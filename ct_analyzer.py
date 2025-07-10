"""
CT Analyzer - Core Analysis Orchestrator
Coordinates different analysis modes and manages the analysis workflow
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

from image_processor import ImageProcessor
from med42_client import Med42Client
from llama_med42_client import LlamaMed42Client
from gemma_client import GemmaClient
from intelligent_analyzer import IntelligentAnalyzer
from comprehensive_analyzer import ComprehensiveAnalyzer

# Попробуем импортировать MedGemma анализатор
try:
    from medgemma_analyzer import MedGemmaAnalyzer
    MEDGEMMA_ANALYZER_AVAILABLE = True
except ImportError as e:
    MEDGEMMA_ANALYZER_AVAILABLE = False
    print(f"⚠️ MedGemma анализатор недоступен: {e}")

import config

class CTAnalyzer:
    """Main CT analysis coordinator"""
    
    def __init__(self):
        """Initialize the CT analyzer with all available clients"""
        self.image_processor = ImageProcessor()
        
        # Initialize analysis clients
        self.med42_client = None
        self.llama_med42_client = None
        self.gemma_client = None
        self.intelligent_analyzer = None
        
        # Create output directory
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _get_client(self, mode: str):
        """Get or initialize the appropriate client for the analysis mode"""
        if mode == "med42":
            if self.med42_client is None:
                self.med42_client = Med42Client()
            return self.med42_client
            
        elif mode == "hybrid":
            if self.llama_med42_client is None:
                self.llama_med42_client = LlamaMed42Client()
            return self.llama_med42_client
            
        elif mode == "gemma":
            if self.gemma_client is None:
                self.gemma_client = GemmaClient()
            return self.gemma_client
            
        elif mode == "medgemma":
            # MedGemma analyzer is always created fresh
            if MEDGEMMA_ANALYZER_AVAILABLE:
                return MedGemmaAnalyzer()
            else:
                raise ValueError("MedGemma анализатор недоступен")
            
        elif mode == "intelligent":
            if self.intelligent_analyzer is None:
                self.intelligent_analyzer = IntelligentAnalyzer()
            return self.intelligent_analyzer
            
        elif mode == "comprehensive":
            # Comprehensive analyzer is always created fresh for each session
            return ComprehensiveAnalyzer()
            
        else:
            raise ValueError(f"Unknown analysis mode: {mode}")
    
    def analyze_directory(self, input_path: str, mode: str = "openai") -> Optional[Dict[str, Any]]:
        """
        Analyze all DICOM files in a directory
        
        Args:
            input_path: Path to directory containing DICOM files
            mode: Analysis mode ("med42", "hybrid")
            
        Returns:
            Analysis results dictionary
        """
        print(f"Начало анализа в режиме: {mode}")
        
        try:
            # Load and process DICOM files
            print("Загрузка DICOM-файлов...")
            dicom_data = self.image_processor.load_dicom_series(input_path)
            
            if not dicom_data:
                print("Не удалось загрузить DICOM-файлы")
                return None
                
            print(f"Загружено {len(dicom_data)} изображений")
            
            # Get analysis client
            client = self._get_client(mode)
            
            # Perform analysis
            print("Выполнение анализа...")
            if mode == "hybrid":
                # Hybrid mode uses different method
                analysis_result = client.analyze_images(dicom_data)
            elif mode == "intelligent":
                # Intelligent mode uses three-stage analysis
                analysis_result = client.analyze_study(dicom_data)
            elif mode == "comprehensive":
                # Comprehensive mode analyzes ALL images with context preservation
                analysis_result = client.analyze_complete_study(dicom_data)
            else:
                # Standard analysis for med42/gemma modes
                analysis_result = client.analyze_images(dicom_data)
            
            if not analysis_result:
                print("Анализ не дал результатов")
                return None
            
            # Prepare final result
            if mode == "intelligent":
                # Intelligent mode has different result structure
                result = {
                    'mode': mode,
                    'timestamp': datetime.now().isoformat(),
                    'image_count': len(dicom_data),
                    'analysis': analysis_result.get('final_report', str(analysis_result)),
                    'stages': analysis_result.get('stages', {}),
                    'intelligent_analysis': analysis_result
                }
            elif mode == "comprehensive":
                # Comprehensive mode has session-based result structure
                result = {
                    'mode': mode,
                    'timestamp': datetime.now().isoformat(),
                    'image_count': analysis_result.get('total_images', len(dicom_data)),
                    'analysis': analysis_result.get('final_report', str(analysis_result)),
                    'session_id': analysis_result.get('session_id'),
                    'context_file': analysis_result.get('context_file'),
                    'comprehensive_analysis': analysis_result
                }
            else:
                result = {
                    'mode': mode,
                    'timestamp': datetime.now().isoformat(),
                    'image_count': len(dicom_data),
                    'analysis': analysis_result if isinstance(analysis_result, str) else analysis_result.get('combined_analysis', analysis_result.get('analysis', str(analysis_result)))
                }
            
            # Save results
            self._save_results(result)
            
            # Create summary
            result['summary'] = self._create_summary(result)
            
            print("Анализ завершён успешно")
            return result
            
        except Exception as e:
            print(f"Ошибка во время анализа: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_results(self, result: Dict[str, Any]):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON result
        json_path = os.path.join(self.output_dir, f"analysis_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Save readable report
        report_path = os.path.join(self.output_dir, f"report_{timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(config.ANALYSIS_REPORT_TEMPLATE.format(
                mode=result['mode'],
                image_count=result['image_count'],
                timestamp=result['timestamp'],
                analysis_content=result['analysis']
            ))
        
        print(f"Результаты сохранены: {json_path}, {report_path}")
    
    def _create_summary(self, result: Dict[str, Any]) -> str:
        """Create a brief summary of the analysis"""
        analysis_text = result.get('analysis', '')
        
        # Extract first few sentences as summary
        sentences = analysis_text.split('.')[:3]
        summary = '. '.join(sentences)
        
        if len(summary) < 100 and len(sentences) < 3:
            # If summary is too short, take more content
            summary = analysis_text[:300]
        
        return summary + "..." if len(analysis_text) > len(summary) else summary

    def get_available_modes(self) -> List[str]:
        """Get list of available analysis modes"""
        modes = ["med42", "hybrid", "gemma", "intelligent", "comprehensive"]
        
        # Добавляем MedGemma если доступна
        if MEDGEMMA_ANALYZER_AVAILABLE:
            modes.append("medgemma")
            
        return modes
    
    def validate_input(self, input_path: str) -> bool:
        """Validate that input directory contains DICOM files"""
        if not os.path.exists(input_path):
            return False
        
        # Check for DICOM files
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(('.dcm', '.dicom')) or '.' not in file:
                    return True
        
        return False 