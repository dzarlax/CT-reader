"""
CT Analyzer - Core Analysis Orchestrator
Coordinates different analysis modes and manages the analysis workflow
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Core components
from config import Config
from image_processor import ImageProcessor
from med42_client import Med42Client
from comprehensive_analyzer import ComprehensiveAnalyzer

# MedGemma integration
try:
    from medgemma_analyzer import MedGemmaAnalyzer
    MEDGEMMA_ANALYZER_AVAILABLE = True
except ImportError:
    MEDGEMMA_ANALYZER_AVAILABLE = False

import config

class CTAnalyzer:
    def __init__(self, config_path: str = "config.py"):
        """Initialize CT Analyzer with simplified modes"""
        self.config = Config()
        self.image_processor = ImageProcessor()
        
        # Core analyzers
        self.med42_client = Med42Client()
        self.comprehensive_analyzer = ComprehensiveAnalyzer()
        
        # MedGemma analyzer (primary mode)
        if MEDGEMMA_ANALYZER_AVAILABLE:
            self.medgemma_analyzer = MedGemmaAnalyzer()
        else:
            self.medgemma_analyzer = None
            
        # Initialize logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def analyze_directory(self, input_path: str, mode: str = "medgemma", user_context: str = "") -> Optional[Dict[str, Any]]:
        """
        Analyze CT images in directory using specified mode
        
        Args:
            input_path: Path to directory containing DICOM files
            mode: Analysis mode ('medgemma', 'med42', 'comprehensive')
            user_context: Additional context from user (symptoms, age, etc.)
            
        Returns:
            Analysis results dictionary
        """
        if not os.path.exists(input_path):
            print(f"❌ Директория не найдена: {input_path}")
            return None
            
        # Process images
        images = self.image_processor.load_dicom_series(input_path)
        if not images:
            print(f"❌ Не найдены DICOM файлы в: {input_path}")
            return None
            
        print(f"📊 Найдено {len(images)} DICOM изображений")
        
        # Show context info
        if user_context:
            print(f"📋 Дополнительный контекст: {user_context}")
        
        # Analyze using selected mode
        try:
            if mode == "medgemma":
                if not self.medgemma_analyzer:
                    print("❌ MedGemma анализатор недоступен")
                    return None
                return self.medgemma_analyzer.analyze_study(images, user_context)
                
            elif mode == "med42":
                return self.med42_client.analyze_ct_study(images, user_context)
                
            elif mode == "comprehensive":
                return self.comprehensive_analyzer.analyze_study(images, user_context)
                
            else:
                print(f"❌ Неизвестный режим: {mode}")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка анализа: {e}")
            return None
    
    def get_available_modes(self) -> List[str]:
        """Get list of available analysis modes"""
        modes = ["medgemma", "med42", "comprehensive"]
        
        # Добавляем MedGemma если доступна (она должна быть основным режимом)
        if not MEDGEMMA_ANALYZER_AVAILABLE:
            print("⚠️ MedGemma недоступна, используйте med42 или comprehensive")
            
        return modes
        
    def get_mode_description(self, mode: str) -> str:
        """Get description of analysis mode"""
        descriptions = {
            "medgemma": "🏥 Специализированная медицинская модель Google - прямой анализ изображений",
            "med42": "📋 Быстрый специализированный медицинский анализ",
            "comprehensive": "🔍 Полный анализ всех изображений с контекстом"
        }
        return descriptions.get(mode, "Неизвестный режим")
        
    def validate_mode(self, mode: str) -> bool:
        """Validate if analysis mode is available"""
        available_modes = self.get_available_modes()
        return mode in available_modes
    
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