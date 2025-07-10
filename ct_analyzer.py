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
from progress_logger import (
    show_step, show_success, show_error, show_info, show_warning, 
    log_to_file, suppress_prints
)

# MedGemma integration
try:
    from medgemma_analyzer import MedGemmaAnalyzer
    MEDGEMMA_ANALYZER_AVAILABLE = True
    log_to_file("MedGemma analyzer available")
except ImportError:
    MEDGEMMA_ANALYZER_AVAILABLE = False
    log_to_file("MedGemma analyzer not available", "WARNING")

import config

class CTAnalyzer:
    def __init__(self, config_path: str = "config.py", max_images_for_medgemma: int = None, enable_parallel: bool = True, batch_size: int = 5):
        """Initialize CT Analyzer with lazy loading
        
        Args:
            config_path: Path to configuration file
            max_images_for_medgemma: Maximum number of images to analyze with MedGemma (None = all images)
            enable_parallel: Enable parallel processing where possible
            batch_size: Number of images to process in each batch for memory management
        """
        self.config = Config()
        self.image_processor = ImageProcessor()
        self.max_images_for_medgemma = max_images_for_medgemma
        self.enable_parallel = enable_parallel
        self.batch_size = batch_size
        
        # Lazy loading - initialize analyzers only when needed
        self._med42_client = None
        self._comprehensive_analyzer = None
        self._medgemma_analyzer = None
        
        # Initialize logging
        self.setup_logging()
        
        # Show configuration
        show_info(f"🔧 Конфигурация CT Analyzer:")
        show_info(f"   - Макс. изображений для MedGemma: {'все' if max_images_for_medgemma is None else max_images_for_medgemma}")
        show_info(f"   - Параллелизация: {enable_parallel}")
        show_info(f"   - Размер батча: {batch_size}")
        
    def setup_logging(self):
        """Setup logging configuration"""
        # Create output directory
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        log_to_file(f"Output directory created: {self.output_dir}")
        
    @property
    def med42_client(self):
        """Lazy load Med42 client"""
        if self._med42_client is None:
            show_step("Инициализация Med42 клиента")
            log_to_file("Initializing Med42 client")
            with suppress_prints():
                self._med42_client = Med42Client()
            show_success("Med42 клиент инициализирован")
            log_to_file("Med42 client initialized successfully")
        return self._med42_client
        
    @property
    def comprehensive_analyzer(self):
        """Lazy load Comprehensive analyzer"""
        if self._comprehensive_analyzer is None:
            show_step("Инициализация Comprehensive анализатора")
            log_to_file("Initializing Comprehensive analyzer")
            with suppress_prints():
                self._comprehensive_analyzer = ComprehensiveAnalyzer()
            show_success("Comprehensive анализатор инициализирован")
            log_to_file("Comprehensive analyzer initialized successfully")
        return self._comprehensive_analyzer
        
    @property
    def medgemma_analyzer(self):
        """Lazy load MedGemma analyzer"""
        if self._medgemma_analyzer is None:
            if MEDGEMMA_ANALYZER_AVAILABLE:
                show_step("Инициализация MedGemma анализатора")
                log_to_file("Initializing MedGemma analyzer")
                with suppress_prints():
                    self._medgemma_analyzer = MedGemmaAnalyzer(
                        max_images_to_analyze=self.max_images_for_medgemma,
                        enable_parallel=self.enable_parallel,
                        batch_size=self.batch_size
                    )
                show_success("MedGemma анализатор инициализирован")
                log_to_file("MedGemma analyzer initialized successfully")
            else:
                show_error("MedGemma анализатор недоступен")
                log_to_file("MedGemma analyzer not available", "ERROR")
                return None
        return self._medgemma_analyzer
    
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
            show_error(f"Директория не найдена: {input_path}")
            log_to_file(f"Directory not found: {input_path}", "ERROR")
            return None
            
        # Process images
        show_step("Загрузка DICOM изображений")
        log_to_file(f"Loading DICOM images from: {input_path}")
        
        with suppress_prints():
            images = self.image_processor.load_dicom_series(input_path)
            
        if not images:
            show_error(f"Не найдены DICOM файлы в: {input_path}")
            log_to_file(f"No DICOM files found in: {input_path}", "ERROR")
            return None
            
        show_success(f"Найдено {len(images)} DICOM изображений")
        log_to_file(f"Found {len(images)} DICOM images")
        
        # Show context info
        if user_context:
            show_info(f"Дополнительный контекст: {user_context}")
            log_to_file(f"User context: {user_context}")
        
        # Analyze using selected mode
        try:
            if mode == "medgemma":
                analyzer = self.medgemma_analyzer
                if not analyzer:
                    show_error("MedGemma анализатор недоступен")
                    log_to_file("MedGemma analyzer not available", "ERROR")
                    return None
                return analyzer.analyze_study(images, user_context)
                
            elif mode == "med42":
                return self.med42_client.analyze_ct_study(images, user_context)
                
            elif mode == "comprehensive":
                return self.comprehensive_analyzer.analyze_study(images, user_context)
                
            else:
                show_error(f"Неизвестный режим: {mode}")
                log_to_file(f"Unknown mode: {mode}", "ERROR")
                return None
                
        except Exception as e:
            show_error(f"Ошибка анализа: {e}")
            log_to_file(f"Analysis error: {e}", "ERROR")
            return None
    
    def get_available_modes(self) -> List[str]:
        """Get list of available analysis modes"""
        modes = ["medgemma", "med42", "comprehensive"]
        
        # Добавляем MedGemma если доступна (она должна быть основным режимом)
        if not MEDGEMMA_ANALYZER_AVAILABLE:
            show_warning("MedGemma недоступна, используйте med42 или comprehensive")
            log_to_file("MedGemma not available, using med42 or comprehensive", "WARNING")
            
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