#!/usr/bin/env python3
"""
MedGemma Analyzer - Google's Medical AI Model Integration
Provides direct medical image analysis using MedGemma model
"""

import os
import sys
import torch
import time
from typing import List, Dict, Any, Optional
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import config
from progress_logger import (
    show_step, show_success, show_error, show_info, show_warning, 
    start_progress, update_progress, complete_progress, 
    log_to_file, suppress_prints
)

# Check if MedGemma is available
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    MEDGEMMA_AVAILABLE = True
    log_to_file("MedGemma dependencies available")
except ImportError as e:
    MEDGEMMA_AVAILABLE = False
    log_to_file(f"MedGemma недоступна: {e}", "WARNING")

class MedGemmaAnalyzer:
    """Medical image analysis using Google's MedGemma model"""
    
    def __init__(self):
        """Initialize MedGemma analyzer"""
        if not MEDGEMMA_AVAILABLE:
            show_error("MedGemma недоступна")
            return
            
        try:
            show_step("Инициализация MedGemma анализатора")
            self.model_name = "google/medgemma-4b-it"
            self.device = self._get_device()
            self.processor = None
            self.model = None
            self.token = None
            self._load_model()
            show_success("MedGemma анализатор инициализирован")
            
        except Exception as e:
            show_error(f"Ошибка инициализации MedGemma: {e}")
            log_to_file(f"MedGemma initialization error: {e}", "ERROR")
            raise
    
    def analyze_study(self, images: List[Dict[str, Any]], user_context: str = "") -> Optional[str]:
        """
        Analyze CT study using MedGemma
        
        Args:
            images: List of processed image data
            user_context: Additional context from user
            
        Returns:
            Medical analysis text
        """
        if not images:
            show_error("Нет изображений для анализа")
            return None
            
        show_step(f"Запуск MedGemma анализа ({len(images)} изображений)")
        log_to_file(f"Starting MedGemma analysis with {len(images)} images")
        
        if user_context:
            show_info(f"Дополнительный контекст: {user_context}")
        
        try:
            # Analyze all images with progress tracking
            result = self._analyze_ct_study(images, user_context)
            
            if result:
                show_success("MedGemma анализ завершён успешно")
                log_to_file("MedGemma analysis completed successfully")
                return result
            else:
                show_warning("MedGemma анализ не дал результатов")
                log_to_file("MedGemma analysis returned no results", "WARNING")
                return None
                
        except Exception as e:
            show_error(f"Ошибка MedGemma анализа: {e}")
            log_to_file(f"MedGemma analysis error: {e}", "ERROR")
            return None
    
    def _get_device(self):
        """Get the device for model loading"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _setup_huggingface_token(self):
        """Setup Hugging Face token"""
        from dotenv import load_dotenv
        from huggingface_hub import login
        load_dotenv()
        
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        if self.token:
            show_success("✅ Токен Hugging Face найден в .env файле")
            log_to_file("Hugging Face token found in .env file")
            show_info(f"🔑 Токен начинается с: {self.token[:10]}...")
            log_to_file(f"Token starts with: {self.token[:10]}...")
            
            # Явная аутентификация в HuggingFace Hub
            try:
                login(token=self.token, add_to_git_credential=False)
                show_success("✅ Аутентификация в HuggingFace Hub успешна")
                log_to_file("HuggingFace Hub authentication successful")
            except Exception as e:
                show_error(f"❌ Ошибка аутентификации в HuggingFace Hub: {e}")
                log_to_file(f"HuggingFace Hub authentication error: {e}", "ERROR")
                raise
        else:
            show_error("❌ Токен Hugging Face не найден в .env файле!")
            show_info("💡 Добавьте HUGGINGFACE_TOKEN=your_token в .env файл")
            log_to_file("Hugging Face token not found in .env file", "ERROR")
            raise ValueError("Токен Hugging Face обязателен для работы с MedGemma")
    
    def _load_model(self):
        """Load the MedGemma model and processor"""
        try:
            show_step("Загрузка модели MedGemma")
            
            # Setup HuggingFace token
            self._setup_huggingface_token()
            
            # Load processor and model with token
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                token=self.token,
                trust_remote_code=True
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                token=self.token,
                torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device != "cpu" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model.to(self.device)
            
            show_success("Модель MedGemma загружена")
        except Exception as e:
            show_error(f"Ошибка загрузки модели MedGemma: {e}")
            log_to_file(f"Model loading error: {e}", "ERROR")
            raise
    
    def _analyze_ct_study(self, images: List[Dict[str, Any]], user_context: str = "") -> Optional[str]:
        """
        Analyze a single CT image using MedGemma
        
        Args:
            image_data: Dictionary containing 'image' (PIL Image) and 'dicom_data' (DICOM data)
            user_context: Additional context from user
            
        Returns:
            Medical analysis text for the image
        """
        if not self.model or not self.processor:
            show_error("Модель MedGemma не инициализирована")
            return None
            
        show_step(f"Анализ изображения {images[0]['dicom_data']['SeriesInstanceUID']}")
        
        try:
            # Prepare image for MedGemma
            image_data = images[0]
            image = Image.fromarray(image_data['image'])
            
            # Process image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate text using MedGemma
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    num_beams=5,
                    temperature=0.3
                )
            
            # Decode the generated text
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Add user context to the analysis
            if user_context:
                generated_text += f"\n\nПредоставленный контекст: {user_context}"
            
            show_success(f"Анализ изображения завершён: {generated_text}")
            return generated_text
            
        except Exception as e:
            show_error(f"Ошибка анализа изображения: {e}")
            log_to_file(f"Image analysis error: {e}", "ERROR")
            return None
    
    def _create_final_report(self, analyses: List[str]) -> str:
        """Создаёт финальный медицинский отчёт"""
        
        report = f"""
=== MEDGEMMA МЕДИЦИНСКИЙ АНАЛИЗ CT ИССЛЕДОВАНИЯ ===

ДАТА АНАЛИЗА: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
КОЛИЧЕСТВО ИЗОБРАЖЕНИЙ: {len(analyses)}
АНАЛИЗАТОР: MedGemma 4B (Google)

=== ДЕТАЛЬНЫЙ АНАЛИЗ ПО ИЗОБРАЖЕНИЯМ ===

"""
        
        # Добавляем все анализы
        for analysis in analyses:
            report += analysis + "\n\n"
        
        # Создаём общее резюме с помощью MedGemma
        if self.use_medgemma:
            try:
                summary_prompt = f"""Based on the following CT study analysis, provide a comprehensive summary:

{chr(10).join(analyses)}

Please provide:
1. OVERALL FINDINGS SUMMARY
2. KEY PATHOLOGICAL FINDINGS
3. CLINICAL SIGNIFICANCE
4. RECOMMENDATIONS
5. FOLLOW-UP SUGGESTIONS

Focus on the most clinically relevant findings and provide actionable recommendations."""
                
                summary = self.medgemma_client.analyze_medical_text(
                    summary_prompt,
                    "CT study comprehensive summary"
                )
                
                if summary:
                    report += f"""
=== ОБЩЕЕ РЕЗЮМЕ ИССЛЕДОВАНИЯ (MedGemma) ===

{summary}

=== КОНЕЦ ОТЧЁТА ==="""
                
            except Exception as e:
                print(f"⚠️ Ошибка создания резюме: {e}")
                report += "\n=== КОНЕЦ ОТЧЁТА ==="
        
        return report


def test_medgemma_analyzer():
    """Тест MedGemma анализатора"""
    print("🧪 ТЕСТ MEDGEMMA АНАЛИЗАТОРА")
    print("=" * 50)
    
    try:
        from image_processor import ImageProcessor
        
        # Проверяем наличие изображений
        if not os.path.exists("input"):
            print("❌ Директория 'input' не найдена")
            return
        
        # Загружаем изображения
        image_processor = ImageProcessor()
        images = image_processor.load_dicom_series("input")
        
        if not images:
            print("❌ Изображения не найдены")
            return
        
        # Тестируем на 2 изображениях
        test_images = images[:2]
        print(f"📊 Тестируем на {len(test_images)} изображениях")
        
        # Инициализируем анализатор
        analyzer = MedGemmaAnalyzer()
        
        # Запускаем анализ
        result = analyzer.analyze_images(test_images)
        
        if result:
            print(f"\n✅ Анализ завершён успешно!")
            print(f"📄 Длина отчёта: {len(result)} символов")
            print(f"🔍 Первые 300 символов:")
            print(result[:300] + "...")
        else:
            print("❌ Анализ не дал результатов")
            
    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_medgemma_analyzer() 