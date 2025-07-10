#!/usr/bin/env python3
"""
MedGemma Analyzer - Google's Medical AI Model Integration
Provides direct medical image analysis using MedGemma model
"""

import os
import sys
import torch
import time
import gc
from typing import List, Dict, Any, Optional
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import config
from datetime import datetime
import concurrent.futures
import threading
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
    
    def __init__(self, max_images_to_analyze: int = None, enable_parallel: bool = True, batch_size: int = 5):
        """Initialize MedGemma analyzer
        
        Args:
            max_images_to_analyze: Maximum number of images to analyze (None = all images)
            enable_parallel: Enable parallel processing where possible
            batch_size: Number of images to process in each batch for memory management
        """
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
            self.max_images_to_analyze = max_images_to_analyze
            self.enable_parallel = enable_parallel
            self.batch_size = batch_size
            self._model_lock = threading.Lock()  # For thread safety
            self._load_model()
            show_success("MedGemma анализатор инициализирован")
            show_info(f"🔧 Настройки: макс. изображений = {'все' if max_images_to_analyze is None else max_images_to_analyze}, параллелизация = {enable_parallel}, размер батча = {batch_size}")
            
        except Exception as e:
            show_error(f"Ошибка инициализации MedGemma: {e}")
            log_to_file(f"MedGemma initialization error: {e}", "ERROR")
            raise
    
    def analyze_study(self, images: List[Dict[str, Any]], user_context: str = "") -> Optional[str]:
        """
        Analyze CT study using MedGemma - processes ALL images
        
        Args:
            images: List of processed image data
            user_context: Additional context from user
            
        Returns:
            Medical analysis text
        """
        if not images:
            show_error("Нет изображений для анализа")
            return None
        
        # Determine how many images to process
        total_images = len(images)
        images_to_process = total_images if self.max_images_to_analyze is None else min(self.max_images_to_analyze, total_images)
        
        show_step(f"Запуск MedGemma анализа ({images_to_process} из {total_images} изображений)")
        log_to_file(f"Starting MedGemma analysis with {images_to_process} out of {total_images} images")
        
        if user_context:
            show_info(f"Дополнительный контекст: {user_context}")
        
        try:
            # Process images in batches
            result = self._analyze_all_images(images[:images_to_process], user_context)
            
            if result:
                show_success(f"MedGemma анализ завершён успешно ({images_to_process} изображений)")
                log_to_file(f"MedGemma analysis completed successfully for {images_to_process} images")
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
    
    def _analyze_all_images(self, images: List[Dict[str, Any]], user_context: str = "") -> Optional[str]:
        """
        Analyze all images in batches with progress tracking
        
        Args:
            images: List of image dictionaries
            user_context: Additional context from user
            
        Returns:
            Combined medical analysis text
        """
        if not self.model or not self.processor:
            show_error("Модель MedGemma не инициализирована")
            return None
            
        total_images = len(images)
        show_step(f"Анализ {total_images} изображений в батчах по {self.batch_size}")
        log_to_file(f"Analyzing {total_images} images in batches of {self.batch_size}")
        
        all_analyses = []
        
        try:
            # Process images in batches
            for batch_start in range(0, total_images, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_images)
                batch_images = images[batch_start:batch_end]
                batch_num = (batch_start // self.batch_size) + 1
                total_batches = (total_images + self.batch_size - 1) // self.batch_size
                
                show_info(f"📦 Обработка батча {batch_num}/{total_batches} ({len(batch_images)} изображений)")
                log_to_file(f"Processing batch {batch_num}/{total_batches} with {len(batch_images)} images")
                
                # Process batch
                batch_analyses = self._process_batch(batch_images, user_context, batch_start)
                all_analyses.extend(batch_analyses)
                
                # Memory cleanup after each batch
                self._cleanup_memory()
                
                # Progress update
                processed = min(batch_end, total_images)
                show_success(f"✅ Обработано {processed}/{total_images} изображений")
                
                # Small delay to prevent overheating
                time.sleep(0.5)
            
            # Create final comprehensive report
            if all_analyses:
                final_report = self._create_comprehensive_report(all_analyses, user_context, total_images)
                show_success(f"🎉 Анализ завершён! Обработано {len(all_analyses)} изображений")
                return final_report
            else:
                show_warning("Не удалось получить анализ ни одного изображения")
                return None
                
        except Exception as e:
            show_error(f"Ошибка пакетного анализа: {e}")
            log_to_file(f"Batch analysis error: {e}", "ERROR")
            self._cleanup_memory()
            return None
    
    def _process_batch(self, batch_images: List[Dict[str, Any]], user_context: str, start_index: int) -> List[str]:
        """
        Process a batch of images
        
        Args:
            batch_images: List of images in this batch
            user_context: User context
            start_index: Starting index for image numbering
            
        Returns:
            List of analysis results
        """
        batch_analyses = []
        
        for i, image_data in enumerate(batch_images):
            global_index = start_index + i + 1
            
            try:
                show_info(f"🔍 Анализ изображения {global_index}")
                
                # Analyze single image
                analysis = self._analyze_single_image(image_data, user_context, global_index)
                
                if analysis:
                    batch_analyses.append(analysis)
                    show_success(f"✅ Изображение {global_index} проанализировано")
                else:
                    show_warning(f"⚠️ Изображение {global_index} не удалось проанализировать")
                
            except Exception as e:
                show_error(f"❌ Ошибка анализа изображения {global_index}: {e}")
                log_to_file(f"Error analyzing image {global_index}: {e}", "ERROR")
                continue
        
        return batch_analyses
    
    def _analyze_single_image(self, image_data: Dict[str, Any], user_context: str, image_index: int) -> Optional[str]:
        """
        Analyze a single image with MedGemma
        
        Args:
            image_data: Image data dictionary
            user_context: User context
            image_index: Image number for reporting
            
        Returns:
            Analysis text or None
        """
        try:
            # Thread-safe model access
            with self._model_lock:
                # Decode base64 image
                import base64
                from io import BytesIO
                image_bytes = base64.b64decode(image_data['base64_image'])
                image = Image.open(BytesIO(image_bytes))
                
                # Create medical prompt
                prompt = f"""Analyze this CT image #{image_index} from a medical study.

{f"Study Context: {user_context}" if user_context else ""}

Please provide:
1. Anatomical structures visible
2. Any pathological findings
3. Normal findings
4. Clinical significance
5. Recommendations

Focus on diagnostic and therapeutic implications."""
                
                # Prepare messages for MedGemma
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an expert radiologist specializing in CT image analysis."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": image}
                        ]
                    }
                ]
                
                # Process with MedGemma
                inputs = self.processor.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=True,
                    return_dict=True, 
                    return_tensors="pt"
                ).to(self.model.device, dtype=torch.bfloat16 if self.device != "cpu" else torch.float32)
                
                # Generate analysis
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                # Decode response
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                
                # Extract generated part (remove input prompt)
                generated_text = response.split("assistant")[-1].strip() if "assistant" in response else response.strip()
                
                if generated_text:
                    return f"=== ИЗОБРАЖЕНИЕ {image_index} ===\n{generated_text}"
                else:
                    return None
                
        except Exception as e:
            log_to_file(f"Single image analysis error for image {image_index}: {e}", "ERROR")
            return None
    
    def _cleanup_memory(self):
        """Clean up GPU/CPU memory"""
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            log_to_file(f"Memory cleanup error: {e}", "WARNING")
    
    def _create_comprehensive_report(self, analyses: List[str], user_context: str, total_images: int) -> str:
        """
        Create a comprehensive medical report from all analyses
        
        Args:
            analyses: List of individual image analyses
            user_context: User context
            total_images: Total number of images processed
            
        Returns:
            Comprehensive medical report
        """
        report = f"""=== ПОЛНЫЙ MEDGEMMA АНАЛИЗ CT ИССЛЕДОВАНИЯ ===

ДАТА АНАЛИЗА: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
ОБЩАЯ ИНФОРМАЦИЯ:
- Всего изображений обработано: {len(analyses)} из {total_images}
- Дополнительный контекст: {user_context if user_context else "Не предоставлен"}
- Анализатор: MedGemma 4B (Google)

=== ДЕТАЛЬНЫЙ АНАЛИЗ ПО ИЗОБРАЖЕНИЯМ ===

{chr(10).join(analyses)}

=== ОБЩИЕ ВЫВОДЫ ===

Исследование включает {len(analyses)} проанализированных изображений.
Каждое изображение было обработано индивидуально с использованием 
специализированной медицинской модели MedGemma.

=== РЕКОМЕНДАЦИИ ===

1. Внимательно изучите каждый раздел анализа
2. Обратите внимание на повторяющиеся находки
3. Консультируйтесь с лечащим врачом для интерпретации результатов
4. При необходимости проведите дополнительные исследования

=== КОНЕЦ АНАЛИЗА ==="""
        
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