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
        
        # Skip first 5 service images (calibration/service data)
        skip_first = 5  # Skip first 5 images (service/calibration data)
        service_filtered_images = images[skip_first:] if len(images) > skip_first else images
        
        # Determine how many images to process
        total_images = len(service_filtered_images)
        images_to_process = total_images if self.max_images_to_analyze is None else min(self.max_images_to_analyze, total_images)
        
        show_info(f"📊 Пропущено первые {skip_first} служебных изображений из {len(images)} общих")
        if skip_first > 0:
            show_info(f"🔍 Будут анализироваться изображения с #{skip_first+1} по #{skip_first+images_to_process}")
        
        show_step(f"Запуск MedGemma анализа ({images_to_process} из {total_images} изображений)")
        log_to_file(f"Starting MedGemma analysis with {images_to_process} out of {total_images} images")
        
        if user_context:
            show_info(f"Дополнительный контекст: {user_context}")
        
        try:
            # Process images in batches
            result = self._analyze_all_images(service_filtered_images[:images_to_process], user_context)
            
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
        
        # Create session for intermediate saves
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join("output", f"session_{session_id}")
        os.makedirs(session_dir, exist_ok=True)
        
        show_info(f"📁 Сессия: {session_id}")
        show_info(f"💾 Промежуточные результаты: {session_dir}")
        
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
                
                # Save intermediate results after each batch
                self._save_intermediate_results(session_dir, batch_num, total_batches, all_analyses, user_context)
                
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
                
                # Save final session summary
                self._save_final_session_summary(session_dir, session_id, total_images, len(all_analyses))
                
                show_success(f"🎉 Анализ завершён! Обработано {len(all_analyses)} изображений")
                show_info(f"📁 Все результаты сохранены в: {session_dir}")
                return final_report
            else:
                show_warning("Не удалось получить анализ ни одного изображения")
                return None
                
        except Exception as e:
            show_error(f"Ошибка пакетного анализа: {e}")
            log_to_file(f"Batch analysis error: {e}", "ERROR")
            
            # Save error state
            self._save_error_state(session_dir, str(e), len(all_analyses), total_images)
            
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
                
                # Ensure image is in RGB format
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Log image info for debugging
                print(f"[DEBUG] Изображение {image_index}: размер {image.size}, режим {image.mode}")
                
                # Resize if too large (MedGemma works better with smaller images)
                max_size = 512
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    print(f"[DEBUG] Изображение уменьшено до {image.size}")
                
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
                
                # Decode only the new tokens (generated part)
                input_len = inputs["input_ids"].shape[-1]
                generated_tokens = outputs[0][input_len:]
                
                # Decode only the generated part
                generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
                
                # Log the full response for debugging
                print(f"=== ОТЛАДКА MEDGEMMA (Изображение {image_index}) ===")
                print(f"Длина входных токенов: {input_len}")
                print(f"Длина выходных токенов: {len(outputs[0])}")
                print(f"Сгенерированный текст: {generated_text[:200]}...")
                print("=" * 50)
                
                if generated_text and len(generated_text.strip()) > 10:
                    return f"=== ИЗОБРАЖЕНИЕ {image_index} ===\n{generated_text.strip()}"
                else:
                    return None
                
        except Exception as e:
            print(f"❌ ОШИБКА АНАЛИЗА ИЗОБРАЖЕНИЯ {image_index}: {e}")
            log_to_file(f"Single image analysis error for image {image_index}: {e}", "ERROR")
            import traceback
            traceback.print_exc()
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
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""=== ПОЛНЫЙ MEDGEMMA АНАЛИЗ CT ИССЛЕДОВАНИЯ ===

ДАТА АНАЛИЗА: {timestamp}
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
        
        # Save report to file
        self._save_report_to_file(report, total_images, timestamp)
        
        return report
    
    def _save_intermediate_results(self, session_dir: str, batch_num: int, total_batches: int, 
                                 analyses: List[str], user_context: str):
        """
        Save intermediate results after each batch
        
        Args:
            session_dir: Session directory path
            batch_num: Current batch number
            total_batches: Total number of batches
            analyses: All analyses so far
            user_context: User context
        """
        try:
            # Save current progress
            progress_file = os.path.join(session_dir, "progress.json")
            progress_data = {
                "timestamp": datetime.now().isoformat(),
                "batch_num": batch_num,
                "total_batches": total_batches,
                "images_processed": len(analyses),
                "user_context": user_context,
                "status": "in_progress"
            }
            
            import json
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            
            # Save analyses so far
            analyses_file = os.path.join(session_dir, f"analyses_batch_{batch_num:03d}.txt")
            with open(analyses_file, 'w', encoding='utf-8') as f:
                f.write(f"=== ПРОМЕЖУТОЧНЫЕ РЕЗУЛЬТАТЫ (Батч {batch_num}/{total_batches}) ===\n\n")
                f.write(f"Обработано изображений: {len(analyses)}\n")
                f.write(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("\n".join(analyses))
            
            # Show progress
            if batch_num % 10 == 0 or batch_num == total_batches:  # Show every 10th batch or last
                show_info(f"💾 Промежуточные результаты сохранены (батч {batch_num}/{total_batches})")
                
        except Exception as e:
            show_warning(f"Ошибка сохранения промежуточных результатов: {e}")
            log_to_file(f"Error saving intermediate results: {e}", "WARNING")
    
    def _save_final_session_summary(self, session_dir: str, session_id: str, total_images: int, processed_images: int):
        """
        Save final session summary
        
        Args:
            session_dir: Session directory path
            session_id: Session ID
            total_images: Total number of images
            processed_images: Number of successfully processed images
        """
        try:
            summary_file = os.path.join(session_dir, "session_summary.json")
            summary_data = {
                "session_id": session_id,
                "start_time": session_id,  # Encoded in session_id
                "end_time": datetime.now().isoformat(),
                "total_images": total_images,
                "processed_images": processed_images,
                "success_rate": (processed_images / total_images * 100) if total_images > 0 else 0,
                "analyzer": "MedGemma 4B",
                "batch_size": self.batch_size,
                "status": "completed"
            }
            
            import json
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            log_to_file(f"Error saving session summary: {e}", "WARNING")
    
    def _save_error_state(self, session_dir: str, error_msg: str, processed_images: int, total_images: int):
        """
        Save error state for debugging
        
        Args:
            session_dir: Session directory path
            error_msg: Error message
            processed_images: Number of images processed before error
            total_images: Total number of images
        """
        try:
            error_file = os.path.join(session_dir, "error_state.json")
            error_data = {
                "timestamp": datetime.now().isoformat(),
                "error_message": error_msg,
                "processed_images": processed_images,
                "total_images": total_images,
                "status": "error"
            }
            
            import json
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, ensure_ascii=False, indent=2)
                
            show_warning(f"💾 Состояние ошибки сохранено: {error_file}")
                
        except Exception as e:
            log_to_file(f"Error saving error state: {e}", "WARNING")
    
    def _save_report_to_file(self, report: str, total_images: int, timestamp: str):
        """
        Save analysis report to file
        
        Args:
            report: Complete analysis report
            total_images: Number of images processed
            timestamp: Analysis timestamp
        """
        try:
            # Create output directory
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename with timestamp
            safe_timestamp = timestamp.replace(':', '-').replace(' ', '_')
            filename = f"medgemma_analysis_{safe_timestamp}_{total_images}images.txt"
            filepath = os.path.join(output_dir, filename)
            
            # Save report
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            
            show_success(f"📄 Отчёт сохранён: {filepath}")
            log_to_file(f"Analysis report saved to: {filepath}")
            
            # Also save as JSON for programmatic access
            json_filename = f"medgemma_analysis_{safe_timestamp}_{total_images}images.json"
            json_filepath = os.path.join(output_dir, json_filename)
            
            analysis_data = {
                "timestamp": timestamp,
                "total_images": total_images,
                "analyzer": "MedGemma 4B",
                "report": report,
                "metadata": {
                    "processed_images": total_images,
                    "batch_size": self.batch_size,
                    "model_name": self.model_name,
                    "device": self.device
                }
            }
            
            import json
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            
            show_success(f"📊 Данные сохранены: {json_filepath}")
            log_to_file(f"Analysis data saved to: {json_filepath}")
            
        except Exception as e:
            show_error(f"Ошибка сохранения отчёта: {e}")
            log_to_file(f"Error saving report: {e}", "ERROR")


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