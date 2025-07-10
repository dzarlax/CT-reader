#!/usr/bin/env python3
"""
MedGemma Client - Google's Medical AI Model
Direct integration with MedGemma for medical image analysis
"""

import os
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

class MedGemmaClient:
    """Client for Google's MedGemma medical AI model"""
    
    def __init__(self):
        """Initialize MedGemma client"""
        self.model_name = "google/medgemma-4b-it"
        self.device = self._get_device()
        self.processor = None
        self.model = None
        self.token = None
        
        # Load model with progress tracking
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize MedGemma model with progress tracking"""
        try:
            show_step("Инициализация MedGemma клиента")
            log_to_file("Starting MedGemma client initialization")
            
            # Setup device and memory
            self._setup_device()
            self._setup_cuda_environment()
            self._check_gpu_memory()
            self._setup_huggingface_token()
            
            # Load model with suppressed prints
            with suppress_prints():
                self._load_model()
            
            show_success("MedGemma клиент инициализирован успешно")
            log_to_file("MedGemma client initialized successfully")
            
        except Exception as e:
            show_error(f"Ошибка загрузки MedGemma: {e}")
            log_to_file(f"MedGemma initialization error: {e}", "ERROR")
            raise
    
    def _get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _setup_device(self):
        """Setup device and clear cache"""
        self.device = self._get_device()
        show_info(f"📱 Устройство: {self.device}")
        log_to_file(f"Device: {self.device}")
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            show_info("🧹 CUDA кэш очищен")
            log_to_file("CUDA cache cleared")
    
    def _setup_cuda_environment(self):
        """Setup CUDA environment variables"""
        if self.device == "cuda":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            show_info("🔧 Настройки CUDA для стабильной работы установлены")
            log_to_file("CUDA environment variables set for stability")
    
    def _check_gpu_memory(self):
        """Check GPU memory availability"""
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_memory_free = torch.cuda.memory_reserved(0) / 1024**3
            show_info(f"💾 GPU память: {gpu_memory:.1f}GB общая, {gpu_memory_free:.1f}GB свободная")
            log_to_file(f"GPU memory: {gpu_memory:.1f}GB total, {gpu_memory_free:.1f}GB free")
            
            if gpu_memory < 8:
                show_warning("⚠️ Предупреждение: Мало GPU памяти для MedGemma. Рекомендуется минимум 8GB")
                log_to_file("Warning: Low GPU memory for MedGemma. Minimum 8GB recommended", "WARNING")
    
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
        """Загружает модель и процессор"""
        try:
            show_step(f"📥 Загрузка MedGemma модели: {self.model_name}")
            
            # Загружаем модель для работы с изображениями и текстом
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                token=self.token,
                torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device != "cpu" else None,
                trust_remote_code=True
            )
            
            # Загружаем процессор
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                token=self.token,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            show_success(f"✅ MedGemma модель успешно загружена на {self.device}")
            log_to_file(f"MedGemma model loaded successfully on {self.device}")
            
        except Exception as e:
            show_error(f"❌ Ошибка загрузки MedGemma: {e}")
            log_to_file(f"MedGemma model loading error: {e}", "ERROR")
            raise
    
    def analyze_medical_image(self, image_data: Dict[str, Any], prompt: str = "") -> Optional[str]:
        """
        Анализирует медицинское изображение напрямую с помощью MedGemma
        
        Args:
            image_data: Данные изображения с base64_image
            prompt: Дополнительный промпт для анализа
            
        Returns:
            Медицинский анализ изображения
        """
        if not self.model or not self.processor:
            show_error("❌ Модель не загружена")
            log_to_file("Model not loaded for image analysis")
            return None
        
        try:
            # Декодируем изображение из base64
            image_bytes = base64.b64decode(image_data['base64_image'])
            image = Image.open(BytesIO(image_bytes))
            
            # Формируем медицинский промпт для анализа изображения
            if not prompt:
                prompt = """Analyze this medical CT image and provide a comprehensive medical assessment.

Please provide:

1. VISUAL FINDINGS:
   - Anatomical structures visible
   - Tissue densities and morphology
   - Any abnormal findings or pathology

2. CLINICAL INTERPRETATION:
   - Medical significance of findings
   - Differential diagnosis considerations
   - Severity assessment

3. RECOMMENDATIONS:
   - Additional imaging if needed
   - Clinical correlation required
   - Follow-up suggestions
   - Urgency level

Provide detailed, clinically relevant analysis focused on diagnostic and therapeutic implications."""
            
            # Подготавливаем сообщения для модели
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
            
            # Обрабатываем входные данные
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16 if self.device != "cpu" else torch.float32)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Генерируем анализ с правильными параметрами для MedGemma
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
            
            # Декодируем ответ
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Логируем полный ответ
            show_info("🔍 ПОЛНЫЙ ОТВЕТ MEDGEMMA (Анализ изображения):")
            log_to_file("Full MedGemma response (Image Analysis):")
            log_to_file("=" * 50)
            log_to_file(response)
            log_to_file("=" * 50)
            
            # Очищаем память после анализа
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
                
            return response.strip()
            
        except Exception as e:
            show_error(f"❌ Ошибка анализа изображения MedGemma: {e}")
            log_to_file(f"MedGemma image analysis error: {e}", "ERROR")
            
            # Очищаем память при ошибке
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
                
            return None
    
    def analyze_ct_study(self, images: List[Dict[str, Any]], study_context: str = "") -> Optional[str]:
        """
        Анализирует серию CT изображений как единое исследование
        
        Args:
            images: Список изображений для анализа
            study_context: Контекст исследования
            
        Returns:
            Комплексный анализ исследования
        """
        if not images:
            return None
        
        show_info(f"🔍 MedGemma анализ CT исследования ({len(images)} изображений)...")
        log_to_file(f"MedGemma CT study analysis ({len(images)} images)...")
        show_info("🏥 Обрабатываем ВСЕ изображения для полного анализа")
        log_to_file("Processing ALL images for comprehensive analysis")
        
        try:
            # Анализируем ВСЕ изображения
            individual_analyses = []
            
            # Обрабатываем изображения пакетами для стабильности
            batch_size = 5  # Уменьшаем размер пакета для GPU
            total_batches = (len(images) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(images))
                batch_images = images[start_idx:end_idx]
                
                show_info(f"📦 Обработка пакета {batch_idx + 1}/{total_batches} ({len(batch_images)} изображений)...")
                log_to_file(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_images)} images)...")
                
                # Очищаем память перед обработкой пакета
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()
                
                for i, image_data in enumerate(batch_images):
                    global_idx = start_idx + i + 1
                    show_info(f"📊 Анализ изображения {global_idx}/{len(images)}: ", end="")
                    log_to_file(f"Analyzing image {global_idx}/{len(images)}: ")
                    
                    slice_prompt = f"""Analyze this CT slice #{global_idx} from a medical study.

{f"Study Context: {study_context}" if study_context else ""}

Please provide:
1. Anatomical structures visible in this slice
2. Any pathological findings
3. Clinical significance of findings
4. Slice position and anatomical level

Focus on medically relevant observations. Be concise but thorough."""
                    
                    analysis = self.analyze_medical_image(image_data, slice_prompt)
                    
                    if analysis:
                        individual_analyses.append(f"=== CT SLICE {global_idx} ===\n{analysis}")
                        show_success("✅")
                        log_to_file(f"CT Slice {global_idx} analysis successful")
                    else:
                        show_error("❌")
                        log_to_file(f"CT Slice {global_idx} analysis failed")
                
                # Пауза между пакетами для стабильности GPU
                if batch_idx < total_batches - 1:
                    show_warning("⏸️ Пауза между пакетами для стабильности GPU...")
                    log_to_file("⏸️ Pause between batches for GPU stability...")
                    time.sleep(3)  # Увеличиваем паузу для GPU
            
            if not individual_analyses:
                return None
            
            show_info(f"📋 Создание общего отчёта из {len(individual_analyses)} проанализированных изображений...")
            log_to_file(f"Creating comprehensive study report from {len(individual_analyses)} analyzed images...")
            
            # Создаём общий анализ исследования
            study_summary = self.analyze_medical_text(
                f"""Based on the following comprehensive CT study analysis, provide a detailed radiology report:

ANALYZED SLICES: {len(individual_analyses)} of {len(images)} total images

{chr(10).join(individual_analyses)}

Study Context: {study_context}

Please provide a structured radiology report including:

1. TECHNIQUE AND QUALITY
2. FINDINGS BY ANATOMICAL REGION
3. IMPRESSION/CONCLUSION
4. RECOMMENDATIONS

Format as a professional radiology report with detailed findings.""",
                "Complete CT study comprehensive analysis"
            )
            
            # Объединяем результаты
            final_report = f"""=== MEDGEMMA COMPLETE CT STUDY ANALYSIS ===

STUDY DETAILS:
- Total images in study: {len(images)}
- Images successfully analyzed: {len(individual_analyses)}
- Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Model: MedGemma 4B (Google) - Vision + Text
- Analysis method: Direct image analysis
- Processing method: Batch processing for stability

=== INDIVIDUAL SLICE ANALYSES ===

{chr(10).join(individual_analyses)}

=== COMPREHENSIVE STUDY REPORT ===

{study_summary if study_summary else "Summary generation failed"}

=== END OF REPORT ==="""
            
            return final_report
            
        except Exception as e:
            show_error(f"❌ Ошибка анализа CT исследования: {e}")
            log_to_file(f"MedGemma CT study analysis error: {e}", "ERROR")
            
            # Специальная обработка ошибок CUDA
            if "CUDA" in str(e) or "NVML" in str(e):
                show_warning("🔧 Обнаружена ошибка CUDA - попытка восстановления...")
                log_to_file("CUDA error detected - attempting recovery...")
                
                # Очищаем всю CUDA память
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                show_info("💡 Рекомендации:")
                show_warning("   - Перезапустите программу")
                show_warning("   - Убедитесь что другие GPU процессы не используют память")
                show_warning("   - Рассмотрите уменьшение размера пакета")
                
            return None
    
    def analyze_medical_text(self, text: str, context: str = "") -> Optional[str]:
        """
        Анализирует медицинский текст с помощью MedGemma
        
        Args:
            text: Текст для анализа
            context: Дополнительный контекст
            
        Returns:
            Медицинский анализ от MedGemma
        """
        if not self.model or not self.processor:
            show_error("❌ Модель не загружена")
            log_to_file("Model not loaded for text analysis")
            return None
        
        try:
            # Формируем медицинский промпт
            if context:
                prompt = f"""Context: {context}

Medical Analysis Request: {text}

Please provide a detailed medical analysis including:
1. Clinical findings interpretation
2. Differential diagnosis considerations  
3. Recommended follow-up actions
4. Risk assessment

Analysis:"""
            else:
                prompt = f"""Medical Analysis Request: {text}

Please provide a detailed medical analysis including:
1. Clinical findings interpretation
2. Differential diagnosis considerations
3. Recommended follow-up actions  
4. Risk assessment

Analysis:"""
            
            # Подготавливаем сообщения для текстового анализа
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert medical AI assistant."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            
            # Обрабатываем входные данные
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16 if self.device != "cpu" else torch.float32)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Генерируем ответ
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.7 if self.device != "cpu" else 1.0
                )
                generation = generation[0][input_len:]
            
            # Декодируем ответ
            response = self.processor.decode(generation, skip_special_tokens=True)
            
            # Логируем полный ответ
            show_info("🔍 ПОЛНЫЙ ОТВЕТ MEDGEMMA (Текстовый анализ):")
            log_to_file("Full MedGemma response (Text Analysis):")
            log_to_file("=" * 50)
            log_to_file(response)
            log_to_file("=" * 50)
            
            return response.strip()
            
        except Exception as e:
            show_error(f"❌ Ошибка анализа MedGemma: {e}")
            log_to_file(f"MedGemma text analysis error: {e}", "ERROR")
            return None
    
    def analyze_radiology_finding(self, finding: str, image_context: str = "") -> Optional[str]:
        """
        Специализированный анализ радиологических находок
        
        Args:
            finding: Радиологическая находка
            image_context: Контекст изображения
            
        Returns:
            Медицинская интерпретация
        """
        prompt = f"""Radiology Finding: {finding}

{f"Image Context: {image_context}" if image_context else ""}

As a medical AI assistant specialized in radiology, please provide:

1. CLINICAL SIGNIFICANCE:
   - What does this finding indicate?
   - Severity assessment
   
2. DIFFERENTIAL DIAGNOSIS:
   - Most likely diagnoses
   - Alternative considerations
   
3. FOLLOW-UP RECOMMENDATIONS:
   - Additional imaging needed
   - Clinical correlation required
   - Urgent vs routine follow-up
   
4. PATIENT COUNSELING POINTS:
   - Key information for patient
   - Prognosis considerations"""
        
        return self.analyze_medical_text(prompt)


def test_medgemma():
    """Тест MedGemma клиента с изображениями"""
    show_step("🧪 ТЕСТ MEDGEMMA КЛИЕНТА (С ИЗОБРАЖЕНИЯМИ)")
    log_to_file("Starting MedGemma client test (with images)")
    
    try:
        # Инициализируем клиент
        client = MedGemmaClient()
        
        # Тест анализа изображений
        show_info("\n📊 Тест анализа изображений...")
        log_to_file("\n📊 MedGemma image analysis test...")
        
        from image_processor import ImageProcessor
        
        # Проверяем наличие изображений
        if not os.path.exists("input"):
            show_error("❌ Директория 'input' не найдена")
            log_to_file("Input directory 'input' not found")
            return
        
        # Загружаем изображения
        image_processor = ImageProcessor()
        images = image_processor.load_dicom_series("input")
        
        if not images:
            show_error("❌ Изображения не найдены")
            log_to_file("Images not found")
            return
        
        # Тестируем на первом изображении
        test_image = images[0]
        show_info(f"📷 Тестируем на изображении: {test_image.get('filename', 'unknown')}")
        log_to_file(f"Testing on image: {test_image.get('filename', 'unknown')}")
        
        result = client.analyze_medical_image(test_image)
        
        if result:
            show_success(f"\n✅ Анализ изображения получен (длина: {len(result)} символов)")
            log_to_file(f"\n✅ Image analysis received (length: {len(result)} characters)")
            show_info(f"📄 Первые 200 символов: {result[:200]}...")
            log_to_file(f"📄 First 200 characters: {result[:200]}...")
        else:
            show_error("❌ Анализ изображения не получен")
            log_to_file("Image analysis not received")
        
        # Тест текстового анализа
        show_info("\n📝 Тест текстового анализа...")
        log_to_file("\n📝 MedGemma text analysis test...")
        test_finding = "CT показывает увеличение плотности в правой доле печени"
        
        text_result = client.analyze_radiology_finding(test_finding)
        
        if text_result:
            show_success(f"✅ Текстовый анализ получен (длина: {len(text_result)} символов)")
            log_to_file(f"✅ Text analysis received (length: {len(text_result)} characters)")
            show_info(f"📄 Первые 200 символов: {text_result[:200]}...")
            log_to_file(f"📄 First 200 characters: {text_result[:200]}...")
        else:
            show_error("❌ Текстовый анализ не получен")
            log_to_file("Text analysis not received")
            
    except Exception as e:
        show_error(f"❌ Ошибка теста: {e}")
        log_to_file(f"Test error: {e}", "ERROR")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_medgemma() 