#!/usr/bin/env python3
"""
MedGemma Client
Клиент для работы с MedGemma 4B - специализированной медицинской моделью от Google
Поддерживает анализ изображений и текста
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import base64
from io import BytesIO
from typing import Optional, Dict, Any, List
import logging
import os
from dotenv import load_dotenv
from datetime import datetime

# Загружаем переменные окружения из .env файла
load_dotenv()

class MedGemmaClient:
    """Клиент для MedGemma 4B - медицинской модели от Google с поддержкой изображений"""
    
    def __init__(self, model_name: str = "google/medgemma-4b-it"):
        """
        Инициализация MedGemma клиента
        
        Args:
            model_name: Название модели на Hugging Face
        """
        self.model_name = model_name
        self.device = self._get_device()
        self.processor = None
        self.model = None
        
        print(f"🔧 Инициализация MedGemma клиента...")
        print(f"📱 Устройство: {self.device}")
        
        # Проверяем наличие токена авторизации
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        if self.token:
            print("✅ Токен Hugging Face найден в .env файле")
            print(f"🔑 Токен начинается с: {self.token[:10]}...")
        else:
            print("❌ Токен Hugging Face не найден в .env файле!")
            print("💡 Добавьте HUGGINGFACE_TOKEN=your_token в .env файл")
            raise ValueError("Токен Hugging Face обязателен для работы с MedGemma")
        
        self._load_model()
    
    def _get_device(self) -> str:
        """Определяет оптимальное устройство для вычислений"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        else:
            return "cpu"
    
    def _load_model(self):
        """Загружает модель и процессор"""
        try:
            print(f"📥 Загрузка MedGemma модели: {self.model_name}")
            
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
            
            print(f"✅ MedGemma модель успешно загружена на {self.device}")
            print(f"🔧 Модель поддерживает: изображения + текст")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки MedGemma: {e}")
            print("💡 Убедитесь, что:")
            print("   1. У вас есть доступ к модели google/medgemma-4b-it")
            print("   2. Установлена переменная окружения HUGGINGFACE_TOKEN")
            print("   3. Токен имеет необходимые права доступа")
            print("   4. Установлена библиотека accelerate: pip install accelerate")
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
            print("❌ Модель не загружена")
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
            print("🔍 ПОЛНЫЙ ОТВЕТ MEDGEMMA (Анализ изображения):")
            print("=" * 50)
            print(response)
            print("=" * 50)
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Ошибка анализа изображения MedGemma: {e}")
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
        
        print(f"🔍 MedGemma анализ CT исследования ({len(images)} изображений)...")
        
        try:
            # Анализируем каждое изображение
            individual_analyses = []
            
            # Ограничиваем количество изображений для анализа
            max_images = min(len(images), 5)
            
            for i, image_data in enumerate(images[:max_images], 1):
                print(f"📊 Анализ изображения {i}/{max_images}: ", end="")
                
                slice_prompt = f"""Analyze this CT slice #{i} from a medical study.

{f"Study Context: {study_context}" if study_context else ""}

Please provide:
1. Anatomical structures visible in this slice
2. Any pathological findings
3. Clinical significance of findings
4. Relationship to adjacent slices (if applicable)

Focus on medically relevant observations."""
                
                analysis = self.analyze_medical_image(image_data, slice_prompt)
                
                if analysis:
                    individual_analyses.append(f"=== CT SLICE {i} ===\n{analysis}")
                    print("✅")
                else:
                    print("❌")
            
            if not individual_analyses:
                return None
            
            # Создаём общий анализ исследования
            study_summary = self.analyze_medical_text(
                f"""Based on the following CT study analysis, provide a comprehensive radiology report:

{chr(10).join(individual_analyses)}

Study Context: {study_context}

Please provide a structured radiology report including:

1. TECHNIQUE AND QUALITY
2. FINDINGS BY ANATOMICAL REGION
3. IMPRESSION/CONCLUSION
4. RECOMMENDATIONS

Format as a professional radiology report.""",
                "CT study comprehensive analysis"
            )
            
            # Объединяем результаты
            final_report = f"""=== MEDGEMMA CT STUDY ANALYSIS ===

STUDY DETAILS:
- Images analyzed: {len(individual_analyses)} of {len(images)} total
- Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Model: MedGemma 4B (Google) - Vision + Text
- Analysis method: Direct image analysis

=== INDIVIDUAL SLICE ANALYSES ===

{chr(10).join(individual_analyses)}

=== COMPREHENSIVE STUDY REPORT ===

{study_summary if study_summary else "Summary generation failed"}

=== END OF REPORT ==="""
            
            return final_report
            
        except Exception as e:
            print(f"❌ Ошибка анализа CT исследования: {e}")
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
            print("❌ Модель не загружена")
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
            print("🔍 ПОЛНЫЙ ОТВЕТ MEDGEMMA (Текстовый анализ):")
            print("=" * 50)
            print(response)
            print("=" * 50)
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Ошибка анализа MedGemma: {e}")
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
    print("🧪 ТЕСТ MEDGEMMA КЛИЕНТА (С ИЗОБРАЖЕНИЯМИ)")
    print("=" * 50)
    
    try:
        # Инициализируем клиент
        client = MedGemmaClient()
        
        # Тест анализа изображений
        print("\n📊 Тест анализа изображений...")
        
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
        
        # Тестируем на первом изображении
        test_image = images[0]
        print(f"📷 Тестируем на изображении: {test_image.get('filename', 'unknown')}")
        
        result = client.analyze_medical_image(test_image)
        
        if result:
            print(f"\n✅ Анализ изображения получен (длина: {len(result)} символов)")
            print(f"📄 Первые 200 символов: {result[:200]}...")
        else:
            print("❌ Анализ изображения не получен")
        
        # Тест текстового анализа
        print("\n📝 Тест текстового анализа...")
        test_finding = "CT показывает увеличение плотности в правой доле печени"
        
        text_result = client.analyze_radiology_finding(test_finding)
        
        if text_result:
            print(f"✅ Текстовый анализ получен (длина: {len(text_result)} символов)")
            print(f"📄 Первые 200 символов: {text_result[:200]}...")
        else:
            print("❌ Текстовый анализ не получен")
            
    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_medgemma() 