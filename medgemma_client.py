#!/usr/bin/env python3
"""
MedGemma Client
Клиент для работы с MedGemma 4B - специализированной медицинской моделью от Google
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import base64
from io import BytesIO
from typing import Optional, Dict, Any
import logging
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

class MedGemmaClient:
    """Клиент для MedGemma 4B - медицинской модели от Google"""
    
    def __init__(self, model_name: str = "google/medgemma-4b-it"):
        """
        Инициализация MedGemma клиента
        
        Args:
            model_name: Название модели на Hugging Face
        """
        self.model_name = model_name
        self.device = self._get_device()
        self.tokenizer = None
        self.model = None
        self.max_length = 4096  # Максимальная длина токенов
        
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
        """Загружает модель и токенизатор"""
        try:
            print(f"📥 Загрузка MedGemma модели: {self.model_name}")
            
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.token,
                trust_remote_code=True
            )
            
            # Загружаем модель
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.token,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device != "cpu" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print(f"✅ MedGemma модель успешно загружена на {self.device}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки MedGemma: {e}")
            print("💡 Убедитесь, что:")
            print("   1. У вас есть доступ к модели google/medgemma-4b-it")
            print("   2. Установлена переменная окружения HUGGINGFACE_TOKEN")
            print("   3. Токен имеет необходимые права доступа")
            raise
    
    def analyze_medical_text(self, text: str, context: str = "") -> Optional[str]:
        """
        Анализирует медицинский текст с помощью MedGemma
        
        Args:
            text: Текст для анализа
            context: Дополнительный контекст
            
        Returns:
            Медицинский анализ от MedGemma
        """
        if not self.model or not self.tokenizer:
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
            
            # Токенизируем
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length - 512  # Оставляем место для ответа
            ).to(self.device)
            
            # Генерируем ответ
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Декодируем ответ
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Логируем полный ответ
            print("🔍 ПОЛНЫЙ ОТВЕТ MEDGEMMA:")
            print("=" * 50)
            print(response)
            print("=" * 50)
            
            return response
            
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
    """Тест MedGemma клиента"""
    print("🧪 ТЕСТ MEDGEMMA КЛИЕНТА")
    print("=" * 40)
    
    try:
        # Инициализируем клиент
        client = MedGemmaClient()
        
        # Тест медицинского анализа
        test_finding = "CT показывает увеличение плотности в правой доле печени"
        
        print(f"\n📝 Тестовая находка: {test_finding}")
        result = client.analyze_radiology_finding(test_finding)
        
        if result:
            print(f"\n✅ Анализ получен (длина: {len(result)} символов)")
            print(f"📄 Первые 200 символов: {result[:200]}...")
        else:
            print("❌ Анализ не получен")
            
    except Exception as e:
        print(f"❌ Ошибка теста: {e}")

if __name__ == "__main__":
    test_medgemma() 