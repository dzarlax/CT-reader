#!/usr/bin/env python3
"""
MedGemma Analyzer
Специализированный анализатор для медицинского анализа CT изображений с использованием MedGemma
"""

import os
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
import base64

# Попробуем импортировать MedGemma
try:
    from medgemma_client import MedGemmaClient
    MEDGEMMA_AVAILABLE = True
except ImportError as e:
    MEDGEMMA_AVAILABLE = False
    print(f"⚠️ MedGemma недоступна: {e}")


class MedGemmaAnalyzer:
    """Анализатор с использованием MedGemma для медицинского анализа"""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.vision_model = "llama3.2-vision:latest"  # Для получения визуального описания
        
        # Инициализируем MedGemma клиент
        if MEDGEMMA_AVAILABLE:
            try:
                self.medgemma_client = MedGemmaClient()
                print("✅ MedGemma анализатор инициализирован")
                self.use_medgemma = True
            except Exception as e:
                print(f"❌ Ошибка инициализации MedGemma: {e}")
                self.medgemma_client = None
                self.use_medgemma = False
                raise ValueError("MedGemma недоступна для анализа")
        else:
            raise ValueError("MedGemma клиент не найден")
    
    def analyze_study(self, images: List[Dict[str, Any]], user_context: str = "") -> Optional[Dict[str, Any]]:
        """
        Analyze complete CT study using MedGemma
        
        Args:
            images: List of DICOM images to analyze
            user_context: Additional context from user (symptoms, age, etc.)
            
        Returns:
            Complete analysis results
        """
        if not images:
            print("❌ Нет изображений для анализа")
            return None
            
        print(f"🏥 Запуск MedGemma анализа ({len(images)} изображений)")
        
        # Prepare study context
        study_context = "CT Study Analysis"
        if user_context:
            study_context += f"\n\nПредоставленный контекст: {user_context}"
            
        try:
            # Analyze using MedGemma client
            analysis_result = self.medgemma_client.analyze_ct_study(images, study_context)
            
            if analysis_result:
                print("✅ MedGemma анализ завершён успешно")
                
                # Return structured result
                return {
                    'mode': 'medgemma',
                    'model': 'MedGemma 4B (Google)',
                    'timestamp': datetime.now().isoformat(),
                    'image_count': len(images),
                    'user_context': user_context,
                    'analysis': analysis_result,
                    'success': True
                }
            else:
                print("❌ MedGemma анализ не дал результатов")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка MedGemma анализа: {e}")
            return None
    
    def _get_visual_description(self, image_data: Dict[str, Any], image_num: int) -> Optional[str]:
        """Получает визуальное описание изображения от Vision модели"""
        
        try:
            prompt = f"""Analyze this CT image #{image_num} and provide a detailed visual description.

Focus on:
1. Anatomical structures visible
2. Organ appearance and morphology
3. Tissue densities and contrast
4. Any abnormal findings or variations
5. Image quality and technical factors

Provide objective, detailed visual findings that can be used for medical interpretation."""
            
            payload = {
                "model": self.vision_model,
                "prompt": prompt,
                "images": [image_data['base64_image']],
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Более низкая температура для точности
                    "num_predict": 600
                }
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=180)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return None
                
        except Exception as e:
            print(f"Ошибка получения визуального описания: {e}")
            return None
    
    def _analyze_with_medgemma(self, visual_description: str, image_num: int) -> Optional[str]:
        """Анализирует визуальное описание с помощью MedGemma"""
        
        try:
            # Формируем специализированный медицинский промпт
            medical_prompt = f"""CT Image #{image_num} Medical Analysis

VISUAL FINDINGS:
{visual_description}

As a specialized medical AI, please provide comprehensive medical interpretation:

1. ANATOMICAL ASSESSMENT:
   - Identify anatomical structures and regions
   - Assess normal vs abnormal anatomy
   
2. PATHOLOGICAL EVALUATION:
   - Identify any pathological findings
   - Assess severity and clinical significance
   
3. DIFFERENTIAL DIAGNOSIS:
   - List possible diagnoses based on findings
   - Prioritize by likelihood
   
4. CLINICAL RECOMMENDATIONS:
   - Suggest additional imaging if needed
   - Recommend clinical correlation
   - Indicate urgency level

Please provide detailed, clinically relevant analysis."""
            
            # Используем MedGemma для анализа
            analysis = self.medgemma_client.analyze_radiology_finding(
                visual_description,
                f"CT image #{image_num} analysis"
            )
            
            return analysis
            
        except Exception as e:
            print(f"Ошибка MedGemma анализа: {e}")
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