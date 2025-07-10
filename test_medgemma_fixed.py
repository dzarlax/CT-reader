#!/usr/bin/env python3
"""
Тест исправленного MedGemma анализатора
"""

import os
from image_processor import ImageProcessor
from medgemma_analyzer import MedGemmaAnalyzer

def test_medgemma_analyzer():
    """Тестирует MedGemma анализатор с реальными данными"""
    print("🧪 ТЕСТ ИСПРАВЛЕННОГО MEDGEMMA АНАЛИЗАТОРА")
    print("=" * 60)
    
    # Проверяем наличие изображений
    if not os.path.exists("input"):
        print("❌ Директория 'input' не найдена")
        return
    
    try:
        # Загружаем изображения
        print("📁 Загрузка изображений...")
        processor = ImageProcessor()
        images = processor.load_dicom_series("input")
        
        if not images:
            print("❌ Изображения не найдены")
            return
            
        print(f"✅ Загружено {len(images)} изображений")
        
        # Тестируем на 2 изображениях для быстроты
        test_images = images[:2]
        print(f"🔍 Тестируем на {len(test_images)} изображениях")
        
        # Проверяем структуру данных
        print("\n📊 Структура данных:")
        img = test_images[0]
        for key in img.keys():
            if key == 'base64_image':
                print(f"  ✅ {key}: {len(img[key])} символов base64")
            else:
                print(f"  ✅ {key}: {img[key]}")
        
        # Инициализируем анализатор
        print("\n🤖 Инициализация MedGemma анализатора...")
        analyzer = MedGemmaAnalyzer()
        
        # Запускаем анализ
        print("\n🔬 Запуск анализа...")
        result = analyzer.analyze_study(test_images, "Dog's full body CT scan from Serbia")
        
        if result:
            print("\n✅ Анализ завершён успешно!")
            print(f"📄 Длина отчёта: {len(result)} символов")
            print(f"\n🔍 Первые 500 символов отчёта:")
            print("-" * 50)
            print(result[:500] + "..." if len(result) > 500 else result)
            print("-" * 50)
        else:
            print("\n❌ Анализ не дал результатов")
            
    except Exception as e:
        print(f"\n❌ Ошибка теста: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_medgemma_analyzer() 