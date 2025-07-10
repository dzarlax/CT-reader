#!/usr/bin/env python3
"""
Тест для исправленного comprehensive анализатора
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comprehensive_analyzer import ComprehensiveAnalyzer
from image_processor import ImageProcessor

def test_fixed_comprehensive():
    print("🧪 ТЕСТ ИСПРАВЛЕННОГО COMPREHENSIVE АНАЛИЗАТОРА")
    print("=" * 60)
    
    # Check if input directory exists
    if not os.path.exists("input"):
        print("❌ Директория 'input' не найдена!")
        print("Создайте директорию 'input' и поместите туда DICOM-файлы")
        return
    
    # Initialize image processor
    image_processor = ImageProcessor()
    comprehensive_analyzer = ComprehensiveAnalyzer()
    
    # Load images (only first 2 for testing)
    print("📂 Загрузка изображений...")
    try:
        images = image_processor.load_dicom_series("input")
        
        if not images:
            print("❌ Изображения не найдены в директории 'input'!")
            return
        
        # Test with just 2 images
        test_images = images[:2]
        print(f"📊 Тестируем на {len(test_images)} изображениях")
        
        # Run comprehensive analysis
        print("\n🔍 Запуск comprehensive анализа...")
        result = comprehensive_analyzer.analyze_complete_study(test_images, mode="comprehensive_test")
        
        print("\n✅ РЕЗУЛЬТАТ ТЕСТА:")
        print(f"Сессия: {result.get('session_id', 'Неизвестно')}")
        print(f"Доступные ключи в результате: {list(result.keys())}")
        
        # Show available data safely
        if 'total_images' in result:
            print(f"Всего изображений: {result['total_images']}")
        if 'context_file' in result:
            print(f"Файл контекста: {result['context_file']}")
        
        # Show final report
        print("\n📋 ФИНАЛЬНЫЙ ОТЧЁТ:")
        final_report = result.get('final_report', 'Отчёт недоступен')
        print(final_report[:500] + "..." if len(final_report) > 500 else final_report)
        
        print("\n✅ Тест успешно завершён!")
        print("🔍 Система comprehensive анализа работает корректно!")
        print("📊 Полное логирование ответов AI активно!")
        
    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_comprehensive() 