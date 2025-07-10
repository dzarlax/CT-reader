#!/usr/bin/env python3
"""
Hybrid Demo Script - Specialized Hybrid Analysis Demonstration
Dedicated demonstration of the Llama Vision + Med42 hybrid analysis system
"""

import os
import sys
from datetime import datetime
from ct_analyzer import CTAnalyzer

def main():
    """Main hybrid demo function"""
    print("=== CT READER - ГИБРИДНЫЙ АНАЛИЗ (LLAMA VISION + MED42) ===")
    print("Демонстрация комбинированного анализа КТ-снимков")
    print("Этап 1: Анализ изображений с Llama Vision")
    print("Этап 2: Медицинская интерпретация с Med42")
    print("Этап 3: Объединение результатов")
    
    # Check for input data
    input_dir = "input"
    if not os.path.exists(input_dir):
        print(f"\nОшибка: Директория {input_dir} не найдена!")
        print("Создайте директорию 'input' и поместите в неё DICOM-файлы")
        return
    
    # Count DICOM files
    dicom_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.dcm', '.dicom')) or not '.' in file:
                dicom_files.append(os.path.join(root, file))
    
    if not dicom_files:
        print(f"\nОшибка: DICOM-файлы не найдены в директории {input_dir}")
        print("Поместите DICOM-файлы в директорию 'input'")
        return
    
    print(f"\nНайдено {len(dicom_files)} DICOM-файлов")
    
    # Initialize analyzer
    try:
        print("\nИнициализация гибридной системы...")
        analyzer = CTAnalyzer()
        print("✅ Система инициализирована успешно")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return
    
    # Confirm analysis
    try:
        confirm = input("\nЗапустить гибридный анализ? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes', 'да', 'д']:
            print("Анализ отменён")
            return
    except KeyboardInterrupt:
        print("\nОтмена операции")
        return
    
    # Run hybrid analysis
    try:
        print("\n" + "="*50)
        print("ЗАПУСК ГИБРИДНОГО АНАЛИЗА")
        print("="*50)
        
        start_time = datetime.now()
        result = analyzer.analyze_directory(input_dir, mode="hybrid")
        end_time = datetime.now()
        
        if result:
            print("\n" + "="*50)
            print("✅ ГИБРИДНЫЙ АНАЛИЗ ЗАВЕРШЁН УСПЕШНО")
            print("="*50)
            
            # Display detailed results
            print(f"Обработано изображений: {result['image_count']}")
            print(f"Время выполнения: {end_time - start_time}")
            print(f"Этапы анализа: {result.get('analysis_stages', 'N/A')}")
            print(f"Использовано токенов: {result.get('tokens_used', 'N/A')}")
            print(f"Время анализа: {result['timestamp']}")
            
            # Show analysis summary
            if 'summary' in result:
                print(f"\n📋 КРАТКОЕ РЕЗЮМЕ:")
                print("-" * 40)
                summary = result['summary']
                if len(summary) > 300:
                    print(summary[:300] + "...")
                else:
                    print(summary)
            
            # Show where results are saved
            print(f"\n💾 РЕЗУЛЬТАТЫ СОХРАНЕНЫ:")
            print("- Директория: output/")
            print("- JSON файл: analysis_[timestamp].json")
            print("- Текстовый отчёт: report_[timestamp].txt")
            
            # Option to view full analysis
            try:
                view_full = input("\nПоказать полный анализ? (y/n): ").strip().lower()
                if view_full in ['y', 'yes', 'да', 'д']:
                    print("\n" + "="*60)
                    print("ПОЛНЫЙ АНАЛИЗ")
                    print("="*60)
                    print(result['analysis'])
            except KeyboardInterrupt:
                print("\nЗавершение просмотра")
                
        else:
            print("\n❌ АНАЛИЗ ЗАВЕРШИЛСЯ БЕЗ РЕЗУЛЬТАТОВ")
            
    except Exception as e:
        print(f"\n❌ ОШИБКА ВО ВРЕМЯ АНАЛИЗА: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 