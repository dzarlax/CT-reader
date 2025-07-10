#!/usr/bin/env python3
"""
CT Reader - Main Application Entry Point
Advanced Medical Image Analysis System
"""

import os
import sys
from datetime import datetime
from ct_analyzer import CTAnalyzer

def main():
    """Main application entry point"""
    print("=== CT Reader - Advanced Medical Image Analysis ===")
    print("Инициализация системы анализа КТ-снимков...")
    
    # Check if input directory exists and has DICOM files
    input_dir = "input"
    if not os.path.exists(input_dir):
        print(f"Ошибка: Директория {input_dir} не найдена!")
        print("Создайте директорию 'input' и поместите в неё DICOM-файлы")
        return
    
    # Count DICOM files
    dicom_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.dcm', '.dicom')) or not '.' in file:
                dicom_files.append(os.path.join(root, file))
    
    if not dicom_files:
        print(f"Ошибка: DICOM-файлы не найдены в директории {input_dir}")
        print("Поместите DICOM-файлы в директорию 'input'")
        return
    
    print(f"Найдено {len(dicom_files)} DICOM-файлов")
    
    # Initialize analyzer
    try:
        analyzer = CTAnalyzer()
        print("Анализатор инициализирован успешно")
    except Exception as e:
        print(f"Ошибка инициализации анализатора: {e}")
        return
    
    # Analysis mode selection
    print("\nДоступные режимы анализа:")
    print("1. Med42 - Специализированная медицинская ИИ модель")
    print("2. Hybrid - Комбинированный анализ (Llama Vision + Med42)")
    
    while True:
        try:
            choice = input("\nВыберите режим анализа (1-2): ").strip()
            if choice in ['1', '2']:
                break
            print("Пожалуйста, введите 1 или 2")
        except KeyboardInterrupt:
            print("\nОтмена операции")
            return
    
    # Map choice to analysis mode
    mode_map = {
        '1': 'med42', 
        '2': 'hybrid'
    }
    
    analysis_mode = mode_map[choice]
    print(f"\nВыбран режим: {analysis_mode}")
    
    # Run analysis
    try:
        print("Запуск анализа...")
        result = analyzer.analyze_directory(input_dir, mode=analysis_mode)
        
        if result:
            print("\n=== АНАЛИЗ ЗАВЕРШЁН ===")
            print("Результаты сохранены в директории 'output'")
            
            # Display summary
            if 'summary' in result:
                print("\nКраткое резюме:")
                print(result['summary'][:500] + "..." if len(result['summary']) > 500 else result['summary'])
                
        else:
            print("Анализ завершился без результатов")
            
    except Exception as e:
        print(f"Ошибка во время анализа: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 