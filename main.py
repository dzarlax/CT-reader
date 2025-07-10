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
    
    # Get additional context from user
    print("\n=== ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ===")
    print("Предоставьте дополнительный контекст для более точного анализа:")
    print("(Например: возраст пациента, симптомы, область исследования, подозрения и т.д.)")
    print("Нажмите Enter для пропуска")
    
    try:
        user_context = input("\nВведите дополнительную информацию: ").strip()
        if user_context:
            print(f"✅ Контекст добавлен: {user_context[:100]}{'...' if len(user_context) > 100 else ''}")
        else:
            user_context = ""
            print("⚪ Анализ без дополнительного контекста")
    except KeyboardInterrupt:
        print("\nОтмена операции")
        return
    
    # Initialize analyzer
    try:
        analyzer = CTAnalyzer()
        print("✅ Анализатор инициализирован (модели загружаются по требованию)")
    except Exception as e:
        print(f"Ошибка инициализации анализатора: {e}")
        return
    
    # Analysis mode selection
    print("\nДоступные режимы анализа:")
    print("1. 🏥 MedGemma - Специализированная медицинская ИИ модель Google (РЕКОМЕНДУЕТСЯ)")
    print("2. Med42 - Специализированная медицинская ИИ модель")
    print("3. 🔍 Comprehensive - Полный анализ всех изображений")
    
    while True:
        try:
            choice = input("\nВыберите режим анализа (1-3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("Пожалуйста, введите 1, 2 или 3")
        except KeyboardInterrupt:
            print("\nОтмена операции")
            return
    
    # Map choice to analysis mode
    mode_map = {
        '1': 'medgemma',
        '2': 'med42', 
        '3': 'comprehensive'
    }
    
    analysis_mode = mode_map[choice]
    print(f"\nВыбран режим: {analysis_mode}")
    
    # Run analysis
    try:
        print("Запуск анализа...")
        result = analyzer.analyze_directory(input_dir, mode=analysis_mode, user_context=user_context)
        
        if result:
            print("\n=== АНАЛИЗ ЗАВЕРШЁН ===")
            print("Результаты отображены выше")
            
            # Show context info if provided
            if user_context:
                print(f"\n📋 Использованный контекст: {user_context}")
                
        else:
            print("Анализ завершился без результатов")
            
    except Exception as e:
        print(f"Ошибка во время анализа: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 