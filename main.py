#!/usr/bin/env python3
"""
CT Reader - Main Application Entry Point
Advanced Medical Image Analysis System
"""

import os
import sys
from datetime import datetime
from ct_analyzer import CTAnalyzer
from progress_logger import show_step, show_success, show_error, show_info, show_warning, get_log_file

def main():
    """Main application entry point"""
    print("=== CT Reader - Advanced Medical Image Analysis ===")
    show_step("Инициализация системы анализа КТ-снимков")
    
    # Check if input directory exists and has DICOM files
    input_dir = "input"
    if not os.path.exists(input_dir):
        show_error(f"Директория {input_dir} не найдена!")
        show_info("Создайте директорию 'input' и поместите в неё DICOM-файлы")
        return
    
    # Count DICOM files
    dicom_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.dcm', '.dicom')) or not '.' in file:
                dicom_files.append(os.path.join(root, file))
    
    if not dicom_files:
        show_error(f"DICOM-файлы не найдены в директории {input_dir}")
        show_info("Поместите DICOM-файлы в директорию 'input'")
        return
    
    show_success(f"Найдено {len(dicom_files)} DICOM-файлов")
    
    # Get additional context from user
    print("\n=== ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ===")
    print("Предоставьте дополнительный контекст для более точного анализа:")
    print("(Например: возраст пациента, симптомы, область исследования, подозрения и т.д.)")
    print("Нажмите Enter для пропуска")
    
    try:
        user_context = input("\nВведите дополнительную информацию: ").strip()
        if user_context:
            show_success(f"Контекст добавлен: {user_context[:100]}{'...' if len(user_context) > 100 else ''}")
        else:
            user_context = ""
            show_info("Анализ без дополнительного контекста")
    except KeyboardInterrupt:
        show_warning("Отмена операции")
        return
    
    # Initialize analyzer
    try:
        show_step("Инициализация анализатора")
        analyzer = CTAnalyzer()
        show_success("Анализатор инициализирован (модели загружаются по требованию)")
    except Exception as e:
        show_error(f"Ошибка инициализации анализатора: {e}")
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
            show_warning("Пожалуйста, введите 1, 2 или 3")
        except KeyboardInterrupt:
            show_warning("Отмена операции")
            return
    
    # Map choice to analysis mode
    mode_map = {
        '1': 'medgemma',
        '2': 'med42', 
        '3': 'comprehensive'
    }
    
    analysis_mode = mode_map[choice]
    show_success(f"Выбран режим: {analysis_mode}")
    
    # Show logging info
    log_file = get_log_file()
    show_info(f"Детальные логи сохраняются в: {log_file}")
    
    # Run analysis
    try:
        show_step("Запуск анализа")
        result = analyzer.analyze_directory(input_dir, mode=analysis_mode, user_context=user_context)
        
        if result:
            show_success("Анализ завершён успешно!")
            print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===")
            print("Результаты отображены выше")
            
            # Show context info if provided
            if user_context:
                show_info(f"Использованный контекст: {user_context}")
                
            # Show log file location
            show_info(f"Полные логи сохранены в: {log_file}")
                
        else:
            show_warning("Анализ завершился без результатов")
            
    except Exception as e:
        show_error(f"Ошибка во время анализа: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 