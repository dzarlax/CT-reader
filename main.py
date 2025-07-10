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

def get_analysis_settings():
    """Get analysis settings from user"""
    print("\n=== НАСТРОЙКИ АНАЛИЗА ===")
    print("Настройте параметры анализа:")
    
    # Max images setting
    print("\n1. Количество изображений для анализа:")
    print("   - Введите число (например, 100) для ограничения")
    print("   - Нажмите Enter для анализа ВСЕХ изображений")
    
    max_images = None
    while True:
        try:
            user_input = input("Максимальное количество изображений (Enter = все): ").strip()
            if not user_input:
                max_images = None
                show_success("Будут проанализированы ВСЕ изображения")
                break
            else:
                max_images = int(user_input)
                if max_images > 0:
                    show_success(f"Будет проанализировано максимум {max_images} изображений")
                    break
                else:
                    show_warning("Введите положительное число")
        except ValueError:
            show_warning("Введите корректное число или нажмите Enter")
        except KeyboardInterrupt:
            show_warning("Отмена операции")
            return None, None, None
    
    # Batch size setting
    print("\n2. Размер батча (количество изображений, обрабатываемых одновременно):")
    print("   - Меньше = меньше памяти, медленнее")
    print("   - Больше = больше памяти, быстрее")
    print("   - Рекомендуется: 3-10")
    
    batch_size = 5
    while True:
        try:
            user_input = input("Размер батча (по умолчанию 5): ").strip()
            if not user_input:
                batch_size = 5
                break
            else:
                batch_size = int(user_input)
                if 1 <= batch_size <= 50:
                    break
                else:
                    show_warning("Введите число от 1 до 50")
        except ValueError:
            show_warning("Введите корректное число")
        except KeyboardInterrupt:
            show_warning("Отмена операции")
            return None, None, None
    
    show_success(f"Размер батча: {batch_size}")
    
    # Parallel processing (always enabled for now)
    enable_parallel = True
    show_info("Параллелизация: включена")
    
    return max_images, enable_parallel, batch_size

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
    
    # Get analysis settings
    max_images, enable_parallel, batch_size = get_analysis_settings()
    if max_images is None and enable_parallel is None and batch_size is None:
        return  # User cancelled
    
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
    
    # Initialize analyzer with settings
    try:
        show_step("Инициализация анализатора")
        analyzer = CTAnalyzer(
            max_images_for_medgemma=max_images,
            enable_parallel=enable_parallel,
            batch_size=batch_size
        )
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
    
    # Show final configuration
    print("\n=== ИТОГОВАЯ КОНФИГУРАЦИЯ ===")
    show_info(f"📁 DICOM файлов: {len(dicom_files)}")
    show_info(f"🔍 Режим анализа: {analysis_mode}")
    show_info(f"🖼️ Макс. изображений: {'все' if max_images is None else max_images}")
    show_info(f"📦 Размер батча: {batch_size}")
    show_info(f"⚡ Параллелизация: {'включена' if enable_parallel else 'выключена'}")
    if user_context:
        show_info(f"📝 Контекст: {user_context[:50]}{'...' if len(user_context) > 50 else ''}")
    
    # Show logging info
    log_file = get_log_file()
    show_info(f"📋 Логи: {log_file}")
    
    # Confirm start
    print("\n" + "="*50)
    try:
        input("Нажмите Enter для начала анализа или Ctrl+C для отмены...")
    except KeyboardInterrupt:
        show_warning("Анализ отменён пользователем")
        return
    
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