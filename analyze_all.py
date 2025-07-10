#!/usr/bin/env python3
"""
CT Reader - Quick Analysis Script
Analyze ALL images without interactive prompts
"""

import os
import sys
from datetime import datetime
from ct_analyzer import CTAnalyzer
from progress_logger import show_step, show_success, show_error, show_info, show_warning, get_log_file

def main():
    """Quick analysis of all images"""
    print("=== CT Reader - Анализ ВСЕХ изображений ===")
    show_step("Быстрый запуск анализа всех изображений")
    
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
    
    # Quick settings for processing ALL images
    max_images = None  # Process ALL images
    enable_parallel = True
    batch_size = 5  # Conservative batch size
    
    # Default context
    user_context = "Полный анализ всех изображений CT исследования"
    
    # Initialize analyzer with settings
    try:
        show_step("Инициализация анализатора")
        analyzer = CTAnalyzer(
            max_images_for_medgemma=max_images,
            enable_parallel=enable_parallel,
            batch_size=batch_size
        )
        show_success("Анализатор инициализирован")
    except Exception as e:
        show_error(f"Ошибка инициализации анализатора: {e}")
        return
    
    # Use MedGemma mode by default
    analysis_mode = 'medgemma'
    
    # Show final configuration
    print("\n=== КОНФИГУРАЦИЯ БЫСТРОГО АНАЛИЗА ===")
    show_info(f"📁 DICOM файлов: {len(dicom_files)}")
    show_info(f"🔍 Режим анализа: {analysis_mode}")
    show_info(f"🖼️ Изображений: ВСЕ ({len(dicom_files)})")
    show_info(f"📦 Размер батча: {batch_size}")
    show_info(f"⚡ Параллелизация: включена")
    show_info(f"📝 Контекст: {user_context}")
    
    # Show logging info
    log_file = get_log_file()
    show_info(f"📋 Логи: {log_file}")
    
    print("\n" + "="*60)
    show_step("Начинаем анализ всех изображений...")
    
    # Run analysis
    try:
        start_time = datetime.now()
        result = analyzer.analyze_directory(input_dir, mode=analysis_mode, user_context=user_context)
        end_time = datetime.now()
        
        if result:
            duration = end_time - start_time
            show_success(f"🎉 Анализ завершён успешно! Время: {duration}")
            print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===")
            print("Результаты отображены выше")
            
            # Show summary
            show_info(f"📊 Обработано изображений: {len(dicom_files)}")
            show_info(f"⏱️ Время анализа: {duration}")
            show_info(f"📋 Полные логи: {log_file}")
                
        else:
            show_warning("Анализ завершился без результатов")
            
    except Exception as e:
        show_error(f"Ошибка во время анализа: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 