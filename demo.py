#!/usr/bin/env python3
"""
Demo Script - CT Analysis Modes Demonstration
Demonstrates different analysis modes with sample data
"""

import os
import sys
from datetime import datetime
from ct_analyzer import CTAnalyzer

# =============================================================================
# КОНФИГУРАЦИЯ ПУТЕЙ
# =============================================================================

# Путь к директории с DICOM-файлами для анализа
# Измените этот путь для использования другой папки с изображениями
INPUT_DIRECTORY = "input"

# Альтернативные пути для разных наборов данных (раскомментируйте нужный):
# INPUT_DIRECTORY = "input_test"        # Для тестовых данных
# INPUT_DIRECTORY = "input_production"  # Для продакшн данных  
# INPUT_DIRECTORY = "samples"           # Для образцов
# INPUT_DIRECTORY = "/path/to/dicom"    # Абсолютный путь

print(f"📁 Используемая директория: {INPUT_DIRECTORY}")
print(f"📍 Полный путь: {os.path.abspath(INPUT_DIRECTORY)}")

# =============================================================================
# ФУНКЦИИ ДЕМОНСТРАЦИИ
# =============================================================================

def demo_med42_analysis():
    """Demonstrate Med42 specialized analysis"""
    print("\n=== ДЕМОНСТРАЦИЯ MED42 АНАЛИЗА ===")
    print("Использует специализированную медицинскую ИИ модель")
    
    try:
        analyzer = CTAnalyzer()
        
        # Check for input data
        if not analyzer.validate_input(INPUT_DIRECTORY):
            print(f"Ошибка: Нет DICOM-файлов в директории '{INPUT_DIRECTORY}'")
            return
        
        # Run Med42 analysis
        result = analyzer.analyze_directory(INPUT_DIRECTORY, mode="med42")
        
        if result:
            print("✅ Med42 анализ завершён успешно")
            print(f"Обработано изображений: {result['image_count']}")
            print(f"Время анализа: {result['timestamp']}")
        else:
            print("❌ Med42 анализ завершился с ошибкой")
            
    except Exception as e:
        print(f"Ошибка демонстрации Med42: {e}")

def demo_hybrid_analysis():
    """Demonstrate hybrid Llama Vision + Med42 analysis"""
    print("\n=== ДЕМОНСТРАЦИЯ ГИБРИДНОГО АНАЛИЗА ===")
    print("Комбинирует Llama Vision (анализ изображений) + Med42 (медицинская интерпретация)")
    
    try:
        analyzer = CTAnalyzer()
        
        # Check for input data
        if not analyzer.validate_input(INPUT_DIRECTORY):
            print(f"Ошибка: Нет DICOM-файлов в директории '{INPUT_DIRECTORY}'")
            return
        
        # Run hybrid analysis
        result = analyzer.analyze_directory(INPUT_DIRECTORY, mode="hybrid")
        
        if result:
            print("✅ Гибридный анализ завершён успешно")
            print(f"Обработано изображений: {result['image_count']}")
            print(f"Этапы анализа: {result.get('analysis_stages', 'N/A')}")
            print(f"Использовано токенов: {result.get('tokens_used', 'N/A')}")
            print(f"Время анализа: {result['timestamp']}")
        else:
            print("❌ Гибридный анализ завершился с ошибкой")
            
    except Exception as e:
        print(f"Ошибка демонстрации гибридного анализа: {e}")

def demo_gemma_analysis():
    """Demonstrate Gemma 3 analysis with enhanced image selection"""
    print("\n=== ДЕМОНСТРАЦИЯ GEMMA 3 АНАЛИЗА ===")
    print("Анализ с улучшенным выбором изображений и комплексной оценкой органов")
    
    try:
        analyzer = CTAnalyzer()
        
        # Check for input data
        if not analyzer.validate_input(INPUT_DIRECTORY):
            print(f"Ошибка: Нет DICOM-файлов в директории '{INPUT_DIRECTORY}'")
            return
        
        # Run Gemma 3 analysis
        result = analyzer.analyze_directory(INPUT_DIRECTORY, mode="gemma")
        
        if result:
            print("✅ Gemma 3 анализ завершён успешно")
            print(f"Обработано изображений: {result['image_count']}")
            print(f"Время анализа: {result['timestamp']}")
            
            # Show enhanced features
            print("\n=== ОСОБЕННОСТИ УЛУЧШЕННОГО АНАЛИЗА ===")
            print("- Анатомическое покрытие по регионам")
            print("- Комплексная оценка всех органов")
            print("- Увеличенное количество анализируемых изображений")
            print("- Приоритизация критических анатомических областей")
        else:
            print("❌ Gemma 3 анализ завершился с ошибкой")
            
    except Exception as e:
        print(f"Ошибка демонстрации Gemma 3: {e}")

def demo_intelligent_analysis():
    """Demonstrate intelligent three-stage analysis"""
    print("\n=== ДЕМОНСТРАЦИЯ ИНТЕЛЛЕКТУАЛЬНОГО АНАЛИЗА ===")
    print("Трёхэтапный анализ: 1) Определение субъекта 2) Картирование анатомии 3) Медицинский анализ")
    
    try:
        analyzer = CTAnalyzer()
        
        # Check for input data
        if not analyzer.validate_input(INPUT_DIRECTORY):
            print(f"Ошибка: Нет DICOM-файлов в директории '{INPUT_DIRECTORY}'")
            return
        
        # Run intelligent analysis
        result = analyzer.analyze_directory(INPUT_DIRECTORY, mode="intelligent")
        
        if result:
            print("✅ Интеллектуальный анализ завершён успешно")
            print(f"Обработано изображений: {result['image_count']}")
            print(f"Время анализа: {result['timestamp']}")
            
            # Show stages information
            stages = result.get('stages', {})
            if stages:
                print("\n=== РЕЗУЛЬТАТЫ ПО ЭТАПАМ ===")
                
                # Stage 1 info
                subject_info = stages.get('subject_identification', {})
                if subject_info:
                    print(f"📋 ЭТАП 1 - Тип субъекта: {subject_info.get('subject_type', 'неизвестно')}")
                    print(f"   Уверенность: {subject_info.get('confidence', 'неизвестно')}")
                
                # Stage 2 info  
                anatomy_info = stages.get('anatomical_mapping', {})
                if anatomy_info:
                    regions = len(anatomy_info.get('important_regions', []))
                    organs = len(anatomy_info.get('organ_locations', {}))
                    print(f"🗺️  ЭТАП 2 - Регионов: {regions}, Органов: {organs}")
                
                # Stage 3 info
                medical_info = stages.get('medical_analysis', {})
                if medical_info:
                    pathological = len(medical_info.get('pathological_findings', []))
                    organ_specific = len(medical_info.get('organ_specific_findings', []))
                    print(f"🏥 ЭТАП 3 - Патологий: {pathological}, Органных находок: {organ_specific}")
            
        else:
            print("❌ Интеллектуальный анализ завершился с ошибкой")
            
    except Exception as e:
        print(f"Ошибка демонстрации интеллектуального анализа: {e}")

def demo_comprehensive_analysis():
    """Demonstrate comprehensive analysis of ALL images"""
    print("\n=== ДЕМОНСТРАЦИЯ ПОЛНОГО АНАЛИЗА ===")
    print("Анализ ВСЕХ изображений с сохранением контекста")
    
    try:
        analyzer = CTAnalyzer()
        
        # Check for input data
        if not analyzer.validate_input(INPUT_DIRECTORY):
            print(f"Ошибка: Нет DICOM-файлов в директории '{INPUT_DIRECTORY}'")
            return
        
        # Run comprehensive analysis
        result = analyzer.analyze_directory(INPUT_DIRECTORY, mode="comprehensive")
        
        if result:
            print("✅ Полный анализ завершён успешно")
            print(f"Обработано изображений: {result['image_count']}")
            print(f"Время анализа: {result['timestamp']}")
            print(f"Сессия: {result.get('session_id', 'неизвестно')}")
            print(f"Контекст сохранён: {result.get('context_file', 'неизвестно')}")
            
            # Show comprehensive features
            print("\n=== ОСОБЕННОСТИ ПОЛНОГО АНАЛИЗА ===")
            print("- Анализ ВСЕХ изображений в исследовании")
            print("- Сохранение контекста между изображениями")
            print("- Пакетная обработка с накоплением знаний")
            print("- Сохранение промежуточных результатов")
            print("- Возможность возобновления анализа")
            print("- Полное логирование всех ответов ИИ")
        else:
            print("❌ Полный анализ завершился с ошибкой")
            
    except Exception as e:
        print(f"Ошибка демонстрации полного анализа: {e}")

def show_comprehensive_sessions():
    """Show available comprehensive analysis sessions"""
    from comprehensive_analyzer import ComprehensiveAnalyzer
    
    print("\n=== ДОСТУПНЫЕ СЕССИИ ПОЛНОГО АНАЛИЗА ===")
    
    analyzer = ComprehensiveAnalyzer()
    sessions = analyzer.list_sessions()
    
    if sessions:
        print(f"Найдено сессий: {len(sessions)}")
        for i, session_id in enumerate(sessions[:10], 1):  # Show last 10
            session_data = analyzer.load_session(session_id)
            if session_data:
                status = session_data.get('status', 'unknown')
                total_images = session_data.get('total_images', '?')
                processed = session_data.get('progress', {}).get('processed', '?')
                print(f"  {i}. {session_id} - {status} ({processed}/{total_images} изображений)")
            else:
                print(f"  {i}. {session_id} - ошибка загрузки")
    else:
        print("Сессий не найдено")
        print("Запустите полный анализ для создания первой сессии")

def compare_analysis_modes():
    """Compare all available analysis modes"""
    print("\n=== СРАВНЕНИЕ РЕЖИМОВ АНАЛИЗА ===")
    
    analyzer = CTAnalyzer()
    
    if not analyzer.validate_input(INPUT_DIRECTORY):
        print("Ошибка: Нет DICOM-файлов для сравнения")
        return
    
    # Получаем доступные режимы
    available_modes = analyzer.get_available_modes()
    
    # Исключаем comprehensive режим из сравнения (слишком долгий)
    modes = [mode for mode in available_modes if mode != "comprehensive"]
    
    print(f"Доступные режимы для сравнения: {', '.join(modes)}")
    
    results = {}
    
    for mode in modes:
        print(f"\nЗапуск анализа в режиме: {mode}")
        try:
            result = analyzer.analyze_directory(INPUT_DIRECTORY, mode=mode)
            if result:
                results[mode] = {
                    'success': True,
                    'image_count': result['image_count'],
                    'timestamp': result['timestamp'],
                    'summary': result.get('summary', 'Нет резюме')[:200]
                }
                print(f"✅ Режим {mode} завершён успешно")
            else:
                results[mode] = {'success': False, 'error': 'Нет результатов'}
                print(f"❌ Режим {mode} завершился с ошибкой")
        except Exception as e:
            results[mode] = {'success': False, 'error': str(e)}
            print(f"❌ Режим {mode} завершился с ошибкой: {e}")
    
    # Display comparison results
    print("\n=== РЕЗУЛЬТАТЫ СРАВНЕНИЯ ===")
    for mode, result in results.items():
        print(f"\n{mode.upper()}:")
        if result['success']:
            print(f"  ✅ Успешно обработано изображений: {result['image_count']}")
            print(f"  📅 Время: {result['timestamp']}")
            print(f"  📝 Краткое резюме: {result['summary']}...")
        else:
            print(f"  ❌ Ошибка: {result['error']}")
    
    print(f"\n=== РЕКОМЕНДАЦИИ ПО РЕЖИМАМ ===")
    print("• medgemma: 🏥 Специализированная медицинская модель Google (РЕКОМЕНДУЕТСЯ)")
    print("  - Прямой анализ изображений")
    print("  - Высокая точность медицинской интерпретации")
    print("  - Обрабатывает ВСЕ изображения")
    print("  - Лучшая производительность на медицинских задачах")
    print("")
    print("• med42: 📋 Быстрый специализированный медицинский анализ")
    print("  - Текстовая медицинская модель")
    print("  - Быстрая обработка")
    print("  - Хорошо для базового анализа")
    print("")
    print("• comprehensive: 🔍 Полный анализ с контекстом")
    print("  - Анализирует ВСЕ изображения")
    print("  - Сохраняет контекст между изображениями")
    print("  - Использует несколько моделей")
    print("  - Самый детальный анализ")

def show_system_info():
    """Display system information"""
    print("=== ИНФОРМАЦИЯ О СИСТЕМЕ ===")
    print(f"Python версия: {sys.version}")
    print(f"Рабочая директория: {os.getcwd()}")
    print(f"Доступные режимы: {CTAnalyzer().get_available_modes()}")
    
    # Check input directory
    if os.path.exists(INPUT_DIRECTORY):
        dicom_count = 0
        for root, dirs, files in os.walk(INPUT_DIRECTORY):
            for file in files:
                if file.lower().endswith(('.dcm', '.dicom')) or '.' not in file:
                    dicom_count += 1
        print(f"DICOM-файлов в {INPUT_DIRECTORY}: {dicom_count}")
    else:
        print(f"Директория {INPUT_DIRECTORY} не найдена")
    
    # Check output directory
    if os.path.exists("output"):
        output_files = len([f for f in os.listdir("output") if os.path.isfile(os.path.join("output", f))])
        print(f"Файлов результатов в output/: {output_files}")
    else:
        print("Директория output/ не найдена")

def demo_medgemma_analysis():
    """Demonstrate MedGemma specialized medical analysis"""
    print("\n=== ДЕМОНСТРАЦИЯ MEDGEMMA АНАЛИЗА ===")
    print("Использует специализированную медицинскую модель Google MedGemma 4B")
    print("Комбинирует визуальный анализ с медицинской интерпретацией")
    
    try:
        analyzer = CTAnalyzer()
        
        # Check for input data
        if not analyzer.validate_input(INPUT_DIRECTORY):
            print(f"Ошибка: Нет DICOM-файлов в директории '{INPUT_DIRECTORY}'")
            return
        
        # Run MedGemma analysis
        result = analyzer.analyze_directory(INPUT_DIRECTORY, mode="medgemma")
        
        if result:
            print("✅ MedGemma анализ завершён успешно")
            print(f"Обработано изображений: {result['image_count']}")
            print(f"Время анализа: {result['timestamp']}")
            
            # Show MedGemma features
            print("\n=== ОСОБЕННОСТИ MEDGEMMA АНАЛИЗА ===")
            print("- Специализированная медицинская модель Google")
            print("- Двухэтапный анализ: визуальный + медицинский")
            print("- Клинически ориентированные рекомендации")
            print("- Дифференциальная диагностика")
            print("- Оценка клинической значимости")
            print("- Рекомендации по дальнейшему обследованию")
        else:
            print("❌ MedGemma анализ завершился с ошибкой")
            
    except Exception as e:
        print(f"Ошибка демонстрации MedGemma: {e}")
        if "MedGemma анализатор недоступен" in str(e):
            print("💡 Убедитесь, что:")
            print("   1. Установлен токен HUGGINGFACE_TOKEN в .env файле")
            print("   2. У вас есть доступ к модели google/medgemma-4b-it")
            print("   3. MedGemma клиент правильно настроен")

def main():
    """Main demo function"""
    print("=== CT READER - ДЕМОНСТРАЦИЯ СИСТЕМЫ АНАЛИЗА ===")
    
    # Show system info
    show_system_info()
    
    # Demo menu
    while True:
        print("\nВыберите демонстрацию:")
        print("1. 🏥 MedGemma анализ (Google медицинская модель - РЕКОМЕНДУЕТСЯ)")
        print("2. Med42 специализированный анализ")
        print("3. 🔍 Полный анализ (ВСЕ изображения с контекстом)")
        print("4. Сравнение режимов")
        print("5. 📋 Просмотр сессий полного анализа")
        print("6. Информация о системе")
        print("0. Выход")
        
        try:
            choice = input("\nВведите номер (0-6): ").strip()
            
            if choice == "0":
                print("Завершение демонстрации")
                break
            elif choice == "1":
                demo_medgemma_analysis()
            elif choice == "2":
                demo_med42_analysis()
            elif choice == "3":
                demo_comprehensive_analysis()
            elif choice == "4":
                compare_analysis_modes()
            elif choice == "5":
                show_comprehensive_sessions()
            elif choice == "6":
                show_system_info()
            else:
                print("Неверный выбор, попробуйте снова")
                
        except KeyboardInterrupt:
            print("\nЗавершение демонстрации")
            break
        except Exception as e:
            print(f"Ошибка: {e}")
            continue

if __name__ == "__main__":
    main() 