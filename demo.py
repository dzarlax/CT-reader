#!/usr/bin/env python3
"""
Demo Script - CT Analysis Modes Demonstration
Demonstrates different analysis modes with sample data
"""

import os
import sys
from datetime import datetime
from ct_analyzer import CTAnalyzer

def demo_med42_analysis():
    """Demonstrate Med42 specialized analysis"""
    print("\n=== ДЕМОНСТРАЦИЯ MED42 АНАЛИЗА ===")
    print("Использует специализированную медицинскую ИИ модель")
    
    try:
        analyzer = CTAnalyzer()
        
        # Check for input data
        if not analyzer.validate_input("input"):
            print("Ошибка: Нет DICOM-файлов в директории 'input'")
            return
        
        # Run Med42 analysis
        result = analyzer.analyze_directory("input", mode="med42")
        
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
        if not analyzer.validate_input("input"):
            print("Ошибка: Нет DICOM-файлов в директории 'input'")
            return
        
        # Run hybrid analysis
        result = analyzer.analyze_directory("input", mode="hybrid")
        
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
        if not analyzer.validate_input("input"):
            print("Ошибка: Нет DICOM-файлов в директории 'input'")
            return
        
        # Run Gemma 3 analysis
        result = analyzer.analyze_directory("input", mode="gemma")
        
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
        if not analyzer.validate_input("input"):
            print("Ошибка: Нет DICOM-файлов в директории 'input'")
            return
        
        # Run intelligent analysis
        result = analyzer.analyze_directory("input", mode="intelligent")
        
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
        if not analyzer.validate_input("input"):
            print("Ошибка: Нет DICOM-файлов в директории 'input'")
            return
        
        # Run comprehensive analysis
        result = analyzer.analyze_directory("input", mode="comprehensive")
        
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
    
    if not analyzer.validate_input("input"):
        print("Ошибка: Нет DICOM-файлов для сравнения")
        return
    
    modes = ["med42", "hybrid", "gemma", "intelligent"]
    results = {}
    
    for mode in modes:
        print(f"\nЗапуск анализа в режиме: {mode}")
        try:
            result = analyzer.analyze_directory("input", mode=mode)
            results[mode] = result
            
            if result:
                print(f"✅ {mode.upper()} - успешно")
            else:
                print(f"❌ {mode.upper()} - ошибка")
                
        except Exception as e:
            print(f"❌ {mode.upper()} - исключение: {e}")
            results[mode] = None
    
    # Display comparison summary
    print("\n=== СВОДКА СРАВНЕНИЯ ===")
    for mode, result in results.items():
        if result:
            print(f"{mode.upper()}:")
            print(f"  - Изображений: {result['image_count']}")
            print(f"  - Время: {result['timestamp']}")
            print(f"  - Длина анализа: {len(result['analysis'])} символов")
        else:
            print(f"{mode.upper()}: ОШИБКА")

def show_system_info():
    """Display system information"""
    print("=== ИНФОРМАЦИЯ О СИСТЕМЕ ===")
    print(f"Python версия: {sys.version}")
    print(f"Рабочая директория: {os.getcwd()}")
    print(f"Доступные режимы: {CTAnalyzer().get_available_modes()}")
    
    # Check input directory
    if os.path.exists("input"):
        dicom_count = 0
        for root, dirs, files in os.walk("input"):
            for file in files:
                if file.lower().endswith(('.dcm', '.dicom')) or '.' not in file:
                    dicom_count += 1
        print(f"DICOM-файлов в input/: {dicom_count}")
    else:
        print("Директория input/ не найдена")
    
    # Check output directory
    if os.path.exists("output"):
        output_files = len([f for f in os.listdir("output") if os.path.isfile(os.path.join("output", f))])
        print(f"Файлов результатов в output/: {output_files}")
    else:
        print("Директория output/ не найдена")

def main():
    """Main demo function"""
    print("=== CT READER - ДЕМОНСТРАЦИЯ СИСТЕМЫ АНАЛИЗА ===")
    
    # Show system info
    show_system_info()
    
    # Demo menu
    while True:
        print("\nВыберите демонстрацию:")
        print("1. Med42 специализированный анализ")
        print("2. Гибридный анализ (Llama Vision + Med42)")
        print("3. Gemma 3 анализ (улучшенный выбор изображений)")
        print("4. 🧠 Интеллектуальный анализ (трёхэтапный)")
        print("5. 🔍 Полный анализ (ВСЕ изображения с контекстом)")
        print("6. Сравнение всех режимов")
        print("7. 📋 Просмотр сессий полного анализа")
        print("8. Информация о системе")
        print("0. Выход")
        
        try:
            choice = input("\nВведите номер (0-8): ").strip()
            
            if choice == "0":
                print("Завершение демонстрации")
                break
            elif choice == "1":
                demo_med42_analysis()
            elif choice == "2":
                demo_hybrid_analysis()
            elif choice == "3":
                demo_gemma_analysis()
            elif choice == "4":
                demo_intelligent_analysis()
            elif choice == "5":
                demo_comprehensive_analysis()
            elif choice == "6":
                compare_analysis_modes()
            elif choice == "7":
                show_comprehensive_sessions()
            elif choice == "8":
                show_system_info()
            else:
                print("Неверный выбор. Введите число от 0 до 8")
                
        except KeyboardInterrupt:
            print("\nЗавершение демонстрации")
            break
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    main() 