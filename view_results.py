#!/usr/bin/env python3
"""
CT Reader - Results Viewer
Utility to view and manage saved analysis results
"""

import os
import json
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
from progress_logger import show_step, show_success, show_error, show_info, show_warning

def list_saved_results() -> List[Dict[str, Any]]:
    """List all saved analysis results"""
    results = []
    output_dir = "output"
    
    if not os.path.exists(output_dir):
        return results
    
    # Find all session directories
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        
        if os.path.isdir(item_path) and item.startswith("session_"):
            session_id = item.replace("session_", "")
            
            # Check for session summary
            summary_file = os.path.join(item_path, "session_summary.json")
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    
                    results.append({
                        "session_id": session_id,
                        "session_dir": item_path,
                        "summary": summary,
                        "type": "session"
                    })
                except Exception as e:
                    show_warning(f"Ошибка чтения сессии {session_id}: {e}")
        
        # Find standalone report files
        elif item.endswith(".txt") and item.startswith("medgemma_analysis_"):
            try:
                # Parse filename to extract info
                parts = item.replace("medgemma_analysis_", "").replace(".txt", "").split("_")
                if len(parts) >= 3:
                    date_part = parts[0]
                    time_part = parts[1]
                    images_part = parts[2]
                    
                    results.append({
                        "filename": item,
                        "filepath": item_path,
                        "date": date_part,
                        "time": time_part,
                        "images": images_part,
                        "type": "standalone"
                    })
            except Exception as e:
                show_warning(f"Ошибка парсинга файла {item}: {e}")
    
    # Sort by date (newest first)
    results.sort(key=lambda x: x.get("session_id", x.get("date", "")), reverse=True)
    return results

def display_results_list(results: List[Dict[str, Any]]):
    """Display list of available results"""
    if not results:
        show_warning("Нет сохранённых результатов анализа")
        return
    
    print("\n=== СОХРАНЁННЫЕ РЕЗУЛЬТАТЫ АНАЛИЗА ===")
    
    session_count = 0
    standalone_count = 0
    
    for i, result in enumerate(results, 1):
        if result["type"] == "session":
            session_count += 1
            summary = result["summary"]
            session_id = result["session_id"]
            
            print(f"\n{i}. 📁 СЕССИЯ: {session_id}")
            print(f"   📊 Изображений: {summary.get('processed_images', 'N/A')}/{summary.get('total_images', 'N/A')}")
            print(f"   ✅ Успешность: {summary.get('success_rate', 0):.1f}%")
            print(f"   🏥 Анализатор: {summary.get('analyzer', 'N/A')}")
            print(f"   📦 Размер батча: {summary.get('batch_size', 'N/A')}")
            print(f"   📅 Статус: {summary.get('status', 'N/A')}")
            
        elif result["type"] == "standalone":
            standalone_count += 1
            print(f"\n{i}. 📄 ФАЙЛ: {result['filename']}")
            print(f"   📅 Дата: {result['date']}")
            print(f"   🕐 Время: {result['time']}")
            print(f"   🖼️ Изображения: {result['images']}")
    
    print(f"\n📊 ИТОГО: {session_count} сессий, {standalone_count} отдельных файлов")

def view_session_details(session_dir: str, session_id: str):
    """View detailed information about a session"""
    print(f"\n=== ДЕТАЛИ СЕССИИ: {session_id} ===")
    
    # Load session summary
    summary_file = os.path.join(session_dir, "session_summary.json")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            print(f"📅 Начало: {summary.get('start_time', 'N/A')}")
            print(f"📅 Конец: {summary.get('end_time', 'N/A')}")
            print(f"📊 Всего изображений: {summary.get('total_images', 'N/A')}")
            print(f"✅ Обработано: {summary.get('processed_images', 'N/A')}")
            print(f"📈 Успешность: {summary.get('success_rate', 0):.1f}%")
            print(f"🏥 Анализатор: {summary.get('analyzer', 'N/A')}")
            print(f"📦 Размер батча: {summary.get('batch_size', 'N/A')}")
            print(f"📋 Статус: {summary.get('status', 'N/A')}")
            
        except Exception as e:
            show_error(f"Ошибка чтения сводки сессии: {e}")
    
    # List all files in session
    print(f"\n📁 ФАЙЛЫ В СЕССИИ:")
    try:
        files = os.listdir(session_dir)
        files.sort()
        
        for file in files:
            file_path = os.path.join(session_dir, file)
            file_size = os.path.getsize(file_path)
            
            if file.endswith('.json'):
                print(f"   📊 {file} ({file_size:,} байт)")
            elif file.endswith('.txt'):
                print(f"   📄 {file} ({file_size:,} байт)")
            else:
                print(f"   📋 {file} ({file_size:,} байт)")
                
    except Exception as e:
        show_error(f"Ошибка чтения файлов сессии: {e}")

def view_file_content(filepath: str, lines: int = 50):
    """View content of a file (first N lines)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines_list = content.split('\n')
        
        print(f"\n=== СОДЕРЖИМОЕ ФАЙЛА: {os.path.basename(filepath)} ===")
        print(f"📊 Размер файла: {len(content):,} символов")
        print(f"📄 Строк: {len(lines_list):,}")
        
        if len(lines_list) > lines:
            print(f"📋 Показаны первые {lines} строк:")
            print('\n'.join(lines_list[:lines]))
            print(f"\n... (ещё {len(lines_list) - lines} строк)")
        else:
            print(f"📋 Полное содержимое:")
            print(content)
            
    except Exception as e:
        show_error(f"Ошибка чтения файла: {e}")

def main():
    """Main function"""
    print("=== CT Reader - Просмотр результатов ===")
    
    # List all results
    results = list_saved_results()
    display_results_list(results)
    
    if not results:
        return
    
    # Interactive menu
    while True:
        print("\n=== МЕНЮ ===")
        print("1. Показать детали сессии")
        print("2. Просмотреть содержимое файла")
        print("3. Обновить список результатов")
        print("4. Выход")
        
        try:
            choice = input("\nВыберите действие (1-4): ").strip()
            
            if choice == "1":
                # View session details
                session_results = [r for r in results if r["type"] == "session"]
                if not session_results:
                    show_warning("Нет доступных сессий")
                    continue
                
                print("\nДоступные сессии:")
                for i, result in enumerate(session_results, 1):
                    print(f"{i}. {result['session_id']}")
                
                try:
                    session_choice = int(input("Выберите сессию: ")) - 1
                    if 0 <= session_choice < len(session_results):
                        selected = session_results[session_choice]
                        view_session_details(selected["session_dir"], selected["session_id"])
                    else:
                        show_warning("Неверный выбор")
                except ValueError:
                    show_warning("Введите число")
                    
            elif choice == "2":
                # View file content
                print("\nДоступные файлы:")
                all_files = []
                
                for result in results:
                    if result["type"] == "session":
                        session_dir = result["session_dir"]
                        try:
                            for file in os.listdir(session_dir):
                                if file.endswith(('.txt', '.json')):
                                    all_files.append({
                                        "name": f"{result['session_id']}/{file}",
                                        "path": os.path.join(session_dir, file)
                                    })
                        except:
                            pass
                    elif result["type"] == "standalone":
                        all_files.append({
                            "name": result["filename"],
                            "path": result["filepath"]
                        })
                
                if not all_files:
                    show_warning("Нет доступных файлов")
                    continue
                
                for i, file_info in enumerate(all_files, 1):
                    print(f"{i}. {file_info['name']}")
                
                try:
                    file_choice = int(input("Выберите файл: ")) - 1
                    if 0 <= file_choice < len(all_files):
                        selected_file = all_files[file_choice]
                        view_file_content(selected_file["path"])
                    else:
                        show_warning("Неверный выбор")
                except ValueError:
                    show_warning("Введите число")
                    
            elif choice == "3":
                # Refresh results
                results = list_saved_results()
                display_results_list(results)
                
            elif choice == "4":
                show_success("До свидания!")
                break
                
            else:
                show_warning("Неверный выбор. Введите 1-4")
                
        except KeyboardInterrupt:
            show_success("\nДо свидания!")
            break
        except Exception as e:
            show_error(f"Ошибка: {e}")

if __name__ == "__main__":
    main() 