#!/usr/bin/env python3
"""
Скрипт проверки системы для CT Reader
Проверяет доступность GPU, памяти и зависимостей
"""

import os
import sys
import torch
import psutil
from datetime import datetime

def check_gpu():
    """Проверка GPU и CUDA"""
    print("🔧 ПРОВЕРКА GPU")
    print("=" * 40)
    
    # CUDA проверка
    cuda_available = torch.cuda.is_available()
    print(f"CUDA доступна: {'✅' if cuda_available else '❌'}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA устройств: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name}")
            print(f"    Память: {memory_gb:.1f}GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
            
            # Проверка доступной памяти
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            memory_free = torch.cuda.memory_reserved(i) / 1024**3
            memory_used = torch.cuda.memory_allocated(i) / 1024**3
            print(f"    Используется: {memory_used:.1f}GB")
            print(f"    Свободно: {memory_gb - memory_used:.1f}GB")
            
            # Рекомендации
            if memory_gb >= 8:
                print(f"    Статус: ✅ Подходит для MedGemma")
            elif memory_gb >= 4:
                print(f"    Статус: ⚠️ Ограниченно подходит")
            else:
                print(f"    Статус: ❌ Недостаточно памяти")
    
    # MPS проверка (для Mac)
    mps_available = torch.backends.mps.is_available()
    if mps_available:
        print(f"MPS доступна: ✅")
    
    print()

def check_memory():
    """Проверка системной памяти"""
    print("💾 ПРОВЕРКА СИСТЕМНОЙ ПАМЯТИ")
    print("=" * 40)
    
    memory = psutil.virtual_memory()
    memory_gb = memory.total / 1024**3
    memory_available_gb = memory.available / 1024**3
    memory_used_percent = memory.percent
    
    print(f"Общая память: {memory_gb:.1f}GB")
    print(f"Доступная память: {memory_available_gb:.1f}GB")
    print(f"Использовано: {memory_used_percent:.1f}%")
    
    if memory_gb >= 16:
        print("Статус: ✅ Достаточно памяти")
    elif memory_gb >= 8:
        print("Статус: ⚠️ Минимальная память")
    else:
        print("Статус: ❌ Недостаточно памяти")
    
    print()

def check_dependencies():
    """Проверка зависимостей"""
    print("📦 ПРОВЕРКА ЗАВИСИМОСТЕЙ")
    print("=" * 40)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("Pillow", "PIL"),
        ("pydicom", "PyDICOM"),
        ("numpy", "NumPy"),
        ("requests", "Requests")
    ]
    
    for package, name in dependencies:
        try:
            __import__(package)
            print(f"{name}: ✅")
        except ImportError:
            print(f"{name}: ❌ Не установлен")
    
    print()

def check_huggingface_token():
    """Проверка токена Hugging Face"""
    print("🔑 ПРОВЕРКА ТОКЕНА HUGGING FACE")
    print("=" * 40)
    
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        print(f"Токен найден: ✅")
        print(f"Токен начинается с: {token[:10]}...")
    else:
        print("Токен не найден: ❌")
        print("Создайте .env файл с HUGGINGFACE_TOKEN=your_token")
    
    print()

def check_directories():
    """Проверка директорий"""
    print("📁 ПРОВЕРКА ДИРЕКТОРИЙ")
    print("=" * 40)
    
    directories = ["input", "output", "context", "temp"]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"{directory}/: ✅")
        else:
            print(f"{directory}/: ❌ Будет создана автоматически")
    
    # Проверка DICOM файлов
    if os.path.exists("input"):
        dicom_files = []
        for root, dirs, files in os.walk("input"):
            for file in files:
                if file.lower().endswith(('.dcm', '.dicom')) or '.' not in file:
                    dicom_files.append(os.path.join(root, file))
        
        print(f"DICOM файлов найдено: {len(dicom_files)}")
    
    print()

def test_medgemma_init():
    """Тестирование инициализации MedGemma"""
    print("🏥 ТЕСТИРОВАНИЕ MEDGEMMA")
    print("=" * 40)
    
    try:
        from medgemma_client import MedGemmaClient
        print("Импорт MedGemma клиента: ✅")
        
        # Попытка инициализации (может быть медленной)
        print("Инициализация MedGemma клиента...")
        client = MedGemmaClient()
        print("MedGemma клиент инициализирован: ✅")
        
    except Exception as e:
        print(f"Ошибка MedGemma: ❌ {e}")
    
    print()

def main():
    """Основная функция проверки"""
    print("🔍 CT READER - ПРОВЕРКА СИСТЕМЫ")
    print("=" * 50)
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    # Выполняем все проверки
    check_gpu()
    check_memory()
    check_dependencies()
    check_huggingface_token()
    check_directories()
    
    # Опциональный тест MedGemma (может быть медленным)
    response = input("Протестировать инициализацию MedGemma? (y/n): ").lower()
    if response == 'y':
        test_medgemma_init()
    
    print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 50)

if __name__ == "__main__":
    main() 