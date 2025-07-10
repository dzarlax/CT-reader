#!/usr/bin/env python3
"""
Диагностический скрипт для проверки доступа к HuggingFace модели
Запустите этот скрипт на сервере для диагностики проблем
"""

import os
import sys
from dotenv import load_dotenv

def main():
    """Основная функция диагностики"""
    print("🚀 Диагностика HuggingFace...")
    print("=" * 50)
    
    # Проверка .env файла
    print("📁 Проверка .env файла:")
    if os.path.exists('.env'):
        print("✅ .env файл найден")
        with open('.env', 'r') as f:
            content = f.read()
            print(f"📄 Содержимое .env файла:")
            print(content)
    else:
        print("❌ .env файл не найден")
    
    # Загрузка переменных окружения
    load_dotenv()
    
    # Проверка токена
    print("\n🔑 Проверка токена:")
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token:
        print(f"✅ Токен найден: {token[:10]}...")
        print(f"📏 Длина токена: {len(token)} символов")
    else:
        print("❌ Токен не найден в переменных окружения")
        return
    
    # Проверка доступа к HuggingFace
    print("\n🌐 Проверка доступа к HuggingFace:")
    try:
        from huggingface_hub import HfApi, login
        
        # Попытка логина
        print("🔐 Попытка аутентификации...")
        login(token=token, add_to_git_credential=False)
        print("✅ Аутентификация успешна")
        
        # Проверка доступа к модели
        print("🤖 Проверка доступа к модели google/medgemma-4b-it...")
        api = HfApi(token=token)
        model_info = api.model_info('google/medgemma-4b-it')
        print("✅ Доступ к модели есть")
        print(f"📊 Модель: {model_info.modelId}")
        
    except Exception as e:
        print(f"❌ Ошибка доступа: {e}")
        return
    
    # Проверка transformers
    print("\n🔧 Проверка библиотеки transformers:")
    try:
        import transformers
        print(f"✅ Transformers версия: {transformers.__version__}")
        
        # Попытка загрузки токенизатора
        print("🔤 Проверка токенизатора...")
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            'google/medgemma-4b-it',
            token=token,
            trust_remote_code=True
        )
        print("✅ Токенизатор загружен успешно")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки токенизатора: {e}")
        return
    
    print("\n✅ Все проверки пройдены успешно!")

if __name__ == "__main__":
    main() 