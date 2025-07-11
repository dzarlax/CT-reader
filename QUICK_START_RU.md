# CT Reader - Быстрый старт

## 🚀 Анализ ВСЕХ изображений

Теперь система может анализировать **ВСЕ** изображения в исследовании, а не только первые 5-10!

### 1. Быстрый запуск (рекомендуется)

```bash
python analyze_all.py
```

**Что делает:**
- ✅ Анализирует **ВСЕ** DICOM файлы в папке `input/`
- ✅ Использует MedGemma для медицинского анализа
- ✅ Автоматически управляет памятью
- ✅ Показывает прогресс в реальном времени

### 2. Интерактивный режим с настройками

```bash
python main.py
```

**Позволяет настроить:**
- 🖼️ Количество изображений (все или ограничение)
- 📦 Размер батча (для управления памятью)
- 🏥 Режим анализа (MedGemma, Med42, Comprehensive)
- 📝 Дополнительный медицинский контекст

## ⚙️ Настройки производительности

### Для больших исследований (1000+ изображений)

- **Консервативно** (мало памяти): размер батча = 3
- **Сбалансированно** (рекомендуется): размер батча = 5
- **Агрессивно** (много памяти): размер батча = 10

### Время обработки

- **С GPU**: ~30-60 секунд на изображение
- **Без GPU**: ~2-3 минуты на изображение
- **1000 изображений**: ~8-50 часов (зависит от железа)

## 🛠️ Системные требования

### Память

- **Малые исследования** (< 100 изображений): 8GB RAM
- **Средние исследования** (100-500 изображений): 16GB RAM
- **Большие исследования** (500+ изображений): 32GB RAM

### GPU

- **NVIDIA GPU**: CUDA поддержка (быстрее)
- **Apple Silicon**: MPS поддержка (M1/M2 Mac)
- **CPU**: Работает, но медленнее

## 📊 Пример использования

### У вас есть 1797 DICOM файлов в папке `input/`

```bash
# Быстрый анализ всех изображений
python analyze_all.py

# Результат:
# ✅ Найдено 1797 DICOM-файлов
# 🔧 Настройки: макс. изображений = все, размер батча = 5
# 📦 Обработка батча 1/360 (5 изображений)
# 📦 Обработка батча 2/360 (5 изображений)
# ...
# 🎉 Анализ завершён! Обработано 1797 изображений
```

## 🔧 Оптимизация

### Если не хватает памяти:

1. Уменьшите размер батча до 3
2. Закройте другие приложения
3. Рассмотрите обработку частями

### Если медленно работает:

1. Проверьте драйверы GPU
2. Убедитесь что CUDA/MPS настроены
3. Попробуйте меньший размер батча

## 📋 Логи и результаты

- **Прогресс**: Отображается в консоли
- **Детальные логи**: Сохраняются в `output/ct_analysis_YYYY-MM-DD_HH-MM-SS.log`
- **Результаты**: Полный медицинский анализ каждого изображения

### 📁 Где хранятся результаты анализа?

При анализе 1000+ изображений результаты автоматически сохраняются в нескольких форматах:

#### 1. Сессии (для больших анализов)
```
output/
├── session_20250710_143022/          # Папка сессии
│   ├── session_summary.json          # Сводка сессии
│   ├── progress.json                  # Текущий прогресс
│   ├── analyses_batch_001.txt         # Результаты 1-го батча
│   ├── analyses_batch_002.txt         # Результаты 2-го батча
│   └── ...                           # Результаты всех батчей
```

#### 2. Итоговые отчёты
```
output/
├── medgemma_analysis_2025-07-10_14-30-22_1797images.txt  # Полный отчёт
├── medgemma_analysis_2025-07-10_14-30-22_1797images.json # Данные в JSON
└── ...
```

#### 3. Просмотр результатов
```bash
# Утилита для просмотра сохранённых результатов
python view_results.py
```

**Что показывает view_results.py:**
- 📊 Список всех сессий и файлов
- 📈 Статистика по каждой сессии
- 📄 Просмотр содержимого любого файла
- 🔍 Детали каждого анализа

### 💾 Преимущества сохранения

✅ **Промежуточные результаты** - сохраняются после каждого батча  
✅ **Защита от потерь** - если анализ прервётся, данные останутся  
✅ **Полная история** - все результаты сохраняются навсегда  
✅ **Удобный просмотр** - специальная утилита для просмотра  
✅ **Форматы данных** - текст для чтения, JSON для программ  

### 📊 Пример структуры для 1797 изображений

```
output/
├── session_20250710_143022/
│   ├── session_summary.json          # Сводка: 1797 изображений, 100% успех
│   ├── analyses_batch_001.txt         # Изображения 1-5
│   ├── analyses_batch_002.txt         # Изображения 6-10
│   ├── ...
│   └── analyses_batch_360.txt         # Изображения 1796-1797
├── medgemma_analysis_2025-07-10_14-30-22_1797images.txt  # Полный отчёт
├── medgemma_analysis_2025-07-10_14-30-22_1797images.json # JSON данные
└── ct_analysis_2025-07-10_14-30-22.log                   # Технические логи
```

## 🎯 Преимущества новой системы

✅ **Полный анализ** - все изображения, не только первые 5-10  
✅ **Умное управление памятью** - автоматическая очистка после каждого батча  
✅ **Пакетная обработка** - оптимизация для больших исследований  
✅ **Прогресс в реальном времени** - видите что происходит  
✅ **Настраиваемость** - можете адаптировать под свои ресурсы  

## 💡 Советы

1. **Начните с `analyze_all.py`** - это самый простой способ
2. **Для больших исследований** используйте размер батча 3-5
3. **Добавляйте медицинский контекст** для более точного анализа
4. **Следите за логами** - там много полезной информации
5. **Будьте терпеливы** - качественный анализ требует времени

Удачи в анализе! 🏥✨ 