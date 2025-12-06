# Руководство по использованию PaddleOCR провайдера

**Дата создания:** 06.12.2025  
**Версия:** 1.0  
**Статус:** Готов к использованию

---

## Обзор

PaddleOCR провайдер - это реализация OCR-движка на основе библиотеки PaddleOCR с поддержкой распознавания русского текста. Провайдер использует модели PP-OCRv4 и предоставляет унифицированный интерфейс для интеграции с системой распознавания паспортов.

---

## Основные возможности

- Распознавание русского текста
- Определение координат текстовых блоков
- Оценка уверенности распознавания
- Поддержка работы с bytes и numpy arrays
- Автоматическая классификация угла поворота текста
- Унифицированный формат вывода

---

## Технические характеристики

| Параметр | Значение |
|----------|----------|
| Модель | PP-OCRv4 |
| Язык | Русский (ru) |
| GPU | Опционально |
| Скорость | ~500-800ms на изображение 800x600 |
| Средняя уверенность | 85-90% на качественных изображениях |

---

## Установка и настройка

### Зависимости
```txt
paddleocr==2.8.1
paddlepaddle==2.6.2
```

### Переменные окружения

Добавьте в файл `.env`:
```bash
# PaddleOCR Settings
PADDLEOCR_USE_GPU=false
PADDLEOCR_LANG=ru
# PADDLEOCR_DET_MODEL_DIR=/app/models/paddleocr/det
# PADDLEOCR_REC_MODEL_DIR=/app/models/paddleocr/rec
# PADDLEOCR_CLS_MODEL_DIR=/app/models/paddleocr/cls
```

### Первый запуск

При первом запуске PaddleOCR автоматически скачает модели для русского языка (~200-300 MB). Это займет несколько минут в зависимости от скорости интернета.

---

## Использование

### Базовое использование
```python
from backend.providers import paddleocr_provider

# Инициализация (выполняется один раз)
paddleocr_provider.initialize()

# Проверка доступности
if paddleocr_provider.is_available():
    # Распознавание изображения
    with open('passport.jpg', 'rb') as f:
        image_data = f.read()
    
    result = paddleocr_provider.recognize(image_data)
    
    # Работа с результатом
    print(f"Текст: {result.full_text}")
    print(f"Уверенность: {result.average_confidence:.2%}")
    print(f"Блоков: {len(result.blocks)}")
```

### Работа с numpy array
```python
import numpy as np
from PIL import Image

# Загрузка изображения как numpy array
image = Image.open('passport.jpg')
numpy_array = np.array(image)
# Конвертация RGB -> BGR
numpy_array = numpy_array[:, :, ::-1]

# Распознавание
result = paddleocr_provider.recognize(numpy_array)
```

### Обработка результатов
```python
# Полный текст
full_text = result.full_text

# Перебор блоков
for block in result.blocks:
    print(f"Текст блока: {block.text}")
    print(f"Уверенность: {block.confidence:.2%}")
    
    # Координаты
    if block.bounding_box:
        bbox = block.bounding_box
        print(f"Координаты: ({bbox.x}, {bbox.y}) - {bbox.width}x{bbox.height}")
    
    # Строки в блоке
    for line in block.lines:
        print(f"  Строка: {line.text}")
        
        # Слова в строке
        for word in line.words:
            print(f"    Слово: {word.text} ({word.confidence:.2%})")

# Метаданные
metadata = result.metadata
print(f"Провайдер: {metadata.provider}")
print(f"Время обработки: {metadata.processing_time_ms:.2f}ms")
print(f"Размер: {metadata.image_width}x{metadata.image_height}")
print(f"Версия модели: {metadata.model_version}")

# Сырые данные (для отладки)
raw_data = result.raw_response
```

---

## Структура результата

### OCRResult
```python
class OCRResult:
    full_text: str                    # Полный распознанный текст
    blocks: List[OCRBlock]            # Блоки текста
    average_confidence: float         # Средняя уверенность (0-1)
    passport_fields: List[PassportField]  # Поля паспорта (пока пусто)
    metadata: OCRMetadata             # Метаданные процесса
    raw_response: Optional[Dict]      # Сырой ответ от PaddleOCR
    timestamp: datetime               # Время выполнения
```

### OCRBlock
```python
class OCRBlock:
    text: str                         # Текст блока
    confidence: float                 # Уверенность (0-1)
    lines: List[OCRLine]              # Строки в блоке
    bounding_box: Optional[BoundingBox]  # Координаты
    block_type: Optional[str]         # Тип блока
```

### BoundingBox
```python
class BoundingBox:
    x: int          # X координата левого верхнего угла
    y: int          # Y координата левого верхнего угла
    width: int      # Ширина области
    height: int     # Высота области
```

---

## Производительность

### Тесты на изображении 800x600px

| Операция | Время |
|----------|-------|
| Инициализация | ~15-20 секунд (при первом запуске) |
| Распознавание | 500-800ms |
| Пустое изображение | ~150ms |

### Рекомендации по оптимизации

1. **Предобработка изображений**
   - Используйте `preprocessing_service` перед OCR
   - Применяйте deskew для выравнивания
   - Нормализуйте контраст

2. **Размер изображений**
   - Оптимальный размер: 1500-2500px по большей стороне
   - Слишком маленькие (<1000px) - низкая точность
   - Слишком большие (>3000px) - медленная обработка

3. **GPU ускорение**
   - Установите `PADDLEOCR_USE_GPU=true` при наличии CUDA
   - Ускорение в 3-5 раз

---

## Типичные проблемы

### Проблема 1: Низкая уверенность распознавания

**Причина:** Плохое качество изображения  
**Решение:**
```python
from backend.services import preprocessing_service

# Предобработка перед OCR
preprocessed = preprocessing_service.preprocess_image(
    image_data,
    apply_deskew=True,
    apply_contrast=True,
    apply_sharpening=True
)

result = paddleocr_provider.recognize(preprocessed)
```

### Проблема 2: Не распознается текст

**Причина:** Изображение перевернуто или слишком темное  
**Решение:** Включите классификацию угла и нормализацию контраста

### Проблема 3: Медленная работа

**Причина:** Большой размер изображения  
**Решение:**
```python
from backend.services import preprocessing_service

# Изменение размера
resized = preprocessing_service.resize_for_ocr(
    image,
    target_dpi=300,
    max_dimension=2000
)
```

---

## Интеграция с preprocessing_service

### Рекомендуемый пайплайн
```python
from backend.services import preprocessing_service
from backend.providers import paddleocr_provider

# 1. Инициализация
paddleocr_provider.initialize()

# 2. Предобработка
preprocessed = preprocessing_service.preprocess_image(
    image_data,
    apply_deskew=True,          # Выравнивание
    apply_denoise=True,         # Шумоподавление
    apply_contrast=True,        # Нормализация контраста
    apply_sharpening=True,      # Усиление резкости
    apply_binarization=False    # Бинаризация (опционально)
)

# 3. OCR
result = paddleocr_provider.recognize(preprocessed)

# 4. Обработка результата
if result.average_confidence > 0.7:
    print("Высокое качество распознавания")
else:
    print("Низкое качество, требуется ручная проверка")
```

---

## Тестирование

### Запуск тестов
```bash
docker-compose exec backend python scripts/test_paddleocr.py
```

### Ожидаемые результаты
```
ТЕСТ 1: Инициализация PaddleOCR - ПРОЙДЕН
ТЕСТ 2: Распознавание текста - ПРОЙДЕН
ТЕСТ 3: Распознавание с numpy array - ПРОЙДЕН
ТЕСТ 4: Обработка пустого изображения - ПРОЙДЕН

Пройдено тестов: 4/4
ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО
```

---

## Следующие шаги

1. **Извлечение полей паспорта**
   - Создать модуль парсинга структурированных данных
   - Реализовать regexp для серии/номера
   - Валидация дат и кодов подразделений

2. **Интеграция с Google Vision API**
   - Создать провайдер `GoogleVisionProvider`
   - Сравнительное тестирование
   - Стратегия выбора провайдера

3. **OCR Service Layer**
   - Унифицированный сервис для всех провайдеров
   - Автоматический выбор лучшего провайдера
   - Кеширование результатов

---

## Ссылки

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/)
- [PP-OCRv4 Paper](https://arxiv.org/abs/2109.03144)

---

**Конец руководства**
