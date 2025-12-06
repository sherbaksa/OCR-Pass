# Руководство по использованию PreprocessingService

## Описание

`PreprocessingService` — модуль для предобработки изображений паспортов РФ перед OCR-распознаванием. Включает функции выравнивания, шумоподавления, нормализации контраста и улучшения читаемости.

## Быстрый старт
```python
from backend.services.preprocessing_service import preprocessing_service

# Загрузка изображения
with open("passport.jpg", "rb") as f:
    image_bytes = f.read()

# Комплексная предобработка (рекомендуется)
processed = preprocessing_service.preprocess_image(
    image_bytes,
    apply_deskew=True,
    apply_denoise=True,
    apply_contrast=True,
    apply_sharpening=True,
    apply_binarization=False
)

# Сохранение результата
with open("passport_processed.jpg", "wb") as f:
    f.write(processed)
```

## Основные функции

### 1. Комплексная предобработка
```python
processed = preprocessing_service.preprocess_image(
    image_data,
    apply_deskew=True,         # Выравнивание наклона
    apply_denoise=True,        # Шумоподавление
    apply_contrast=True,       # Нормализация контраста
    apply_sharpening=True,     # Усиление резкости
    apply_binarization=False   # Бинаризация (ч/б)
)
```

**Рекомендации:**
- Для обычных фото паспортов: все флаги `True`, кроме `apply_binarization`
- Для сканов высокого качества: можно отключить `apply_denoise`
- Для очень плохого качества: включить `apply_binarization=True`

### 2. Выравнивание наклона (Deskew)
```python
import numpy as np

# Конвертация из bytes в numpy array
image = preprocessing_service._bytes_to_image(image_bytes)

# Выравнивание
deskewed = preprocessing_service.deskew_image(image)
```

**Параметры:**
- Автоматически определяет угол наклона текста
- Максимальный угол коррекции: 45°
- Использует метод минимального охватывающего прямоугольника

### 3. Шумоподавление
```python
# Легкое шумоподавление
denoised = preprocessing_service.denoise_image(image, strength="light")

# Среднее (по умолчанию)
denoised = preprocessing_service.denoise_image(image, strength="medium")

# Сильное (для очень зашумленных изображений)
denoised = preprocessing_service.denoise_image(image, strength="strong")
```

**Методы:**
- Non-local Means Denoising (cv2.fastNlMeansDenoisingColored)
- Сохраняет детали текста

### 4. Нормализация контраста
```python
# CLAHE (рекомендуется для паспортов)
normalized = preprocessing_service.normalize_contrast(image, method="clahe")

# Простая выравнивание гистограммы
normalized = preprocessing_service.normalize_contrast(image, method="hist_eq")

# Адаптивная нормализация
normalized = preprocessing_service.normalize_contrast(image, method="adaptive")
```

**Методы:**
- `clahe`: Contrast Limited Adaptive Histogram Equalization (лучше для паспортов)
- `hist_eq`: Простое выравнивание гистограммы
- `adaptive`: Адаптивная нормализация (scikit-image)

### 5. Усиление резкости
```python
# Слабое усиление
sharpened = preprocessing_service.sharpen_image(image, strength=0.5)

# Среднее (по умолчанию)
sharpened = preprocessing_service.sharpen_image(image, strength=1.0)

# Сильное
sharpened = preprocessing_service.sharpen_image(image, strength=1.5)
```

### 6. Бинаризация
```python
# Метод Otsu (автоматический порог)
binary = preprocessing_service.binarize_image(image, method="otsu")

# Адаптивная бинаризация (лучше для неравномерного освещения)
binary = preprocessing_service.binarize_image(image, method="adaptive")

# Локальная бинаризация
binary = preprocessing_service.binarize_image(image, method="local")
```

### 7. Удаление рамок
```python
cropped = preprocessing_service.remove_borders(image, threshold=10)
```

### 8. Изменение размера
```python
# Автоматическое масштабирование для OCR
resized = preprocessing_service.resize_for_ocr(
    image,
    target_dpi=300,
    max_dimension=3000
)
```

## Рекомендуемые пайплайны

### Для качественных фото с телефона
```python
processed = preprocessing_service.preprocess_image(
    image_bytes,
    apply_deskew=True,
    apply_denoise=True,
    apply_contrast=True,
    apply_sharpening=True,
    apply_binarization=False
)
```

### Для сканов
```python
processed = preprocessing_service.preprocess_image(
    image_bytes,
    apply_deskew=True,
    apply_denoise=False,      # Сканы обычно чистые
    apply_contrast=True,
    apply_sharpening=False,    # Сканы уже резкие
    apply_binarization=False
)
```

### Для низкокачественных фото
```python
processed = preprocessing_service.preprocess_image(
    image_bytes,
    apply_deskew=True,
    apply_denoise=True,
    apply_contrast=True,
    apply_sharpening=True,
    apply_binarization=True    # Включаем для плохого качества
)
```

## Производительность

- **Комплексная обработка**: ~2-4 секунды на изображение 2000x1500px
- **Только выравнивание**: ~0.3 секунды
- **Только шумоподавление**: ~1-2 секунды (зависит от strength)
- **Только нормализация**: ~0.2 секунды

## Логирование

Все операции логируются через `backend.core.logger`:
```python
2025-12-06 04:06:20 - passport_ocr - INFO - Начало предобработки изображения размером (1500, 2000, 3)
2025-12-06 04:06:20 - passport_ocr - INFO - Изображение повернуто на 2.34°
2025-12-06 04:06:22 - passport_ocr - INFO - Применено шумоподавление силой 'medium'
2025-12-06 04:06:22 - passport_ocr - INFO - Предобработка изображения завершена успешно
```

## Тестирование

Запуск тестов:
```bash
docker-compose exec backend python scripts/test_preprocessing.py
```

## Зависимости

- `opencv-python-headless` (4.8.1.78)
- `numpy` (1.24.4)
- `imutils` (0.5.4)
- `scikit-image` (0.22.0)
- `Pillow` (10.1.0)

## Системные библиотеки (Docker)
```dockerfile
libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```
