"""
Тестовый скрипт для проверки функциональности preprocessing_service
Проверяет все основные функции предобработки изображений
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.preprocessing_service import preprocessing_service
import numpy as np
from PIL import Image
import io


def create_test_image(width: int = 2000, height: int = 1500) -> bytes:
    """Создание тестового изображения"""
    # Создаем простое тестовое изображение с текстом
    img = Image.new('RGB', (width, height), color='white')
    
    # Добавляем шум
    pixels = np.array(img)
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(pixels)
    
    # Конвертируем в байты
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


def test_preprocessing_functions():
    """Тестирование всех функций предобработки"""
    
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ PREPROCESSING SERVICE")
    print("=" * 60)
    
    # Создаем тестовое изображение
    print("\n1. Создание тестового изображения...")
    test_image = create_test_image()
    print(f"   ✓ Тестовое изображение создано: {len(test_image)} байт")
    
    # Тест 1: Комплексная предобработка
    print("\n2. Тест комплексной предобработки...")
    try:
        result = preprocessing_service.preprocess_image(
            test_image,
            apply_deskew=True,
            apply_denoise=True,
            apply_contrast=True,
            apply_sharpening=True,
            apply_binarization=False
        )
        print(f"   ✓ Комплексная предобработка: {len(result)} байт")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        return False
    
    # Тест 2: Выравнивание
    print("\n3. Тест выравнивания (deskew)...")
    try:
        image_array = preprocessing_service._bytes_to_image(test_image)
        deskewed = preprocessing_service.deskew_image(image_array)
        print(f"   ✓ Выравнивание: {deskewed.shape}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        return False
    
    # Тест 3: Шумоподавление
    print("\n4. Тест шумоподавления...")
    try:
        image_array = preprocessing_service._bytes_to_image(test_image)
        
        for strength in ["light", "medium", "strong"]:
            denoised = preprocessing_service.denoise_image(image_array, strength=strength)
            print(f"   ✓ Шумоподавление ({strength}): {denoised.shape}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        return False
    
    # Тест 4: Нормализация контраста
    print("\n5. Тест нормализации контраста...")
    try:
        image_array = preprocessing_service._bytes_to_image(test_image)
        
        for method in ["clahe", "hist_eq", "adaptive"]:
            normalized = preprocessing_service.normalize_contrast(image_array, method=method)
            print(f"   ✓ Нормализация ({method}): {normalized.shape}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        return False
    
    # Тест 5: Усиление резкости
    print("\n6. Тест усиления резкости...")
    try:
        image_array = preprocessing_service._bytes_to_image(test_image)
        
        for strength in [0.5, 1.0, 1.5]:
            sharpened = preprocessing_service.sharpen_image(image_array, strength=strength)
            print(f"   ✓ Усиление резкости ({strength}): {sharpened.shape}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        return False
    
    # Тест 6: Бинаризация
    print("\n7. Тест бинаризации...")
    try:
        image_array = preprocessing_service._bytes_to_image(test_image)
        
        for method in ["otsu", "adaptive", "local"]:
            binary = preprocessing_service.binarize_image(image_array, method=method)
            print(f"   ✓ Бинаризация ({method}): {binary.shape}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        return False
    
    # Тест 7: Удаление рамок
    print("\n8. Тест удаления рамок...")
    try:
        image_array = preprocessing_service._bytes_to_image(test_image)
        cropped = preprocessing_service.remove_borders(image_array)
        print(f"   ✓ Удаление рамок: {cropped.shape}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        return False
    
    # Тест 8: Изменение размера
    print("\n9. Тест изменения размера...")
    try:
        image_array = preprocessing_service._bytes_to_image(test_image)
        resized = preprocessing_service.resize_for_ocr(image_array, max_dimension=2500)
        print(f"   ✓ Изменение размера: {resized.shape}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_preprocessing_functions()
    sys.exit(0 if success else 1)
