import sys
import os
import logging
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Добавляем путь к backend модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.providers.paddleocr_provider import paddleocr_provider
from backend.core.logger import setup_logger

# Настройка логирования
logger = setup_logger("test_paddleocr", level=logging.INFO)


def create_test_image_with_russian_text() -> bytes:
    """
    Создание тестового изображения с русским текстом
    
    Returns:
        bytes: Изображение в формате JPEG
    """
    # Создаем изображение
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Текст паспорта (упрощенный)
    passport_text = [
        "ПАСПОРТ",
        "Фамилия: ИВАНОВ",
        "Имя: ИВАН",
        "Отчество: ИВАНОВИЧ",
        "Дата рождения: 01.01.1990",
        "Серия: 4512",
        "Номер: 123456",
        "Дата выдачи: 15.05.2010",
    ]
    
    # Рисуем текст
    try:
        # Пытаемся использовать системный шрифт
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        # Если не нашли, используем дефолтный
        font = ImageFont.load_default()
        logger.warning("Системный шрифт не найден, используется дефолтный")
    
    y_position = 50
    for line in passport_text:
        draw.text((50, y_position), line, fill='black', font=font)
        y_position += 50
    
    # Конвертируем в bytes
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()


def test_paddleocr_initialization():
    """Тест 1: Инициализация PaddleOCR"""
    logger.info("=" * 60)
    logger.info("ТЕСТ 1: Инициализация PaddleOCR")
    logger.info("=" * 60)
    
    try:
        # Проверяем доступность до инициализации
        assert not paddleocr_provider.is_available(), "Провайдер не должен быть доступен до инициализации"
        logger.info("Статус до инициализации: недоступен")
        
        # Инициализируем
        logger.info("Запуск инициализации PaddleOCR...")
        paddleocr_provider.initialize()
        
        # Проверяем доступность после инициализации
        assert paddleocr_provider.is_available(), "Провайдер должен быть доступен после инициализации"
        logger.info("Статус после инициализации: доступен")
        
        logger.info("ТЕСТ 1: ПРОЙДЕН")
        return True
        
    except Exception as e:
        logger.error(f"ТЕСТ 1: ПРОВАЛЕН - {e}")
        return False


def test_paddleocr_recognize():
    """Тест 2: Распознавание текста"""
    logger.info("=" * 60)
    logger.info("ТЕСТ 2: Распознавание текста с PaddleOCR")
    logger.info("=" * 60)
    
    try:
        # Создаем тестовое изображение
        logger.info("Создание тестового изображения...")
        image_data = create_test_image_with_russian_text()
        logger.info(f"Тестовое изображение создано, размер: {len(image_data)} байт")
        
        # Распознаем текст
        logger.info("Запуск распознавания...")
        result = paddleocr_provider.recognize(image_data)
        
        # Проверяем результат
        assert result is not None, "Результат не должен быть None"
        assert hasattr(result, 'full_text'), "Результат должен содержать full_text"
        assert hasattr(result, 'blocks'), "Результат должен содержать blocks"
        assert hasattr(result, 'metadata'), "Результат должен содержать metadata"
        
        # Выводим статистику
        logger.info(f"Распознанный текст ({len(result.full_text)} символов):")
        logger.info("-" * 60)
        logger.info(result.full_text)
        logger.info("-" * 60)
        logger.info(f"Количество блоков: {len(result.blocks)}")
        logger.info(f"Средняя уверенность: {result.average_confidence:.2%}")
        logger.info(f"Провайдер: {result.metadata.provider}")
        logger.info(f"Время обработки: {result.metadata.processing_time_ms:.2f}ms")
        logger.info(f"Размер изображения: {result.metadata.image_width}x{result.metadata.image_height}")
        logger.info(f"Версия модели: {result.metadata.model_version}")
        
        # Проверяем блоки
        if result.blocks:
            logger.info(f"\nДетали первого блока:")
            first_block = result.blocks[0]
            logger.info(f"  Текст: {first_block.text}")
            logger.info(f"  Уверенность: {first_block.confidence:.2%}")
            if first_block.bounding_box:
                bbox = first_block.bounding_box
                logger.info(f"  Координаты: x={bbox.x}, y={bbox.y}, w={bbox.width}, h={bbox.height}")
        
        logger.info("ТЕСТ 2: ПРОЙДЕН")
        return True
        
    except Exception as e:
        logger.error(f"ТЕСТ 2: ПРОВАЛЕН - {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_paddleocr_with_numpy():
    """Тест 3: Распознавание с numpy array"""
    logger.info("=" * 60)
    logger.info("ТЕСТ 3: Распознавание с numpy array")
    logger.info("=" * 60)
    
    try:
        # Создаем изображение как numpy array
        logger.info("Создание numpy array изображения...")
        image_bytes = create_test_image_with_russian_text()
        pil_image = Image.open(BytesIO(image_bytes))
        numpy_array = np.array(pil_image)
        # Конвертируем RGB -> BGR для OpenCV/PaddleOCR
        numpy_array = numpy_array[:, :, ::-1]
        
        logger.info(f"Numpy array создан, shape: {numpy_array.shape}")
        
        # Распознаем
        logger.info("Запуск распознавания...")
        result = paddleocr_provider.recognize(numpy_array)
        
        # Проверяем
        assert result is not None, "Результат не должен быть None"
        assert len(result.full_text) > 0, "Текст должен быть распознан"
        
        logger.info(f"Распознано символов: {len(result.full_text)}")
        logger.info(f"Средняя уверенность: {result.average_confidence:.2%}")
        
        logger.info("ТЕСТ 3: ПРОЙДЕН")
        return True
        
    except Exception as e:
        logger.error(f"ТЕСТ 3: ПРОВАЛЕН - {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_paddleocr_empty_image():
    """Тест 4: Обработка пустого изображения"""
    logger.info("=" * 60)
    logger.info("ТЕСТ 4: Обработка пустого изображения")
    logger.info("=" * 60)
    
    try:
        # Создаем пустое белое изображение
        logger.info("Создание пустого изображения...")
        image = Image.new('RGB', (800, 600), color='white')
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        image_data = buffer.getvalue()
        
        # Распознаем
        logger.info("Запуск распознавания...")
        result = paddleocr_provider.recognize(image_data)
        
        # Проверяем
        assert result is not None, "Результат не должен быть None"
        logger.info(f"Распознанный текст: '{result.full_text}'")
        logger.info(f"Количество блоков: {len(result.blocks)}")
        logger.info(f"Средняя уверенность: {result.average_confidence:.2%}")
        
        logger.info("ТЕСТ 4: ПРОЙДЕН")
        return True
        
    except Exception as e:
        logger.error(f"ТЕСТ 4: ПРОВАЛЕН - {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Запуск всех тестов"""
    logger.info("\n" + "=" * 60)
    logger.info("ЗАПУСК ТЕСТОВ PADDLEOCR ПРОВАЙДЕРА")
    logger.info("=" * 60 + "\n")
    
    tests = [
        test_paddleocr_initialization,
        test_paddleocr_recognize,
        test_paddleocr_with_numpy,
        test_paddleocr_empty_image,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"Критическая ошибка в тесте {test_func.__name__}: {e}")
            results.append(False)
        logger.info("")  # Пустая строка между тестами
    
    # Итоговый отчет
    logger.info("=" * 60)
    logger.info("ИТОГОВЫЙ ОТЧЕТ")
    logger.info("=" * 60)
    passed = sum(results)
    total = len(results)
    logger.info(f"Пройдено тестов: {passed}/{total}")
    
    if passed == total:
        logger.info("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО")
        return 0
    else:
        logger.error(f"ПРОВАЛЕНО ТЕСТОВ: {total - passed}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
