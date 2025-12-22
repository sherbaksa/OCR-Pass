import time
import logging
from typing import Union, List
import numpy as np
from PIL import Image
import io

import easyocr
from backend.providers.base_ocr import BaseOCRProvider
from backend.schemas.ocr import (
    OCRResult,
    OCRBlock,
    OCRLine,
    OCRWord,
    BoundingBox,
    OCRMetadata,
)
from backend.core.settings import settings

logger = logging.getLogger(__name__)


class EasyOCRProvider(BaseOCRProvider):
    """
    OCR-провайдер на основе EasyOCR.
    Поддерживает распознавание русского и английского текста.
    """

    def __init__(self):
        """Инициализация EasyOCR провайдера"""
        super().__init__(provider_name="easyocr")
        self.reader = None
        self.model_version = "EasyOCR-v1.7"

    def initialize(self) -> None:
        """
        Инициализация EasyOCR движка.
        Загружает модели для русского и английского языков.
        """
        try:
            logger.info("Инициализация EasyOCR провайдера...")

            # Инициализация EasyOCR для русского и английского языков
            # gpu=False для CPU, gpu=True для GPU
            self.reader = easyocr.Reader(
                ['ru', 'en'],
                gpu=settings.paddleocr_use_gpu,  # Используем ту же настройку GPU
                verbose=False
            )

            self._initialized = True
            logger.info(f"EasyOCR провайдер инициализирован успешно (GPU: {settings.paddleocr_use_gpu})")

        except Exception as e:
            logger.error(f"Ошибка инициализации EasyOCR: {e}")
            raise RuntimeError(f"Не удалось инициализировать EasyOCR: {e}")

    def is_available(self) -> bool:
        """
        Проверка доступности EasyOCR

        Returns:
            bool: True если провайдер готов к работе
        """
        return self._initialized and self.reader is not None

    def recognize(
        self,
        image_data: Union[bytes, np.ndarray],
        language: str = "ru"
    ) -> OCRResult:
        """
        Распознавание текста на изображении с помощью EasyOCR.

        Args:
            image_data: Изображение в виде bytes или numpy array
            language: Язык распознавания (игнорируется, используются ru+en)

        Returns:
            OCRResult: Унифицированный результат распознавания

        Raises:
            ValueError: Если провайдер не инициализирован
            RuntimeError: Если произошла ошибка при распознавании
        """
        self._ensure_initialized()

        start_time = time.time()

        try:
            # Конвертация данных в numpy array
            image_array = self._prepare_image(image_data)

            # Получаем размеры изображения
            img_height, img_width = image_array.shape[:2]

            logger.info(f"Запуск EasyOCR на изображении {img_width}x{img_height}")

            # Выполнение OCR
            # EasyOCR возвращает список: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence), ...]
            ocr_result = self.reader.readtext(image_array)

            logger.info(f"=== EasyOCR RAW RESULT START === Length: {len(ocr_result)}")
            for idx, item in enumerate(ocr_result):
                bbox, text, confidence = item
                logger.info(f"Line {idx}: '{text}' | Conf: {confidence:.3f}")
            logger.info("=== EasyOCR RAW RESULT END ===")

            # Обработка результатов
            if not ocr_result:
                logger.warning("EasyOCR не распознал текст на изображении")
                return self._create_empty_result(img_width, img_height, start_time)

            # Парсинг результатов в унифицированный формат
            unified_result = self._parse_easyocr_result(
                ocr_result,
                img_width,
                img_height,
                start_time
            )

            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"EasyOCR завершен за {processing_time:.2f}ms. "
                f"Распознано блоков: {len(unified_result.blocks)}, "
                f"средняя уверенность: {unified_result.average_confidence:.2f}"
            )

            return unified_result

        except Exception as e:
            logger.error(f"Ошибка при распознавании с EasyOCR: {e}")
            raise RuntimeError(f"Ошибка распознавания EasyOCR: {e}")

    def _prepare_image(self, image_data: Union[bytes, np.ndarray]) -> np.ndarray:
        """
        Подготовка изображения для EasyOCR

        Args:
            image_data: Изображение в bytes или numpy array

        Returns:
            np.ndarray: Изображение в формате numpy array (RGB)
        """
        if isinstance(image_data, bytes):
            # Конвертация bytes -> PIL -> numpy
            pil_image = Image.open(io.BytesIO(image_data))
            # Конвертация в RGB если нужно
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # EasyOCR работает с RGB
            image_array = np.array(pil_image)
            return image_array

        elif isinstance(image_data, np.ndarray):
            # Если BGR (из OpenCV), конвертируем в RGB
            if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                # Предполагаем BGR, конвертируем в RGB
                return image_data[:, :, ::-1]
            return image_data

        else:
            raise ValueError(f"Неподдерживаемый тип данных изображения: {type(image_data)}")

    def _parse_easyocr_result(
        self,
        easyocr_result: List,
        img_width: int,
        img_height: int,
        start_time: float
    ) -> OCRResult:
        """
        Парсинг результата EasyOCR в унифицированный формат

        Args:
            easyocr_result: Результат от EasyOCR
            img_width: Ширина изображения
            img_height: Высота изображения
            start_time: Время начала обработки

        Returns:
            OCRResult: Унифицированный результат
        """
        blocks = []
        full_text_parts = []
        total_confidence = 0.0
        confidence_count = 0

        # EasyOCR возвращает: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence), ...]
        for item in easyocr_result:
            if not item or len(item) < 3:
                continue

            bbox_points = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = item[1]         # text
            confidence = float(item[2])  # confidence

            # Пропускаем пустые результаты
            if not text or not text.strip():
                continue

            # Создаем bounding box из координат
            bounding_box = self._create_bounding_box_from_points(bbox_points)

            # Создаем слово
            word = OCRWord(
                text=text,
                confidence=confidence,
                bounding_box=bounding_box
            )

            # Создаем линию с одним словом
            line = OCRLine(
                text=text,
                confidence=confidence,
                words=[word],
                bounding_box=bounding_box
            )

            # Создаем блок с одной линией
            block = OCRBlock(
                text=text,
                confidence=confidence,
                lines=[line],
                bounding_box=bounding_box,
                block_type="text"
            )

            blocks.append(block)
            full_text_parts.append(text)
            total_confidence += confidence
            confidence_count += 1

        # Вычисляем среднюю уверенность
        average_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0

        # Собираем полный текст
        full_text = "\n".join(full_text_parts)

        # Метаданные
        processing_time_ms = (time.time() - start_time) * 1000
        metadata = OCRMetadata(
            provider=self.provider_name,
            processing_time_ms=processing_time_ms,
            image_width=img_width,
            image_height=img_height,
            preprocessed=False,
            preprocessing_steps=[],
            language="ru+en",
            model_version=self.model_version
        )

        return OCRResult(
            full_text=full_text,
            blocks=blocks,
            average_confidence=average_confidence,
            passport_fields=[],
            metadata=metadata,
            raw_response={"easyocr_result": easyocr_result}
        )

    def _create_bounding_box_from_points(
        self,
        points: List[List[float]]
    ) -> BoundingBox:
        """
        Создание BoundingBox из точек координат

        Args:
            points: Список точек [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            BoundingBox: Прямоугольная область
        """
        # Находим минимальные и максимальные координаты
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        x_max = int(max(x_coords))
        y_max = int(max(y_coords))

        return BoundingBox(
            x=x_min,
            y=y_min,
            width=x_max - x_min,
            height=y_max - y_min
        )

    def _create_empty_result(
        self,
        img_width: int,
        img_height: int,
        start_time: float
    ) -> OCRResult:
        """
        Создание пустого результата если текст не распознан

        Args:
            img_width: Ширина изображения
            img_height: Высота изображения
            start_time: Время начала обработки

        Returns:
            OCRResult: Пустой результат
        """
        processing_time_ms = (time.time() - start_time) * 1000

        metadata = OCRMetadata(
            provider=self.provider_name,
            processing_time_ms=processing_time_ms,
            image_width=img_width,
            image_height=img_height,
            preprocessed=False,
            preprocessing_steps=[],
            language="ru+en",
            model_version=self.model_version
        )

        return OCRResult(
            full_text="",
            blocks=[],
            average_confidence=0.0,
            passport_fields=[],
            metadata=metadata,
            raw_response=None
        )


# Singleton instance
easyocr_provider = EasyOCRProvider()
