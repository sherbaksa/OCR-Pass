import time
import logging
from typing import Union, List
import numpy as np
from PIL import Image
import io
import tempfile
import os

from paddleocr import PaddleOCR
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


class PaddleOCRProvider(BaseOCRProvider):
    """
    OCR-провайдер на основе PaddleOCR 3.x (новый API).
    Использует метод predict() со стандартной моделью eslav_PP-OCRv5_mobile_rec.
    """

    def __init__(self):
        """Инициализация PaddleOCR провайдера"""
        super().__init__(provider_name="paddleocr")
        self.ocr_engine = None
        self.model_version = "PaddleOCR-3.x-Standard"
        self._last_raw_result = None

    def initialize(self) -> None:
        """
        Инициализация PaddleOCR движка со стандартной моделью.
        Согласно рабочей конфигурации из Colab (второй вариант).
        """
        try:
            logger.info("Инициализация PaddleOCR провайдера...")

            # ТОЛЬКО стандартная модель (как работает в Colab)
            # Это загружает eslav_PP-OCRv5_mobile_rec - лучшую модель для русского языка
            self.ocr_engine = PaddleOCR(lang='ru')

            self._initialized = True
            logger.info("✓ PaddleOCR провайдер инициализирован успешно (стандартная модель)")

        except Exception as e:
            logger.error(f"❌ Ошибка инициализации PaddleOCR: {e}")
            raise RuntimeError(f"Не удалось инициализировать PaddleOCR: {e}")

    def is_available(self) -> bool:
        """Проверка доступности PaddleOCR"""
        return self._initialized and self.ocr_engine is not None

    def recognize(
        self,
        image_data: Union[bytes, np.ndarray],
        language: str = "ru"
    ) -> OCRResult:
        """
        Распознавание текста с помощью PaddleOCR 3.x (метод predict).

        Args:
            image_data: Изображение в виде bytes или numpy array
            language: Язык (игнорируется, используется ru)

        Returns:
            OCRResult: Унифицированный результат
        """
        self._ensure_initialized()

        start_time = time.time()

        try:
            # Сохраняем изображение во временный файл
            temp_image_path = self._save_temp_image(image_data)

            logger.info(f"Запуск PaddleOCR predict() на {temp_image_path}")

            # Используем predict() API
            result = self.ocr_engine.predict(temp_image_path)

            # Сохраняем сырой результат для отладки
            self._last_raw_result = result

            # Удаляем временный файл
            os.remove(temp_image_path)

            # Парсим результат
            unified_result = self._parse_predict_result(result, start_time)

            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"PaddleOCR завершен за {processing_time:.2f}ms. "
                f"Распознано текстов: {len(unified_result.blocks)}, "
                f"средняя уверенность: {unified_result.average_confidence:.2%}"
            )

            return unified_result

        except Exception as e:
            logger.error(f"Ошибка при распознавании с PaddleOCR: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Ошибка распознавания PaddleOCR: {e}")

    def _save_temp_image(self, image_data: Union[bytes, np.ndarray]) -> str:
        """Сохранить изображение во временный файл"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_path = temp_file.name

        if isinstance(image_data, bytes):
            temp_file.write(image_data)
        elif isinstance(image_data, np.ndarray):
            # BGR -> RGB если нужно
            if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                image_data = image_data[:, :, ::-1]
            pil_image = Image.fromarray(image_data)
            pil_image.save(temp_path, 'JPEG')
        else:
            raise ValueError(f"Неподдерживаемый тип: {type(image_data)}")

        temp_file.close()
        return temp_path

    def _parse_predict_result(
        self,
        predict_result: list,
        start_time: float
    ) -> OCRResult:
        """
        Парсинг результата predict() в унифицированный формат.
        
        predict_result[0] - это объект OCRResult с данными напрямую (БЕЗ 'res'!)
        """
        blocks = []
        full_text_parts = []
        total_confidence = 0.0
        confidence_count = 0

        try:
            # predict() возвращает СПИСОК, берем первый элемент
            if not predict_result or len(predict_result) == 0:
                logger.warning("predict() вернул пустой список")
                return self._create_empty_result(start_time)

            # Первый элемент списка - результат для нашего изображения
            result_item = predict_result[0]

            # Данные напрямую в объекте, БЕЗ 'res'!
            if isinstance(result_item, dict):
                rec_texts = result_item.get('rec_texts', [])
                rec_scores = result_item.get('rec_scores', [])
                rec_boxes = result_item.get('rec_boxes', [])
            else:
                rec_texts = getattr(result_item, 'rec_texts', [])
                rec_scores = getattr(result_item, 'rec_scores', [])
                rec_boxes = getattr(result_item, 'rec_boxes', [])

            logger.info(f"Распознано текстов: {len(rec_texts)}")

            # Обрабатываем каждый распознанный текст
            for idx, text in enumerate(rec_texts):
                if not text or not text.strip():
                    continue

                # Получаем уверенность
                confidence = float(rec_scores[idx]) if idx < len(rec_scores) else 0.0

                # Получаем координаты если есть
                bounding_box = None
                if idx < len(rec_boxes):
                    box = rec_boxes[idx]
                    if len(box) >= 4:
                        bounding_box = BoundingBox(
                            x=int(box[0]),
                            y=int(box[1]),
                            width=int(box[2] - box[0]),
                            height=int(box[3] - box[1])
                        )

                # Создаем структуру
                word = OCRWord(
                    text=text,
                    confidence=confidence,
                    bounding_box=bounding_box
                )

                line = OCRLine(
                    text=text,
                    confidence=confidence,
                    words=[word],
                    bounding_box=bounding_box
                )

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

                # Логируем первые 10 распознанных текстов
                if idx < 10:
                    logger.info(f"  [{idx}] '{text}' (conf: {confidence:.3f})")

        except Exception as e:
            logger.error(f"Ошибка парсинга predict результата: {e}")
            import traceback
            traceback.print_exc()

        # Вычисляем среднюю уверенность
        average_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0

        # Собираем полный текст
        full_text = "\n".join(full_text_parts)

        # Метаданные
        processing_time_ms = (time.time() - start_time) * 1000
        metadata = OCRMetadata(
            provider=self.provider_name,
            processing_time_ms=processing_time_ms,
            image_width=0,
            image_height=0,
            preprocessed=False,
            preprocessing_steps=[],
            language="ru",
            model_version=self.model_version
        )

        return OCRResult(
            full_text=full_text,
            blocks=blocks,
            average_confidence=average_confidence,
            passport_fields=[],
            metadata=metadata,
            raw_response=self._convert_result_to_dict(predict_result)
        )


#   _convert_result_to_dict

    def _convert_result_to_dict(self, predict_result: list) -> dict:
        """Конвертировать только важные данные для JSON (БЕЗ огромных массивов координат)"""
        if not predict_result or len(predict_result) == 0:
            return {}
        
        result_item = predict_result[0]
        
        # Извлекаем только тексты и scores - БЕЗ координат!
        if isinstance(result_item, dict):
            rec_texts = result_item.get('rec_texts', [])
            rec_scores = result_item.get('rec_scores', [])
        else:
            rec_texts = getattr(result_item, 'rec_texts', [])
            rec_scores = getattr(result_item, 'rec_scores', [])
        
        # Конвертируем numpy массив scores в обычный список
        if hasattr(rec_scores, 'tolist'):
                rec_scores = rec_scores.tolist()
        
        return {
            'rec_texts': rec_texts,
            'rec_scores': rec_scores,
                'total_texts': len(rec_texts)
        }


#       _convert_result_to_dict
    def _create_empty_result(self, start_time: float) -> OCRResult:
        """Создание пустого результата"""
        processing_time_ms = (time.time() - start_time) * 1000

        metadata = OCRMetadata(
            provider=self.provider_name,
            processing_time_ms=processing_time_ms,
            image_width=0,
            image_height=0,
            preprocessed=False,
            preprocessing_steps=[],
            language="ru",
            model_version=self.model_version
        )

        return OCRResult(
            full_text="",
            blocks=[],
            average_confidence=0.0,
            passport_fields=[],
            metadata=metadata,
            raw_response={}
        )


# Singleton instance
paddleocr_provider = PaddleOCRProvider()
