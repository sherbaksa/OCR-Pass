import time
import logging
from typing import Union, List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import io

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
    OCR-провайдер на основе PaddleOCR.
    Поддерживает распознавание русского текста с помощью моделей PP-OCRv4.
    """

    def __init__(self):
        """Инициализация PaddleOCR провайдера"""
        super().__init__(provider_name="paddleocr")
        self.ocr_engine = None
        self.model_version = "PP-OCRv4"
        self._last_raw_result = None  # Хранение последнего сырого результата

    def initialize(self) -> None:
        """
        Инициализация PaddleOCR движка.
        Загружает модели для детекции, распознавания и классификации.
        """
        try:
            logger.info("Инициализация PaddleOCR провайдера...")

            # Параметры инициализации
            ocr_params = {
                'use_angle_cls': True,  # Использовать классификацию угла поворота
                'lang': settings.paddleocr_lang,
                'use_gpu': settings.paddleocr_use_gpu,
                'show_log': False,  # Отключаем логи PaddleOCR
            }

            # Добавляем пути к кастомным моделям, если указаны
            if settings.paddleocr_det_model_dir:
                ocr_params['det_model_dir'] = settings.paddleocr_det_model_dir

            if settings.paddleocr_rec_model_dir:
                ocr_params['rec_model_dir'] = settings.paddleocr_rec_model_dir

            if settings.paddleocr_cls_model_dir:
                ocr_params['cls_model_dir'] = settings.paddleocr_cls_model_dir

            # Инициализация движка
            self.ocr_engine = PaddleOCR(**ocr_params)

            self._initialized = True
            logger.info(f"PaddleOCR провайдер инициализирован успешно (язык: {settings.paddleocr_lang}, GPU: {settings.paddleocr_use_gpu})")

        except Exception as e:
            logger.error(f"Ошибка инициализации PaddleOCR: {e}")
            raise RuntimeError(f"Не удалось инициализировать PaddleOCR: {e}")

    def is_available(self) -> bool:
        """
        Проверка доступности PaddleOCR

        Returns:
            bool: True если провайдер готов к работе
        """
        return self._initialized and self.ocr_engine is not None

    def recognize(
        self,
        image_data: Union[bytes, np.ndarray],
        language: str = "ru"
    ) -> OCRResult:
        """
        Распознавание текста на изображении с помощью PaddleOCR.

        Args:
            image_data: Изображение в виде bytes или numpy array
            language: Язык распознавания (игнорируется, используется язык из настроек)

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

            logger.info(f"Запуск PaddleOCR на изображении {img_width}x{img_height}")

            # Выполнение OCR
            ocr_result = self.ocr_engine.ocr(image_array, cls=True)
            
            # СОХРАНЯЕМ СЫРОЙ РЕЗУЛЬТАТ
            self._last_raw_result = ocr_result
            
            print(f"\n{'='*80}")
            print(f"PADDLEOCR DEBUG: ocr_result type = {type(ocr_result)}")
            print(f"PADDLEOCR DEBUG: ocr_result length = {len(ocr_result) if ocr_result else 0}")
            if ocr_result and len(ocr_result) > 0:
                print(f"PADDLEOCR DEBUG: ocr_result[0] length = {len(ocr_result[0])}")
                for idx, line in enumerate(ocr_result[0]):
                    print(f"  Line {idx}: {line[1][0]} (conf: {line[1][1]:.3f})")
            print(f"{'='*80}\n")

            # ОТЛАДКА: Логируем весь распознанный текст
            logger.info(f"=== PaddleOCR RAW RESULT START === Type: {type(ocr_result)}, Length: {len(ocr_result) if ocr_result else 0}")
            if ocr_result:
                logger.info(f"ocr_result[0] type: {type(ocr_result[0])}, Length: {len(ocr_result[0]) if ocr_result[0] else 0}")
                if ocr_result[0]:
                    for idx, line_data in enumerate(ocr_result[0]):
                        try:
                            text = line_data[1][0]
                            confidence = line_data[1][1]
                            logger.info(f"Line {idx}: '{text}' | Conf: {confidence:.3f}")
                        except Exception as e:
                            logger.error(f"Error parsing line {idx}: {e}, Data: {line_data}")
                else:
                    logger.warning("ocr_result[0] is empty")
            else:
                logger.warning("ocr_result is None or empty")
            logger.info("=== PaddleOCR RAW RESULT END ===")

            # Обработка результатов
            if not ocr_result or not ocr_result[0]:
                logger.warning("PaddleOCR не распознал текст на изображении")
                return self._create_empty_result(img_width, img_height, start_time)

            # Парсинг результатов в унифицированный формат
            unified_result = self._parse_paddle_result(
                ocr_result[0],
                img_width,
                img_height,
                start_time
            )

            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"PaddleOCR завершен за {processing_time:.2f}ms. "
                f"Распознано блоков: {len(unified_result.blocks)}, "
                f"средняя уверенность: {unified_result.average_confidence:.2f}"
            )

            return unified_result

        except Exception as e:
            logger.error(f"Ошибка при распознавании с PaddleOCR: {e}")
            raise RuntimeError(f"Ошибка распознавания PaddleOCR: {e}")

    def _convert_result_to_dict(self, ocr_result: Optional[List]) -> Optional[Dict[str, Any]]:
        """
        Конвертация сырого результата PaddleOCR в словарь для API.
        
        Args:
            ocr_result: Сырой результат от PaddleOCR
            
        Returns:
            Dict с полями rec_texts, rec_scores, total_texts или None
        """
        if not ocr_result or not ocr_result[0]:
            return None
            
        try:
            rec_texts = []
            rec_scores = []
            
            for item in ocr_result[0]:
                if not item or len(item) < 2:
                    continue
                    
                text_info = item[1]
                if not text_info or len(text_info) < 2:
                    continue
                    
                text = text_info[0]
                confidence = float(text_info[1])
                
                rec_texts.append(text)
                rec_scores.append(confidence)
            
            return {
                "rec_texts": rec_texts,
                "rec_scores": rec_scores,
                "total_texts": len(rec_texts)
            }
            
        except Exception as e:
            logger.error(f"Ошибка конвертации результата в словарь: {e}")
            return None

    def _prepare_image(self, image_data: Union[bytes, np.ndarray]) -> np.ndarray:
        """
        Подготовка изображения для PaddleOCR

        Args:
            image_data: Изображение в bytes или numpy array

        Returns:
            np.ndarray: Изображение в формате numpy array (BGR)
        """
        if isinstance(image_data, bytes):
            # Конвертация bytes -> PIL -> numpy
            pil_image = Image.open(io.BytesIO(image_data))
            # Конвертация в RGB если нужно
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # PIL использует RGB, OpenCV использует BGR
            image_array = np.array(pil_image)
            # Конвертация RGB -> BGR для OpenCV/PaddleOCR
            image_array = image_array[:, :, ::-1]
            return image_array

        elif isinstance(image_data, np.ndarray):
            return image_data

        else:
            raise ValueError(f"Неподдерживаемый тип данных изображения: {type(image_data)}")

    def _parse_paddle_result(
        self,
        paddle_result: List,
        img_width: int,
        img_height: int,
        start_time: float
    ) -> OCRResult:
        """
        Парсинг результата PaddleOCR в унифицированный формат

        Args:
            paddle_result: Результат от PaddleOCR
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

        # PaddleOCR возвращает список элементов [bbox, (text, confidence)]
        for item in paddle_result:
            if not item or len(item) < 2:
                continue

            bbox_points = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text_info = item[1]    # (text, confidence)

            if not text_info or len(text_info) < 2:
                continue

            text = text_info[0]
            confidence = float(text_info[1])

            # Пропускаем пустые результаты
            if not text or not text.strip():
                continue

            # Создаем bounding box из координат
            bounding_box = self._create_bounding_box_from_points(bbox_points)

            # Создаем слово (в PaddleOCR каждый элемент - это строка/слово)
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
            language=settings.paddleocr_lang,
            model_version=self.model_version
        )

        return OCRResult(
            full_text=full_text,
            blocks=blocks,
            average_confidence=average_confidence,
            passport_fields=[],
            metadata=metadata,
            raw_response={"paddle_result": paddle_result}
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
            language=settings.paddleocr_lang,
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
paddleocr_provider = PaddleOCRProvider()
