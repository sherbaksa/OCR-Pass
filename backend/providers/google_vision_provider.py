import time
import logging
from typing import Union, List
import numpy as np
from PIL import Image
import io
import os

from google.cloud import vision
from google.oauth2 import service_account

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


class GoogleVisionProvider(BaseOCRProvider):
    """
    OCR-провайдер на основе Google Cloud Vision API.
    Поддерживает распознавание текста с использованием TEXT_DETECTION.
    """
    
    def __init__(self):
        """Инициализация Google Vision провайдера"""
        super().__init__(provider_name="google_vision")
        self.client = None
        self.model_version = "Google Cloud Vision API v1"
    
    def initialize(self) -> None:
        """
        Инициализация Google Vision API клиента.
        Использует service account credentials из файла или mock-режим.
        """
        try:
            logger.info("Инициализация Google Vision провайдера...")
            
            # Проверка включения провайдера
            if not settings.google_vision_enabled:
                raise ValueError("Google Vision API отключен в настройках")
            
            # Mock-режим (для тестирования без реального API)
            if settings.google_vision_mock_mode:
                logger.warning("Google Vision работает в MOCK-РЕЖИМЕ (без реальных API вызовов)")
                self.client = "MOCK_CLIENT"  # Заглушка
                self._initialized = True
                logger.info("Google Vision провайдер инициализирован в mock-режиме")
                return
            
            # Проверка пути к credentials
            if not settings.google_application_credentials:
                raise ValueError("Не указан путь к Google credentials (GOOGLE_APPLICATION_CREDENTIALS)")
            
            credentials_path = settings.google_application_credentials
            
            # Проверка существования файла
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"Файл credentials не найден: {credentials_path}")
            
            logger.info(f"Загрузка credentials из: {credentials_path}")
            
            # Создание credentials из файла
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            
            # Инициализация клиента
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
            
            self._initialized = True
            logger.info("Google Vision провайдер инициализирован успешно")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации Google Vision: {e}")
            raise RuntimeError(f"Не удалось инициализировать Google Vision: {e}")
    
    def is_available(self) -> bool:
        """
        Проверка доступности Google Vision
        
        Returns:
            bool: True если провайдер готов к работе
        """
        return (
            self._initialized 
            and self.client is not None 
            and settings.google_vision_enabled
        )
    
    def recognize(
        self,
        image_data: Union[bytes, np.ndarray],
        language: str = "ru"
    ) -> OCRResult:
        """
        Распознавание текста на изображении с помощью Google Vision API.
        
        Args:
            image_data: Изображение в виде bytes или numpy array
            language: Язык распознавания (используется для hints)
        
        Returns:
            OCRResult: Унифицированный результат распознавания
        
        Raises:
            ValueError: Если провайдер не инициализирован
            RuntimeError: Если произошла ошибка при распознавании
        """
        self._ensure_initialized()
        
        start_time = time.time()
        
        # Mock-режим: возвращаем синтетический результат
        if settings.google_vision_mock_mode:
            return self._create_mock_result(image_data, start_time)
        
        try:
            # Конвертация данных в bytes
            image_bytes = self._prepare_image(image_data)
            
            # Создание объекта Image для Google Vision
            vision_image = vision.Image(content=image_bytes)
            
            # Получаем размеры изображения
            pil_image = Image.open(io.BytesIO(image_bytes))
            img_width, img_height = pil_image.size
            
            logger.info(f"Запуск Google Vision на изображении {img_width}x{img_height}")
            
            # Настройка языковых подсказок
            image_context = vision.ImageContext(language_hints=[language])
            
            # Выполнение TEXT_DETECTION
            response = self.client.text_detection(
                image=vision_image,
                image_context=image_context
            )
            
            # Проверка ошибок
            if response.error.message:
                raise RuntimeError(f"Google Vision API error: {response.error.message}")
            
            # Обработка результатов
            if not response.text_annotations:
                logger.warning("Google Vision не распознал текст на изображении")
                return self._create_empty_result(img_width, img_height, start_time)
            
            # Парсинг результатов в унифицированный формат
            unified_result = self._parse_vision_result(
                response.text_annotations,
                img_width,
                img_height,
                start_time
            )
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Google Vision завершен за {processing_time:.2f}ms. "
                f"Распознано блоков: {len(unified_result.blocks)}, "
                f"средняя уверенность: {unified_result.average_confidence:.2f}"
            )
            
            return unified_result
            
        except Exception as e:
            logger.error(f"Ошибка при распознавании с Google Vision: {e}")
            raise RuntimeError(f"Ошибка распознавания Google Vision: {e}")
    
    def _prepare_image(self, image_data: Union[bytes, np.ndarray]) -> bytes:
        """
        Подготовка изображения для Google Vision
        
        Args:
            image_data: Изображение в bytes или numpy array
        
        Returns:
            bytes: Изображение в формате bytes
        """
        if isinstance(image_data, bytes):
            return image_data
        
        elif isinstance(image_data, np.ndarray):
            # Конвертация numpy array -> PIL -> bytes
            # Проверяем формат (BGR -> RGB если нужно)
            if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                # Предполагаем BGR от OpenCV, конвертируем в RGB
                image_data = image_data[:, :, ::-1]
            
            pil_image = Image.fromarray(image_data)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            return buffer.getvalue()
        
        else:
            raise ValueError(f"Неподдерживаемый тип данных изображения: {type(image_data)}")
    
    def _parse_vision_result(
        self,
        text_annotations: List,
        img_width: int,
        img_height: int,
        start_time: float
    ) -> OCRResult:
        """
        Парсинг результата Google Vision в унифицированный формат
        
        Args:
            text_annotations: Результат от Google Vision API
            img_width: Ширина изображения
            img_height: Высота изображения
            start_time: Время начала обработки
        
        Returns:
            OCRResult: Унифицированный результат
        """
        # Первый элемент - полный текст документа
        full_text = text_annotations[0].description if text_annotations else ""
        
        blocks = []
        total_confidence = 0.0
        confidence_count = 0
        
        # Остальные элементы - отдельные слова/фрагменты
        # Пропускаем первый элемент (полный текст)
        for annotation in text_annotations[1:]:
            text = annotation.description
            
            # Google Vision не возвращает confidence для TEXT_DETECTION
            # Используем значение по умолчанию или извлекаем из других полей
            # Для TEXT_DETECTION confidence обычно не доступен, ставим 0.9
            confidence = 0.9
            
            # Получаем bounding box
            vertices = annotation.bounding_poly.vertices
            bounding_box = self._create_bounding_box_from_vertices(vertices)
            
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
            total_confidence += confidence
            confidence_count += 1
        
        # Вычисляем среднюю уверенность
        average_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.9
        
        # Метаданные
        processing_time_ms = (time.time() - start_time) * 1000
        metadata = OCRMetadata(
            provider=self.provider_name,
            processing_time_ms=processing_time_ms,
            image_width=img_width,
            image_height=img_height,
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
            raw_response={"text_annotations": [self._annotation_to_dict(a) for a in text_annotations]}
        )
    
    def _create_bounding_box_from_vertices(
        self,
        vertices: List
    ) -> BoundingBox:
        """
        Создание BoundingBox из вершин Google Vision
        
        Args:
            vertices: Список вершин от Google Vision
        
        Returns:
            BoundingBox: Прямоугольная область
        """
        # Извлекаем координаты
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]
        
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        return BoundingBox(
            x=x_min,
            y=y_min,
            width=x_max - x_min,
            height=y_max - y_min
        )
    
    def _annotation_to_dict(self, annotation) -> dict:
        """
        Конвертация annotation в словарь для хранения в raw_response
        
        Args:
            annotation: TextAnnotation от Google Vision
        
        Returns:
            dict: Словарь с данными annotation
        """
        return {
            "description": annotation.description,
            "bounding_poly": {
                "vertices": [
                    {"x": v.x, "y": v.y} 
                    for v in annotation.bounding_poly.vertices
                ]
            }
        }
    
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
            language="ru",
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
    
    def _create_mock_result(
        self,
        image_data: Union[bytes, np.ndarray],
        start_time: float
    ) -> OCRResult:
        """
        Создание mock-результата для тестирования без реального API
        
        Args:
            image_data: Изображение
            start_time: Время начала обработки
        
        Returns:
            OCRResult: Синтетический результат
        """
        # Конвертация в bytes для получения размеров
        image_bytes = self._prepare_image(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = pil_image.size
        
        logger.info(f"MOCK MODE: Создание синтетического результата для {img_width}x{img_height}")
        
        # Синтетический текст паспорта
        mock_text = """ПАСПОРТ
Фамилия: ИВАНОВ
Имя: ИВАН
Отчество: ИВАНОВИЧ
Дата рождения: 01.01.1990
Серия: 4512
Номер: 123456
Дата выдачи: 15.05.2010"""
        
        # Создаём блоки для каждой строки
        blocks = []
        lines = mock_text.split('\n')
        y_offset = 50
        
        for line_text in lines:
            # Создаём bounding box
            bbox = BoundingBox(
                x=50,
                y=y_offset,
                width=len(line_text) * 12,
                height=20
            )
            
            # Создаём слово
            word = OCRWord(
                text=line_text,
                confidence=0.95,
                bounding_box=bbox
            )
            
            # Создаём линию
            line = OCRLine(
                text=line_text,
                confidence=0.95,
                words=[word],
                bounding_box=bbox
            )
            
            # Создаём блок
            block = OCRBlock(
                text=line_text,
                confidence=0.95,
                lines=[line],
                bounding_box=bbox,
                block_type="text"
            )
            
            blocks.append(block)
            y_offset += 50
        
        # Метаданные
        processing_time_ms = (time.time() - start_time) * 1000
        metadata = OCRMetadata(
            provider=f"{self.provider_name}_mock",
            processing_time_ms=processing_time_ms,
            image_width=img_width,
            image_height=img_height,
            preprocessed=False,
            preprocessing_steps=[],
            language="ru",
            model_version=f"{self.model_version} (MOCK)"
        )
        
        return OCRResult(
            full_text=mock_text,
            blocks=blocks,
            average_confidence=0.95,
            passport_fields=[],
            metadata=metadata,
            raw_response={"mock": True, "message": "This is a synthetic result for testing"}
        )


# Singleton instance
google_vision_provider = GoogleVisionProvider()
