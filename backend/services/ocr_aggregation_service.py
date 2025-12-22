"""
OCR Aggregation Service
Координирует работу всех OCR провайдеров и агрегирует результаты.
Использует Field Extraction и Field Scoring для получения лучших значений.
"""

import time
from typing import List, Union, Optional
import numpy as np

from backend.providers import (
    paddleocr_provider, 
    easyocr_provider, 
    google_vision_provider,
    PADDLEOCR_AVAILABLE,
    EASYOCR_AVAILABLE,
    GOOGLE_VISION_AVAILABLE
)
from backend.schemas.ocr import OCRResult
from backend.schemas.passport_fields import PassportFieldsResult, ProviderExtraction
from backend.services.field_extraction_service import field_extraction_service
from backend.services.field_scoring_service import field_scoring_service
from backend.core.logger import logger
from backend.core.settings import settings
from backend.services.preprocessing_service import preprocessing_service

class OCRAggregationService:
    """
    Сервис агрегации OCR результатов от нескольких провайдеров.
    Выполняет распознавание, извлечение полей и голосование.
    """

    def __init__(self):
        """Инициализация сервиса"""
        self.providers = []
        self._initialized = False
        logger.info("OCR Aggregation Service created")

    def initialize(self) -> None:
        """
        Инициализация всех доступных OCR провайдеров.
        """
        if self._initialized:
            logger.info("OCR Aggregation Service уже инициализирован")
            return

        logger.info("Инициализация OCR Aggregation Service...")

        # Инициализация PaddleOCR
        if settings.paddleocr_enabled:
            if PADDLEOCR_AVAILABLE:
                try:
                    paddleocr_provider.initialize()
                    self.providers.append(paddleocr_provider)
                    logger.info("PaddleOCR provider добавлен")
                except Exception as e:
                    logger.error(f"Ошибка инициализации PaddleOCR: {e}")
            else:
                logger.warning("PaddleOCR включен в настройках, но модуль не установлен")

        # Инициализация EasyOCR
        if settings.easyocr_enabled:
            if EASYOCR_AVAILABLE:
                try:
                    easyocr_provider.initialize()
                    self.providers.append(easyocr_provider)
                    logger.info("EasyOCR provider добавлен")
                except Exception as e:
                    logger.error(f"Ошибка инициализации EasyOCR: {e}")
            else:
                logger.warning("EasyOCR включен в настройках, но модуль не установлен")

        # Google Vision (опционально, но не в mock режиме)
        if settings.google_vision_enabled and GOOGLE_VISION_AVAILABLE:
            try:
                google_vision_provider.initialize()
                # НЕ добавляем mock-провайдер в список для реальной обработки
                if not settings.google_vision_mock_mode:
                    self.providers.append(google_vision_provider)
                    logger.info("Google Vision provider добавлен")
                else:
                    logger.info("Google Vision в MOCK-режиме - пропускаем для тестирования")
            except Exception as e:
                logger.error(f"Ошибка инициализации Google Vision: {e}")
        elif settings.google_vision_enabled and not GOOGLE_VISION_AVAILABLE:
            logger.warning("Google Vision включен в настройках, но модуль google-cloud-vision не установлен")

        if not self.providers:
            raise RuntimeError("Ни один OCR провайдер не инициализирован")

        self._initialized = True
        logger.info(f"OCR Aggregation Service инициализирован. Провайдеров: {len(self.providers)}")

    def process_image(
        self,
        image_data: Union[bytes, np.ndarray],
        language: str = "ru",
        use_all_providers: bool = True,
        skip_preprocessing: bool = False
    ) -> PassportFieldsResult:
        """
        Обработать изображение паспорта через все провайдеры.

        Args:
            image_data: Изображение в виде bytes или numpy array
            language: Язык распознавания
            use_all_providers: Использовать все провайдеры или только первый доступный
            skip_preprocessing: Пропустить встроенный препроцессинг

        Returns:
            PassportFieldsResult: Результат с извлечёнными и проголосованными полями

        Raises:
            ValueError: Если сервис не инициализирован
            RuntimeError: Если все провайдеры вернули ошибку
        """
        if not self._initialized:
            raise ValueError("OCR Aggregation Service не инициализирован. Вызовите initialize()")

        start_time = time.time()

        # Препроцессинг изображения для улучшения качества
        image_for_ocr = image_data  # По умолчанию используем оригинал
        logger.info("Preprocessing отключен, используется оригинальное изображение")

        logger.info(
            f"Начало обработки изображения. "
            f"Провайдеров: {len(self.providers)}, "
            f"Режим: {'все' if use_all_providers else 'первый доступный'}"
        )

        # Шаг 1: Распознавание через OCR провайдеры
        ocr_results = self._recognize_with_providers(
            image_data=image_for_ocr,
            language=language,
            use_all_providers=use_all_providers
        )

        if not ocr_results:
            raise RuntimeError("Все OCR провайдеры вернули ошибку")

        logger.info(f"Получено {len(ocr_results)} OCR результатов")

        # Шаг 2: Извлечение полей из каждого OCR результата
        provider_extractions = self._extract_fields_from_results(ocr_results)

        logger.info(f"Извлечено полей от {len(provider_extractions)} провайдеров")

        # Шаг 3: Скоринг и голосование
        final_result = field_scoring_service.score_and_vote(provider_extractions)

        total_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Обработка завершена за {total_time_ms:.2f}ms. "
            f"Извлечено полей: {len(final_result.final_values)}, "
            f"Средняя уверенность: {final_result.average_confidence:.2%}"
        )

        return final_result

    def _recognize_with_providers(
        self,
        image_data: Union[bytes, np.ndarray],
        language: str,
        use_all_providers: bool
    ) -> List[OCRResult]:
        """
        Выполнить распознавание через OCR провайдеры.
        БЕЗ ручного поворота - PaddleOCR сам определяет ориентацию!
        """
        ocr_results = []

        for provider in self.providers:
            try:
                logger.info(f"Запуск распознавания через {provider.provider_name}...")
            
                # Просто вызываем распознавание БЕЗ поворота
                # PaddleOCR имеет встроенную детекцию ориентации
                result = provider.recognize(image_data, language)
            
                ocr_results.append(result)
                logger.info(
                    f"{provider.provider_name}: распознано {len(result.full_text)} символов, "
                    f"conf {result.average_confidence:.2%}"
                )

                if not use_all_providers:
                    break

            except Exception as e:
                logger.error(f"Ошибка в провайдере {provider.provider_name}: {e}")
                continue

        return ocr_results 


    def _extract_fields_from_results(
        self,
        ocr_results: List[OCRResult]
    ) -> List[ProviderExtraction]:
        """
        Извлечь поля из всех OCR результатов.

        Args:
            ocr_results: Результаты от OCR провайдеров

        Returns:
            List[ProviderExtraction]: Извлечённые поля от каждого провайдера
        """
        provider_extractions = []

        for ocr_result in ocr_results:
            try:
                logger.info(f"Извлечение полей из результата {ocr_result.metadata.provider}...")

                extraction = field_extraction_service.extract_fields(ocr_result)
                provider_extractions.append(extraction)

                logger.info(
                    f"{ocr_result.metadata.provider}: "
                    f"извлечено {extraction.total_fields_found} полей"
                )

            except Exception as e:
                logger.error(
                    f"Ошибка извлечения полей из {ocr_result.metadata.provider}: {e}"
                )
                # Продолжаем с другими результатами
                continue

        return provider_extractions

    def get_available_providers(self) -> List[str]:
        """
        Получить список доступных провайдеров.

        Returns:
            List[str]: Названия провайдеров
        """
        if not self._initialized:
            return []

        return [provider.provider_name for provider in self.providers]

    def is_initialized(self) -> bool:
        """Проверка инициализации сервиса"""
        return self._initialized

    def get_provider_count(self) -> int:
        """Получить количество доступных провайдеров"""
        return len(self.providers)


# Singleton instance
ocr_aggregation_service = OCRAggregationService()
