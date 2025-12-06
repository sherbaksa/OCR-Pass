from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from backend.schemas.ocr import OCRResult


class BaseOCRProvider(ABC):
    """
    Базовый абстрактный класс для всех OCR-провайдеров.
    Обеспечивает унифицированный интерфейс для разных OCR-движков.
    """
    
    def __init__(self, provider_name: str):
        """
        Инициализация OCR-провайдера
        
        Args:
            provider_name: Название провайдера (google_vision, paddleocr)
        """
        self.provider_name = provider_name
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Инициализация провайдера (загрузка моделей, настройка API и т.д.)
        Должен быть реализован в каждом конкретном провайдере.
        """
        pass
    
    @abstractmethod
    def recognize(
        self,
        image_data: Union[bytes, np.ndarray],
        language: str = "ru"
    ) -> OCRResult:
        """
        Распознавание текста на изображении.
        
        Args:
            image_data: Изображение в виде bytes или numpy array
            language: Язык распознавания (по умолчанию русский)
        
        Returns:
            OCRResult: Унифицированный результат распознавания
        
        Raises:
            ValueError: Если провайдер не инициализирован
            RuntimeError: Если произошла ошибка при распознавании
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Проверка доступности провайдера (наличие ключей, моделей и т.д.)
        
        Returns:
            bool: True если провайдер готов к работе
        """
        pass
    
    def _ensure_initialized(self) -> None:
        """
        Проверка инициализации провайдера перед использованием
        
        Raises:
            ValueError: Если провайдер не инициализирован
        """
        if not self._initialized:
            raise ValueError(
                f"OCR провайдер '{self.provider_name}' не инициализирован. "
                f"Вызовите метод initialize() перед использованием."
            )
    
    def __repr__(self) -> str:
        """Строковое представление провайдера"""
        status = "initialized" if self._initialized else "not initialized"
        return f"<{self.__class__.__name__}(provider={self.provider_name}, status={status})>"
