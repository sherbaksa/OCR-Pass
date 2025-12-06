from backend.providers.base_ocr import BaseOCRProvider
from backend.providers.paddleocr_provider import PaddleOCRProvider, paddleocr_provider

__all__ = [
    "BaseOCRProvider",
    "PaddleOCRProvider",
    "paddleocr_provider",
]
