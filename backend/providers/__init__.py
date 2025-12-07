from backend.providers.base_ocr import BaseOCRProvider
from backend.providers.paddleocr_provider import PaddleOCRProvider, paddleocr_provider
from backend.providers.google_vision_provider import GoogleVisionProvider, google_vision_provider

__all__ = [
    "BaseOCRProvider",
    "PaddleOCRProvider",
    "paddleocr_provider",
    "GoogleVisionProvider",
    "google_vision_provider",
]
