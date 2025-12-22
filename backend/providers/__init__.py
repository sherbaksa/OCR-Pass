from backend.providers.base_ocr import BaseOCRProvider

# PaddleOCR - основной провайдер (обязательный)
try:
    from backend.providers.paddleocr_provider import PaddleOCRProvider, paddleocr_provider
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    PaddleOCRProvider = None
    paddleocr_provider = None
    PADDLEOCR_AVAILABLE = False
    print(f"WARNING: PaddleOCR не доступен: {e}")

# EasyOCR - опциональный провайдер
try:
    from backend.providers.easyocr_provider import EasyOCRProvider, easyocr_provider
    EASYOCR_AVAILABLE = True
except ImportError as e:
    EasyOCRProvider = None
    easyocr_provider = None
    EASYOCR_AVAILABLE = False
    print(f"WARNING: EasyOCR не доступен: {e}")

# Google Vision - опциональный провайдер
try:
    from backend.providers.google_vision_provider import GoogleVisionProvider, google_vision_provider
    GOOGLE_VISION_AVAILABLE = True
except ImportError as e:
    GoogleVisionProvider = None
    google_vision_provider = None
    GOOGLE_VISION_AVAILABLE = False
    print(f"WARNING: Google Vision не доступен: {e}")

__all__ = [
    "BaseOCRProvider",
    "PaddleOCRProvider",
    "paddleocr_provider",
    "PADDLEOCR_AVAILABLE",
    "EasyOCRProvider",
    "easyocr_provider",
    "EASYOCR_AVAILABLE",
    "GoogleVisionProvider",
    "google_vision_provider",
    "GOOGLE_VISION_AVAILABLE",
]
