"""
Модели базы данных
"""
from backend.models.document import Document, DocumentStatus
from backend.models.ocr_result import OCRResult
from backend.models.log import Log

__all__ = [
    "Document",
    "DocumentStatus",
    "OCRResult",
    "Log",
]
