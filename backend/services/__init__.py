"""
Модуль сервисов OCR-Pass
Экспорт основных сервисов для использования в приложении
"""
from backend.services.storage_service import storage_service, StorageService
from backend.services.upload_service import UploadService
from backend.services.preprocessing_service import preprocessing_service, PreprocessingService
from backend.services.field_extraction_service import field_extraction_service, FieldExtractionService
from backend.services.field_scoring_service import field_scoring_service, FieldScoringService
from backend.services.ocr_aggregation_service import ocr_aggregation_service, OCRAggregationService

__all__ = [
    "storage_service",
    "StorageService",
    "UploadService",
    "preprocessing_service",
    "PreprocessingService",
    "field_extraction_service",
    "FieldExtractionService",
    "field_scoring_service",
    "FieldScoringService",
    "ocr_aggregation_service",
    "OCRAggregationService",
]
