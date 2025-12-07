from backend.schemas.ocr import (
    BoundingBox,
    OCRWord,
    OCRLine,
    OCRBlock,
    PassportField,
    OCRMetadata,
    OCRResult,
    OCRRequest,
    OCRResponse,
    PassportData,
)

from backend.schemas.passport_fields import (
    PassportFieldType,
    ExtractedField,
    FieldScore,
    ProviderExtraction,
    FieldVotingResult,
    PassportFieldsResult,
    ScoringWeights,
    RegexPattern,
    PASSPORT_REGEX_PATTERNS,
    EXPECTED_FIELD_LENGTHS,
)

__all__ = [
    # OCR базовые схемы
    "BoundingBox",
    "OCRWord",
    "OCRLine",
    "OCRBlock",
    "PassportField",
    "OCRMetadata",
    "OCRResult",
    "OCRRequest",
    "OCRResponse",
    "PassportData",
    # Полевой скоринг
    "PassportFieldType",
    "ExtractedField",
    "FieldScore",
    "ProviderExtraction",
    "FieldVotingResult",
    "PassportFieldsResult",
    "ScoringWeights",
    "RegexPattern",
    "PASSPORT_REGEX_PATTERNS",
    "EXPECTED_FIELD_LENGTHS",
]
