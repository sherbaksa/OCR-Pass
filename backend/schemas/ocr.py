from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class BoundingBox(BaseModel):
    """Координаты bounding box для распознанного текста"""
    x: int = Field(..., description="X координата левого верхнего угла")
    y: int = Field(..., description="Y координата левого верхнего угла")
    width: int = Field(..., description="Ширина области")
    height: int = Field(..., description="Высота области")


class OCRWord(BaseModel):
    """Распознанное слово с координатами и уверенностью"""
    text: str = Field(..., description="Распознанный текст")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность распознавания (0-1)")
    bounding_box: Optional[BoundingBox] = Field(None, description="Координаты области")


class OCRLine(BaseModel):
    """Распознанная строка текста"""
    text: str = Field(..., description="Полный текст строки")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Средняя уверенность для строки")
    words: List[OCRWord] = Field(default_factory=list, description="Слова в строке")
    bounding_box: Optional[BoundingBox] = Field(None, description="Координаты строки")


class OCRBlock(BaseModel):
    """Блок текста (параграф или логическая группа)"""
    text: str = Field(..., description="Полный текст блока")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Средняя уверенность для блока")
    lines: List[OCRLine] = Field(default_factory=list, description="Строки в блоке")
    bounding_box: Optional[BoundingBox] = Field(None, description="Координаты блока")
    block_type: Optional[str] = Field(None, description="Тип блока (text, heading, etc.)")


class PassportField(BaseModel):
    """Распознанное поле паспорта"""
    field_name: str = Field(..., description="Название поля (surname, name, etc.)")
    value: Optional[str] = Field(None, description="Значение поля")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность распознавания")
    bounding_box: Optional[BoundingBox] = Field(None, description="Координаты поля")
    raw_value: Optional[str] = Field(None, description="Необработанное значение до постобработки")


class OCRMetadata(BaseModel):
    """Метаданные процесса OCR"""
    provider: str = Field(..., description="Название OCR-провайдера (google_vision, paddleocr)")
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах")
    image_width: int = Field(..., description="Ширина обработанного изображения")
    image_height: int = Field(..., description="Высота обработанного изображения")
    preprocessed: bool = Field(default=False, description="Была ли применена предобработка")
    preprocessing_steps: List[str] = Field(default_factory=list, description="Примененные шаги предобработки")
    language: str = Field(default="ru", description="Язык распознавания")
    model_version: Optional[str] = Field(None, description="Версия модели OCR")


class OCRResult(BaseModel):
    """Унифицированный результат OCR от любого провайдера"""
    
    # Основной распознанный текст
    full_text: str = Field(..., description="Полный распознанный текст")
    
    # Структурированные данные
    blocks: List[OCRBlock] = Field(default_factory=list, description="Блоки текста")
    
    # Средняя уверенность по всему документу
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="Средняя уверенность (0-1)")
    
    # Распознанные поля паспорта (если доступно)
    passport_fields: List[PassportField] = Field(
        default_factory=list, 
        description="Извлеченные поля паспорта"
    )
    
    # Метаданные
    metadata: OCRMetadata = Field(..., description="Метаданные процесса OCR")
    
    # Сырые данные от провайдера (для отладки)
    raw_response: Optional[Dict[str, Any]] = Field(
        None, 
        description="Необработанный ответ от OCR-провайдера"
    )
    
    # Временная метка
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Время выполнения OCR")


class OCRRequest(BaseModel):
    """Запрос на распознавание"""
    image_data: bytes = Field(..., description="Данные изображения")
    provider: str = Field(
        default="paddleocr", 
        description="OCR провайдер (google_vision, paddleocr)"
    )
    language: str = Field(default="ru", description="Язык распознавания")
    apply_preprocessing: bool = Field(
        default=True, 
        description="Применить предобработку изображения"
    )
    preprocessing_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Конфигурация предобработки"
    )
    extract_passport_fields: bool = Field(
        default=True,
        description="Извлекать поля паспорта автоматически"
    )


class OCRResponse(BaseModel):
    """Ответ API с результатами OCR"""
    success: bool = Field(..., description="Успешность операции")
    result: Optional[OCRResult] = Field(None, description="Результат OCR")
    error: Optional[str] = Field(None, description="Сообщение об ошибке")
    processing_time_ms: float = Field(..., description="Общее время обработки")


class PassportData(BaseModel):
    """Извлеченные данные паспорта РФ"""
    
    # Личные данные
    surname: Optional[str] = Field(None, description="Фамилия")
    name: Optional[str] = Field(None, description="Имя")
    patronymic: Optional[str] = Field(None, description="Отчество")
    birth_date: Optional[str] = Field(None, description="Дата рождения (DD.MM.YYYY)")
    birth_place: Optional[str] = Field(None, description="Место рождения")
    gender: Optional[str] = Field(None, description="Пол (М/Ж)")
    
    # Паспортные данные
    series: Optional[str] = Field(None, description="Серия паспорта (4 цифры)")
    number: Optional[str] = Field(None, description="Номер паспорта (6 цифр)")
    issue_date: Optional[str] = Field(None, description="Дата выдачи (DD.MM.YYYY)")
    issued_by: Optional[str] = Field(None, description="Кем выдан")
    department_code: Optional[str] = Field(None, description="Код подразделения (XXX-XXX)")
    
    # Адрес регистрации
    registration_address: Optional[str] = Field(None, description="Адрес регистрации")
    
    # Метаданные извлечения
    extraction_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Общая уверенность извлечения данных"
    )
    field_confidences: Dict[str, float] = Field(
        default_factory=dict,
        description="Уверенность для каждого поля"
    )
    
    # Валидация
    is_valid: bool = Field(default=True, description="Прошла ли валидация")
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Ошибки валидации"
    )
