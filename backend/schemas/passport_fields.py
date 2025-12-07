"""
Схемы данных для извлечения и скоринга полей паспорта РФ.
Используется в Field Extraction Service и Field Scoring Service.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum


class PassportFieldType(str, Enum):
    """Типы полей паспорта РФ"""
    SURNAME = "surname"
    NAME = "name"
    PATRONYMIC = "patronymic"
    BIRTH_DATE = "birth_date"
    BIRTH_PLACE = "birth_place"
    GENDER = "gender"
    SERIES = "series"
    NUMBER = "number"
    ISSUE_DATE = "issue_date"
    ISSUED_BY = "issued_by"
    DEPARTMENT_CODE = "department_code"
    REGISTRATION_ADDRESS = "registration_address"


class ExtractedField(BaseModel):
    """
    Одно извлечённое поле паспорта от одного провайдера.
    """
    field_type: PassportFieldType = Field(..., description="Тип поля")
    value: Optional[str] = Field(None, description="Извлечённое значение")
    raw_value: Optional[str] = Field(None, description="Необработанное значение")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Базовая уверенность OCR")
    source_provider: str = Field(..., description="Провайдер-источник (paddleocr, google_vision)")
    
    class Config:
        use_enum_values = True


class FieldScore(BaseModel):
    """
    Оценка качества извлечённого поля на основе многофакторного скоринга.
    """
    field_type: PassportFieldType = Field(..., description="Тип поля")
    value: Optional[str] = Field(None, description="Значение поля")
    
    # Базовая уверенность от OCR провайдера
    ocr_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Уверенность OCR (0-1)")
    
    # Факторы скоринга
    regex_match_score: float = Field(0.0, ge=0.0, le=1.0, description="Совпадение с regex-паттерном (0-1)")
    cyrillic_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Доля кириллических символов (0-1)")
    length_score: float = Field(0.0, ge=0.0, le=1.0, description="Соответствие ожидаемой длине (0-1)")
    format_score: float = Field(0.0, ge=0.0, le=1.0, description="Соответствие формату (даты, коды) (0-1)")
    
    # Итоговый скор
    total_score: float = Field(0.0, ge=0.0, le=1.0, description="Итоговая оценка поля (0-1)")
    
    # Метаданные
    source_provider: str = Field(..., description="Провайдер-источник")
    scoring_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Детализация скоринга для отладки"
    )
    
    class Config:
        use_enum_values = True


class ProviderExtraction(BaseModel):
    """
    Результат извлечения полей от одного провайдера.
    """
    provider_name: str = Field(..., description="Название провайдера")
    fields: List[ExtractedField] = Field(default_factory=list, description="Извлечённые поля")
    extraction_time_ms: float = Field(..., description="Время извлечения в миллисекундах")
    total_fields_found: int = Field(0, description="Количество найденных полей")
    
    def get_field(self, field_type: PassportFieldType) -> Optional[ExtractedField]:
        """Получить поле по типу"""
        for field in self.fields:
            if field.field_type == field_type:
                return field
        return None


class FieldVotingResult(BaseModel):
    """
    Результат голосования для одного поля.
    """
    field_type: PassportFieldType = Field(..., description="Тип поля")
    selected_value: Optional[str] = Field(None, description="Выбранное значение")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Уверенность в выбранном значении")
    
    # Все кандидаты с оценками
    candidates: List[FieldScore] = Field(default_factory=list, description="Все варианты от провайдеров")
    
    # Победитель
    winner_provider: Optional[str] = Field(None, description="Провайдер-победитель")
    winner_score: float = Field(0.0, description="Оценка победителя")
    
    # Статистика голосования
    total_candidates: int = Field(0, description="Количество кандидатов")
    agreement_score: float = Field(0.0, ge=0.0, le=1.0, description="Согласованность провайдеров (0-1)")
    
    class Config:
        use_enum_values = True


class PassportFieldsResult(BaseModel):
    """
    Финальный результат извлечения и скоринга всех полей паспорта.
    """
    # Извлечённые данные по провайдерам
    provider_extractions: List[ProviderExtraction] = Field(
        default_factory=list,
        description="Результаты от каждого провайдера"
    )
    
    # Результаты голосования по полям
    field_results: Dict[str, FieldVotingResult] = Field(
        default_factory=dict,
        description="Результаты голосования по каждому полю"
    )
    
    # Финальные значения (для удобства)
    final_values: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Финальные значения полей после голосования"
    )
    
    # Уверенность по полям
    field_confidences: Dict[str, float] = Field(
        default_factory=dict,
        description="Уверенность для каждого поля (0-1)"
    )
    
    # Общая статистика
    total_providers_used: int = Field(0, description="Количество использованных провайдеров")
    average_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Средняя уверенность по всем полям")
    fields_with_high_confidence: int = Field(0, description="Количество полей с confidence > 0.8")
    fields_with_low_confidence: int = Field(0, description="Количество полей с confidence < 0.5")
    
    # Валидация
    is_valid: bool = Field(True, description="Прошла ли общая валидация")
    validation_errors: List[str] = Field(default_factory=list, description="Ошибки валидации")
    
    def get_field_value(self, field_type: PassportFieldType) -> Optional[str]:
        """Получить финальное значение поля"""
        return self.final_values.get(field_type.value)
    
    def get_field_confidence(self, field_type: PassportFieldType) -> float:
        """Получить уверенность для поля"""
        return self.field_confidences.get(field_type.value, 0.0)


class ScoringWeights(BaseModel):
    """
    Веса для различных факторов скоринга.
    Позволяет настраивать важность каждого фактора.
    """
    ocr_confidence_weight: float = Field(0.3, ge=0.0, le=1.0, description="Вес OCR confidence")
    regex_match_weight: float = Field(0.3, ge=0.0, le=1.0, description="Вес совпадения с regex")
    cyrillic_ratio_weight: float = Field(0.15, ge=0.0, le=1.0, description="Вес доли кириллицы")
    length_score_weight: float = Field(0.15, ge=0.0, le=1.0, description="Вес корректности длины")
    format_score_weight: float = Field(0.1, ge=0.0, le=1.0, description="Вес корректности формата")
    
    def validate_weights(self) -> bool:
        """Проверка, что сумма весов примерно равна 1.0"""
        total = (
            self.ocr_confidence_weight +
            self.regex_match_weight +
            self.cyrillic_ratio_weight +
            self.length_score_weight +
            self.format_score_weight
        )
        return abs(total - 1.0) < 0.01
    
    @classmethod
    def default_weights(cls) -> "ScoringWeights":
        """Веса по умолчанию"""
        return cls()
    
    @classmethod
    def strict_regex_weights(cls) -> "ScoringWeights":
        """Веса с акцентом на regex (для номеров, кодов)"""
        return cls(
            ocr_confidence_weight=0.2,
            regex_match_weight=0.5,
            cyrillic_ratio_weight=0.1,
            length_score_weight=0.1,
            format_score_weight=0.1
        )
    
    @classmethod
    def text_quality_weights(cls) -> "ScoringWeights":
        """Веса с акцентом на качество текста (для ФИО, адресов)"""
        return cls(
            ocr_confidence_weight=0.3,
            regex_match_weight=0.15,
            cyrillic_ratio_weight=0.35,
            length_score_weight=0.1,
            format_score_weight=0.1
        )


class RegexPattern(BaseModel):
    """
    Regex-паттерн для валидации поля паспорта.
    """
    field_type: PassportFieldType = Field(..., description="Тип поля")
    pattern: str = Field(..., description="Regex паттерн")
    description: str = Field(..., description="Описание паттерна")
    examples: List[str] = Field(default_factory=list, description="Примеры корректных значений")
    
    class Config:
        use_enum_values = True


# Предопределённые regex-паттерны для полей паспорта РФ
PASSPORT_REGEX_PATTERNS: Dict[PassportFieldType, RegexPattern] = {
    PassportFieldType.SERIES: RegexPattern(
        field_type=PassportFieldType.SERIES,
        pattern=r"^\d{4}$",
        description="Серия паспорта: 4 цифры",
        examples=["4509", "4510", "4620"]
    ),
    PassportFieldType.NUMBER: RegexPattern(
        field_type=PassportFieldType.NUMBER,
        pattern=r"^\d{6}$",
        description="Номер паспорта: 6 цифр",
        examples=["123456", "987654", "555555"]
    ),
    PassportFieldType.BIRTH_DATE: RegexPattern(
        field_type=PassportFieldType.BIRTH_DATE,
        pattern=r"^\d{2}\.\d{2}\.\d{4}$",
        description="Дата рождения: ДД.ММ.ГГГГ",
        examples=["15.06.1990", "01.01.1985", "31.12.2000"]
    ),
    PassportFieldType.ISSUE_DATE: RegexPattern(
        field_type=PassportFieldType.ISSUE_DATE,
        pattern=r"^\d{2}\.\d{2}\.\d{4}$",
        description="Дата выдачи: ДД.ММ.ГГГГ",
        examples=["20.07.2010", "15.03.2015", "01.01.2020"]
    ),
    PassportFieldType.DEPARTMENT_CODE: RegexPattern(
        field_type=PassportFieldType.DEPARTMENT_CODE,
        pattern=r"^\d{3}-\d{3}$",
        description="Код подразделения: XXX-XXX",
        examples=["770-001", "450-089", "500-123"]
    ),
    PassportFieldType.GENDER: RegexPattern(
        field_type=PassportFieldType.GENDER,
        pattern=r"^[МЖ]$",
        description="Пол: М или Ж",
        examples=["М", "Ж"]
    ),
    PassportFieldType.SURNAME: RegexPattern(
        field_type=PassportFieldType.SURNAME,
        pattern=r"^[А-ЯЁ][а-яё]+(-[А-ЯЁ][а-яё]+)?$",
        description="Фамилия: кириллица, заглавная первая буква",
        examples=["Иванов", "Петров-Водкин", "Сидорова"]
    ),
    PassportFieldType.NAME: RegexPattern(
        field_type=PassportFieldType.NAME,
        pattern=r"^[А-ЯЁ][а-яё]+$",
        description="Имя: кириллица, заглавная первая буква",
        examples=["Иван", "Мария", "Александр"]
    ),
    PassportFieldType.PATRONYMIC: RegexPattern(
        field_type=PassportFieldType.PATRONYMIC,
        pattern=r"^[А-ЯЁ][а-яё]+(ович|евич|ич|овна|евна|ична)$",
        description="Отчество: кириллица, типичные окончания",
        examples=["Иванович", "Петровна", "Сергеевич"]
    ),
}


# Ожидаемые длины полей
EXPECTED_FIELD_LENGTHS: Dict[PassportFieldType, tuple] = {
    PassportFieldType.SERIES: (4, 4),  # точно 4 цифры
    PassportFieldType.NUMBER: (6, 6),  # точно 6 цифр
    PassportFieldType.DEPARTMENT_CODE: (7, 7),  # XXX-XXX = 7 символов
    PassportFieldType.BIRTH_DATE: (10, 10),  # ДД.ММ.ГГГГ = 10 символов
    PassportFieldType.ISSUE_DATE: (10, 10),  # ДД.ММ.ГГГГ = 10 символов
    PassportFieldType.GENDER: (1, 1),  # М или Ж = 1 символ
    PassportFieldType.SURNAME: (2, 40),  # от 2 до 40 символов
    PassportFieldType.NAME: (2, 30),  # от 2 до 30 символов
    PassportFieldType.PATRONYMIC: (3, 30),  # от 3 до 30 символов
    PassportFieldType.BIRTH_PLACE: (10, 200),  # от 10 до 200 символов
    PassportFieldType.ISSUED_BY: (20, 300),  # от 20 до 300 символов
    PassportFieldType.REGISTRATION_ADDRESS: (20, 500),  # от 20 до 500 символов
}
