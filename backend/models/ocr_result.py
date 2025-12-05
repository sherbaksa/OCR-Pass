from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from backend.core.database import Base


class OCRResult(Base):
    """
    Модель результата распознавания (OCR)
    """
    __tablename__ = "ocr_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Связь с документом
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Распознанные данные паспорта
    surname = Column(String(100), nullable=True)
    name = Column(String(100), nullable=True)
    patronymic = Column(String(100), nullable=True)
    birth_date = Column(String(20), nullable=True)  # Храним как строку для валидации
    
    # Паспортные данные
    series = Column(String(10), nullable=True)
    number = Column(String(10), nullable=True)
    issued_by = Column(Text, nullable=True)
    issue_date = Column(String(20), nullable=True)
    department_code = Column(String(10), nullable=True)
    
    # Адрес регистрации
    registration_address = Column(Text, nullable=True)
    
    # Сырые данные OCR (для отладки и улучшения)
    raw_ocr_data = Column(JSONB, nullable=True)
    
    # Метрики качества распознавания
    confidence_score = Column(String(10), nullable=True)  # Средняя уверенность OCR (0-100)
    
    # Проверка валидности
    is_valid = Column(Boolean, default=True, nullable=False)
    validation_errors = Column(JSONB, nullable=True)  # Список ошибок валидации
    
    # Проверка паспорта в базе МВД
    passport_check_status = Column(String(50), nullable=True)  # 'valid', 'invalid', 'not_checked'
    passport_check_date = Column(DateTime, nullable=True)
    
    # Временные метки
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<OCRResult(id={self.id}, document_id={self.document_id}, passport={self.series} {self.number})>"
