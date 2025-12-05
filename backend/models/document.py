from sqlalchemy import Column, String, DateTime, Integer, Enum as SQLEnum, Text
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import enum
from backend.core.database import Base


class DocumentStatus(str, enum.Enum):
    """Статус обработки документа"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"


class Document(Base):
    """
    Модель документа (паспорта)
    """
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Метаданные файла
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)  # Путь в MinIO
    file_size = Column(Integer, nullable=False)  # Размер в байтах
    mime_type = Column(String(100), nullable=False)
    
    # Статус обработки
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.UPLOADED, nullable=False, index=True)
    
    # Тип страницы паспорта
    page_type = Column(String(50))  # 'main' или 'registration'
    
    # Временные метки
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    
    # Дополнительная информация
    error_message = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"
