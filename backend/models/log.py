from sqlalchemy import Column, String, DateTime, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid
from backend.core.database import Base


class Log(Base):
    """
    Модель для логирования событий системы
    """
    __tablename__ = "logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Уровень лога
    level = Column(String(20), nullable=False, index=True)  # INFO, WARNING, ERROR, CRITICAL
    
    # Источник события
    source = Column(String(100), nullable=False)  # module.function
    
    # Сообщение
    message = Column(Text, nullable=False)
    
    # Связь с документом (опционально)
    document_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Дополнительные данные (JSON)
    extra_data = Column(JSONB, nullable=True)
    
    # Информация об ошибке (если есть)
    exception = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)
    
    # HTTP request info (если применимо)
    request_id = Column(String(100), nullable=True, index=True)
    user_agent = Column(String(255), nullable=True)
    ip_address = Column(String(50), nullable=True)
    
    # Временная метка
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f"<Log(id={self.id}, level={self.level}, source={self.source}, created_at={self.created_at})>"
