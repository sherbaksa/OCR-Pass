"""
API endpoint для загрузки файлов паспортов
"""

from fastapi import APIRouter, UploadFile, File, Depends, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from backend.core.database import get_db
from backend.services.upload_service import upload_service
from backend.models.document import Document
from pydantic import BaseModel, Field


router = APIRouter(prefix="/upload", tags=["Upload"])


class UploadResponse(BaseModel):
    """Ответ на загрузку файла"""
    document_id: str = Field(..., description="ID созданного документа")
    filename: str = Field(..., description="Имя файла")
    file_size: int = Field(..., description="Размер файла в байтах")
    mime_type: str = Field(..., description="MIME тип файла")
    status: str = Field(..., description="Статус документа")
    page_type: Optional[str] = Field(None, description="Тип страницы паспорта")
    created_at: str = Field(..., description="Дата создания")
    
    class Config:
        from_attributes = True


@router.post(
    "",
    response_model=UploadResponse,
    summary="Загрузка файла паспорта",
    description="""
    Загрузка изображения или PDF файла паспорта.
    
    **Требования к файлу:**
    - Форматы: JPG, PNG, PDF
    - Максимальный размер: 10 MB
    - Минимальное разрешение: 1000px по длинной стороне (для изображений)
    
    **Параметры:**
    - file: файл для загрузки
    - page_type: тип страницы паспорта (main - основная, registration - прописка)
    """
)
async def upload_file(
    file: UploadFile = File(..., description="Файл паспорта (JPG, PNG или PDF)"),
    page_type: Optional[str] = Form(None, description="Тип страницы: main или registration"),
    db: AsyncSession = Depends(get_db)
) -> UploadResponse:
    """
    Загрузка файла паспорта
    
    Args:
        file: Загруженный файл
        page_type: Тип страницы паспорта
        db: Сессия базы данных
        
    Returns:
        UploadResponse: Информация о загруженном документе
    """
    # Валидация page_type
    if page_type and page_type not in ["main", "registration"]:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="page_type должен быть 'main' или 'registration'"
        )
    
    # Загрузка файла
    document = await upload_service.upload_file(
        db=db,
        file=file,
        page_type=page_type
    )
    
    # Формируем ответ
    return UploadResponse(
        document_id=str(document.id),
        filename=document.filename,
        file_size=document.file_size,
        mime_type=document.mime_type,
        status=document.status.value,
        page_type=document.page_type,
        created_at=document.created_at.isoformat()
    )
