"""
Сервис загрузки и валидации файлов паспортов
Обрабатывает загрузку, валидацию и сохранение в MinIO и БД
"""

from fastapi import UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Tuple, Optional
from PIL import Image
from io import BytesIO
import os
import uuid
from datetime import datetime

from backend.core.settings import settings
from backend.core.logger import logger
from backend.models.document import Document, DocumentStatus
from backend.services.storage_service import storage_service


class UploadService:
    """Сервис для загрузки и обработки файлов паспортов"""
    
    # Разрешенные MIME типы
    ALLOWED_MIME_TYPES = {
        "image/jpeg": [".jpg", ".jpeg"],
        "image/png": [".png"],
        "application/pdf": [".pdf"]
    }
    
    # Минимальное разрешение (по длинной стороне)
    MIN_RESOLUTION = 1000
    
    # Максимальный размер файла (байты)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    
    @classmethod
    async def validate_file(cls, file: UploadFile) -> Tuple[bool, Optional[str]]:
        """
        Валидация загруженного файла
        
        Args:
            file: Загруженный файл
            
        Returns:
            Tuple[bool, Optional[str]]: (валиден, сообщение об ошибке)
        """
        # Проверка размера файла
        file.file.seek(0, 2)  # Переходим в конец файла
        file_size = file.file.tell()
        file.file.seek(0)  # Возвращаемся в начало
        
        if file_size > cls.MAX_FILE_SIZE:
            return False, f"Размер файла превышает максимально допустимый ({cls.MAX_FILE_SIZE / 1024 / 1024} MB)"
        
        if file_size == 0:
            return False, "Файл пустой"
        
        # Проверка MIME типа
        content_type = file.content_type
        if content_type not in cls.ALLOWED_MIME_TYPES:
            allowed = ", ".join(cls.ALLOWED_MIME_TYPES.keys())
            return False, f"Неподдерживаемый тип файла. Разрешены: {allowed}"
        
        # Проверка расширения файла
        file_ext = os.path.splitext(file.filename)[1].lower()
        allowed_extensions = cls.ALLOWED_MIME_TYPES[content_type]
        
        if file_ext not in allowed_extensions:
            return False, f"Расширение файла {file_ext} не соответствует типу {content_type}"
        
        # Для изображений проверяем разрешение
        if content_type.startswith("image/"):
            is_valid, error = await cls.validate_image_resolution(file)
            if not is_valid:
                return False, error
        
        return True, None
    
    @classmethod
    async def validate_image_resolution(cls, file: UploadFile) -> Tuple[bool, Optional[str]]:
        """
        Проверка разрешения изображения
        
        Args:
            file: Загруженный файл изображения
            
        Returns:
            Tuple[bool, Optional[str]]: (валиден, сообщение об ошибке)
        """
        try:
            file_content = await file.read()
            image = Image.open(BytesIO(file_content))
            width, height = image.size
            
            # Возвращаем указатель в начало файла
            await file.seek(0)
            
            max_dimension = max(width, height)
            
            if max_dimension < cls.MIN_RESOLUTION:
                return False, f"Разрешение изображения слишком низкое. Минимум {cls.MIN_RESOLUTION}px по длинной стороне. Текущее: {max_dimension}px"
            
            logger.info(f"Изображение валидно: {width}x{height}px")
            return True, None
            
        except Exception as e:
            logger.error(f"Ошибка при проверке разрешения изображения: {e}")
            return False, f"Не удалось прочитать изображение: {str(e)}"
    
    @classmethod
    async def create_document_record(
        cls,
        db: AsyncSession,
        filename: str,
        file_path: str,
        file_size: int,
        mime_type: str,
        page_type: Optional[str] = None
    ) -> Document:
        """
        Создание записи документа в БД
        
        Args:
            db: Сессия базы данных
            filename: Имя файла
            file_path: Путь в MinIO
            file_size: Размер файла
            mime_type: MIME тип
            page_type: Тип страницы паспорта (main/registration)
            
        Returns:
            Document: Созданная запись документа
        """
        document = Document(
            filename=filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=mime_type,
            status=DocumentStatus.UPLOADED,
            page_type=page_type
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        logger.info(f"Создана запись документа: {document.id}")
        return document
    
    @classmethod
    async def upload_file(
        cls,
        db: AsyncSession,
        file: UploadFile,
        page_type: Optional[str] = None
    ) -> Document:
        """
        Полный процесс загрузки файла
        
        Args:
            db: Сессия базы данных
            file: Загруженный файл
            page_type: Тип страницы паспорта
            
        Returns:
            Document: Созданная запись документа
            
        Raises:
            HTTPException: При ошибке валидации или загрузки
        """
        # Валидация файла
        is_valid, error_message = await cls.validate_file(file)
        if not is_valid:
            logger.warning(f"Файл не прошел валидацию: {error_message}")
            raise HTTPException(status_code=400, detail=error_message)
        
        try:
            # Генерируем уникальное имя файла
            file_ext = os.path.splitext(file.filename)[1].lower()
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            
            # Формируем путь в MinIO (папка по датам)
            date_path = datetime.utcnow().strftime("%Y/%m/%d")
            minio_key = f"passports/{date_path}/{unique_filename}"
            
            # Читаем содержимое файла
            file_content = await file.read()
            file_size = len(file_content)
            
            # Загружаем в MinIO
            logger.info(f"Загрузка файла в MinIO: {minio_key}")
            storage_service.upload_file_object(
                file_content,
                minio_key,
                content_type=file.content_type
            )
            
            # Создаем запись в БД
            document = await cls.create_document_record(
                db=db,
                filename=file.filename,
                file_path=minio_key,
                file_size=file_size,
                mime_type=file.content_type,
                page_type=page_type
            )
            
            logger.info(f"Файл успешно загружен: document_id={document.id}")
            return document
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка при загрузке файла: {str(e)}")
    
    @classmethod
    async def update_document_status(
        cls,
        db: AsyncSession,
        document_id: uuid.UUID,
        status: DocumentStatus,
        error_message: Optional[str] = None
    ) -> Document:
        """
        Обновление статуса документа
        
        Args:
            db: Сессия базы данных
            document_id: ID документа
            status: Новый статус
            error_message: Сообщение об ошибке (опционально)
            
        Returns:
            Document: Обновленный документ
        """
        from sqlalchemy import select
        
        stmt = select(Document).where(Document.id == document_id)
        result = await db.execute(stmt)
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail="Документ не найден")
        
        document.status = status
        if error_message:
            document.error_message = error_message
        
        if status in [DocumentStatus.COMPLETED, DocumentStatus.FAILED]:
            document.processed_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(document)
        
        logger.info(f"Статус документа {document_id} обновлен: {status}")
        return document


upload_service = UploadService()
