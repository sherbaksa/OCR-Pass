"""
Сервис для работы с S3-совместимым хранилищем (MinIO)
Предоставляет функции для загрузки, получения и удаления файлов
"""

from minio import Minio
from minio.error import S3Error
from typing import Optional
import os
from io import BytesIO
from datetime import timedelta

from backend.core.settings import settings
from backend.core.logger import logger


class StorageService:
    """Сервис для работы с MinIO хранилищем"""
    
    def __init__(self):
        """Инициализация клиента MinIO"""
        # Убираем http:// из endpoint
        endpoint = settings.minio_endpoint.replace("http://", "").replace("https://", "")
        
        self.client = Minio(
            endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=False  # HTTP для локальной разработки
        )
        self.bucket_name = settings.minio_bucket_name
        
        # Проверяем существование bucket при инициализации
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self) -> None:
        """Проверка и создание bucket если не существует"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Bucket '{self.bucket_name}' создан")
            else:
                logger.info(f"Bucket '{self.bucket_name}' существует")
        except S3Error as e:
            logger.error(f"Ошибка при проверке bucket: {e}")
            raise
    
    def upload_file(self, file_path: str, object_name: str) -> str:
        """
        Загрузка файла в MinIO
        
        Args:
            file_path: Путь к локальному файлу
            object_name: Имя объекта в хранилище (ключ)
        
        Returns:
            str: Ключ загруженного файла
        
        Raises:
            FileNotFoundError: Если файл не найден
            S3Error: Если ошибка при загрузке
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        try:
            # Определяем content type по расширению
            content_type = self._get_content_type(file_path)
            
            # Загружаем файл
            self.client.fput_object(
                self.bucket_name,
                object_name,
                file_path,
                content_type=content_type
            )
            
            logger.info(f"Файл загружен: {object_name}")
            return object_name
            
        except S3Error as e:
            logger.error(f"Ошибка загрузки файла {object_name}: {e}")
            raise
    
    def upload_file_object(self, file_data: bytes, object_name: str, content_type: str = "application/octet-stream") -> str:
        """
        Загрузка файла из байтов в MinIO
        
        Args:
            file_data: Данные файла в байтах
            object_name: Имя объекта в хранилище (ключ)
            content_type: MIME тип файла
        
        Returns:
            str: Ключ загруженного файла
        
        Raises:
            S3Error: Если ошибка при загрузке
        """
        try:
            file_stream = BytesIO(file_data)
            file_size = len(file_data)
            
            self.client.put_object(
                self.bucket_name,
                object_name,
                file_stream,
                file_size,
                content_type=content_type
            )
            
            logger.info(f"Файл загружен из памяти: {object_name}")
            return object_name
            
        except S3Error as e:
            logger.error(f"Ошибка загрузки файла {object_name}: {e}")
            raise
    
    def get_file_url(self, object_name: str, expires: int = 3600) -> str:
        """
        Получение presigned URL для доступа к файлу
        
        Args:
            object_name: Имя объекта в хранилище
            expires: Время жизни URL в секундах (по умолчанию 1 час)
        
        Returns:
            str: Presigned URL для доступа к файлу
        
        Raises:
            S3Error: Если ошибка при генерации URL
        """
        try:
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=timedelta(seconds=expires)
            )
            logger.info(f"Сгенерирован URL для {object_name}, expires={expires}s")
            return url
            
        except S3Error as e:
            logger.error(f"Ошибка генерации URL для {object_name}: {e}")
            raise
    
    def download_file(self, object_name: str, file_path: str) -> str:
        """
        Скачивание файла из MinIO
        
        Args:
            object_name: Имя объекта в хранилище
            file_path: Путь для сохранения файла
        
        Returns:
            str: Путь к скачанному файлу
        
        Raises:
            S3Error: Если ошибка при скачивании
        """
        try:
            self.client.fget_object(
                self.bucket_name,
                object_name,
                file_path
            )
            logger.info(f"Файл скачан: {object_name} -> {file_path}")
            return file_path
            
        except S3Error as e:
            logger.error(f"Ошибка скачивания файла {object_name}: {e}")
            raise
    
    def delete_file(self, object_name: str) -> None:
        """
        Удаление файла из MinIO
        
        Args:
            object_name: Имя объекта в хранилище
        
        Raises:
            S3Error: Если ошибка при удалении
        """
        try:
            self.client.remove_object(self.bucket_name, object_name)
            logger.info(f"Файл удален: {object_name}")
            
        except S3Error as e:
            logger.error(f"Ошибка удаления файла {object_name}: {e}")
            raise
    
    def file_exists(self, object_name: str) -> bool:
        """
        Проверка существования файла
        
        Args:
            object_name: Имя объекта в хранилище
        
        Returns:
            bool: True если файл существует, False иначе
        """
        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error:
            return False
    
    def get_file_metadata(self, object_name: str) -> Optional[dict]:
        """
        Получение метаданных файла
        
        Args:
            object_name: Имя объекта в хранилище
        
        Returns:
            dict: Метаданные файла или None если файл не найден
        """
        try:
            stat = self.client.stat_object(self.bucket_name, object_name)
            return {
                "size": stat.size,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified,
                "etag": stat.etag,
                "metadata": stat.metadata
            }
        except S3Error as e:
            logger.error(f"Ошибка получения метаданных {object_name}: {e}")
            return None
    
    @staticmethod
    def _get_content_type(file_path: str) -> str:
        """Определение MIME типа по расширению файла"""
        ext = os.path.splitext(file_path)[1].lower()
        content_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.json': 'application/json',
        }
        return content_types.get(ext, 'application/octet-stream')


# Singleton instance
storage_service = StorageService()
