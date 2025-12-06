#!/usr/bin/env python3
"""
Скрипт для инициализации MinIO bucket
Создает bucket если он не существует
"""

from minio import Minio
from minio.error import S3Error
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

def init_minio():
    """Инициализация MinIO и создание bucket"""
    
    # Получаем настройки из .env
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    bucket_name = os.getenv("MINIO_BUCKET_NAME", "passport-documents")
    
    # Убираем http:// из endpoint если есть
    endpoint = endpoint.replace("http://", "").replace("https://", "")
    
    print(f"[INFO] Подключение к MinIO: {endpoint}")
    print(f"[INFO] Bucket name: {bucket_name}")
    
    try:
        # Создаем клиент MinIO
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False  # HTTP, не HTTPS для локальной разработки
        )
        
        # Проверяем существование bucket
        if client.bucket_exists(bucket_name):
            print(f"[OK] Bucket '{bucket_name}' уже существует")
        else:
            # Создаем bucket
            client.make_bucket(bucket_name)
            print(f"[OK] Bucket '{bucket_name}' успешно создан")
        
        # Проверяем доступность bucket
        buckets = client.list_buckets()
        print(f"\n[INFO] Список всех buckets:")
        for bucket in buckets:
            print(f"  - {bucket.name} (создан: {bucket.creation_date})")
        
        print(f"\n[OK] MinIO инициализирован успешно!")
        return True
        
    except S3Error as e:
        print(f"[ERROR] Ошибка S3: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
        return False

if __name__ == "__main__":
    init_minio()
