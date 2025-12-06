#!/usr/bin/env python3
"""
Тестовый скрипт для проверки storage_service
Проверяет все основные функции: upload, get_url, download, delete
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.storage_service import storage_service
from backend.core.logger import logger


def test_storage_service():
    """Тестирование всех функций storage_service"""
    
    print("[TEST] Начало тестирования storage_service")
    print("-" * 60)
    
    # 1. Создаем тестовый файл
    test_file_path = "/tmp/test_passport.txt"
    test_content = "Это тестовый файл паспорта для проверки MinIO storage"
    
    with open(test_file_path, "w") as f:
        f.write(test_content)
    
    print(f"[OK] Создан тестовый файл: {test_file_path}")
    
    # 2. Загрузка файла
    test_key = "test/passport_test_001.txt"
    print(f"\n[TEST] Загрузка файла с ключом: {test_key}")
    
    try:
        uploaded_key = storage_service.upload_file(test_file_path, test_key)
        print(f"[OK] Файл успешно загружен: {uploaded_key}")
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки: {e}")
        return False
    
    # 3. Проверка существования файла
    print(f"\n[TEST] Проверка существования файла")
    exists = storage_service.file_exists(test_key)
    print(f"[OK] Файл существует: {exists}")
    
    # 4. Получение метаданных
    print(f"\n[TEST] Получение метаданных файла")
    metadata = storage_service.get_file_metadata(test_key)
    if metadata:
        print(f"[OK] Метаданные получены:")
        print(f"  - Размер: {metadata['size']} bytes")
        print(f"  - Content-Type: {metadata['content_type']}")
        print(f"  - Последнее изменение: {metadata['last_modified']}")
    else:
        print(f"[ERROR] Не удалось получить метаданные")
    
    # 5. Получение presigned URL
    print(f"\n[TEST] Генерация presigned URL")
    try:
        url = storage_service.get_file_url(test_key, expires=3600)
        print(f"[OK] URL сгенерирован:")
        print(f"  {url}")
    except Exception as e:
        print(f"[ERROR] Ошибка генерации URL: {e}")
        return False
    
    # 6. Скачивание файла
    download_path = "/tmp/test_passport_downloaded.txt"
    print(f"\n[TEST] Скачивание файла в: {download_path}")
    
    try:
        storage_service.download_file(test_key, download_path)
        print(f"[OK] Файл успешно скачан")
        
        # Проверяем содержимое
        with open(download_path, "r") as f:
            downloaded_content = f.read()
        
        if downloaded_content == test_content:
            print(f"[OK] Содержимое файла совпадает")
        else:
            print(f"[ERROR] Содержимое файла не совпадает!")
            
    except Exception as e:
        print(f"[ERROR] Ошибка скачивания: {e}")
        return False
    
    # 7. Тест upload_file_object (загрузка из байтов)
    print(f"\n[TEST] Загрузка файла из байтов")
    test_key_bytes = "test/passport_test_002.txt"
    test_bytes = b"Test content from bytes"
    
    try:
        storage_service.upload_file_object(test_bytes, test_key_bytes, "text/plain")
        print(f"[OK] Файл из байтов успешно загружен: {test_key_bytes}")
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки из байтов: {e}")
    
    # 8. Удаление файлов
    print(f"\n[TEST] Удаление тестовых файлов")
    
    try:
        storage_service.delete_file(test_key)
        print(f"[OK] Удален файл: {test_key}")
        
        storage_service.delete_file(test_key_bytes)
        print(f"[OK] Удален файл: {test_key_bytes}")
        
        # Проверяем что файлы удалены
        if not storage_service.file_exists(test_key):
            print(f"[OK] Подтверждено удаление: {test_key}")
        
    except Exception as e:
        print(f"[ERROR] Ошибка удаления: {e}")
        return False
    
    # Очистка локальных файлов
    os.remove(test_file_path)
    os.remove(download_path)
    print(f"\n[OK] Локальные тестовые файлы удалены")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Все тесты пройдены успешно!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_storage_service()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
