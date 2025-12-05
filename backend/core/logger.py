import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "passport_ocr",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Настройка логгера для приложения
    
    Args:
        name: Имя логгера
        level: Уровень логирования
        log_to_file: Писать ли логи в файл
        log_dir: Директория для логов
    
    Returns:
        Настроенный logger
    """
    
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Удаляем существующие обработчики (если есть)
    if logger.handlers:
        logger.handlers.clear()
    
    # Формат логов
    log_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Консольный обработчик (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # Файловый обработчик (опционально)
    if log_to_file:
        # Создаем директорию для логов, если её нет
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Создаем файл с датой в имени
        log_filename = log_path / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    
    return logger


# Создаем глобальный экземпляр логгера
logger = setup_logger()


# Вспомогательные функции для быстрого логирования
def log_info(message: str):
    """Логирование информационного сообщения"""
    logger.info(message)


def log_error(message: str, exc_info: bool = False):
    """Логирование ошибки"""
    logger.error(message, exc_info=exc_info)


def log_warning(message: str):
    """Логирование предупреждения"""
    logger.warning(message)


def log_debug(message: str):
    """Логирование отладочного сообщения"""
    logger.debug(message)
