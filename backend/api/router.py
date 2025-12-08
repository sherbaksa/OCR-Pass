from fastapi import APIRouter
from datetime import datetime
from backend.core.logger import log_info
from backend.api.upload import router as upload_router
from backend.api.ocr import router as ocr_router
from backend.api.passport import router as passport_router

# Создаем роутер для API v1
router = APIRouter()

# Подключаем upload роутер
router.include_router(upload_router)

# Подключаем OCR роутер
router.include_router(ocr_router)

# Подключаем Passport parser роутер
router.include_router(passport_router)


@router.get("/ping")
async def ping():
    """
    Endpoint для проверки работоспособности API
    Returns:
        dict: Статус сервиса и текущее время
    """
    log_info("Ping endpoint called")
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Passport OCR Service"
    }


@router.get("/health")
async def health_check():
    """
    Health check endpoint для мониторинга
    Returns:
        dict: Детальная информация о состоянии сервиса
    """
    log_info("Health check endpoint called")
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "api": "ok",
            "database": "not_implemented",
            "storage": "not_implemented"
        }
    }

# Debug router
from backend.api.debug import router as debug_router
router.include_router(debug_router)
