from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.settings import settings
from backend.core.logger import logger, log_info
from backend.api.router import router as api_router


# Инициализация приложения
app = FastAPI(
    title=settings.app_name,
    description="Сервис для загрузки, обработки и распознавания изображений паспортов РФ",
    version="0.1.0",
    debug=settings.debug
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутера API v1
app.include_router(api_router, prefix=settings.api_v1_prefix, tags=["API v1"])


@app.on_event("startup")
async def startup_event():
    """Событие при запуске приложения"""
    log_info(f"Starting {settings.app_name}")
    log_info(f"Debug mode: {settings.debug}")
    log_info(f"API v1 prefix: {settings.api_v1_prefix}")


@app.on_event("shutdown")
async def shutdown_event():
    """Событие при остановке приложения"""
    log_info(f"Shutting down {settings.app_name}")


@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": f"{settings.app_name} API",
        "status": "healthy",
        "version": "0.1.0",
        "docs_url": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check для корневого уровня"""
    return {"status": "healthy"}
