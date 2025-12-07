"""
API endpoint для OCR обработки паспортов
Обрабатывает изображения через агрегацию нескольких OCR провайдеров
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional
import time

from backend.services.ocr_aggregation_service import ocr_aggregation_service
from backend.schemas.passport_fields import PassportFieldsResult
from backend.core.logger import logger
from pydantic import BaseModel, Field


router = APIRouter(prefix="/ocr", tags=["OCR"])


class OCRProcessResponse(BaseModel):
    """Ответ на запрос обработки OCR"""
    success: bool = Field(..., description="Успешность операции")
    result: Optional[PassportFieldsResult] = Field(None, description="Результат обработки")
    error: Optional[str] = Field(None, description="Сообщение об ошибке")
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах")
    providers_used: int = Field(..., description="Количество использованных провайдеров")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "result": {
                    "final_values": {
                        "series": "4509",
                        "number": "123456",
                        "surname": "Иванов",
                        "name": "Иван"
                    },
                    "field_confidences": {
                        "series": 0.95,
                        "number": 0.93,
                        "surname": 0.88,
                        "name": 0.91
                    },
                    "average_confidence": 0.92,
                    "is_valid": True
                },
                "error": None,
                "processing_time_ms": 1234.56,
                "providers_used": 2
            }
        }


class OCRHealthResponse(BaseModel):
    """Ответ на health check OCR сервиса"""
    status: str = Field(..., description="Статус сервиса")
    initialized: bool = Field(..., description="Инициализирован ли сервис")
    providers_available: int = Field(..., description="Количество доступных провайдеров")
    providers_list: list[str] = Field(..., description="Список провайдеров")


@router.post(
    "/process",
    response_model=OCRProcessResponse,
    summary="Обработка изображения паспорта через OCR",
    description="""
    Обрабатывает изображение паспорта через несколько OCR провайдеров.
    
    **Процесс обработки:**
    1. Распознавание текста через все доступные OCR провайдеры (PaddleOCR, Google Vision)
    2. Извлечение структурированных полей паспорта из распознанного текста
    3. Многофакторный скоринг каждого поля (regex, кириллица, длина, формат)
    4. Голосование между провайдерами для выбора лучших значений
    
    **Извлекаемые поля:**
    - Серия и номер паспорта
    - ФИО (фамилия, имя, отчество)
    - Даты (рождения, выдачи)
    - Код подразделения
    - Пол
    - Место рождения
    - Кем выдан
    - Адрес регистрации
    
    **Требования к файлу:**
    - Форматы: JPG, PNG, JPEG
    - Максимальный размер: 10 MB
    - Рекомендуемое разрешение: 1500x1000 или выше
    
    **Параметры:**
    - file: файл изображения паспорта
    - use_all_providers: использовать все провайдеры (по умолчанию true)
    - language: язык распознавания (по умолчанию ru)
    """
)
async def process_ocr(
    file: UploadFile = File(..., description="Файл изображения паспорта (JPG, PNG)"),
    use_all_providers: bool = Query(
        default=True,
        description="Использовать все доступные провайдеры или только первый"
    ),
    language: str = Query(
        default="ru",
        description="Язык распознавания"
    )
) -> OCRProcessResponse:
    """
    Обработать изображение паспорта через OCR агрегацию
    
    Args:
        file: Загруженный файл изображения
        use_all_providers: Использовать все провайдеры
        language: Язык распознавания
        
    Returns:
        OCRProcessResponse: Результат обработки с извлечёнными полями
    """
    start_time = time.time()
    
    logger.info(
        f"Получен запрос на OCR обработку. "
        f"Файл: {file.filename}, "
        f"Размер: {file.size if hasattr(file, 'size') else 'unknown'}, "
        f"Все провайдеры: {use_all_providers}"
    )
    
    # Валидация типа файла
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый тип файла: {file.content_type}. "
                   f"Разрешены только JPG и PNG"
        )
    
    try:
        # Инициализация сервиса если ещё не инициализирован
        if not ocr_aggregation_service.is_initialized():
            logger.info("Инициализация OCR Aggregation Service...")
            ocr_aggregation_service.initialize()
        
        # Чтение файла
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="Загружен пустой файл"
            )
        
        # Обработка через OCR агрегацию
        result = ocr_aggregation_service.process_image(
            image_data=image_data,
            language=language,
            use_all_providers=use_all_providers
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"OCR обработка завершена за {processing_time_ms:.2f}ms. "
            f"Извлечено полей: {len(result.final_values)}, "
            f"Средняя уверенность: {result.average_confidence:.2%}, "
            f"Валидность: {result.is_valid}"
        )
        
        return OCRProcessResponse(
            success=True,
            result=result,
            error=None,
            processing_time_ms=processing_time_ms,
            providers_used=result.total_providers_used
        )
        
    except ValueError as e:
        logger.error(f"Ошибка валидации: {e}")
        processing_time_ms = (time.time() - start_time) * 1000
        
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Ошибка при обработке OCR: {e}", exc_info=True)
        processing_time_ms = (time.time() - start_time) * 1000
        
        return OCRProcessResponse(
            success=False,
            result=None,
            error=str(e),
            processing_time_ms=processing_time_ms,
            providers_used=0
        )


@router.get(
    "/health",
    response_model=OCRHealthResponse,
    summary="Health check OCR сервиса",
    description="Проверка состояния OCR сервиса и доступных провайдеров"
)
async def ocr_health() -> OCRHealthResponse:
    """
    Проверить состояние OCR сервиса
    
    Returns:
        OCRHealthResponse: Статус сервиса и список провайдеров
    """
    try:
        # Инициализация если нужно
        if not ocr_aggregation_service.is_initialized():
            ocr_aggregation_service.initialize()
        
        providers = ocr_aggregation_service.get_available_providers()
        
        return OCRHealthResponse(
            status="healthy",
            initialized=True,
            providers_available=len(providers),
            providers_list=providers
        )
        
    except Exception as e:
        logger.error(f"Ошибка health check OCR: {e}")
        
        return OCRHealthResponse(
            status="error",
            initialized=False,
            providers_available=0,
            providers_list=[]
        )


@router.get(
    "/providers",
    summary="Список доступных OCR провайдеров",
    description="Получить список всех доступных OCR провайдеров"
)
async def get_providers() -> dict:
    """
    Получить список доступных провайдеров
    
    Returns:
        dict: Информация о провайдерах
    """
    try:
        if not ocr_aggregation_service.is_initialized():
            ocr_aggregation_service.initialize()
        
        providers = ocr_aggregation_service.get_available_providers()
        
        return {
            "success": True,
            "providers": providers,
            "count": len(providers)
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения списка провайдеров: {e}")
        
        return {
            "success": False,
            "providers": [],
            "count": 0,
            "error": str(e)
        }
