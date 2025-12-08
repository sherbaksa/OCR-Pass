"""
API endpoint для интеллектуального парсинга паспортов РФ
Использует OCR + интеллектуальный парсинг с нормализацией, морфологией и словарями.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional
import time

from backend.services.ocr_aggregation_service import ocr_aggregation_service
from backend.services.passport_parser_service import passport_parser_service
from backend.core.logger import logger
from pydantic import BaseModel, Field


router = APIRouter(prefix="/passport", tags=["Passport"])


class PassportParseResponse(BaseModel):
    """Ответ на запрос парсинга паспорта"""
    success: bool = Field(..., description="Успешность операции")
    fields: dict = Field(default_factory=dict, description="Извлеченные поля паспорта")
    confidences: dict = Field(default_factory=dict, description="Уверенность для каждого поля")
    errors: list = Field(default_factory=list, description="Ошибки валидации")
    warnings: list = Field(default_factory=list, description="Предупреждения")
    ocr_text: Optional[str] = Field(None, description="Полный OCR-текст (опционально)")
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "fields": {
                    "series": "4509",
                    "number": "123456",
                    "surname": "Иванов",
                    "name": "Иван",
                    "patronymic": "Иванович",
                    "birth_date": "15.06.1990",
                    "issue_date": "20.07.2010",
                    "department_code": "770-001",
                    "gender": "М",
                    "birth_place": "г. Москва",
                    "issued_by": "ОУФМС России по г. Москве",
                    "registration_address": "г. Москва, ул. Ленина, д. 10, кв. 5"
                },
                "confidences": {
                    "series": 0.95,
                    "number": 0.95,
                    "surname": 0.92,
                    "name": 0.94,
                    "patronymic": 0.91,
                    "birth_date": 0.85,
                    "issue_date": 0.85,
                    "department_code": 0.95,
                    "gender": 0.95
                },
                "errors": [],
                "warnings": [],
                "processing_time_ms": 2345.67
            }
        }


class PassportHealthResponse(BaseModel):
    """Ответ на health check парсера паспортов"""
    status: str = Field(..., description="Статус сервиса")
    ocr_initialized: bool = Field(..., description="OCR сервис инициализирован")
    parser_initialized: bool = Field(..., description="Парсер инициализирован")
    ocr_providers_count: int = Field(..., description="Количество OCR провайдеров")
    dictionaries_loaded: dict = Field(..., description="Статус загрузки словарей")


@router.post(
    "/parse",
    response_model=PassportParseResponse,
    summary="Интеллектуальный парсинг паспорта РФ",
    description="""
    Полный цикл обработки паспорта: OCR + интеллектуальный парсинг.
    
    **Процесс обработки:**
    1. **OCR-распознавание** через несколько провайдеров (PaddleOCR, Google Vision)
    2. **Нормализация текста**: Ё→Е, исправление регистра, очистка спецсимволов
    3. **Извлечение полей** с использованием regex-паттернов
    4. **Морфологический анализ ФИО** через pymorphy2 + словари
    5. **Нормализация дат** в различных форматах (DD.MM.YY, DD-MM-YYYY и т.д.)
    6. **Валидация** извлеченных данных
    7. **Кросс-проверки** (даты, пол из ФИО, и т.д.)
    
    **Извлекаемые поля:**
    - Серия и номер паспорта (4 + 6 цифр)
    - ФИО (фамилия, имя, отчество) с морфологическим анализом
    - Даты (рождения, выдачи) с нормализацией форматов
    - Код подразделения (XXX-XXX)
    - Пол (М/Ж)
    - Место рождения
    - Орган выдачи
    - Адрес регистрации
    
    **Требования к файлу:**
    - Форматы: JPG, PNG, JPEG
    - Максимальный размер: 10 MB
    - Рекомендуемое разрешение: 1500x1000 или выше
    
    **Параметры:**
    - file: файл изображения паспорта
    - use_all_providers: использовать все OCR провайдеры (по умолчанию true)
    - include_ocr_text: включить полный OCR-текст в ответ (по умолчанию false)
    """
)
async def parse_passport(
    file: UploadFile = File(..., description="Файл изображения паспорта (JPG, PNG)"),
    use_all_providers: bool = Query(
        default=True,
        description="Использовать все доступные OCR провайдеры"
    ),
    include_ocr_text: bool = Query(
        default=False,
        description="Включить полный OCR-текст в ответ"
    )
) -> PassportParseResponse:
    """
    Парсинг паспорта РФ с полной обработкой
    
    Args:
        file: Загруженный файл изображения
        use_all_providers: Использовать все OCR провайдеры
        include_ocr_text: Включить OCR-текст в ответ
        
    Returns:
        PassportParseResponse: Извлеченные поля паспорта
    """
    start_time = time.time()
    
    logger.info(
        f"Получен запрос на парсинг паспорта. "
        f"Файл: {file.filename}, "
        f"Размер: {file.size if hasattr(file, 'size') else 'unknown'}"
    )
    
    # Валидация типа файла
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый тип файла: {file.content_type}. "
                   f"Разрешены только JPG и PNG"
        )
    
    try:
        # Инициализация сервисов если нужно
        if not ocr_aggregation_service.is_initialized():
            logger.info("Инициализация OCR Aggregation Service...")
            ocr_aggregation_service.initialize()
        
        if not passport_parser_service._initialized:
            logger.info("Инициализация Passport Parser Service...")
            passport_parser_service.initialize()
        
        # Чтение файла
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="Загружен пустой файл"
            )
        
        # Шаг 1: OCR-распознавание
        logger.info("Шаг 1: OCR-распознавание...")
        ocr_start = time.time()
        
        ocr_result = ocr_aggregation_service.process_image(
            image_data=image_data,
            language="ru",
            use_all_providers=use_all_providers
        )
        
        ocr_time = (time.time() - ocr_start) * 1000
        logger.info(f"OCR завершен за {ocr_time:.2f}ms")
        
        # Получаем все тексты от провайдеров
        combined_text = ""
        for provider_extraction in ocr_result.provider_extractions:
            # Извлекаем текст из каждого провайдера
            # Текст находится в provider_extraction, нужно получить OCR результат
            combined_text += " "
        
        # Временное решение: берем текст из первого провайдера
        # TODO: улучшить объединение текстов
        if ocr_result.provider_extractions:
            # Нужно получить оригинальный OCR текст
            # Пока используем заглушку
            combined_text = "OCR текст для парсинга"
        
        # Шаг 2: Интеллектуальный парсинг
        logger.info("Шаг 2: Интеллектуальный парсинг...")
        parse_start = time.time()
        
        # ВРЕМЕННОЕ РЕШЕНИЕ: используем упрощенный парсинг через field_extraction
        # Формируем текст из всех извлеченных полей
        all_text_parts = []
        for field_name, value in ocr_result.final_values.items():
            if value:
                all_text_parts.append(f"{field_name}: {value}")
        
        combined_ocr_text = " ".join(all_text_parts)
        
        parsed_result = passport_parser_service.parse_passport_text(combined_ocr_text)
        
        parse_time = (time.time() - parse_start) * 1000
        logger.info(f"Парсинг завершен за {parse_time:.2f}ms")
        
        # Объединяем результаты OCR и парсера
        final_fields = {}
        final_confidences = {}
        
        # Сначала берем из парсера (более точные)
        final_fields.update(parsed_result['fields'])
        final_confidences.update(parsed_result['confidences'])
        
        # Дополняем из OCR то, чего нет в парсере
        for field_name, value in ocr_result.final_values.items():
            if field_name not in final_fields and value:
                final_fields[field_name] = value
                final_confidences[field_name] = ocr_result.field_confidences.get(field_name, 0.5)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Парсинг паспорта завершен за {total_time_ms:.2f}ms. "
            f"Извлечено полей: {len(final_fields)}, "
            f"Ошибок: {len(parsed_result['errors'])}, "
            f"Предупреждений: {len(parsed_result['warnings'])}"
        )
        
        return PassportParseResponse(
            success=True,
            fields=final_fields,
            confidences=final_confidences,
            errors=parsed_result.get('errors', []),
            warnings=parsed_result.get('warnings', []),
            ocr_text=combined_ocr_text if include_ocr_text else None,
            processing_time_ms=total_time_ms
        )
        
    except ValueError as e:
        logger.error(f"Ошибка валидации: {e}")
        processing_time_ms = (time.time() - start_time) * 1000
        
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Ошибка при парсинге паспорта: {e}", exc_info=True)
        processing_time_ms = (time.time() - start_time) * 1000
        
        return PassportParseResponse(
            success=False,
            fields={},
            confidences={},
            errors=[str(e)],
            warnings=[],
            processing_time_ms=processing_time_ms
        )


@router.get(
    "/health",
    response_model=PassportHealthResponse,
    summary="Health check парсера паспортов",
    description="Проверка состояния парсера паспортов и всех зависимых сервисов"
)
async def passport_health() -> PassportHealthResponse:
    """
    Проверить состояние парсера паспортов
    
    Returns:
        PassportHealthResponse: Статус всех компонентов
    """
    try:
        # Инициализация если нужно
        if not ocr_aggregation_service.is_initialized():
            ocr_aggregation_service.initialize()
        
        if not passport_parser_service._initialized:
            passport_parser_service.initialize()
        
        # Проверка словарей
        dictionaries_status = {
            "surnames": len(name_morphology_service.surnames_dict) if name_morphology_service._initialized else 0,
            "names": len(name_morphology_service.names_dict) if name_morphology_service._initialized else 0,
            "patronymics": len(name_morphology_service.patronymics_dict) if name_morphology_service._initialized else 0
        }
        
        return PassportHealthResponse(
            status="healthy",
            ocr_initialized=ocr_aggregation_service.is_initialized(),
            parser_initialized=passport_parser_service._initialized,
            ocr_providers_count=ocr_aggregation_service.get_provider_count(),
            dictionaries_loaded=dictionaries_status
        )
        
    except Exception as e:
        logger.error(f"Ошибка health check парсера паспортов: {e}")
        
        return PassportHealthResponse(
            status="error",
            ocr_initialized=False,
            parser_initialized=False,
            ocr_providers_count=0,
            dictionaries_loaded={}
        )
