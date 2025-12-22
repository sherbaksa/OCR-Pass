"""
API endpoint для интеллектуального парсинга паспортов РФ
Использует OCR + парсинг МЧЗ + интеллектуальный парсинг с нормализацией, морфологией и словарями.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional
import time

from backend.services.ocr_aggregation_service import ocr_aggregation_service
from backend.services.passport_parser_service import passport_parser_service
from backend.services.passport_mrz_parser_service import passport_mrz_parser_service
from backend.services.name_morphology_service import name_morphology_service
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
    raw_ocr_data: Optional[dict] = Field(None, description="Полные данные OCR от модели (для отладки)")
    debug_image_url: Optional[str] = Field(None, description="URL визуализации в MinIO (для отладки)")
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "fields": {
                    "series": "0520",
                    "number": "794472",
                    "surname": "Шербак",
                    "name": "Сергей",
                    "patronymic": "Алексеевич",
                    "birth_date": "01.01.1976",
                    "issue_date": "14.01.2021",
                    "department_code": "250-036",
                    "gender": "М",
                    "birth_place": "С.КРЕНОВО МИХАЙЛОВСКИЙ Р-ОН ПРИМОРСКИЙ КРАЙ",
                    "issued_by": "ПО ПРИМОРСКОМУ КРАЮ"
                },
                "confidences": {
                    "series": 0.98,
                    "number": 0.98,
                    "surname": 0.95,
                    "name": 0.95,
                    "patronymic": 0.95,
                    "birth_date": 0.98,
                    "issue_date": 0.98,
                    "department_code": 0.98,
                    "gender": 0.98
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
    Полный цикл обработки паспорта: OCR + парсинг МЧЗ + интеллектуальный парсинг.

    **Процесс обработки:**
    1. **OCR-распознавание** через PaddleOCR, EasyOCR, Google Vision
    2. **Парсинг МЧЗ** (машиночитаемой зоны) - приоритетный источник данных
    3. **Извлечение дополнительных полей** (место рождения, кем выдан)
    4. **Нормализация и валидация** всех данных

    **Извлекаемые поля (первая страница):**
    - Серия и номер паспорта (из МЧЗ)
    - ФИО (из МЧЗ)
    - Дата рождения (из МЧЗ)
    - Дата выдачи (из МЧЗ)
    - Код подразделения (из МЧЗ)
    - Пол (из МЧЗ)
    - Место рождения (из OCR текста)
    - Кем выдан (из OCR текста)

    **Требования к файлу:**
    - Форматы: JPG, PNG, JPEG
    - Максимальный размер: 10 MB
    - Рекомендуемое разрешение: 1500x1000 или выше

    **Параметры:**
    - file: файл изображения паспорта
    - use_all_providers: использовать все OCR провайдеры (по умолчанию true)
    - skip_preprocessing: пропустить встроенный препроцессинг (по умолчанию false)
    - include_ocr_text: включить полный OCR-текст в ответ (по умолчанию false)
    - include_raw_ocr: включить сырые данные OCR (по умолчанию false)
    """
)
async def parse_passport(
    file: UploadFile = File(..., description="Файл изображения паспорта (JPG, PNG)"),
    use_all_providers: bool = Query(
        default=True,
        description="Использовать все доступные OCR провайдеры"
    ),
    skip_preprocessing: bool = Query(
        default=False,
        description="Пропустить встроенный препроцессинг (deskew, denoise, contrast, sharpening)"
    ),
    include_ocr_text: bool = Query(
        default=False,
        description="Включить полный OCR-текст в ответ"
    ),
    include_raw_ocr: bool = Query(
        default=False,
        description="Включить полные сырые данные OCR (для отладки)"
    )
) -> PassportParseResponse:
    """
    Парсинг паспорта РФ с полной обработкой

    Args:
        file: Загруженный файл изображения
        use_all_providers: Использовать все OCR провайдеры
        skip_preprocessing: Пропустить препроцессинг
        include_ocr_text: Включить OCR-текст в ответ
        include_raw_ocr: Включить сырые данные OCR

    Returns:
        PassportParseResponse: Извлеченные поля паспорта
    """
    start_time = time.time()

    logger.info(
        f"Получен запрос на парсинг паспорта. "
        f"Файл: {file.filename}, "
        f"Размер: {file.size if hasattr(file, 'size') else 'unknown'}, "
        f"skip_preprocessing: {skip_preprocessing}"
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
            use_all_providers=use_all_providers,
            skip_preprocessing=skip_preprocessing
        )

        ocr_time = (time.time() - ocr_start) * 1000
        logger.info(f"OCR завершен за {ocr_time:.2f}ms")

        # Получаем raw_ocr_data из PaddleOCR провайдера
        raw_ocr = None
        rec_texts = []
        
        if ocr_result.provider_extractions and len(ocr_result.provider_extractions) > 0:
            from backend.providers import paddleocr_provider
            if paddleocr_provider._last_raw_result:
                # Конвертируем результат в словарь
                raw_ocr = paddleocr_provider._convert_result_to_dict(paddleocr_provider._last_raw_result)
                if raw_ocr:
                    rec_texts = raw_ocr.get('rec_texts', [])
                    logger.info(f"Получено {len(rec_texts)} распознанных текстов от PaddleOCR")
            else:
                logger.warning("paddleocr_provider._last_raw_result пустой")

        # Шаг 2: Парсинг МЧЗ (приоритетный источник данных)
        logger.info("Шаг 2: Парсинг МЧЗ...")
        mrz_start = time.time()
        
        mrz_result = passport_mrz_parser_service.parse_from_ocr_texts(rec_texts)
        
        mrz_time = (time.time() - mrz_start) * 1000
        logger.info(f"Парсинг МЧЗ завершен за {mrz_time:.2f}ms. Извлечено полей: {len(mrz_result['fields'])}")

        # Шаг 3: Извлечение дополнительных полей из OCR текста
        logger.info("Шаг 3: Извлечение дополнительных полей...")
        parse_start = time.time()

        # Формируем полный текст из всех распознанных строк
        combined_text = "\n".join(rec_texts) if rec_texts else ""
        
        # Используем стандартный парсер для извлечения дополнительных полей
        # (место рождения, кем выдан)
        additional_result = passport_parser_service.parse_passport_text(combined_text)
        
        parse_time = (time.time() - parse_start) * 1000
        logger.info(f"Извлечение дополнительных полей завершено за {parse_time:.2f}ms")

        # Шаг 4: Объединение результатов
        final_fields = {}
        final_confidences = {}
        all_errors = []
        all_warnings = []

        # Приоритет 1: Данные из МЧЗ (самая высокая точность)
        final_fields.update(mrz_result['fields'])
        final_confidences.update(mrz_result['confidences'])
        all_errors.extend(mrz_result.get('errors', []))
        all_warnings.extend(mrz_result.get('warnings', []))

        # Приоритет 2: Дополнительные поля из стандартного парсера
        # (место рождения, кем выдан - их нет в МЧЗ)
        additional_fields = ['birth_place', 'issued_by']
        for field in additional_fields:
            if field in additional_result['fields'] and field not in final_fields:
                final_fields[field] = additional_result['fields'][field]
                final_confidences[field] = additional_result['confidences'].get(field, 0.7)

        # Добавляем ошибки и предупреждения из дополнительного парсинга
        all_errors.extend(additional_result.get('errors', []))
        all_warnings.extend(additional_result.get('warnings', []))

        # Приоритет 3: Данные из OCR aggregation (если что-то осталось)
        for field_name, value in ocr_result.final_values.items():
            if field_name not in final_fields and value:
                final_fields[field_name] = value
                final_confidences[field_name] = ocr_result.field_confidences.get(field_name, 0.5)

        total_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Парсинг паспорта завершен за {total_time_ms:.2f}ms. "
            f"Извлечено полей: {len(final_fields)}, "
            f"Ошибок: {len(all_errors)}, "
            f"Предупреждений: {len(all_warnings)}"
        )

        return PassportParseResponse(
            success=True,
            fields=final_fields,
            confidences=final_confidences,
            errors=all_errors,
            warnings=all_warnings,
            ocr_text=combined_text if include_ocr_text else None,
            raw_ocr_data=raw_ocr if include_raw_ocr else None,
            debug_image_url=None,
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
