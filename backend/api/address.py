"""
API endpoint для парсинга адреса регистрации и синхронизации ФИО паспорта РФ.
"""

from fastapi import APIRouter, HTTPException
import time

from backend.services.address_parser_service import address_parser_service
from backend.schemas.address import (
    AddressParseRequest,
    AddressParseResponse,
    NameSyncRequest,
    NameSyncResponse,
    StructuredAddress,
)
from backend.core.logger import logger


router = APIRouter(prefix="/passport", tags=["Passport Address"])


@router.post(
    "/address/parse",
    response_model=AddressParseResponse,
    summary="Парсинг адреса регистрации",
    description="""
    Структурирование адреса регистрации из паспорта РФ на компоненты.

    **Извлекаемые компоненты:**
    - Почтовый индекс (6 цифр)
    - Регион (область, край, республика)
    - Район
    - Город
    - Населенный пункт (поселок, деревня)
    - Улица (проспект, переулок, бульвар)
    - Номер дома
    - Корпус / строение
    - Номер квартиры

    **Процесс обработки:**
    1. Нормализация адреса (очистка, приведение к стандартному виду)
    2. Извлечение компонентов через regex-паттерны
    3. Валидация полноты адреса (обязательные компоненты)
    4. Расчет уверенности для каждого компонента
    5. Форматирование в юридически корректный вид

    **Юридически корректный адрес должен содержать:**
    - Регион
    - Город или населенный пункт
    - Улица
    - Номер дома
    """
)
async def parse_address(request: AddressParseRequest) -> AddressParseResponse:
    """
    Распарсить адрес регистрации на структурированные компоненты.

    Args:
        request: Запрос с исходным адресом

    Returns:
        AddressParseResponse: Структурированный адрес с компонентами
    """
    start_time = time.time()

    logger.info(f"Получен запрос на парсинг адреса ({len(request.raw_address)} символов)")

    try:
        # Парсинг адреса
        structured_address = address_parser_service.parse_address(
            raw_address=request.raw_address,
            extract_postal_code=request.extract_postal_code,
            validate_completeness=request.validate_completeness
        )

        # Форматирование адреса
        formatted_address = structured_address.to_formatted_string(
            include_postal_code=request.extract_postal_code
        )

        # Проверка ошибок
        errors = []
        if request.validate_completeness:
            missing = structured_address.validate_completeness()
            if missing:
                errors.append(
                    f"Адрес неполный. Отсутствуют обязательные компоненты: {', '.join(missing)}"
                )

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Парсинг адреса завершен за {processing_time_ms:.2f}ms. "
            f"Компонентов: {len(structured_address.component_confidences)}, "
            f"Средняя уверенность: {structured_address.get_confidence():.2f}"
        )

        return AddressParseResponse(
            success=True,
            address=structured_address,
            formatted_address=formatted_address,
            errors=errors,
            warnings=structured_address.parsing_warnings,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.error(f"Ошибка при парсинге адреса: {e}", exc_info=True)
        processing_time_ms = (time.time() - start_time) * 1000

        return AddressParseResponse(
            success=False,
            address=None,
            formatted_address=None,
            errors=[str(e)],
            warnings=[],
            processing_time_ms=processing_time_ms
        )


@router.post(
    "/address/sync-names",
    response_model=NameSyncResponse,
    summary="Синхронизация ФИО между страницами",
    description="""
    Сравнение ФИО со страницы 1 (основные данные) и страницы 2 (адрес регистрации).

    **Процесс проверки:**
    1. Нормализация ФИО (регистр, пробелы)
    2. Точное сравнение каждого компонента (фамилия, имя, отчество)
    3. Вычисление схожести через расстояние Левенштейна
    4. Определение общей схожести (взвешенное среднее)
    5. Обнаружение расхождений

    **Критерии совпадения:**
    - Точное совпадение: 100% схожесть по всем компонентам
    - Приемлемое совпадение: схожесть >= порога (по умолчанию 0.85)
    - Расхождение: схожесть < порога

    **Используется для:**
    - Проверки корректности распознавания
    - Обнаружения опечаток OCR
    - Валидации данных паспорта
    """
)
async def sync_names(request: NameSyncRequest) -> NameSyncResponse:
    """
    Синхронизация и сравнение ФИО между страницами паспорта.

    Args:
        request: Запрос с ФИО из двух страниц

    Returns:
        NameSyncResponse: Результат сравнения с метриками схожести
    """
    start_time = time.time()

    logger.info("Получен запрос на синхронизацию ФИО между страницами")

    try:
        # Проверка наличия хотя бы одного компонента ФИО
        page1_has_data = any([
            request.page1_surname,
            request.page1_name,
            request.page1_patronymic
        ])

        page2_has_data = any([
            request.page2_surname,
            request.page2_name,
            request.page2_patronymic
        ])

        if not page1_has_data and not page2_has_data:
            raise ValueError("Необходимо предоставить ФИО хотя бы с одной страницы")

        # Сравнение ФИО
        comparison = address_parser_service.compare_names(
            page1_surname=request.page1_surname,
            page1_name=request.page1_name,
            page1_patronymic=request.page1_patronymic,
            page2_surname=request.page2_surname,
            page2_name=request.page2_name,
            page2_patronymic=request.page2_patronymic,
            threshold=request.similarity_threshold
        )

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Синхронизация ФИО завершена за {processing_time_ms:.2f}ms. "
            f"Совпадение: {comparison.full_match}, "
            f"Схожесть: {comparison.overall_similarity:.2f}, "
            f"Расхождений: {len(comparison.discrepancies)}"
        )

        return NameSyncResponse(
            success=True,
            comparison=comparison,
            is_match=comparison.is_acceptable_match(request.similarity_threshold),
            errors=[],
            processing_time_ms=processing_time_ms
        )

    except ValueError as e:
        logger.error(f"Ошибка валидации при синхронизации ФИО: {e}")
        processing_time_ms = (time.time() - start_time) * 1000

        return NameSyncResponse(
            success=False,
            comparison=None,
            is_match=False,
            errors=[str(e)],
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.error(f"Ошибка при синхронизации ФИО: {e}", exc_info=True)
        processing_time_ms = (time.time() - start_time) * 1000

        return NameSyncResponse(
            success=False,
            comparison=None,
            is_match=False,
            errors=[str(e)],
            processing_time_ms=processing_time_ms
        )
