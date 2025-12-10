"""
Debug API endpoint для визуализации препроцессинга изображений
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional, Dict, List, Tuple
import time
import cv2
import os
from datetime import datetime
import numpy as np

from backend.services.preprocessing_service import preprocessing_service
from backend.core.logger import logger
from pydantic import BaseModel, Field


router = APIRouter(prefix="/debug", tags=["Debug"])


class PreprocessDebugResponse(BaseModel):
    """Ответ debug endpoint'а"""
    success: bool
    original_size: tuple
    preprocessed_size: tuple
    preprocessed_image_path: str
    download_url: str
    parameters_used: dict
    processing_time_ms: float


class PassportPreprocessDebugResponse(BaseModel):
    """Ответ endpoint'а для пошаговой обработки паспортов"""
    success: bool
    original_size: tuple
    steps: List[Dict]  # Список шагов с информацией о каждом
    download_urls: dict
    parameters_used: dict
    execution_order: List[str]  # Порядок выполнения этапов
    processing_time_ms: float


@router.post(
    "/preprocess",
    response_model=PreprocessDebugResponse,
    summary="Визуализация препроцессинга изображения",
    description="Применяет препроцессинг и возвращает ссылку на результат для визуальной проверки"
)
async def debug_preprocess(
    file: UploadFile = File(..., description="Изображение для препроцессинга"),
    apply_deskew: bool = Query(default=True, description="Выравнивание наклона"),
    apply_denoise: bool = Query(default=True, description="Шумоподавление"),
    apply_contrast: bool = Query(default=True, description="Улучшение контраста"),
    apply_sharpening: bool = Query(default=True, description="Повышение резкости"),
    apply_binarization: bool = Query(default=False, description="Бинаризация"),
) -> PreprocessDebugResponse:
    """
    Debug endpoint для визуализации результатов препроцессинга
    """
    start_time = time.time()
    
    try:

        # Чтение файла
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Пустой файл")
        
        # Декодирование с сохранением ориентации через PIL
        from PIL import Image, ImageOps
        import io
        
        original_pil = Image.open(io.BytesIO(image_data))
        # Автоматически корректируем ориентацию на основе EXIF
        original_pil = ImageOps.exif_transpose(original_pil)
        
        # Конвертируем в numpy для preprocessing_service
        original_img = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
        original_size = original_img.shape[:2]
        
        # Конвертируем обратно в bytes для preprocessing
        _, buffer = cv2.imencode('.jpg', original_img)
        corrected_image_data = buffer.tobytes()
        
        logger.info(f"Препроцессинг изображения: {original_size}")
        
        # Применение препроцессинга (используем скорректированное изображение)
        preprocessed_bytes = preprocessing_service.preprocess_image(
            corrected_image_data,
            apply_deskew=apply_deskew,
            apply_denoise=apply_denoise,
            apply_contrast=apply_contrast,
            apply_sharpening=apply_sharpening,
            apply_binarization=apply_binarization
        )

        
        # Декодируем preprocessed bytes обратно в numpy array для получения размера
        preprocessed_nparr = np.frombuffer(preprocessed_bytes, np.uint8)
        preprocessed_img = cv2.imdecode(preprocessed_nparr, cv2.IMREAD_COLOR)
        preprocessed_size = preprocessed_img.shape[:2]
        
        # Сохранение результата
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_preprocessed_{timestamp}.jpg"
        output_path = f"/mnt/user-data/outputs/{filename}"
        
        # Создаем директорию если нет
        os.makedirs("/mnt/user-data/outputs", exist_ok=True)
        
        # Сохраняем напрямую bytes
        with open(output_path, 'wb') as f:
            f.write(preprocessed_bytes)        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Препроцессированное изображение сохранено: {output_path}")
        
        return PreprocessDebugResponse(
            success=True,
            original_size=original_size,
            preprocessed_size=preprocessed_size,
            preprocessed_image_path=filename,
            download_url=f"/api/v1/debug/download/{filename}",
            parameters_used={
                "deskew": apply_deskew,
                "denoise": apply_denoise,
                "contrast": apply_contrast,
                "sharpening": apply_sharpening,
                "binarization": apply_binarization
            },
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Ошибка препроцессинга: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/preprocess-passport-steps",
    response_model=PassportPreprocessDebugResponse,
    summary="Пошаговая обработка паспорта с настраиваемым порядком этапов",
    description="""
    Применяет специализированный пайплайн обработки паспортов с динамическим порядком этапов.
    
    Доступные этапы:
    1. Повышение яркости (brightness) - высветление водяных знаков
    2. Усиление контраста (contrast) - усиление текста в темных зонах
    3. Конвертация в градации серого (grayscale)
    4. Бинаризация (binarization) - опционально
    
    Порядок выполнения определяется весами (weight):
    - Чем ВЫШЕ вес (1-100), тем РАНЬШЕ выполняется этап
    - Дефолтные веса: brightness=10, contrast=20, grayscale=30, binarization=40
    - При одинаковых весах используется алфавитный порядок
    
    Примеры:
    - brightness_weight=50 → яркость выполнится последней
    - contrast_weight=5 → контраст выполнится первым
    
    Возвращает результат каждого этапа для визуального анализа.
    """
)
async def debug_passport_preprocess_steps(
    file: UploadFile = File(..., description="Изображение паспорта"),
    brightness_increase: int = Query(default=30, ge=0, le=100, description="Увеличение яркости (0-100)"),
    brightness_weight: int = Query(default=10, ge=1, le=100, description="Вес этапа яркости (1-100, выше=раньше)"),
    contrast_clip_limit: float = Query(default=3.0, ge=1.0, le=5.0, description="Сила контраста (1.0-5.0)"),
    contrast_weight: int = Query(default=20, ge=1, le=100, description="Вес этапа контраста (1-100, выше=раньше)"),
    grayscale_weight: int = Query(default=30, ge=1, le=100, description="Вес этапа grayscale (1-100, выше=раньше)"),
    apply_binarization: bool = Query(default=True, description="Применять бинаризацию"),
    binarization_method: str = Query(default="adaptive", description="Метод бинаризации (adaptive/otsu/simple)"),
    binarization_weight: int = Query(default=40, ge=1, le=100, description="Вес этапа бинаризации (1-100, выше=раньше)")
) -> PassportPreprocessDebugResponse:
    """
    Пошаговая обработка паспорта с динамическим порядком этапов на основе весов
    """
    start_time = time.time()
    
    try:
        # Чтение файла
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Пустой файл")
        
        # Декодирование и коррекция ориентации
        from PIL import Image, ImageOps
        import io
        
        original_pil = Image.open(io.BytesIO(image_data))
        original_pil = ImageOps.exif_transpose(original_pil)
        
        # Конвертация в numpy для обработки
        original_img = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
        original_size = original_img.shape[:2]
        
        logger.info(f"=== НАЧАЛО ПОШАГОВОЙ ОБРАБОТКИ ПАСПОРТА === Размер: {original_size}")
        
        # Создаем директорию для выходных файлов
        os.makedirs("/mnt/user-data/outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Определяем порядок выполнения этапов на основе весов
        steps_config = [
            ("brightness", brightness_weight, brightness_increase),
            ("contrast", contrast_weight, contrast_clip_limit),
            ("grayscale", grayscale_weight, None),
        ]
        
        if apply_binarization:
            steps_config.append(("binarization", binarization_weight, binarization_method))
        
        # Сортировка: сначала по весу (DESC), потом по имени (ASC) при равных весах
        steps_config.sort(key=lambda x: (-x[1], x[0]))
        
        execution_order = [step[0] for step in steps_config]
        logger.info(f"Порядок выполнения этапов: {execution_order}")
        
        # Выполнение этапов в определенном порядке
        current_img = original_img.copy()
        steps_results = []
        download_urls = {}
        
        for step_num, (step_name, step_weight, step_param) in enumerate(steps_config, 1):
            logger.info(f"Этап {step_num}/{len(steps_config)}: {step_name} (вес={step_weight})")
            
            # Выполнение преобразования
            if step_name == "brightness":
                current_img = preprocessing_service.increase_brightness(current_img, value=step_param)
                step_description = f"Яркость +{step_param}"
                
            elif step_name == "contrast":
                current_img = preprocessing_service.enhance_contrast_for_dark_zones(
                    current_img,
                    clip_limit=step_param,
                    tile_size=8
                )
                step_description = f"Контраст (clip={step_param})"
                
            elif step_name == "grayscale":
                current_img = preprocessing_service.convert_to_grayscale(current_img)
                step_description = "Градации серого"
                
            elif step_name == "binarization":
                current_img = preprocessing_service.binarize_image(current_img, method=step_param)
                step_description = f"Бинаризация ({step_param})"
            
            # Сохранение результата этапа
            step_filename = f"passport_step{step_num}_{step_name}_{timestamp}.jpg"
            step_path = f"/mnt/user-data/outputs/{step_filename}"
            cv2.imwrite(step_path, current_img)
            
            # Сохранение информации об этапе
            step_info = {
                "step_number": step_num,
                "step_name": step_name,
                "step_weight": step_weight,
                "description": step_description,
                "filename": step_filename
            }
            steps_results.append(step_info)
            
            download_urls[f"step{step_num}_{step_name}"] = f"/api/v1/debug/download/{step_filename}"
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"=== ПОШАГОВАЯ ОБРАБОТКА ЗАВЕРШЕНА === Время: {processing_time:.2f}ms")
        
        return PassportPreprocessDebugResponse(
            success=True,
            original_size=original_size,
            steps=steps_results,
            download_urls=download_urls,
            execution_order=execution_order,
            parameters_used={
                "brightness_increase": brightness_increase,
                "brightness_weight": brightness_weight,
                "contrast_clip_limit": contrast_clip_limit,
                "contrast_weight": contrast_weight,
                "grayscale_weight": grayscale_weight,
                "apply_binarization": apply_binarization,
                "binarization_method": binarization_method if apply_binarization else None,
                "binarization_weight": binarization_weight if apply_binarization else None
            },
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Ошибка пошаговой обработки паспорта: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


from fastapi.responses import FileResponse


@router.get(
    "/download/{filename}",
    summary="Скачать препроцессированное изображение",
    response_class=FileResponse
)
async def download_preprocessed(filename: str):
    """Скачать файл по имени"""
    filepath = f"/mnt/user-data/outputs/{filename}"
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    return FileResponse(
        path=filepath,
        media_type="image/jpeg",
        filename=filename
    )
@router.post(
    "/test-preprocessing-ocr",
    summary="Тестирование влияния препроцессинга на качество OCR",
    description="""
    Endpoint для тестирования различных комбинаций препроцессинга и оценки их влияния на качество OCR.
    
    Процесс:
    1. Применяет препроцессинг с заданными параметрами и порядком (веса)
    2. Запускает OCR распознавание
    3. Извлекает поля паспорта
    4. Возвращает результаты распознавания + метрики качества
    
    Позволяет найти оптимальную комбинацию параметров препроцессинга для лучшего распознавания.
    """
)
async def test_preprocessing_ocr(
    file: UploadFile = File(..., description="Изображение паспорта"),
    # Параметры препроцессинга
    brightness_increase: int = Query(default=30, ge=0, le=100, description="Увеличение яркости (0-100)"),
    brightness_weight: int = Query(default=10, ge=1, le=100, description="Вес яркости"),
    contrast_clip_limit: float = Query(default=3.0, ge=1.0, le=5.0, description="Сила контраста"),
    contrast_weight: int = Query(default=20, ge=1, le=100, description="Вес контраста"),
    grayscale_weight: int = Query(default=30, ge=1, le=100, description="Вес grayscale"),
    apply_binarization: bool = Query(default=True, description="Применять бинаризацию"),
    binarization_method: str = Query(default="adaptive", description="Метод бинаризации"),
    binarization_weight: int = Query(default=40, ge=1, le=100, description="Вес бинаризации"),
    # Параметры OCR
    use_all_providers: bool = Query(default=True, description="Использовать все OCR провайдеры")
):
    """
    Тестирование препроцессинга с OCR распознаванием
    """
    from backend.services.ocr_aggregation_service import ocr_aggregation_service
    from backend.services.passport_parser_service import passport_parser_service
    
    start_time = time.time()
    
    try:
        # Чтение файла
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Пустой файл")
        
        logger.info("=== ТЕСТ ПРЕПРОЦЕССИНГА + OCR ===")
        
        # Декодирование и коррекция ориентации
        from PIL import Image, ImageOps
        import io
        
        original_pil = Image.open(io.BytesIO(image_data))
        original_pil = ImageOps.exif_transpose(original_pil)
        original_img = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
        
        # Определяем порядок выполнения этапов
        steps_config = [
            ("brightness", brightness_weight, brightness_increase),
            ("contrast", contrast_weight, contrast_clip_limit),
            ("grayscale", grayscale_weight, None),
        ]
        
        if apply_binarization:
            steps_config.append(("binarization", binarization_weight, binarization_method))
        
        steps_config.sort(key=lambda x: (-x[1], x[0]))
        execution_order = [step[0] for step in steps_config]
        
        logger.info(f"Порядок препроцессинга: {execution_order}")
        
        # Применяем препроцессинг
        preprocessed_img = original_img.copy()
        
        for step_name, step_weight, step_param in steps_config:
            if step_name == "brightness":
                preprocessed_img = preprocessing_service.increase_brightness(preprocessed_img, value=step_param)
            elif step_name == "contrast":
                preprocessed_img = preprocessing_service.enhance_contrast_for_dark_zones(
                    preprocessed_img, clip_limit=step_param, tile_size=8
                )
            elif step_name == "grayscale":
                preprocessed_img = preprocessing_service.convert_to_grayscale(preprocessed_img)
            elif step_name == "binarization":
                preprocessed_img = preprocessing_service.binarize_image(preprocessed_img, method=step_param)
        
        # Конвертируем обработанное изображение в bytes для OCR
        _, buffer = cv2.imencode('.jpg', preprocessed_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        preprocessed_bytes = buffer.tobytes()
        
        # Инициализация сервисов
        if not ocr_aggregation_service.is_initialized():
            ocr_aggregation_service.initialize()
        if not passport_parser_service._initialized:
            passport_parser_service.initialize()
        
        # OCR распознавание
        logger.info("Запуск OCR...")
        ocr_start = time.time()

        ocr_result = ocr_aggregation_service.process_image(
            image_data=preprocessed_bytes,
            language="ru",
            use_all_providers=use_all_providers,
            skip_preprocessing=True  # <- добавь эту строку
        )

        ocr_time = (time.time() - ocr_start) * 1000
        
        # Парсинг паспорта
        logger.info("Парсинг полей паспорта...")
        parse_start = time.time()
        
        # Получаем OCR текст из результата
        ocr_text = ""
        if ocr_result.provider_extractions:
            for extraction in ocr_result.provider_extractions:
                for field in extraction.fields:
                    if field.value:
                        ocr_text += field.value + " "
        
        # Парсим текст
        parsed_result = passport_parser_service.parse_passport_text(
            ocr_text=ocr_text,
            parse_address_structure=True
        )
        parse_time = (time.time() - parse_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        # Подсчет метрик качества
        fields_found = len([v for v in parsed_result.get("fields", {}).values() if v])
        total_fields = len(parsed_result.get("fields", {}))
        
        # Формируем ответ
        result = {
            "success": True,
            "preprocessing": {
                "execution_order": execution_order,
                "parameters": {
                    "brightness_increase": brightness_increase,
                    "brightness_weight": brightness_weight,
                    "contrast_clip_limit": contrast_clip_limit,
                    "contrast_weight": contrast_weight,
                    "grayscale_weight": grayscale_weight,
                    "apply_binarization": apply_binarization,
                    "binarization_method": binarization_method if apply_binarization else None,
                    "binarization_weight": binarization_weight if apply_binarization else None
                }
            },
            "ocr_results": {
                "providers_used": ocr_result.total_providers_used,
                "average_confidence": ocr_result.average_confidence,
                "fields_with_high_confidence": ocr_result.fields_with_high_confidence,
                "fields_with_low_confidence": ocr_result.fields_with_low_confidence
            },
            "parsed_fields": parsed_result.get("fields", {}),
            "field_confidences": parsed_result.get("confidences", {}),
            "quality_metrics": {
                "fields_found": fields_found,
                "total_fields": total_fields,
                "completion_rate": round(fields_found / total_fields * 100, 2) if total_fields > 0 else 0,
                "errors_count": len(parsed_result.get("errors", [])),
                "warnings_count": len(parsed_result.get("warnings", []))
            },
            "errors": parsed_result.get("errors", []),
            "warnings": parsed_result.get("warnings", []),
            "timing": {
                "ocr_time_ms": round(ocr_time, 2),
                "parsing_time_ms": round(parse_time, 2),
                "total_time_ms": round(total_time, 2)
            }
        }
        
        logger.info(f"=== ТЕСТ ЗАВЕРШЕН === Поля: {fields_found}/{total_fields}, Время: {total_time:.2f}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка теста препроцессинга+OCR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
