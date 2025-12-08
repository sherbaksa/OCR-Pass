"""
Debug API endpoint для визуализации препроцессинга изображений
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional
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
            corrected_image_data,  # ← Используем исправленное
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
            preprocessed_image_path=output_path,
            download_url=f"computer://{output_path}",
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
