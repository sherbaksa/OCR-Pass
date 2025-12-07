"""
Сервис предобработки изображений паспортов с использованием OpenCV
Включает выравнивание, шумоподавление, нормализацию контраста и фильтры читаемости
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union
from PIL import Image
import io
import imutils
from skimage import exposure
from skimage.filters import threshold_local

from backend.core.logger import logger


class PreprocessingService:
    """Сервис для предобработки изображений паспортов перед OCR"""

    # Константы для обработки
    DEFAULT_DPI = 300
    MIN_CONTOUR_AREA = 1000
    ROTATION_ANGLE_THRESHOLD = 95  # Максимальный угол поворота для коррекции

    def __init__(self):
        """Инициализация сервиса предобработки"""
        logger.info("PreprocessingService инициализирован")

    def preprocess_image(
        self,
        image_data: Union[bytes, np.ndarray],
        apply_deskew: bool = True,
        apply_denoise: bool = True,
        apply_contrast: bool = True,
        apply_sharpening: bool = True,
        apply_binarization: bool = False
    ) -> bytes:
        """
        Комплексная предобработка изображения паспорта

        Args:
            image_data: Изображение в байтах или numpy array
            apply_deskew: Применять выравнивание наклона
            apply_denoise: Применять шумоподавление
            apply_contrast: Применять нормализацию контраста
            apply_sharpening: Применять усиление резкости
            apply_binarization: Применять бинаризацию (черно-белое)

        Returns:
            bytes: Обработанное изображение в формате JPEG

        Raises:
            ValueError: Если не удается декодировать изображение
        """
        try:
            # Конвертация входных данных в numpy array
            if isinstance(image_data, bytes):
                image = self._bytes_to_image(image_data)
            else:
                image = image_data

            logger.info(f"Начало предобработки изображения размером {image.shape}")

            # Применяем последовательность обработок
            if apply_deskew:
                image = self.deskew_image(image)

            if apply_denoise:
                image = self.denoise_image(image)

            if apply_contrast:
                image = self.normalize_contrast(image)

            if apply_sharpening:
                image = self.sharpen_image(image)

            if apply_binarization:
                image = self.binarize_image(image)

            # Конвертация обратно в байты
            result_bytes = self._image_to_bytes(image)

            logger.info("Предобработка изображения завершена успешно")
            return result_bytes

        except Exception as e:
            logger.error(f"Ошибка при предобработке изображения: {e}", exc_info=True)
            raise

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Выравнивание наклона изображения (deskew)
        Использует определение углов по проекционному профилю текста

        Args:
            image: Входное изображение (BGR или grayscale)

        Returns:
            np.ndarray: Выровненное изображение
        """
        try:
            # Конвертация в grayscale если цветное
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Инвертируем если фон темный
            if np.mean(gray) < 127:
                gray = cv2.bitwise_not(gray)

            # Бинаризация для определения текста
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Определение угла наклона
            angle = self._compute_skew_angle(thresh)

            # Если угол слишком большой, скорее всего ошибка определения
            if abs(angle) > self.ROTATION_ANGLE_THRESHOLD:
                logger.warning(f"Угол наклона {angle:.2f}° превышает порог, пропускаем поворот")
                return image

            # Поворачиваем изображение
            if abs(angle) > 0.5:  # Поворачиваем только если угол значительный
                rotated = imutils.rotate_bound(image, angle)
                logger.info(f"Изображение повернуто на {angle:.2f}°")
                return rotated
            else:
                logger.info(f"Угол наклона {angle:.2f}° незначителен, поворот не требуется")
                return image

        except Exception as e:
            logger.error(f"Ошибка при выравнивании изображения: {e}")
            return image  # Возвращаем исходное при ошибке

    def _compute_skew_angle(self, binary_image: np.ndarray) -> float:
        """
        Вычисление угла наклона текста методом проекционного профиля

        Args:
            binary_image: Бинаризованное изображение

        Returns:
            float: Угол наклона в градусах
        """
        # Находим координаты всех ненулевых пикселей
        coords = np.column_stack(np.where(binary_image > 0))

        # Если слишком мало точек, возвращаем 0
        if len(coords) < 100:
            return 0.0

        # Вычисляем угол через минимальный охватывающий прямоугольник
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # Корректировка угла (OpenCV возвращает угол от -90 до 0)
        if angle < -45:
            angle = 90 + angle
        
        return -angle

    def denoise_image(self, image: np.ndarray, strength: str = "medium") -> np.ndarray:
        """
        Шумоподавление с использованием различных методов фильтрации

        Args:
            image: Входное изображение
            strength: Сила шумоподавления ("light", "medium", "strong")

        Returns:
            np.ndarray: Изображение с удаленным шумом
        """
        try:
            # Параметры в зависимости от силы
            params = {
                "light": {"h": 5, "templateWindowSize": 7, "searchWindowSize": 21},
                "medium": {"h": 10, "templateWindowSize": 7, "searchWindowSize": 21},
                "strong": {"h": 15, "templateWindowSize": 9, "searchWindowSize": 25}
            }

            config = params.get(strength, params["medium"])

            # Применяем Non-local Means Denoising
            if len(image.shape) == 3:
                # Цветное изображение
                denoised = cv2.fastNlMeansDenoisingColored(
                    image,
                    None,
                    h=config["h"],
                    hColor=config["h"],
                    templateWindowSize=config["templateWindowSize"],
                    searchWindowSize=config["searchWindowSize"]
                )
            else:
                # Grayscale изображение
                denoised = cv2.fastNlMeansDenoising(
                    image,
                    None,
                    h=config["h"],
                    templateWindowSize=config["templateWindowSize"],
                    searchWindowSize=config["searchWindowSize"]
                )

            logger.info(f"Применено шумоподавление силой '{strength}'")
            return denoised

        except Exception as e:
            logger.error(f"Ошибка при шумоподавлении: {e}")
            return image

    def normalize_contrast(self, image: np.ndarray, method: str = "clahe") -> np.ndarray:
        """
        Нормализация контраста изображения

        Args:
            image: Входное изображение
            method: Метод нормализации ("clahe", "hist_eq", "adaptive")

        Returns:
            np.ndarray: Изображение с улучшенным контрастом
        """
        try:
            if method == "clahe":
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                if len(image.shape) == 3:
                    # Для цветных изображений применяем в LAB пространстве
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    
                    enhanced_lab = cv2.merge([l, a, b])
                    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                else:
                    # Для grayscale
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    result = clahe.apply(image)

            elif method == "hist_eq":
                # Простая выравнивание гистограммы
                if len(image.shape) == 3:
                    result = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    result[:, :, 0] = cv2.equalizeHist(result[:, :, 0])
                    result = cv2.cvtColor(result, cv2.COLOR_YUV2BGR)
                else:
                    result = cv2.equalizeHist(image)

            elif method == "adaptive":
                # Адаптивная нормализация с использованием scikit-image
                if len(image.shape) == 3:
                    # Конвертируем в grayscale для обработки
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    normalized = exposure.equalize_adapthist(gray, clip_limit=0.03)
                    result = (normalized * 255).astype(np.uint8)
                    # Конвертируем обратно в BGR
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                else:
                    normalized = exposure.equalize_adapthist(image, clip_limit=0.03)
                    result = (normalized * 255).astype(np.uint8)
            else:
                result = image

            logger.info(f"Применена нормализация контраста методом '{method}'")
            return result

        except Exception as e:
            logger.error(f"Ошибка при нормализации контраста: {e}")
            return image

    def sharpen_image(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Усиление резкости изображения для улучшения читаемости текста

        Args:
            image: Входное изображение
            strength: Сила усиления (0.5 - слабое, 1.0 - среднее, 1.5 - сильное)

        Returns:
            np.ndarray: Изображение с усиленной резкостью
        """
        try:
            # Ядро для усиления резкости (unsharp mask)
            kernel = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ]) * strength

            # Применяем фильтр
            sharpened = cv2.filter2D(image, -1, kernel)

            logger.info(f"Применено усиление резкости с силой {strength}")
            return sharpened

        except Exception as e:
            logger.error(f"Ошибка при усилении резкости: {e}")
            return image

    def binarize_image(self, image: np.ndarray, method: str = "adaptive") -> np.ndarray:
        """
        Бинаризация изображения (преобразование в черно-белое)

        Args:
            image: Входное изображение
            method: Метод бинаризации ("otsu", "adaptive", "local")

        Returns:
            np.ndarray: Бинаризованное изображение
        """
        try:
            # Конвертация в grayscale если цветное
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            if method == "otsu":
                # Метод Otsu
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            elif method == "adaptive":
                # Адаптивная бинаризация
                binary = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    blockSize=11,
                    C=2
                )

            elif method == "local":
                # Локальная бинаризация (scikit-image)
                local_thresh = threshold_local(gray, block_size=35, offset=10)
                binary = (gray > local_thresh).astype(np.uint8) * 255

            else:
                binary = gray

            # Морфологическая обработка для улучшения результата
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            logger.info(f"Применена бинаризация методом '{method}'")
            return binary

        except Exception as e:
            logger.error(f"Ошибка при бинаризации: {e}")
            return image

    def remove_borders(self, image: np.ndarray, threshold: int = 10) -> np.ndarray:
        """
        Удаление черных рамок по краям изображения

        Args:
            image: Входное изображение
            threshold: Порог для определения границы (яркость пикселей)

        Returns:
            np.ndarray: Изображение без рамок
        """
        try:
            # Конвертация в grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Находим ненулевые пиксели
            coords = cv2.findNonZero(cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1])

            if coords is not None:
                # Находим границы
                x, y, w, h = cv2.boundingRect(coords)
                
                # Обрезаем изображение
                cropped = image[y:y+h, x:x+w]
                
                logger.info(f"Удалены рамки: обрезка с ({x}, {y}) размером {w}x{h}")
                return cropped
            else:
                return image

        except Exception as e:
            logger.error(f"Ошибка при удалении рамок: {e}")
            return image

    def resize_for_ocr(
        self,
        image: np.ndarray,
        target_dpi: int = None,
        max_dimension: int = 3000
    ) -> np.ndarray:
        """
        Изменение размера изображения для оптимальной работы OCR

        Args:
            image: Входное изображение
            target_dpi: Целевое разрешение (если None, используется DEFAULT_DPI)
            max_dimension: Максимальный размер по длинной стороне

        Returns:
            np.ndarray: Изображение оптимального размера
        """
        try:
            if target_dpi is None:
                target_dpi = self.DEFAULT_DPI

            height, width = image.shape[:2]
            max_current = max(height, width)

            # Если изображение слишком большое, уменьшаем
            if max_current > max_dimension:
                scale = max_dimension / max_current
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.info(f"Изображение уменьшено до {new_width}x{new_height}")
                return resized

            # Если изображение слишком маленькое, увеличиваем
            elif max_current < 1000:
                scale = 1000 / max_current
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                logger.info(f"Изображение увеличено до {new_width}x{new_height}")
                return resized

            else:
                logger.info("Размер изображения оптимален, изменение не требуется")
                return image

        except Exception as e:
            logger.error(f"Ошибка при изменении размера: {e}")
            return image

    @staticmethod
    def _bytes_to_image(image_bytes: bytes) -> np.ndarray:
        """
        Конвертация байтов в numpy array (OpenCV формат)

        Args:
            image_bytes: Изображение в байтах

        Returns:
            np.ndarray: Изображение в формате OpenCV (BGR)

        Raises:
            ValueError: Если не удается декодировать изображение
        """
        try:
            # Используем PIL для загрузки
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Конвертация в RGB если необходимо
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Конвертация в numpy array
            image_array = np.array(pil_image)
            
            # Конвертация RGB -> BGR для OpenCV
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return image_bgr

        except Exception as e:
            raise ValueError(f"Не удалось декодировать изображение: {e}")

    @staticmethod
    def _image_to_bytes(image: np.ndarray, format: str = "JPEG", quality: int = 95) -> bytes:
        """
        Конвертация numpy array в байты

        Args:
            image: Изображение в формате OpenCV
            format: Формат выходного изображения (JPEG, PNG)
            quality: Качество JPEG (1-100)

        Returns:
            bytes: Изображение в байтах
        """
        try:
            # Конвертация BGR -> RGB для PIL
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Конвертация в PIL Image
            pil_image = Image.fromarray(image_rgb)

            # Сохранение в байты
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format, quality=quality)
            
            return buffer.getvalue()

        except Exception as e:
            raise ValueError(f"Не удалось конвертировать изображение в байты: {e}")


# Singleton instance
preprocessing_service = PreprocessingService()
