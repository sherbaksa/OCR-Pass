"""
Preprocessing Service
Сервис предобработки изображений для улучшения качества OCR.
Включает методы для удаления шума, нормализации контраста, бинаризации и т.д.
"""

import cv2
import numpy as np
from typing import List, Tuple, Union
from io import BytesIO
from PIL import Image

from backend.core.logger import logger


class PreprocessingService:
    """
    Сервис предобработки изображений для OCR.
    Предоставляет методы для улучшения качества изображений перед распознаванием.
    """
    
    def __init__(self):
        """Инициализация сервиса предобработки"""
        logger.info("Preprocessing Service initialized")
    
    def preprocess_image(
        self,
        image_data: Union[bytes, np.ndarray],
        apply_deskew: bool = True,
        apply_denoise: bool = True,
        apply_contrast: bool = True,
        apply_sharpening: bool = True,
        apply_binarization: bool = False,
        apply_border_removal: bool = False,
        resize_width: int = None
    ) -> bytes:
        """
        Применить полный пайплайн предобработки к изображению.
        
        Args:
            image_data: Изображение в виде bytes или numpy array
            apply_deskew: Исправить наклон текста
            apply_denoise: Применить шумоподавление
            apply_contrast: Нормализовать контраст
            apply_sharpening: Повысить резкость
            apply_binarization: Применить бинаризацию
            apply_border_removal: Удалить рамки
            resize_width: Ширина для изменения размера (опционально)
            
        Returns:
            bytes: Обработанное изображение в формате JPEG
        """
        logger.info("Начало предобработки изображения")
        
        # Конвертация в numpy array
        image = self._bytes_to_image(image_data)
        original_shape = image.shape
        
        steps_applied = []
        
        # 1. Удаление рамок (если нужно)
        if apply_border_removal:
            image = self.remove_borders(image)
            steps_applied.append("border_removal")
            logger.debug("Применено: удаление рамок")
        
        # 2. Изменение размера (если нужно)
        if resize_width:
            image = self.resize_for_ocr(image, target_width=resize_width)
            steps_applied.append(f"resize_to_{resize_width}")
            logger.debug(f"Применено: изменение размера до ширины {resize_width}")
        
        # 3. Исправление наклона
        if apply_deskew:
            image = self.deskew_image(image)
            steps_applied.append("deskew")
            logger.debug("Применено: исправление наклона")
        
        # 4. Шумоподавление
        if apply_denoise:
            image = self.denoise_image(image, strength="medium")
            steps_applied.append("denoise")
            logger.debug("Применено: шумоподавление")
        
        # 5. Нормализация контраста
        if apply_contrast:
            image = self.normalize_contrast(image, method="clahe")
            steps_applied.append("contrast_clahe")
            logger.debug("Применено: нормализация контраста (CLAHE)")
        
        # 6. Повышение резкости
        if apply_sharpening:
            image = self.sharpen_image(image, strength=1.0)
            steps_applied.append("sharpen")
            logger.debug("Применено: повышение резкости")
        
        # 7. Бинаризация (опционально)
        if apply_binarization:
            image = self.binarize_image(image, method="adaptive")
            steps_applied.append("binarization")
            logger.debug("Применено: бинаризация")
        
        logger.info(
            f"Предобработка завершена. "
            f"Размер: {original_shape} -> {image.shape}. "
            f"Применено шагов: {len(steps_applied)}"
        )
        
        # Конвертация обратно в bytes
        return self._image_to_bytes(image)
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Исправить наклон текста на изображении.
        
        Алгоритм:
        1. Конвертация в градации серого
        2. Бинаризация
        3. Вычисление угла наклона через преобразование Хафа
        4. Поворот изображения
        
        Args:
            image: Входное изображение
            
        Returns:
            np.ndarray: Изображение с исправленным наклоном
        """
        logger.debug("Начало исправления наклона изображения")
        
        # Конвертация в градации серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Бинаризация для лучшего определения линий
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Вычисление угла наклона
        angle = self._compute_skew_angle(binary)
        
        if abs(angle) < 0.5:
            logger.debug(f"Угол наклона {angle:.2f}° слишком мал, пропускаем поворот")
            return image
        
        logger.debug(f"Обнаружен угол наклона: {angle:.2f}°")
        
        # Поворот изображения
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        logger.debug("Наклон исправлен успешно")
        return rotated
    
    def _compute_skew_angle(self, binary_image: np.ndarray) -> float:
        """
        Вычислить угол наклона текста на бинарном изображении.
        
        Args:
            binary_image: Бинарное изображение
            
        Returns:
            float: Угол наклона в градусах
        """
        # Детекция линий через преобразование Хафа
        lines = cv2.HoughLinesP(
            binary_image,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            logger.debug("Линии не обнаружены, угол = 0")
            return 0.0
        
        # Вычисление углов всех линий
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        # Медианный угол (более устойчив к выбросам)
        median_angle = np.median(angles)
        
        return median_angle
    
    def denoise_image(self, image: np.ndarray, strength: str = "medium") -> np.ndarray:
        """
        Удалить шум с изображения.
        
        Использует Gaussian Blur с разными параметрами в зависимости от силы.
        
        Args:
            image: Входное изображение
            strength: Сила шумоподавления (weak, medium, strong)
            
        Returns:
            np.ndarray: Изображение без шума
        """
        logger.debug(f"Применение шумоподавления (strength={strength})")
        
        # Параметры для разных уровней силы
        strength_params = {
            "weak": (3, 3),      # kernel size
            "medium": (5, 5),
            "strong": (7, 7)
        }
        
        if strength not in strength_params:
            logger.warning(f"Неизвестная сила '{strength}', использую 'medium'")
            strength = "medium"
        
        kernel_size = strength_params[strength]
        
        # Применение Gaussian Blur
        denoised = cv2.GaussianBlur(
            image,
            kernel_size,
            0  # sigma вычисляется автоматически
        )
        
        logger.debug(f"Шумоподавление применено с kernel={kernel_size}")
        return denoised
    
    def normalize_contrast(self, image: np.ndarray, method: str = "clahe") -> np.ndarray:
        """
        Нормализовать контраст изображения.
        
        Методы:
        - clahe: Adaptive Histogram Equalization (рекомендуется)
        - histogram: Обычная эквализация гистограммы
        - auto: Автоматическая нормализация яркости
        
        Args:
            image: Входное изображение
            method: Метод нормализации
            
        Returns:
            np.ndarray: Изображение с нормализованным контрастом
        """
        logger.debug(f"Нормализация контраста методом '{method}'")
        
        # Конвертация в градации серого для обработки
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray = image
            is_color = False
        
        if method == "clahe":
            # CLAHE - лучший метод для документов
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )
            result = clahe.apply(gray)
            
        elif method == "histogram":
            # Простая эквализация гистограммы
            result = cv2.equalizeHist(gray)
            
        elif method == "auto":
            # Автоматическая нормализация
            result = cv2.normalize(
                gray,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX
            )
        else:
            logger.warning(f"Неизвестный метод '{method}', использую 'clahe'")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result = clahe.apply(gray)
        
        # Если исходное изображение цветное, конвертируем обратно
        if is_color:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        logger.debug("Контраст нормализован")
        return result
    
    def sharpen_image(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Повысить резкость изображения.
        
        Использует технику Unsharp Mask.
        
        Args:
            image: Входное изображение
            strength: Сила повышения резкости (0.0 - 2.0)
            
        Returns:
            np.ndarray: Изображение с повышенной резкостью
        """
        logger.debug(f"Повышение резкости (strength={strength})")
        
        # Размытие изображения
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        
        # Unsharp mask: original + strength * (original - blurred)
        sharpened = cv2.addWeighted(
            image,
            1.0 + strength,
            blurred,
            -strength,
            0
        )
        
        logger.debug("Резкость повышена")
        return sharpened
    
    def binarize_image(self, image: np.ndarray, method: str = "adaptive") -> np.ndarray:
        """
        Бинаризировать изображение (черно-белое).
        
        Методы:
        - adaptive: Адаптивная пороговая обработка (рекомендуется)
        - otsu: Метод Оцу
        - simple: Простая пороговая обработка
        
        Args:
            image: Входное изображение
            method: Метод бинаризации
            
        Returns:
            np.ndarray: Бинарное изображение
        """
        logger.debug(f"Бинаризация методом '{method}'")
        
        # Конвертация в градации серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == "adaptive":
            # Адаптивная пороговая обработка - лучше для неравномерного освещения
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )
            
        elif method == "otsu":
            # Метод Оцу - автоматический выбор порога
            _, binary = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
        elif method == "simple":
            # Простая пороговая обработка с фиксированным порогом
            _, binary = cv2.threshold(
                gray,
                127,
                255,
                cv2.THRESH_BINARY
            )
        else:
            logger.warning(f"Неизвестный метод '{method}', использую 'adaptive'")
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
        
        logger.debug("Бинаризация завершена")
        return binary
    
    def remove_borders(self, image: np.ndarray, threshold: int = 10) -> np.ndarray:
        """
        Удалить черные рамки по краям изображения.
        
        Args:
            image: Входное изображение
            threshold: Порог для определения рамки (0-255)
            
        Returns:
            np.ndarray: Изображение без рамок
        """
        logger.debug("Удаление рамок")
        
        # Конвертация в градации серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Поиск ненулевых пикселей
        coords = cv2.findNonZero((gray > threshold).astype(np.uint8))
        
        if coords is None:
            logger.warning("Не удалось найти контент для обрезки рамок")
            return image
        
        # Получение bounding box
        x, y, w, h = cv2.boundingRect(coords)
        
        # Обрезка
        cropped = image[y:y+h, x:x+w]
        
        logger.debug(f"Рамки удалены. Новый размер: {cropped.shape}")
        return cropped
    
    def resize_for_ocr(
        self,
        image: np.ndarray,
        target_width: int = 2000,
        maintain_aspect: bool = True
    ) -> np.ndarray:
        """
        Изменить размер изображения для оптимального OCR.
        
        OCR движки работают лучше всего с изображениями
        определенного размера (обычно 1500-2500px по ширине).
        
        Args:
            image: Входное изображение
            target_width: Целевая ширина
            maintain_aspect: Сохранить соотношение сторон
            
        Returns:
            np.ndarray: Изображение измененного размера
        """
        current_height, current_width = image.shape[:2]
        
        logger.debug(
            f"Изменение размера с {current_width}x{current_height} "
            f"до ширины {target_width}"
        )
        
        if current_width == target_width:
            logger.debug("Размер уже соответствует целевому")
            return image
        
        if maintain_aspect:
            # Вычисление новой высоты с сохранением пропорций
            aspect_ratio = current_height / current_width
            target_height = int(target_width * aspect_ratio)
        else:
            target_height = current_height
        
        # Выбор метода интерполяции
        if current_width > target_width:
            # Уменьшение - используем INTER_AREA
            interpolation = cv2.INTER_AREA
        else:
            # Увеличение - используем INTER_CUBIC
            interpolation = cv2.INTER_CUBIC
        
        resized = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=interpolation
        )
        
        logger.debug(f"Размер изменен на {target_width}x{target_height}")
        return resized
    
    # === НОВЫЕ МЕТОДЫ ДЛЯ УЛУЧШЕННОЙ ОБРАБОТКИ ПАСПОРТОВ ===
    
    def increase_brightness(self, image: np.ndarray, value: int = 30) -> np.ndarray:
        """
        Повысить яркость изображения для "высветления" водяных знаков.
        
        Args:
            image: Входное изображение
            value: Значение увеличения яркости (0-100, рекомендуется 20-40)
            
        Returns:
            np.ndarray: Изображение с повышенной яркостью
        """
        logger.debug(f"Повышение яркости на {value}")
        
        # Конвертация в HSV для работы с яркостью
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
        else:
            # Для grayscale используем напрямую
            v = image.copy()
            h, s = None, None
        
        # Увеличение яркости с защитой от переполнения
        v = cv2.add(v, np.array([value], dtype=np.uint8))
        
        # Собираем обратно
        if h is not None:
            hsv = cv2.merge([h, s, v])
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            result = v
        
        logger.debug("Яркость повышена")
        return result
    
    def enhance_contrast_for_dark_zones(
        self,
        image: np.ndarray,
        clip_limit: float = 3.0,
        tile_size: int = 8
    ) -> np.ndarray:
        """
        Повысить контрастность для темных зон изображения.
        Использует CLAHE с настройками для усиления темных областей.
        
        Для цветных изображений применяет CLAHE к L-каналу в LAB пространстве,
        сохраняя информацию о цвете (каналы A и B).
        
        Args:
            image: Входное изображение
            clip_limit: Предел отсечения для CLAHE (выше = сильнее контраст)
            tile_size: Размер сетки для адаптивной обработки
            
        Returns:
            np.ndarray: Изображение с усиленным контрастом в темных зонах
        """
        logger.debug(f"Усиление контраста для темных зон (clip_limit={clip_limit})")
        
        # Создание CLAHE объекта
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )
        
        # Обработка в зависимости от типа изображения
        if len(image.shape) == 3:
            # Цветное изображение - работаем в LAB пространстве
            # LAB: L - яркость, A и B - цветовые каналы
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Применяем CLAHE только к L-каналу (яркость)
            l_enhanced = clahe.apply(l)
            
            # Собираем обратно LAB изображение
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            
            # Конвертируем обратно в BGR
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale изображение
            enhanced = clahe.apply(image)
        
        logger.debug("Контраст для темных зон усилен")
        return enhanced
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Конвертировать изображение в градации серого.
        
        Args:
            image: Входное изображение (BGR или уже grayscale)
            
        Returns:
            np.ndarray: Изображение в градациях серого
        """
        logger.debug("Конвертация в градации серого")
        
        if len(image.shape) == 2:
            logger.debug("Изображение уже в градациях серого")
            return image
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.debug("Конвертация завершена")
        return gray
    
    def preprocess_for_passport(
        self,
        image_data: Union[bytes, np.ndarray],
        brightness_increase: int = 30,
        contrast_clip_limit: float = 3.0,
        apply_binarization: bool = True,
        binarization_method: str = "adaptive"
    ) -> bytes:
        """
        Специализированный пайплайн предобработки для паспортов РФ.
        
        Этапы:
        1. Повышение яркости (высветление водяных знаков)
        2. Усиление контраста для темных зон (усиление текста)
        3. Конвертация в градации серого
        4. Бинаризация (опционально)
        
        Args:
            image_data: Изображение в виде bytes или numpy array
            brightness_increase: Значение увеличения яркости (0-100)
            contrast_clip_limit: Сила контраста для темных зон (1.0-5.0)
            apply_binarization: Применять ли бинаризацию
            binarization_method: Метод бинаризации (adaptive/otsu/simple)
            
        Returns:
            bytes: Обработанное изображение в формате JPEG
        """
        logger.info("=== НАЧАЛО СПЕЦИАЛИЗИРОВАННОЙ ОБРАБОТКИ ПАСПОРТА ===")
        
        # Конвертация в numpy array
        image = self._bytes_to_image(image_data)
        original_shape = image.shape
        logger.info(f"Исходный размер: {original_shape}")
        
        # Этап 1: Повышение яркости
        logger.info(f"Этап 1: Повышение яркости (+{brightness_increase})")
        image = self.increase_brightness(image, value=brightness_increase)
        
        # Этап 2: Усиление контраста для темных зон
        logger.info(f"Этап 2: Усиление контраста (clip_limit={contrast_clip_limit})")
        image = self.enhance_contrast_for_dark_zones(
            image,
            clip_limit=contrast_clip_limit,
            tile_size=8
        )
        
        # Этап 3: Конвертация в градации серого
        logger.info("Этап 3: Конвертация в градации серого")
        image = self.convert_to_grayscale(image)
        
        # Этап 4: Бинаризация (если нужно)
        if apply_binarization:
            logger.info(f"Этап 4: Бинаризация (метод={binarization_method})")
            image = self.binarize_image(image, method=binarization_method)
        else:
            logger.info("Этап 4: Бинаризация пропущена")
        
        logger.info(f"=== ОБРАБОТКА ЗАВЕРШЕНА === Финальный размер: {image.shape}")
        
        # Конвертация обратно в bytes
        return self._image_to_bytes(image)
    
    # === СЛУЖЕБНЫЕ МЕТОДЫ ===
    
    @staticmethod
    def _bytes_to_image(image_bytes: bytes) -> np.ndarray:
        """
        Конвертировать bytes в numpy array (OpenCV формат).
        
        Args:
            image_bytes: Изображение в виде bytes
            
        Returns:
            np.ndarray: Изображение в формате BGR (OpenCV)
        """
        if isinstance(image_bytes, np.ndarray):
            return image_bytes
        
        # Декодирование через OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            # Попытка через PIL если OpenCV не справился
            pil_image = Image.open(BytesIO(image_bytes))
            
            # Конвертация в RGB если нужно
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # PIL использует RGB, OpenCV использует BGR
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    @staticmethod
    def _image_to_bytes(image: np.ndarray, format: str = "JPEG", quality: int = 95) -> bytes:
        """
        Конвертировать numpy array в bytes.
        
        Args:
            image: Изображение в numpy array
            format: Формат кодирования (JPEG, PNG)
            quality: Качество JPEG (1-100)
            
        Returns:
            bytes: Изображение в виде bytes
        """
        # Определение расширения
        if format.upper() == "JPEG":
            ext = ".jpg"
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif format.upper() == "PNG":
            ext = ".png"
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        else:
            ext = ".jpg"
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        
        # Кодирование
        success, encoded_image = cv2.imencode(ext, image, encode_params)
        
        if not success:
            raise RuntimeError("Не удалось закодировать изображение")
        
        return encoded_image.tobytes()


# Singleton instance
preprocessing_service = PreprocessingService()
