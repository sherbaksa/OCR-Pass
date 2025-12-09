"""
Text Normalization Service
Нормализация OCR-текста для улучшения качества парсинга паспортных данных.
"""

import re
from typing import Dict, List, Optional
from backend.core.logger import logger


class TextNormalizationService:
    """
    Сервис нормализации текста после OCR.
    Применяет различные правила для очистки и стандартизации текста.
    """
    
    def __init__(self):
        """Инициализация сервиса"""
        self._ocr_artifacts = self._build_ocr_artifacts_map()
        logger.info("Text Normalization Service инициализирован")
    
    def _build_ocr_artifacts_map(self) -> Dict[str, str]:
        """
        Создание карты замен для типичных артефактов OCR.
        
        Returns:
            Dict[str, str]: Карта замен символов
        """
        return {
            # Буква Ё -> Е (стандарт для паспортов РФ)
            'ё': 'е',
            'Ё': 'Е',
            
            # Типичные ошибки OCR: похожие символы
            '0': 'О',  # ноль -> О (в именах)
            '1': 'I',  # единица -> I (в редких случаях)
            '|': 'I',  # вертикальная черта -> I
            
            # Английские буквы -> русские (частая ошибка OCR)
            'A': 'А',
            'B': 'В', 
            'C': 'С',
            'E': 'Е',
            'H': 'Н',
            'K': 'К',
            'M': 'М',
            'O': 'О',
            'P': 'Р',
            'T': 'Т',
            'X': 'Х',
            'Y': 'У',
            
            # Нижний регистр
            'a': 'а',
            'c': 'с',
            'e': 'е',
            'o': 'о',
            'p': 'р',
            'x': 'х',
            'y': 'у',
        }
    
    def normalize_text(
        self,
        text: str,
        replace_yo: bool = True,
        fix_ocr_artifacts: bool = True,
        remove_extra_spaces: bool = True,
        remove_special_chars: bool = False
    ) -> str:
        """
        Полная нормализация текста.
        
        Args:
            text: Исходный текст
            replace_yo: Заменять Ё на Е
            fix_ocr_artifacts: Исправлять типичные ошибки OCR
            remove_extra_spaces: Удалять лишние пробелы
            remove_special_chars: Удалять спецсимволы
            
        Returns:
            str: Нормализованный текст
        """
        if not text:
            return ""
        
        result = text
        
        # 1. Замена Ё -> Е
        if replace_yo:
            result = result.replace('ё', 'е').replace('Ё', 'Е')
        
        # 2. Исправление артефактов OCR
        if fix_ocr_artifacts:
            for old_char, new_char in self._ocr_artifacts.items():
                if old_char in result:
                    result = result.replace(old_char, new_char)
        
        # 3. Удаление лишних пробелов
        if remove_extra_spaces:
            result = ' '.join(result.split())
        
        # 4. Удаление спецсимволов (опционально)
        if remove_special_chars:
            result = re.sub(r'[^\w\s\-\.\,]', '', result, flags=re.UNICODE)
        
        return result.strip()
    
    def normalize_fio_part(self, text: str) -> str:
        """
        Нормализация части ФИО (фамилия, имя, отчество).
        
        Args:
            text: Часть ФИО
            
        Returns:
            str: Нормализованная часть ФИО с правильным регистром
        """
        if not text:
            return ""
        
        # Базовая нормализация
        text = self.normalize_text(text, remove_special_chars=True)
        
        # Приведение к правильному регистру: первая буква заглавная, остальные строчные
        parts = text.split('-')  # Для двойных фамилий типа Петров-Водкин
        normalized_parts = []
        
        for part in parts:
            if part:
                # Первая буква заглавная, остальные строчные
                normalized_part = part[0].upper() + part[1:].lower() if len(part) > 0 else part
                normalized_parts.append(normalized_part)
        
        return '-'.join(normalized_parts)
    
    def normalize_passport_number(self, text: str) -> str:
        """
        Нормализация серии/номера паспорта.
        Оставляет только цифры.
        
        Args:
            text: Текст с серией/номером
            
        Returns:
            str: Только цифры
        """
        if not text:
            return ""
        
        # Удаляем все кроме цифр
        digits_only = re.sub(r'\D', '', text)
        return digits_only
    
    def normalize_department_code(self, text: str) -> str:
        """
        Нормализация кода подразделения.
        Формат: XXX-XXX
        
        Args:
            text: Текст с кодом
            
        Returns:
            str: Код в формате XXX-XXX
        """
        if not text:
            return ""
        
        # Удаляем все кроме цифр и дефисов
        code = re.sub(r'[^\d\-]', '', text)
        
        # Удаляем лишние дефисы
        code = re.sub(r'-+', '-', code)
        
        # Если дефиса нет, но есть 6 цифр - добавляем дефис
        digits_only = re.sub(r'\D', '', code)
        if len(digits_only) == 6 and '-' not in code:
            code = f"{digits_only[:3]}-{digits_only[3:]}"
        
        return code
    
    def normalize_date_string(self, text: str) -> str:
        """
        Предварительная нормализация строки даты.
        Унификация разделителей к точке.
        
        Args:
            text: Строка даты
            
        Returns:
            str: Нормализованная строка даты
        """
        if not text:
            return ""
        
        # Удаляем пробелы
        text = text.strip()
        
        # Заменяем различные разделители на точку
        text = re.sub(r'[\/\-]', '.', text)
        
        # Удаляем лишние точки
        text = re.sub(r'\.+', '.', text)
        
        return text
    
    def clean_ocr_text(self, text: str) -> str:
        """
        Агрессивная очистка OCR-текста от мусора.
        Используется для полей с адресами и длинными текстовыми строками.
        
        Args:
            text: OCR текст
            
        Returns:
            str: Очищенный текст
        """
        if not text:
            return ""
        
        # Базовая нормализация
        text = self.normalize_text(text)
        
        # Удаление множественных пробелов и переносов строк
        text = re.sub(r'\s+', ' ', text)
        
        # Удаление странных символов, которые могут появиться после OCR
        text = re.sub(r'[^\w\s\.\,\-\(\)№]', '', text, flags=re.UNICODE)
        
        return text.strip()
    
    def extract_cyrillic_only(self, text: str) -> str:
        """
        Извлечение только кириллических символов и пробелов.
        Полезно для имен и адресов.
        
        Args:
            text: Исходный текст
            
        Returns:
            str: Только кириллица и пробелы
        """
        if not text:
            return ""
        
        # Оставляем только кириллицу, пробелы и дефисы
        cyrillic_text = re.sub(r'[^а-яА-ЯёЁ\s\-]', '', text)
        
        # Нормализуем пробелы
        cyrillic_text = ' '.join(cyrillic_text.split())
        
        return cyrillic_text
    
    def extract_digits_only(self, text: str) -> str:
        """
        Извлечение только цифр.
        
        Args:
            text: Исходный текст
            
        Returns:
            str: Только цифры
        """
        if not text:
            return ""
        
        return re.sub(r'\D', '', text)
    
    def normalize_address(self, text: str) -> str:
        """
        Нормализация адреса регистрации.
        ВАЖНО: НЕ применяет замены цифр, чтобы сохранить номера домов и индексы.
        
        Args:
            text: Адрес
        Returns:
            str: Нормализованный адрес
        """
        if not text:
            return ""
        
        # Используем normalize_text БЕЗ замен цифр
        text = self.normalize_text(
            text,
            replace_yo=True,
            fix_ocr_artifacts=False,  # ОТКЛЮЧАЕМ замены цифр!
            remove_extra_spaces=True,
            remove_special_chars=False
        )
        
        # Замена распространенных сокращений на стандартные
        replacements = {
            r'\bг\.?\s*': 'г. ',
            r'\bул\.?\s*': 'ул. ',
            r'\bд\.?\s*': 'д. ',
            r'\bкв\.?\s*': 'кв. ',
            r'\bкорп\.?\s*': 'корп. ',
            r'\bстр\.?\s*': 'стр. ',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Финальная нормализация пробелов
        text = ' '.join(text.split())
        
        return text.strip()
    
    def calculate_cyrillic_ratio(self, text: str) -> float:
        """
        Вычисление доли кириллических символов в тексте.
        
        Args:
            text: Исходный текст
            
        Returns:
            float: Доля кириллицы (0.0 - 1.0)
        """
        if not text:
            return 0.0
        
        # Убираем пробелы для подсчета
        text_no_spaces = text.replace(' ', '')
        
        if len(text_no_spaces) == 0:
            return 0.0
        
        # Считаем кириллические символы
        cyrillic_count = len(re.findall(r'[а-яА-ЯёЁ]', text_no_spaces))
        
        return cyrillic_count / len(text_no_spaces)
    
    def is_valid_cyrillic_text(self, text: str, min_ratio: float = 0.8) -> bool:
        """
        Проверка, что текст преимущественно кириллический.
        
        Args:
            text: Текст для проверки
            min_ratio: Минимальная доля кириллицы
            
        Returns:
            bool: True если текст валидный
        """
        if not text:
            return False
        
        ratio = self.calculate_cyrillic_ratio(text)
        return ratio >= min_ratio


# Singleton instance
text_normalization_service = TextNormalizationService()
