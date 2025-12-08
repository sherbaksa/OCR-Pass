"""
Date Normalization Service
Нормализация и парсинг дат из различных форматов для паспортных данных.
"""

import re
from datetime import datetime
from typing import Optional, Tuple
from dateutil import parser as date_parser
from backend.core.logger import logger


class DateNormalizationService:
    """
    Сервис нормализации дат из OCR-текста.
    Поддерживает множество форматов и приводит к стандартному виду DD.MM.YYYY.
    """
    
    def __init__(self):
        """Инициализация сервиса"""
        self._date_patterns = self._build_date_patterns()
        logger.info("Date Normalization Service инициализирован")
    
    def _build_date_patterns(self) -> list:
        """
        Создание списка regex-паттернов для различных форматов дат.
        
        Returns:
            list: Список паттернов
        """
        return [
            # DD.MM.YYYY или DD.MM.YY
            r'(\d{1,2})\.(\d{1,2})\.(\d{2,4})',
            
            # DD/MM/YYYY или DD/MM/YY
            r'(\d{1,2})/(\d{1,2})/(\d{2,4})',
            
            # DD-MM-YYYY или DD-MM-YY
            r'(\d{1,2})-(\d{1,2})-(\d{2,4})',
            
            # DD MM YYYY (с пробелами)
            r'(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})',
            
            # DDMMYYYY (слитно, 8 цифр)
            r'(\d{2})(\d{2})(\d{4})',
        ]
    
    def normalize_date(
        self,
        date_string: str,
        field_type: str = "unknown"
    ) -> Optional[str]:
        """
        Нормализация даты к формату DD.MM.YYYY.
        
        Args:
            date_string: Строка с датой в любом формате
            field_type: Тип поля (birth_date, issue_date) для логирования
            
        Returns:
            Optional[str]: Дата в формате DD.MM.YYYY или None
        """
        if not date_string:
            return None
        
        # Очистка строки от лишних символов
        date_string = date_string.strip()
        
        # Попытка 1: Через regex-паттерны
        normalized = self._normalize_with_patterns(date_string)
        if normalized:
            logger.debug(f"Дата {field_type} нормализована через паттерн: {date_string} -> {normalized}")
            return normalized
        
        # Попытка 2: Через dateutil.parser (более гибкий парсинг)
        normalized = self._normalize_with_parser(date_string)
        if normalized:
            logger.debug(f"Дата {field_type} нормализована через parser: {date_string} -> {normalized}")
            return normalized
        
        logger.warning(f"Не удалось нормализовать дату {field_type}: {date_string}")
        return None
    
    def _normalize_with_patterns(self, date_string: str) -> Optional[str]:
        """
        Нормализация даты через regex-паттерны.
        
        Args:
            date_string: Строка с датой
            
        Returns:
            Optional[str]: Нормализованная дата или None
        """
        for pattern in self._date_patterns:
            match = re.search(pattern, date_string)
            if match:
                day, month, year = match.groups()
                
                # Преобразуем в числа
                try:
                    day_int = int(day)
                    month_int = int(month)
                    year_int = int(year)
                    
                    # Обработка двухзначного года
                    if year_int < 100:
                        year_int = self._expand_two_digit_year(year_int)
                    
                    # Валидация
                    if not self._is_valid_date(day_int, month_int, year_int):
                        continue
                    
                    # Форматирование
                    return f"{day_int:02d}.{month_int:02d}.{year_int:04d}"
                    
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _normalize_with_parser(self, date_string: str) -> Optional[str]:
        """
        Нормализация даты через dateutil.parser (более гибкий подход).
        
        Args:
            date_string: Строка с датой
            
        Returns:
            Optional[str]: Нормализованная дата или None
        """
        try:
            # dateutil умеет парсить много форматов
            # dayfirst=True для европейского формата (день-месяц-год)
            parsed_date = date_parser.parse(date_string, dayfirst=True)
            
            # Валидация года (паспорта не выдавались до 1900 и после текущего года)
            current_year = datetime.now().year
            if parsed_date.year < 1900 or parsed_date.year > current_year + 1:
                return None
            
            return parsed_date.strftime("%d.%m.%Y")
            
        except (ValueError, TypeError, OverflowError):
            return None
    
    def _expand_two_digit_year(self, year: int) -> int:
        """
        Преобразование двухзначного года в четырехзначный.
        Логика: 00-40 -> 2000-2040, 41-99 -> 1941-1999
        
        Args:
            year: Двухзначный год (0-99)
            
        Returns:
            int: Четырехзначный год
        """
        if year <= 40:
            # 00-40 -> 2000-2040
            return 2000 + year
        else:
            # 41-99 -> 1941-1999
            return 1900 + year
    
    def _is_valid_date(self, day: int, month: int, year: int) -> bool:
        """
        Валидация даты.
        
        Args:
            day: День
            month: Месяц
            year: Год
            
        Returns:
            bool: True если дата валидна
        """
        try:
            # Проверка диапазонов
            if not (1 <= day <= 31):
                return False
            if not (1 <= month <= 12):
                return False
            if not (1900 <= year <= datetime.now().year + 1):
                return False
            
            # Проверка через datetime (учитывает високосные годы и количество дней в месяце)
            datetime(year, month, day)
            return True
            
        except (ValueError, TypeError):
            return False
    
    def extract_dates_from_text(self, text: str) -> list:
        """
        Извлечение всех дат из текста.
        
        Args:
            text: Текст с возможными датами
            
        Returns:
            list: Список найденных дат в формате DD.MM.YYYY
        """
        if not text:
            return []
        
        dates = []
        
        for pattern in self._date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                date_str = match.group(0)
                normalized = self.normalize_date(date_str)
                if normalized and normalized not in dates:
                    dates.append(normalized)
        
        return dates
    
    def validate_birth_date(self, date_string: str) -> Tuple[bool, Optional[str]]:
        """
        Валидация даты рождения с дополнительными проверками.
        
        Args:
            date_string: Дата рождения
            
        Returns:
            Tuple[bool, Optional[str]]: (валидна, сообщение об ошибке)
        """
        normalized = self.normalize_date(date_string, "birth_date")
        
        if not normalized:
            return False, "Невозможно распознать дату рождения"
        
        try:
            # Парсим нормализованную дату
            date_obj = datetime.strptime(normalized, "%d.%m.%Y")
            
            # Проверка: дата рождения не может быть в будущем
            if date_obj > datetime.now():
                return False, "Дата рождения не может быть в будущем"
            
            # Проверка: возраст должен быть от 14 до 120 лет (паспорт РФ с 14 лет)
            age = (datetime.now() - date_obj).days / 365.25
            if age < 14:
                return False, "Возраст меньше 14 лет (паспорт не выдается)"
            if age > 120:
                return False, "Возраст больше 120 лет (нереалистично)"
            
            return True, None
            
        except (ValueError, TypeError):
            return False, "Ошибка валидации даты рождения"
    
    def validate_issue_date(
        self,
        issue_date_string: str,
        birth_date_string: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Валидация даты выдачи паспорта.
        
        Args:
            issue_date_string: Дата выдачи
            birth_date_string: Дата рождения (опционально, для доп. проверок)
            
        Returns:
            Tuple[bool, Optional[str]]: (валидна, сообщение об ошибке)
        """
        normalized_issue = self.normalize_date(issue_date_string, "issue_date")
        
        if not normalized_issue:
            return False, "Невозможно распознать дату выдачи"
        
        try:
            issue_date = datetime.strptime(normalized_issue, "%d.%m.%Y")
            
            # Проверка: дата выдачи не может быть в будущем
            if issue_date > datetime.now():
                return False, "Дата выдачи не может быть в будущем"
            
            # Проверка: паспорта РФ выдаются с 1997 года
            if issue_date.year < 1997:
                return False, "Паспорта РФ начали выдавать с 1997 года"
            
            # Если есть дата рождения - проверяем логику
            if birth_date_string:
                normalized_birth = self.normalize_date(birth_date_string, "birth_date")
                if normalized_birth:
                    birth_date = datetime.strptime(normalized_birth, "%d.%m.%Y")
                    
                    # Минимальный возраст для получения паспорта - 14 лет
                    age_at_issue = (issue_date - birth_date).days / 365.25
                    if age_at_issue < 14:
                        return False, "Паспорт не мог быть выдан раньше 14 лет"
            
            return True, None
            
        except (ValueError, TypeError):
            return False, "Ошибка валидации даты выдачи"
    
    def compare_dates(self, date1: str, date2: str) -> Optional[int]:
        """
        Сравнение двух дат.
        
        Args:
            date1: Первая дата
            date2: Вторая дата
            
        Returns:
            Optional[int]: -1 если date1 < date2, 0 если равны, 1 если date1 > date2, None при ошибке
        """
        norm1 = self.normalize_date(date1)
        norm2 = self.normalize_date(date2)
        
        if not norm1 or not norm2:
            return None
        
        try:
            dt1 = datetime.strptime(norm1, "%d.%m.%Y")
            dt2 = datetime.strptime(norm2, "%d.%m.%Y")
            
            if dt1 < dt2:
                return -1
            elif dt1 > dt2:
                return 1
            else:
                return 0
                
        except (ValueError, TypeError):
            return None


# Singleton instance
date_normalization_service = DateNormalizationService()
