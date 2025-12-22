"""
Passport MRZ Parser Service
Сервис парсинга машиночитаемой зоны (МЧЗ) паспорта РФ для извлечения данных первой страницы.
Включает гибридный подход: структура из МЧЗ + кириллические значения из OCR текста.
"""

import re
from typing import Optional, Dict, Tuple, List
from backend.core.logger import logger


class PassportMRZParserService:
    """
    Сервис парсинга машиночитаемой зоны паспорта РФ с гибридным извлечением данных.
    
    Структура МЧЗ российского паспорта (2 строки по 44 символа):
    
    Верхняя строка:
    - Позиции 1-2: Тип документа (PN)
    - Позиции 3-5: Код государства (RUS)
    - Позиции 6-44: ФИО (разделители <<)
    
    Нижняя строка:
    - Позиции 1-9: Серия (3 цифры) + Номер (6 цифр)
    - Позиция 10: Контрольная цифра
    - Позиции 11-13: Гражданство (RUS)
    - Позиции 14-19: Дата рождения (YYMMDD)
    - Позиция 20: Контрольная цифра
    - Позиция 21: Пол (M/F)
    - Позиции 22-27: Дата истечения (заполнители)
    - Позиция 28: Контрольная цифра
    - Позиции 29-42: Дополнительные данные (последняя цифра серии + дата выдачи + код подразделения)
    - Позиция 43: Контрольная цифра
    - Позиция 44: Заключительная контрольная цифра
    """

    def __init__(self):
        """Инициализация сервиса"""
        self._mrz_pattern_line1 = re.compile(r'^PN[A-Z]{3}[A-Z<]{39}$')
        self._mrz_pattern_line2 = re.compile(r'^\d{9}\d[A-Z]{3}\d{6}\d[MF][<\d]{6}\d[<\d]{14}\d\d$')
        
        # Таблица транслитерации для сопоставления латиницы и кириллицы
        self._translit_table = {
            'A': 'А', 'B': 'Б', 'V': 'В', 'G': 'Г', 'D': 'Д',
            'E': 'Е', 'Z': 'З', 'I': 'И', 'K': 'К', 'L': 'Л',
            'M': 'М', 'N': 'Н', 'O': 'О', 'P': 'П', 'R': 'Р',
            'S': 'С', 'T': 'Т', 'U': 'У', 'F': 'Ф', 'H': 'Х',
            'C': 'Ц', 'Y': 'Ы', 'Q': 'К', 'W': 'В', 'J': 'Й'
        }
        
        logger.info("Passport MRZ Parser Service создан")

    def find_mrz_in_texts(self, rec_texts: list) -> Optional[Tuple[str, str]]:
        """
        Найти две строки МЧЗ в списке распознанных текстов.
        
        Args:
            rec_texts: Список распознанных текстов от OCR
            
        Returns:
            Tuple[str, str] или None: (верхняя строка, нижняя строка) МЧЗ
        """
        if not rec_texts:
            return None
            
        mrz_lines = []
        
        for text in rec_texts:
            # Очистка текста от пробелов
            cleaned = text.replace(' ', '').strip()
            
            # Проверка на строки МЧЗ (начинаются с PN или содержат RUS и цифры)
            if self._is_mrz_line(cleaned):
                mrz_lines.append(cleaned)
                
        # МЧЗ должна состоять из 2 строк
        if len(mrz_lines) >= 2:
            # Ищем пару: верхняя строка (начинается с PN) и нижняя (начинается с цифр)
            upper = None
            lower = None
            
            for line in mrz_lines:
                if line.startswith('PN') or line.startswith('P<'):
                    upper = line
                elif re.match(r'^\d', line) and len(line) >= 40:
                    lower = line
                    
            if upper and lower:
                logger.info(f"МЧЗ найдена: верхняя={upper[:20]}..., нижняя={lower[:20]}...")
                return (upper, lower)
                
        return None

    def _is_mrz_line(self, text: str) -> bool:
        """
        Проверка, является ли строка частью МЧЗ.
        
        Args:
            text: Очищенная строка текста
            
        Returns:
            bool: True если похоже на МЧЗ
        """
        # МЧЗ содержит только заглавные латинские буквы, цифры и знак <
        if not re.match(r'^[A-Z0-9<]+$', text):
            return False
            
        # МЧЗ имеет длину около 44 символов
        if len(text) < 40 or len(text) > 50:
            return False
            
        # Верхняя строка начинается с PN или P<
        if text.startswith('PN') or text.startswith('P<'):
            return True
            
        # Нижняя строка начинается с цифр и содержит RUS
        if re.match(r'^\d{9}', text) and 'RUS' in text:
            return True
            
        return False

    def parse_mrz(self, upper_line: str, lower_line: str, rec_texts: List[str] = None) -> Dict[str, any]:
        """
        Парсинг МЧЗ паспорта РФ с гибридным извлечением ФИО.
        
        Args:
            upper_line: Верхняя строка МЧЗ
            lower_line: Нижняя строка МЧЗ
            rec_texts: Список всех распознанных текстов (для поиска кириллических ФИО)
            
        Returns:
            Dict с извлеченными данными
        """
        result = {
            'fields': {},
            'confidences': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Парсинг верхней строки (ФИО) - получаем латинские версии
            fio_data_latin = self._parse_upper_line(upper_line)
            
            # Гибридный подход: ищем кириллические соответствия в rec_texts
            if rec_texts:
                fio_data = self._find_cyrillic_fio(fio_data_latin, rec_texts)
            else:
                # Если rec_texts нет, используем латинские версии
                fio_data = fio_data_latin
                result['warnings'].append("Кириллические ФИО не найдены, используются латинские из МЧЗ")
            
            result['fields'].update(fio_data)
            
            # Высокая уверенность для ФИО
            for field in ['surname', 'name', 'patronymic']:
                if field in fio_data:
                    result['confidences'][field] = 0.95
            
            # Парсинг нижней строки (серия, номер, даты, пол)
            data = self._parse_lower_line(lower_line)
            result['fields'].update(data)
            
            # Высокая уверенность для всех полей из МЧЗ
            for field in ['series', 'number', 'birth_date', 'issue_date', 'department_code', 'gender']:
                if field in data:
                    result['confidences'][field] = 0.98
            
            # Извлечение дополнительных полей из rec_texts
            if rec_texts:
                birth_place = self._extract_birth_place_from_texts(rec_texts)
                if birth_place:
                    result['fields']['birth_place'] = birth_place
                    result['confidences']['birth_place'] = 0.85
                    
                issued_by = self._extract_issued_by_from_texts(rec_texts)
                if issued_by:
                    result['fields']['issued_by'] = issued_by
                    result['confidences']['issued_by'] = 0.85
                    
            logger.info(f"МЧЗ успешно распарсена. Извлечено полей: {len(result['fields'])}")
            
        except Exception as e:
            error_msg = f"Ошибка парсинга МЧЗ: {e}"
            logger.error(error_msg)
            result['errors'].append(error_msg)
            
        return result

    def _find_cyrillic_fio(self, fio_latin: Dict[str, str], rec_texts: List[str]) -> Dict[str, str]:
        """
        Поиск кириллических соответствий для латинских ФИО из МЧЗ.
        
        Args:
            fio_latin: Словарь с латинскими ФИО из МЧЗ
            rec_texts: Список распознанных текстов
            
        Returns:
            Dict с кириллическими ФИО
        """
        result = {}
        
        # Ищем только в первых 50 строках (там обычно находятся ФИО)
        search_range = rec_texts[:50] if len(rec_texts) > 50 else rec_texts
        
        # Поиск фамилии
        if 'surname' in fio_latin:
            cyrillic = self._find_cyrillic_match(
                fio_latin['surname'], 
                search_range,
                min_length=4,
                max_length=20
            )
            if cyrillic:
                result['surname'] = cyrillic
                logger.info(f"Найдена кириллическая фамилия: {cyrillic} (вместо {fio_latin['surname']})")
            else:
                result['surname'] = fio_latin['surname']
                
        # Поиск имени
        if 'name' in fio_latin:
            cyrillic = self._find_cyrillic_match(
                fio_latin['name'],
                search_range,
                min_length=3,
                max_length=15
            )
            if cyrillic:
                result['name'] = cyrillic
                logger.info(f"Найдено кириллическое имя: {cyrillic} (вместо {fio_latin['name']})")
            else:
                result['name'] = fio_latin['name']
                
        # Поиск отчества
        if 'patronymic' in fio_latin:
            cyrillic = self._find_cyrillic_match(
                fio_latin['patronymic'],
                search_range,
                min_length=5,
                max_length=20
            )
            if cyrillic:
                result['patronymic'] = cyrillic
                logger.info(f"Найдено кириллическое отчество: {cyrillic} (вместо {fio_latin['patronymic']})")
            else:
                result['patronymic'] = fio_latin['patronymic']
                
        return result

    def _find_cyrillic_match(
        self, 
        latin_word: str, 
        texts: List[str],
        min_length: int = 3,
        max_length: int = 20
    ) -> Optional[str]:
        """
        Найти кириллическое слово, соответствующее латинскому.
        
        Args:
            latin_word: Латинское слово
            texts: Список текстов для поиска
            min_length: Минимальная длина слова
            max_length: Максимальная длина слова
            
        Returns:
            Кириллическое слово или None
        """
        # Транслитерируем латинское слово для сравнения
        expected_cyrillic = self._transliterate_to_cyrillic(latin_word)
        target_length = len(latin_word)
        
        # Ищем кириллические слова подходящей длины
        for text in texts:
            # Проверяем, что строка содержит кириллицу
            if not re.search(r'[А-ЯЁ]', text):
                continue
                
            # Извлекаем кириллические слова
            words = re.findall(r'[А-ЯЁ][А-ЯЁа-яё]+', text)
            
            for word in words:
                word_upper = word.upper()
                word_len = len(word_upper)
                
                # Проверка длины (допускаем отклонение ±2 символа)
                if word_len < min_length or word_len > max_length:
                    continue
                    
                if abs(word_len - target_length) > 2:
                    continue
                
                # Проверка похожести
                similarity = self._calculate_similarity(expected_cyrillic, word_upper)
                
                if similarity > 0.6:  # Порог похожести
                    return word_upper
                    
        return None

    def _transliterate_to_cyrillic(self, latin_text: str) -> str:
        """
        Транслитерация латиницы в кириллицу для сопоставления.
        
        Args:
            latin_text: Латинский текст
            
        Returns:
            Кириллический текст
        """
        result = []
        for char in latin_text.upper():
            result.append(self._translit_table.get(char, char))
        return ''.join(result)

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Расчет похожести двух строк (простой алгоритм).
        
        Args:
            str1: Первая строка
            str2: Вторая строка
            
        Returns:
            Коэффициент похожести (0.0 - 1.0)
        """
        if not str1 or not str2:
            return 0.0
            
        # Считаем совпадающие символы на тех же позициях
        matches = sum(1 for a, b in zip(str1, str2) if a == b)
        max_len = max(len(str1), len(str2))
        
        return matches / max_len if max_len > 0 else 0.0

    def _extract_birth_place_from_texts(self, rec_texts: List[str]) -> Optional[str]:
        """
        Извлечение места рождения из распознанных текстов.
        
        Args:
            rec_texts: Список распознанных текстов
            
        Returns:
            Место рождения или None
        """
        # Ищем маркер "рождения"
        marker_indices = []
        for i, text in enumerate(rec_texts):
            if 'рождения' in text.lower() or 'рожден' in text.lower():
                marker_indices.append(i)
        
        if not marker_indices:
            return None
        
        # Берем последний маркер (обычно там идет место рождения)
        marker_idx = marker_indices[-1]
        
        # Собираем 2-4 строки после маркера
        parts = []
        for i in range(marker_idx + 1, min(marker_idx + 5, len(rec_texts))):
            text = rec_texts[i].strip()
            
            # Пропускаем мусор
            if len(text) < 2:
                continue
            if text in ['RUSSIA', 'RUSSLA', 'POCCИA', 'ОССИЯ', 'llox', 'Pe', 'Mecro']:
                continue
            if re.match(r'^\d+$', text):  # Только цифры
                continue
                
            # Проверяем, что это кириллический текст
            if re.search(r'[А-ЯЁ]', text):
                parts.append(text)
                
            # Останавливаемся, если собрали достаточно
            if len(parts) >= 3:
                break
        
        if parts:
            birth_place = ' '.join(parts)
            logger.info(f"Извлечено место рождения: {birth_place}")
            return birth_place
            
        return None

    def _extract_issued_by_from_texts(self, rec_texts: List[str]) -> Optional[str]:
        """
        Извлечение органа выдачи из распознанных текстов.
        Обычно состоит из 2 строк после маркера "выдан".
        
        Args:
            rec_texts: Список распознанных текстов
            
        Returns:
            Орган выдачи или None
        """
        # Ищем маркер "выдан"
        for i, text in enumerate(rec_texts):
            if 'выдан' in text.lower():
                # Берем следующие 2 строки
                parts = []
                for j in range(i + 1, min(i + 3, len(rec_texts))):
                    line = rec_texts[j].strip()
                    
                    # Пропускаем мусор
                    if len(line) < 3:
                        continue
                    if line in ['RUSSIA', 'RUSSLA', 'POCCИA', 'ОССИЯ']:
                        continue
                    if re.match(r'^\d+$', line):
                        continue
                        
                    # Проверяем наличие кириллицы
                    if re.search(r'[А-ЯЁ]', line):
                        # Очищаем от лишних латинских символов в конце
                        cleaned = re.sub(r'[A-Z0-9\s]+$', '', line).strip()
                        if cleaned:
                            parts.append(cleaned)
                
                if parts:
                    issued_by = ' '.join(parts)
                    logger.info(f"Извлечен орган выдачи: {issued_by}")
                    return issued_by
        
        return None

    def _parse_upper_line(self, line: str) -> Dict[str, str]:
        """
        Парсинг верхней строки МЧЗ (ФИО) - возвращает латинские версии.
        
        Args:
            line: Верхняя строка МЧЗ
            
        Returns:
            Dict с фамилией, именем, отчеством (латиница)
        """
        # Убираем PN и код страны (первые 5 символов)
        fio_part = line[5:].rstrip('<')
        
        # ФИО разделены символами <<
        # Формат: ФАМИЛИЯ<<ИМЯ<ОТЧЕСТВО
        parts = fio_part.split('<<')
        
        result = {}
        
        if len(parts) >= 1:
            # Фамилия
            surname = parts[0].replace('<', '-').strip('-')
            if surname:
                result['surname'] = surname.capitalize()
                
        if len(parts) >= 2:
            # Имя и отчество разделены одним <
            name_patronymic = parts[1].split('<')
            
            if len(name_patronymic) >= 1:
                name = name_patronymic[0].strip()
                if name:
                    result['name'] = name.capitalize()
                    
            if len(name_patronymic) >= 2:
                patronymic = name_patronymic[1].strip()
                if patronymic:
                    result['patronymic'] = patronymic.capitalize()
                    
        return result

    def _parse_lower_line(self, line: str) -> Dict[str, str]:
        """
        Парсинг нижней строки МЧЗ.
        
        Args:
            line: Нижняя строка МЧЗ (44 символа)
            
        Returns:
            Dict с данными паспорта
        """
        result = {}
        
        # Убеждаемся, что строка достаточной длины
        if len(line) < 44:
            logger.warning(f"МЧЗ нижняя строка слишком короткая: {len(line)} символов")
            # Дополняем строку заполнителями если нужно
            line = line.ljust(44, '<')
        
        # Позиции 1-9: Серия (3 цифры) + Номер (6 цифр)
        series_number = line[0:9]
        if series_number.isdigit():
            series_first_part = series_number[0:3]  # Первые 3 цифры серии
            number = series_number[3:9]  # 6 цифр номера
            result['number'] = number
        
        # Позиции 14-19: Дата рождения (YYMMDD)
        birth_date_raw = line[13:19]
        if birth_date_raw.isdigit():
            birth_date = self._format_date(birth_date_raw)
            if birth_date:
                result['birth_date'] = birth_date
        
        # Позиция 21: Пол (M/F)
        gender = line[20]
        if gender in ['M', 'F']:
            result['gender'] = 'М' if gender == 'M' else 'Ж'
        
        # Позиции 29-42: Дополнительные данные
        additional = line[28:42]
        
        # Первая цифра - последняя цифра серии
        if len(additional) >= 1 and additional[0].isdigit():
            series_last_digit = additional[0]
            # Собираем полную серию
            if 'number' in result:  # Если успешно извлекли первую часть
                full_series = series_first_part + series_last_digit
                result['series'] = full_series
        
        # Позиции 2-7: Дата выдачи (YYMMDD)
        if len(additional) >= 7:
            issue_date_raw = additional[1:7]
            if issue_date_raw.isdigit():
                issue_date = self._format_date(issue_date_raw)
                if issue_date:
                    result['issue_date'] = issue_date
        
        # Позиции 8-13: Код подразделения (6 цифр)
        if len(additional) >= 13:
            dept_code_raw = additional[7:13]
            if dept_code_raw.isdigit():
                # Форматируем как XXX-XXX
                dept_code = f"{dept_code_raw[0:3]}-{dept_code_raw[3:6]}"
                result['department_code'] = dept_code
        
        return result

    def _format_date(self, date_str: str) -> Optional[str]:
        """
        Форматирование даты из YYMMDD в DD.MM.YYYY.
        
        Args:
            date_str: Дата в формате YYMMDD
            
        Returns:
            Дата в формате DD.MM.YYYY или None
        """
        if len(date_str) != 6 or not date_str.isdigit():
            return None
            
        yy = date_str[0:2]
        mm = date_str[2:4]
        dd = date_str[4:6]
        
        # Определяем век (19xx или 20xx)
        year_int = int(yy)
        if year_int >= 0 and year_int <= 25:
            yyyy = f"20{yy}"
        else:
            yyyy = f"19{yy}"
        
        # Проверка корректности даты
        try:
            day = int(dd)
            month = int(mm)
            
            if day < 1 or day > 31:
                return None
            if month < 1 or month > 12:
                return None
                
            return f"{dd}.{mm}.{yyyy}"
            
        except ValueError:
            return None

    def parse_from_ocr_texts(self, rec_texts: list) -> Dict[str, any]:
        """
        Полный цикл: поиск МЧЗ в текстах и парсинг с гибридным извлечением.
        
        Args:
            rec_texts: Список распознанных текстов от OCR
            
        Returns:
            Dict с извлеченными данными
        """
        # Ищем МЧЗ
        mrz_lines = self.find_mrz_in_texts(rec_texts)
        
        if not mrz_lines:
            logger.warning("МЧЗ не найдена в распознанных текстах")
            return {
                'fields': {},
                'confidences': {},
                'errors': ['МЧЗ не найдена'],
                'warnings': []
            }
        
        # Парсим МЧЗ с передачей rec_texts для гибридного извлечения
        upper_line, lower_line = mrz_lines
        return self.parse_mrz(upper_line, lower_line, rec_texts)


# Singleton instance
passport_mrz_parser_service = PassportMRZParserService()
