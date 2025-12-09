"""
Passport Parser Service
Интеллектуальный парсер паспортных данных с нормализацией, морфологией и валидацией.
"""

import re
from typing import Optional, Dict, List, Tuple
from backend.core.logger import logger
from backend.services.text_normalization_service import text_normalization_service
from backend.services.date_normalization_service import date_normalization_service
from backend.services.name_morphology_service import name_morphology_service
from backend.services.address_parser_service import address_parser_service
from backend.schemas.passport_fields import PassportFieldType


class PassportParserService:
    """
    Сервис интеллектуального парсинга паспортных данных.
    Использует нормализацию, морфологию и словари для извлечения полей.
    """

    def __init__(self):
        """Инициализация сервиса"""
        self._initialized = False
        self._patterns = self._build_extraction_patterns()
        logger.info("Passport Parser Service создан")

    def initialize(self) -> None:
        """Инициализация всех зависимых сервисов"""
        if self._initialized:
            logger.info("Passport Parser Service уже инициализирован")
            return

        logger.info("Инициализация Passport Parser Service...")

        # Инициализация морфологического анализатора
        try:
            name_morphology_service.initialize()
            logger.info("Name Morphology Service инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации Name Morphology Service: {e}")
            raise

        self._initialized = True
        logger.info("Passport Parser Service инициализирован успешно")

    def _build_extraction_patterns(self) -> Dict[str, List[str]]:
        """
        Создание regex-паттернов для извлечения полей паспорта.

        Returns:
            Dict[str, List[str]]: Словарь паттернов по типам полей
        """
        return {
            'series': [
                r'серия[:\s]*(\d{4})',
                r'(\d{4})\s*№',
                r'паспорт[:\s]*(\d{4})',
            ],
            'number': [
                r'номер[:\s]*(\d{6})',
                r'№[:\s]*(\d{6})',
                r'\d{4}\s*[№-]\s*(\d{6})',
            ],
            'series_number': [
                r'(\d{4})\s*[№-]?\s*(\d{6})',
                r'серия\s*(\d{4})\s*номер\s*(\d{6})',
            ],
            'department_code': [
                r'код\s*подразделения[:\s]*(\d{3}[-]?\d{3})',
                r'(\d{3}[-]\d{3})',
            ],
            'gender': [
                r'пол[:\s]*([МЖ])',
                r'\bпол\b[:\s]*([МЖ])',
            ],
            'issued_by_markers': [
                r'кем\s+выдан',
                r'выдан',
                r'орган',
            ],
            'birth_place_markers': [
                r'место\s+рождения',
                r'рожден[ия]?',
            ],
            'registration_markers': [
                r'место\s+жительства',
                r'зарегистрирован',
                r'адрес',
            ],
        }

    def parse_passport_text(self, ocr_text: str, parse_address_structure: bool = True) -> Dict[str, any]:
        """
        Главный метод парсинга паспортного текста.

        Args:
            ocr_text: OCR-текст паспорта
            parse_address_structure: Структурировать адрес на компоненты

        Returns:
            Dict: Извлеченные и нормализованные поля паспорта
        """
        if not self._initialized:
            self.initialize()

        logger.info(f"Начало парсинга паспортного текста ({len(ocr_text)} символов)")

        # Нормализация текста
        normalized_text = text_normalization_service.normalize_text(ocr_text)

        result = {
            'fields': {},
            'confidences': {},
            'errors': [],
            'warnings': [],
            'structured_address': None  # Новое поле для структурированного адреса
        }

        # Извлечение серии и номера
        series, number, conf_series, conf_number = self._extract_series_number(normalized_text)
        if series:
            result['fields']['series'] = series
            result['confidences']['series'] = conf_series
        if number:
            result['fields']['number'] = number
            result['confidences']['number'] = conf_number

        # Извлечение кода подразделения
        dept_code, conf_dept = self._extract_department_code(normalized_text)
        if dept_code:
            result['fields']['department_code'] = dept_code
            result['confidences']['department_code'] = conf_dept

        # Извлечение дат
        dates = self._extract_dates(normalized_text)
        if dates.get('birth_date'):
            result['fields']['birth_date'] = dates['birth_date']
            result['confidences']['birth_date'] = dates.get('birth_date_conf', 0.7)
        if dates.get('issue_date'):
            result['fields']['issue_date'] = dates['issue_date']
            result['confidences']['issue_date'] = dates.get('issue_date_conf', 0.7)

        # Извлечение пола
        gender, conf_gender = self._extract_gender(normalized_text)
        if gender:
            result['fields']['gender'] = gender
            result['confidences']['gender'] = conf_gender

        # Извлечение ФИО
        fio = self._extract_fio(normalized_text)
        if fio.get('surname'):
            result['fields']['surname'] = fio['surname']
            result['confidences']['surname'] = fio.get('surname_conf', 0.7)
        if fio.get('name'):
            result['fields']['name'] = fio['name']
            result['confidences']['name'] = fio.get('name_conf', 0.7)
        if fio.get('patronymic'):
            result['fields']['patronymic'] = fio['patronymic']
            result['confidences']['patronymic'] = fio.get('patronymic_conf', 0.7)

        # Извлечение места рождения
        birth_place, conf_bp = self._extract_birth_place(normalized_text)
        if birth_place:
            result['fields']['birth_place'] = birth_place
            result['confidences']['birth_place'] = conf_bp

        # Извлечение органа выдачи
        issued_by, conf_ib = self._extract_issued_by(normalized_text)
        if issued_by:
            result['fields']['issued_by'] = issued_by
            result['confidences']['issued_by'] = conf_ib

        # Извлечение адреса регистрации (с опциональным структурированием)
        address_data = self._extract_registration_address(normalized_text, parse_address_structure)
        if address_data:
            # Основной адрес (строка)
            result['fields']['registration_address'] = address_data['address']
            result['confidences']['registration_address'] = address_data['confidence']
            
            # Структурированный адрес (если запрошено)
            if parse_address_structure and address_data.get('structured_address'):
                result['structured_address'] = address_data['structured_address']
                
                # Добавляем предупреждения из парсинга адреса
                if address_data['structured_address'].parsing_warnings:
                    result['warnings'].extend(address_data['structured_address'].parsing_warnings)

        # Валидация и кросс-проверки
        self._validate_parsed_data(result)

        logger.info(f"Парсинг завершен: извлечено {len(result['fields'])} полей")

        return result

    def _extract_series_number(self, text: str) -> Tuple[Optional[str], Optional[str], float, float]:
        """Извлечение серии и номера паспорта"""

        # Попытка 1: Серия и номер вместе
        for pattern in self._patterns['series_number']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                series = text_normalization_service.normalize_passport_number(match.group(1))
                number = text_normalization_service.normalize_passport_number(match.group(2))

                if len(series) == 4 and len(number) == 6:
                    return series, number, 0.95, 0.95

        # Попытка 2: Серия отдельно
        series = None
        conf_series = 0.0
        for pattern in self._patterns['series']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                series = text_normalization_service.normalize_passport_number(match.group(1))
                if len(series) == 4:
                    conf_series = 0.9
                    break

        # Попытка 3: Номер отдельно
        number = None
        conf_number = 0.0
        for pattern in self._patterns['number']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                number = text_normalization_service.normalize_passport_number(match.group(1))
                if len(number) == 6:
                    conf_number = 0.9
                    break

        return series, number, conf_series, conf_number

    def _extract_department_code(self, text: str) -> Tuple[Optional[str], float]:
        """Извлечение кода подразделения"""

        for pattern in self._patterns['department_code']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                code = text_normalization_service.normalize_department_code(match.group(1))

                # Проверка формата XXX-XXX
                if re.match(r'^\d{3}-\d{3}$', code):
                    return code, 0.95

        return None, 0.0

    def _extract_dates(self, text: str) -> Dict[str, any]:
        """Извлечение дат из текста"""

        result = {}

        # Извлекаем все возможные даты
        all_dates = date_normalization_service.extract_dates_from_text(text)

        if not all_dates:
            return result

        # Эвристика: первая дата обычно - дата рождения, вторая - дата выдачи
        if len(all_dates) >= 1:
            # Валидация даты рождения
            is_valid, error = date_normalization_service.validate_birth_date(all_dates[0])
            if is_valid:
                result['birth_date'] = all_dates[0]
                result['birth_date_conf'] = 0.85

        if len(all_dates) >= 2:
            # Валидация даты выдачи
            birth_date = result.get('birth_date')
            is_valid, error = date_normalization_service.validate_issue_date(
                all_dates[1],
                birth_date_string=birth_date
            )
            if is_valid:
                result['issue_date'] = all_dates[1]
                result['issue_date_conf'] = 0.85

        return result

    def _extract_gender(self, text: str) -> Tuple[Optional[str], float]:
        """Извлечение пола"""

        for pattern in self._patterns['gender']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gender = match.group(1).upper()
                if gender in ['М', 'Ж']:
                    return gender, 0.95

        return None, 0.0

    def _extract_fio(self, text: str) -> Dict[str, any]:
        """Извлечение ФИО с морфологическим анализом"""

        result = {}

        # Извлечение кириллических слов
        cyrillic_words = re.findall(r'[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?', text)

        if not cyrillic_words:
            return result

        # Кандидаты на ФИО - первые 3-5 слов
        fio_candidates = cyrillic_words[:5]

        # Пытаемся найти фамилию, имя, отчество
        surname = None
        name = None
        patronymic = None

        for word in fio_candidates:
            # Проверка на отчество (имеет специфичные окончания)
            is_valid, conf, corrected = name_morphology_service.validate_patronymic(word)
            if is_valid and conf > 0.7 and not patronymic:
                patronymic = corrected or word
                result['patronymic'] = patronymic
                result['patronymic_conf'] = conf
                continue

            # Проверка на имя
            is_valid, conf, corrected = name_morphology_service.validate_name(word)
            if is_valid and conf > 0.7 and not name:
                name = corrected or word
                result['name'] = name
                result['name_conf'] = conf
                continue

            # Проверка на фамилию
            is_valid, conf, corrected = name_morphology_service.validate_surname(word)
            if is_valid and conf > 0.7 and not surname:
                surname = corrected or word
                result['surname'] = surname
                result['surname_conf'] = conf
                continue

        # Если пол не определен - пытаемся извлечь из ФИО
        if surname or name or patronymic:
            gender = name_morphology_service.extract_gender_from_name(surname, name, patronymic)
            if gender:
                result['gender_from_name'] = gender

        return result

    def _extract_birth_place(self, text: str) -> Tuple[Optional[str], float]:
        """Извлечение места рождения"""

        # Ищем маркер "место рождения"
        for marker in self._patterns['birth_place_markers']:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                # Берем текст после маркера (до 200 символов)
                start_pos = match.end()
                fragment = text[start_pos:start_pos + 200]

                # Очищаем
                cleaned = text_normalization_service.clean_ocr_text(fragment)

                # Берем первое предложение или до точки
                sentences = re.split(r'[\.;]', cleaned)
                if sentences:
                    birth_place = sentences[0].strip()
                    if len(birth_place) >= 10:
                        return birth_place, 0.75

        return None, 0.0

    def _extract_issued_by(self, text: str) -> Tuple[Optional[str], float]:
        """Извлечение органа выдачи"""

        # Ищем маркер "кем выдан"
        for marker in self._patterns['issued_by_markers']:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                # Берем текст после маркера
                start_pos = match.end()
                fragment = text[start_pos:start_pos + 300]

                # Очищаем
                cleaned = text_normalization_service.clean_ocr_text(fragment)

                # Берем до точки или до следующего маркера
                sentences = re.split(r'[\.;]', cleaned)
                if sentences:
                    issued_by = sentences[0].strip()
                    if len(issued_by) >= 20:
                        return issued_by, 0.75

        return None, 0.0

    def _extract_registration_address(
        self, 
        text: str, 
        parse_structure: bool = True
    ) -> Optional[Dict[str, any]]:
        """
        Извлечение адреса регистрации с опциональным структурированием.
        
        Args:
            text: Текст для парсинга
            parse_structure: Структурировать адрес на компоненты
            
        Returns:
            Dict с ключами:
                - address: строка адреса
                - confidence: уверенность
                - structured_address: структурированный адрес (если parse_structure=True)
        """
        # Ищем маркер адреса
        for marker in self._patterns['registration_markers']:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                # Берем текст после маркера
                start_pos = match.end()
                fragment = text[start_pos:start_pos + 500]

                # Нормализуем адрес
                address = text_normalization_service.normalize_address(fragment)

                if len(address) >= 20:
                    result = {
                        'address': address,
                        'confidence': 0.7,
                        'structured_address': None
                    }
                    
                    # Опционально структурируем адрес
                    if parse_structure:
                        try:
                            structured = address_parser_service.parse_address(
                                raw_address=address,
                                extract_postal_code=True,
                                validate_completeness=True
                            )
                            result['structured_address'] = structured
                            
                            # Обновляем уверенность на основе структурирования
                            struct_confidence = structured.get_confidence()
                            if struct_confidence > 0:
                                result['confidence'] = max(result['confidence'], struct_confidence)
                                
                        except Exception as e:
                            logger.warning(f"Ошибка структурирования адреса: {e}")
                    
                    return result

        return None

    def _validate_parsed_data(self, result: Dict) -> None:
        """Валидация и кросс-проверки извлеченных данных"""

        fields = result['fields']

        # Проверка серии и номера
        if 'series' in fields and len(fields['series']) != 4:
            result['errors'].append("Серия паспорта должна содержать 4 цифры")

        if 'number' in fields and len(fields['number']) != 6:
            result['errors'].append("Номер паспорта должен содержать 6 цифр")

        # Проверка кода подразделения
        if 'department_code' in fields:
            if not re.match(r'^\d{3}-\d{3}$', fields['department_code']):
                result['errors'].append("Неверный формат кода подразделения")

        # Кросс-проверка дат
        if 'birth_date' in fields and 'issue_date' in fields:
            comparison = date_normalization_service.compare_dates(
                fields['birth_date'],
                fields['issue_date']
            )
            if comparison is not None and comparison >= 0:
                result['errors'].append("Дата выдачи должна быть позже даты рождения")

        # Проверка ФИО
        if 'surname' in fields or 'name' in fields or 'patronymic' in fields:
            validation = name_morphology_service.validate_full_name(
                fields.get('surname'),
                fields.get('name'),
                fields.get('patronymic')
            )

            if not validation['is_valid']:
                result['warnings'].extend(validation['errors'])

            # Применяем исправления если есть
            if validation['corrected']:
                for field_name, corrected_value in validation['corrected'].items():
                    result['fields'][field_name] = corrected_value
                    result['warnings'].append(f"Исправлено {field_name}: {corrected_value}")

        # Валидация структурированного адреса
        if result.get('structured_address'):
            missing = result['structured_address'].validate_completeness()
            if missing:
                result['warnings'].append(
                    f"Адрес неполный. Отсутствуют: {', '.join(missing)}"
                )


# Singleton instance
passport_parser_service = PassportParserService()
