"""
Address Parser Service
Интеллектуальный парсинг адреса регистрации паспорта РФ с структурированием на компоненты.
"""

import re
from typing import Optional, Dict, Tuple, List
from Levenshtein import distance as levenshtein_distance

from backend.core.logger import logger
from backend.schemas.address import (
    StructuredAddress,
    NameComparisonResult,
    STREET_TYPE_ABBREVIATIONS,
    SETTLEMENT_TYPE_ABBREVIATIONS,
    BUILDING_TYPE_ABBREVIATIONS,
)
from backend.services.text_normalization_service import text_normalization_service


class AddressParserService:
    """
    Сервис интеллектуального парсинга адресов регистрации.
    Структурирует адрес на компоненты: регион, город, улица, дом, корпус, квартира.
    """

    def __init__(self):
        """Инициализация сервиса"""
        self._patterns = self._build_extraction_patterns()
        logger.info("Address Parser Service создан")

    def _build_extraction_patterns(self) -> Dict[str, List[str]]:
        """
        Создание regex-паттернов для извлечения компонентов адреса.

        Returns:
            Dict[str, List[str]]: Словарь паттернов по типам компонентов
        """

        return {
            'postal_code': [
                r'\b(\d{6})\b',  # 6 цифр - почтовый индекс
            ],
            'region': [
                r'((?:г\.|город)\s*Москва)',
                r'((?:г\.|город)\s*Санкт-Петербург)',
                r'((?:г\.|город)\s*Севастополь)',
                r'([А-ЯЁ][а-яё]+\s+(?:область|обл|обл\.))',
                r'([А-ЯЁ][а-яё]+\s+(?:край))',
                r'([А-ЯЁ][а-яё]+\s+(?:Республика|респ|респ\.))',
                r'(Республика\s+[А-ЯЁ][а-яё]+)',
                r'([А-ЯЁ][а-яё]+\s+(?:автономный округ|АО))',
            ],
            'district': [
                r'([А-ЯЁ][а-яё]+\s+(?:район|р-н|р-н\.))',
            ],
            'city': [
                r'(?:г\.|город)\s*([А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?)',
            ],
            'settlement': [
                r'(?:п\.|пос\.|поселок)\s*([А-ЯЁ][а-яё]+)',
                r'(?:д\.|дер\.|деревня)\s*([А-ЯЁ][а-яё]+)',
                r'(?:с\.|село)\s*([А-ЯЁ][а-яё]+)',
            ],
            'street': [
                r'(?:ул\.?|ул|улица)\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*)',
                r'(?:пр-кт|просп\.?|проспект)\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*)',
                r'([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*)\s+(?:проспект|пр-кт|просп\.?)',  # НОВЫЙ: проспект после названия
                r'(?:пер\.?|переулок)\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*)',
                r'(?:б-р|бульв\.?|бульвар)\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*)',
                r'(?:пл\.?|площадь)\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*)',
                r'(?:наб\.?|набережная)\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*)',
            ],
            'house': [
                r'(?:д\.?|дом)\s*(\d+[а-яА-Я]?)',
                r'(?:владение|влд\.?)\s*(\d+)',
            ],
            'building': [
                r'(?:корп\.?|корпус|к\.?)\s*(\d+[а-яА-Я]?)',
                r'(?:стр\.?|строение)\s*(\d+)',
                r'(?:лит\.?|литера)\s*([А-ЯЁа-яё])',
            ],
            'apartment': [
                r'(?:кв\.?|квартира)\s*(\d+[а-яА-Я]?)',
                r'(?:оф\.?|офис)\s*(\d+)',
                r'(?:пом\.?|помещение)\s*(\d+)',
            ],
        }

    def parse_address(
        self,
        raw_address: str,
        extract_postal_code: bool = True,
        validate_completeness: bool = True
    ) -> StructuredAddress:
        """
        Главный метод парсинга адреса.

        Args:
            raw_address: Исходный неструктурированный адрес
            extract_postal_code: Извлекать почтовый индекс
            validate_completeness: Проверять полноту адреса

        Returns:
            StructuredAddress: Структурированный адрес с компонентами
        """
        logger.info(f"Начало парсинга адреса ({len(raw_address)} символов)")

        # Нормализация адреса
        normalized_address = text_normalization_service.normalize_address(raw_address)

        # Инициализация результата
        result = StructuredAddress(
            raw_address=raw_address,
            region=None,
            district=None,
            city=None,
            settlement=None,
            street=None,
            house=None,
            building=None,
            apartment=None,
            postal_code=None,
            component_confidences={},
            is_fully_parsed=False,
            parsing_warnings=[]
        )

        # Извлечение компонентов
        if extract_postal_code:
            postal_code, conf = self._extract_postal_code(normalized_address)
            if postal_code:
                result.postal_code = postal_code
                result.component_confidences['postal_code'] = conf

        region, conf = self._extract_region(normalized_address)
        if region:
            result.region = region
            result.component_confidences['region'] = conf

        district, conf = self._extract_district(normalized_address)
        if district:
            result.district = district
            result.component_confidences['district'] = conf

        city, conf = self._extract_city(normalized_address)
        if city:
            result.city = city
            result.component_confidences['city'] = conf

        settlement, conf = self._extract_settlement(normalized_address)
        if settlement:
            result.settlement = settlement
            result.component_confidences['settlement'] = conf

        street, conf = self._extract_street(normalized_address)
        if street:
            result.street = street
            result.component_confidences['street'] = conf

        house, conf = self._extract_house(normalized_address)
        if house:
            result.house = house
            result.component_confidences['house'] = conf

        building, conf = self._extract_building(normalized_address)
        if building:
            result.building = building
            result.component_confidences['building'] = conf

        apartment, conf = self._extract_apartment(normalized_address)
        if apartment:
            result.apartment = apartment
            result.component_confidences['apartment'] = conf

        # Проверка полноты
        if validate_completeness:
            missing = result.validate_completeness()
            if missing:
                result.parsing_warnings.append(
                    f"Отсутствуют обязательные компоненты: {', '.join(missing)}"
                )
            else:
                result.is_fully_parsed = True

        logger.info(
            f"Парсинг адреса завершен. "
            f"Извлечено компонентов: {len(result.component_confidences)}, "
            f"Средняя уверенность: {result.get_confidence():.2f}"
        )

        return result

    def _extract_postal_code(self, text: str) -> Tuple[Optional[str], float]:
        """Извлечение почтового индекса"""
        for pattern in self._patterns['postal_code']:
            match = re.search(pattern, text)
            if match:
                postal_code = match.group(1)
                # Почтовый индекс обычно в начале адреса
                if match.start() < 50:
                    return postal_code, 0.95
                return postal_code, 0.80
        return None, 0.0

    def _extract_region(self, text: str) -> Tuple[Optional[str], float]:
        """Извлечение региона"""
        for pattern in self._patterns['region']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                region = match.group(1).strip()
                # Нормализация
                region = self._normalize_component(region)
                return region, 0.90
        return None, 0.0

    def _extract_district(self, text: str) -> Tuple[Optional[str], float]:
        """Извлечение района"""
        for pattern in self._patterns['district']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                district = match.group(1).strip()
                district = self._normalize_component(district)
                return district, 0.85
        return None, 0.0

    def _extract_city(self, text: str) -> Tuple[Optional[str], float]:
        """Извлечение города"""
        for pattern in self._patterns['city']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                city = match.group(1).strip()
                city = self._normalize_component(city)
                return city, 0.90
        return None, 0.0

    def _extract_settlement(self, text: str) -> Tuple[Optional[str], float]:
        """Извлечение населенного пункта"""
        for pattern in self._patterns['settlement']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                settlement = match.group(0).strip()
                settlement = self._normalize_component(settlement)
                return settlement, 0.85
        return None, 0.0

    def _extract_street(self, text: str) -> Tuple[Optional[str], float]:
        """Извлечение улицы"""
        for pattern in self._patterns['street']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Получаем полное совпадение
                full_match = match.group(0).strip()
                
                # Если совпадение - только тип улицы, пропускаем
                if full_match.lower() in ['проспект', 'пр-кт', 'улица', 'ул', 'переулок', 'пер']:
                    continue
                
                street = self._normalize_component(full_match)
                return street, 0.90
        
        return None, 0.0

    def _extract_house(self, text: str) -> Tuple[Optional[str], float]:
        """Извлечение номера дома"""
        for pattern in self._patterns['house']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                house = match.group(1).strip()
                return house, 0.95
        return None, 0.0

    def _extract_building(self, text: str) -> Tuple[Optional[str], float]:
        """Извлечение корпуса/строения"""
        for pattern in self._patterns['building']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                building = match.group(1).strip()
                # Добавляем префикс если его нет
                if not re.match(r'^(?:корп|к|стр|лит)', building, re.IGNORECASE):
                    building = f"корп. {building}"
                return building, 0.90
        return None, 0.0

    def _extract_apartment(self, text: str) -> Tuple[Optional[str], float]:
        """Извлечение номера квартиры"""
        for pattern in self._patterns['apartment']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                apartment = match.group(1).strip()
                return apartment, 0.95
        return None, 0.0

    def _normalize_component(self, component: str) -> str:
        """
        Нормализация компонента адреса.

        Args:
            component: Компонент адреса

        Returns:
            str: Нормализованный компонент
        """
        # Убираем лишние пробелы
        component = re.sub(r'\s+', ' ', component).strip()

        # Делаем первую букву заглавной
        if component:
            component = component[0].upper() + component[1:]

        return component

    def compare_names(
        self,
        page1_surname: Optional[str],
        page1_name: Optional[str],
        page1_patronymic: Optional[str],
        page2_surname: Optional[str],
        page2_name: Optional[str],
        page2_patronymic: Optional[str],
        threshold: float = 0.85
    ) -> NameComparisonResult:
        """
        Сравнение ФИО между страницами паспорта.

        Args:
            page1_surname: Фамилия со страницы 1
            page1_name: Имя со страницы 1
            page1_patronymic: Отчество со страницы 1
            page2_surname: Фамилия со страницы 2
            page2_name: Имя со страницы 2
            page2_patronymic: Отчество со страницы 2
            threshold: Порог схожести для определения совпадения

        Returns:
            NameComparisonResult: Результат сравнения
        """
        logger.info("Начало сравнения ФИО между страницами")

        result = NameComparisonResult(
            page1_surname=page1_surname,
            page1_name=page1_name,
            page1_patronymic=page1_patronymic,
            page2_surname=page2_surname,
            page2_name=page2_name,
            page2_patronymic=page2_patronymic,
            surname_match=False,
            name_match=False,
            patronymic_match=False,
            surname_similarity=0.0,
            name_similarity=0.0,
            patronymic_similarity=0.0,
            full_match=False,
            overall_similarity=0.0,
            discrepancies=[],
            warnings=[]
        )

        # Сравнение фамилий
        if page1_surname and page2_surname:
            result.surname_similarity = self._calculate_similarity(page1_surname, page2_surname)
            result.surname_match = result.surname_similarity >= threshold

            if not result.surname_match:
                result.discrepancies.append(
                    f"Фамилии не совпадают: '{page1_surname}' vs '{page2_surname}' "
                    f"(схожесть: {result.surname_similarity:.2f})"
                )

        elif page1_surname or page2_surname:
            result.warnings.append("Фамилия присутствует только на одной странице")

        # Сравнение имен
        if page1_name and page2_name:
            result.name_similarity = self._calculate_similarity(page1_name, page2_name)
            result.name_match = result.name_similarity >= threshold

            if not result.name_match:
                result.discrepancies.append(
                    f"Имена не совпадают: '{page1_name}' vs '{page2_name}' "
                    f"(схожесть: {result.name_similarity:.2f})"
                )

        elif page1_name or page2_name:
            result.warnings.append("Имя присутствует только на одной странице")

        # Сравнение отчеств
        if page1_patronymic and page2_patronymic:
            result.patronymic_similarity = self._calculate_similarity(
                page1_patronymic,
                page2_patronymic
            )
            result.patronymic_match = result.patronymic_similarity >= threshold

            if not result.patronymic_match:
                result.discrepancies.append(
                    f"Отчества не совпадают: '{page1_patronymic}' vs '{page2_patronymic}' "
                    f"(схожесть: {result.patronymic_similarity:.2f})"
                )

        elif page1_patronymic or page2_patronymic:
            result.warnings.append("Отчество присутствует только на одной странице")

        # Общая схожесть (средневзвешенная)
        similarities = []
        weights = []

        if page1_surname and page2_surname:
            similarities.append(result.surname_similarity)
            weights.append(0.4)  # Фамилия важнее

        if page1_name and page2_name:
            similarities.append(result.name_similarity)
            weights.append(0.35)

        if page1_patronymic and page2_patronymic:
            similarities.append(result.patronymic_similarity)
            weights.append(0.25)

        if similarities:
            result.overall_similarity = sum(
                s * w for s, w in zip(similarities, weights)
            ) / sum(weights)

        # Полное совпадение
        result.full_match = (
            result.surname_match and
            result.name_match and
            result.patronymic_match and
            len(result.discrepancies) == 0
        )

        logger.info(
            f"Сравнение завершено. "
            f"Полное совпадение: {result.full_match}, "
            f"Общая схожесть: {result.overall_similarity:.2f}, "
            f"Расхождений: {len(result.discrepancies)}"
        )

        return result

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Вычисление схожести строк через нормализованное расстояние Левенштейна.

        Args:
            str1: Первая строка
            str2: Вторая строка

        Returns:
            float: Схожесть (0-1), где 1 - полное совпадение
        """
        if not str1 or not str2:
            return 0.0

        # Нормализация: нижний регистр, удаление пробелов
        str1_norm = str1.lower().strip()
        str2_norm = str2.lower().strip()

        if str1_norm == str2_norm:
            return 1.0

        # Расстояние Левенштейна
        distance = levenshtein_distance(str1_norm, str2_norm)
        max_length = max(len(str1_norm), len(str2_norm))

        if max_length == 0:
            return 0.0

        # Нормализованная схожесть
        similarity = 1.0 - (distance / max_length)

        return max(0.0, min(1.0, similarity))


# Singleton instance
address_parser_service = AddressParserService()
