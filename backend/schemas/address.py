"""
Схемы данных для структурированного адреса и синхронизации ФИО паспорта РФ.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum


class AddressComponentType(str, Enum):
    """Типы компонентов адреса"""
    REGION = "region"  # Регион / область / край / республика
    DISTRICT = "district"  # Район
    CITY = "city"  # Город
    SETTLEMENT = "settlement"  # Населенный пункт / поселок / деревня
    STREET = "street"  # Улица / проспект / переулок
    HOUSE = "house"  # Номер дома
    BUILDING = "building"  # Корпус / строение
    APARTMENT = "apartment"  # Квартира
    POSTAL_CODE = "postal_code"  # Почтовый индекс


class StructuredAddress(BaseModel):
    """
    Структурированный адрес регистрации из паспорта РФ.
    """
    # Основные компоненты
    region: Optional[str] = Field(None, description="Регион (область, край, республика)")
    district: Optional[str] = Field(None, description="Район")
    city: Optional[str] = Field(None, description="Город")
    settlement: Optional[str] = Field(None, description="Населенный пункт (поселок, деревня)")
    street: Optional[str] = Field(None, description="Улица (проспект, переулок)")
    house: Optional[str] = Field(None, description="Номер дома")
    building: Optional[str] = Field(None, description="Корпус / строение")
    apartment: Optional[str] = Field(None, description="Номер квартиры")
    postal_code: Optional[str] = Field(None, description="Почтовый индекс")

    # Исходный адрес
    raw_address: str = Field(..., description="Исходный неструктурированный адрес")

    # Уверенность по компонентам
    component_confidences: Dict[str, float] = Field(
        default_factory=dict,
        description="Уверенность для каждого компонента (0-1)"
    )

    # Метаданные
    is_fully_parsed: bool = Field(False, description="Все ли компоненты извлечены")
    parsing_warnings: List[str] = Field(default_factory=list, description="Предупреждения при парсинге")

    class Config:
        json_schema_extra = {
            "example": {
                "region": "г. Москва",
                "district": None,
                "city": "Москва",
                "settlement": None,
                "street": "ул. Ленина",
                "house": "10",
                "building": "корп. 2",
                "apartment": "кв. 5",
                "postal_code": "123456",
                "raw_address": "г. Москва, ул. Ленина, д. 10, корп. 2, кв. 5",
                "component_confidences": {
                    "region": 0.95,
                    "city": 0.95,
                    "street": 0.90,
                    "house": 0.95,
                    "building": 0.85,
                    "apartment": 0.90
                },
                "is_fully_parsed": True,
                "parsing_warnings": []
            }
        }

    def to_formatted_string(self, include_postal_code: bool = False) -> str:
        """
        Форматирование адреса в читаемую строку.

        Args:
            include_postal_code: Включить почтовый индекс

        Returns:
            str: Отформатированный адрес
        """
        parts = []

        if include_postal_code and self.postal_code:
            parts.append(self.postal_code)

        if self.region:
            parts.append(self.region)

        if self.district:
            parts.append(self.district)

        if self.city:
            parts.append(f"г. {self.city}" if not self.city.startswith("г.") else self.city)

        if self.settlement:
            parts.append(self.settlement)

        if self.street:
            parts.append(self.street)

        if self.house:
            house_str = f"д. {self.house}" if not self.house.startswith("д.") else self.house
            if self.building:
                building_str = f"корп. {self.building}" if not self.building.startswith("корп.") else self.building
                house_str += f", {building_str}"
            parts.append(house_str)

        if self.apartment:
            apt_str = f"кв. {self.apartment}" if not self.apartment.startswith("кв.") else self.apartment
            parts.append(apt_str)

        return ", ".join(parts)

    def get_confidence(self) -> float:
        """
        Получить среднюю уверенность по всем компонентам.

        Returns:
            float: Средняя уверенность (0-1)
        """
        if not self.component_confidences:
            return 0.0

        return sum(self.component_confidences.values()) / len(self.component_confidences)

    def validate_completeness(self) -> List[str]:
        """
        Проверить полноту адреса (обязательные компоненты).

        Returns:
            List[str]: Список отсутствующих обязательных компонентов
        """
        missing = []

        # Обязательные компоненты для юридически корректного адреса РФ
        if not self.region:
            missing.append("region")

        if not self.city and not self.settlement:
            missing.append("city_or_settlement")

        if not self.street:
            missing.append("street")

        if not self.house:
            missing.append("house")

        return missing


class NameComparisonResult(BaseModel):
    """
    Результат сравнения ФИО между страницами паспорта.
    """
    # Сравниваемые имена
    page1_surname: Optional[str] = Field(None, description="Фамилия со страницы 1")
    page1_name: Optional[str] = Field(None, description="Имя со страницы 1")
    page1_patronymic: Optional[str] = Field(None, description="Отчество со страницы 1")

    page2_surname: Optional[str] = Field(None, description="Фамилия со страницы 2 (адрес)")
    page2_name: Optional[str] = Field(None, description="Имя со страницы 2 (адрес)")
    page2_patronymic: Optional[str] = Field(None, description="Отчество со страницы 2 (адрес)")

    # Результаты сравнения
    surname_match: bool = Field(False, description="Фамилии совпадают")
    name_match: bool = Field(False, description="Имена совпадают")
    patronymic_match: bool = Field(False, description="Отчества совпадают")

    # Метрики схожести (расстояние Левенштейна нормализованное)
    surname_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Схожесть фамилий (0-1)")
    name_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Схожесть имен (0-1)")
    patronymic_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Схожесть отчеств (0-1)")

    # Общий результат
    full_match: bool = Field(False, description="Полное совпадение всех компонентов")
    overall_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Общая схожесть ФИО (0-1)")

    # Расхождения
    discrepancies: List[str] = Field(default_factory=list, description="Обнаруженные расхождения")
    warnings: List[str] = Field(default_factory=list, description="Предупреждения")

    class Config:
        json_schema_extra = {
            "example": {
                "page1_surname": "Иванов",
                "page1_name": "Иван",
                "page1_patronymic": "Иванович",
                "page2_surname": "Иванов",
                "page2_name": "Иван",
                "page2_patronymic": "Иванович",
                "surname_match": True,
                "name_match": True,
                "patronymic_match": True,
                "surname_similarity": 1.0,
                "name_similarity": 1.0,
                "patronymic_similarity": 1.0,
                "full_match": True,
                "overall_similarity": 1.0,
                "discrepancies": [],
                "warnings": []
            }
        }

    def is_acceptable_match(self, threshold: float = 0.85) -> bool:
        """
        Проверить, является ли совпадение приемлемым.

        Args:
            threshold: Порог схожести (по умолчанию 0.85)

        Returns:
            bool: True если схожесть выше порога
        """
        return self.overall_similarity >= threshold


class AddressParseRequest(BaseModel):
    """
    Запрос на парсинг адреса.
    """
    raw_address: str = Field(..., description="Исходный адрес для парсинга", min_length=10)
    extract_postal_code: bool = Field(True, description="Извлекать почтовый индекс")
    validate_completeness: bool = Field(True, description="Проверять полноту адреса")

    class Config:
        json_schema_extra = {
            "example": {
                "raw_address": "123456, г. Москва, ул. Ленина, д. 10, корп. 2, кв. 5",
                "extract_postal_code": True,
                "validate_completeness": True
            }
        }


class AddressParseResponse(BaseModel):
    """
    Ответ на запрос парсинга адреса.
    """
    success: bool = Field(..., description="Успешность парсинга")
    address: Optional[StructuredAddress] = Field(None, description="Структурированный адрес")
    formatted_address: Optional[str] = Field(None, description="Отформатированный адрес")
    errors: List[str] = Field(default_factory=list, description="Ошибки парсинга")
    warnings: List[str] = Field(default_factory=list, description="Предупреждения")
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "address": {
                    "region": "г. Москва",
                    "city": "Москва",
                    "street": "ул. Ленина",
                    "house": "10",
                    "building": "корп. 2",
                    "apartment": "кв. 5",
                    "postal_code": "123456",
                    "raw_address": "123456, г. Москва, ул. Ленина, д. 10, корп. 2, кв. 5",
                    "is_fully_parsed": True
                },
                "formatted_address": "123456, г. Москва, ул. Ленина, д. 10, корп. 2, кв. 5",
                "errors": [],
                "warnings": [],
                "processing_time_ms": 45.67
            }
        }


class NameSyncRequest(BaseModel):
    """
    Запрос на синхронизацию ФИО между страницами.
    """
    page1_surname: Optional[str] = Field(None, description="Фамилия со страницы 1")
    page1_name: Optional[str] = Field(None, description="Имя со страницы 1")
    page1_patronymic: Optional[str] = Field(None, description="Отчество со страницы 1")

    page2_surname: Optional[str] = Field(None, description="Фамилия со страницы 2")
    page2_name: Optional[str] = Field(None, description="Имя со страницы 2")
    page2_patronymic: Optional[str] = Field(None, description="Отчество со страницы 2")

    similarity_threshold: float = Field(
        0.85,
        ge=0.0,
        le=1.0,
        description="Порог схожести для определения совпадения"
    )


class NameSyncResponse(BaseModel):
    """
    Ответ на запрос синхронизации ФИО.
    """
    success: bool = Field(..., description="Успешность операции")
    comparison: Optional[NameComparisonResult] = Field(None, description="Результат сравнения")
    is_match: bool = Field(False, description="ФИО совпадают")
    errors: List[str] = Field(default_factory=list, description="Ошибки")
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах")


# Справочник сокращений для типов улиц (для нормализации)
STREET_TYPE_ABBREVIATIONS = {
    "улица": ["ул", "ул.", "улица"],
    "проспект": ["пр-кт", "просп", "пр", "проспект"],
    "переулок": ["пер", "пер.", "переулок"],
    "бульвар": ["бульв", "б-р", "бульвар"],
    "площадь": ["пл", "пл.", "площадь"],
    "шоссе": ["ш", "шоссе"],
    "набережная": ["наб", "наб.", "набережная"],
    "проезд": ["пр-д", "проезд"],
    "аллея": ["ал", "аллея"],
    "тупик": ["туп", "тупик"],
}

# Справочник сокращений для типов населенных пунктов
SETTLEMENT_TYPE_ABBREVIATIONS = {
    "город": ["г", "г.", "город"],
    "поселок": ["п", "пос", "пос.", "поселок"],
    "деревня": ["д", "дер", "дер.", "деревня"],
    "село": ["с", "с.", "село"],
    "станица": ["ст", "ст.", "станица"],
    "хутор": ["х", "хут", "хутор"],
    "рабочий поселок": ["рп", "р.п.", "рабочий поселок"],
    "поселок городского типа": ["пгт", "п.г.т.", "поселок городского типа"],
}

# Справочник сокращений для типов зданий
BUILDING_TYPE_ABBREVIATIONS = {
    "дом": ["д", "д.", "дом"],
    "корпус": ["корп", "корп.", "к", "к.", "корпус"],
    "строение": ["стр", "стр.", "с", "строение"],
    "литера": ["лит", "лит.", "литера"],
    "владение": ["влд", "влд.", "владение"],
    "квартира": ["кв", "кв.", "квартира"],
    "офис": ["оф", "оф.", "офис"],
    "помещение": ["пом", "пом.", "помещение"],
}
