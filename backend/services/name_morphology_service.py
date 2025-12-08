"""
Name Morphology Service
Морфологический анализ и валидация ФИО с использованием словарей и pymorphy3.
"""

import json
import os
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import pymorphy3
from Levenshtein import distance as levenshtein_distance

from backend.core.logger import logger
from backend.services.text_normalization_service import text_normalization_service


class NameMorphologyService:
    """
    Сервис морфологического анализа и валидации ФИО.
    Использует pymorphy3 для морфологии и словари для проверки.
    """
    
    def __init__(self):
        """Инициализация сервиса"""
        self.morph = None
        self.surnames_dict = set()
        self.names_dict = set()
        self.patronymics_dict = set()
        self._initialized = False
        logger.info("Name Morphology Service создан")
    
    def initialize(self) -> None:
        """Инициализация морфологического анализатора и загрузка словарей"""
        if self._initialized:
            logger.info("Name Morphology Service уже инициализирован")
            return
        
        logger.info("Инициализация Name Morphology Service...")
        
        # Инициализация pymorphy3
        try:
            self.morph = pymorphy3.MorphAnalyzer()
            logger.info("Pymorphy2 инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации pymorphy3: {e}")
            raise
        
        # Загрузка словарей
        try:
            self._load_dictionaries()
            logger.info(
                f"Словари загружены: "
                f"фамилий={len(self.surnames_dict)}, "
                f"имен={len(self.names_dict)}, "
                f"отчеств={len(self.patronymics_dict)}"
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки словарей: {e}")
            raise
        
        self._initialized = True
        logger.info("Name Morphology Service инициализирован успешно")
    
    def _load_dictionaries(self) -> None:
        """Загрузка JSON-словарей с типовыми ФИО"""
        # Путь к директории со словарями
        dict_dir = Path(__file__).parent.parent / "dictionaries"
        
        # Загрузка фамилий
        surnames_file = dict_dir / "surnames.json"
        if surnames_file.exists():
            with open(surnames_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.surnames_dict = set(data.get('surnames', []))
        else:
            logger.warning(f"Файл словаря фамилий не найден: {surnames_file}")
        
        # Загрузка имен
        names_file = dict_dir / "names.json"
        if names_file.exists():
            with open(names_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.names_dict = set(data.get('names', []))
        else:
            logger.warning(f"Файл словаря имен не найден: {names_file}")
        
        # Загрузка отчеств
        patronymics_file = dict_dir / "patronymics.json"
        if patronymics_file.exists():
            with open(patronymics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.patronymics_dict = set(data.get('patronymics', []))
        else:
            logger.warning(f"Файл словаря отчеств не найден: {patronymics_file}")
    
    def analyze_name_part(self, text: str) -> Dict[str, any]:
        """
        Морфологический анализ части ФИО.
        
        Args:
            text: Часть ФИО (фамилия, имя или отчество)
            
        Returns:
            Dict: Результат анализа с морфологической информацией
        """
        if not self._initialized:
            self.initialize()
        
        if not text:
            return {
                'original': '',
                'normalized': '',
                'is_name': False,
                'gender': None,
                'case': None,
                'confidence': 0.0
            }
        
        # Нормализация
        normalized = text_normalization_service.normalize_fio_part(text)
        
        # Морфологический разбор
        parsed = self.morph.parse(normalized)[0]
        
        # Определение типа (имя собственное?)
        is_name = 'Name' in parsed.tag
        
        # Определение пола
        gender = None
        if 'masc' in parsed.tag:
            gender = 'М'
        elif 'femn' in parsed.tag:
            gender = 'Ж'
        
        # Определение падежа
        case = None
        if 'nomn' in parsed.tag:
            case = 'nominative'  # именительный
        elif 'gent' in parsed.tag:
            case = 'genitive'    # родительный
        
        return {
            'original': text,
            'normalized': normalized,
            'is_name': is_name,
            'gender': gender,
            'case': case,
            'confidence': 0.8 if is_name else 0.3,
            'normal_form': parsed.normal_form
        }
    
    def validate_surname(self, surname: str) -> Tuple[bool, float, Optional[str]]:
        """
        Валидация фамилии.
        
        Args:
            surname: Фамилия для проверки
            
        Returns:
            Tuple[bool, float, Optional[str]]: (валидна, уверенность, исправленная версия)
        """
        if not self._initialized:
            self.initialize()
        
        if not surname:
            return False, 0.0, None
        
        normalized = text_normalization_service.normalize_fio_part(surname)
        
        # Точное совпадение
        if normalized in self.surnames_dict:
            return True, 1.0, normalized
        
        # Нечеткий поиск (для исправления OCR-ошибок)
        best_match, confidence = self._fuzzy_match(normalized, self.surnames_dict)
        
        if confidence > 0.85:
            return True, confidence, best_match
        elif confidence > 0.7:
            return True, confidence, best_match  # Вероятное совпадение
        else:
            # Морфологическая проверка
            analysis = self.analyze_name_part(normalized)
            if analysis['is_name']:
                return True, 0.6, normalized
            else:
                return False, confidence, best_match
    
    def validate_name(self, name: str) -> Tuple[bool, float, Optional[str]]:
        """
        Валидация имени.
        
        Args:
            name: Имя для проверки
            
        Returns:
            Tuple[bool, float, Optional[str]]: (валидно, уверенность, исправленная версия)
        """
        if not self._initialized:
            self.initialize()
        
        if not name:
            return False, 0.0, None
        
        normalized = text_normalization_service.normalize_fio_part(name)
        
        # Точное совпадение
        if normalized in self.names_dict:
            return True, 1.0, normalized
        
        # Нечеткий поиск
        best_match, confidence = self._fuzzy_match(normalized, self.names_dict)
        
        if confidence > 0.85:
            return True, confidence, best_match
        elif confidence > 0.7:
            return True, confidence, best_match
        else:
            # Морфологическая проверка
            analysis = self.analyze_name_part(normalized)
            if analysis['is_name']:
                return True, 0.6, normalized
            else:
                return False, confidence, best_match
    
    def validate_patronymic(self, patronymic: str) -> Tuple[bool, float, Optional[str]]:
        """
        Валидация отчества.
        
        Args:
            patronymic: Отчество для проверки
            
        Returns:
            Tuple[bool, float, Optional[str]]: (валидно, уверенность, исправленная версия)
        """
        if not self._initialized:
            self.initialize()
        
        if not patronymic:
            return False, 0.0, None
        
        normalized = text_normalization_service.normalize_fio_part(patronymic)
        
        # Точное совпадение
        if normalized in self.patronymics_dict:
            return True, 1.0, normalized
        
        # Нечеткий поиск
        best_match, confidence = self._fuzzy_match(normalized, self.patronymics_dict)
        
        if confidence > 0.85:
            return True, confidence, best_match
        elif confidence > 0.7:
            return True, confidence, best_match
        else:
            # Проверка на типичные окончания отчеств
            if self._has_patronymic_ending(normalized):
                return True, 0.7, normalized
            else:
                return False, confidence, best_match
    
    def _fuzzy_match(self, text: str, dictionary: set, max_distance: int = 2) -> Tuple[Optional[str], float]:
        """
        Нечеткий поиск совпадения в словаре (исправление OCR-ошибок).
        
        Args:
            text: Текст для поиска
            dictionary: Словарь для поиска
            max_distance: Максимальное расстояние Левенштейна
            
        Returns:
            Tuple[Optional[str], float]: (лучшее совпадение, уверенность)
        """
        if not text or not dictionary:
            return None, 0.0
        
        best_match = None
        min_distance = float('inf')
        
        for word in dictionary:
            dist = levenshtein_distance(text.lower(), word.lower())
            if dist < min_distance:
                min_distance = dist
                best_match = word
        
        # Расчет уверенности на основе расстояния
        if min_distance == 0:
            confidence = 1.0
        elif min_distance <= max_distance:
            # Чем меньше расстояние, тем выше уверенность
            confidence = 1.0 - (min_distance / (max_distance + 1))
        else:
            confidence = 0.0
        
        return best_match, confidence
    
    def _has_patronymic_ending(self, text: str) -> bool:
        """
        Проверка на типичные окончания отчеств.
        
        Args:
            text: Текст для проверки
            
        Returns:
            bool: True если есть типичное окончание отчества
        """
        patronymic_endings = [
            'ович', 'евич', 'ич',      # мужские
            'овна', 'евна', 'ична'     # женские
        ]
        
        text_lower = text.lower()
        return any(text_lower.endswith(ending) for ending in patronymic_endings)
    
    def validate_full_name(
        self,
        surname: Optional[str],
        name: Optional[str],
        patronymic: Optional[str]
    ) -> Dict[str, any]:
        """
        Комплексная валидация полного ФИО.
        
        Args:
            surname: Фамилия
            name: Имя
            patronymic: Отчество
            
        Returns:
            Dict: Результат валидации с исправлениями и уверенностью
        """
        if not self._initialized:
            self.initialize()
        
        result = {
            'is_valid': True,
            'confidence': 0.0,
            'errors': [],
            'corrected': {}
        }
        
        confidences = []
        
        # Валидация фамилии
        if surname:
            is_valid, conf, corrected = self.validate_surname(surname)
            confidences.append(conf)
            if corrected and corrected != surname:
                result['corrected']['surname'] = corrected
            if not is_valid:
                result['errors'].append(f"Фамилия '{surname}' не распознана")
        
        # Валидация имени
        if name:
            is_valid, conf, corrected = self.validate_name(name)
            confidences.append(conf)
            if corrected and corrected != name:
                result['corrected']['name'] = corrected
            if not is_valid:
                result['errors'].append(f"Имя '{name}' не распознано")
        
        # Валидация отчества
        if patronymic:
            is_valid, conf, corrected = self.validate_patronymic(patronymic)
            confidences.append(conf)
            if corrected and corrected != patronymic:
                result['corrected']['patronymic'] = corrected
            if not is_valid:
                result['errors'].append(f"Отчество '{patronymic}' не распознано")
        
        # Общая уверенность - среднее от всех компонентов
        if confidences:
            result['confidence'] = sum(confidences) / len(confidences)
        
        result['is_valid'] = len(result['errors']) == 0
        
        return result
    
    def extract_gender_from_name(
        self,
        surname: Optional[str],
        name: Optional[str],
        patronymic: Optional[str]
    ) -> Optional[str]:
        """
        Определение пола по ФИО.
        
        Args:
            surname: Фамилия
            name: Имя
            patronymic: Отчество
            
        Returns:
            Optional[str]: 'М' или 'Ж', или None если не удалось определить
        """
        if not self._initialized:
            self.initialize()
        
        # Приоритет определения: отчество > имя > фамилия
        
        # 1. По отчеству (самый надежный способ)
        if patronymic:
            if patronymic.lower().endswith(('ович', 'евич', 'ич')):
                return 'М'
            elif patronymic.lower().endswith(('овна', 'евна', 'ична')):
                return 'Ж'
        
        # 2. По имени
        if name:
            analysis = self.analyze_name_part(name)
            if analysis['gender']:
                return analysis['gender']
        
        # 3. По фамилии (менее надежно)
        if surname:
            if surname.lower().endswith('ова') or surname.lower().endswith('ева'):
                return 'Ж'
            elif surname.lower().endswith('ов') or surname.lower().endswith('ев'):
                return 'М'
        
        return None


# Singleton instance
name_morphology_service = NameMorphologyService()
