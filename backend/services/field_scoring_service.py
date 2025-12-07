"""
Field Scoring Service
Выполняет многофакторный скоринг извлечённых полей паспорта.
Реализует голосование между провайдерами для выбора лучших значений.
"""

import re
import time
from typing import List, Dict, Optional
from statistics import mean

from backend.schemas.passport_fields import (
    PassportFieldType,
    ExtractedField,
    FieldScore,
    ProviderExtraction,
    FieldVotingResult,
    PassportFieldsResult,
    ScoringWeights,
    PASSPORT_REGEX_PATTERNS,
    EXPECTED_FIELD_LENGTHS,
)
from backend.core.logger import logger


class FieldScoringService:
    """
    Сервис для скоринга и голосования по полям паспорта.
    Оценивает качество извлечённых полей и выбирает лучшие значения.
    """
    
    def __init__(self):
        """Инициализация сервиса"""
        self.patterns = PASSPORT_REGEX_PATTERNS
        self.expected_lengths = EXPECTED_FIELD_LENGTHS
        logger.info("Field Scoring Service initialized")
    
    def score_and_vote(
        self,
        provider_extractions: List[ProviderExtraction]
    ) -> PassportFieldsResult:
        """
        Выполнить скоринг и голосование по всем полям.
        
        Args:
            provider_extractions: Результаты извлечения от всех провайдеров
            
        Returns:
            PassportFieldsResult: Финальный результат с выбранными значениями
        """
        start_time = time.time()
        
        logger.info(f"Начало скоринга и голосования. Провайдеров: {len(provider_extractions)}")
        
        if not provider_extractions:
            logger.warning("Нет данных от провайдеров для скоринга")
            return self._create_empty_result()
        
        # Группируем поля по типам
        fields_by_type = self._group_fields_by_type(provider_extractions)
        
        # Выполняем голосование по каждому полю
        field_results: Dict[str, FieldVotingResult] = {}
        final_values: Dict[str, Optional[str]] = {}
        field_confidences: Dict[str, float] = {}
        
        for field_type, extracted_fields in fields_by_type.items():
            # Скоринг каждого кандидата
            scored_fields = [
                self._score_field(field)
                for field in extracted_fields
            ]
            
            # Голосование
            voting_result = self._vote_for_best_field(field_type, scored_fields)
            
            field_type_key = field_type if isinstance(field_type, str) else field_type.value
            field_results[field_type_key] = voting_result
            final_values[field_type_key] = voting_result.selected_value
            field_confidences[field_type_key] = voting_result.confidence
        
        # Вычисляем общую статистику
        total_providers = len(provider_extractions)
        avg_confidence = mean(field_confidences.values()) if field_confidences else 0.0
        high_conf_count = sum(1 for c in field_confidences.values() if c > 0.8)
        low_conf_count = sum(1 for c in field_confidences.values() if c < 0.5)
        
        # Валидация
        validation_errors = self._validate_fields(final_values)
        is_valid = len(validation_errors) == 0
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        result = PassportFieldsResult(
            provider_extractions=provider_extractions,
            field_results=field_results,
            final_values=final_values,
            field_confidences=field_confidences,
            total_providers_used=total_providers,
            average_confidence=avg_confidence,
            fields_with_high_confidence=high_conf_count,
            fields_with_low_confidence=low_conf_count,
            is_valid=is_valid,
            validation_errors=validation_errors
        )
        
        logger.info(
            f"Скоринг завершён за {processing_time_ms:.2f}ms. "
            f"Извлечено полей: {len(final_values)}, "
            f"Средняя уверенность: {avg_confidence:.2%}, "
            f"Валидность: {is_valid}"
        )
        
        return result
    
    def _group_fields_by_type(
        self,
        provider_extractions: List[ProviderExtraction]
    ) -> Dict[PassportFieldType, List[ExtractedField]]:
        """
        Сгруппировать извлечённые поля по типам.
        
        Returns:
            Dict[PassportFieldType, List[ExtractedField]]: Поля сгруппированные по типам
        """
        grouped: Dict[PassportFieldType, List[ExtractedField]] = {}
        
        for extraction in provider_extractions:
            for field in extraction.fields:
                if field.field_type not in grouped:
                    grouped[field.field_type] = []
                grouped[field.field_type].append(field)
        
        return grouped
    
    def _score_field(self, field: ExtractedField) -> FieldScore:
        """
        Выполнить многофакторный скоринг одного поля.
        
        Args:
            field: Извлечённое поле
            
        Returns:
            FieldScore: Оценка поля
        """
        # Выбираем веса в зависимости от типа поля
        weights = self._get_weights_for_field(field.field_type)
        
        # Базовая уверенность от OCR
        ocr_confidence = field.confidence
        
        # Regex matching score
        regex_score = self._calculate_regex_score(field)
        
        # Cyrillic ratio (для текстовых полей)
        cyrillic_ratio = self._calculate_cyrillic_ratio(field)
        
        # Length score
        length_score = self._calculate_length_score(field)
        
        # Format score (для дат и кодов)
        format_score = self._calculate_format_score(field)
        
        # Итоговый скор (взвешенная сумма)
        total_score = (
            ocr_confidence * weights.ocr_confidence_weight +
            regex_score * weights.regex_match_weight +
            cyrillic_ratio * weights.cyrillic_ratio_weight +
            length_score * weights.length_score_weight +
            format_score * weights.format_score_weight
        )
        
        # Детализация для отладки
        scoring_details = {
            "ocr_confidence": ocr_confidence,
            "regex_score": regex_score,
            "cyrillic_ratio": cyrillic_ratio,
            "length_score": length_score,
            "format_score": format_score,
            "weights": {
                "ocr": weights.ocr_confidence_weight,
                "regex": weights.regex_match_weight,
                "cyrillic": weights.cyrillic_ratio_weight,
                "length": weights.length_score_weight,
                "format": weights.format_score_weight,
            }
        }
        
        return FieldScore(
            field_type=field.field_type,
            value=field.value,
            ocr_confidence=ocr_confidence,
            regex_match_score=regex_score,
            cyrillic_ratio=cyrillic_ratio,
            length_score=length_score,
            format_score=format_score,
            total_score=total_score,
            source_provider=field.source_provider,
            scoring_details=scoring_details
        )
    
    def _get_weights_for_field(self, field_type: PassportFieldType) -> ScoringWeights:
        """Получить веса для конкретного типа поля"""
        # Для номеров и кодов - строгие regex
        if field_type in [
            PassportFieldType.SERIES,
            PassportFieldType.NUMBER,
            PassportFieldType.DEPARTMENT_CODE
        ]:
            return ScoringWeights.strict_regex_weights()
        
        # Для текстовых полей - качество текста
        if field_type in [
            PassportFieldType.SURNAME,
            PassportFieldType.NAME,
            PassportFieldType.PATRONYMIC,
            PassportFieldType.BIRTH_PLACE,
            PassportFieldType.ISSUED_BY,
            PassportFieldType.REGISTRATION_ADDRESS
        ]:
            return ScoringWeights.text_quality_weights()
        
        # Для остальных - стандартные веса
        return ScoringWeights.default_weights()
    
    def _calculate_regex_score(self, field: ExtractedField) -> float:
        """Оценка соответствия regex-паттерну"""
        if not field.value:
            return 0.0
        
        # Проверяем наличие паттерна для данного типа поля
        if field.field_type not in self.patterns:
            return 0.5  # Нейтральная оценка если нет паттерна
        
        pattern = self.patterns[field.field_type].pattern
        
        if re.match(pattern, field.value):
            return 1.0
        else:
            return 0.0
    
    def _calculate_cyrillic_ratio(self, field: ExtractedField) -> float:
        """Доля кириллических символов в тексте"""
        if not field.value:
            return 0.0
        
        # Для числовых полей кириллица не нужна
        if field.field_type in [
            PassportFieldType.SERIES,
            PassportFieldType.NUMBER,
            PassportFieldType.BIRTH_DATE,
            PassportFieldType.ISSUE_DATE,
            PassportFieldType.DEPARTMENT_CODE
        ]:
            return 1.0  # Максимальный скор (не применимо)
        
        # Для пола - проверяем только М/Ж
        if field.field_type == PassportFieldType.GENDER:
            return 1.0 if field.value in ['М', 'Ж'] else 0.0
        
        # Подсчёт кириллических символов
        total_chars = len(field.value)
        cyrillic_chars = len(re.findall(r'[А-Яа-яЁё]', field.value))
        
        if total_chars == 0:
            return 0.0
        
        return cyrillic_chars / total_chars
    
    def _calculate_length_score(self, field: ExtractedField) -> float:
        """Оценка соответствия ожидаемой длине"""
        if not field.value:
            return 0.0
        
        # Проверяем наличие ожидаемой длины для данного типа поля
        if field.field_type not in self.expected_lengths:
            return 0.5  # Нейтральная оценка
        
        min_len, max_len = self.expected_lengths[field.field_type]
        actual_len = len(field.value)
        
        # Идеальная длина
        if min_len <= actual_len <= max_len:
            return 1.0
        
        # Небольшое отклонение
        if min_len - 2 <= actual_len <= max_len + 2:
            return 0.7
        
        # Значительное отклонение
        if min_len - 5 <= actual_len <= max_len + 5:
            return 0.4
        
        # Слишком большое отклонение
        return 0.0
    
    def _calculate_format_score(self, field: ExtractedField) -> float:
        """Оценка корректности формата (даты, коды)"""
        if not field.value:
            return 0.0
        
        # Для дат - проверяем валидность
        if field.field_type in [PassportFieldType.BIRTH_DATE, PassportFieldType.ISSUE_DATE]:
            return self._validate_date_format(field.value)
        
        # Для кода подразделения - проверяем формат XXX-XXX
        if field.field_type == PassportFieldType.DEPARTMENT_CODE:
            if re.match(r'^\d{3}-\d{3}$', field.value):
                return 1.0
            return 0.0
        
        # Для остальных полей - нейтральная оценка
        return 0.5
    
    def _validate_date_format(self, date_str: str) -> float:
        """Валидация формата даты ДД.ММ.ГГГГ"""
        if not re.match(r'^\d{2}\.\d{2}\.\d{4}$', date_str):
            return 0.0
        
        try:
            parts = date_str.split('.')
            day = int(parts[0])
            month = int(parts[1])
            year = int(parts[2])
            
            # Базовая валидация диапазонов
            if not (1 <= day <= 31):
                return 0.0
            if not (1 <= month <= 12):
                return 0.0
            if not (1900 <= year <= 2100):
                return 0.0
            
            return 1.0
        except (ValueError, IndexError):
            return 0.0
    
    def _vote_for_best_field(
        self,
        field_type: PassportFieldType,
        scored_fields: List[FieldScore]
    ) -> FieldVotingResult:
        """
        Выбрать лучшее значение поля на основе скоринга.
        
        Args:
            field_type: Тип поля
            scored_fields: Оценённые варианты от разных провайдеров
            
        Returns:
            FieldVotingResult: Результат голосования
        """
        if not scored_fields:
            return FieldVotingResult(
                field_type=field_type,
                selected_value=None,
                confidence=0.0,
                candidates=[],
                winner_provider=None,
                winner_score=0.0,
                total_candidates=0,
                agreement_score=0.0
            )
        
        # Сортируем по итоговому скору (от большего к меньшему)
        sorted_fields = sorted(scored_fields, key=lambda f: f.total_score, reverse=True)
        
        # Победитель - поле с наибольшим скором
        winner = sorted_fields[0]
        
        # Вычисляем согласованность провайдеров
        # Если несколько провайдеров дали одинаковое значение - согласованность выше
        agreement_score = self._calculate_agreement_score(scored_fields)
        
        return FieldVotingResult(
            field_type=field_type,
            selected_value=winner.value,
            confidence=winner.total_score,
            candidates=sorted_fields,
            winner_provider=winner.source_provider,
            winner_score=winner.total_score,
            total_candidates=len(scored_fields),
            agreement_score=agreement_score
        )
    
    def _calculate_agreement_score(self, scored_fields: List[FieldScore]) -> float:
        """
        Вычислить согласованность между провайдерами.
        
        Returns:
            float: Оценка согласованности (0-1)
        """
        if len(scored_fields) <= 1:
            return 1.0  # Полная согласованность если один вариант
        
        # Подсчитываем, сколько провайдеров дали одинаковое значение
        value_counts: Dict[str, int] = {}
        for field in scored_fields:
            if field.value:
                value_counts[field.value] = value_counts.get(field.value, 0) + 1
        
        if not value_counts:
            return 0.0
        
        # Максимальное количество совпадений
        max_count = max(value_counts.values())
        total_count = len(scored_fields)
        
        # Согласованность = доля провайдеров с самым популярным значением
        return max_count / total_count
    
    def _validate_fields(self, final_values: Dict[str, Optional[str]]) -> List[str]:
        """
        Валидация финальных значений полей.
        
        Returns:
            List[str]: Список ошибок валидации
        """
        errors = []
        
        # Обязательные поля
        required_fields = [
            PassportFieldType.SERIES,
            PassportFieldType.NUMBER,
            PassportFieldType.SURNAME,
            PassportFieldType.NAME,
        ]
        
        for field_type in required_fields:
            if not final_values.get(field_type.value):
                errors.append(f"Обязательное поле '{field_type.value}' не найдено")
        
        # Валидация формата серии
        series = final_values.get(PassportFieldType.SERIES.value)
        if series and not re.match(r'^\d{4}$', series):
            errors.append(f"Некорректный формат серии: {series}")
        
        # Валидация формата номера
        number = final_values.get(PassportFieldType.NUMBER.value)
        if number and not re.match(r'^\d{6}$', number):
            errors.append(f"Некорректный формат номера: {number}")
        
        # Валидация дат
        birth_date = final_values.get(PassportFieldType.BIRTH_DATE.value)
        if birth_date and self._validate_date_format(birth_date) == 0.0:
            errors.append(f"Некорректная дата рождения: {birth_date}")
        
        issue_date = final_values.get(PassportFieldType.ISSUE_DATE.value)
        if issue_date and self._validate_date_format(issue_date) == 0.0:
            errors.append(f"Некорректная дата выдачи: {issue_date}")
        
        return errors
    
    def _create_empty_result(self) -> PassportFieldsResult:
        """Создать пустой результат"""
        return PassportFieldsResult(
            provider_extractions=[],
            field_results={},
            final_values={},
            field_confidences={},
            total_providers_used=0,
            average_confidence=0.0,
            fields_with_high_confidence=0,
            fields_with_low_confidence=0,
            is_valid=False,
            validation_errors=["Нет данных от провайдеров"]
        )


# Singleton instance
field_scoring_service = FieldScoringService()
