"""
Field Extraction Service
Извлекает структурированные поля паспорта из OCR результатов.
Использует regex-паттерны и эвристики для поиска данных.
"""

import re
import time
from typing import List, Optional, Dict
from datetime import datetime

from backend.schemas.ocr import OCRResult, OCRBlock, OCRLine, OCRWord
from backend.schemas.passport_fields import (
    PassportFieldType,
    ExtractedField,
    ProviderExtraction,
    PASSPORT_REGEX_PATTERNS,
)
from backend.core.logger import logger


class FieldExtractionService:
    """
    Сервис для извлечения полей паспорта из OCR результатов.
    """
    
    def __init__(self):
        """Инициализация сервиса"""
        self.patterns = PASSPORT_REGEX_PATTERNS
        logger.info("Field Extraction Service initialized")
    
    def extract_fields(self, ocr_result: OCRResult) -> ProviderExtraction:
        """
        Извлечь все поля паспорта из OCR результата.
        
        Args:
            ocr_result: Результат OCR от провайдера
            
        Returns:
            ProviderExtraction: Извлечённые поля с метаданными
        """
        start_time = time.time()
        
        logger.info(f"Начало извлечения полей от провайдера: {ocr_result.metadata.provider}")
        
        # Извлекаем все токены из OCR результата
        tokens = self._extract_tokens(ocr_result)
        full_text = ocr_result.full_text
        
        # Извлекаем каждое поле
        extracted_fields: List[ExtractedField] = []
        
        # Серия паспорта
        series = self._extract_series(tokens, full_text, ocr_result)
        if series:
            extracted_fields.append(series)
        
        # Номер паспорта
        number = self._extract_number(tokens, full_text, ocr_result)
        if number:
            extracted_fields.append(number)
        
        # Фамилия, Имя, Отчество
        fio_fields = self._extract_fio(tokens, full_text, ocr_result)
        extracted_fields.extend(fio_fields)
        
        # Даты
        dates = self._extract_dates(tokens, full_text, ocr_result)
        extracted_fields.extend(dates)
        
        # Код подразделения
        dept_code = self._extract_department_code(tokens, full_text, ocr_result)
        if dept_code:
            extracted_fields.append(dept_code)
        
        # Пол
        gender = self._extract_gender(tokens, full_text, ocr_result)
        if gender:
            extracted_fields.append(gender)
        
        # Место рождения
        birth_place = self._extract_birth_place(tokens, full_text, ocr_result)
        if birth_place:
            extracted_fields.append(birth_place)
        
        # Кем выдан
        issued_by = self._extract_issued_by(tokens, full_text, ocr_result)
        if issued_by:
            extracted_fields.append(issued_by)
        
        # Адрес регистрации
        address = self._extract_registration_address(tokens, full_text, ocr_result)
        if address:
            extracted_fields.append(address)
        
        extraction_time_ms = (time.time() - start_time) * 1000
        
        result = ProviderExtraction(
            provider_name=ocr_result.metadata.provider,
            fields=extracted_fields,
            extraction_time_ms=extraction_time_ms,
            total_fields_found=len(extracted_fields)
        )
        
        logger.info(
            f"Извлечено {len(extracted_fields)} полей от {ocr_result.metadata.provider} "
            f"за {extraction_time_ms:.2f}ms"
        )
        
        return result
    
    def _extract_tokens(self, ocr_result: OCRResult) -> List[Dict]:
        """
        Извлечь все токены (слова) из OCR результата.
        
        Returns:
            List[Dict]: Список словарей с токенами и их метаданными
        """
        tokens = []
        
        for block in ocr_result.blocks:
            for line in block.lines:
                for word in line.words:
                    tokens.append({
                        'text': word.text,
                        'confidence': word.confidence,
                        'bbox': word.bounding_box,
                    })
        
        return tokens
    
    def _extract_series(
        self,
        tokens: List[Dict],
        full_text: str,
        ocr_result: OCRResult
    ) -> Optional[ExtractedField]:
        """Извлечь серию паспорта (4 цифры)"""
        pattern = self.patterns[PassportFieldType.SERIES].pattern
        
        # Ищем в токенах
        for token in tokens:
            text = token['text'].strip()
            if re.match(pattern, text):
                return ExtractedField(
                    field_type=PassportFieldType.SERIES,
                    value=text,
                    raw_value=text,
                    confidence=token['confidence'],
                    source_provider=ocr_result.metadata.provider
                )
        
        # Ищем в полном тексте
        matches = re.findall(r'\b(\d{4})\b', full_text)
        if matches:
            # Берём первое совпадение (обычно серия идёт первой)
            return ExtractedField(
                field_type=PassportFieldType.SERIES,
                value=matches[0],
                raw_value=matches[0],
                confidence=ocr_result.average_confidence,
                source_provider=ocr_result.metadata.provider
            )
        
        return None
    
    def _extract_number(
        self,
        tokens: List[Dict],
        full_text: str,
        ocr_result: OCRResult
    ) -> Optional[ExtractedField]:
        """Извлечь номер паспорта (6 цифр)"""
        pattern = self.patterns[PassportFieldType.NUMBER].pattern
        
        # Ищем в токенах
        for token in tokens:
            text = token['text'].strip()
            if re.match(pattern, text):
                return ExtractedField(
                    field_type=PassportFieldType.NUMBER,
                    value=text,
                    raw_value=text,
                    confidence=token['confidence'],
                    source_provider=ocr_result.metadata.provider
                )
        
        # Ищем в полном тексте
        matches = re.findall(r'\b(\d{6})\b', full_text)
        if matches:
            # Берём первое совпадение
            return ExtractedField(
                field_type=PassportFieldType.NUMBER,
                value=matches[0],
                raw_value=matches[0],
                confidence=ocr_result.average_confidence,
                source_provider=ocr_result.metadata.provider
            )
        
        return None
    
    def _extract_fio(
        self,
        tokens: List[Dict],
        full_text: str,
        ocr_result: OCRResult
    ) -> List[ExtractedField]:
        """Извлечь ФИО (фамилия, имя, отчество)"""
        fields = []
        
        # Паттерны для ФИО
        surname_pattern = self.patterns[PassportFieldType.SURNAME].pattern
        name_pattern = self.patterns[PassportFieldType.NAME].pattern
        patronymic_pattern = self.patterns[PassportFieldType.PATRONYMIC].pattern
        
        # Ключевые слова для поиска ФИО
        fio_keywords = ['Фамилия', 'Имя', 'Отчество']
        
        # Ищем в токенах с контекстом
        cyrillic_tokens = [
            t for t in tokens 
            if re.search(r'[А-Яа-яЁё]', t['text'])
        ]
        
        for i, token in enumerate(cyrillic_tokens):
            text = token['text'].strip()
            
            # Фамилия
            if re.match(surname_pattern, text) and len(text) > 2:
                # Проверяем, что не является ключевым словом
                if text not in fio_keywords:
                    fields.append(ExtractedField(
                        field_type=PassportFieldType.SURNAME,
                        value=text,
                        raw_value=text,
                        confidence=token['confidence'],
                        source_provider=ocr_result.metadata.provider
                    ))
                    break  # Берём только первую найденную фамилию
        
        for i, token in enumerate(cyrillic_tokens):
            text = token['text'].strip()
            
            # Имя
            if re.match(name_pattern, text) and len(text) > 2:
                if text not in fio_keywords and not any(f.value == text for f in fields):
                    fields.append(ExtractedField(
                        field_type=PassportFieldType.NAME,
                        value=text,
                        raw_value=text,
                        confidence=token['confidence'],
                        source_provider=ocr_result.metadata.provider
                    ))
                    break
        
        for i, token in enumerate(cyrillic_tokens):
            text = token['text'].strip()
            
            # Отчество
            if re.match(patronymic_pattern, text):
                fields.append(ExtractedField(
                    field_type=PassportFieldType.PATRONYMIC,
                    value=text,
                    raw_value=text,
                    confidence=token['confidence'],
                    source_provider=ocr_result.metadata.provider
                ))
                break
        
        return fields
    
    def _extract_dates(
        self,
        tokens: List[Dict],
        full_text: str,
        ocr_result: OCRResult
    ) -> List[ExtractedField]:
        """Извлечь даты (рождения и выдачи)"""
        fields = []
        
        # Паттерн для дат ДД.ММ.ГГГГ
        date_pattern = r'\b(\d{2}\.\d{2}\.\d{4})\b'
        
        # Находим все даты в тексте
        dates = re.findall(date_pattern, full_text)
        
        if len(dates) >= 1:
            # Первая дата обычно - дата рождения
            fields.append(ExtractedField(
                field_type=PassportFieldType.BIRTH_DATE,
                value=dates[0],
                raw_value=dates[0],
                confidence=ocr_result.average_confidence,
                source_provider=ocr_result.metadata.provider
            ))
        
        if len(dates) >= 2:
            # Вторая дата обычно - дата выдачи
            fields.append(ExtractedField(
                field_type=PassportFieldType.ISSUE_DATE,
                value=dates[1],
                raw_value=dates[1],
                confidence=ocr_result.average_confidence,
                source_provider=ocr_result.metadata.provider
            ))
        
        return fields
    
    def _extract_department_code(
        self,
        tokens: List[Dict],
        full_text: str,
        ocr_result: OCRResult
    ) -> Optional[ExtractedField]:
        """Извлечь код подразделения (XXX-XXX)"""
        pattern = self.patterns[PassportFieldType.DEPARTMENT_CODE].pattern
        
        # Ищем в полном тексте
        match = re.search(pattern, full_text)
        if match:
            return ExtractedField(
                field_type=PassportFieldType.DEPARTMENT_CODE,
                value=match.group(0),
                raw_value=match.group(0),
                confidence=ocr_result.average_confidence,
                source_provider=ocr_result.metadata.provider
            )
        
        return None
    
    def _extract_gender(
        self,
        tokens: List[Dict],
        full_text: str,
        ocr_result: OCRResult
    ) -> Optional[ExtractedField]:
        """Извлечь пол (М/Ж)"""
        pattern = self.patterns[PassportFieldType.GENDER].pattern
        
        # Ищем в токенах
        for token in tokens:
            text = token['text'].strip()
            if re.match(pattern, text):
                return ExtractedField(
                    field_type=PassportFieldType.GENDER,
                    value=text,
                    raw_value=text,
                    confidence=token['confidence'],
                    source_provider=ocr_result.metadata.provider
                )
        
        # Ищем в полном тексте
        if 'М' in full_text or 'Ж' in full_text:
            # Берём первое вхождение
            for char in full_text:
                if char in ['М', 'Ж']:
                    return ExtractedField(
                        field_type=PassportFieldType.GENDER,
                        value=char,
                        raw_value=char,
                        confidence=ocr_result.average_confidence,
                        source_provider=ocr_result.metadata.provider
                    )
        
        return None
    
    def _extract_birth_place(
        self,
        tokens: List[Dict],
        full_text: str,
        ocr_result: OCRResult
    ) -> Optional[ExtractedField]:
        """Извлечь место рождения"""
        # Ищем ключевое слово и берём текст после него
        keywords = ['МЕСТО РОЖДЕНИЯ', 'МЕСТО РОЖД', 'МЕС РОЖ']
        
        for keyword in keywords:
            if keyword in full_text.upper():
                # Находим позицию и берём следующую строку
                idx = full_text.upper().find(keyword)
                after_keyword = full_text[idx + len(keyword):].strip()
                
                # Берём до следующего ключевого слова или до конца строки
                lines = after_keyword.split('\n')
                if lines:
                    place = lines[0].strip()
                    if len(place) > 5:  # Минимальная длина для валидного места
                        return ExtractedField(
                            field_type=PassportFieldType.BIRTH_PLACE,
                            value=place,
                            raw_value=place,
                            confidence=ocr_result.average_confidence * 0.8,  # Понижаем confidence
                            source_provider=ocr_result.metadata.provider
                        )
        
        return None
    
    def _extract_issued_by(
        self,
        tokens: List[Dict],
        full_text: str,
        ocr_result: OCRResult
    ) -> Optional[ExtractedField]:
        """Извлечь информацию о выдавшем органе"""
        # Ищем ключевое слово и берём текст после него
        keywords = ['КЕМ ВЫДАН', 'ВЫДАН', 'ОТДЕЛЕНИЕМ']
        
        for keyword in keywords:
            if keyword in full_text.upper():
                idx = full_text.upper().find(keyword)
                after_keyword = full_text[idx + len(keyword):].strip()
                
                # Берём несколько строк (орган может занимать 2-3 строки)
                lines = after_keyword.split('\n')[:3]
                issued_by = ' '.join([line.strip() for line in lines if line.strip()])
                
                if len(issued_by) > 10:  # Минимальная длина
                    return ExtractedField(
                        field_type=PassportFieldType.ISSUED_BY,
                        value=issued_by,
                        raw_value=issued_by,
                        confidence=ocr_result.average_confidence * 0.7,
                        source_provider=ocr_result.metadata.provider
                    )
        
        return None
    
    def _extract_registration_address(
        self,
        tokens: List[Dict],
        full_text: str,
        ocr_result: OCRResult
    ) -> Optional[ExtractedField]:
        """Извлечь адрес регистрации"""
        # Ищем ключевые слова адреса
        keywords = ['МЕСТО ЖИТЕЛЬСТВА', 'ЗАРЕГИСТРИРОВАН', 'АДРЕС']
        
        for keyword in keywords:
            if keyword in full_text.upper():
                idx = full_text.upper().find(keyword)
                after_keyword = full_text[idx + len(keyword):].strip()
                
                # Адрес может занимать несколько строк
                lines = after_keyword.split('\n')[:5]
                address = ' '.join([line.strip() for line in lines if line.strip()])
                
                if len(address) > 15:  # Минимальная длина для адреса
                    return ExtractedField(
                        field_type=PassportFieldType.REGISTRATION_ADDRESS,
                        value=address,
                        raw_value=address,
                        confidence=ocr_result.average_confidence * 0.7,
                        source_provider=ocr_result.metadata.provider
                    )
        
        return None


# Singleton instance
field_extraction_service = FieldExtractionService()
