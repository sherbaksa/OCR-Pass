#!/usr/bin/env python3
"""
Простейший тест PaddleOCR - точно как в Colab
"""

from paddleocr import PaddleOCR
import sys

print("=== ТЕСТ PADDLEOCR ===")
print(f"Python: {sys.version}")

# Инициализация ТОЧНО как в Colab
print("\n1. Инициализация PaddleOCR...")
ocr = PaddleOCR(lang='ru')
print("✓ Инициализация завершена")

# Путь к тестовому изображению
test_image = "/app/test_passport.jpg"

print(f"\n2. Распознавание изображения: {test_image}")
result = ocr.predict(test_image)

print(f"\n3. Результат:")
print(f"   Type: {type(result)}")
print(f"   Length: {len(result) if isinstance(result, list) else 'N/A'}")

if result and isinstance(result, list) and len(result) > 0:
    item = result[0]
    print(f"   First item type: {type(item)}")
    
    if isinstance(item, dict) and 'res' in item:
        res = item['res']
        rec_texts = res.get('rec_texts', [])
        rec_scores = res.get('rec_scores', [])
        
        print(f"\n4. Распознано текстов: {len(rec_texts)}")
        print("\nПервые 10 результатов:")
        for i, text in enumerate(rec_texts[:10]):
            score = rec_scores[i] if i < len(rec_scores) else 0
            print(f"   [{i}] {text} (conf: {score:.3f})")
    else:
        print("   ОШИБКА: Нет ключа 'res' в результате!")
else:
    print("   ОШИБКА: Пустой результат!")

print("\n=== КОНЕЦ ТЕСТА ===")
