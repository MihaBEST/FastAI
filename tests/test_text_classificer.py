"""
Тесты для модуля text_classificer
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from text_classificer import t_text_classifier, text_classifier


def test_basic_text_classifier():
    """Тест основной функции классификатора"""
    print("  Тест 1: Базовый классификатор текста")
    texts = [
        "Это очень хороший продукт, мне понравилось",
        "Ужасное качество, не рекомендую",
        "Нормальный товар за свои деньги",
        "Отличное приобретение, буду покупать еще",
        "Разочарован, не оправдало ожиданий"
    ]
    labels = [1, 0, 1, 1, 0]

    model = t_text_classifier(texts, labels, test_size=0.2)

    assert model is not None, "Модель не создана"
    assert hasattr(model, 'coef_'), "Модель должна иметь коэффициенты"
    assert hasattr(model, 'classes_'), "Модель должна иметь классы"
    print("    ✅ Базовый классификатор работает")


def test_text_classifier_no_test():
    """Тест без тестовой выборки"""
    print("  Тест 2: Классификатор без тестовых данных")
    texts = [
        "Отличный сервис",
        "Плохое обслуживание",
        "Все понравилось"
    ]
    labels = [1, 0, 1]

    model = t_text_classifier(texts, labels, test_size=0)

    assert model is not None, "Модель не создана"
    assert hasattr(model, 'coef_'), "Модель должна иметь коэффициенты"
    print("    ✅ Классификатор без теста работает")


def test_text_classifier_untrained():
    """Тест создания необученной модели"""
    print("  Тест 3: Необученная модель")
    model = text_classifier(random_state=42)

    assert model is not None, "Модель не создана"
    assert model.random_state == 42, f"Неверный random_state: {model.random_state}"
    assert model.max_iter == 1000, f"Неверный max_iter: {model.max_iter}"
    assert model.C == 1.0, f"Неверный C: {model.C}"
    assert not hasattr(model, 'coef_'), "Модель не должна быть обучена"
    print("    ✅ Необученная модель создана")


def test_different_parameters():
    """Тест с разными параметрами"""
    print("  Тест 4: Разные параметры")
    texts = ["Текст 1", "Текст 2", "Текст 3"]
    labels = [0, 1, 0]

    model = t_text_classifier(
        texts, labels,
        test_size=0.3,
        random_state=123,
        max_features=5000
    )

    assert model is not None, "Модель не создана"
    print("    ✅ Параметры применены")


def test_russian_stop_words():
    """Тест с русскими стоп-словами"""
    print("  Тест 5: Русские стоп-слова")
    texts = ["Это и или но", "Вот так вот", "И все такое"]
    labels = [0, 1, 0]

    model = t_text_classifier(
        texts, labels,
        test_size=0.2,
        stop_word="russian"
    )

    assert model is not None, "Модель не создана"
    print("    ✅ Русские стоп-слова работают")


def run_tests():
    """Запуск всех тестов текстового классификатора"""
    print("\n" + "=" * 40)
    print("Запуск тестов Text Classifier")
    print("=" * 40)

    tests = [
        test_basic_text_classifier,
        test_text_classifier_no_test,
        test_text_classifier_untrained,
        test_different_parameters,
        test_russian_stop_words
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"    ❌ Ошибка: {e}")
        except Exception as e:
            failed += 1
            print(f"    ❌ Неожиданная ошибка: {e}")

    print(f"\nРезультат: {passed} прошло, {failed} упало")
    return passed, failed


if __name__ == "__main__":
    run_tests()