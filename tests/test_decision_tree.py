"""
Тесты для модуля decision_tree
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sklearn.datasets import make_classification
from decision_tree import t_decision_tree, decision_tree


def test_basic_decision_tree():
    """Тест основной функции с обучением"""
    print("  Тест 1: Базовое дерево решений")
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3, random_state=42)
    model = t_decision_tree(X, y, test_size=0.2, choose_optimal=False, max_depth=3)

    assert model is not None, "Модель не создана"
    assert model.max_depth == 3, f"Неверная глубина: {model.max_depth}"
    print("    ✅ Базовое дерево работает")


def test_decision_tree_optimal():
    """Тест с выбором оптимальной глубины"""
    print("  Тест 2: Выбор оптимальной глубины")
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3, random_state=42)
    model = t_decision_tree(X, y, test_size=0.2, choose_optimal=True)

    assert model is not None, "Модель не создана"
    assert 1 <= model.max_depth <= 10, f"Глубина вне диапазона: {model.max_depth}"
    print(f"    ✅ Оптимальная глубина найдена: {model.max_depth}")


def test_decision_tree_untrained():
    """Тест создания необученной модели"""
    print("  Тест 3: Создание необученной модели")
    model = decision_tree(max_depth=5, random_state=42)

    assert model is not None, "Модель не создана"
    assert model.max_depth == 5, f"Неверная глубина: {model.max_depth}"
    assert model.random_state == 42, f"Неверный random_state: {model.random_state}"
    assert not hasattr(model, 'classes_'), "Модель не должна быть обучена"
    print("    ✅ Необученная модель создана")


def test_no_test_data_error():
    """Тест обработки ошибки без тестовых данных"""
    print("  Тест 4: Обработка ошибки без тестовых данных")
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3, random_state=42)

    try:
        t_decision_tree(X, y, test_size=0, choose_optimal=True)
        assert False, "Должна была возникнуть ошибка ValueError"
    except ValueError as e:
        assert "TEST DATA" in str(e) or "Warning" in str(e), f"Неожиданная ошибка: {e}"
        print("    ✅ Ошибка корректно обработана")


def test_predictions():
    """Тест предсказаний модели"""
    print("  Тест 5: Предсказания модели")
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3, random_state=42)
    model = t_decision_tree(X, y, test_size=0.2, choose_optimal=False, max_depth=3)

    predictions = model.predict(X[:10])
    assert len(predictions) == 10, f"Неверное количество предсказаний: {len(predictions)}"
    assert all(pred in model.classes_ for pred in predictions), "Неверные метки предсказаний"
    print("    ✅ Предсказания работают корректно")


def run_tests():
    """Запуск всех тестов Decision Tree"""
    print("\n" + "=" * 40)
    print("Запуск тестов Decision Tree")
    print("=" * 40)

    tests = [
        test_basic_decision_tree,
        test_decision_tree_optimal,
        test_decision_tree_untrained,
        test_no_test_data_error,
        test_predictions
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