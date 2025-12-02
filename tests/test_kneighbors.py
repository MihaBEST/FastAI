"""
Тесты для модуля kneighbors
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sklearn.datasets import make_classification
from kneighbors import t_kneighbors, kneighbors


def test_basic_kneighbors():
    """Тест основной функции с обучением"""
    print("  Тест 1: Базовая KNN классификация")
    X, y = make_classification(n_samples=100, n_features=4, n_classes=3, random_state=42)
    model = t_kneighbors(X, y, test_size=0.2, choose_optimal=False, neighbors=3)

    assert model is not None, "Модель не создана"
    assert model.n_neighbors == 3, f"Неверное количество соседей: {model.n_neighbors}"
    print("    ✅ Базовая KNN работает")


def test_kneighbors_optimal():
    """Тест с выбором оптимального количества соседей"""
    print("  Тест 2: Выбор оптимального K")
    X, y = make_classification(n_samples=100, n_features=4, n_classes=3, random_state=42)
    model = t_kneighbors(X, y, test_size=0.2, choose_optimal=True)

    assert model is not None, "Модель не создана"
    assert 1 <= model.n_neighbors <= 10, f"K вне диапазона: {model.n_neighbors}"
    print(f"    ✅ Оптимальное K найдено: {model.n_neighbors}")


def test_kneighbors_untrained():
    """Тест создания необученной модели"""
    print("  Тест 3: Создание необученной модели")
    model = kneighbors(neighbors=5)

    assert model is not None, "Модель не создана"
    assert model.n_neighbors == 5, f"Неверное количество соседей: {model.n_neighbors}"
    assert not hasattr(model, 'classes_'), "Модель не должна быть обучена"
    print("    ✅ Необученная модель создана")


def test_no_test_data_error():
    """Тест обработки ошибки без тестовых данных"""
    print("  Тест 4: Обработка ошибки без тестовых данных")
    X, y = make_classification(n_samples=100, n_features=4, n_classes=3, random_state=42)

    try:
        t_kneighbors(X, y, test_size=0, choose_optimal=True)
        assert False, "Должна была возникнуть ошибка ValueError"
    except ValueError as e:
        assert "TEST DATA" in str(e) or "Warning" in str(e), f"Неожиданная ошибка: {e}"
        print("    ✅ Ошибка корректно обработана")


def test_predictions():
    """Тест предсказаний модели"""
    print("  Тест 5: Предсказания модели")
    X, y = make_classification(n_samples=100, n_features=4, n_classes=3, random_state=42)
    model = t_kneighbors(X, y, test_size=0.2, choose_optimal=False, neighbors=3)

    predictions = model.predict(X[:10])
    assert len(predictions) == 10, f"Неверное количество предсказаний: {len(predictions)}"
    assert all(pred in model.classes_ for pred in predictions), "Неверные метки предсказаний"
    print("    ✅ Предсказания работают корректно")


def run_tests():
    """Запуск всех тестов KNN"""
    print("\n" + "=" * 40)
    print("Запуск тестов KNeighbors")
    print("=" * 40)

    tests = [
        test_basic_kneighbors,
        test_kneighbors_optimal,
        test_kneighbors_untrained,
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