"""
Тесты для модуля cnn_network
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import tensorflow as tf
from cnn_network import t_cnn_network, cnn_network


def test_cnn_multiclass():
    """Тест для многоклассовой классификации"""
    print("  Тест 1: Многоклассовая классификация")
    tf.random.set_seed(42)
    np.random.seed(42)

    X_images = np.random.randn(100, 28, 28, 1).astype(np.float32)
    y_cls = np.random.randint(0, 10, size=(100,))

    model, history = t_cnn_network(
        X_images, y_cls,
        test_size=0.2, epochs=1,
        32, 64, 10
    )

    assert model is not None, "Модель не создана"
    assert history is not None, "История обучения не создана"
    assert model.layers[-1].output_shape[-1] == 10, f"Неверный выходной размер: {model.layers[-1].output_shape[-1]}"
    print("    ✅ Многоклассовая классификация работает")


def test_cnn_binary():
    """Тест для бинарной классификации"""
    print("  Тест 2: Бинарная классификация")
    tf.random.set_seed(42)
    np.random.seed(42)

    X_images = np.random.randn(100, 28, 28, 1).astype(np.float32)
    y_bin = np.random.randint(0, 2, size=(100,))

    model, history = t_cnn_network(
        X_images, y_bin,
        test_size=0.2, epochs=1,
        16, 32, 1
    )

    assert model is not None, "Модель не создана"
    assert model.layers[
               -1].activation.__name__ == 'sigmoid', f"Неверная активация: {model.layers[-1].activation.__name__}"
    print("    ✅ Бинарная классификация работает")


def test_cnn_untrained():
    """Тест создания необученной модели"""
    print("  Тест 3: Необученная модель CNN")
    model = cnn_network((28, 28, 1), 32, 64, 128, 10)

    assert model is not None, "Модель не создана"
    assert model.input_shape == (None, 28, 28, 1), f"Неверная входная форма: {model.input_shape}"
    assert model.output_shape[-1] == 10, f"Неверный выходной размер: {model.output_shape[-1]}"
    print("    ✅ Необученная CNN создана")


def test_2d_input():
    """Тест с 2D входными данными (без каналов)"""
    print("  Тест 4: 2D входные данные")
    tf.random.set_seed(42)
    np.random.seed(42)

    X_2d = np.random.randn(100, 28, 28)
    y_cls = np.random.randint(0, 10, size=(100,))

    model, history = t_cnn_network(
        X_2d, y_cls,
        test_size=0.2, epochs=1,
        32, 64, 10
    )

    assert model is not None, "Модель не создана"
    assert model.input_shape == (None, 28, 28, 1), f"Неверная входная форма: {model.input_shape}"
    print("    ✅ 2D данные автоматически преобразованы")


def test_invalid_structure():
    """Тест с некорректной структурой"""
    print("  Тест 5: Некорректная структура")
    X_images = np.random.randn(100, 28, 28, 1).astype(np.float32)
    y_cls = np.random.randint(0, 10, size=(100,))

    try:
        t_cnn_network(X_images, y_cls, test_size=0.2, epochs=1, 10)
        assert False, "Должна была возникнуть ошибка"
    except ValueError as e:
        assert "структуры" in str(e).lower() or "parameters" in str(e).lower(), f"Неожиданная ошибка: {e}"
        print("    ✅ Некорректная структура обработана")


def run_tests():
    """Запуск всех тестов CNN"""
    print("\n" + "=" * 40)
    print("Запуск тестов CNN Network")
    print("=" * 40)

    tests = [
        test_cnn_multiclass,
        test_cnn_binary,
        test_cnn_untrained,
        test_2d_input,
        test_invalid_structure
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