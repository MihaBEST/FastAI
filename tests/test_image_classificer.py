"""
Тесты для модуля image_classificer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import tensorflow as tf
from image_classificer import t_image_classifier

def test_image_classifier_numeric():
    """Тест с числовыми метками"""
    print("  Тест 1: Числовые метки")
    tf.random.set_seed(42)
    np.random.seed(42)

    X_images = np.random.rand(100, 64, 64, 3).astype(np.float32)
    y_labels = np.random.randint(0, 5, size=(100,))

    model = t_image_classifier(
        X_images, y_labels,
        input_shape=(64, 64, 3),
        test_size=0.2,
        epochs=1,
        verbose=0
    )

    assert model is not None, "Модель не создана"
    assert model.output_shape[-1] == 5, f"Неверный выходной размер: {model.output_shape[-1]}"
    print("    ✅ Числовые метки работают")

def test_image_classifier_string():
    """Тест со строковыми метками"""
    print("  Тест 2: Строковые метки")
    tf.random.set_seed(42)
    np.random.seed(42)

    X_images = np.random.rand(100, 64, 64, 3).astype(np.float32)
    y_strings = np.array(['cat', 'dog', 'bird', 'cat', 'dog'] * 20)[:100]

    model = t_image_classifier(
        X_images, y_strings,
        input_shape=(64, 64, 3),
        test_size=0.2,
        epochs=1,
        verbose=0
    )

    assert model is not None, "Модель не создана"
    # Должно быть 3 класса: cat, dog, bird
    assert model.output_shape[-1] == 3, f"Неверный выходной размер: {model.output_shape[-1]}"
    print("    ✅ Строковые метки работают")

def test_image_classifier_binary():
    """Тест бинарной классификации"""
    print("  Тест 3: Бинарная классификация")
    tf.random.set_seed(42)
    np.random.seed(42)

    X_images = np.random.rand(100, 64, 64, 3).astype(np.float32)
    y_binary = np.random.randint(0, 2, size=(100,))

    model = t_image_classifier(
        X_images, y_binary,
        input_shape=(64, 64, 3),
        test_size=0.2,
        epochs=1,
        verbose=0,
        loss='binary_crossentropy'
    )

    assert model is not None, "Модель не создана"
    assert model.output_shape[-1] == 2, f"Неверный выходной размер: {model.output_shape[-1]}"
    print("    ✅ Бинарная классификация работает")

def test_model_structure():
    """Тест структуры модели"""
    print("  Тест 4: Структура модели")
    tf.random.set_seed(42)
    np.random.seed(42)

    X_images = np.random.rand(100, 64, 64, 3).astype(np.float32)
    y_labels = np.random.randint(0, 5, size=(100,))

    model = t_image_classifier(
        X_images, y_labels,
        input_shape=(64, 64, 3),
        test_size=0.2,
        epochs=1,
        verbose=0
    )

    # Проверяем, что модель содержит ожидаемые слои
    layer_types = [type(layer).__name__ for layer in model.layers]
    expected_layers = ['Conv2D', 'MaxPooling2D', 'Flatten', 'Dense']

    for expected in expected_layers:
        assert expected in layer_types, f"Слой {expected} отсутствует"

    print("    ✅ Структура модели корректна")

def test_different_parameters():
    """Тест с разными параметрами"""
    print("  Тест 5: Разные параметры")
    tf.random.set_seed(42)
    np.random.seed(42)

    X_images = np.random.rand(100, 64, 64, 3).astype(np.float32)
    y_labels = np.random.randint(0, 5, size=(100,))

    model = t_image_classifier(
        X_images, y_labels,
        input_shape=(64, 64, 3),
        num_classes=5,
        test_size=0.3,
        random_state=123,
        epochs=2,
        batch_size=16,
        learning_rate=0.01,
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        verbose=0
    )

    assert model is not None, "Модель не создана"
    print("    ✅ Разные параметры работают")

def run_tests():
    """Запуск всех тестов классификатора изображений"""
    print("\n" + "=" * 40)
    print("Запуск тестов Image Classifier")
    print("=" * 40)

    tests = [
        test_image_classifier_numeric,
        test_image_classifier_string,
        test_image_classifier_binary,
        test_model_structure,
        test_different_parameters
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