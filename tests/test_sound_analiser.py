"""
Тесты для модуля sound_analiser
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sound_analiser import t_sound_analiser, sound_analiser


def test_basic_sound_analiser():
    """Тест основной функции"""
    print("  Тест 1: Базовый анализатор звука")
    sounds = [np.random.randn(13) for _ in range(50)]
    labels = np.random.randint(0, 3, size=(50,))

    model = t_sound_analiser(sounds, labels, test_size=0.2)

    assert model is not None, "Модель не создана"
    assert hasattr(model, 'coef_'), "Модель должна иметь коэффициенты"
    print("    ✅ Базовый анализатор работает")


def test_sound_analiser_no_test():
    """Тест без тестовой выборки"""
    print("  Тест 2: Анализатор без тестовых данных")
    sounds = [np.random.randn(13) for _ in range(30)]
    labels = np.random.randint(0, 2, size=(30,))

    model = t_sound_analiser(sounds, labels, test_size=0)

    assert model is not None, "Модель не создана"
    assert hasattr(model, 'coef_'), "Модель должна иметь коэффициенты"
    print("    ✅ Анализатор без теста работает")


def test_sound_analiser_untrained():
    """Тест создания необученной модели"""
    print("  Тест 3: Необученная модель")
    model = sound_analiser()

    assert model is not None, "Модель не создана"
    assert model.max_iter == 1000, f"Неверный max_iter: {model.max_iter}"
    assert model.C == 1.0, f"Неверный C: {model.C}"
    assert model.multi_class == 'auto', f"Неверный multi_class: {model.multi_class}"
    assert model.solver == 'lbfgs', f"Неверный solver: {model.solver}"
    assert not hasattr(model, 'coef_'), "Модель не должна быть обучена"
    print("    ✅ Необученная модель создана")


def test_data_length_mismatch():
    """Тест несоответствия длины данных и меток"""
    print("  Тест 4: Проверка длины данных")
    sounds = [np.random.randn(13) for _ in range(50)]
    wrong_labels = np.random.randint(0, 3, size=(30,))  # Меньше чем звуков

    try:
        t_sound_analiser(sounds, wrong_labels, test_size=0.2)
        assert False, "Должна была возникнуть ошибка"
    except ValueError as e:
        assert "не совпадает" in str(e) or "Количество" in str(e), f"Неожиданная ошибка: {e}"
        print("    ✅ Ошибка длины данных обработана")


def test_prediction_shape():
    """Тест формы предсказаний"""
    print("  Тест 5: Форма предсказаний")
    sounds = [np.random.randn(13) for _ in range(50)]
    labels = np.random.randint(0, 3, size=(50,))

    model = t_sound_analiser(sounds, labels, test_size=0.2)

    # Создаем тестовые данные для предсказания
    test_sounds = [np.random.randn(13) for _ in range(5)]
    test_array = np.array(test_sounds)

    predictions = model.predict(test_array)
    assert len(predictions) == 5, f"Неверное количество предсказаний: {len(predictions)}"
    assert all(pred in model.classes_ for pred in predictions), "Неверные метки предсказаний"
    print("    ✅ Форма предсказаний корректна")


def run_tests():
    """Запуск всех тестов анализатора звука"""
    print("\n" + "=" * 40)
    print("Запуск тестов Sound Analyser")
    print("=" * 40)

    tests = [
        test_basic_sound_analiser,
        test_sound_analiser_no_test,
        test_sound_analiser_untrained,
        test_data_length_mismatch,
        test_prediction_shape
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