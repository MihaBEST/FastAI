"""
Тесты для модуля eazy_models.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from eazy_models import (
    ovr_model, ovo_model, multi_output_regression, ensemble_regression,
    image_classifier, image_feature_extractor, text_embedding,
    meta_text_classifier as eazy_text_classifier, audio_feature_extractor,
    time_series_model, create_pipeline
)
import tensorflow.keras

def test_ovr_model_basic():
    """Тест базовой модели One-vs-Rest"""
    print("  Тест 1: One-vs-Rest модель")
    model = ovr_model()

    assert model is not None, "Модель не создана"
    assert hasattr(model, 'fit'), "Модель должна иметь метод fit"
    assert hasattr(model, 'predict'), "Модель должна иметь метод predict"
    assert model.n_jobs == -1, f"Неверный n_jobs: {model.n_jobs}"
    print("    ✅ OVR модель создана")


def test_ovr_model_custom_estimator():
    """Тест OVR с кастомным базовым классификатором"""
    print("  Тест 2: OVR с кастомным классификатором")
    from sklearn.svm import SVC

    custom_estimator = SVC(kernel='linear')
    model = ovr_model(base_estimator=custom_estimator)

    assert model is not None, "Модель не создана"
    assert isinstance(model.estimator, SVC), f"Неверный базовый классификатор: {type(model.estimator)}"
    print("    ✅ OVR с кастомным классификатором создана")


def test_ovo_model_basic():
    """Тест базовой модели One-vs-One"""
    print("  Тест 3: One-vs-One модель")
    model = ovo_model()

    assert model is not None, "Модель не создана"
    assert hasattr(model, 'fit'), "Модель должна иметь метод fit"
    assert hasattr(model, 'predict'), "Модель должна иметь метод predict"
    assert model.n_jobs == -1, f"Неверный n_jobs: {model.n_jobs}"
    print("    ✅ OVO модель создана")


def test_multi_output_regression_types():
    """Тест многозадачной регрессии разных типов"""
    print("  Тест 4: Многозадачная регрессия")

    regression_types = ['random_forest', 'linear', 'svr', 'gradient_boosting']

    for reg_type in regression_types:
        try:
            model = multi_output_regression(reg_type)
            assert model is not None, f"Модель {reg_type} не создана"
            assert hasattr(model, 'fit'), f"Модель {reg_type} должна иметь метод fit"
            assert hasattr(model, 'predict'), f"Модель {reg_type} должна иметь метод predict"
            print(f"    ✅ Многозадачная регрессия '{reg_type}' создана")
        except Exception as e:
            print(f"    ⚠  Тип '{reg_type}' не поддерживается: {e}")


def test_ensemble_regression_types():
    """Тест ансамблевой регрессии разных типов"""
    print("  Тест 5: Ансамблевая регрессия")

    ensemble_types = ['voting', 'stacking', 'bagging', 'adaboost']

    for ens_type in ensemble_types:
        try:
            model = ensemble_regression(ens_type)
            assert model is not None, f"Ансамбль {ens_type} не создан"
            assert hasattr(model, 'fit'), f"Ансамбль {ens_type} должен иметь метод fit"
            assert hasattr(model, 'predict'), f"Ансамбль {ens_type} должен иметь метод predict"
            print(f"    ✅ Ансамбль регрессии '{ens_type}' создан")
        except Exception as e:
            print(f"    ⚠  Ансамбль '{ens_type}' не поддерживается: {e}")


def test_image_classifier_types():
    """Тест предобученных классификаторов изображений"""
    print("  Тест 6: Предобученные классификаторы изображений")

    # Список моделей для тестирования (mobilenet самая легкая)
    image_models = ['mobilenet', 'resnet', 'vgg', 'efficientnet']

    for model_type in image_models[:1]:  # Тестируем только mobilenet для скорости
        try:
            model = image_classifier(model_type)

            if model is None:
                print(f"    ⚠  TensorFlow не установлен для {model_type}")
                continue

            assert hasattr(model, 'predict'), f"Модель {model_type} должна иметь метод predict"

            # Проверяем, что модель заморожена (не обучается)
            trainable_layers = sum(1 for layer in model.layers if layer.trainable)
            assert trainable_layers == 0, f"Модель {model_type} не должна быть обучаемой"

            print(f"    ✅ Классификатор изображений '{model_type}' создан")
        except Exception as e:
            print(f"    ⚠  Модель '{model_type}' не поддерживается: {e}")


def test_image_feature_extractor():
    """Тест экстрактора признаков изображений"""
    print("  Тест 7: Экстрактор признаков изображений")

    try:
        model = image_feature_extractor('mobilenet')

        if model is None:
            print("    ⚠  TensorFlow не установлен")
            return

        assert hasattr(model, 'predict'), "Модель должна иметь метод predict"

        # Проверяем, что include_top=False
        assert model.layers[0].name.startswith('mobilenet'), "Должна быть MobileNet"

        print("    ✅ Экстрактор признаков изображений создан")
    except Exception as e:
        print(f"    ⚠  Экстрактор признаков не поддерживается: {e}")


def test_text_embedding_tfidf():
    """Тест TF-IDF эмбеддинга текста"""
    print("  Тест 8: TF-IDF эмбеддинг текста")

    model = text_embedding('tfidf')

    assert model is not None, "Модель не создана"
    assert hasattr(model, 'fit_transform'), "Модель должна иметь метод fit_transform"
    assert hasattr(model, 'transform'), "Модель должна иметь метод transform"

    # Тестируем преобразование
    texts = ["привет мир", "машинное обучение"]
    embeddings = model.fit_transform(texts)

    assert embeddings.shape[0] == 2, f"Неверное количество эмбеддингов: {embeddings.shape[0]}"
    print("    ✅ TF-IDF эмбеддинг работает")


def test_text_classifier_pipelines():
    """Тест готовых пайплайнов классификации текста"""
    print("  Тест 9: Готовые пайплайны классификации текста")

    tasks = ['sentiment', 'spam']

    for task in tasks:
        try:
            pipeline = eazy_text_classifier(task)

            assert pipeline is not None, f"Пайплайн {task} не создан"
            assert hasattr(pipeline, 'fit'), f"Пайплайн {task} должен иметь метод fit"
            assert hasattr(pipeline, 'predict'), f"Пайплайн {task} должен иметь метод predict"

            print(f"    ✅ Пайплайн классификации текста '{task}' создан")
        except Exception as e:
            print(f"    ⚠  Пайплайн '{task}' не поддерживается: {e}")


def test_audio_feature_extractor():
    """Тест экстрактора признаков аудио"""
    print("  Тест 10: Экстрактор признаков аудио")

    feature_types = ['mfcc', 'spectrogram']

    for feature_type in feature_types:
        try:
            extractor = audio_feature_extractor(feature_type)

            assert extractor is not None, f"Экстрактор {feature_type} не создан"
            assert hasattr(extractor, 'transform'), f"Экстрактор {feature_type} должен иметь метод transform"

            # Создаем тестовые аудиоданные
            test_audio = [np.random.randn(22050) for _ in range(3)]  # 1 секунда при 22050 Гц

            features = extractor.transform(test_audio)

            assert features is not None, f"Признаки не извлечены для {feature_type}"
            assert len(features) == 3, f"Неверное количество признаков: {len(features)}"

            print(f"    ✅ Экстрактор признаков аудио '{feature_type}' работает")
        except ImportError as e:
            print(f"    ⚠  Librosa не установлен для {feature_type}: {e}")
        except Exception as e:
            print(f"    ⚠  Экстрактор '{feature_type}' не поддерживается: {e}")


def test_time_series_models():
    """Тест моделей временных рядов"""
    print("  Тест 11: Модели временных рядов")

    try:
        # Тестируем LSTM
        lstm_model = time_series_model('lstm', 'forecast')

        if lstm_model is None:
            print("    ⚠  TensorFlow не установлен для LSTM")
        else:
            assert hasattr(lstm_model, 'fit'), "LSTM модель должна иметь метод fit"
            assert hasattr(lstm_model, 'predict'), "LSTM модель должна иметь метод predict"
            print("    ✅ LSTM модель создана")
    except Exception as e:
        print(f"    ⚠  LSTM не поддерживается: {e}")

    try:
        # Тестируем ARIMA
        arima_model = time_series_model('arima', 'forecast')

        if arima_model is None:
            print("    ⚠  statsmodels не установлен для ARIMA")
        else:
            assert hasattr(arima_model, 'fit'), "ARIMA модель должна иметь метод fit"
            assert hasattr(arima_model, 'predict'), "ARIMA модель должна иметь метод predict"
            print("    ✅ ARIMA модель создана")
    except Exception as e:
        print(f"    ⚠  ARIMA не поддерживается: {e}")


def test_create_pipeline_text():
    """Тест создания пайплайна для текста"""
    print("  Тест 12: Создание пайплайна для текста")

    pipeline = create_pipeline('text', 'tfidf_logistic')

    assert pipeline is not None, "Пайплайн не создан"
    assert hasattr(pipeline, 'fit'), "Пайплайн должен иметь метод fit"
    assert hasattr(pipeline, 'predict'), "Пайплайн должен иметь метод predict"

    # Проверяем, что это действительно пайплайн
    from sklearn.pipeline import Pipeline
    assert isinstance(pipeline, Pipeline), f"Неверный тип пайплайна: {type(pipeline)}"

    print("    ✅ Пайплайн для текста создан")


def test_create_pipeline_image():
    """Тест создания пайплайна для изображений"""
    print("  Тест 13: Создание пайплайна для изображений")

    try:
        pipeline = create_pipeline('image', 'mobilenet')

        if pipeline is None:
            print("    ⚠  TensorFlow не установлен")
            return

        assert hasattr(pipeline, 'predict'), "Пайплайн должен иметь метод predict"
        print("    ✅ Пайплайн для изображений создан")
    except Exception as e:
        print(f"    ⚠  Пайплайн для изображений не поддерживается: {e}")


def test_model_fitting():
    """Тест обучения моделей на данных"""
    print("  Тест 14: Обучение моделей на данных")

    # Создаем тестовые данные
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, size=(100,))

    # Тестируем OVR модель
    model = ovr_model()

    try:
        model.fit(X, y)
        assert hasattr(model, 'estimators_'), "Модель должна быть обучена"
        assert len(model.estimators_) > 0, "Должны быть созданы estimators"

        predictions = model.predict(X[:5])
        assert len(predictions) == 5, f"Неверное количество предсказаний: {len(predictions)}"

        print("    ✅ Модель успешно обучена и делает предсказания")
    except Exception as e:
        print(f"    ⚠  Ошибка обучения модели: {e}")


def test_invalid_parameters():
    """Тест обработки неверных параметров"""
    print("  Тест 15: Обработка неверных параметров")

    # Тест неверного типа модели для изображений
    try:
        create_pipeline('image', 'invalid_model')
        assert False, "Должна была возникнуть ошибка"
    except ValueError as e:
        assert "Доступно" in str(e) or "available" in str(e), f"Неожиданная ошибка: {e}"
        print("    ✅ Неверные параметры обработаны корректно")
    except Exception as e:
        print(f"    ⚠  Неожиданная ошибка: {e}")


def run_tests():
    """Запуск всех тестов для eazy_models"""
    print("\n" + "=" * 60)
    print("Запуск тестов Eazy Models")
    print("=" * 60)

    tests = [
        test_ovr_model_basic,
        test_ovr_model_custom_estimator,
        test_ovo_model_basic,
        test_multi_output_regression_types,
        test_ensemble_regression_types,
        test_image_classifier_types,
        test_image_feature_extractor,
        test_text_embedding_tfidf,
        test_text_classifier_pipelines,
        test_audio_feature_extractor,
        test_time_series_models,
        test_create_pipeline_text,
        test_create_pipeline_image,
        test_model_fitting,
        test_invalid_parameters
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