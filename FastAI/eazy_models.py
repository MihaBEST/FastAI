# Мета-модели для многоклассовой классификации
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def ovr_model(base_estimator=None):
    """
    Мета-модель One-vs-Rest (OvR) для многоклассовой классификации.
    One-vs-Rest (один против всех) - стратегия для многоклассовой классификации,
    где для каждого класса создается отдельная бинарная модель, которая отличает
    этот класс от всех остальных классов.

    Параметры:
    ----------
    base_estimator : объект модели, optional
        Базовая модель для использования в OvR. Если None, используется
        LogisticRegression с max_iter=1000.
    """
    from sklearn.multiclass import OneVsRestClassifier

    if base_estimator is None:
        base_estimator = LogisticRegression(max_iter=1000)

    model = OneVsRestClassifier(
        estimator=base_estimator,
        n_jobs=-1
    )
    return model


def ovo_model(base_estimator=None):
    """
    Мета-модель One-vs-One (OvO) для многоклассовой классификации.
    One-vs-One (один против одного) - стратегия для многоклассовой классификации,
    где для каждой пары классов создается отдельная бинарная модель.
    Для N классов создается N*(N-1)/2 моделей.
    Итоговый класс выбирается по большинству голосов.
    """
    from sklearn.multiclass import OneVsOneClassifier

    if base_estimator is None:
        base_estimator = SVC(kernel='linear', probability=True)

    model = OneVsOneClassifier(
        estimator=base_estimator,
        n_jobs=-1
    )
    return model


import numpy as np
import warnings

warnings.filterwarnings('ignore')


# ==================== МОДЕЛИ РЕГРЕССИИ ====================

def multi_output_regression(model_type='random_forest'):
    """
    Создает модель для многозадачной регрессии.
    Многозадачная регрессия - это регрессионная задача, где нужно предсказать
    несколько целевых переменных одновременно.

    Доступные варианты:
    --------------------
    1. 'random_forest' - ансамбль решающих деревьев, устойчив к выбросам
       и переобучению, хорошо работает на нелинейных данных.
    2. 'linear' - линейная регрессия, быстрая и интерпретируемая, но предполагает
       линейную зависимость между признаками и целевыми переменными.
    3. 'svr' - Support Vector Regression с RBF ядром, хорошо работает на
       небольших выборках и нелинейных данных.
    4. 'gradient_boosting' - градиентный бустинг над деревьями, обычно дает
       высокую точность, но требует больше времени на обучение.

    """
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            n_jobs=-1
        )

    elif model_type == 'linear':
        from sklearn.linear_model import LinearRegression
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(LinearRegression(), n_jobs=-1)

    elif model_type == 'svr':
        from sklearn.svm import SVR
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(SVR(kernel='rbf', C=1.0), n_jobs=-1)

    elif model_type == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        )

    return model


def ensemble_regression(ensemble_type='voting'):
    """
    Создает ансамблевую модель для регрессии.
    Ансамбли объединяют несколько базовых моделей для улучшения точности
    и устойчивости предсказаний.

    Доступные варианты:
    --------------------
    1. 'voting' - голосующий ансамбль, усредняет предсказания нескольких моделей
       (линейная регрессия, дерево решений и SVR).
    2. 'stacking' - стекинг, использует предсказания базовых моделей как вход
       для мета-модели (второго уровня).
    3. 'bagging' - бэггинг, обучает несколько одинаковых моделей на разных
       подвыборках данных и усредняет их предсказания.
    4. 'adaboost' - AdaBoost, последовательно обучает модели, где каждая следующая
       модель исправляет ошибки предыдущих.
    """
    if ensemble_type == 'voting':
        from sklearn.ensemble import VotingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.svm import SVR
        model = VotingRegressor([
            ('lr', LinearRegression()),
            ('dt', DecisionTreeRegressor(max_depth=5)),
            ('svr', SVR(kernel='rbf', C=10))
        ])

    elif ensemble_type == 'stacking':
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.svm import SVR
        model = StackingRegressor(
            estimators=[('lr', LinearRegression()), ('ridge', Ridge(alpha=1.0)), ('svr', SVR(kernel='linear'))],
            final_estimator=LinearRegression(),
            cv=5
        )

    elif ensemble_type == 'bagging':
        from sklearn.ensemble import BaggingRegressor
        from sklearn.tree import DecisionTreeRegressor
        model = BaggingRegressor(
            DecisionTreeRegressor(),
            n_estimators=10,
            max_samples=0.8,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )

    elif ensemble_type == 'adaboost':
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        model = AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=4),
            n_estimators=50,
            learning_rate=0.1,
            random_state=42
        )

    return model


# ==================== ПРЕДОБУЧЕННЫЕ МОДЕЛИ ====================

def image_classifier(model_type='mobilenet'):
    """
    Предобученная модель для классификации изображений.
    Предобученные модели были обучены на больших датасетах (обычно ImageNet с 1.2 млн изображений
    и 1000 классов) и могут быть использованы для решения новых задач через transfer learning.

    Доступные варианты:
    --------------------
    1. 'mobilenet' - MobileNetV2: легкая и быстрая архитектура, оптимизированная для мобильных устройств
    2. 'resnet' - ResNet50: архитектура с остаточными связями, глубокая и точная
    3. 'vgg' - VGG16: простая архитектура с малым размером ядер, хороша для извлечения признаков
    4. 'efficientnet' - EfficientNetB0: современная архитектура, балансирует точность и скорость
    """
    try:
        from tensorflow import keras
    except ImportError:
        print("Установите tensorflow: pip install tensorflow")
        return None

    models = {
        'mobilenet': keras.applications.MobileNetV2,
        'resnet': keras.applications.ResNet50,
        'vgg': keras.applications.VGG16,
        'efficientnet': keras.applications.EfficientNetB0
    }

    if model_type not in models:
        raise ValueError(f"Доступно: {list(models.keys())}")

    model = models[model_type](weights='imagenet', include_top=True)
    for layer in model.layers:
        layer.trainable = False

    return model


def image_feature_extractor(model_type='mobilenet'):
    """
    Модель для извлечения признаков из изображений.
    Извлекает высокоуровневые признаки из изображений с помощью предобученных CNN.
    Эти признаки могут быть использованы для решения различных задач:
    классификации, кластеризации, поиска похожих изображений и т.д.

    Доступные варианты:
    --------------------
    1. 'mobilenet' - MobileNetV2: быстрая и легковесная
    2. 'resnet' - ResNet50: глубокая с остаточными связями
    3. 'vgg' - VGG16: хороша для текстуры и деталей
    4. 'efficientnet' - EfficientNetB0: оптимальное соотношение точности и скорости
    """
    try:
        from tensorflow import keras
    except ImportError:
        print("Установите tensorflow: pip install tensorflow")
        return None

    models = {
        'mobilenet': keras.applications.MobileNetV2,
        'resnet': keras.applications.ResNet50,
        'vgg': keras.applications.VGG16,
        'efficientnet': keras.applications.EfficientNetB0
    }

    if model_type not in models:
        raise ValueError(f"Доступно: {list(models.keys())}")

    base_model = models[model_type](weights='imagenet', include_top=False, pooling='avg')
    base_model.trainable = False

    model = keras.Sequential([base_model, keras.layers.Dropout(0.5)])
    return model


def text_embedding(model_type='tfidf'):
    """
    Предобученная модель для векторного представления текстов.

    Что такое векторное представление текста?
    ------------------------------------------
    Векторное представление (эмбеддинг) текста - это преобразование текстовых данных
    в числовые векторы фиксированной длины, которые сохраняют семантическую информацию.
    Это позволяет:
    1. Использовать тексты как вход для машинного обучения
    2. Измерять семантическую близость между текстами
    3. Выполнять операции с текстами в векторном пространстве

    Доступные варианты:
    --------------------
    1. 'tfidf' - TF-IDF векторизатор:
       - Статистический метод, основанный на частоте слов
       - Быстрый и не требует предобучения
       - Не учитывает семантику и порядок слов
       - Подходит для задач с малым количеством данных

    2. 'bert_mini' - RuBERT-tiny2:
       - Предобученная трансформерная модель для русского языка
       - Учитывает контекст и семантику слов
       - Создает контекстно-зависимые эмбеддинги
       - Требует больше вычислительных ресурсов
       - Подходит для сложных семантических задач
    """
    if model_type == 'tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        model = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.7, ngram_range=(1, 2), stop_words='russian')
        model.description = "TF-IDF векторизатор"

    elif model_type == 'bert_mini':
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            print("Установите transformers и torch")
            return None

        tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

        class RuBertEmbedder:
            def __init__(self, tokenizer, model):
                self.tokenizer = tokenizer
                self.model = model
                self.model.eval()

            def transform(self, texts):
                embeddings = []
                for text in texts:
                    inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].numpy()
                    embeddings.append(embedding[0])
                return np.array(embeddings)

        model = RuBertEmbedder(tokenizer, bert_model)

    return model


def eazy_text_classifier(task='sentiment'):
    """
    Готовый пайплайн для классификации текстов.

    Что такое пайплайн (pipeline)?
    --------------------------------
    Пайплайн в машинном обучении - это последовательность преобразований данных
    и моделей, объединенных в единый объект. Преимущества пайплайнов:
    1. Упрощение кода - одна цепочка вместо отдельных шагов
    2. Избежание утечки данных - правильное применение трансформаций
    3. Удобство кросс-валидации и подбора гиперпараметров
    4. Легкость развертывания - один объект для сохранения/загрузки

    Доступные варианты:
    --------------------
    1. 'sentiment' - анализ тональности:
       - TF-IDF векторизация с N-граммами (1-3 слова)
       - Логистическая регрессия с балансировкой классов
       - Подходит для определения позитивных/негативных отзывов

    2. 'spam' - классификация спама:
       - TF-IDF векторизация с биграммами
       - Наивный байесовский классификатор
       - Эффективен для задач бинарной классификации текстов
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer

    if task == 'sentiment':
        from sklearn.linear_model import LogisticRegression
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='russian')),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced'))
        ])

    elif task == 'spam':
        from sklearn.naive_bayes import MultinomialNB
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ('clf', MultinomialNB(alpha=0.1))
        ])

    return pipeline


def audio_feature_extractor(feature_type='mfcc'):
    """
    Модель для извлечения признаков из аудио.

    Доступные варианты:
    --------------------
    1. 'mfcc' - Mel-frequency cepstral coefficients:
       - Воспринимаемые человеком частотные характеристики
       - Включает MFCC, дельты и двойные дельты
       - Стандарт для задач распознавания речи

    2. 'spectrogram' - Спектрограмма:
       - Визуальное представление спектра частот
       - Показывает изменение частот во времени
       - Подходит для анализа музыки и звуков
    """

    class AudioFeatureExtractor:
        def __init__(self, feature_type='mfcc', sr=22050):
            self.feature_type = feature_type
            self.sr = sr

        def transform(self, audio_list):
            try:
                import librosa
            except ImportError:
                print("Установите librosa: pip install librosa")
                return None

            features = []
            for audio in audio_list:
                if self.feature_type == 'mfcc':
                    mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
                    mfcc_delta = librosa.feature.delta(mfcc)
                    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                    feature = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
                elif self.feature_type == 'spectrogram':
                    stft = np.abs(librosa.stft(audio))
                    feature = librosa.amplitude_to_db(stft, ref=np.max)
                feature = np.mean(feature, axis=1)
                features.append(feature)
            return np.array(features)

    extractor = AudioFeatureExtractor(feature_type)
    return extractor


def time_series_model(model_type='lstm', task='forecast'):
    """
    Создает модель для работы с временными рядами.

    Доступные варианты:
    --------------------
    1. 'lstm' для 'forecast' - LSTM для прогнозирования:
       - Рекуррентная нейронная сеть с долгой краткосрочной памятью
       - Учитывает долгосрочные зависимости во временных рядах
       - Подходит для нелинейных и сложных временных паттернов

    2. 'arima' для 'forecast' - ARIMA модель:
       - Статистическая модель для стационарных временных рядов
       - Авторегрессионная интегрированная скользящая средняя
       - Хорошо работает на линейных трендах и сезонности
       - Легко интерпретируется
    """
    if model_type == 'lstm' and task == 'forecast':
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            print("Установите tensorflow: pip install tensorflow")
            return None

        model = keras.Sequential([
            layers.LSTM(50, activation='relu', input_shape=(None, 1), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(50, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.description = "LSTM для прогнозирования"

    elif model_type == 'arima' and task == 'forecast':
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            print("Установите statsmodels: pip install statsmodels")
            return None

        class ARIMAModelWrapper:
            def __init__(self, order=(1, 1, 1)):
                self.order = order
                self.model = None

            def fit(self, data):
                self.model = ARIMA(data, order=self.order)
                self.model_fit = self.model.fit()
                return self

            def predict(self, steps=10):
                return self.model_fit.forecast(steps=steps)

        model = ARIMAModelWrapper(order=(1, 1, 1))
        model.description = "ARIMA модель"

    return model


# ==================== УТИЛИТЫ ====================

def create_pipeline(data_type, model_name=None):
    """
    Создает готовый пайплайн с предобученной моделью.

    Пайплайн (pipeline) - это последовательность шагов обработки данных и модели,
    объединенных в один объект для удобства обучения и предсказания.

    Доступные варианты:
    --------------------
    1. Для data_type='text':
       - 'tfidf_logistic': TF-IDF + логистическая регрессия

    2. Для data_type='image':
       - 'mobilenet': предобученная MobileNetV2
       - 'resnet': предобученная ResNet50
       - 'vgg': предобученная VGG16
       - 'efficientnet': предобученная EfficientNetB0
    """
    if data_type == 'text':
        if model_name is None:
            model_name = 'tfidf_logistic'

        if model_name == 'tfidf_logistic':
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline

            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='russian')),
                ('clf', LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced'))
            ])

    elif data_type == 'image':
        if model_name is None:
            model_name = 'mobilenet'
        pipeline = image_classifier(model_name)

    return pipeline