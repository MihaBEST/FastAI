import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score


def t_cnn_network(X, y, test_size: float = 0.1, epochs: int = 10, *neural_structure):
    """
    Создает и обучает сверточную нейронную сеть (CNN) с простой структурой.

    Parameters:
    -----------
    X : array-like
        Трехмерный или четырехмерный массив изображений
        Формат: (n_samples, height, width, channels) или (n_samples, height, width)
    y : array-like
        Вектор целевых значений
    test_size : float
        Доля тестовой выборки
    *neural_structure : int
        Структура сети в формате: [фильтр1, фильтр2, ..., выходной_размер]
        Последний параметр всегда задает размер выходного слоя

    Examples:
    ---------
    # 3 сверточных слоя с 32, 64, 128 фильтрами и выходной слой на 10 классов
    t_cnn_network(X, y, 0.1, 10, 32, 64, 128, 10)

    # 2 сверточных слоя с 16, 32 фильтрами и выходной слой на 2 класса (бинарная классификация)
    t_cnn_network(X, y, 0.2, 20, 16, 32, 2)

    # 4 сверточных слоя и регрессия с 1 выходным значением
    t_cnn_network(X, y, 0.1, 15, 8, 16, 32, 64, 1)
    """

    # Проверка входных данных
    if len(neural_structure) < 2:
        raise ValueError(
            f"Недостаточно параметров структуры. Минимум 2: фильтры и выходной слой. Получено: {len(neural_structure)}")

    # Проверяем, что все параметры - числа
    for i, param in enumerate(neural_structure):
        if not isinstance(param, (int, np.integer)):
            raise ValueError(f"Параметр {i + 1} должен быть целым числом, получен {type(param)}: {param}")

    # Извлекаем выходной размер (последний параметр)
    output_size = neural_structure[-1]
    conv_filters = neural_structure[:-1]  # Все кроме последнего - фильтры сверточных слоев

    # Определяем тип задачи на основе выходного размера и данных
    if output_size == 1 and len(np.unique(y)) > 10 and np.issubdtype(y.dtype, np.number):
        problem_type = 'regression'
    elif output_size == 1:
        problem_type = 'binary_classification'
    else:
        problem_type = 'multiclass_classification'

    print(f"Тип задачи: {problem_type}")
    print(f"Структура сети: {conv_filters} сверточных слоев -> выходной слой {output_size}")

    # Подготовка данных для классификации
    if problem_type == 'multiclass_classification':
        if y.ndim == 1 or y.shape[1] == 1:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            num_classes = len(np.unique(y_encoded))
            if output_size != num_classes:
                print(
                    f"Внимание: заданный выходной размер ({output_size}) не совпадает с количеством классов ({num_classes})")
                print(f"Используется фактическое количество классов: {num_classes}")
                output_size = num_classes
            y_categorical = keras.utils.to_categorical(y_encoded, num_classes)
        else:
            y_categorical = y
            num_classes = y.shape[1]
    elif problem_type == 'binary_classification':
        # Для бинарной классификации используем одну выходную нейронку с сигмоидой
        y_categorical = y.reshape(-1, 1) if y.ndim == 1 else y
        if output_size != 1:
            print(f"Внимание: для бинарной классификации выходной размер должен быть 1, получен {output_size}")
            output_size = 1
    else:  # regression
        y_categorical = y.reshape(-1, 1) if y.ndim == 1 else y
        if output_size != 1:
            print(f"Внимание: для регрессии выходной размер должен быть 1, получен {output_size}")
            output_size = 1

    # Преобразуем X в правильный формат для CNN
    X_array = np.array(X)

    # Если данные 2D (без каналов), добавляем dimension для каналов
    if X_array.ndim == 3:
        X_array = X_array[..., np.newaxis]  # (samples, height, width, 1)
        print(f"Добавлен канал к данным. Новая форма: {X_array.shape}")

    # Определяем input_shape
    input_shape = X_array.shape[1:]  # (height, width, channels)
    print(f"Input shape: {input_shape}")

    # Разделение данных
    stratify_param = None
    if problem_type in ['multiclass_classification', 'binary_classification'] and (y.ndim == 1 or y.shape[1] == 1):
        stratify_param = y

    X_train, X_test, y_train, y_test = train_test_split(
        X_array,
        y_categorical,
        test_size=test_size,
        random_state=42,
        stratify=stratify_param
    )

    # Нормализация изображений (приведение к диапазону 0-1)
    if X_train.dtype != np.float32 and X_train.dtype != np.float64:
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        print("Данные нормализованы (деление на 255)")
    else:
        if X_train.max() > 1.0:
            X_train = X_train / 255.0
            X_test = X_test / 255.0
            print("Данные нормализованы (деление на 255)")
        else:
            print("Данные уже нормализованы (диапазон 0-1)")

    # Создание модели
    model = keras.Sequential()

    # Добавляем входной слой
    model.add(layers.Input(shape=input_shape))

    # Добавляем сверточные блоки
    for i, filters in enumerate(conv_filters):
        # Сверточный слой
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())

        # Добавляем MaxPooling после каждого сверточного слоя, кроме последнего
        if i < len(conv_filters) - 1:
            model.add(layers.MaxPooling2D((2, 2)))

        # Добавляем Dropout (более высокий для более глубоких слоев)
        dropout_rate = min(0.1 + i * 0.05, 0.5)  # Увеличиваем dropout с глубиной
        model.add(layers.Dropout(dropout_rate))

    # Преобразование в вектор
    model.add(layers.Flatten())

    # Добавляем полносвязный слой перед выходом (опционально)
    if len(conv_filters) > 1:
        dense_units = max(conv_filters[-1] // 2, 32)
        model.add(layers.Dense(dense_units, activation='relu'))
        model.add(layers.Dropout(0.5))
        print(f"Добавлен полносвязный слой с {dense_units} нейронами")

    # Выходной слой
    if problem_type == 'multiclass_classification':
        model.add(layers.Dense(output_size, activation='softmax'))
        print(f"Выходной слой: {output_size} нейронов с активацией softmax")
    elif problem_type == 'binary_classification':
        model.add(layers.Dense(output_size, activation='sigmoid'))
        print(f"Выходной слой: {output_size} нейронов с активацией sigmoid")
    else:  # regression
        model.add(layers.Dense(output_size, activation='linear'))
        print(f"Выходной слой: {output_size} нейронов с активацией linear")

    # Компиляция модели
    if problem_type == 'multiclass_classification':
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Скомпилирована модель для многоклассовой классификации")
    elif problem_type == 'binary_classification':
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("Скомпилирована модель для бинарной классификации")
    else:  # regression
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        print("Скомпилирована модель для регрессии")

    # Вывод структуры сети
    print("\nСтруктура CNN:")
    model.summary()

    # Обучение модели
    print(f"\nНачало обучения на {epochs} эпох...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=epochs,
        validation_split=0.2,
        verbose=1
    )

    # Оценка модели
    print("\nОценка на тестовых данных...")
    if problem_type in ['multiclass_classification', 'binary_classification']:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Точность на тестовых данных: {test_accuracy:.4f}")
        print(f"Потери на тестовых данных: {test_loss:.4f}")

        # Предсказания и дополнительная метрика accuracy
        y_pred = model.predict(X_test, verbose=0)

        if problem_type == 'multiclass_classification':
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
        else:  # binary_classification
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            y_true_classes = y_test.flatten()

        acc_score = accuracy_score(y_true_classes, y_pred_classes)
        print(f"Accuracy score: {acc_score:.4f}")

    else:  # regression
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"MSE на тестовых данных: {test_loss:.4f}")
        print(f"MAE на тестовых данных: {test_mae:.4f}")

        # Пример предсказания
        y_pred = model.predict(X_test[:5], verbose=0)
        print(f"Пример предсказаний (первые 5): {y_pred.flatten()}")
        print(f"Фактические значения (первые 5): {y_test[:5].flatten()}")

    return model, history


def cnn_network(input_shape: tuple, *neural_structure):
    """
    Создает необученную сверточную нейронную сеть с простой структурой.

    Parameters:
    -----------
    input_shape : tuple
        Формат входных данных (height, width, channels)
    *neural_structure : int
        Структура сети в формате: [фильтр1, фильтр2, ..., выходной_размер]
        Последний параметр всегда задает размер выходного слоя

    Returns:
    --------
    model : keras.Model
        Необученная модель CNN

    Examples:
    ---------
    model = cnn_network((28, 28, 1), 32, 64, 128, 10)
    """

    # Проверка входных данных
    if len(neural_structure) < 2:
        raise ValueError(
            f"Недостаточно параметров структуры. Минимум 2: фильтры и выходной слой. Получено: {len(neural_structure)}")

    # Проверяем, что все параметры - числа
    for i, param in enumerate(neural_structure):
        if not isinstance(param, (int, np.integer)):
            raise ValueError(f"Параметр {i + 1} должен быть целым числом, получен {type(param)}: {param}")

    # Извлекаем выходной размер (последний параметр)
    output_size = neural_structure[-1]
    conv_filters = neural_structure[:-1]  # Все кроме последнего - фильтры сверточных слоев

    # Определяем тип задачи на основе выходного размера
    if output_size == 1:
        problem_type = 'regression_or_binary'
    else:
        problem_type = 'multiclass_classification'

    print(f"Создание CNN с структурой: {conv_filters} сверточных слоев -> выходной слой {output_size}")
    print(f"Тип задачи предполагается как: {problem_type}")

    # Создание модели
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Добавляем сверточные блоки
    for i, filters in enumerate(conv_filters):
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())

        if i < len(conv_filters) - 1:
            model.add(layers.MaxPooling2D((2, 2)))

        dropout_rate = min(0.1 + i * 0.05, 0.5)
        model.add(layers.Dropout(dropout_rate))

    # Преобразование в вектор
    model.add(layers.Flatten())

    # Опциональный полносвязный слой
    if len(conv_filters) > 1:
        dense_units = max(conv_filters[-1] // 2, 32)
        model.add(layers.Dense(dense_units, activation='relu'))
        model.add(layers.Dropout(0.5))

    # Выходной слой
    if problem_type == 'multiclass_classification':
        model.add(layers.Dense(output_size, activation='softmax'))
    else:  # regression_or_binary
        model.add(layers.Dense(output_size, activation='sigmoid' if output_size == 1 else 'linear'))

    # Компиляция модели (универсальная)
    if problem_type == 'multiclass_classification':
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy' if output_size == 1 else 'mse',
            metrics=['accuracy' if output_size == 1 else 'mae']
        )

    # Вывод структуры
    print("\nСтруктура CNN:")
    model.summary()

    return model