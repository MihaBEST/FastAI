import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score


def t_basic_network(X, y, test_size: float = 0.1, epochs: int = 10, *neural_structure):
    """
    Создает и обучает полносвязную нейронную сеть.

    Parameters:
    -----------
    X : array-like
        Матрица признаков
    y : array-like
        Вектор целевых значений
    test_size : float
        Доля тестовой выборки
    epochs : int
        Количество эпох обучения
    *neural_structure : int or str
        Структура сети в формате: [нейроны_скрытые_слои, ..., выходной_размер]
        Последний параметр всегда задает размер выходного слоя
        Тип 1: только целые числа (количество нейронов)
        Тип 2: строки с функциями активации

    Act Functions (для типа 2):
    --------
    "relu" : [0, ∞) : скрытые слои, CNN\n
    "sigmoid" : (0, 1) : бинарная классификация, выходной слой\n
    "tanh" : (-1, 1) : скрытые слои RNN, центрированные данные\n
    "softmax" : (0, 1) : многоклассовая классификация, выходной слой\n
    "leaky_relu" : (-∞, ∞) : улучшенная версия relu\n
    "elu" : (-1, ∞) : улучшенная версия relu\n
    "swish" : (-∞, ∞) : современная альтернатива relu\n
    "linear" : (-∞, ∞) : регрессионные задачи, выходной слой\n

    Returns:
    --------
    model : keras.Model
        Обученная модель
    history : keras.History
        История обучения

    Examples:
    ---------
    # Тип 1: Простая структура
    t_basic_network(X, y, 0.1, 10, 64, 128, 256, 10)
    # 3 скрытых слоя: 64, 128, 256 нейронов, выходной слой 10 нейронов

    # Тип 2: Расширенная структура с функциями активации
    t_basic_network(X, y, 0.1, 10, "relu", 64, "relu", 128, "relu", 256, "softmax", 10)
    """

    # Проверка входных данных
    if len(neural_structure) < 2:
        raise ValueError(
            f"Недостаточно параметров структуры. Минимум 2: скрытые слои и выходной слой. Получено: {len(neural_structure)}")

    # Определяем тип структуры
    structure_type = 2 if any(isinstance(item, str) for item in neural_structure) else 1

    if structure_type == 1:
        # Проверяем, что все параметры - числа для типа 1
        for i, param in enumerate(neural_structure):
            if not isinstance(param, (int, np.integer)):
                raise ValueError(
                    f"Параметр {i + 1} должен быть целым числом для типа 1, получен {type(param)}: {param}")

        # Извлекаем выходной размер (последний параметр)
        output_size = neural_structure[-1]
        hidden_neurons = neural_structure[:-1]  # Все кроме последнего - скрытые слои
        print(f"Тип структуры: 1 (простая)")
        print(f"Скрытые слои: {hidden_neurons} нейронов")
        print(f"Выходной слой: {output_size} нейронов")

    else:
        # Для типа 2 проверяем, что последний параметр - число (выходной размер)
        if not isinstance(neural_structure[-1], (int, np.integer)):
            raise ValueError(
                f"Последний параметр должен быть целым числом (размер выходного слоя), получен {type(neural_structure[-1])}: {neural_structure[-1]}")

        output_size = neural_structure[-1]
        structure_params = neural_structure[:-1]  # Все кроме последнего - параметры структуры
        print(f"Тип структуры: 2 (расширенная)")
        print(f"Выходной слой: {output_size} нейронов")

    # Преобразуем данные в numpy массивы
    X = np.array(X)
    y = np.array(y)

    # Определяем тип задачи на основе выходного размера и данных
    if output_size == 1 and len(np.unique(y)) > 10 and np.issubdtype(y.dtype, np.number):
        problem_type = 'regression'
    elif output_size == 1:
        problem_type = 'binary_classification'
    else:
        problem_type = 'multiclass_classification'

    print(f"\nТип задачи: {problem_type}")

    # Подготовка данных для классификации
    if problem_type == 'multiclass_classification':
        if y.ndim == 1 or y.shape[1] == 1:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            num_classes = len(np.unique(y_encoded))

            # Проверяем соответствие выходного размера количеству классов
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

    # Разделяем данные
    stratify_param = None
    if problem_type in ['multiclass_classification', 'binary_classification'] and (y.ndim == 1 or y.shape[1] == 1):
        stratify_param = y_encoded if 'y_encoded' in locals() else y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_categorical,
        test_size=test_size,
        random_state=42,
        stratify=stratify_param
    )

    # Масштабируем признаки
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Признаки масштабированы (StandardScaler)")

    # Создание модели
    model = keras.Sequential()
    input_shape = X.shape[1]

    if structure_type == 1:
        # Тип 1: Простая структура
        # Входной слой
        model.add(layers.Dense(hidden_neurons[0], activation='relu', input_shape=(input_shape,)))
        model.add(layers.Dropout(0.2))
        print(f"Входной слой: {hidden_neurons[0]} нейронов, активация relu")

        # Скрытые слои
        for i in range(1, len(hidden_neurons)):
            model.add(layers.Dense(hidden_neurons[i], activation='relu'))
            if i < len(hidden_neurons) - 1:  # Не добавляем Dropout перед последним скрытым слоем
                model.add(layers.Dropout(0.2))
            print(f"Скрытый слой {i}: {hidden_neurons[i]} нейронов, активация relu")

    else:
        # Тип 2: Расширенная структура
        # Парсим структуру
        structure = []
        current_activation = 'relu'  # По умолчанию

        for item in structure_params:
            if isinstance(item, str):
                # Проверяем, является ли строка функцией активации
                activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax',
                                        'leaky_relu', 'elu', 'swish', 'linear']
                if item.lower() in activation_functions:
                    current_activation = item.lower()
                else:
                    raise ValueError(f"Неизвестная функция активации: {item}")
            elif isinstance(item, (int, np.integer)):
                structure.append((item, current_activation))
            else:
                raise ValueError(f"Неподдерживаемый тип параметра: {type(item)}")

        if not structure:
            raise ValueError("Не указаны слои нейронов в структуре")

        # Добавляем слои в модель
        # Первый слой (входной)
        neurons, activation = structure[0]
        model.add(layers.Dense(neurons, activation=activation, input_shape=(input_shape,)))
        model.add(layers.Dropout(0.2))
        print(f"Входной слой: {neurons} нейронов, активация {activation}")

        # Скрытые слои
        for i in range(1, len(structure)):
            neurons, activation = structure[i]
            model.add(layers.Dense(neurons, activation=activation))
            if i < len(structure) - 1:  # Не добавляем Dropout перед последним скрытым слоем
                model.add(layers.Dropout(0.2))
            print(f"Скрытый слой {i}: {neurons} нейронов, активация {activation}")

    # Выходной слой
    if problem_type == 'multiclass_classification':
        model.add(layers.Dense(output_size, activation='softmax'))
        print(f"Выходной слой: {output_size} нейронов, активация softmax")
    elif problem_type == 'binary_classification':
        model.add(layers.Dense(output_size, activation='sigmoid'))
        print(f"Выходной слой: {output_size} нейронов, активация sigmoid")
    else:  # regression
        model.add(layers.Dense(output_size, activation='linear'))
        print(f"Выходной слой: {output_size} нейронов, активация linear")

    # Компиляция модели
    if problem_type == 'multiclass_classification':
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Модель скомпилирована для многоклассовой классификации")
    elif problem_type == 'binary_classification':
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("Модель скомпилирована для бинарной классификации")
    else:  # regression
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        print("Модель скомпилирована для регрессии")

    # Вывод структуры сети
    print("\nСтруктура сети:")
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


def basic_network(input_shape: int, *neural_structure):
    """
    Создает необученную полносвязную нейронную сеть.

    Parameters:
    -----------
    input_shape : int
        Размерность входных данных (количество признаков)
    *neural_structure : int or str
        Структура сети в формате: [нейроны_скрытые_слои, ..., выходной_размер]
        Последний параметр всегда задает размер выходного слоя

    Returns:
    --------
    model : keras.Model
        Необученная модель

    Examples:
    ---------
    # Тип 1: Простая структура
    model = basic_network(784, 64, 128, 256, 10)

    # Тип 2: Расширенная структура
    model = basic_network(784, "relu", 64, "relu", 128, "relu", 256, "softmax", 10)
    """

    # Проверка входных данных
    if len(neural_structure) < 2:
        raise ValueError(
            f"Недостаточно параметров структуры. Минимум 2: скрытые слои и выходной слой. Получено: {len(neural_structure)}")

    # Определяем тип структуры
    structure_type = 2 if any(isinstance(item, str) for item in neural_structure) else 1

    if structure_type == 1:
        # Проверяем, что все параметры - числа для типа 1
        for i, param in enumerate(neural_structure):
            if not isinstance(param, (int, np.integer)):
                raise ValueError(
                    f"Параметр {i + 1} должен быть целым числом для типа 1, получен {type(param)}: {param}")

        output_size = neural_structure[-1]
        hidden_neurons = neural_structure[:-1]
        print(f"Создание полносвязной сети (тип 1)")
        print(f"Скрытые слои: {hidden_neurons} нейронов")
        print(f"Выходной слой: {output_size} нейронов")

    else:
        # Для типа 2 проверяем, что последний параметр - число
        if not isinstance(neural_structure[-1], (int, np.integer)):
            raise ValueError(
                f"Последний параметр должен быть целым числом (размер выходного слоя), получен {type(neural_structure[-1])}: {neural_structure[-1]}")

        output_size = neural_structure[-1]
        structure_params = neural_structure[:-1]
        print(f"Создание полносвязной сети (тип 2)")
        print(f"Выходной слой: {output_size} нейронов")

    # Определяем тип задачи на основе выходного размера
    if output_size == 1:
        problem_type = 'regression_or_binary'
    else:
        problem_type = 'multiclass_classification'

    print(f"Тип задачи предполагается как: {problem_type}")

    # Создание модели
    model = keras.Sequential()

    if structure_type == 1:
        # Тип 1: Простая структура
        # Входной слой
        model.add(layers.Dense(hidden_neurons[0], activation='relu', input_shape=(input_shape,)))
        model.add(layers.Dropout(0.2))

        # Скрытые слои
        for i in range(1, len(hidden_neurons)):
            model.add(layers.Dense(hidden_neurons[i], activation='relu'))
            if i < len(hidden_neurons) - 1:
                model.add(layers.Dropout(0.2))

    else:
        # Тип 2: Расширенная структура
        # Парсим структуру
        structure = []
        current_activation = 'relu'  # По умолчанию

        for item in structure_params:
            if isinstance(item, str):
                activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax',
                                        'leaky_relu', 'elu', 'swish', 'linear']
                if item.lower() in activation_functions:
                    current_activation = item.lower()
                else:
                    raise ValueError(f"Неизвестная функция активации: {item}")
            elif isinstance(item, (int, np.integer)):
                structure.append((item, current_activation))
            else:
                raise ValueError(f"Неподдерживаемый тип параметра: {type(item)}")

        if not structure:
            raise ValueError("Не указаны слои нейронов в структуре")

        # Первый слой (входной)
        neurons, activation = structure[0]
        model.add(layers.Dense(neurons, activation=activation, input_shape=(input_shape,)))
        model.add(layers.Dropout(0.2))

        # Скрытые слои
        for i in range(1, len(structure)):
            neurons, activation = structure[i]
            model.add(layers.Dense(neurons, activation=activation))
            if i < len(structure) - 1:
                model.add(layers.Dropout(0.2))

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
    print("\nСтруктура сети:")
    model.summary()

    return model