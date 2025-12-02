import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


def t_image_classifier(X, y,/,
                           input_shape=(64, 64, 3),  # Пример входной формы (высота, ширина, каналы)
                           num_classes=None,  # Количество классов, если известно заранее
                           test_size=0.2,  # Доля данных для валидации
                           random_state=42,  # Для воспроизводимости разделения данных
                           epochs=20,  # Количество эпох обучения
                           batch_size=32,  # Размер батча
                           learning_rate=0.001,  # Скорость обучения
                           optimizer='adam',  # Оптимизатор
                           loss='categorical_crossentropy',  # Функция потерь
                           metrics=['accuracy'],  # Метрики для оценки
                           verbose = 0
                           ):
    """
    Обучает модель классификации изображений.

    Args:
        X (np.ndarray): Массив изображений (предварительно обработанных).
                        Ожидаемая форма: (количество_изображений, высота, ширина, каналы).
        y (np.ndarray): Массив меток классов для каждого изображения.
                        Может быть числовым или строковым.
        input_shape (tuple, optional): Форма одного входного изображения (без учета батча).
                                       По умолчанию (64, 64, 3).
        num_classes (int, optional): Количество уникальных классов. Если None, будет определено
                                     из данных.
        test_size (float, optional): Доля данных, выделяемая для валидационной выборки.
                                     По умолчанию 0.2.
        random_state (int, optional): Число для инициализации генератора случайных чисел,
                                      чтобы обеспечить воспроизводимость разделения данных.
                                      По умолчанию 42.
        epochs (int, optional): Количество эпох обучения. По умолчанию 20.
        batch_size (int, optional): Размер батча для обучения. По умолчанию 32.
        learning_rate (float, optional): Скорость обучения оптимизатора. По умолчанию 0.001.
        optimizer (str, optional): Строка или объект оптимизатора. По умолчанию 'adam'.
        loss (str, optional): Строка или объект функции потерь. По умолчанию 'categorical_crossentropy'.
                              Используйте 'binary_crossentropy' для бинарной классификации.
        metrics (list, optional): Список метрик для оценки. По умолчанию ['accuracy'].

    Returns:
        tensorflow.keras.models.Model: Обученная модель классификации изображений.
                                       Возвращает None, если произошла ошибка.
    """

    print("--- Начало обучения модели ---")

    # 1. Предобработка меток классов
    print("Обработка меток классов...")
    if y.dtype == 'object' or y.dtype == 'str':
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print(f"Обнаружено классов: {len(label_encoder.classes_)}. Метки преобразованы.")
    else:
        y_encoded = y
        if num_classes is None:
            num_classes = len(np.unique(y_encoded))
        print(f"Обнаружено классов: {num_classes}. Используются числовые метки.")

    # Преобразование меток в one-hot encoding
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)

    # 2. Разделение данных на обучающую и валидационную выборки
    print(f"Разделение данных на обучающую ({1 - test_size:.0%}) и валидационную ({test_size:.0%}) выборки...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=test_size, random_state=random_state, stratify=y_categorical
    )
    print(f"Размеры: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")

    # 3. Создание модели (простая CNN)
    print("Создание модели CNN...")
    model = Sequential([
        # Сверточные слои
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Полносвязные слои
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),  # Регуляризация для предотвращения переобучения
        Dense(num_classes, activation='softmax')  # Softmax для многоклассовой классификации
    ])

    # 4. Компиляция модели
    print(f"Компиляция модели с оптимизатором '{optimizer}', функцией потерь '{loss}' и метриками {metrics}...")

    # Настройка скорости обучения для оптимизатора, если это Adam
    if optimizer == 'adam':
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.summary()  # Вывод структуры модели

    # 5. Обучение модели
    print(f"Начало обучения модели ({epochs} эпох, батч={batch_size})...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose  # 1 - показывать прогресс, 0 - молча, 2 - одна строка на эпоху
    )

    print("End of learning")

    # Дополнительно: можно вернуть history, чтобы посмотреть графики обучения
    # return model, history 
    return model
