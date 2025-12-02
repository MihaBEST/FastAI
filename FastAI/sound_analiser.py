import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def t_sound_analiser(sounds, labels, /, test_size=0.2, random_state: int = 42):
    """
    Создает и обучает модель классификации звуков

    Args:
        sounds (list): Список числовых массивов с аудиоданными
        labels (list): Список меток классов
        test_size (float): Доля данных для тестирования (по умолчанию 0.2)
        random_state (int): Seed для воспроизводимости
    """

    # Проверка входных данных
    if len(sounds) != len(labels):
        raise ValueError(f"Количество звуков ({len(sounds)}) не совпадает с количеством меток ({len(labels)})")

    # Преобразование списка звуков в массив признаков
    # Предполагаем, что sounds уже содержит числовые признаки
    X = np.array(sounds)
    y = np.array(labels)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Создание и обучение модели
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        C=1.0,
        multi_class='auto',
        solver='lbfgs'
    )

    model.fit(X_train, y_train)

    # Оценка модели если есть тестовая выборка
    if test_size != 0:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Точность модели: {accuracy:.4f}")
        print("\nОтчет классификации:")
        print(classification_report(y_test, y_pred))

    return model

def sound_analiser():
    return LogisticRegression(
        max_iter=1000,
        C=1.0,
        multi_class='auto',
        solver='lbfgs'
    )