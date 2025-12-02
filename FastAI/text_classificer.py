from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def t_text_classifier(texts, labels,/, test_size=0.2, random_state:int=42, max_features=10000, stop_word:str="russian"):
    """
    Создает и обучает модель классификации текста на векторизованных данных

    Args:
        texts (list): Список текстов для обучения
        labels (list): Список меток классов
        test_size (float): Доля данных для тестирования (по умолчанию 0.2)
        random_state (int): Seed для воспроизводимости (по умолчанию 42)
        max_features (int): Максимальное количество признаков для векторизации

    Returns:
        dict: taught model
    """

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Векторизация текста с помощью TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_word,  # Удаление стоп-слов (можно заменить на 'russian')
        ngram_range=(1, 2),  # Учитываем униграммы и биграммы
        min_df=2,  # Минимальная частота термина
        max_df=0.8  # Максимальная частота термина
    )

    # Преобразование текстов в числовые векторы
    X_train_vec = vectorizer.fit_transform(X_train)

    # Создание и обучение модели
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        C=1.0
    )

    model.fit(X_train_vec, y_train)

    # Предсказания и оценка модели
    if test_size != 0:
        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)
        print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")

    return model


def text_classifier(random_state:int=42):
    """Not taught model"""
    return LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        C=1.0
    )
