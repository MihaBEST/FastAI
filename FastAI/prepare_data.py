import pandas as pd
import json
import os
import warnings
import numpy as np

def load_filedata(file: str, Y_columns: int = 1, **kwargs):
    """
    Loads data from various file formats and databases, returns x (all columns except the last one) and y (the last column)

    Parameters:
    file (str): the path to the file or database connection string
    Y_columns (int): how many columns is y data
    **kwargs: additional parameters for specific loaders (e.g., sheet_name for Excel, table_name for SQL)

    Returns:
    tuple: (x, y) where x is all signs except the last column, y is the last column
    """

    # Defining the file extension
    file_extension = os.path.splitext(file)[1].lower() if not file.startswith(
        ('sql://', 'postgresql://', 'mysql://', 'oracle://')) else 'sql'

    try:
        df = None

        if file_extension == '.csv':
            # Load CSV
            df = pd.read_csv(file, **kwargs)

        elif file_extension == '.json':
            # Load JSON
            df = pd.read_json(file, **kwargs)

        elif file_extension in ['.xlsx', '.xls', '.xlsm']:
            # Load Excel files
            sheet_name = kwargs.get('sheet_name', 0)
            df = pd.read_excel(file, sheet_name=sheet_name, **{k: v for k, v in kwargs.items() if k != 'sheet_name'})

        elif file_extension == '.parquet':
            # Load Parquet
            df = pd.read_parquet(file, **kwargs)

        elif file_extension == '.feather':
            # Load Feather
            df = pd.read_feather(file, **kwargs)

        elif file_extension == '.h5' or file_extension == '.hdf5':
            # Load HDF5
            key = kwargs.get('key', 'data')
            df = pd.read_hdf(file, key=key, **{k: v for k, v in kwargs.items() if k != 'key'})

        elif file_extension == '.pkl' or file_extension == '.pickle':
            # Load Pickle
            df = pd.read_pickle(file, **kwargs)

        elif file_extension == '.dta':
            # Load Stata
            df = pd.read_stata(file, **kwargs)

        elif file_extension == '.sas7bdat':
            # Load SAS
            df = pd.read_sas(file, **kwargs)

        elif file_extension == '.sav':
            # Load SPSS
            df = pd.read_spss(file, **kwargs)

        elif file_extension == '.xml':
            # Load XML
            xpath = kwargs.get('xpath', './/row')
            df = pd.read_xml(file, xpath=xpath, **{k: v for k, v in kwargs.items() if k != 'xpath'})

        elif file_extension == '.html':
            # Load HTML tables
            match = kwargs.get('match', '.+')
            df = pd.read_html(file, match=match, **{k: v for k, v in kwargs.items() if k != 'match'})[0]

        elif file_extension == '.txt' or file_extension == '.tsv':
            # Try different separators for text files
            separators = [',', '\t', ' ', ';', '|']
            for sep in separators:
                try:
                    df = pd.read_csv(file, sep=sep, **kwargs)
                    break
                except:
                    continue
            if df is None:
                raise ValueError(f"Could not parse text file with common separators: {separators}")

        elif file_extension == 'sql':
            # Load from SQL database
            return load_from_sql(file, Y_columns, **kwargs)

        elif file_extension == '.orc':
            # Load ORC file
            df = pd.read_orc(file, **kwargs)

        else:
            # Try to load with generic read_csv as fallback
            try:
                df = pd.read_csv(file, **kwargs)
                warnings.warn(f"Unknown extension {file_extension}, loaded as CSV")
            except:
                raise ValueError(f"Unsupported format: {file_extension}")

        # File has data
        if df is None or df.empty:
            raise ValueError("Empty file or no data loaded")

        return split_features_target(df, Y_columns)

    except Exception as e:
        raise Exception(f"Load Error {file}: {str(e)}")


def load_from_sql(connection_string: str, Y_columns: int = 1, **kwargs):
    """
    Load data from SQL database

    Parameters:
    connection_string (str): SQLAlchemy connection string
    Y_columns (int): how many columns is y data
    **kwargs: table_name or query required, plus other SQL parameters
    """
    from sqlalchemy import create_engine, text

    table_name = kwargs.get('table_name')
    query = kwargs.get('query')

    if not table_name and not query:
        raise ValueError("Either 'table_name' or 'query' must be provided for SQL loading")

    engine = create_engine(connection_string)

    try:
        if query:
            df = pd.read_sql(query, engine)
        else:
            df = pd.read_sql_table(table_name, engine)

        return split_features_target(df, Y_columns)

    finally:
        engine.dispose()


def split_features_target(df: pd.DataFrame, Y_columns: int):
    """
    Split dataframe into features (X) and target (Y)
    """
    if Y_columns != 0:
        x = df.iloc[:, :-Y_columns]
        y = df.iloc[:, -Y_columns:]
        # If Y_columns == 1, return series instead of dataframe for backward compatibility
        if Y_columns == 1:
            y = y.iloc[:, 0]
        return x, y
    else:
        return df


def del_columns(data: pd.DataFrame, *names, by_index: bool = False) -> pd.DataFrame:
    """
    Удаляет указанные столбцы из DataFrame по именам или индексам

    Parameters:
    data (pd.DataFrame): исходный DataFrame
    *names: названия столбцов или индексы для удаления
    by_index (bool): если True, удаляет по индексам, иначе по именам

    Returns:
    pd.DataFrame: DataFrame без указанных столбцов
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data должен быть pandas DataFrame")

    if by_index:
        # Удаление по индексам
        indices_to_drop = list(names)
        valid_indices = [idx for idx in indices_to_drop if 0 <= idx < len(data.columns)]

        if valid_indices:
            return data.drop(data.columns[valid_indices], axis=1)
        else:
            print("Предупреждение: не найдено ни одного валидного индекса для удаления")
            return data
    else:
        # Удаление по именам (стандартное поведение)
        return data.drop(columns=list(names))


from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
import pickle
import json


def text_to_num(texts: Union[List[str], pd.Series, np.ndarray],
                          max_sequence_length: Optional[int] = None,
                          vocabulary_size: Optional[int] = None,
                          tokenization_method: str = 'word',
                          padding_strategy: str = 'post',
                          truncation_strategy: str = 'post',
                          return_vocabulary: bool = False,
                          existing_vocabulary: Optional[Dict] = None) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Преобразует тексты в числовой формат для нейросетей.

    Args:
        texts: Список текстов для преобразования
        max_sequence_length: Максимальная длина последовательности (обрезает/дополняет)
        vocabulary_size: Максимальный размер словаря (ограничивает количество слов)
        tokenization_method: Метод токенизации ('word', 'char', 'subword')
        padding_strategy: Стратегия дополнения ('pre', 'post')
        truncation_strategy: Стратегия обрезки ('pre', 'post')
        return_vocabulary: Возвращать ли словарь вместе с результатом
        existing_vocabulary: Существующий словарь для кодирования (для предобученных моделей)

    Returns:
        Массив в формате (n_samples, max_sequence_length) или кортеж (массив, словарь)
    """

    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()

    if not texts:
        raise ValueError("Список текстов не должен быть пустым")

    # Токенизация текстов
    tokenized_texts = []

    for text in texts:
        if not isinstance(text, str):
            text = str(text)

        if tokenization_method == 'word':
            # Простая токенизация по словам
            tokens = text.split()
        elif tokenization_method == 'char':
            # Токенизация по символам
            tokens = list(text)
        elif tokenization_method == 'subword':
            # Базовая субсловная токенизация
            tokens = []
            words = text.split()
            for word in words:
                if len(word) <= 3:
                    tokens.append(word)
                else:
                    # Разбиваем длинные слова на подстроки
                    for i in range(0, len(word), 2):
                        tokens.append(word[i:i + 2])
        else:
            raise ValueError(f"Неизвестный метод токенизации: {tokenization_method}")

        tokenized_texts.append(tokens)

    # Создание или использование словаря
    if existing_vocabulary is not None:
        vocab = existing_vocabulary
    else:
        vocab = build_vocabulary(tokenized_texts, vocabulary_size)

    # Преобразование текстов в числовые последовательности
    sequences = []

    for tokens in tokenized_texts:
        sequence = []
        for token in tokens:
            if token in vocab:
                sequence.append(vocab[token])
            else:
                sequence.append(vocab.get('<UNK>', 0))  # Для неизвестных токенов

        sequences.append(sequence)

    # Определение максимальной длины последовательности
    if max_sequence_length is None:
        max_sequence_length = max(len(seq) for seq in sequences)

    # Паддинг и обрезка последовательностей
    padded_sequences = pad_and_truncate_sequences(
        sequences,
        max_length=max_sequence_length,
        padding_strategy=padding_strategy,
        truncation_strategy=truncation_strategy
    )

    result_array = np.array(padded_sequences, dtype=np.int32)

    if return_vocabulary:
        return result_array, vocab
    else:
        return result_array


def build_vocabulary(tokenized_texts: List[List[str]],
                     vocabulary_size: Optional[int] = None) -> Dict[str, int]:
    """
    Строит словарь из токенизированных текстов.

    Args:
        tokenized_texts: Список токенизированных текстов
        vocabulary_size: Максимальный размер словаря

    Returns:
        Словарь {токен: индекс}
    """
    # Подсчет частоты токенов
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    # Сортировка по частоте
    sorted_tokens = [token for token, _ in counter.most_common()]

    # Ограничение размера словаря
    if vocabulary_size is not None:
        if vocabulary_size > 0:
            sorted_tokens = sorted_tokens[:vocabulary_size - 2]  # Оставляем место для спецтокенов

    # Создание словаря
    vocabulary = {
        '<PAD>': 0,  # Для дополнения
        '<UNK>': 1,  # Для неизвестных слов
        '<START>': 2,  # Начало последовательности
        '<END>': 3  # Конец последовательности
    }

    # Добавление токенов с учетом оффсета
    offset = len(vocabulary)
    for idx, token in enumerate(sorted_tokens):
        vocabulary[token] = idx + offset

    return vocabulary


def pad_and_truncate_sequences(sequences: List[List[int]],
                               max_length: int,
                               padding_strategy: str = 'post',
                               truncation_strategy: str = 'post',
                               padding_value: int = 0) -> List[List[int]]:
    """
    Дополняет и обрезает последовательности до фиксированной длины.

    Args:
        sequences: Список числовых последовательностей
        max_length: Желаемая длина последовательностей
        padding_strategy: 'pre' или 'post'
        truncation_strategy: 'pre' или 'post'
        padding_value: Значение для дополнения

    Returns:
        Список последовательностей одинаковой длины
    """
    processed_sequences = []

    for sequence in sequences:
        # Обрезка если нужно
        if len(sequence) > max_length:
            if truncation_strategy == 'pre':
                sequence = sequence[-max_length:]
            else:  # 'post'
                sequence = sequence[:max_length]

        # Дополнение если нужно
        if len(sequence) < max_length:
            padding = [padding_value] * (max_length - len(sequence))
            if padding_strategy == 'pre':
                sequence = padding + sequence
            else:  # 'post'
                sequence = sequence + padding

        processed_sequences.append(sequence)

    return processed_sequences


def create_text_embeddings(texts: Union[List[str], pd.Series],
                           method: str = 'tfidf',
                           max_features: int = 1000,
                           ngram_range: Tuple[int, int] = (1, 1),
                           return_vectorizer: bool = False):
    """
    Создает эмбеддинги текстов для нейросетей.

    Args:
        texts: Список текстов
        method: Метод векторизации ('tfidf', 'count', 'binary')
        max_features: Максимальное количество признаков
        ngram_range: Диапазон для n-грамм
        return_vectorizer: Возвращать ли векторизатор вместе с результатом

    Returns:
        Матрица эмбеддингов или кортеж (матрица, векторизатор)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    if isinstance(texts, pd.Series):
        texts = texts.tolist()

    # Выбор векторизатора
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,
            lowercase=True
        )
    elif method == 'count':
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,
            lowercase=True,
            binary=False
        )
    elif method == 'binary':
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,
            lowercase=True,
            binary=True
        )
    else:
        raise ValueError(f"Неизвестный метод векторизации: {method}")

    # Преобразование текстов в матрицу признаков
    embeddings = vectorizer.fit_transform(texts).toarray()

    if return_vectorizer:
        return embeddings, vectorizer
    else:
        return embeddings


def save_text_preprocessor(vocabulary: Dict,
                           params: Dict,
                           filepath: str):
    """
    Сохраняет предобработчик текста для последующего использования.

    Args:
        vocabulary: Словарь токенов
        params: Параметры предобработки
        filepath: Путь для сохранения
    """
    preprocessor_data = {
        'vocabulary': vocabulary,
        'params': params
    }

    with open(filepath, 'wb') as f:
        pickle.dump(preprocessor_data, f)


def load_text_preprocessor(filepath: str) -> Tuple[Dict, Dict]:
    """
    Загружает сохраненный предобработчик текста.

    Args:
        filepath: Путь к файлу

    Returns:
        Кортеж (словарь, параметры)
    """
    with open(filepath, 'rb') as f:
        preprocessor_data = pickle.load(f)

    return preprocessor_data['vocabulary'], preprocessor_data['params']


def data_normalise(data: pd.DataFrame,
                            numeric_method: str = 'zscore',
                            numeric_columns: Optional[List] = None,
                            passthrough_columns: Optional[List] = None,
                            add_imputation: bool = False) -> pd.DataFrame:
    """
    Продвинутая нормализация с Pipeline и ColumnTransformer.

    Parameters:
    -----------
    data : pd.DataFrame
        Входные данные
    numeric_method : str
        Метод для числовых колонок
    numeric_columns : list, optional
        Список числовых колонок для нормализации
        minmax, zscore, maxabs, robust, l2, power, quantile
    passthrough_columns : list, optional
        Колонки, которые нужно пропустить без изменений
    add_imputation : bool
        Добавить импутацию пропущенных значений
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, \
        PowerTransformer, QuantileTransformer
    from sklearn.compose import ColumnTransformer

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    if passthrough_columns is None:
        passthrough_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()

    # Определение steps для pipeline
    steps = []
    if add_imputation:
        steps.append(('imputer', SimpleImputer(strategy='median')))

    # Добавление нормализации
    scalers = {
        'minmax': MinMaxScaler(),
        'zscore': StandardScaler(),
        'maxabs': MaxAbsScaler(),
        'robust': RobustScaler(),
        'l2': Normalizer(),
        'power': PowerTransformer(),
        'quantile': QuantileTransformer(output_distribution='normal')
    }

    steps.append(('scaler', scalers.get(numeric_method, StandardScaler())))

    # Создание pipeline
    numeric_pipeline = Pipeline(steps)

    # Создание ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_columns),
            ('pass', 'passthrough', passthrough_columns)
        ],
        remainder='drop'
    )

    # Применение преобразований
    normalized_array = preprocessor.fit_transform(data)

    # Восстановление DataFrame
    normalized_columns = numeric_columns + passthrough_columns
    result = pd.DataFrame(normalized_array, columns=normalized_columns, index=data.index)

    return result