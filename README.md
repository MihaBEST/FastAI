
# FastAI Library

Быстрая библиотека для работы с машинным обучением и нейросетями.

## Установка

```bash
pip install -r requirements.txt
```

## Основные возможности

### Классификация
- Метод k-ближайших соседей (`kneighbors`)
- Дерево решений (`decision_tree`)
- Логистическая регрессия (`text_classifier`)
- Нейронные сети (`basic_network`, `cnn_network`)
- Классификация изображений (`t_image_classifier`)
- Классификация звуков (`t_sound_analiser`)

### Кластеризация
- K-means, DBSCAN, Agglomerative, GMM (`cluster_data`)

### Обработка данных
- Загрузка данных из различных форматов (`load_filedata`)
- Подготовка изображений (`prepare_images`)
- Подготовка аудио (`prepare_sound`)
- Текстовая нормализация (`text_normalise`, `ru_custom_normaliser`)
- Преобразование текста в числа (`text_to_num`)

### Готовые модели
- Многоклассовая классификация (`ovr_model`, `ovo_model`)
- Многозадачная регрессия (`multi_output_regression`)
- Ансамблевые модели (`ensemble_regression`)
- Предобученные модели изображений (`image_classifier`)
- Эмбеддинги текста (`text_embedding`)
- Анализ временных рядов (`time_series_model`)

## Быстрый старт

```python
import FastAI as ai

# Загрузка данных
X, y = ai.load_filedata('data.csv', Y_columns=1)

# Обучение модели
model = ai.t_kneighbors(X, y, test_size=0.2, choose_optimal=True)

# Нормализация текста
text = ai.ru_custom_normaliser("Привет, как дела?")
```

## Требования

Смотри `requirements.txt` для полного списка зависимостей.
