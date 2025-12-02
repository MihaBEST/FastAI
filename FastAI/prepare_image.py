from PIL import Image, ImageFilter, ImageEnhance
import os
import numpy as np


def prepare_images(images, output_dir: str = None, target_width: int = 16,
                   target_height: int = 16, filters_count: int = 0):
    """
    Преобразует изображения: изменяет размер, применяет фильтры и сохраняет при необходимости.

    Args:
        images: Список изображений (PIL.Image или пути к файлам)
        output_dir (str): Директория для сохранения результатов
        target_width (int): Ширина целевого изображения
        target_height (int): Высота целевого изображения
        filters_count (int): Количество фильтров для применения к каждому изображению

    Returns:
        list: Список преобразованных изображений
    """

    processed_images = []

    # Создаем выходную директорию если указана
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, img in enumerate(images):
        # Если передан путь к файлу, загружаем изображение
        if isinstance(img, str):
            img = Image.open(img)

        # Базовое преобразование: изменение размера
        resized_img = img.resize((target_width, target_height), Image.LANCZOS)
        processed_images.append(resized_img)

        # Применяем фильтры если нужно
        if filters_count > 0:
            filtered_imgs = apply_filters(resized_img, filters_count)
            processed_images.extend(filtered_imgs)

            # Сохраняем изображения с фильтрами если нужно
            if output_dir:
                for j, filtered_img in enumerate(filtered_imgs):
                    filter_filename = f"image_{i}_filter_{j}.png"
                    filtered_img.save(os.path.join(output_dir, filter_filename))

    return processed_images


def apply_filters(image, count: int):
    """
    Применяет различные фильтры к изображению

    Args:
        image: Исходное изображение (PIL.Image)
        count (int): Количество фильтров для применения

    Returns:
        list: Список изображений с примененными фильтрами
    """
    filtered_images = []
    filters = [
        # Фильтр размытия
        lambda img: img.filter(ImageFilter.BLUR),
        # Фильтр резкости
        lambda img: img.filter(ImageFilter.SHARPEN),
        # Фильтр контура
        lambda img: img.filter(ImageFilter.CONTOUR),
        # Фильтр детализации
        lambda img: img.filter(ImageFilter.DETAIL),
        # Фильтр рельефа
        lambda img: img.filter(ImageFilter.EMBOSS),
        # Фильтр поиска краев
        lambda img: img.filter(ImageFilter.FIND_EDGES),
        # Увеличение яркости
        lambda img: ImageEnhance.Brightness(img).enhance(1.5),
        # Уменьшение яркости
        lambda img: ImageEnhance.Brightness(img).enhance(0.7),
        # Увеличение контрастности
        lambda img: ImageEnhance.Contrast(img).enhance(1.5),
        # Уменьшение контрастности
        lambda img: ImageEnhance.Contrast(img).enhance(0.7),
        # Увеличение насыщенности
        lambda img: ImageEnhance.Color(img).enhance(1.5),
        # Уменьшение насыщенности
        lambda img: ImageEnhance.Color(img).enhance(0.7),
        # Чёрно-белое
        lambda img: img.convert('L').convert('RGB'),
        # Сепия
        lambda img: apply_sepia_filter(img),
        # Инверсия цветов
        lambda img: Image.eval(img, lambda x: 255 - x),
        # Гауссово размытие
        lambda img: img.filter(ImageFilter.GaussianBlur(radius=1)),
    ]

    # Ограничиваем количество фильтров доступным списком
    count = min(count, len(filters))

    # Применяем случайные фильтры
    import random
    selected_filters = random.sample(filters, count)

    for filter_func in selected_filters:
        try:
            filtered_img = filter_func(image.copy())
            filtered_images.append(filtered_img)
        except Exception as e:
            print(f"Ошибка применения фильтра: {e}")

    return filtered_images


def apply_sepia_filter(image):
    """
    Применяет сепия-фильтр к изображению
    """
    # Преобразуем в numpy array для обработки
    img_array = np.array(image)

    # Коэффициенты для сепии
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    # Применяем матричное преобразование
    sepia_img = np.dot(img_array, sepia_matrix.T)
    # Ограничиваем значения 0-255
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)

    return Image.fromarray(sepia_img)


def load_images(path_dir):
    """
    Загружает все изображения из указанной директории

    Args:
        path_dir (str): Путь к директории с изображениями

    Returns:
        list: Список объектов PIL.Image
    """

    images = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')

    if not os.path.exists(path_dir):
        raise FileNotFoundError(f"Can't find {path_dir}")

    for filename in os.listdir(path_dir):
        if filename.lower().endswith(supported_formats):
            try:
                img_path = os.path.join(path_dir, filename)
                img = Image.open(img_path)
                images.append(img)
            except Exception as e:
                raise ValueError(f"Loading error, image {filename}: {e}")

    return images


def images_to_num(image_dir, target_width=16, target_height=16,
                               filters_count=0, normalize=True, flatten=False):
    """
    Преобразует все изображения из папки в массив numpy для нейросети

    Args:
        image_dir (str): Путь к директории с изображениями
        target_width (int): Ширина целевого изображения
        target_height (int): Высота целевого изображения
        filters_count (int): Количество фильтров для применения к каждому изображению
        normalize (bool): Нормализовать ли значения пикселей в диапазон [0, 1]
        flatten (bool): Преобразовать ли изображения в одномерные векторы

    Returns:
        tuple: (X, y) где X - массив изображений, y - массив меток (если есть)
               или только X если нет файлов с метками
    """

    # Загружаем изображения
    images = load_images(image_dir)

    if not images:
        print(f"В директории {image_dir} не найдено изображений")
        return np.array([]), np.array([])

    # Обрабатываем изображения
    processed_images = prepare_images(
        images,
        output_dir=None,
        target_width=target_width,
        target_height=target_height,
        filters_count=filters_count
    )

    # Преобразуем PIL изображения в numpy массивы
    image_arrays = []

    for img in processed_images:
        # Конвертируем в RGB если не RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Преобразуем в numpy array
        img_array = np.array(img)

        # Нормализуем если нужно
        if normalize:
            img_array = img_array.astype(np.float32) / 255.0

        # Выравниваем если нужно
        if flatten:
            img_array = img_array.flatten()

        image_arrays.append(img_array)

    # Преобразуем список в numpy array
    X = np.array(image_arrays)

    # Пытаемся загрузить метки если есть соответствующий файл
    y = None
    labels_file = os.path.join(image_dir, 'labels.txt')
    labels_npy_file = os.path.join(image_dir, 'labels.npy')

    if os.path.exists(labels_file):
        # Загружаем метки из текстового файла
        try:
            with open(labels_file, 'r') as f:
                labels = [int(line.strip()) for line in f.readlines()]
            # Расширяем метки для аугментированных изображений
            if filters_count > 0:
                expanded_labels = []
                for label in labels:
                    expanded_labels.extend([label] * (1 + filters_count))
                y = np.array(expanded_labels[:len(X)])  # Обрезаем до нужной длины
            else:
                y = np.array(labels[:len(X)])
        except Exception as e:
            print(f"Ошибка загрузки меток из {labels_file}: {e}")

    elif os.path.exists(labels_npy_file):
        # Загружаем метки из numpy файла
        try:
            labels = np.load(labels_npy_file)
            if filters_count > 0:
                expanded_labels = []
                for label in labels:
                    expanded_labels.extend([label] * (1 + filters_count))
                y = np.array(expanded_labels[:len(X)])
            else:
                y = labels[:len(X)]
        except Exception as e:
            print(f"Ошибка загрузки меток из {labels_npy_file}: {e}")

    # Если метки не найдены, возвращаем только X
    if y is None or len(y) == 0:
        print("Метки не найдены. Возвращаются только изображения.")
        return X

    return X, y