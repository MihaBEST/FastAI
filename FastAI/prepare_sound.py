import librosa
import soundfile as sf
from pathlib import Path
from typing import Union, List, Tuple
import numpy as np
import os

def prepare_sound(input_data: Union[str, np.ndarray, List],
                  target_sr: int = 22050,
                  duration: float = None,
                  target_length: int = None,
                  mono: bool = True,
                  normalize: bool = True,
                  preemph: float = 0.97) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Подготавливает звук(и) для нейросети

    Parameters:
    input_data: путь к файлу, папке, numpy array или список путей
    target_sr: целевая частота дискретизации (default: 22050)
    duration: целевая длительность в секундах (если None - исходная длительность)
    target_length: целевое количество семплов (приоритет над duration)
    mono: преобразовать в моно (default: True)
    normalize: нормализовать амплитуду (default: True)
    preemph: коэффициент предэмфазы (default: 0.97, None чтобы отключить)

    Returns:
    numpy array или список numpy arrays с подготовленными звуками
    """

    # Обработка разных типов входных данных
    if isinstance(input_data, (str, Path)):
        if os.path.isdir(input_data):
            return _process_folder(input_data, target_sr, duration, target_length, mono, normalize, preemph)
        else:
            return _process_single_file(input_data, target_sr, duration, target_length, mono, normalize, preemph)

    elif isinstance(input_data, np.ndarray):
        return _process_array(input_data, target_sr, duration, target_length, mono, normalize, preemph)

    elif isinstance(input_data, list):
        return _process_list(input_data, target_sr, duration, target_length, mono, normalize, preemph)

    else:
        raise TypeError(f"Неподдерживаемый тип входных данных: {type(input_data)}")


def _process_folder(folder_path: str, target_sr: int, duration: float, target_length: int,
                    mono: bool, normalize: bool, preemph: float) -> List[np.ndarray]:
    """Обработка папки с аудиофайлами"""
    audio_files = []
    supported_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac', '.wma'}

    for file_path in Path(folder_path).iterdir():
        if file_path.suffix.lower() in supported_extensions and file_path.is_file():
            audio_files.append(str(file_path))

    if not audio_files:
        raise ValueError(f"В папке {folder_path} не найдено поддерживаемых аудиофайлов")

    return _process_list(audio_files, target_sr, duration, target_length, mono, normalize, preemph)


def _process_single_file(file_path: str, target_sr: int, duration: float, target_length: int,
                         mono: bool, normalize: bool, preemph: float) -> np.ndarray:
    """Обработка одиночного файла"""
    try:
        # Загрузка аудио с помощью librosa
        audio, sr = librosa.load(file_path, sr=target_sr, mono=mono, duration=duration)

        # Обработка аудио
        processed_audio = _process_audio(audio, sr, target_sr, target_length, normalize, preemph)

        return processed_audio

    except Exception as e:
        raise ValueError(f"Ошибка загрузки файла {file_path}: {str(e)}")


def _process_array(audio: np.ndarray, target_sr: int, duration: float, target_length: int,
                   mono: bool, normalize: bool, preemph: float) -> np.ndarray:
    """Обработка numpy array"""
    # Предполагаем, что массив уже загружен с правильной частотой
    sr = target_sr

    # Конвертация в моно если нужно
    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=1) if audio.ndim == 2 else audio

    # Обрезка по длительности если указана
    if duration is not None:
        max_samples = int(duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

    return _process_audio(audio, sr, target_sr, target_length, normalize, preemph)


def _process_list(file_list: List, target_sr: int, duration: float, target_length: int,
                  mono: bool, normalize: bool, preemph: float) -> List[np.ndarray]:
    """Обработка списка файлов или массивов"""
    processed_audios = []

    for item in file_list:
        if isinstance(item, (str, Path)):
            audio = _process_single_file(item, target_sr, duration, target_length, mono, normalize, preemph)
        elif isinstance(item, np.ndarray):
            audio = _process_array(item, target_sr, duration, target_length, mono, normalize, preemph)
        else:
            raise TypeError(f"Неподдерживаемый тип в списке: {type(item)}")

        processed_audios.append(audio)

    return processed_audios


def _process_audio(audio: np.ndarray, original_sr: int, target_sr: int, target_length: int,
                   normalize: bool, preemph: float) -> np.ndarray:
    """Основная обработка аудиосигнала"""

    # Ресемплинг если нужно
    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)

    # Предэмфаза если нужно
    if preemph is not None:
        audio = librosa.effects.preemphasis(audio, coef=preemph)

    # Приведение к целевой длине
    if target_length is not None:
        audio = _adjust_length(audio, target_length)

    # Нормализация амплитуды
    if normalize:
        audio = _normalize_audio(audio)

    return audio


def _adjust_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Приведение аудио к целевой длине"""
    current_length = len(audio)

    if current_length == target_length:
        return audio
    elif current_length > target_length:
        # Обрезка по центру
        start = (current_length - target_length) // 2
        return audio[start:start + target_length]
    else:
        # Дополнение нулями
        padding = target_length - current_length
        return np.pad(audio, (0, padding), mode='constant')


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Нормализация амплитуды аудио"""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio

def get_audio_segment(audio: np.ndarray, start_time: float, end_time: float, sample_rate: int) -> np.ndarray:
    """Извлекает сегмент аудио по времени"""
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    return audio[start_sample:end_sample]


def get_track_rate(file_path: str) -> int:
    """
    Получает частоту дискретизации аудиотрека

    Parameters:
    file_path: путь к аудиофайлу

    Returns:
    int: частота дискретизации в Гц

    Raises:
    FileNotFoundError: если файл не существует
    ValueError: если файл не поддерживается или поврежден
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        # Способ 1: Быстрое получение только частоты дискретизации
        sample_rate = librosa.get_samplerate(file_path)
        return sample_rate

    except Exception as e:
        try:
            # Способ 2: Загрузка файла с помощью soundfile
            info = sf.info(file_path)
            return info.samplerate

        except Exception:
            try:
                # Способ 3: Загрузка небольшого сегмента через librosa
                audio, sr = librosa.load(file_path, sr=None, duration=0.1)
                return sr

            except Exception as e2:
                raise ValueError(f"Не удалось определить частоту дискретизации файла {file_path}: {e2}")


def get_detailed_audio_info(file_path: str) -> dict:
    """
    Получает подробную информацию об аудиофайле

    Returns:
    dict: словарь с информацией об аудио
    """
    info = {}

    try:
        # Базовая информация через soundfile
        sf_info = sf.info(file_path)
        info.update({
            'sample_rate': sf_info.samplerate,
            'channels': sf_info.channels,
            'duration': sf_info.duration,
            'frames': sf_info.frames,
            'format': sf_info.format,
            'subtype': sf_info.subtype
        })

    except Exception as e:
        print(f"Soundfile info failed: {e}")
        # Альтернатива через librosa
        try:
            sr = librosa.get_samplerate(file_path)
            audio, _ = librosa.load(file_path, sr=sr, duration=1.0)
            info.update({
                'sample_rate': sr,
                'channels': 1 if audio.ndim == 1 else audio.shape[0],
                'duration': len(audio) / sr,
                'frames': len(audio)
            })
        except Exception as e2:
            print(f"Librosa info also failed: {e2}")

    return info