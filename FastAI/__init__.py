"""
FastAI - you can create AI relly fast
"""

__version__ = "0.1.0"
__all__ = [
    'kneighbors',
    'decision_tree',
    't_decision_tree',
    'load_filedata',
    'text_classifier',
    't_text_classifier',
    'basic_network',
    't_basic_network',
    'cnn_network',
    't_cnn_network',
    'cluster_data',
    'prepare_images',
    'prepare_sound',
    'sound_analiser',
    't_sound_analiser',
    't_image_classifier',
    'text_to_num',
    'ovr_model',
    'ovo_model',
    'multi_output_regression',
    'ensemble_regression',
    'image_classifier',
    'image_feature_extractor',
    'text_embedding',
    'eazy_text_classifier',
    'audio_feature_extractor',
    'time_series_model',
    'create_pipeline',
    'data_normalise'
]

from .basic_network import t_basic_network
from .clusters import cluster_data
from .cnn_network import t_cnn_network
from .decision_tree import t_decision_tree
from .eazy_models import ovo_model, ovr_model, multi_output_regression, ensemble_regression, image_classifier, \
    image_feature_extractor, text_embedding, audio_feature_extractor, time_series_model, create_pipeline, \
    eazy_text_classifier
from .image_classificer import t_image_classifier
from .prepare_data import load_filedata, text_to_num, data_normalise
from .prepare_image import prepare_images
from .sound_analiser import t_sound_analiser
from .text_classificer import text_classifier, t_text_classifier
