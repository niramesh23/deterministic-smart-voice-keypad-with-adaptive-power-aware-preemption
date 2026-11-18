"""
ml Subsystem
============
Responsibilities:
- Dataset loading, batching, and split parsing (Google Speech Commands v0.02 train/val/test).
- Neural network model definitions (e.g. compact DS-CNN for Keyword Spotting).
- Training loop orchestration, loss calculation, early stopping, and checkpointing.
- Evaluation metrics computation (Top-1 accuracy, confusion matrix, precision/recall per command).
- Low-latency CPU inference engine.

Design Principles:
- Decouple training pipelines from real-time inference runtimes.
- Compact, parameter-efficient architectures (< 30k parameters) for real-time CPU execution.
- Identical preprocessing contracts across offline training and live inference.
"""

from src.ml.classes import (
    CLASS_NAMES,
    CLASS_NAMES_12,
    CLASS_NAMES_21,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CORE_COMMANDS,
    NUMBER_COMMANDS,
    TARGET_COMMANDS,
    NUMBER_TO_DIGIT,
    AUXILIARY_WORDS,
    NUM_CLASSES,
    map_word_to_class,
    get_display_command,
)
from src.ml.manifests import build_manifests
from src.ml.dataset import SpeechCommandsDataset, load_wav_16k_mono
from src.ml.model import DSCNNKeywordSpotter, ResidualDSCNNKeywordSpotter, DepthwiseSeparableBlock, get_model
from src.ml.inference import KeywordSpotterInferenceService, RecognitionResult
from src.ml.train import train
from src.ml.evaluate import evaluate

__all__ = [
    "CLASS_NAMES",
    "CLASS_NAMES_12",
    "CLASS_NAMES_21",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "CORE_COMMANDS",
    "NUMBER_COMMANDS",
    "TARGET_COMMANDS",
    "NUMBER_TO_DIGIT",
    "AUXILIARY_WORDS",
    "NUM_CLASSES",
    "map_word_to_class",
    "get_display_command",
    "build_manifests",
    "SpeechCommandsDataset",
    "load_wav_16k_mono",
    "DSCNNKeywordSpotter",
    "ResidualDSCNNKeywordSpotter",
    "DepthwiseSeparableBlock",
    "get_model",
    "KeywordSpotterInferenceService",
    "RecognitionResult",
    "train",
    "evaluate",
]
