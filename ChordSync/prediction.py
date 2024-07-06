"""
Prediction module for running ChordSync at inference time.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import librosa
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from data.jams_processing import JAMSProcessor
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.utils.data import DataLoader, Dataset
from torchaudio.models import Conformer
from utils.torch_utils import PositionalEncoding

from ChordSync.models import ConformerModel


def _get_model_weights(model: L.LightningModule, model_path: str) -> nn.Module:
    """
    Load model weights from a file.

    Args:
        model: The model to load weights into.
        model_path: The path to the model weights file.

    Returns:
        The model with the loaded weights.
    """
    # load model weights
    model = model.load_from_checkpoint(model_path)
    model.eval()
    return model


def _preprocess_audio(audio: str | Path) -> torch.Tensor:
    """
    Preprocess the audio file for inference.

    Args:
        audio: The path to the audio file.

    Returns:
        The audio waveform and the sample rate.
    """
    # load audio
    waveform, _ = librosa.load(audio, sr=22_050)
    waveform = torch.tensor(waveform).unsqueeze(0)
    return waveform


def _preprocess_labels(labels: List[str]) -> torch.Tensor:
    """
    Preprocess the labels for inference.

    Args:
        labels: The list of labels.

    Returns:
        The label tensor.
    """
    # load labels
    pass


if __name__ == "__main__":
    # load model
    model_path = "model_weights.ckpt"
    model = _get_model_weights(ConformerModel(), model_path)

    # preprocess audio
    audio = "audio.wav"
    waveform = _preprocess_audio(audio)

    # preprocess labels
    labels = ["C:maj", "G:maj", "A:min", "F:maj"]
    labels = _preprocess_labels(labels)

    # run inference
    with torch.no_grad():
        output = model(waveform, labels)
        print(output)
