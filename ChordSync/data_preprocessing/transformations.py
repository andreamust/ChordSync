"""
Preprocessing function for the ChordSync dataset.
"""
import librosa
import numpy as np
import torch
from data_preprocessing.hcqt import compute_efficient_hcqt
from librosa.feature import chroma_stft


def chroma_transformation(
    signal: torch.Tensor,
    target_sample_rate: int = 22050,
    n_chroma: int = 12,
    hop_length: int = 512,
    n_fft: int = 1024,
) -> np.ndarray:
    """
    Applies chroma transformation to the input signal.

    Args:
        signal (torch.Tensor): Input signal to be transformed.
        n_chroma (int): Number of chroma bins to use (default: 12).
        n_fft (int): Length of the FFT window (default: 1024).

    Returns:
        torch.Tensor: Chroma representation of the input signal.
    """
    signal = signal.detach().cpu().numpy()
    chroma = chroma_stft(
        y=signal,  # type: ignore
        sr=target_sample_rate,
        n_chroma=n_chroma,
        hop_length=hop_length,
        n_fft=n_fft,
    )

    return chroma


def hcqt_transformation(
    signal: torch.Tensor,
    sr: int = 22050,
    fs_hcqt_target: int = 50,
    bins_per_semitone: int = 5,
    num_octaves: int = 6,
    num_harmonics: int = 5,
    num_subharmonics: int = 1,
    center_bins: bool = True,
) -> np.ndarray:
    """
    Applies HCQT transformation to the input signal.

    Args:
        signal (torch.Tensor): Input signal to be transformed.
        file_name (str): Name of the file to save the HCQT to.
        fs_hcqt_target (int): Target sample rate of the HCQT (default: 50).
        bins_per_semitone (int): Number of bins per semitone (default: 5).
        num_octaves (int): Number of octaves to use (default: 6).
        num_harmonics (int): Number of harmonics to use (default: 5).
        num_subharmonics (int): Number of subharmonics to use (default: 1).
        center_bins (bool): Whether to center the bins (default: True).
        save_hcqt (bool): Whether to save the HCQT to a file (default: False).

    Returns:
        torch.Tensor: HCQT representation of the input signal.
    """
    # CQT parameters
    fmin = librosa.note_to_hz("C1")  # MIDI pitch 24
    bins_per_octave = 12 * bins_per_semitone

    # extract HCQT
    signal = signal.detach().cpu().numpy()
    f_hcqt, _, _ = compute_efficient_hcqt(
        signal.squeeze(),
        fs=sr,
        fmin=fmin,
        fs_hcqt_target=fs_hcqt_target,
        bins_per_octave=bins_per_octave,
        num_octaves=num_octaves,
        num_harmonics=num_harmonics,
        num_subharmonics=num_subharmonics,
        center_bins=center_bins,
    )

    # reshape to (2, 0, 1)
    cqtq = np.transpose(f_hcqt, (2, 0, 1))

    return cqtq
