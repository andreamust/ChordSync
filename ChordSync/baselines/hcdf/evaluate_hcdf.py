import os
import sys

sys.path.append(os.path.dirname(os.path.realpath("./HCDF")))
sys.path.append(os.path.dirname(os.path.realpath("./HCDF/TIVlib")))

from HCDF import harmonic_change
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import jams
import random

AUDIO_PATH = "/media/data/andrea/choco_audio/audio"
JAMS_PATH = "/media/data/andrea/choco_audio/jams"


def _get_audio_list(audio_path: Union[str, Path]) -> list:
    """
    Get the list of audio files path in the audio path.

    Parameters
    ----------
    audio_path : Union[str, Path]
        Path to the audio files.

    Returns
    -------
    list
        List of audio files paths.
    """
    if isinstance(audio_path, str):
        audio_path = Path(audio_path)
    return list(audio_path.glob("*.flac"))


def _get_random_sample(audio_list: list, n: int) -> list:
    """
    Gets a random sample of n audio files from the audio list.

    Parameters
    ----------
    audio_list : list
        List of audio files.
    n : int
        Number of audio files to sample.

    Returns
    -------
    list
        Random sample of n audio files.
    """
    random.seed(111)
    return random.sample(audio_list, n)


def _get_jams_list(random_audio: list, jams_path: Union[str, Path]) -> list:
    """
    Get the list of jams files corresponding to the random audio files.

    Parameters
    ----------
    random_audio : list
        List of random audio files.
    audio_path : Union[str, Path]
        Path to the audio files.

    Returns
    -------
    list
        List of jams files corresponding to the random audio files.
    """
    if isinstance(jams_path, str):
        jams_path = Path(jams_path)
    return [jams_path / f.name.replace(".flac", ".jams") for f in random_audio]


def get_data(
    audio_path: Union[str, Path], jams_path: Union[str, Path], n: int
) -> tuple:
    """
    Get the random sample of audio and jams files.

    Parameters
    ----------
    audio_path : Union[str, Path]
        Path to the audio files.
    jams_path : Union[str, Path]
        Path to the jams files.
    n : int
        Number of audio files to sample.

    Returns
    -------
    tuple
        Random sample of audio and jams files.
    """
    audio_list = _get_audio_list(audio_path)
    random_audio = _get_random_sample(audio_list, n)
    jams_list = _get_jams_list(random_audio, jams_path)
    return random_audio, jams_list


def _compute_f1_peaks(audio_file: Path) -> np.ndarray:
    """
    Compute the f1 peaks from the audio file.

    Parameters
    ----------
    audio_file : Path
        Path to the audio file.

    Returns
    -------
    np.ndarray
        F1 peaks.
    """
    f_score = harmonic_change(
        audio_file,
        audio_file.name,
        chroma="nnls-8000-1024-2",
        hpss=True,
        tonal_model="TIV2",
        blur="full",
        sigma=5,
    )

    return f_score["changes"]


def _compute_recall_peaks(audio_file: Path) -> np.ndarray:
    """
    Compute the recall peaks from the audio file.

    Parameters
    ----------
    audio_file : Path
        Path to the audio file.

    Returns
    -------
    np.ndarray
        Recall peaks.
    """
    recall = harmonic_change(
        audio_file,
        audio_file.name,
        chroma="stft-44100-2048-4",
        hpss=True,
        tonal_model="TIV2",
        blur="full",
        sigma=17,
        distance="euclidean",
    )

    return recall["changes"]


def _get_jams_onset(jams_file: Path) -> np.ndarray:
    """
    Get the onset times from the jams file.

    Parameters
    ----------
    jams_file : Path
        Path to the jams file.

    Returns
    -------
    np.ndarray
        Onset times.
    """
    jam = jams.load(str(jams_file), validate=False)
    event_values = jam.annotations[0].to_event_values()[0]

    return event_values


def _evaluate(changes: np.ndarray, ground_truth: np.ndarray, window=0.3) -> float:
    """
    Evaluate the changes with the ground truth.

    Parameters
    ----------
    changes : np.ndarray
        Changes to evaluate.
    ground_truth : np.ndarray
        Ground truth.

    Returns
    -------
    float
        F1 score.
    """

    def _calculate_matches(seq1, seq2, window=window):
        matches = 0
        for time1 in seq1:
            for time2 in seq2:
                if abs(time1 - time2) <= window:
                    matches += 1
                    # Break the loop to avoid double counting
                    break

        return matches

    matches = _calculate_matches(changes, ground_truth, window)
    precision = matches / len(changes)
    recall = matches / len(ground_truth)
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


def _save_results(results: dict, output_path: Union[str, Path]):
    """
    Save the results to a csv file.

    Parameters
    ----------
    results : dict
        Results to save.
    output_path : Union[str, Path]
        Path to the output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).T
    df.to_csv(output_path)


def main(audio_path: Union[str, Path], jams_path: Union[str, Path], n: int):
    """
    Evaluate the HCDF algorithm on the random sample of audio and jams files.

    Parameters
    ----------
    audio_path : Union[str, Path]
        Path to the audio files.
    jams_path : Union[str, Path]
        Path to the jams files.
    n : int
        Number of audio files to sample.
    """
    results = {}
    random_audio, jams_list = get_data(audio_path, jams_path, n)
    for audio, jams in zip(random_audio, jams_list):
        print(f"Processing {audio.stem}")

        # compute f1 peaks
        f1_peaks = _compute_f1_peaks(audio)
        # compute recall peaks
        recall_peaks = _compute_recall_peaks(audio)

        # get jams onset
        target_onset = _get_jams_onset(jams)

        # evaluate
        precision, recall, f1_score = _evaluate(f1_peaks, target_onset)
        results[audio.stem] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    output_path = Path("results/HCDF_results.csv")
    _save_results(results, output_path)


if __name__ == "__main__":
    main(AUDIO_PATH, JAMS_PATH, 100)
