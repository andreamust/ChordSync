import sys
from pathlib import Path

sys.path.append(Path(__file__).parents[2].as_posix())


from typing import Union

import jams
import jams_namespaces
import numpy as np
import pandas as pd
from audio_score_alignment import AudioScoreAlignment


def get_audio_ids(
    score_file: str,
    data_path: Union[str, Path] = "ChordSync/baselines/dtw/data/schubert-winterreise",
) -> list:
    """
    Get the audio ids for a given score file.

    Args:
        score_file (str): The score file to get the audio ids for.
        data_path (str | Path): The path to the data.

    Returns:
        List[str]: The audio ids for the given score file.
    """
    # get paths
    data_path = Path(data_path)
    assert data_path.exists(), f"Data path {data_path} does not exist."
    meta_audio = pd.read_csv(data_path / "meta_audio.csv")
    meta_score = pd.read_csv(data_path / "meta_score.csv")

    # get audio ids
    score_id = meta_score[meta_score["id"] == score_file]["score_file"].values[0]
    results = meta_audio[meta_audio["track_file"].str.contains(score_id)]["id"].values

    return list(results)


def _find_audio_jams(file_stem: str, jams_path: Union[str, Path]) -> str:
    """
    Find the JAMS file for a given audio file stem.

    Args:
        file_stem (str): The stem of the audio file.
        jams_path (str | Path): The path to the JAMS files.

    Returns:
        str: The path to the JAMS file.
    """
    # get paths
    jams_path = Path(jams_path)
    assert jams_path.exists(), f"JAMS path {jams_path} does not exist."

    # find JAMS file
    jams_file = list(jams_path.glob(f"{file_stem}.jams"))

    return str(jams_file[0])


def _get_jams_onsets(jams_file: Union[str, Path]) -> pd.DataFrame:
    """
    Get the onsets from the JAMS file.

    Args:
        jams_file (str | Path): The path to the JAMS file.

    Returns:
        pd.DataFrame: The onsets from the JAMS file.
    """
    jams_file = str(jams_file)
    # load JAMS
    jam = jams.load(jams_file)
    ann = jam.search(namespace="chord_harte")

    # get data
    onsets, _ = ann[0].to_event_values()  # type: ignore

    return onsets


def get_csv_onsets(csv_path: Union[str, Path]) -> np.ndarray:
    """
    Get the onsets from the CSV file.

    Args:
        csv_path (str | Path): The path to the CSV file.

    Returns:
        pd.DataFrame: The onsets from the CSV file.
    """
    # load CSV
    onsets = pd.read_csv(csv_path, header=None)
    # get unique values of the column "Start" and do not consider the first row
    onsets = onsets[0].unique()[1:]

    return onsets


def get_audio_jams_onsets(file_stem: str, jams_path: Union[str, Path]) -> pd.DataFrame:
    """
    Get the onsets from the JAMS file for a given audio file stem.

    Args:
        file_stem (str): The stem of the audio file.
        jams_path (str | Path): The path to the JAMS files.

    Returns:
        pd.DataFrame: The onsets from the JAMS file.
    """
    # find JAMS file
    jams_file = _find_audio_jams(file_stem, jams_path)

    # load JAMS
    onsets = _get_jams_onsets(jams_file)

    return onsets


def map_timing(alignment: np.ndarray, score_time: float) -> float:
    # get the indices of the alignment matrix
    audio_indices = alignment[0]
    score_indices = alignment[1]
    # get the indices closest to the score time
    score_time_indices = np.argmin(np.abs(score_indices - score_time))
    # get the audio time
    audio_time = audio_indices[score_time_indices]
    return audio_time


def convert_onsets(
    alignment: np.ndarray, onsets: np.ndarray, feature_rate: int
) -> np.ndarray:
    """
    Convert the onsets to the alignment grid.

    Args:
        alignment (np.ndarray): The alignment grid.
        onsets (np.ndarray): The onsets to convert.
        feature_rate (int): The feature rate.

    Returns:
        np.ndarray: The converted onsets.
    """
    converted_onsets = [
        map_timing(alignment, float(onset) * feature_rate)  # type: ignore
        for onset in onsets
    ]
    converted_onsets = np.array(converted_onsets)

    converted_onsets = converted_onsets / feature_rate

    return converted_onsets


def align_with_csv(
    audio_path: str, csv_path: str, feature_rate: int = 10
) -> np.ndarray:
    # create the audio score alignment object
    alignment = AudioScoreAlignment(
        audio_path=audio_path, score_path=csv_path, feature_rate=feature_rate
    )
    alignment = alignment.get_alignment(visualize=True)

    return alignment


if __name__ == "__main__":
    # # get audio ids
    score_file = "schubert-winterreise-score_22"
    audio_ids = get_audio_ids(score_file)
    print(audio_ids)

    # get audio onsets
    # audio_file = "schubert-winterreise-audio_4"
    # jams_path = "/media/data/andrea/choco_audio/jams"

    # onsets = get_audio_jams_onsets(audio_file, jams_path)
    # print(onsets)
