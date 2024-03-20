"""
Baselines experiments for DTW on Schubert dataset. 
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from pathlib import Path

import numpy as np
import pandas as pd
from audio_score_alignment import AudioScoreAlignment
from dtw_utils import (
    convert_onsets,
    get_audio_ids,
    get_audio_jams_onsets,
    get_csv_onsets,
)
from joblib import Parallel, delayed
from mir_eval import alignment


class AlignSchubertDTW:
    """
    Main class for DTW experiments on Schubert dataset.
    It takes the audio and score paths and performs the alignment for the whole
    dataset. It exposes methods for getting the evaluation and storing the results.
    """

    def __init__(
        self, audio_path: Path, score_path: Path, feature_rate: int = 100
    ) -> None:
        """
        Initialize the class with the paths to the audio and score files.

        Parameters
        ----------
        audio_path : Path
            Path to the audio files.
        score_path : Path
            Path to the score files.

        Returns
        -------
        None
        """
        self.audio_path = audio_path
        self.score_path = score_path
        self.feature_rate = feature_rate
        self.meta_path = score_path.parent
        self.jams_audio_path = self.audio_path.parent / "jams"

        # get the list of score files
        self.score_files = self._get_score_files()

        # prepare results
        self.results = []

    def _get_score_files(self) -> list:
        """
        Get the list of score files.

        Returns
        -------
        list
            List of score files.
        """
        assert self.score_path.exists(), "Score path does not exist."
        return list(self.score_path.glob("*.csv"))

    def _get_audio_ids(self, score_name: str) -> list:
        """
        Get the list of audio ids for a given score file.

        Parameters
        ----------
        score_name : str
            Name of the score file.

        Returns
        -------
        list
            List of audio ids.
        """
        return get_audio_ids(score_name, self.meta_path)

    def _get_evaluation(self, target_onsets: np.ndarray, predicted_onsets: np.ndarray):
        """
        Get the evaluation for the alignment.

        Parameters
        ----------
        target_onsets : np.ndarray
            The target onsets.
        predicted_onsets : np.ndarray
            The predicted onsets.

        Returns
        -------
        dict
            The evaluation results.
        """
        # get the evaluation
        eval_results = alignment.evaluate(target_onsets, predicted_onsets)

        return eval_results

    def _save_dataframes_to_csv(self, save_path: Path):
        """
        Save the dataframes to a CSV file.

        Parameters
        ----------
        save_path : Path
            The path to save the dataframes.
        filename : str
            The filename for the CSV file.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # save self.results as CSV
        results = pd.DataFrame(self.results)
        results.to_csv(save_path, index=False)

    def save_evaluation(self, filename: str = "schubert_dtw_results.csv"):
        """
        Save the evaluation results to a CSV file.
        """
        if not self.results:
            self.evaluate()
        self._save_dataframes_to_csv(self.meta_path / "results" / filename)

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate the alignment for the whole dataset.
        """

        for score_file in self.score_files:

            score_name = score_file.stem
            audio_ids = self._get_audio_ids(score_name)
            for audio_id in audio_ids:
                try:
                    audio_file = self.audio_path / f"{audio_id}.flac"
                    # audio_onsets = get_audio_jams_onsets(audio_file, FEATURE_RATE)
                    alignment = AudioScoreAlignment(
                        audio_file, score_file, feature_rate=self.feature_rate
                    )
                    alignment = alignment.get_alignment(visualize=False)

                    # get score onsets
                    score_onsets = get_csv_onsets(score_file)

                    # convert the onsets
                    converted_onsets = convert_onsets(
                        alignment, score_onsets, self.feature_rate
                    )

                    # get the corresponding audio onsets
                    audio_onsets = get_audio_jams_onsets(
                        audio_file.stem, self.jams_audio_path
                    )

                    # get the evaluation
                    eval_results = self._get_evaluation(audio_onsets, converted_onsets)  # type: ignore

                    # store the results
                    result = {
                        "audio_id": audio_id,
                        "score_name": score_name,
                    }
                    result.update(eval_results)

                    self.results.append(result)

                except Exception as e:
                    print(f"Error for {audio_id} and {score_name}: {e}")
                    continue

        return pd.DataFrame(self.results)


def run_experiment(feature_rate: int):
    """
    Run the experiment for a given feature rate.
    """
    # define paths
    audio_path = Path("/media/data/andrea/choco_audio/audio")
    score_path = Path("ChordSync/baselines/dtw/data/schubert-winterreise/csv-score")

    # get the alignment
    align = AlignSchubertDTW(audio_path, score_path, feature_rate=feature_rate)
    align.evaluate()
    align.save_evaluation(filename=f"schubert_dtw_results-{feature_rate}.csv")


def main():
    """
    Main function for DTW experiments on Schubert dataset.
    """

    ranges = [1, 10, 100, 1000]

    # parallelize the experiment
    Parallel(n_jobs=8, verbose=5)(delayed(run_experiment)(fr) for fr in ranges)


if __name__ == "__main__":
    main()
