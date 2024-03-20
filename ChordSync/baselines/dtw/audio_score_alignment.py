"""
Audio to score alignment using the synctoolbox library.
"""
from pathlib import Path
from typing import Union, Tuple, Optional

import librosa
import numpy as np
from libfmp.b import plot_chromagram
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw, sync_via_mrmsdtw_with_anchors
from synctoolbox.dtw.utils import compute_optimal_chroma_shift, shift_chroma_vectors
from synctoolbox.feature.chroma import (
    pitch_to_chroma,
    quantize_chroma,
    quantized_chroma_to_CENS,
)
from synctoolbox.feature.csv_tools import (
    read_csv_to_df,
    df_to_pitch_features,
    df_to_pitch_onset_features,
)
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning


class AudioScoreAlignment:
    """
    Class to align the audio and score using the synctoolbox library.
    """

    def __init__(
        self,
        /,
        audio_path: Union[str, Path],
        score_path: Union[str, Path],
        fs: int = 22050,
        feature_rate: int = 10,
        step_weights: np.ndarray = np.array([1.5, 1.5, 2.0]),
        threshold_rec: int = 10**6,
    ) -> None:
        """
        Initialize the class with the path to the audio and score.
        :param audio_path: the path to the audio file
        :param score_path: the path to the score file
        :param fs: the sampling rate of the audio
        :param feature_rate: the feature rate
        :param step_weights: the step weights
        :param threshold_rec: the threshold for the recursion
        """
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
        if isinstance(score_path, str):
            score_path = Path(score_path)
        self.audio_path = audio_path
        self.score_path = score_path
        if not self.audio_path.exists():
            raise FileNotFoundError(f"{self.audio_path} not found.")
        if not self.score_path.exists():
            raise FileNotFoundError(f"{self.score_path} not found.")

        # basic parameters
        self.fs = fs
        self.feature_rate = feature_rate
        self.step_weights = step_weights
        self.threshold_rec = threshold_rec
        # load the audio and score
        self.audio, _ = librosa.load(str(self.audio_path), sr=self.fs, mono=True)
        self.score = read_csv_to_df(str(self.score_path), csv_delimiter=",")
        # normalize the audio
        self.audio = librosa.util.normalize(self.audio)

    @property
    def tuning(self) -> float:
        """
        Get the tuning of the audio.
        :return: the tuning of the audio
        :rtype: int
        """
        return estimate_tuning(self.audio, self.fs)

    def get_audio_features(
        self, visualize: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the audio features.
        :return:
        """
        f_pitch = audio_to_pitch_features(
            f_audio=self.audio,
            Fs=self.fs,
            tuning_offset=self.tuning, # type: ignore
            feature_rate=self.feature_rate,
            verbose=visualize,
        )
        f_chroma = pitch_to_chroma(f_pitch=f_pitch)
        f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
        if visualize:
            plot_chromagram(
                f_chroma_quantized,
                title="Quantized chroma features - Audio",
                Fs=self.feature_rate,
                figsize=(9, 3),
            )

        f_pitch_onset = audio_to_pitch_onset_features(
            f_audio=self.audio,
            Fs=self.fs,
            tuning_offset=self.tuning,
            verbose=visualize,
        )
        f_DLNCO = pitch_onset_features_to_DLNCO(
            f_peaks=f_pitch_onset,
            feature_rate=self.feature_rate,
            feature_sequence_length=f_chroma_quantized.shape[1],
            visualize=visualize,
        )
        return f_chroma_quantized, f_DLNCO

    def get_score_features(
        self, visualize: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the score features.
        :return:
        """
        f_pitch = df_to_pitch_features(self.score, feature_rate=self.feature_rate)
        f_chroma = pitch_to_chroma(f_pitch=f_pitch)
        f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
        if visualize:
            plot_chromagram(
                f_chroma_quantized,
                title="Quantized chroma features - Annotation",
                Fs=self.feature_rate,
                figsize=(9, 3),
            )
        f_pitch_onset = df_to_pitch_onset_features(self.score)
        f_DLNCO = pitch_onset_features_to_DLNCO(
            f_peaks=f_pitch_onset,
            feature_rate=self.feature_rate,
            feature_sequence_length=f_chroma_quantized.shape[1],
            visualize=visualize,
        )
        return f_chroma_quantized, f_DLNCO

    def get_alignment(
        self, algorithm: Optional[str] = "mrmsdtw", visualize: bool = False
    ) -> np.ndarray:
        """
        Get the alignment matrix.
        :param algorithm: the algorithm used for the alignment
        :type algorithm: str
        :param visualize: whether to visualize the alignment matrix or not
        :type visualize: bool
        :return: the alignment matrix
        :rtype: np.ndarray
        """
        f_chroma_quantized_audio, f_DLNCO_audio = self.get_audio_features(
            visualize=visualize
        )
        f_chroma_quantized_annotation, f_DLNCO_annotation = self.get_score_features(
            visualize=visualize
        )
        f_cens_1hz_audio = quantized_chroma_to_CENS(
            f_chroma_quantized_audio, 201, 50, self.feature_rate
        )[0]
        f_cens_1hz_annotation = quantized_chroma_to_CENS(
            f_chroma_quantized_annotation, 201, 50, self.feature_rate
        )[0]
        opt_chroma_shift = compute_optimal_chroma_shift(
            f_cens_1hz_audio, f_cens_1hz_annotation
        )

        f_chroma_quantized_annotation = shift_chroma_vectors(
            f_chroma_quantized_annotation, opt_chroma_shift
        )
        f_DLNCO_annotation = shift_chroma_vectors(f_DLNCO_annotation, opt_chroma_shift)

        if algorithm == "mrmsdtw":
            wp = sync_via_mrmsdtw(
                f_chroma1=f_chroma_quantized_audio,
                f_onset1=f_DLNCO_audio,
                f_chroma2=f_chroma_quantized_annotation,
                f_onset2=f_DLNCO_annotation,
                input_feature_rate=self.feature_rate,
                step_weights=self.step_weights,
                threshold_rec=self.threshold_rec,
                verbose=visualize,
            )
        elif algorithm == "mrmsdtw_anchors":
            wp = sync_via_mrmsdtw_with_anchors(
                f_chroma1=f_chroma_quantized_audio,
                f_onset1=f_DLNCO_audio,
                f_chroma2=f_chroma_quantized_annotation,
                f_onset2=f_DLNCO_annotation,
                input_feature_rate=self.feature_rate,
                step_weights=self.step_weights,
                threshold_rec=self.threshold_rec,
                verbose=visualize,
            )
        else:
            raise ValueError(f"Algorithm {algorithm} not supported.")

        return wp


if __name__ == "__main__":
    audio_path = Path("./data/schubert-winterreise/audio/schubert-winterreise_0_0.wav")
    score_path = Path(
        "./data/schubert-winterreise/converted/csv/schubert-winterreise_0.csv"
    )
    alignment = AudioScoreAlignment(audio_path=audio_path, score_path=score_path)
    al = alignment.get_alignment(visualize=False)
    print(al.flat)
    print(al.data)
    print(al[:, 0])
    print(al[:, 1])
