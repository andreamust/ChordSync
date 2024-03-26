import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import jams
import librosa
import numpy as np
import torch
from costants import Paths
from jams_processing import JAMSProcessor
from joblib import Parallel, delayed
from librosa import load
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, Resample
from tqdm import tqdm
from utils.chord_utils import (
    MajminChordEncoder,
    ModeEncoder,
    NoteEncoder,
    SimpleChordEncoder,
)
from utils.jams_utils import preprocess_jams, trim_jams


class ChocoAudioPreprocessor(Dataset):
    """
    Audio Dataset
    """

    def __init__(
        self,
        audio_path: str,
        jams_path: str,
        transform=None,
        target_sample_rate: int = 22_050,
        max_sequence_length: int = 15,
        excerpt_per_song: int = 3,
        excerpt_distance: int = 30,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Args:
            audio_path (str): Path to audio files
            jams_path (str): Path to jams files
            transform (callable, optional): Optional transform to be applied
                on a sample.
            target_sample_rate (int): Target sample rate
            num_samples (int): Number of samples to be extracted from audio
            device: Device to be used for processing
        """
        self.audio_path = Path(audio_path)
        self.jams_path = Path(jams_path)
        self.target_sample_rate = target_sample_rate
        self.device = device
        self.transform = transform
        self.excerpt_per_song = excerpt_per_song
        self.excerpt_distance = excerpt_distance

        # initialise max sequence length
        self._max_sequence_length = max_sequence_length
        # get number of samples which is sample rate * sequence length
        self.num_samples = int(self.target_sample_rate * self._max_sequence_length)

        # get audio and jams files
        audio_extensions = [".mp3", ".wav", ".flac"]
        self.audio_files = [
            file
            for file in os.listdir(str(self.audio_path))
            if file.endswith(tuple(audio_extensions))
        ]
        self.jams_files = [
            file for file in os.listdir(str(self.jams_path)) if file.endswith(".jams")
        ]

        # get the list of songs with the corresponding exerpts
        self.song_list = self._preprocess()

        # all vocabularies length
        self.vocabularies = {
            "simplified_sequence": len(SimpleChordEncoder),
            "root_sequence": len(NoteEncoder),
            "bass_sequence": len(NoteEncoder),
            "mode_sequence": len(ModeEncoder),
            "majmin_sequence": len(MajminChordEncoder),
        }

    def _create_song_list(self, distance: int = 30):
        """
        Creates a list of songs with the corresponding exerpts.

        Returns:
            list: A list of dictionaries containing the song name and the
            corresponding onsets.
        """
        song_list = []
        for audio in self.audio_files:
            for i in range(self.excerpt_per_song):
                song_list.append(
                    {
                        "audio": audio,
                        "onset": i * distance,
                    }
                )
        return song_list

    def __len__(self):
        """
        Returns the number of audio files in the dataset.
        """
        return len(self.song_list)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, dict]:
        """
        Returns the audio signal and chord annotation for the given index.

        Args:
            idx (int): Index of the audio file to retrieve.

        Returns:
            A tuple containing the audio signal and chord annotation as torch tensors.
            The audio signal is a tensor of shape (1, num_samples), where num_samples is
            the number of audio samples in the file. The chord annotation is a tensor of
            shape (1, longest_target), where longest_target is the length of the longest
            chord sequence in the dataset. Chord labels are represented as integers.
        """
        # the jams path is inferred from the audio path since the audio are
        # derived from the jams
        file = self.song_list[idx]
        onset = file["onset"]
        audio_file = file["audio"]
        audio_path = self.audio_path / audio_file
        file_name = audio_path.stem
        file_path = self.jams_path / f"{file_name}.jams"
        # print(f"Loading file {idx} of {len(self.song_list)}: {file_name}@{onset}")

        # load audio
        sample_onset = int(onset * self.target_sample_rate)
        signal, sr = load(audio_path, sr=None, mono=False)
        signal = torch.from_numpy(signal)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal, sample_onset)
        signal = self._right_pad_if_necessary(signal)

        # load annotations
        annotation = preprocess_jams(file_path)
        annotation = trim_jams(annotation, onset, self._max_sequence_length)
        assert (
            annotation and len(annotation.data) > 0
        ), f"File {file_name} @ {onset} has no annotation. \n {annotation}"
        annotation = self._chord_sequence_dict(annotation)

        # apply transform
        if self.transform == "cqt":
            signal = signal.numpy()
            signal = np.abs(
                librosa.cqt(signal, sr=self.target_sample_rate, hop_length=1024)
            )
            signal = torch.from_numpy(signal)
            print(signal.shape)
        elif self.transform:
            signal = self.transform(signal)

        excerpt_id = f"{file_name}-{onset}"

        return excerpt_id, signal, annotation

    def _get_jams_path(self, audio_path: Path) -> Path:
        """
        Returns the path to the JAMS file corresponding to the given audio
        file path.

        Args:
            audio_path (Path): The path to the audio file.

        Returns:
            Path: The path to the corresponding JAMS file.
        """
        file_name = audio_path.stem
        file_path = self.jams_path / f"{file_name}.jams"
        return file_path

    def _resample_if_necessary(self, signal: torch.Tensor, sr: float) -> torch.Tensor:
        """
        Resamples the input signal to the target sample rate if necessary.

        Args:
            signal (torch.Tensor): The input signal to resample.
            sr (int): The sample rate of the input signal.

        Returns:
            torch.Tensor: The resampled signal, or the original signal if its
            sample rate matches the target sample rate.
        """
        if sr != self.target_sample_rate:
            resample = Resample(int(sr), self.target_sample_rate)
            signal = resample(signal)
        return signal

    @staticmethod
    def _mix_down_if_necessary(signal: torch.Tensor) -> torch.Tensor:
        """
        Mixes down the given signal if it has more than one channel.

        Args:
            signal (torch.Tensor): The input signal to mix down.

        Returns:
            torch.Tensor: The mixed down signal.
        """
        # if the signal has no channels, add one
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        # if the signal has more than one channel, mix it down
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(
        self, signal: torch.Tensor, sample_onset: int
    ) -> torch.Tensor:
        """
        Cuts the signal tensor to the specified number of samples if it exceeds
        the limit, starting from the given onset.

        Args:
            signal (Tensor): The input signal tensor.
            sample_onset (int): The onset of the signal in samples.

        Returns:
            Tensor: The signal tensor with the specified number of samples.
        """
        # if the signal has more samples than the specified number of samples,
        # cut it starting from the onset
        if signal.shape[1] > self.num_samples:
            signal = signal[:, sample_onset : sample_onset + self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Pads the last dimension of the input signal tensor with zeros on the
        right side if its length is smaller than `self.num_samples`.

        Args:
            signal (torch.Tensor): The input signal tensor to pad.

        Returns:
            torch.Tensor: The padded signal tensor.
        """
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            last_dim_padding = (0, self.num_samples - length_signal)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _preprocess(self) -> list:
        """
        Preprocesses the dataset by iterating over all the JAMS files in it and
        extracting the chord annotations from them. It then creates a chord
        embedding for each chord annotation and adds it to the vocabulary.

        Returns:
            int: The length of the longest chord sequence in the dataset.
        """
        song_list = self._create_song_list(distance=self.excerpt_distance)
        # make a deep copy of the song list
        song_list_copy = song_list.copy()
        for song in song_list_copy:
            file_name = song["audio"].split(".")[0]
            onset = song["onset"]
            jams_path = self.jams_path / f"{file_name}.jams"
            annotation = preprocess_jams(jams_path=jams_path)
            annotation = trim_jams(annotation, onset, self._max_sequence_length)
            if not annotation:
                song_list.remove(song)

        return song_list

    def _chord_sequence_dict(self, jams_annotation: jams.Annotation):
        """
        Returns the chord sequence vector for the given JAMS annotation.

        Args:
            jams_annotation (jams.Annotation): The JAMS annotation to get the
                chord sequence vector for.
            onset (int): The onset of the audio file in seconds.

        Returns:
            dict: The chord sequence dictionary.
        """
        preprocessor = JAMSProcessor(
            sr=self.target_sample_rate,
            hop_length=512,
            duration=self._max_sequence_length,
        )
        # get the chord sequences
        simplified_sequence = preprocessor.simplified_sequence(jams_annotation)
        # onehot_sequence = preprocessor.onehot_sequence(jams_annotation)
        majmin_sequence = preprocessor.majmin_sequence(jams_annotation)
        root_sequence = preprocessor.root_sequence(jams_annotation)
        mode_sequence = preprocessor.mode_sequence(jams_annotation)
        onsets_sequence = preprocessor.onsets_sequence(jams_annotation)
        complete_sequence = preprocessor.complete_sequence(jams_annotation)
        # get the sequences with non-repeating chords
        simplified_symbols = preprocessor.simplified_unique(jams_annotation)
        majmin_symbols = preprocessor.majmin_unique(jams_annotation)
        root_symbols = preprocessor.root_unique(jams_annotation)
        mode_symbols = preprocessor.mode_unique(jams_annotation)
        complete_symbols = preprocessor.complete_unique(jams_annotation)

        # return the dictionary
        return {
            "simplified_sequence": simplified_sequence,
            "complete_sequence": complete_sequence,
            "majmin_sequence": majmin_sequence,
            "root_sequence": root_sequence,
            "mode_sequence": mode_sequence,
            "onsets_sequence": onsets_sequence,
            "simplified_symbols": simplified_symbols,
            "complete_symbols": complete_symbols,
            "majmin_symbols": majmin_symbols,
            "root_symbols": root_symbols,
            "mode_symbols": mode_symbols,
        }


# precompute the preprocessing results
def _parallel_preprocess(dataset: Dataset, idx: int, cache_path: Path) -> None:
    """
    Utility function for parallel preprocessing.
    """
    name, out_1, out_2 = dataset.__getitem__(idx)
    torch.save((out_1, out_2), cache_path / f"{name}.pt")


def _save_cache(dataset: Dataset, cache_path: Path, num_workers: int = 4):
    """
    Saves the preprocessing results on disk.

    Args:
        func (callable): The preprocessing function to cache.
        audio_files (list): The list of audio files to cache.
        jams_files (list): The list of JAMS files to cache.

    Returns:
        None
    """
    # if the directory does not exist, create it
    cache_path.mkdir(exist_ok=True)
    # if the directory is not empty, delete all files in it
    if os.listdir(cache_path):
        print("Deleting cache...")
        for file in os.listdir(cache_path):
            os.remove(cache_path / file)

    # cache the preprocessing results
    Parallel(n_jobs=num_workers)(
        delayed(_parallel_preprocess)(dataset, idx, cache_path)
        for idx in tqdm(range(len(dataset)))  # type: ignore
    )


def preprocess_data(
    audio_path: str,
    jams_path: str,
    max_sequence_length: int = 15,
    excerpt_per_song: int = 3,
    excerpt_distance: int = 30,
    cache_name: str = "cache_cqt_short",
    device: str | torch.device = "gpu",
    transform: str | torch.nn.Module | None = None,
    num_workers: int = 4,
) -> None:
    """
    Preprocesses the dataset by iterating over all the JAMS files in it and
    extracting the chord annotations from them. It then creates a chord
    embedding for each chord annotation and adds it to the vocabulary.

    Parameters
    ----------
    max_sequence_length : int, optional
        [description], by default 15
    excerpt_per_song : int, optional
        [description], by default 3
    excerpt_distance : int, optional
        [description], by default 30
    save_cache : bool, optional
        [description], by default False
    cache_name : str, optional
        [description], by default "cache_cqt_short"
    device : str | torch.device, optional
        [description], by default "cpu"

    Returns
    -------
    [type]
        [description]
    """
    print("Preprocessing dataset...")
    # initialize the dataset
    dataset = ChocoAudioPreprocessor(
        audio_path,
        jams_path,
        max_sequence_length=max_sequence_length,
        excerpt_distance=excerpt_distance,
        excerpt_per_song=excerpt_per_song,
        transform=transform,
        device=device,
    )

    # define the cache path
    cache_path = Path(audio_path).parent / "cache" / cache_name
    print(f"Cache path: {cache_path}")

    # get total number of samples
    num_samples = len(dataset)
    print(f"Number of samples: {num_samples}")

    # save the cache
    _save_cache(dataset, cache_path, num_workers=num_workers)


if __name__ == "__main__":

    mel_spectrogram = MelSpectrogram(
        sample_rate=22_050, n_fft=2048, hop_length=1024, n_mels=128
    )

    parser = ArgumentParser()
    parser.add_argument("--audio_path", type=str, default=Paths.audio.value)
    parser.add_argument("--jams_path", type=str, default=Paths.jams.value)
    parser.add_argument("--max_sequence_length", type=int, default=15)
    parser.add_argument("--excerpt_per_song", type=int, default=25)
    parser.add_argument("--excerpt_distance", type=int, default=13)
    parser.add_argument("--cache_name", type=str, default="mel_dict")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--transform", type=str, default=mel_spectrogram)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    preprocess_data(
        audio_path=args.audio_path,
        jams_path=args.jams_path,
        max_sequence_length=args.max_sequence_length,
        excerpt_distance=args.excerpt_distance,
        excerpt_per_song=args.excerpt_per_song,
        cache_name=args.cache_name,
        transform=args.transform,
        device=args.device,
        num_workers=args.num_workers,
    )

    # example usage
    # python preprocess_data.py --audio_path /path/to/audio --jams_path /path/to/jams --max_sequence_length 15 --excerpt_per_song 3 --excerpt_distance 30 --cache_name cache_cqt_short --device cpu --transform cqt --num_workers 4
