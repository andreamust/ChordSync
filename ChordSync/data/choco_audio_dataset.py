import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import os
from pathlib import Path
from typing import Tuple

import jams
import torch
from ChordSync.data.jams_processing import JAMSProcessor
from data_preprocessing.transformations import (
    chroma_transformation,
    hcqt_transformation,
)
from joblib import Parallel, delayed
from librosa import load
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, Resample
from utils.chord_utils import (
    MajminChordEncoder,
    ModeEncoder,
    NoteEncoder,
    SimpleChordEncoder,
)
from utils.jams_utils import preprocess_jams, trim_jams


class ChocoAudioDataset(Dataset):
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
        save_cache: bool = False,
        cache_name: str = "cache_cqt_short",
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
        self.cache_name = cache_name

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
            # exclude jazz partitions
            and "weimar" not in file and "jaah" not in file
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
        # cache the preprocessing by saving preprocessing results on disk
        if save_cache:
            self._save_cache()

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

    def _parallel_preprocess(self, idx: int):
        """
        Utility function for parallel preprocessing.
        """
        output = self.__getitem__(idx)
        torch.save(output, self.jams_path.parent / self.cache_name / f"{idx}.pt")

    def _save_cache(self):
        """
        Saves the preprocessing results on disk.

        Args:
            func (callable): The preprocessing function to cache.
            audio_files (list): The list of audio files to cache.
            jams_files (list): The list of JAMS files to cache.

        Returns:
            None
        """
        # create cache directory
        cache_path = self.jams_path.parent / self.cache_name
        # if the directory does not exist, create it
        cache_path.mkdir(exist_ok=True)
        # if the directory is not empty, delete all files in it
        if os.listdir(cache_path):
            print("Deleting cache...")
            for file in os.listdir(cache_path):
                os.remove(cache_path / file)

        # cache the preprocessing results
        Parallel(n_jobs=4)(
            delayed(self._parallel_preprocess)(idx)
            for idx in range(len(self.song_list))
        )

    def __len__(self):
        """
        Returns the number of audio files in the dataset.
        """
        return len(self.song_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # if the cached file exists, load it
        if (self.jams_path.parent / self.cache_name / f"{idx}.pt").exists():
            audio, target = torch.load(
                self.jams_path.parent / self.cache_name / f"{idx}.pt"
            )
            return audio, target

        # the jams path is inferred from the audio path since the audio are
        # derived from the jams
        file = self.song_list[idx]
        onset = file["onset"]
        audio_file = file["audio"]
        audio_path = self.audio_path / audio_file
        file_name = audio_path.stem
        file_path = self.jams_path / f"{file_name}.jams"
        print(f"Loading file {idx} of {len(self.song_list)}: {file_name}@{onset}")

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
        abc = annotation
        annotation = trim_jams(annotation, onset, self._max_sequence_length)
        assert annotation, f"File {file_name} @ {onset} has no annotation. \n {abc}"
        annotation = self._chord_sequence_vector(annotation)

        # apply transform
        if self.transform == "chroma":
            signal = chroma_transformation(
                signal=signal,
                n_chroma=12,
                hop_length=512,
                n_fft=1024,
            )
            signal = torch.from_numpy(signal)  # .to(self.device)
        elif self.transform == "hcqt":
            signal = hcqt_transformation(
                signal,
                sr=22050,
                fs_hcqt_target=50,
                bins_per_semitone=5,
                num_octaves=6,
                num_harmonics=5,
                num_subharmonics=1,
                center_bins=True,
            )
            signal = torch.from_numpy(signal)  # .to(self.device)
        elif self.transform:
            signal = self.transform(signal)  # .to(self.device)

        # assure that the signal and the annotation have the same length
        assert signal.shape[2] == annotation.shape[0], (
            f"Signal and annotation have different length: "
            f"{signal.shape[2]} != {annotation.shape[0]}"
        )

        return signal, annotation

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

    def _chord_sequence_vector(self, jams_annotation: jams.Annotation):
        """
        Returns the chord sequence vector for the given JAMS annotation.

        Args:
            jams_annotation (jams.Annotation): The JAMS annotation to get the
                chord sequence vector for.
            onset (int): The onset of the audio file in seconds.

        Returns:
            np.ndarray: The chord sequence vector.
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
        # get the sequences with non-repeating chords
        simplified_symbols = preprocessor.simplified_unique(jams_annotation)
        majmin_symbols = preprocessor.majmin_unique(jams_annotation)
        root_symbols = preprocessor.root_unique(jams_annotation)
        mode_symbols = preprocessor.mode_unique(jams_annotation)

        # stack the sequences
        sequence = torch.cat(
            (
                # onehot_sequence,
                simplified_sequence,
                majmin_sequence,
                root_sequence,
                mode_sequence,
                onsets_sequence,
                simplified_symbols,
                majmin_symbols,
                root_symbols,
                mode_symbols,
            ),
            dim=1,
        )

        return sequence


if __name__ == "__main__":
    audio_path = "/media/data/andrea/choco_audio/audio"
    jams_path = "/media/data/andrea/choco_audio/jams"

    mel_spectrogram = MelSpectrogram(
        sample_rate=22_050, n_fft=1024, hop_length=512, n_mels=128
    )

    dataset = ChocoAudioDataset(
        audio_path,
        jams_path,
        max_sequence_length=15,
        excerpt_distance=17,
        excerpt_per_song=8,
        transform="hcqt",
        cache_name="cache_hcqt_short_nojazz",
        save_cache=False,
    )
    print(f"Dataset_shape: {len(dataset)}")

    signal, annotation = dataset[12]
    print(f"signal_shape: {signal.shape}")
    print(f"annotation_shape: {annotation.shape}")
