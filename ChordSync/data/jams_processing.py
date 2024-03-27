"""
Utility functions for processing JAMS files and extracting the chord annotations
in different formats and shapes.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parents[3]))

import jams
import numpy as np
import torch
from pumpp.task import ChordTransformer
from pumpp.task.base import BaseTaskTransformer
from pumpp.task.chord import ChordTagTransformer
from utils.chord_utils import (
    ChordEncoder,
    Encoding,
    MajminChordEncoder,
    ModeEncoder,
    NoteEncoder,
    SimpleChordEncoder,
)


class PumppChordConverter(ChordTagTransformer):
    def __init__(self, vocab="3567s", sparse=True):
        super().__init__(vocab, sparse=sparse, vocab=vocab)

    def convert(self, value):

        converted = self.encoder.transform([self.simplify(value)])
        return converted[0]


class JAMSProcessor:
    """
    Interface for processing JAMS files and extracting the chord annotations in
    different formats and shapes, e.g. symbols sequences, one-hot encoded vectors,
    etc.
    """

    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        duration: float = 15,
    ) -> None:
        """
        Constructor of the JAMSProcessor class.

        Args:
            jams_annotation (jams.Annotation): The JAMS annotation to process.
            sr (int): The sampling rate of the audio file.
            hop_length (int): The hop length of the audio file.


        Returns:
            None
        """
        self.sr = sr + 1  # TODO: fix this. It is a workaround for a bug in librosa
        self.hop_length = hop_length
        self.duration = duration
        # compute sequence duration
        self.sequence_duration = self.sr * self.duration / self.hop_length
        # initialize the chord encoder
        self.custom_encoder = ChordEncoder()

        # initialize the pump extractors
        self.chord_transformer = ChordTransformer(
            name="chord",
            sr=self.sr,
            hop_length=self.hop_length,
        )
        self.base_transformer = BaseTaskTransformer(
            name="chord",
            namespace="chord",
            sr=self.sr,
            hop_length=self.hop_length,
        )

    def _create_sequence(
        self, intervals: np.ndarray, values: np.ndarray, fill: int = 0
    ) -> np.ndarray:
        """
        Transforms the given JAMS annotation into a sequence of chord symbols
        and a sequence of chord tags.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.
            values (np.ndarray): The values to encode. The array must have shape
                (n_frames, 1).
            fill (int): The value to use for filling the sequence. Defaults to 0.

        Returns:
            tuple: A tuple containing the chord symbols and the chord tags.
        """
        # check dimensionality of input arrays
        assert intervals.ndim == 2, ValueError(
            "The intervals array must have shape (n_frames, 2)."
        )
        assert values.ndim == 2, ValueError(
            "The values array must have shape (n_frames, 1)."
        )
        assert values.shape[1] == 1, ValueError(
            "The values array must have shape (n_frames, 1)."
        )
        # encode modes
        sequence = self.base_transformer.encode_intervals(
            duration=self.duration,
            intervals=intervals,
            values=values,
            dtype=int,  # type: ignore
            multi=False,
            fill=fill,
        )

        # check if the sequence is of the same length as self.sequence_duration
        assert sequence.shape[0] == int(self.sequence_duration), ValueError(
            "Sequence mismatch."
        )

        return sequence

    def _transform_annotation(self, annotation: jams.Annotation) -> dict:
        """
        Transforms the given JAMS annotation into a sequence of chord symbols
        and a sequence of chord tags.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            tuple: A tuple containing the chord symbols and the chord tags.
        """
        # get the chord symbols
        chord_symbols = self.chord_transformer.transform_annotation(
            ann=annotation, duration=self.duration
        )

        return chord_symbols

    def _pad_sequence(self, sequence: np.ndarray, pad_value: int) -> np.ndarray:
        """
        Pads the given sequence to the sequence duration.

        Args:
            sequence (np.ndarray): The sequence to pad.

        Returns:
            np.ndarray: The padded sequence.
        """
        # pad the roots to the sequence duration
        sequence = np.pad(
            sequence,
            (0, int(self.sequence_duration - len(sequence))),
            mode="constant",
            constant_values=pad_value,
        )

        return sequence.reshape(-1, 1)

    def onehot_sequence(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a one-hot encoded sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A one-hot encoded sequence.
        """
        # get the chord symbols
        chord_symbols = self._transform_annotation(annotation)["pitch"]

        return torch.Tensor(chord_symbols).type(torch.int)

    def _encode_sequence(self, chords: list, encoding: Encoding) -> np.ndarray:
        """
        Transforms the given JAMS annotation into a sequence of chord symbols.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            np.ndarray: A sequence of chord symbols.
        """
        encoded = [
            self.custom_encoder.encode(chord, encoding=encoding) for chord in chords
        ]
        return np.array(encoded, dtype=int)

    def _convert_sequence(
        self, annotation: jams.Annotation, encoding: Encoding, pad_value: int
    ) -> torch.Tensor:
        """
        Converts a sequence of chord symbols into a sequence of notes.

        Args:
            sequence (np.ndarray): The sequence of chord symbols to convert.

        Returns:
            np.ndarray: The sequence of notes.
        """
        intervals, chords = annotation.to_interval_values()
        converted = self._encode_sequence(chords, encoding=encoding)
        converted = converted.reshape(-1, 1)

        # unroll the sequence
        converted = self._create_sequence(intervals, converted, fill=pad_value)

        return torch.Tensor(converted).type(torch.int)

    def root_sequence(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a root sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A root sequence of shape (n_frames, 1).
        """
        return self._convert_sequence(
            annotation, encoding=Encoding.ROOT, pad_value=NoteEncoder.N.value
        )

    def mode_sequence(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a mode sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A mode sequence of shape (n_frames, 1).
        """
        return self._convert_sequence(
            annotation, encoding=Encoding.MODE, pad_value=ModeEncoder.N.value
        )

    def bass_sequence(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a bass sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A bass sequence of shape (n_frames, 1).
        """
        return self._convert_sequence(
            annotation, encoding=Encoding.BASS, pad_value=NoteEncoder.N.value
        )

    def simplified_sequence(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a mode sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A mode sequence of shape (n_frames, 1).
        """
        return self._convert_sequence(annotation, encoding=Encoding.SIMPLIFIED, pad_value=SimpleChordEncoder.N.value)  # type: ignore

    def complete_unique(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a mode sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A mode sequence of shape (n_frames, 1).
        """
        converter = PumppChordConverter()
        intervals, chords = annotation.to_interval_values()
        converted = np.array([converter.convert(c) for c in chords])

        converted = self._pad_sequence(converted, pad_value=0)

        return torch.Tensor(converted).type(torch.int)

    def complete_sequence(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a mode sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A mode sequence of shape (n_frames, 1).
        """
        converter = PumppChordConverter()
        intervals, chords = annotation.to_interval_values()
        converted = np.array([converter.convert(c) for c in chords])
        converted = converted.reshape(-1, 1)

        # unroll the sequence
        converted = self._create_sequence(intervals, converted, fill=0)

        return torch.Tensor(converted).type(torch.int)

    def majmin_sequence(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a mode sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A mode sequence of shape (n_frames, 1).
        """
        return self._convert_sequence(annotation, encoding=Encoding.MAJMIN, pad_value=MajminChordEncoder.N.value)  # type: ignore

    def mode_unique(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a mode sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A mode sequence of shape (n_frames, 1).
        """
        _, chords = annotation.to_interval_values()
        modes = self._encode_sequence(chords, encoding=Encoding.MODE)
        # pad the modes to the sequence duration
        modes = self._pad_sequence(modes, pad_value=0)

        return torch.Tensor(modes).type(torch.int)

    def onsets_sequence(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a sequence of onsets.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A sequence of onsets of shape (n_frames, 1).
        """
        # get the onsets for the annotation
        intervals, _ = annotation.to_interval_values()
        # get only the fitst column of the intervals
        onsets = intervals[:, 0]
        # pad the onsets to the sequence duration
        onsets = self._pad_sequence(onsets, pad_value=0)

        return torch.Tensor(onsets).type(torch.float)

    def root_unique(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a root sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A root sequence of shape (n_frames, 1).
        """
        _, chords = annotation.to_interval_values()
        roots = self._encode_sequence(chords, encoding=Encoding.ROOT)
        # pad the roots to the sequence duration
        roots = self._pad_sequence(roots, pad_value=0)

        return torch.Tensor(roots).type(torch.int)

    def bass_unique(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a root sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A root sequence of shape (n_frames, 1).
        """
        _, chords = annotation.to_interval_values()
        basses = self._encode_sequence(chords, encoding=Encoding.BASS)
        # pad the basses to the sequence duration
        basses = self._pad_sequence(basses, pad_value=0)

        return torch.Tensor(basses).type(torch.int)

    def simplified_unique(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a root sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A root sequence of shape (n_frames, 1).
        """
        _, chords = annotation.to_interval_values()
        simplified = self._encode_sequence(chords, encoding=Encoding.SIMPLIFIED)
        # pad the simplified to the sequence duration
        simplified = self._pad_sequence(simplified, pad_value=0)  # type: ignore

        return torch.Tensor(simplified).type(torch.int)

    def majmin_unique(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a root sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A root sequence of shape (n_frames, 1).
        """
        _, chords = annotation.to_interval_values()
        majmin = self._encode_sequence(chords, encoding=Encoding.MAJMIN)
        # pad the majmin to the sequence duration
        majmin = self._pad_sequence(majmin, pad_value=0)

        return torch.Tensor(majmin).type(torch.int)


if __name__ == "__main__":
    # test the class
    jams_path = "/media/data/andrea/choco_audio/jams/isophonics_1.jams"
    jam = jams.load(jams_path, validate=False)
    annotation = jam.annotations[0]
    # append to annotation a fake N
    processor = JAMSProcessor()

    converter = PumppChordConverter()
    print(converter.convert("F"))
