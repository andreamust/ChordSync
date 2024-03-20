"""
Main script for encoding chord as numerical values in different formats.
The implementations available are:
    - root: encodes the root of the chord as a number from 1 to 12, plus 13 for "N"
    - mode: encodes the mode of the chord as a number from 1 to 7, plus 8 for "N"
    - bass: encodes the bass of the chord as a number from 1 to 12, plus 13 for "N"
    - majmin: encodes the chord as a number from 1 to 25, plus 26 for "N"
    - simplified: encodes the chord as a product of the root and the mode, as a 
      number from 1 to 84, plus 85 for "N"
"""

from ast import Or
from collections import OrderedDict
from enum import CONTINUOUS, Enum, verify
from json import decoder
from re import S

from harte.harte import Harte
from music21 import pitch


class Encoding(Enum):
    """
    Enum for the different encodings available.
    """

    ROOT = 1
    BASS = 2
    MODE = 3
    MAJMIN = 4
    MAJMIN7 = 5
    SIMPLIFIED = 6
    COMPLETE = 7


@verify(CONTINUOUS)
class ModeEncoder(Enum):
    """
    Enum for the different encodings available.
    """

    MAJOR = 1
    MINOR = 2
    MAJOR_SEVENTH = 3
    MINOR_SEVENTH = 4
    AUGMENTED = 5
    AUGMENTED_SEVENTH = 5
    DIMINISHED = 6
    DIMINISHED_SEVENTH = 6
    OTHER = 7
    OTHER_SEVENTH = 7
    N = 8


class ModeMajminEncoder(Enum):
    """
    Enum for the different encodings available.
    """

    MAJOR = 1
    MINOR = 2
    OTHER = 3
    AUGMENTED = 3
    DIMINISHED = 3
    N = 4


@verify(CONTINUOUS)
class NoteEncoder(Enum):
    """
    Enum for the different encodings available.
    """

    C = 1
    C_SHARP = 2
    D = 3
    D_SHARP = 4
    E = 5
    F = 6
    F_SHARP = 7
    G = 8
    G_SHARP = 9
    A = 10
    A_SHARP = 11
    B = 12
    N = 13


def simple_combinations() -> list:
    """
    Create all combinations between notes and modes.
    """
    notes = [note for note in NoteEncoder if note.name != "N"]
    modes = [mode for mode in ModeEncoder if mode.name != "N"]

    combinations = [
        (f"{note.name}_{mode.name}", (note.value + mode.value - 1) + (i * 6))
        for i, note in enumerate(notes)
        for mode in modes
    ]

    # add N to combinations assigning it the max value + 1
    combinations.append(("N", max([c[1] for c in combinations]) + 1))

    return combinations


SimpleChordEncoder = Enum("SimpleChordEncoder", simple_combinations())


def majmin_combinations() -> list:
    """
    Create all combinations between notes and modes.
    """
    notes = [note for note in NoteEncoder if note.name != "N"]
    modes = [
        mode
        for mode in ModeMajminEncoder
        if mode.name != "N" and mode.name in ["MAJOR", "MINOR", "OTHER"]
    ]

    combinations = [
        (f"{note.name}_{mode.name}", (note.value + mode.value - 1) + (i * 2))
        for i, note in enumerate(notes)
        for mode in modes
    ]

    # add N to combinations assigning it the max value + 1
    combinations.append(("N", max([c[1] for c in combinations]) + 1))

    return combinations


MajminChordEncoder = Enum("MajminChordEncoder", majmin_combinations())


class ChordEncoder:
    """
    Class for encoding chords as numerical values.
    """

    def __init__(self):
        """
        Initializes the encoder with the given encoding.

        Args:
            encoding (Encoding): The encoding to use.
        """

    def encode(self, chord: str, encoding: Encoding = Encoding.SIMPLIFIED) -> int:
        """
        Encodes the given chord as a numerical value.

        Args:
            chord (str): The chord to encode.

        Returns:
            int: The numerical value of the chord.
        """
        if encoding == Encoding.ROOT:
            return self._encode_root(chord)
        elif encoding == Encoding.BASS:
            return self._encode_bass(chord)
        elif encoding == Encoding.MODE:
            return self._encode_mode(chord)
        elif encoding == Encoding.SIMPLIFIED:
            return self._encode_simplified(chord)
        elif encoding == Encoding.MAJMIN:
            return self._encode_majmin(chord)
        else:
            raise ValueError("Invalid encoding.")

    def _open_harte(self, chord: str) -> Harte:
        """
        Creates a Harte chord object for the given chord.

        Args:
            chord (str): The chord to create a Harte object for.

        Returns:
            Harte: The Harte object for the given chord.
        """
        if "/bb1" in chord:
            chord = chord.replace("/bb1", "/7")

        return Harte(chord)

    def _encode_note(self, pitch: pitch.Pitch) -> int:
        """
        Creates a chord embedding for the given chord.

        Args:
            chord (str): The chord to create an embedding for.

        Returns:
            int: The encoded chord embedding.
        """
        pitch.octave = 0
        return pitch.pitchClass + 1

    def _encode_root(self, chord: str) -> int:
        """
        Creates a chord embedding for the given chord.

        Args:
            chord (str): The chord to create an embedding for.

        Returns:
            int: The encoded chord embedding.
        """
        if chord == "N":
            return NoteEncoder.N.value

        harte_chord = self._open_harte(chord)
        # get root from m21
        root: pitch.Pitch = harte_chord.root()  # type: ignore

        return self._encode_note(root)

    def _encode_bass(self, chord: str) -> int:
        """
        Creates a chord embedding for the given chord.

        Args:
            chord (str): The chord to create an embedding for.

        Returns:
            int: The encoded chord embedding.
        """

        if chord == "N":
            return NoteEncoder.N.value

        harte_chord = self._open_harte(chord)
        # get root from m21
        root: pitch.Pitch = harte_chord.bass()  # type: ignore

        return self._encode_note(root)

    def _encode_mode(self, chord: str) -> int:
        """
        Creates a chord embedding for the given chord.

        Args:
            chord (str): The chord to create an embedding for.

        Returns:
            int: The encoded chord embedding.
        """
        if chord == "N":
            return ModeEncoder.N.value

        harte_chord = self._open_harte(chord)
        # get mode from m21
        mode = harte_chord.quality
        # get seventh from m21
        is_seventh = harte_chord.isSeventh()
        # merge mode and seventh
        is_seventh = "_seventh" if is_seventh else ""
        mode = mode + is_seventh

        # get encoding from enum
        mode = ModeEncoder[mode.upper()].value

        return mode

    def _encode_majmin(self, chord: str) -> int:
        """
        Creates a chord embedding for the given chord.

        Args:
            chord (str): The chord to create an embedding for.

        Returns:
            int: The encoded chord embedding.
        """
        valid_modes = ["MAJOR", "MINOR", "OTHER", "N"]
        mode = self._encode_mode_majmin(chord)
        root = self._encode_root(chord)

        mode_name = ModeMajminEncoder(mode).name
        root_name = NoteEncoder(root).name

        mode_name = mode_name if mode_name in valid_modes else "OTHER"

        search_name = f"{root_name}_{mode_name}" if mode_name != "N" else "N"

        return MajminChordEncoder[search_name].value

    def _encode_mode_majmin(self, chord: str) -> int:
        """
        Creates a chord embedding for the given chord.

        Args:
            chord (str): The chord to create an embedding for.

        Returns:
            int: The encoded chord embedding.
        """
        if chord == "N":
            return ModeMajminEncoder.N.value

        harte_chord = self._open_harte(chord)
        # get mode from m21
        mode = harte_chord.quality

        # get encoding from enum
        mode = ModeMajminEncoder[mode.upper()].value

        return mode

    def _encode_simplified(self, chord: str) -> int:
        """
        Creates a chord embedding for the given chord.

        Args:
            chord (str): The chord to create an embedding for.

        Returns:
            int: The encoded chord embedding.
        """
        mode = self._encode_mode(chord)
        root = self._encode_root(chord)

        mode_name = ModeEncoder(mode).name
        root_name = NoteEncoder(root).name

        search_name = f"{root_name}_{mode_name}" if mode_name != "N" else "N"

        return SimpleChordEncoder[search_name].value


class ChordDecoder:
    """
    Class for decoding chords from numerical values.
    """

    MODE_MAP = OrderedDict(
        {
            "MAJOR_SEVENTH": "maj7",
            "MINOR_SEVENTH": "min7",
            "AUGMENTED_SEVENTH": "aug",
            "DIMINISHED_SEVENTH": "dim",
            "OTHER_SEVENTH": "maj7",
            "MAJOR": "maj",
            "MINOR": "min",
            "OTHER": "maj",
            "AUGMENTED": "aug",
            "DIMINISHED": "dim",
            "": "maj",
        }
    )

    def __init__(self):
        """
        Initializes the decoder with the given encoding.

        Args:
            encoding (Encoding): The encoding to use.
        """

    def decode(self, chord: int, encoding: Encoding = Encoding.SIMPLIFIED) -> str:
        """
        Decodes the given chord from a numerical value.

        Args:
            chord (int): The chord to decode.

        Returns:
            str: The chord corresponding to the given value.
        """
        if encoding == Encoding.ROOT:
            decoded = self._decode_root(chord)
            return self._decode_label(decoded)
        elif encoding == Encoding.BASS:
            decoded = self._decode_bass(chord)
            return self._decode_label(decoded)
        elif encoding == Encoding.MODE:
            decoded = self._decode_mode(chord)
            return self._decode_label(decoded)
        elif encoding == Encoding.SIMPLIFIED:
            decoded = self._decode_simplified(chord)
            return self._decode_label(decoded)
        elif encoding == Encoding.MAJMIN:
            decoded = self._decode_majmin(chord)
            return self._decode_label(decoded)
        else:
            raise ValueError("Invalid encoding.")

    def _decode_label(self, chord: str) -> str:
        """
        Converts the given chord key to a harte label.
        """
        if chord == "N":
            return "N"

        root = chord.split("_")[0]
        sharp = "#" if "SHARP" in chord else ""
        flat = "b" if "FLAT" in chord else ""
        modes = chord.replace("_SHARP", "").replace("_FLAT", "").split("_")[1:]
        modes = "_".join(modes)
        modes = self.MODE_MAP[modes]
        return f"{root}{sharp}{flat}:{modes}"

    def _decode_root(self, chord: int) -> str:
        """
        Decodes the given chord from a numerical value.

        Args:
            chord (int): The chord to decode.

        Returns:
            str: The chord corresponding to the given value.
        """
        if chord == NoteEncoder.N.value:
            return "N"

        # get note from enum
        note = NoteEncoder(chord).name

        return note

    def _decode_bass(self, chord: int) -> str:
        """
        Decodes the given chord from a numerical value.

        Args:
            chord (int): The chord to decode.

        Returns:
            str: The chord corresponding to the given value.
        """
        if chord == NoteEncoder.N.value:
            return "N"

        # get note from enum
        note = NoteEncoder(chord).name

        return note

    def _decode_mode(self, chord: int) -> str:
        """
        Decodes the given chord from a numerical value.

        Args:
            chord (int): The chord to decode.

        Returns:
            str: The chord corresponding to the given value.
        """
        if chord == ModeEncoder.N.value:
            return "N"

        # get mode from enum
        mode = ModeEncoder(chord).name

        return mode

    def _decode_majmin(self, chord: int) -> str:
        """
        Decodes the given chord from a numerical value.

        Args:
            chord (int): The chord to decode.

        Returns:
            str: The chord corresponding to the given value.
        """
        if chord == MajminChordEncoder.N.value:  # type: ignore
            return "N"

        # get mode from enum
        mode = MajminChordEncoder(chord).name

        return mode

    def _decode_simplified(self, chord: int) -> str:
        """
        Decodes the given chord from a numerical value.

        Args:
            chord (int): The chord to decode.

        Returns:
            str: The chord corresponding to the given value.
        """
        if chord == SimpleChordEncoder.N.value:  # type: ignore
            return "N"

        # get mode from enum
        mode = SimpleChordEncoder(chord).name

        return mode


if __name__ == "__main__":
    # test cases
    encoder = ChordEncoder()
    # print(encoder.encode("F:maj6", Encoding.ROOT))
    # print(encoder.encode("F#:(2,3,6)/5", Encoding.BASS))
    print(encoder.encode("C:maj7", Encoding.MAJMIN))

    decoder = ChordDecoder()
    for i in range(1, 9):
        print(decoder.decode(i, Encoding.MAJMIN))
