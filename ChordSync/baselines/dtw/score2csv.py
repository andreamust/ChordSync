from pathlib import Path
from typing import Optional, Union

import jams
import pandas as pd
from harte.harte import Harte
from music21 import converter


def midi_to_csv(
    midi_file: str,
    types: tuple = ("Note", "Chord"),
    save: bool = True,
    output_path: Optional[Union[str, Path]] = ".",
) -> pd.DataFrame:
    """
    Converts a MIDI file to a Pandas DataFrame.
    :param midi_file: the path to the MIDI file
    :type midi_file: str
    :param types: the types of elements to extract from the MIDI file
    :type types: tuple
    :param save: whether to save the DataFrame to a CSV file or not
    :type save: bool
    :param output_path: the path to the output CSV file
    :type output_path: str | Path
    :return: the DataFrame containing the MIDI file information
    :rtype: pd.DataFrame
    """
    midi_stream = converter.parse(midi_file)
    data = []

    for element in midi_stream.flat:
        if "Note" in element.classes and "Note" in types:
            start_time = element.offset
            duration = element.duration.quarterLength
            pitch = element.pitch.midi
            velocity = element.volume.velocity
            instrument_name = element.activeSite.getInstrument().instrumentName

            data.append(
                {
                    "Start": float(start_time),
                    "Duration": float(duration),
                    "Pitch": int(pitch),
                    "Velocity": velocity,
                    "Instrument": instrument_name,
                }
            )

        # decompose chords
        if "Chord" in element.classes and "Chord" in types:
            print(element, element.offset, element.duration.quarterLength)
            for chord_note in element.pitches:
                start_time = element.offset
                duration = element.duration.quarterLength
                pitch = chord_note.midi
                velocity = element.volume.velocity
                instrument_name = element.activeSite.getInstrument().instrumentName

                data.append(
                    {
                        "Start": float(start_time),
                        "Duration": float(duration),
                        "Pitch": int(pitch),
                        "Velocity": velocity,
                        "Instrument": instrument_name,
                    }
                )

    if save:
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

    return pd.DataFrame(data)


def jams_to_csv(jams_file: Union[str, Path], csv_file: Union[str, Path]) -> None:
    """
    Convert a JAMS file to a CSV file.
    :param jams_file: the JAMS file to convert
    :type jams_file: str
    :param csv_file: the CSV file to save the data to
    :type csv_file: str
    :return: None
    :rtype: None
    """
    jams_file = str(jams_file)

    # load JAMS
    jam = jams.load(jams_file)
    # get the first annotation that contains "chord" in its namespace
    ann = jam.annotations.search(namespace="chord_harte")[0]

    # get data
    data = []

    time = ann.data[0].time  # type: ignore
    for obs in ann.data:  # type: ignore
        if obs.value != "N":
            # open the chord with hartelib
            harte = Harte(obs.value)
            pitches = harte.pitches
            for pitch in pitches:
                # convert pitch to midi
                pitch = pitch.midi
                # append to DataFrame
                data.append(
                    {
                        "Start": float(time),
                        "Duration": float(obs.duration),
                        "Pitch": int(pitch),
                        "Velocity": 127,
                        "Instrument": "instrument",
                    }
                )
            time += obs.duration

    # save to CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    # convert a MIDI file to a DataFrame
    # midi_to_dataframe(
    #     midi_file="../data/schubert-winterreise/row/score_midi/Schubert_D911-01.mid",
    #     types=("Chord",),
    #     save=True,
    #     output_path="./ciao.csv",
    # )

    JAMS_PATH = Path("/media/data/andrea/choco_audio/jams/")

    # convert a JAMS file to a CSV file
    jams_to_csv(
        jams_file=JAMS_PATH / "schubert-winterreise-audio_22.jams",
        csv_file="ChordSync/baselines/dtw/data/ciao.csv",
    )
