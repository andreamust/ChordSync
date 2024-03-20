"""
Preocessing script for Schubert dataset. Converts JAMS files to CSV files.
"""

import sys
from pathlib import Path

sys.path.append("ChordSync")


import jams_namespaces
from score2csv import jams_to_csv


def process_jams(
    dataset_path: Path,
    partition_name: str,
    output_path: Path = Path("ChordSync/baselines/dtw/data"),
) -> None:
    """
    Converts the JAMS files to a CSV file containing the information about
    the notes and chords.
    :param dataset_path: the path to the dataset
    :type dataset_path: Path
    :param partition_name: the name of the partition
    :type partition_name: str
    :param output_path: the path to the output CSV file
    :type output_path: Path
    :return: None
    :rtype: None
    """
    # iterate over the JAMS files if partition name is contained in the file name
    for jams_file in dataset_path.glob("*.jams"):
        if partition_name in jams_file.name:
            print(f"Processing {jams_file.name}")
            # if output_path does not exist, create it
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)

            # convert the JAMS file to a DataFrame
            jams_to_csv(jams_file, output_path / f"{jams_file.stem}.csv")


if __name__ == "__main__":
    # process the dataset
    process_jams(
        dataset_path=Path(
            "ChordSync/baselines/dtw/data/schubert-winterreise/jams-score"
        ),
        partition_name="schubert-winterreise",
        output_path=Path("ChordSync/baselines/dtw/data/schubert-winterreise/csv-score"),
    )
