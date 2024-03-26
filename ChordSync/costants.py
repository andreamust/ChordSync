"""
Define the constants used in ChordSync.
"""

from enum import Enum

PROGRAM_NAME = "ChordSync"
PROGRAM_VERSION = "0.1.0"
PROGRAM_DESCRIPTION = "Chord alignment using conformer networks"

# Data


class Paths(Enum):
    audio = "/media/data/andrea/choco_audio/audio"
    jams = "/media/data/andrea/choco_audio/jams"
