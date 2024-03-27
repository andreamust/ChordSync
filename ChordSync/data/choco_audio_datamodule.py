"""
PyTorch Lightning DataModule for loading audio data from the Choco dataset.
"""

import glob
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import random
from collections import defaultdict

import lightning as L
import torch
from augmentations import AudioAugmentation
from torch.utils.data import DataLoader, Dataset
from utils.chord_utils import (
    MajminChordEncoder,
    ModeEncoder,
    NoteEncoder,
    SimpleChordEncoder,
)


def _get_exerpt_name(name: str) -> str:
    """
    Get the track name and the excerpt number from the file name.
    """
    dashes = name.split("-")
    track_name = "-".join(dashes[:-1])
    return track_name


class ChocoAudioDataset(Dataset):
    """
    Dataset for loading audio data from the Choco dataset.
    """

    def __init__(self, data_path: Path, excerpt_dict: dict, augmentation: bool = False):
        super().__init__()

        # Data paths
        self.data_path = data_path

        # Data parameters
        self.excerpt_dict = excerpt_dict

        # Augmentation
        self.augmentation = augmentation

        # Data list
        self.data_list = self._get_data_list()

    def __len__(self):
        """
        Return the number of excerpts in the dataset.
        """
        return sum(self.excerpt_dict.values())

    def _get_exerpt_name(self, name: str) -> str:
        """
        Get the track name and the excerpt number from the file name.
        """
        dashes = name.split("-")
        track_name = "-".join(dashes[:-1])
        return track_name

    def _get_data_list(self):
        """
        Reconstruct the data list based on the excerpt_dict.
        """
        data_list = []
        for track in glob.glob(str(self.data_path / "*.pt")):
            track = Path(track)
            if _get_exerpt_name(track.stem) in list(self.excerpt_dict.keys()):
                data_list.append(track.name)

        return data_list

    def __getitem__(self, index):
        """
        Return the item at the given index, by taking into account the excerpt_dict.
        """
        data = torch.load(self.data_path / self.data_list[index])
        if self.augmentation:
            aug = AudioAugmentation(data[0], 40, 60)
            augmented = aug.random_masking()
            data = (augmented, *data[1:])

        return data


class ChocoAudioDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading audio data from the Choco dataset.
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 64,
        num_workers: int = 0,
        augmentation: bool = False,
    ):
        super().__init__()

        # Data paths
        self.data_path = Path(data_path)

        # Data parameters
        self.batch_size = batch_size

        # Get excerpt
        self.excerpt = self._count_excerpt()

        # vocab size
        self.vocabularies = {
            "simplified": len(SimpleChordEncoder),
            "root": len(NoteEncoder),
            "bass": len(NoteEncoder),
            "mode": len(ModeEncoder),
            "majmin": len(MajminChordEncoder),
            "onehot": 12,
            "complete": 170,
        }

        # Data loaders parameters
        self.num_workers = num_workers

        # Augmentation
        self.augmentation = augmentation

    def __len__(self):
        """
        Return the number of excerpt total.
        """
        return sum(self.excerpt.values())

    def _count_excerpt(self) -> dict:
        """
        Count the number of excerpts per track.
        """
        excerpt_count = defaultdict(int)
        for track in self.data_path.glob("*.pt"):
            # # filter out jazz partitions
            # if "jaah" in track.stem or "weimar" in track.stem:
            #     continue
            track_id = _get_exerpt_name(track.stem)
            excerpt_count[track_id] += 1

        return excerpt_count

    def _split_dataset(self, train_ratio=0.65, val_ratio=0.20, seed=None):
        """
        Split dataset based on the number of excerpts per track.

        Parameters:
        - track_counts: A dictionary where keys are track names and values are the count of excerpts.
        - train_ratio: The ratio of the dataset to be used for training.
        - val_ratio: The ratio of the dataset to be used for validation.
        - test_ratio: The ratio of the dataset to be used for testing.
        - seed: Seed for reproducibility.

        Returns:
        - train_set: Dictionary containing tracks in the training set.
        - val_set: Dictionary containing tracks in the validation set.
        - test_set: Dictionary containing tracks in the test set.
        """
        random.seed(seed)

        total_tracks = len(self)
        track_list = list(self.excerpt.keys())
        random.shuffle(track_list)

        train_size = int(total_tracks * train_ratio)
        val_size = int(total_tracks * val_ratio)
        test_size = total_tracks - train_size - val_size
        print(f"Train size: {train_size}")
        print(f"Val size: {val_size}")
        print(f"Test size: {test_size}")

        train_set, val_set, test_set = {}, {}, {}

        for track in track_list:
            if sum(train_set.values()) < train_size:
                train_set[track] = self.excerpt[track]
            elif sum(val_set.values()) < val_size:
                val_set[track] = self.excerpt[track]
            elif sum(test_set.values()) < test_size:
                test_set[track] = self.excerpt[track]

        return train_set, val_set, test_set

    def setup(self, stage) -> None:
        """
        Split dataset based on the number of excerpts per track.
        """
        train_dict, val_dict, test_dict = self._split_dataset(seed=3)

        if stage == "fit" or stage is None:
            self.train_dataset = ChocoAudioDataset(
                self.data_path, train_dict, self.augmentation
            )
            self.val_dataset = ChocoAudioDataset(
                self.data_path, val_dict, augmentation=False
            )

        if stage == "test":
            self.test_dataset = ChocoAudioDataset(
                self.data_path, test_dict, augmentation=False
            )

    def train_dataloader(self):
        """
        Return the training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Return the validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        Return the test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    data_path = "/media/data/andrea/choco_audio/cache/mel_all/"
    data_module = ChocoAudioDataModule(data_path, augmentation=True)
    a, b, c = data_module._split_dataset()
    print(sum(a.values()))
    print(sum(b.values()))
    print(sum(c.values()))
    data_module.setup("fit")
    print(len(data_module.train_dataset))
