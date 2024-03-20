"""
Audio augmentations for ChordSync.
"""

import random

import torch
import torchaudio.transforms as T
import torchvision.transforms as V


class AudioAugmentation:
    """
    Class to apply audio augmentations.
    """

    def __init__(
        self,
        mel_spectrogram: torch.Tensor,
        time_mask_param: int = 80,
        frequency_mask_param: int = 80,
    ):
        self.mel_spectrogram = mel_spectrogram
        self.time_mask_param = int(time_mask_param)
        self.frequency_mask_param = int(frequency_mask_param)

    def time_masking(self):
        """
        Apply time masking to the mel spectrogram.
        """
        time_mask = T.TimeMasking(self.time_mask_param)
        return time_mask(self.mel_spectrogram)

    def frequency_masking(self):
        """
        Apply frequency masking to the mel spectrogram.
        """
        freq_mask = T.FrequencyMasking(self.frequency_mask_param)
        return freq_mask(self.mel_spectrogram)

    def time_frequency_masking(self):
        """
        Apply both time and frequency masking to the mel spectrogram.
        """
        time_mask = T.TimeMasking(self.time_mask_param)
        freq_mask = T.FrequencyMasking(self.frequency_mask_param)
        return freq_mask(time_mask(self.mel_spectrogram))

    def gaussian_blur(self):
        """
        Apply Gaussian blur to the mel spectrogram.
        """
        gaussian_blur = V.GaussianBlur(3, (1, 3))
        return gaussian_blur(self.mel_spectrogram)

    def random_masking(self):
        """
        Apply random masking to the mel spectrogram. The random masking can be
        either time masking, frequency masking, or both.
        """
        random_options = ["time", "freq", "both", "blur"]
        random_mask = random.choice(random_options)

        if random_mask == "time":
            return self.time_masking()
        elif random_mask == "freq":
            return self.frequency_masking()
        elif "both":
            return self.time_frequency_masking()
        else:
            return self.gaussian_blur()
