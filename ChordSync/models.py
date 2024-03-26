import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.optimizer import Optimizer
from torchaudio.models import Conformer
from utils.torch_utils import PositionalEncoding, smooth_probabilities


class ConvolutionalConformer(L.LightningModule):
    """
    TBD
    """

    def __init__(
        self,
        num_classes_chord: int,
        min_learning_rate: float,
        max_learning_rate: float,
        **kwargs,
    ):
        super().__init__()

        # setup learning rate
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate

        # num  classes
        self.num_classes_chord = num_classes_chord

        # positional encoding
        self.positional_encoding = PositionalEncoding(
            128,
        )

        # unpack kwargs for convoltuion layers and conformer
        self.conformer_kwargs = kwargs.get("conformer_kwargs", {})

        # get dimensions
        self.conformer_dimension = self.conformer_kwargs.get("input_dim", 128)

        # convolution layers
        self.convolution = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            ),
            nn.LeakyReLU(negative_slope=0.3),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)),
            nn.Dropout(p=0.4),
        )

        # adapt dimensionality for the first
        self.fully_connected_preconformer = nn.Sequential(
            nn.Linear(64, self.conformer_dimension),
            nn.Dropout(0.4),
        )

        # convolution layers
        self.conformer_encoder = Conformer(**self.conformer_kwargs)

        # fully connected
        self.fully_connected_boundaries = nn.Linear(
            self.conformer_dimension, self.num_classes_chord
        )

    def forward(self, x):

        # define durations
        durations = torch.full(
            size=(x.shape[0],),
            fill_value=x.shape[-1],
            dtype=torch.long,
            device=self.device,
        )

        # conformer
        x = x.squeeze(1).permute(0, 2, 1)
        x = self.positional_encoding(x)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        convolution = self.convolution(x)

        # reshape for conformer (B, T, N)
        convolution = convolution.squeeze()
        convolution = convolution.permute(0, 2, 1)

        # fully connected
        fc_preconformer = self.fully_connected_preconformer(convolution)

        # conformer
        conformer_encoder, _ = self.conformer_encoder(fc_preconformer, durations)

        # fully connected
        y_pred_boundaries = self.fully_connected_boundaries(conformer_encoder)
        # y_pred_boundaries = torch.sigmoid(y_pred_boundaries)

        return y_pred_boundaries.squeeze()

    def configure_optimizers(self) -> OptimizerLRScheduler | Optimizer | None | dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.max_learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=self.min_learning_rate, last_epoch=-1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_acc",
        }

    def _shared_step(self, batch, batch_idx, mireval: bool = True):
        # initialize mireval metrics
        alignment_evaluation = None
        # get outputs
        x, y = batch
        # y.shape = torch.Size([batch, len, 10])
        (
            simplified,
            majmin,
            _,
            _,
            onset_sequence,
            simplified_sequence,
            majmin_sequence,
            *_,
        ) = torch.split(y, 1, dim=2)

        # convert to long and squeeze
        simplified = simplified.type(torch.long).squeeze(-1)
        simplified_sequence = simplified_sequence.type(torch.long).squeeze(-1)
        onset_sequence = onset_sequence.type(torch.float).squeeze(-1)
        majmin = simplified
        majmin_sequence = simplified_sequence

        # reshape inputs
        x = x.float()

        # forward pass
        out_majmin = self(x)
        out_majmin = smooth_probabilities(out_majmin, majmin_sequence, sigma=0.7)
        out_majmin = out_majmin.permute(0, 2, 1)

        # calculate loss
        loss = F.cross_entropy(out_majmin, majmin_sequence)

        # out_boundaries = torch.sigmoid(out_boundaries)

        if mireval:
            # mireval metrics
            alignment_evaluation = self.alignment_evaluation.evaluate(
                out_majmin, majmin_sequence, onset_sequence
            )

        # group targets outputs and losses
        targets = {
            "simplified": simplified,
            "onset_sequence": onset_sequence,
            "majmin": majmin,
        }
        outputs = {
            "out_majmin": out_majmin,
        }
        losses = {
            "loss": loss,
        }

        return targets, outputs, losses, alignment_evaluation
