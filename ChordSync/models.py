import lightning as L
import torch
import torch.nn as nn

#
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.optimizer import Optimizer
from torchaudio.models import Conformer
from utils.torch_utils import PositionalEncoding


class ConvolutionalConformer(L.LightningModule):
    """
    TBD
    """

    def __init__(
        self,
        vocabularies: dict,
        min_learning_rate: float,
        max_learning_rate: float,
        criterion: nn.Module,
        prediction_mode: list[str] = ["simplified"],
        convolution: bool = False,
        **kwargs,
    ):
        super().__init__()

        # prediction modes
        self.prediction_mode = prediction_mode

        # setup learning rate
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate

        # vocabularies
        self.vocabularies = vocabularies

        # loss
        self.loss = criterion

        # positional encoding
        self.positional_encoding = PositionalEncoding(
            128,
        )

        # unpack kwargs for convoltuion layers and conformer
        self.conformer_kwargs = kwargs.get("conformer_kwargs", {})

        # get dimensions
        self.conformer_dimension = self.conformer_kwargs.get("input_dim", 128)

        # convolution layers
        self.convolution = None

        if convolution:
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

        # dynamically create fully connected layers as self.fully_connected_{mode}
        for mode in self.prediction_mode:
            setattr(
                self,
                f"fully_connected_{mode}",
                nn.Linear(self.conformer_dimension, self.vocabularies[mode]),
            )

    def forward(self, x):

        # itialize outputs
        outputs = {}

        # define durations
        durations = torch.full(
            size=(x.shape[0],),
            fill_value=x.shape[-1],
            dtype=torch.long,
            device=self.device,
        )

        # convolution
        if self.convolution:
            # x = x.squeeze(1).permute(0, 2, 1)
            # x = self.positional_encoding(x)
            # x = x.permute(0, 2, 1)
            # x = x.unsqueeze(1)
            convolution = self.convolution(x)

            # reshape for conformer (B, T, N)
            convolution = convolution.squeeze()
            x = convolution.permute(0, 2, 1)

        # positional encoding
        x = self.positional_encoding(x)

        # fully connected
        fc_preconformer = self.fully_connected_preconformer(convolution)

        # conformer
        conformer_encoder, _ = self.conformer_encoder(fc_preconformer, durations)

        # fully connected
        for mode in self.prediction_mode:
            outputs[mode] = getattr(self, f"fully_connected_{mode}")(conformer_encoder)

        return outputs

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

        # reshape inputs
        x = x.float()

        # forward pass
        outputs = self(x)

        # calculate loss
        losses = self.loss(outputs, y)

        # if mireval:
        #     # mireval metrics
        #     alignment_evaluation = self.alignment_evaluation.evaluate(
        #         out_majmin, majmin_sequence, onset_sequence
        #     )

        return y, outputs, losses, alignment_evaluation
