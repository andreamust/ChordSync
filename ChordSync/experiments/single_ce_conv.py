import os
import sys

import torchmetrics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

#
from data.choco_audio_datamodule import ChocoAudioDataModule
from evaluation.mireaval_metrics import EvaluateAlignment
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.optimizer import Optimizer
from torchaudio.models import Conformer
from utils.torch_utils import (
    PositionalEncoding,
    remove_probabilities,
    smooth_probabilities,
    wrong_probabilities_loss,
)

import wandb


class ForcedAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        # penalize wrong probabilities
        loss_wrong = wrong_probabilities_loss(predictions, targets)
        # calculate loss
        loss_chord = F.cross_entropy(predictions, targets)

        loss = loss_chord + (loss_wrong * 2)
        return loss


class DeepChromaConformer(L.LightningModule):
    """
    TBD
    """

    def __init__(
        self,
        num_classes_root: int,
        num_classes_mode: int,
        num_classes_chord: int,
        num_classes_majmin: int,
        min_learning_rate: float,
        max_learning_rate: float,
        **kwargs,
    ):
        super().__init__()

        # setup learning rate
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate

        # num  classes
        self.num_classes_root = num_classes_root
        self.num_classes_mode = num_classes_mode
        self.num_classes_chord = num_classes_chord
        self.num_classes_majmin = num_classes_majmin

        # save parameters
        self.save_hyperparameters()

        # cache evaluation metrics for each epoch
        self.training_step_outputs = []
        self.eval_step_outputs = []

        # positional encoding
        self.positional_encoding = PositionalEncoding(
            128,
        )

        # unpack kwargs for convoltuion layers and conformer
        self.conformer_kwargs = kwargs.get("conformer_kwargs", {})

        # get dimensions
        self.conformer_dimension = self.conformer_kwargs.get("input_dim", 128)

        # init metrics
        self.accuracy_majmin = torchmetrics.Accuracy(
            num_classes=num_classes_chord, task="multiclass", average="macro"
        )
        self.alignment_evaluation = EvaluateAlignment(
            blank=0, window_size=0.8, audio_length=15
        )

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
        loss = ForcedAlignmentLoss()(out_majmin, majmin_sequence)

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

    def training_step(self, batch, batch_idx):
        # get outputs and losses
        targets, outputs, losses, alignment_evaluation = self._shared_step(
            batch, batch_idx, mireval=True
        )

        # get mireval evaluations
        absolute_error, percentage_correct = alignment_evaluation  # type: ignore

        # log loss
        self.log("train_loss", losses["loss"])

        # calculate accuracy using pytorch lightning metrics
        self.accuracy_majmin(outputs["out_majmin"], targets["majmin"])

        # log accuracy
        self.log("train_acc_majmin", self.accuracy_majmin)

        # log mireval metrics
        if percentage_correct is not None:
            self.log("train_mireval", percentage_correct)
            self.log("train_mireval_aberror", absolute_error)

        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        # get outputs and losses
        targets, outputs, losses, alignment_evaluation = self._shared_step(
            batch, batch_idx, mireval=True
        )

        # get mireval evaluations
        absolute_error, percentage_correct = alignment_evaluation  # type: ignore

        # log loss
        self.log("val_loss", losses["loss"])

        # calculate accuracy using pytorch lightning metrics
        self.accuracy_majmin(outputs["out_majmin"], targets["majmin"])

        # log accuracy
        self.log("val_acc_majmin", self.accuracy_majmin)

        # log mireval metrics
        if percentage_correct is not None:
            self.log("val_mireval", percentage_correct)
            self.log("val_mireval_aberror", absolute_error)

        # log samples
        if batch_idx == 0:
            targets = targets["majmin"][1]
            outs = outputs["out_majmin"][1]

            # log the plot as an image to wandb
            image = wandb.Image(torch.sigmoid(outs), caption="val_predictions")
            wandb.log({"val_predictions": image})

            print("targets", targets)
            print("outs", outs.argmax(0))

        return losses["loss"]


def main(
    project_name: str,
    name: str,
    data_path: str,
    conformer_kwargs: dict,
    max_learning_rate: float,
    min_learning_rate: float,
    max_epochs: int,
    batch_size: int,
    num_workers: int,
    log_every_n_steps: int = 1,
    check_val_every_n_epoch: int = 5,
    accumulate_grad_batches: int = 1,
    precision: str = "bf16-mixed",
    debug: bool = False,
):
    # set precision
    torch.set_float32_matmul_precision("medium")

    # set debug mode
    limit_train_batches = 0.2 if debug else None
    limit_val_batches = 0.1 if debug else None
    detect_anomaly = True if debug else False

    # create dataset
    data_module = ChocoAudioDataModule(
        data_path, batch_size=batch_size, num_workers=num_workers, augmentation=True
    )
    print(f"Dataset size: {len(data_module)}")
    # get vocabularies
    vocabularies = data_module.vocabularies

    wandb_logger = WandbLogger(
        log_model=False,
        project=project_name,
        group="majmin",
        name=name,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = L.Trainer(
        accelerator="cuda",
        devices=[0],
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        logger=wandb_logger,  # type: ignore
        accumulate_grad_batches=accumulate_grad_batches,
        detect_anomaly=detect_anomaly,
        callbacks=[lr_monitor],
        precision=precision,  # type: ignore
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )

    model = DeepChromaConformer(
        num_classes_root=vocabularies["root_sequence"] + 1,
        num_classes_mode=vocabularies["mode_sequence"] + 1,
        num_classes_chord=vocabularies["simplified_sequence"] + 1,
        num_classes_majmin=vocabularies["majmin_sequence"] + 1,
        min_learning_rate=min_learning_rate,
        max_learning_rate=max_learning_rate,
        conformer_kwargs=conformer_kwargs,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        # ckpt_path="last",
    )
    # trainer.test(
    #     model,
    #     datamodule=data_module,
    #     ckpt_path="best",
    # )


if __name__ == "__main__":
    DATA_PATH = "/media/data/andrea/choco_audio/cache/mel_all_new/"

    CONFORMER_KWARGS = {
        "input_dim": 256,
        "num_layers": 16,
        "ffn_dim": 128,
        "num_heads": 4,
        "dropout": 0.5,
        "depthwise_conv_kernel_size": 31,
        "use_group_norm": False,
        "convolution_first": True,
    }

    main(
        project_name="forced-alignment",
        name="double-ce-conv#",
        data_path=DATA_PATH,
        conformer_kwargs=CONFORMER_KWARGS,
        max_learning_rate=3e-4,
        min_learning_rate=3e-5,
        max_epochs=150,
        batch_size=64,
        num_workers=16,
        log_every_n_steps=10,
        check_val_every_n_epoch=5,
        accumulate_grad_batches=1,
        precision="bf16-mixed",
        debug=False,
    )
