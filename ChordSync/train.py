import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.choco_audio_datamodule import ChocoAudioDataModule
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import MulticlassAccuracy

import wandb
from models import ConformerModel  # type: ignore

# from evaluation.mireaval_metrics import EvaluateAlignment


class ConformerLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        # calculate loss
        loss = F.cross_entropy(predictions["root"], targets["root"])

        return {"loss": loss}


class MultiConformer(ConformerModel):
    """
    TBD
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # save parameters
        self.save_hyperparameters(ignore=["criterion"])

        # init metrics for each of the prediction modes
        self.metrics = nn.ModuleDict()
        for mode in self.prediction_mode:
            self.metrics[mode] = MulticlassAccuracy(
                num_classes=self.vocabularies[mode] + 1
            )

        # self.alignment_evaluation = EvaluateAlignment(
        #     blank=0, window_size=0.8, audio_length=15
        # )

    def training_step(self, batch, batch_idx):
        # get outputs and losses
        targets, outputs, losses, alignment_evaluation = self._shared_step(
            batch, batch_idx, mireval=False
        )

        # get mireval evaluations
        if alignment_evaluation is not None:
            absolute_error, percentage_correct = alignment_evaluation  # type: ignore

        # log loss
        for loss in losses:
            self.log(f"train_{loss}", losses[loss])

        # calculate and log accuracy using pytorch lightning metrics
        for mode in self.prediction_mode:
            self.log(
                f"train_acc_{mode}", self.metrics[mode](outputs[mode], targets[mode])
            )

        # log mireval metrics
        if alignment_evaluation is not None:
            self.log("train_mireval", percentage_correct)
            self.log("train_mireval_aberror", absolute_error)

        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        # get outputs and losses
        targets, outputs, losses, alignment_evaluation = self._shared_step(
            batch, batch_idx, mireval=False
        )

        # get mireval evaluations
        if alignment_evaluation is not None:
            absolute_error, percentage_correct = alignment_evaluation  # type: ignore

        # log loss
        for loss in losses:
            self.log(f"val_{loss}", losses[loss])

        # calculate and log accuracy using pytorch lightning metrics
        for mode in self.prediction_mode:
            self.log(
                f"val_acc_{mode}",
                self.metrics[mode](outputs[mode], targets[mode]).to(self.device),
            )

        # log mireval metrics
        if alignment_evaluation is not None:
            self.log("val_mireval", percentage_correct)
            self.log("val_mireval_aberror", absolute_error)

        # log samples
        if batch_idx == 0:
            targets = targets["root"][1]
            outs = outputs["root"][1]

            # log the plot as an image to wandb
            image = wandb.Image(torch.softmax(outs, dim=0), caption="val_predictions")
            wandb.log({"val_predictions": image})

            image = wandb.Image(
                torch.eye(self.vocabularies["root"] + 1, device=targets.device)[
                    targets
                ].T,
                caption="val_targets",
            )
            wandb.log({"val_targets": image})

            print("targets", targets)
            print("outs", outs.argmax(0))

        return losses["loss"]


def main(
    project_name: str,
    group_name: str,
    run_name: str,
    data_path: str,
    criterion: nn.Module,
    max_learning_rate: float,
    min_learning_rate: float,
    max_epochs: int,
    batch_size: int,
    num_workers: int,
    prediction_mode: list[str] = ["simplified"],
    convolution: bool = False,
    log_every_n_steps: int = 1,
    check_val_every_n_epoch: int = 5,
    accumulate_grad_batches: int = 1,
    precision: str = "bf16-mixed",
    debug: bool = False,
    test: bool = False,
    test_checkpoint: str | None = None,
    **kwargs,
):
    # set precision
    torch.set_float32_matmul_precision("medium")

    # set debug mode
    limit_train_batches = 0.2 if debug else None
    limit_val_batches = 0.1 if debug else None
    detect_anomaly = True if debug else False

    # create datamodule
    data_module = ChocoAudioDataModule(
        data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation=False,
    )
    print(f"Dataset size: {len(data_module)}")

    # get vocabularies
    vocabularies = data_module.vocabularies

    # create wandb logger
    wandb_logger = WandbLogger(
        log_model=False,
        project=project_name,
        group=group_name,
        name=run_name,
    )

    # create learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # create trainer
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

    # create model
    model = MultiConformer(
        vocabularies=vocabularies,
        convolution=convolution,
        criterion=criterion,
        min_learning_rate=min_learning_rate,
        max_learning_rate=max_learning_rate,
        prediction_mode=prediction_mode,
        **kwargs,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        # ckpt_path="last",
    )

    # test model
    if test:
        trainer.test(
            model,
            datamodule=data_module,
            ckpt_path=test_checkpoint,
        )


if __name__ == "__main__":
    DATA_PATH = "/media/data/andrea/choco_audio/cache/mel_dict/"

    CONFORMER_KWARGS = {
        "input_dim": 256,
        "num_layers": 16,
        "ffn_dim": 128,
        "num_heads": 4,
        "dropout": 0.4,
        "depthwise_conv_kernel_size": 31,
        "use_group_norm": False,
        "convolution_first": True,
    }

    main(
        project_name="chord-sync",
        group_name="forced-alignment",
        run_name="root",
        data_path=DATA_PATH,
        prediction_mode=["root"],
        convolution=False,
        criterion=ConformerLoss(),
        max_learning_rate=3e-4,
        min_learning_rate=3e-4,
        max_epochs=150,
        batch_size=64,
        num_workers=16,
        log_every_n_steps=10,
        check_val_every_n_epoch=5,
        accumulate_grad_batches=1,
        precision="bf16-mixed",
        debug=False,
        convolution_kwargs={},
        conformer_kwargs=CONFORMER_KWARGS,
        test=False,
    )
