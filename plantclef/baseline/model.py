import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from .losses import ASLSingleLabel


class LinearClassifier(pl.LightningModule):
    def __init__(self, num_features: int, num_classes: int, asl_loss: bool = False):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.save_hyperparameters()  # Saves hyperparams in the checkpoints
        self.loss = torch.nn.functional.nll_loss
        self.asl_loss = asl_loss
        if self.asl_loss:
            self.loss = ASLSingleLabel()
        self.model = nn.Linear(num_features, num_classes)
        self.learning_rate = 0.002
        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average="weighted")
        self.f1_score = MulticlassF1Score(num_classes=num_classes, average="weighted")
        self.precision = MulticlassPrecision(
            num_classes=num_classes, average="weighted"
        )
        self.recall = MulticlassRecall(num_classes=num_classes, average="weighted")

    def forward(self, x):
        return torch.log_softmax(self.model(x), dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _run_step(self, batch, batch_idx, step_name):
        x, y = batch["features"], batch["label"]
        logits = self(x)
        loss = self.loss(logits, y)
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        self.log(
            f"{step_name}_accuracy",
            self.accuracy(logits, y),
            on_step=False,
            on_epoch=True,
        )
        if step_name != "train":
            self.log(
                f"{step_name}_f1",
                self.f1_score(logits, y),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step_name}_precision",
                self.precision(logits, y),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step_name}_recall",
                self.recall(logits, y),
                on_step=False,
                on_epoch=True,
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "test")


class TwoLayerClassifier(LinearClassifier):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        asl_loss: bool = False,
        hidden_layer_size: int = 768,
    ):
        super().__init__(num_features, num_classes, asl_loss)
        self.model = nn.Sequential(
            nn.Linear(num_features, hidden_layer_size),
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_size, num_classes),
        )
