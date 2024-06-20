import pytorch_lightning as pl
import torch

from utils.metrics import get_accuracy, get_f1, get_auc
from utils.training_utils import get_optimizer, get_loss


class LitClassifier(pl.LightningModule):
    def __init__(self, model, config, **kwargs):
        super().__init__()
        self.loss = None
        self.optim = None
        self.model = model
        self.config = config

    def forward(self, x):
        x = self.model(x)
        return x

    def default_step(self, x, y, stage):
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_dict(
            {f"{stage}_loss": loss,
             f"{stage}_acc": self.get_accuracy(y_hat, y),
             f"{stage}_auc": self.get_auc(y_hat, y),
             f"{stage}_f1": self.get_f1(y_hat, y),
             },
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.default_step(x, y, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.default_step(x, y, stage="valid")

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.default_step(x, y, stage="test")

    def set_optimizer(self, optim_name, params, lr):
        self.optim = get_optimizer(optim_name, params, lr)

    def set_loss(self, loss_name, weights=None):
        self.loss = get_loss(loss_name, weights)

    def configure_optimizers(self, milestones=None):
        if milestones is None:
            milestones = [50, 75]
        sche = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optim, milestones=milestones, gamma=0.1)
        scheduler = {
            "scheduler": sche,
            "name": "lr_history",
        }

        return [self.optim], [scheduler]

    @staticmethod
    def get_accuracy(y_hat, y):
        return get_accuracy(y_hat, y)

    @staticmethod
    def get_f1(y_hat, y):
        return get_f1(y_hat, y)

    @staticmethod
    def get_auc(y_hat, y):
        return get_auc(y_hat, y)

    def state_dict(self, **kwargs):
        return self.model.state_dict()


class Vanilla(LitClassifier):
    def __init__(self, model, config):
        super().__init__(model, config)

    def default_step(self, x, y, stage):
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_dict(
            {f"{stage}_loss": loss,
             f"{stage}_acc": self.get_accuracy(y_hat, y),
             f"{stage}_auc": self.get_auc(y_hat, y),
             f"{stage}_f1": self.get_f1(y_hat, y),
             },
            prog_bar=True,
            sync_dist=True,
        )
        return loss
