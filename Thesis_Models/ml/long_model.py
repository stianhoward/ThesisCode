"""
model.py

Contains the basic NN model
"""


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class Network(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(10,16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
        self.learning_rate = learning_rate

    def forward(self, x):
        predicts = self.linear_stack(x)
        return predicts

    def training_step(self, batch, batch_idx):
        # Look into tensorboard logging
        x, y = batch
        x_hat = self.linear_stack(x)
        loss = F.mse_loss(x_hat, y)
        self.log("training loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.linear_stack(x)
        loss = F.mse_loss(x_hat, y)
        self.log("Validation loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.linear_stack(x)
        loss = F.mse_loss(x_hat, y)
        self.log("Test loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



