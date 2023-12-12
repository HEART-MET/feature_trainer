import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim import lr_scheduler

from feature_dataset import FeatureDataset

import pytorch_lightning as pl
import torchnet.meter as meter
import pdb

class FeatureTrainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = nn.Linear(1024, self.hparams.num_classes, bias=True)
        self.meter = meter.mAPMeter()

    def forward(self, batch):
        inputs, labels, vidx = batch
        per_frame_logits = self.model(inputs)
        predictions = torch.argmax(per_frame_logits, axis=1)
        return per_frame_logits, predictions

    def calculate_loss(self, batch, per_frame_logits):
        inputs, labels, vidx = batch
        loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        per_frame_logits, predictions = self(batch)
        loss = self.calculate_loss(batch, per_frame_logits)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        inputs, labels, vidx = batch
        per_frame_logits, predictions = self(batch)
        loss = self.calculate_loss(batch, per_frame_logits)
        self.meter.add(per_frame_logits, labels)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("train_loss", avg_loss)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        mAP = self.meter.value()
        self.log("val_mAP", mAP)
        self.log("val_loss", avg_loss)
        self.meter.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, weight_decay=0.0000001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = FeatureDataset(self.hparams.train_root, self.hparams.train_labels, num_classes=self.hparams.num_classes)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                           shuffle=True, num_workers=self.hparams.n_threads)
    def val_dataloader(self):
        train_dataset = FeatureDataset(self.hparams.validation_root, self.hparams.validation_labels, num_classes=self.hparams.num_classes)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.n_threads)
