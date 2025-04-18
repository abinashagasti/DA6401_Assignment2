import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

class LitCNN(pl.LightningModule):
    def __init__(self, 
                 num_filters=[32, 64, 128, 256, 512], 
                 kernel_size=[3,3,3,3,3], 
                 num_dense=[512], 
                 num_classes=10, 
                 activation=F.relu, 
                 use_batchnorm=True, 
                 use_dropout=True, 
                 dropout_prob=0.12, 
                 padding=1, 
                 img_size=224,
                 lr=1e-3,
                 weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.activation = activation
        self.pool = nn.MaxPool2d(2, 2)
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_channels = 3
        for i in range(len(num_filters)):
            if padding is None:
                padding = kernel_size[i] // 2
            self.convs.append(nn.Conv2d(in_channels, num_filters[i], kernel_size[i], padding=padding))
            if self.use_batchnorm:
                self.bns.append(nn.BatchNorm2d(num_filters[i]))
            in_channels = num_filters[i]

        self.flattened_size = self._get_flattened_size((3, img_size, img_size))

        self.fcs = nn.ModuleList()
        input_size = self.flattened_size
        for output_size in num_dense:
            self.fcs.append(nn.Linear(input_size, output_size))
            input_size = output_size
        self.fcs.append(nn.Linear(input_size, num_classes))

        self.dropout = nn.Dropout(dropout_prob) if self.use_dropout else nn.Identity()
        self._initialize_weights()
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def _get_flattened_size(self, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            for i, conv in enumerate(self.convs):
                x = self.pool(self.activation(conv(x)))
            return x.numel()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            if self.use_batchnorm:
                x = self.bns[i](x)
            x = self.pool(self.activation(x))
        x = x.view(x.size(0), -1)
        for i in range(len(self.fcs) - 1):
            x = self.activation(self.fcs[i](x))
            x = self.dropout(x)
        return self.fcs[-1](x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        acc = self.train_acc(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        acc = self.val_acc(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        acc = self.test_acc(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
