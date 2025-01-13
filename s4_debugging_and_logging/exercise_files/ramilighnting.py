import lightning as l
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn, optim


class MyLightningModule(l.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x) -> torch.Tensor:
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc1(x)

    def training_step(self, batch, batch_idx):
        img, target = batch
        y_pred = self(img)
        return self.loss_fn(y_pred, target)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    net = MyLightningModule()
    print(f"Model architecture: {net}")
    print(f"Number of parameters: {sum(p.numel() for p in net.parameters())}")
    dummy_input = torch.randn(1, 1, 28, 28)
    output = net(dummy_input)
    print(f"Output shape: {output.shape}")

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")
    trainer = l.Trainer(max_epochs=10, max_steps=100, limit_train_batches=0.2, callbacks=[early_stopping, checkpoint])
    trainer.fit(net)
