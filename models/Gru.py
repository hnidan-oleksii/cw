import os

import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.parent = nn.Module.__class__
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input (batch_size, seq_len, input_size).
            hidden (torch.Tensor): Hidden state (num_layers, batch_size, hidden_size).

        Returns:
            torch.Tensor: Output (batch_size, seq_len, output_size).
            torch.Tensor: Final hidden state (num_layers, batch_size, hidden_size).
        """

        out, hidden = self.gru(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        save_path: str,
        device: torch.device,
        writer: SummaryWriter,
    ) -> tuple[np.array]:
        """
        Train a GRU model.

        Args:
            train_loader (DataLoader): Training GRU dataset loader.
            val_loader (DataLoader): Validation GRU dataset loader.
            input_size (int): Number of input features per timestep.
            hidden_size (int): Hidden state size of the GRU.
            output_size (int): Output size of the GRU.
            hidden_layers (int): Number of GRU layers.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            save_path (str): Directory to save trained model.
            device (torch.device): PyTorch device.
            writer (SummaryWriter): Tensorboard writer.

        Returns:
            np.array: Training predictions.
            np.array: Training targets.
            np.array: Validation predictions.
            np.array: Validation targets.
        """

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.L1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=4, threshold=1e-2, cooldown=2
        )

        for epoch in range(epochs):
            self.parent.train()
            train_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                hidden = self.init_hidden(x_batch.size(0)).to(device)
                output, _ = self(x_batch, hidden)
                output = output[:, -1, 0]

                loss = criterion(output, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.eval()
            val_loss = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    hidden = self.init_hidden(x_batch.size(0)).to(device)
                    output, _ = self(x_batch, hidden)
                    output = output[:, -1, 0]
                    val_loss += criterion(output, y_batch).item()
            scheduler.step(val_loss)

            writer.add_scalars(
                "Loss/GRU",
                {
                    "train": train_loss / len(train_loader),
                    "val": val_loss / len(val_loader),
                },
                epoch,
            )
        torch.save(self.state_dict(), os.path.join(save_path, "gru_model.pt"))

        train_preds, train_targets = self.get_predictions(train_loader, device)
        val_preds, val_targets = self.get_predictions(val_loader, device)
        return train_preds, train_targets, val_preds, val_targets

    def predict(self, data_loader: DataLoader, device: torch.device) -> tuple[np.array]:
        """
        Generates GRU predictions during inference.

        Args:
            model (GRUModel): Trained GRU model.
            data_loader (DataLoader): GRU DataLoader.

        Returns:
            np.array: Predictions.
            np.array: Targets.
        """
        preds, targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device)
                hidden = self.init_hidden(x_batch.size(0)).to(device)
                output, _ = self(x_batch, hidden)
                preds.append(output[:, -1, 0].cpu().numpy())
                targets.append(y_batch.numpy())

        return np.concatenate(preds), np.concatenate(targets)
