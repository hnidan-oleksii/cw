import os

import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from env import Features


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, input_dim)
        )

    def forward(self, x) -> tuple[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The reconstructed output tensor.
            torch.Tensor: The latent space representation.
        """

        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def get_latent_representation(
        self, num_df: pd.DataFrame, device: torch.device
    ) -> np.array:
        """
        Computes autoencoder latent representations.

        Args:
            autoencoder (Autoencoder): Trained autoencoder.
            num_df (pd.DataFrame): Numerical feature dataframe.

        Returns:
            np.array: Latent representations.
        """
        with torch.no_grad():
            data_tensor = torch.tensor(
                num_df[Features.NUMERICAL_FEATURES].values, dtype=torch.float32
            ).to(device)
            _, latent = self.__init__(data_tensor)

        return latent.cpu().detach().numpy()

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        numerical_features: list[str],
        epochs: int,
        lr: float,
        save_path: str,
        device: torch.device,
        writer: SummaryWriter,
    ) -> tuple[np.array]:
        """
        Train AutoEncoder.

        Args:
            train_df (pd.DataFrame): Full training dataset.
            val_df (pd.DataFrame): Full validation dataset.
            numerical_features (list[str]): List of numerical features.
            latent_dim (int): Number of latent dimensions.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            save_path (str): Path at which to save the model.
            device (torch.device): PyTorch device.

        Returns:
            np.array: Train dataset representation from latent dimensions.
            np.array: Validation dataset representation from latent dimensions.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        train_tensor = torch.tensor(
            train_df[numerical_features].values, dtype=torch.float32
        )
        val_tensor = torch.tensor(
            val_df[numerical_features].values, dtype=torch.float32
        ).to(device)
        train_loader = DataLoader(
            train_tensor, batch_size=1024, shuffle=True, num_workers=8, pin_memory=True
        )

        for epoch in range(epochs):
            nn.Module.train()
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                reconstructed, _ = self(batch)
                loss = criterion(reconstructed, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.eval()
            with torch.no_grad():
                val_reconstructed, _ = self(val_tensor)
                val_loss = criterion(val_reconstructed, val_tensor)

            writer.add_scalars(
                "Loss/Autoencoder",
                {
                    "train": loss.item(),
                    "val": val_loss.item(),
                },
                epoch,
            )
            scheduler.step(val_loss.item())
        torch.save(self.state_dict(), os.path.join(save_path, "autoencoder.pt"))

        self.eval()
        _, train_latent = self(train_tensor.to(device))
        _, val_latent = self(val_tensor)
        train_latent = train_latent.cpu().detach().numpy()
        val_latent = val_latent.cpu().detach().numpy()
        return train_latent, val_latent
