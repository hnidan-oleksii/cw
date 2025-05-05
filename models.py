import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=num_layers, dropout=0.2,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
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

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
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


class CategoricalEmbeddingEncoder:
    def __init__(self, categorical_cols, embed_dim):
        self.categorical_cols = categorical_cols
        self.label_encoders = {}
        self.cardinalities = []
        self.embedding_dim = embed_dim
        self.fitted = False

    def fit(self, df):
        df[self.categorical_cols] = df[self.categorical_cols].astype("object")
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            le.fit(df[col])
            self.label_encoders[col] = le
            self.cardinalities.append(len(le.classes_))
        self.fitted = True

    def transform(self, df):
        if not self.fitted:
            raise ValueError("Call fit() before transform().")
        encoded = []
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            encoded.append(le.transform(df[col].astype(str)))
        return np.stack(encoded, axis=1)

    def get_embedding_layer(self):
        if not self.fitted:
            raise ValueError("Call fit() before get_embedding_layer().")
        return CategoricalEmbeddingModule(
            self.cardinalities, self.embedding_dim
        )


class CategoricalEmbeddingModule(nn.Module):
    def __init__(self, cardinalities, embedding_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embedding_dim) for card in cardinalities
        ])
        self.output_dim = embedding_dim * len(cardinalities)

    def forward(self, x_cat):
        x_cat = x_cat.long()
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(embedded, dim=1)
