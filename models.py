import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder


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
        return CategoricalEmbeddingModule(self.cardinalities, self.embedding_dim)


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
