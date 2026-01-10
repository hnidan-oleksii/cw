import pandas as pd
import torch
from torch.utils.data import Dataset


class GRUDataset(Dataset):
    """
    PyTorch Dataset for GRU time-series windows.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dt_col: str,
        numerical_cols: list[str],
        groupby_cols: list[str],
        target_col: str,
        valid_idx: pd.Index,
        window_size: str = 24,
    ):
        self.window_size = window_size
        self.features = []
        self.targets = []

        df = df.sort_values([*groupby_cols, dt_col])
        df = df.reset_index(drop=True)

        group_col_values = df[groupby_cols].astype(str).agg("_".join, axis=1)
        group_id_series = group_col_values.astype("category").cat.codes
        df["group_id"] = group_id_series

        group_indices = df.groupby("group_id").indices

        for group_id, indices in group_indices.items():
            indices = list(indices)
            if len(indices) < window_size:
                continue

            for i in range(len(indices) - window_size + 1):
                idx_range = indices[i : i + window_size]
                window = df.iloc[idx_range]

                if window.isna().any().any():
                    continue

                end_time = window[dt_col].iloc[-1]
                group_vals = df.loc[idx_range[0], groupby_cols].tolist()
                key = tuple([end_time] + group_vals)

                if key in valid_idx:
                    self.features.append(idx_range)
                    self.targets.append(indices[i + window_size - 1])

        self.X = df[numerical_cols].values
        self.y = df[target_col].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        idx_range = self.features[idx]
        target_idx = self.targets[idx]

        x = torch.tensor(self.X[idx_range], dtype=torch.float32)
        y = torch.tensor(self.y[target_idx], dtype=torch.float32)

        return x, y
