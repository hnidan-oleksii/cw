import os
import joblib

import numpy as np
from sklearn.linear_model import LinearRegression
from torch.utils.tensorboard import SummaryWriter


class MetaLearner(LinearRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = LinearRegression.__class__

    @staticmethod
    def get_meta_input(gru_preds: np.array, lgbm_preds: np.array) -> np.array:
        """
        Builds stacked ensemble input from GRU and LGBM predictions.

        Args:
            gru_preds (np.array): GRU predictions.
            lgbm_preds (np.array): LGBM predictions.

        Returns:
            np.array: Meta-model feature matrix.
        """

        common_len = min(len(gru_preds), len(lgbm_preds))
        return np.vstack([gru_preds[:common_len], lgbm_preds[:common_len]]).T

    def train(
        train_gru_preds: np.array,
        train_gru_targets: np.array,
        train_lgbm_preds: np.array,
        val_gru_preds: np.array,
        val_gru_targets: np.array,
        val_lgbm_preds: np.array,
        save_path: str,
        writer: SummaryWriter,
    ) -> None:
        """
        Trains a linear meta-learner for ensembling GRU and LGBM predictions.

        Args:
            train_gru_preds (np.array): GRU training predictions.
            train_gru_targets (np.array): GRU training targets.
            train_lgbm_preds (np.array): LGBM training predictions.
            val_gru_preds (np.array): GRU validation predictions.
            val_gru_targets (np.array): GRU validation targets.
            val_lgbm_preds (np.array): LGBM validation predictions.
            save_path (str): Directory to save meta-learner.

        Returns:
            None
        """

        meta_learner = LinearRegression()
        meta_learner.fit(
            MetaLearner.get_meta_input(train_gru_preds, train_lgbm_preds),
            train_gru_targets[: len(train_lgbm_preds)],
        )
        joblib.dump(meta_learner, os.path.join(save_path, "meta_learner.pkl"))

        train_meta_preds = meta_learner.predict(
            MetaLearner.get_meta_input(train_gru_preds, train_lgbm_preds)
        )
        val_meta_preds = meta_learner.predict(
            MetaLearner.get_meta_input(val_gru_preds, val_lgbm_preds)
        )

        train_loss = np.mean(
            np.abs(train_meta_preds - train_gru_targets[: len(train_meta_preds)])
        )
        val_loss = np.mean(
            np.abs(val_meta_preds - val_gru_targets[: len(val_meta_preds)])
        )
        writer.add_scalars(
            "Loss/Meta",
            {
                "train": train_loss,
                "val": val_loss,
            },
        )

    def predict(self, gru_preds, lgbm_preds):
        """
        Generate ensemble predictions.

        Args:
            meta: Trained meta-learner.
            gru_preds (np.array): GRU predictions.
            lgbm_preds (np.array): LGBM predictions.

        Returns:
            np.array: Ensemble predictions.
        """

        common_len = min(len(gru_preds), len(lgbm_preds))
        meta_input = np.vstack([gru_preds[:common_len], lgbm_preds[:common_len]]).T
        return self.parent.predict(meta_input)
