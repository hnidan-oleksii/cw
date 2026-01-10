import joblib
import os

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from env import Features


class LGBMModel(LGBMRegressor):
    def __init__(self, config: dict):
        LGBMRegressor.__init__(config)
        self.parent = LGBMRegressor.__class__

    @staticmethod
    def get_lgbm_input(weekly_df, latent, cat_df):
        return np.hstack(
            [
                weekly_df.drop(
                    columns=[Features.DATETIME_COL, *Features.GROUPBY_COLS]
                ).values,
                latent,
                cat_df.drop(
                    columns=[Features.DATETIME_COL, *Features.GROUPBY_COLS]
                ).values,
            ]
        )

    def train(
        self,
        train_weekly_df: pd.DataFrame,
        train_latent: pd.DataFrame,
        train_cat_df: pd.DataFrame,
        train_num_df: pd.DataFrame,
        val_weekly_df: pd.DataFrame,
        val_latent: np.array,
        val_cat_df: pd.DataFrame,
        val_num_df: pd.DataFrame,
        target_col: str,
        save_path: str,
        writer: SummaryWriter,
    ) -> tuple[np.array]:
        """
        Train a LightGBM regressor on engineered weekly features,
        autoencoder latent features, and categorical embeddings.

        Args:
            train_weekly_df (pd.DataFrame): Weekly aggregated training features.
            train_latent (np.array): Training latent representations.
            train_cat_df (pd.DataFrame): Training categorical features.
            train_num_df (pd.DataFrame): Training numerical dataframe.
            val_weekly_df (pd.DataFrame): Weekly aggregated validation features.
            val_latent (np.array): Validation latent representations.
            val_cat_df (pd.DataFrame): Validation categorical features.
            val_num_df (pd.DataFrame): Validation numerical dataframe.
            target_col (str): Target column name.
            lgbm_params (dict): LightGBM hyperparameters.
            save_path (str): Directory to save trained model.

        Returns:
            np.array: Training predictions.
            np.array: Validation predictions.
        """

        train_lgbm_input = LGBMModel.get_lgbm_input(
            train_weekly_df, train_latent, train_cat_df
        )
        val_lgbm_input = LGBMModel.get_lgbm_input(val_weekly_df, val_latent, val_cat_df)
        train_lgbm_target = train_num_df[target_col].values
        val_lgbm_target = val_num_df[target_col].values

        self.fit(train_lgbm_input, train_lgbm_target)
        joblib.dump(self, os.path.join(save_path, "lgbm_model.pkl"))
        train_lgbm_preds = self.parent.predict(train_lgbm_input)
        val_lgbm_preds = self.parent.predict(val_lgbm_input)

        train_loss = np.mean(np.abs(train_lgbm_preds - train_lgbm_target))
        val_loss = np.mean(np.abs(val_lgbm_preds - val_lgbm_target))
        writer.add_scalars(
            "Loss/LGBM",
            {
                "train": train_loss,
                "val": val_loss,
            },
        )

        return train_lgbm_preds, val_lgbm_preds

    def predict(self, weekly_df, latent, cat_df):
        lgbm_input = np.hstack(
            [
                weekly_df.drop(
                    columns=[Features.DATETIME_COL, *Features.GROUPBY_COLS]
                ).values,
                latent,
                cat_df.drop(
                    columns=[Features.DATETIME_COL, *Features.GROUPBY_COLS]
                ).values,
            ]
        )
        return self.parent.predict(lgbm_input)
