import os
import joblib
import argparse

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

from data_utils import get_all_data
from models import GRUModel, Autoencoder
from env import (
    DATETIME_COL, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, GROUPBY_COLS,
    TARGET_COL, MODEL_DIR, GRU_INPUT_SIZE, GRU_HIDDEN_SIZE, GRU_OUTPUT_SIZE,
    GRU_LAYERS, LATENT_DIM, PT_DATA_PATH
)


def get_models(device):
    gru_model = GRUModel(
        GRU_INPUT_SIZE, GRU_HIDDEN_SIZE, GRU_OUTPUT_SIZE, GRU_LAYERS
    ).to(device)
    gru_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gru_model.pt")))
    gru_model.eval()

    autoencoder = Autoencoder(len(NUMERICAL_FEATURES), LATENT_DIM).to(device)
    autoencoder.load_state_dict(torch.load(os.path.join(MODEL_DIR, "autoencoder.pt")))
    autoencoder.eval()

    lgbm_model = joblib.load(os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    meta_learner = joblib.load(os.path.join(MODEL_DIR, "meta_learner.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    embedder = joblib.load(os.path.join(MODEL_DIR, "embedder.pkl"))

    return gru_model, autoencoder, lgbm_model, meta_learner, scaler, embedder


def predict(gru, autoencoder, lgbm, meta, scaler, embedder, df_type, suffix, rerun_gru_dataset):
    num_df, cat_df, weekly_df, gru_loader, _, _ = get_all_data(
        dt_col=DATETIME_COL,
        numerical_cols=NUMERICAL_FEATURES,
        categorical_cols=CATEGORICAL_FEATURES,
        groupby_cols=GROUPBY_COLS,
        target_col=TARGET_COL,
        gru_dataset_path=os.path.join(
            PT_DATA_PATH, f"train_dataset{suffix}.pt"
        ),
        scaler=scaler,
        embedder=embedder,
        df_type=df_type,
        rerun_gru_dataset=rerun_gru_dataset
    )

    gru_preds, gru_targets = get_gru_predictions(gru, gru_loader)
    latent = get_latent_representation(autoencoder, num_df)
    lgbm_preds = get_lgbm_predictions(lgbm, weekly_df, latent, cat_df)
    meta_preds = get_meta_predictions(meta, gru_preds, lgbm_preds)

    target_idx = num_df.columns.get_loc(TARGET_COL) - len(GROUPBY_COLS) - 1
    mean = scaler.mean_[target_idx]
    scale = scaler.scale_[target_idx]

    gru_preds = unscale(mean, scale, gru_preds)
    gru_targets = unscale(mean, scale, gru_targets)
    lgbm_preds = unscale(mean, scale, lgbm_preds)
    meta_preds = unscale(mean, scale, meta_preds)

    results = create_results_df(
        num_df, gru_preds, lgbm_preds, meta_preds, gru_targets
    )
    metrics = calculate_metrics(results)

    return results, metrics


def get_gru_predictions(model, data_loader):
    preds, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            hidden = model.init_hidden(x_batch.size(0)).to(device)
            output, _ = model(x_batch, hidden)
            preds.append(output[:, -1, 0].cpu().numpy())
            targets.append(y_batch.numpy())

    return np.concatenate(preds), np.concatenate(targets)


def get_latent_representation(autoencoder, num_df):
    with torch.no_grad():
        data_tensor = torch.tensor(
            num_df[NUMERICAL_FEATURES].values, dtype=torch.float32
        ).to(device)
        _, latent = autoencoder(data_tensor)

    return latent.cpu().detach().numpy()


def get_lgbm_predictions(lgbm, weekly_df, latent, cat_df):
    lgbm_input = np.hstack([
        weekly_df.drop(columns=[DATETIME_COL, *GROUPBY_COLS]).values,
        latent,
        cat_df.drop(columns=[DATETIME_COL, *GROUPBY_COLS]).values
    ])

    return lgbm.predict(lgbm_input)


def get_meta_predictions(meta, gru_preds, lgbm_preds):
    common_len = min(len(gru_preds), len(lgbm_preds))
    meta_input = np.vstack([
        gru_preds[:common_len], lgbm_preds[:common_len]
    ]).T

    return meta.predict(meta_input)


def create_results_df(num_df, gru_preds, lgbm_preds, meta_preds, targets):
    common_len = min(len(gru_preds), len(lgbm_preds), len(meta_preds))

    df_subset = num_df.iloc[:common_len].copy()

    df_subset["gru_prediction"] = gru_preds[:common_len]
    df_subset["lgbm_prediction"] = lgbm_preds[:common_len]
    df_subset["ensemble_prediction"] = meta_preds[:common_len]

    df_subset["actual"] = targets[:common_len]

    return df_subset


def calculate_metrics(results_df):
    metrics = {}

    for model in ["gru", "lgbm", "ensemble"]:
        pred_col = f"{model}_prediction"
        metrics[f"{model}_mae"] = np.mean(
            np.abs(results_df[pred_col] - results_df["actual"])
        )

    return metrics


def unscale(mean, scale, arr):
    return mean + arr * scale


def main(gru, autoencoder, lgbm, meta, scaler, embedder, suffix, rerun_gru_dataset):
    results, metrics = predict(
        gru, autoencoder, lgbm, meta, scaler, embedder, suffix, rerun_gru_dataset
    )

    print("Performance on test set:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    csv_results_path = f"test_predictions{suffix}.csv"
    results.to_csv(csv_results_path, index=False)
    print(f"Predictions saved to {csv_results_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--models", type=str, default=MODEL_DIR)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--rerun_gru_datasets", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gru, autoencoder, lgbm, meta, scaler, embedder = get_models(device)
    args = parse_args()
    main(gru, autoencoder, lgbm, meta, scaler, embedder, args.suffix, args.rerun_gru_datasets)
