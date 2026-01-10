import os
import joblib
import argparse

from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import torch

from env import Features, GruParams, AEParams, PT_DATA_PATH, MODEL_DIR, DataSourceType
from models import (
    Autoencoder,
    CategoricalEmbeddingEncoder,
    GRUModel,
    LGBMModel,
    MetaLearner,
)
from utils.data import get_all_data, create_results_df
from utils.metrics import calculate_metrics
from utils.unscale import unscale


def get_models(
    device: torch.device,
) -> (
    GRUModel,
    Autoencoder,
    LGBMRegressor,
    LinearRegression,
    StandardScaler,
    CategoricalEmbeddingEncoder,
):
    """
    Load all trained models and preprocessing artifacts.

    Args:
        device (torch.device): PyTorch device.

    Returns:
        GRUModel: Trained GRU model.
        Autoencoder: Trained autoencoder.
        LGBMRegressor: Trained LightGBM model.
        LinearRegression: Trained meta-learner.
        StandardScaler: Fitted scaler.
        CategoricalEmbeddingEncoder: Fitted categorical embedder.
    """

    gru_model = GRUModel(
        input_size=GruParams.GRU_INPUT_SIZE,
        hidden_size=GruParams.GRU_HIDDEN_SIZE,
        output_size=GruParams.GRU_OUTPUT_SIZE,
        num_layers=GruParams.GRU_LAYERS,
    ).to(device)
    gru_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gru_model.pt")))
    gru_model.eval()

    autoencoder = Autoencoder(len(Features.NUMERICAL_FEATURES), AEParams.LATENT_DIM).to(
        device
    )
    autoencoder.load_state_dict(torch.load(os.path.join(MODEL_DIR, "autoencoder.pt")))
    autoencoder.eval()

    lgbm_model = joblib.load(os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    meta_learner = joblib.load(os.path.join(MODEL_DIR, "meta_learner.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    embedder = joblib.load(os.path.join(MODEL_DIR, "embedder.pkl"))

    return gru_model, autoencoder, lgbm_model, meta_learner, scaler, embedder


def predict(
    gru: GRUModel,
    autoencoder: Autoencoder,
    lgbm: LGBMModel,
    meta: MetaLearner,
    scaler: StandardScaler,
    embedder: CategoricalEmbeddingEncoder,
    df_type: DataSourceType,
    suffix: str,
    rerun_gru_dataset: bool,
):
    """
    Run full inference pipeline and computes evaluation metrics.

    Args:
        gru (GRUModel): Trained GRU model.
        autoencoder (Autoencoder): Trained autoencoder.
        lgbm: Trained LightGBM model.
        meta: Trained meta-learner.
        scaler (StandardScaler): Fitted scaler.
        embedder (CategoricalEmbeddingEncoder): Fitted embedder.
        df_type (DataSourceType): Dataset split to use.
        suffix (str): Dataset suffix.
        rerun_gru_dataset (bool): Whether to rebuild GRU dataset.

    Returns:
        pd.DataFrame: Prediction results.
        dict: Evaluation metrics.
    """

    num_df, cat_df, weekly_df, gru_loader, _, _ = get_all_data(
        dt_col=Features.DATETIME_COL,
        numerical_cols=Features.NUMERICAL_FEATURES,
        categorical_cols=Features.CATEGORICAL_FEATURES,
        groupby_cols=Features.GROUPBY_COLS,
        target_col=Features.TARGET_COL,
        gru_dataset_path=os.path.join(PT_DATA_PATH, f"train_dataset{suffix}.pt"),
        scaler=scaler,
        embedder=embedder,
        df_type=df_type,
        rerun_gru_dataset=rerun_gru_dataset,
    )

    gru_preds, gru_targets = gru.get_predictions(gru_loader)
    latent = autoencoder.get_latent_representation(autoencoder, num_df)
    lgbm_preds = lgbm.predict(weekly_df, latent, cat_df)
    meta_preds = meta.predict(gru_preds, lgbm_preds)

    target_idx = (
        num_df.columns.get_loc(Features.TARGET_COL) - len(Features.GROUPBY_COLS) - 1
    )
    mean = scaler.mean_[target_idx]
    scale = scaler.scale_[target_idx]

    gru_preds = unscale(mean, scale, gru_preds)
    gru_targets = unscale(mean, scale, gru_targets)
    lgbm_preds = unscale(mean, scale, lgbm_preds)
    meta_preds = unscale(mean, scale, meta_preds)

    results = create_results_df(num_df, gru_preds, lgbm_preds, meta_preds, gru_targets)
    metrics = calculate_metrics(results)

    return results, metrics


def main(
    gru: GRUModel,
    autoencoder: Autoencoder,
    lgbm: LGBMModel,
    meta: MetaLearner,
    scaler: StandardScaler,
    embedder: CategoricalEmbeddingEncoder,
    suffix: str,
    rerun_gru_dataset: bool,
):
    """
    Entry point for inference execution.

    Args:
        gru (GRUModel): Trained GRU.
        autoencoder (Autoencoder): Trained autoencoder.
        lgbm: Trained LightGBM.
        meta: Trained meta-learner.
        scaler (StandardScaler): Fitted scaler.
        embedder (CategoricalEmbeddingEncoder): Fitted embedder.
        suffix (str): Dataset suffix.
        rerun_gru_dataset (bool): Whether to rebuild GRU dataset.

    Returns:
        None
    """

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
    parser.add_argument(
        "--models", type=str, default=MODEL_DIR, help="Path to the dir with models"
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="Suffix for the model and dataset names"
    )
    parser.add_argument(
        "--rerun_gru_datasets",
        action="store_false",
        help="Binary, whether to recreate PyTorch datasets for GRU model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gru, autoencoder, lgbm, meta, scaler, embedder = get_models(device)
    args = parse_args()
    main(
        gru,
        autoencoder,
        lgbm,
        meta,
        scaler,
        embedder,
        args.suffix,
        args.rerun_gru_datasets,
    )
