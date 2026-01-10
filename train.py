import os
import argparse
import joblib

import torch
from torch.utils.tensorboard import SummaryWriter

from env import (
    Features,
    GruParams,
    AEParams,
    LGBM_PARAMS,
    MODEL_DIR,
    LOG_DIR,
    PT_DATA_PATH,
    DataSourceType,
)
from models import Autoencoder, GRUModel, LGBMModel, MetaLearner
from utils.data import get_all_data


def main(args: argparse.Namespace) -> None:
    model_dir = args.models
    rerun_gru_datasets = args.rerun_gru_datasets

    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    writer = SummaryWriter(args.logs)

    train_num_df, train_cat_df, train_weekly_df, train_gru_loader, scaler, embedder = (
        get_all_data(
            dt_col=Features.DATETIME_COL,
            categorical_cols=Features.CATEGORICAL_FEATURES,
            numerical_cols=Features.NUMERICAL_FEATURES,
            groupby_cols=Features.GROUPBY_COLS,
            target_col=Features.TARGET_COL,
            gru_dataset_path=os.path.join(
                PT_DATA_PATH, f"{DataSourceType.TRAIN}_dataset{args.suffix}.pt"
            ),
            rerun_gru_dataset=rerun_gru_datasets,
        )
    )

    val_num_df, val_cat_df, val_weekly_df, val_gru_loader, _, _ = get_all_data(
        dt_col=Features.DATETIME_COL,
        categorical_cols=Features.CATEGORICAL_FEATURES,
        numerical_cols=Features.NUMERICAL_FEATURES,
        groupby_cols=Features.GROUPBY_COLS,
        target_col=Features.TARGET_COL,
        gru_dataset_path=os.path.join(
            PT_DATA_PATH, f"{DataSourceType.VALIDATION}_dataset{args.suffix}.pt"
        ),
        scaler=scaler,
        embedder=embedder,
        df_type=DataSourceType.VALIDATION,
        rerun_gru_dataset=rerun_gru_datasets,
    )

    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(embedder, os.path.join(model_dir, "embedder.pkl"))

    if args.no_gru:
        gru_model = GRUModel(
            input_size=GruParams.GRU_INPUT_SIZE,
            hidden_size=GruParams.HIDDEN_SIZE,
            output_size=GruParams.GRU_OUTPUT_SIZE,
            num_layers=GruParams.LAYERS,
        ).to(device)
        gru_model.load_state_dict(torch.load(os.path.join(model_dir, "gru_model.pt")))
        gru_model.eval()
        train_gru_preds, train_gru_targets = gru_model.predict(train_gru_loader, device)
        val_gru_preds, val_gru_targets = gru_model.predict(val_gru_loader, device)
    else:
        gru_model = GRUModel(
            input_size=GruParams.GRU_INPUT_SIZE,
            hidden_size=args.gru_hidden_size,
            output_size=GruParams.GRU_OUTPUT_SIZE,
            num_layers=args.gru_layers,
        )
        train_gru_preds, train_gru_targets, val_gru_preds, val_gru_targets = (
            gru_model.train(
                train_loader=train_gru_loader,
                val_loader=val_gru_loader,
                epochs=args.gru_epochs,
                lr=args.gru_lr,
                save_path=model_dir,
                device=device,
            )
        )

    if args.no_ae:
        autoencoder = Autoencoder(
            input_dim=len(Features.NUMERICAL_FEATURES),
            latent_dim=args.latent_dim,
        ).to(device)
        autoencoder.load_state_dict(
            torch.load(os.path.join(model_dir, "autoencoder.pt"))
        )
        autoencoder.eval()
        train_ae_latents = autoencoder.get_latent_representation(train_num_df)
        val_ae_latents = autoencoder.get_latent_representation(val_num_df)
    else:
        autoencoder = Autoencoder(
            input_dim=len(Features.NUMERICAL_FEATURES),
            latent_dim=args.latent_dim,
        )
        train_latent, val_latent = autoencoder.train(
            train_df=train_num_df,
            val_df=val_num_df,
            numerical_features=Features.NUMERICAL_FEATURES,
            epochs=args.ae_epochs,
            lr=args.ae_lr,
            save_path=model_dir,
            device=device,
        )

    if args.no_lgbm:
        lgbm_model = joblib.load(os.path.join(model_dir, "lgbm_model.pkl"))
        train_lgbm_preds = lgbm_model.predict(train_gru_preds, train_ae_latents)
        val_lgbm_preds = lgbm_model.predict(val_gru_preds, val_ae_latents)
    else:
        lgbm_model = LGBMModel(LGBM_PARAMS)
        train_lgbm_preds, val_lgbm_preds = lgbm_model.train(
            train_weekly_df=train_weekly_df,
            train_latent=train_latent,
            train_cat_df=train_cat_df,
            train_num_df=train_num_df,
            val_weekly_df=val_weekly_df,
            val_latent=val_latent,
            val_cat_df=val_cat_df,
            val_num_df=val_num_df,
            target_col=Features.TARGET_COL,
            save_path=model_dir,
        )

    if not args.no_meta:
        meta_learner = MetaLearner()
        meta_learner.train(
            train_gru_preds=train_gru_preds,
            train_gru_targets=train_gru_targets,
            train_lgbm_preds=train_lgbm_preds,
            val_gru_preds=val_gru_preds,
            val_gru_targets=val_gru_targets,
            val_lgbm_preds=val_lgbm_preds,
            save_path=model_dir,
        )

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train model pipeline")

    parser.add_argument("--gru_lr", type=float, default=GruParams.GRU_LR)
    parser.add_argument("--gru_epochs", type=int, default=GruParams.GRU_EPOCHS)
    parser.add_argument(
        "--gru_hidden-size", type=int, default=GruParams.GRU_HIDDEN_SIZE
    )
    parser.add_argument("--gru_layers", type=int, default=GruParams.GRU_LAYERS)
    parser.add_argument("--no_gru", action="store_true", help="Skip GRU model training")

    parser.add_argument("--ae_lr", type=float, default=AEParams.AE_LR)
    parser.add_argument("--ae_epochs", type=int, default=AEParams.AE_EPOCHS)
    parser.add_argument("--latent_dim", type=int, default=AEParams.LATENT_DIM)
    parser.add_argument(
        "--no_ae", action="store_true", help="Skip Autoencoder training"
    )

    parser.add_argument(
        "--no_lgbm", action="store_true", help="Skip LGBM model training"
    )

    parser.add_argument(
        "--no_meta", action="store_true", help="Skip meta-learner model training"
    )

    parser.add_argument(
        "--models", type=str, default=MODEL_DIR, help="Path to the dir with models"
    )
    parser.add_argument(
        "--logs", type=str, default=LOG_DIR, help="Path to the dir with logs"
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
    args = parse_args()
    main(args)
