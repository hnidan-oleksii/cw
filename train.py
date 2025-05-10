import os
import argparse
import joblib

import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

from data_utils import get_all_data, GRUDataset
from models import GRUModel, Autoencoder
from infere import get_gru_predictions, get_latent_representation, get_lgbm_predictions
from env import (
    DATETIME_COL, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, GROUPBY_COLS,
    TARGET_COL, MODEL_DIR, LOG_DIR, GRU_EPOCHS, GRU_LR, AE_EPOCHS, AE_LR,
    LGBM_PARAMS, LATENT_DIM, GRU_INPUT_SIZE, GRU_HIDDEN_SIZE, GRU_OUTPUT_SIZE,
    GRU_LAYERS, PT_DATA_PATH
)


def train_gru(
    train_loader,
    val_loader,
    input_size,
    hidden_size,
    output_size,
    hidden_layers,
    epochs,
    lr,
    save_path,
    device
):
    gru_model = GRUModel(
        input_size, hidden_size, output_size, num_layers=hidden_layers
    ).to(device)
    optimizer = optim.Adam(gru_model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, threshold=1e-2, cooldown=2
    )

    for epoch in range(epochs):
        gru_model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            hidden = gru_model.init_hidden(x_batch.size(0)).to(device)
            output, _ = gru_model(x_batch, hidden)
            output = output[:, -1, 0]

            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        gru_model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                hidden = gru_model.init_hidden(x_batch.size(0)).to(device)
                output, _ = gru_model(x_batch, hidden)
                output = output[:, -1, 0]
                val_loss += criterion(output, y_batch).item()
        scheduler.step(val_loss)

        writer.add_scalars("Loss/GRU", {
            "train": train_loss / len(train_loader),
            "val": val_loss / len(val_loader),
        }, epoch)
    torch.save(gru_model.state_dict(), os.path.join(save_path, "gru_model.pt"))

    train_preds, train_targets = get_gru_preds(
        gru_model, train_loader, device
    )
    val_preds, val_targets = get_gru_preds(
        gru_model, val_gru_loader, device
    )
    return train_preds, train_targets, val_preds, val_targets


def get_gru_preds(model, data_loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            hidden = model.init_hidden(x_batch.size(0)).to(device)
            output, _ = model(x_batch, hidden)
            preds.append(output[:, -1, 0].cpu().numpy())
            targets.append(y_batch.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def train_autoencoder(
    train_df,
    val_df,
    numerical_features,
    latent_dim,
    epochs,
    lr,
    save_path,
    device
):
    autoencoder = Autoencoder(len(numerical_features), latent_dim).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
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
        train_tensor, batch_size=1024, shuffle=True, num_workers=8,
        pin_memory=True
    )

    for epoch in range(epochs):
        autoencoder.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            reconstructed, _ = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        autoencoder.eval()
        with torch.no_grad():
            val_reconstructed, _ = autoencoder(val_tensor)
            val_loss = criterion(val_reconstructed, val_tensor)

        writer.add_scalars("Loss/Autoencoder", {
            "train": loss.item(),
            "val": val_loss.item(),
        }, epoch)
        scheduler.step(val_loss.item())
    torch.save(
        autoencoder.state_dict(), os.path.join(save_path, "autoencoder.pt")
    )

    autoencoder.eval()
    _, train_latent = autoencoder(train_tensor.to(device))
    _, val_latent = autoencoder(val_tensor)
    train_latent = train_latent.cpu().detach().numpy()
    val_latent = val_latent.cpu().detach().numpy()
    return train_latent, val_latent


def get_lgbm_input(weekly_df, latent, cat_df):
    return np.hstack([
        weekly_df.drop(columns=[DATETIME_COL, *GROUPBY_COLS]).values,
        latent,
        cat_df.drop(columns=[DATETIME_COL, *GROUPBY_COLS]).values
    ])


def train_lgbm(
    train_weekly_df,
    train_latent,
    train_cat_df,
    train_num_df,
    val_weekly_df,
    val_latent,
    val_cat_df,
    val_num_df,
    target_col,
    lgbm_params,
    save_path
):
    train_lgbm_input = get_lgbm_input(train_weekly_df, train_latent, train_cat_df)
    val_lgbm_input = get_lgbm_input(val_weekly_df, val_latent, val_cat_df)
    train_lgbm_target = train_num_df[target_col].values
    val_lgbm_target = val_num_df[target_col].values

    lgbm_model = LGBMRegressor(**lgbm_params)
    lgbm_model.fit(train_lgbm_input, train_lgbm_target)
    joblib.dump(lgbm_model, os.path.join(save_path, "lgbm_model.pkl"))
    train_lgbm_preds = lgbm_model.predict(train_lgbm_input)
    val_lgbm_preds = lgbm_model.predict(val_lgbm_input)

    train_loss = np.mean(np.abs(train_lgbm_preds - train_lgbm_target))
    val_loss = np.mean(np.abs(val_lgbm_preds - val_lgbm_target))
    writer.add_scalars("Loss/LGBM", {
        "train": train_loss,
        "val": val_loss,
    })

    return train_lgbm_preds, val_lgbm_preds


def get_meta_input(gru_preds, lgbm_preds):
    common_len = min(len(gru_preds), len(lgbm_preds))
    return np.vstack([gru_preds[:common_len], lgbm_preds[:common_len]]).T


def train_meta(
    train_gru_preds,
    train_gru_targets,
    train_lgbm_preds,
    val_gru_preds,
    val_gru_targets,
    val_lgbm_preds,
    save_path
):
    meta_learner = LinearRegression()
    meta_learner.fit(
        get_meta_input(train_gru_preds, train_lgbm_preds),
        train_gru_targets[:len(train_lgbm_preds)]
    )
    joblib.dump(meta_learner, os.path.join(model_dir, "meta_learner.pkl"))

    train_meta_preds = meta_learner.predict(
        get_meta_input(train_gru_preds, train_lgbm_preds)
    )
    val_meta_preds = meta_learner.predict(
        get_meta_input(val_gru_preds, val_lgbm_preds)
    )

    train_loss = np.mean(
        np.abs(train_meta_preds - train_gru_targets[:len(train_meta_preds)])
    )
    val_loss = np.mean(
        np.abs(val_meta_preds - val_gru_targets[:len(val_meta_preds)])
    )
    writer.add_scalars("Loss/Meta", {
        "train": train_loss,
        "val": val_loss,
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Train model pipeline")

    parser.add_argument("--gru_lr", type=float, default=GRU_LR)
    parser.add_argument("--gru_epochs", type=int, default=GRU_EPOCHS)
    parser.add_argument("--gru_hidden-size", type=int, default=GRU_HIDDEN_SIZE)
    parser.add_argument("--gru_layers", type=int, default=GRU_LAYERS)
    parser.add_argument("--no_gru", action="store_true")

    parser.add_argument("--ae_lr", type=float, default=AE_LR)
    parser.add_argument("--ae_epochs", type=int, default=AE_EPOCHS)
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM)
    parser.add_argument("--no_ae", action="store_true")

    parser.add_argument("--no_lgbm", action="store_true")

    parser.add_argument("--no_meta", action="store_true")

    parser.add_argument("--models", type=str, default=MODEL_DIR)
    parser.add_argument("--logs", type=str, default=LOG_DIR)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--rerun_gru_datasets", action="store_false")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_dir = args.models
    rerun_gru_datasets = args.rerun_gru_datasets

    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    writer = SummaryWriter(args.logs)

    train_num_df, train_cat_df, train_weekly_df, train_gru_loader, scaler, embedder = get_all_data(
        dt_col=DATETIME_COL,
        categorical_cols=CATEGORICAL_FEATURES,
        numerical_cols=NUMERICAL_FEATURES,
        groupby_cols=GROUPBY_COLS,
        target_col=TARGET_COL,
        gru_dataset_path=os.path.join(
            PT_DATA_PATH, f"train_dataset{args.suffix}.pt"
        ),
        rerun_gru_dataset=rerun_gru_datasets,
    )

    val_num_df, val_cat_df, val_weekly_df, val_gru_loader, _, _ = get_all_data(
        dt_col=DATETIME_COL,
        categorical_cols=CATEGORICAL_FEATURES,
        numerical_cols=NUMERICAL_FEATURES,
        groupby_cols=GROUPBY_COLS,
        target_col=TARGET_COL,
        gru_dataset_path=os.path.join(
            PT_DATA_PATH, f"val_dataset{args.suffix}.pt"
        ),
        scaler=scaler,
        embedder=embedder,
        df_type="val",
        rerun_gru_dataset=rerun_gru_datasets
    )

    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(embedder, os.path.join(model_dir, "embedder.pkl"))

    if args.no_gru:
        gru_model = GRUModel(
            GRU_INPUT_SIZE, args.gru_hidden_size, GRU_OUTPUT_SIZE, args.gru_layers
        ).to(device)
        gru_model.load_state_dict(
            torch.load(os.path.join(model_dir, "gru_model.pt"))
        )
        gru_model.eval()
        train_gru_preds, train_gru_targets = get_gru_predictions(
            gru_model, train_gru_loader, device
        )
        val_gru_preds, val_gru_targets = get_gru_predictions(
            gru_model, val_gru_loader, device
        )
    else:
        train_gru_preds, train_gru_targets, val_gru_preds, val_gru_targets = train_gru(
            train_loader=train_gru_loader,
            val_loader=val_gru_loader,
            input_size=GRU_INPUT_SIZE,
            hidden_size=args.gru_hidden_size,
            output_size=GRU_OUTPUT_SIZE,
            hidden_layers=args.gru_layers,
            epochs=args.gru_epochs,
            lr=args.gru_lr,
            save_path=model_dir,
            device=device
        )

    if args.no_ae:
        autoencoder = Autoencoder(
            len(NUMERICAL_FEATURES), args.latent_dim
        ).to(device)
        autoencoder.load_state_dict(
            torch.load(os.path.join(model_dir, "autoencoder.pt"))
        )
        autoencoder.eval()
        train_ae_latents = get_latent_representation(autoencoder, train_num_df)
        val_ae_latents = get_latent_representation(autoencoder, val_num_df)
    else:
        train_latent, val_latent = train_autoencoder(
            train_df=train_num_df,
            val_df=val_num_df,
            numerical_features=NUMERICAL_FEATURES,
            latent_dim=args.latent_dim,
            epochs=args.ae_epochs,
            lr=args.ae_lr,
            save_path=model_dir,
            device=device
        )

    if args.no_lgbm:
        lgbm_model = joblib.load(os.path.join(model_dir, "lgbm_model.pkl"))
        train_lgbm_preds = get_lgbm_predictions(
            lgbm_model, train_gru_preds, train_ae_latents
        )
        val_lgbm_preds = get_lgbm_predictions(
            lgbm_model, val_gru_preds, val_ae_latents
        )
    else:
        train_lgbm_preds, val_lgbm_preds = train_lgbm(
            train_weekly_df=train_weekly_df,
            train_latent=train_latent,
            train_cat_df=train_cat_df,
            train_num_df=train_num_df,
            val_weekly_df=val_weekly_df,
            val_latent=val_latent,
            val_cat_df=val_cat_df,
            val_num_df=val_num_df,
            target_col=TARGET_COL,
            lgbm_params=LGBM_PARAMS,
            save_path=model_dir
        )

    if not args.no_meta:
        train_meta(
            train_gru_preds=train_gru_preds,
            train_gru_targets=train_gru_targets,
            train_lgbm_preds=train_lgbm_preds,
            val_gru_preds=val_gru_preds,
            val_gru_targets=val_gru_targets,
            val_lgbm_preds=val_lgbm_preds,
            save_path=model_dir
        )

    writer.close()
