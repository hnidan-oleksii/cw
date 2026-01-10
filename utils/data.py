import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from env import (
    TRAIN_PATH,
    CLIENT_PATH,
    ELECTRICITY_PRICES_PATH,
    FORECAST_WEATHER_PATH,
    HISTORICAL_WEATHER_PATH,
    WEATHER_STATION_MAPPING_PATH,
    Features,
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
    EMBED_DIM,
    DataSourceType,
)
from models import CategoricalEmbeddingEncoder
from utils.GruDataset import GRUDataset


def get_all_data(
    dt_col: str,
    numerical_cols: str,
    categorical_cols: str,
    groupby_cols: str,
    target_col: str,
    gru_dataset_path: str = None,
    rerun_gru_dataset: bool = False,
    scaler: StandardScaler | None = None,
    embedder: CategoricalEmbeddingEncoder | None = None,
    split_ratios: tuple[float] = (0.85, 0.10, 0.05),
    df_type: DataSourceType = DataSourceType.TRAIN,
) -> (tuple[pd.DataFrame], DataLoader, StandardScaler, CategoricalEmbeddingEncoder):
    """
    Load, preprocess, and prepare all datasets for training or inference.

    Args:
        dt_col (str): Datetime column name.
        numerical_cols (list[str]): Numerical feature columns.
        categorical_cols (list[str]): Categorical feature columns.
        groupby_cols (list[str]): Grouping columns.
        target_col (str): Target column.
        gru_dataset_path (str): Path to cached GRU dataset.
        rerun_gru_dataset (bool): Whether to regenerate GRU dataset.
        scaler (StandardScaler | None): Existing scaler.
        embedder (CategoricalEmbeddingEncoder | None): Existing embedder.
        split_ratios (tuple[float]): Train/val/test split ratios.
        df_type (DataSourceType): Dataset split.

    Returns:
        pd.DataFrame: Numerical dataframe.
        pd.DataFrame: Categorical dataframe.
        pd.DataFrame: Weekly aggregated dataframe.
        DataLoader: GRU DataLoader.
        StandardScaler: Fitted scaler.
        CategoricalEmbeddingEncoder: Fitted embedder.
    """

    df = read_and_preprocess_raw(dt_col=dt_col)

    if round(sum(split_ratios), 2) != 1:
        raise ValueError("Split ratios do sum to 1")

    block_ids_num = df["data_block_id"].nunique()
    train_id_cutoff = int(block_ids_num * split_ratios[0])
    val_id_cutoff = train_id_cutoff + int(block_ids_num * split_ratios[1])

    if df_type == DataSourceType.TRAIN:
        block_ids = list(range(0, train_id_cutoff))
    elif df_type == DataSourceType.VALIDATION:
        block_ids = list(range(train_id_cutoff, val_id_cutoff))
    elif df_type == DataSourceType.TEST:
        block_ids = list(range(val_id_cutoff, block_ids_num))
    else:
        raise ValueError(
            f"Incorrect df_type: {df_type}. Must be one of: train, validation, test."
        )

    df = df[df["data_block_id"].isin(block_ids)]

    num_df, scaler = process_numerical(
        df=df,
        dt_cols=[dt_col],
        numerical_cols=numerical_cols,
        merge_cols=groupby_cols,
        target_col=target_col,
        scaler=scaler,
    )
    cat_df, embedder = process_categorical(
        df=df,
        dt_cols=[dt_col],
        categorical_cols=categorical_cols,
        merge_cols=groupby_cols,
        embedder=embedder,
    )

    weekly_df = process_ts_weekly(
        df=num_df,
        dt_col=dt_col,
        numerical_cols=numerical_cols,
        groupby_cols=groupby_cols,
        target_col=target_col,
    )
    valid_idx = pd.MultiIndex.from_frame(
        weekly_df[[dt_col, *groupby_cols]]
    ).sortlevel()[0]

    if rerun_gru_dataset or not gru_dataset_path:
        gru_dataset = GRUDataset(
            df=num_df,
            dt_col=dt_col,
            numerical_cols=numerical_cols,
            groupby_cols=groupby_cols,
            target_col=target_col,
            valid_idx=valid_idx,
        )
    else:
        gru_dataset = torch.load(gru_dataset_path, weights_only=False)
    if rerun_gru_dataset and gru_dataset_path:
        torch.save(gru_dataset, gru_dataset_path)

    batch_size = TRAIN_BATCH_SIZE if df_type == DataSourceType.TRAIN else VAL_BATCH_SIZE
    gru_loader = DataLoader(
        gru_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    weekly_keys = weekly_df.set_index([dt_col, *groupby_cols]).index
    num_df = num_df[num_df.set_index([dt_col, *groupby_cols]).index.isin(weekly_keys)]
    cat_df = cat_df[cat_df.set_index([dt_col, *groupby_cols]).index.isin(weekly_keys)]

    return (num_df, cat_df, weekly_df), gru_loader, scaler, embedder


def read_and_preprocess_raw(dt_col: str) -> pd.DataFrame:
    """
    Read raw data sources and perform feature engineering.

    Args:
        dt_col (str): Datetime column name.

    Returns:
        pd.DataFrame: Fully preprocessed dataframe.
    """

    df = pd.read_csv(TRAIN_PATH, parse_dates=[dt_col])
    client_df = pd.read_csv(CLIENT_PATH, parse_dates=["date"])
    electricity_df = pd.read_csv(
        ELECTRICITY_PRICES_PATH, parse_dates=["origin_date", "forecast_date"]
    )
    fw_df = pd.read_csv(
        FORECAST_WEATHER_PATH, parse_dates=["origin_datetime", "forecast_datetime"]
    )
    hw_df = pd.read_csv(HISTORICAL_WEATHER_PATH, parse_dates=[dt_col])

    weather_station_to_county_mapping = pd.read_csv(WEATHER_STATION_MAPPING_PATH)

    client_df = client_df.drop(columns=["date"]).sort_values(
        by=["data_block_id"], ascending=True
    )
    client_merge_cols = [
        "county",
        "is_business",
        "product_type",
        "data_block_id",
        "eic_count",
        "installed_capacity",
    ]
    df = pd.merge(
        df,
        client_df[client_merge_cols],
        how="left",
        on=["county", "is_business", "product_type", "data_block_id"],
    )

    electricity_df = electricity_df.drop(columns=["origin_date"])
    electricity_df = electricity_df.rename(columns={"forecast_date": "datetime"})
    electricity_df["datetime"] = pd.to_datetime(
        electricity_df["datetime"]
    ) + pd.Timedelta(days=1)
    electricity_df["euros_per_mwh"] = electricity_df["euros_per_mwh"].abs() + 0.1
    df = pd.merge(
        df,
        electricity_df[["data_block_id", "datetime", "euros_per_mwh"]],
        how="left",
        on=["data_block_id", "datetime"],
    )

    fw_df["datetime"] = pd.to_datetime(fw_df["origin_datetime"]) + pd.to_timedelta(
        fw_df["hours_ahead"], unit="h"
    )
    fw_df["latitude"] = np.round(fw_df["latitude"].astype(np.float32), 1)
    fw_df["longitude"] = np.round(fw_df["longitude"].astype(np.float32), 1)
    fw_df["data_block_id"] = fw_df["data_block_id"].astype(np.int64)

    weather_station_to_county_mapping = weather_station_to_county_mapping.drop(
        columns=["county_name"]
    )
    weather_station_to_county_mapping["latitude"] = np.round(
        weather_station_to_county_mapping["latitude"].astype(np.float32), 1
    )
    weather_station_to_county_mapping["longitude"] = np.round(
        weather_station_to_county_mapping["longitude"].astype(np.float32), 1
    )

    fw_df = pd.merge(
        fw_df,
        weather_station_to_county_mapping,
        how="left",
        on=["longitude", "latitude"],
    )
    fw_df = fw_df.drop(columns=["longitude", "latitude", "origin_datetime"])

    fw_df = (
        fw_df.groupby(["county", "datetime", "data_block_id"])
        .agg({col: "mean" for col in Features.FORECAST_WEATHER_COLS})
        .reset_index()
    )

    for col in Features.FORECAST_WEATHER_COLS:
        fw_df = fw_df.rename(columns={col: f"fw_{col}"})

    fw_df["county"] = fw_df["county"].astype(np.int64)
    fw_df["data_block_id"] = fw_df["data_block_id"].astype(np.int64)

    df = pd.merge(df, fw_df, how="left", on=["county", "datetime", "data_block_id"])

    df["fw_new_feature"] = (
        df["installed_capacity"] * df["fw_surface_solar_radiation_downwards"]
    ) / (df["fw_temperature"] + 273.15)

    hw_df["datetime"] = pd.to_datetime(hw_df["datetime"]) + pd.Timedelta(days=1)
    hw_df["latitude"] = np.round(hw_df["latitude"].astype(np.float32), 1)
    hw_df["longitude"] = np.round(hw_df["longitude"].astype(np.float32), 1)
    hw_df["data_block_id"] = hw_df["data_block_id"].astype(np.int64)

    hw_df = pd.merge(
        hw_df,
        weather_station_to_county_mapping,
        how="left",
        on=["longitude", "latitude"],
    )
    hw_df = hw_df.drop(columns=["longitude", "latitude"])

    hw_df = (
        hw_df.groupby(["county", "datetime", "data_block_id"])
        .agg({col: "mean" for col in Features.HISTORICAL_WEATHER_COLS})
        .reset_index()
    )

    for col in Features.HISTORICAL_WEATHER_COLS:
        hw_df = hw_df.rename(columns={col: f"hw_{col}"})

    hw_df["datetime"] = hw_df.apply(
        lambda row: (
            row["datetime"] + pd.Timedelta(days=1)
            if row["datetime"].hour > 10
            else row["datetime"]
        ),
        axis=1,
    )

    hw_df["county"] = hw_df["county"].astype(np.int64)
    hw_df["data_block_id"] = hw_df["data_block_id"].astype(np.int64)

    df = pd.merge(df, hw_df, how="left", on=["county", "datetime", "data_block_id"])

    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday
    df["day"] = df["datetime"].dt.day

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    grouped_df = df.groupby("data_block_id")["target"]
    lagged_cols = []
    for lag in Features.TARGET_LAGS:
        col_name = f"target_lag_{lag}"
        lagged_col = grouped_df.shift(lag)
        lagged_cols.append(lagged_col.rename(col_name))

    df = pd.concat([df] + lagged_cols, axis=1)
    return df.dropna()


def process_numerical(
    df, dt_cols, numerical_cols, merge_cols, target_col, scaler=None
) -> (pd.DataFrame, StandardScaler):
    """
    Scale numerical features and target.

    Args:
        df (pd.DataFrame): Input dataframe.
        dt_cols (list[str]): Datetime columns.
        numerical_cols (list[str]): Numerical feature columns.
        merge_cols (list[str]): Merge columns.
        target_col (str): Target column.
        scaler (StandardScaler | None): Existing scaler.

    Returns:
        pd.DataFrame: Scaled numerical dataframe.
        StandardScaler: Fitted scaler.
    """

    cols = dt_cols + numerical_cols + merge_cols
    scale_cols = numerical_cols + [target_col]
    if target_col is not None:
        cols.append(target_col)

    df_scaled = df.copy()[cols]

    if scaler is None:
        scaler = StandardScaler()
        df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
    else:
        df_scaled[scale_cols] = scaler.transform(df_scaled[scale_cols])

    return df_scaled, scaler


def process_categorical(
    df: pd.DataFrame,
    dt_cols: list[str],
    categorical_cols: list[str],
    merge_cols: list[str],
    embedder=None,
) -> (pd.DataFrame, CategoricalEmbeddingEncoder):
    """
    Encode categorical features using embeddings.

    Args:
        df (pd.DataFrame): Input dataframe.
        dt_cols (list[str]): Datetime columns.
        categorical_cols (list[str]): Categorical columns.
        merge_cols (list[str]): Merge columns.
        embedder (CategoricalEmbeddingEncoder | None): Existing embedder.

    Returns:
        pd.DataFrame: Encoded categorical dataframe.
        CategoricalEmbeddingEncoder: Fitted embedder.
    """

    cols = dt_cols + merge_cols + categorical_cols
    df_cat = df.copy()[cols]

    if embedder is None:
        embedder = CategoricalEmbeddingEncoder(categorical_cols, EMBED_DIM)
        embedder.fit(df_cat[categorical_cols])

    encoded = embedder.transform(df_cat[categorical_cols])
    embedded_df = pd.DataFrame(
        encoded, columns=[f"{col}_idx" for col in categorical_cols], index=df_cat.index
    )

    for col in dt_cols + merge_cols:
        embedded_df[col] = df_cat[col].values

    return embedded_df, embedder


def process_ts_weekly(
    df: pd.DataFrame,
    dt_col: str,
    numerical_cols: list[str],
    groupby_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """
    Compute rolling weekly aggregations for time-series features.

    Args:
        df (pd.DataFrame): Input dataframe.
        dt_col (str): Datetime column.
        numerical_cols (list[str]): Numerical columns.
        groupby_cols (list[str]): Grouping columns.
        target_col (str): Target column.

    Returns:
        pd.DataFrame: Weekly aggregated dataframe.
    """

    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values([dt_col])

    def weekly_agg(group):
        group = group.set_index(dt_col)
        rolled = group[numerical_cols].rolling("7D").agg(["min", "max", "mean"])
        rolled.columns = ["_".join(col) for col in rolled.columns]

        for col in groupby_cols:
            rolled[col] = group[col]
        rolled = rolled.reset_index().dropna()
        return rolled

    weekly_df = df.groupby(groupby_cols, group_keys=True).apply(weekly_agg)
    weekly_df.index = np.arange(weekly_df.shape[0])
    return weekly_df


def create_results_df(
    num_df: pd.DataFrame,
    gru_preds: np.array,
    lgbm_preds: np.array,
    meta_preds: np.array,
    targets: np.array,
) -> pd.DataFrame:
    """
    Builds results dataframe containing predictions and targets.

    Args:
        num_df (pd.DataFrame): Numerical dataframe.
        gru_preds (np.array): GRU predictions.
        lgbm_preds (np.array): LGBM predictions.
        meta_preds (np.array): Ensemble predictions.
        targets (np.array): Ground truth targets.

    Returns:
        pd.DataFrame: Results dataframe.
    """
    common_len = min(len(gru_preds), len(lgbm_preds), len(meta_preds))

    df_subset = num_df.iloc[:common_len].copy()

    df_subset["gru_prediction"] = gru_preds[:common_len]
    df_subset["lgbm_prediction"] = lgbm_preds[:common_len]
    df_subset["ensemble_prediction"] = meta_preds[:common_len]

    df_subset["actual"] = targets[:common_len]

    return df_subset
