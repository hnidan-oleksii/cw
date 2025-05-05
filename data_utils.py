import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

from models import CategoricalEmbeddingEncoder
from env import (
    TRAIN_PATH, CLIENT_PATH, ELECTRICITY_PRICES_PATH, FORECAST_WEATHER_PATH,
    HISTORICAL_WEATHER_PATH, FORECAST_WEATHER_COLS, HISTORICAL_WEATHER_COLS,
    WEATHER_STATION_MAPPING_PATH, DATETIME_COL, TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE, TARGET_LAGS, EMBED_DIM
)


def get_all_data(
    dt_col,
    numerical_cols,
    categorical_cols,
    groupby_cols,
    target_col,
    gru_dataset_path: str = None,
    rerun_gru_dataset=False,
    scaler=None,
    embedder=None,
    split_ratios=(0.85, 0.10, 0.05),
    df_type="train"
):
    df = read_all()

    if round(sum(split_ratios), 2) != 1:
        raise ValueError("Split ratios do sum to 1")

    block_ids_num = df["data_block_id"].nunique()
    train_id_cutoff = int(block_ids_num * split_ratios[0])
    val_id_cutoff = train_id_cutoff + int(block_ids_num * split_ratios[1])

    if df_type == "train":
        block_ids = list(range(0, train_id_cutoff))
    elif df_type == "val":
        block_ids = list(range(train_id_cutoff, val_id_cutoff))
    elif df_type == "test":
        block_ids = list(range(val_id_cutoff, block_ids_num))
    else:
        raise ValueError(
            f"Incorrect df_type: {df_type}. Must be one of: train, val, test."
        )

    df = df[df["data_block_id"].isin(block_ids)]

    num_df, cat_df, scaler, embedder = get_numerical_categorical_preprocessed(
        full_df=df,
        dt_col=dt_col,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        merge_cols=groupby_cols,
        target_col=target_col,
        scaler=scaler,
        embedder=embedder
    )

    weekly_df = process_ts_weekly(
        df=num_df,
        dt_col=dt_col,
        numerical_cols=numerical_cols,
        groupby_cols=groupby_cols,
        target_col=target_col
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
            valid_idx=valid_idx
        )
    else:
        gru_dataset = torch.load(gru_dataset_path, weights_only=False)
    if rerun_gru_dataset and gru_dataset_path:
        torch.save(gru_dataset, gru_dataset_path)

    batch_size = TRAIN_BATCH_SIZE if df_type == "train" else VAL_BATCH_SIZE
    gru_loader = DataLoader(
        gru_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )

    weekly_keys = weekly_df.set_index([dt_col, *groupby_cols]).index
    num_df = num_df[num_df.set_index(
        [dt_col, *groupby_cols]
    ).index.isin(weekly_keys)]
    cat_df = cat_df[cat_df.set_index(
        [dt_col, *groupby_cols]
    ).index.isin(weekly_keys)]

    return num_df, cat_df, weekly_df, gru_loader, scaler, embedder


def read_all():
    df = pd.read_csv(TRAIN_PATH, parse_dates=[DATETIME_COL])
    client_df = pd.read_csv(CLIENT_PATH, parse_dates=['date'])
    electricity_df = pd.read_csv(
        ELECTRICITY_PRICES_PATH, parse_dates=['origin_date', 'forecast_date']
    )
    fw_df = pd.read_csv(
        FORECAST_WEATHER_PATH,
        parse_dates=['origin_datetime', 'forecast_datetime']
    )
    hw_df = pd.read_csv(HISTORICAL_WEATHER_PATH, parse_dates=[DATETIME_COL])

    weather_station_to_county_mapping = pd.read_csv(
        WEATHER_STATION_MAPPING_PATH
    )

    client_df = client_df \
        .drop(columns=["date"]) \
        .sort_values(by=["data_block_id"], ascending=True)
    client_merge_cols = ["county", "is_business", "product_type",
                         "data_block_id", "eic_count", "installed_capacity"]
    df = pd.merge(
        df, client_df[client_merge_cols], how="left",
        on=["county", "is_business", "product_type", "data_block_id"]
    )

    electricity_df = electricity_df.drop(columns=["origin_date"])
    electricity_df = electricity_df.rename(
        columns={"forecast_date": "datetime"}
    )
    electricity_df["datetime"] = pd.to_datetime(electricity_df["datetime"]) \
        + pd.Timedelta(days=1)
    electricity_df["euros_per_mwh"] = electricity_df["euros_per_mwh"].abs() + 0.1
    df = pd.merge(
        df, electricity_df[["data_block_id", "datetime", "euros_per_mwh"]],
        how="left", on=["data_block_id", "datetime"]
    )

    fw_df["datetime"] = pd.to_datetime(fw_df["origin_datetime"]) \
        + pd.to_timedelta(fw_df["hours_ahead"], unit='h')
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
        fw_df, weather_station_to_county_mapping,
        how="left", on=["longitude", "latitude"]
    )
    fw_df = fw_df.drop(columns=["longitude", "latitude", "origin_datetime"])

    fw_df = fw_df.groupby(["county", "datetime", "data_block_id"]).agg(
        {col: "mean" for col in FORECAST_WEATHER_COLS}
    ).reset_index()

    for col in FORECAST_WEATHER_COLS:
        fw_df = fw_df.rename(columns={col: f"fw_{col}"})

    fw_df["county"] = fw_df["county"].astype(np.int64)
    fw_df["data_block_id"] = fw_df["data_block_id"].astype(np.int64)

    df = pd.merge(
        df, fw_df,
        how="left", on=["county", "datetime", "data_block_id"]
    )

    df["fw_new_feature"] = (
        (df["installed_capacity"] * df["fw_surface_solar_radiation_downwards"])
        / (df["fw_temperature"] + 273.15)
    )

    hw_df["datetime"] = pd.to_datetime(hw_df["datetime"]) \
        + pd.Timedelta(days=1)
    hw_df["latitude"] = np.round(hw_df["latitude"].astype(np.float32), 1)
    hw_df["longitude"] = np.round(hw_df["longitude"].astype(np.float32), 1)
    hw_df["data_block_id"] = hw_df["data_block_id"].astype(np.int64)

    hw_df = pd.merge(
        hw_df, weather_station_to_county_mapping,
        how="left", on=["longitude", "latitude"]
    )
    hw_df = hw_df.drop(columns=["longitude", "latitude"])

    hw_df = hw_df.groupby(["county", "datetime", "data_block_id"]).agg(
        {col: "mean" for col in HISTORICAL_WEATHER_COLS}
    ).reset_index()

    for col in HISTORICAL_WEATHER_COLS:
        hw_df = hw_df.rename(columns={col: f"hw_{col}"})

    hw_df["datetime"] = hw_df.apply(
        lambda row: row["datetime"] + pd.Timedelta(days=1)
            if row["datetime"].hour > 10
            else row["datetime"],
        axis=1
    )

    hw_df["county"] = hw_df["county"].astype(np.int64)
    hw_df["data_block_id"] = hw_df["data_block_id"].astype(np.int64)

    df = pd.merge(
        df, hw_df,
        how="left", on=["county", "datetime", "data_block_id"]
    )

    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['weekday'] = df['datetime'].dt.weekday
    df['day'] = df['datetime'].dt.day

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    grouped_df = df.groupby('data_block_id')['target']
    lagged_cols = []
    for lag in TARGET_LAGS:
        col_name = f'target_lag_{lag}'
        lagged_col = grouped_df.shift(lag)
        lagged_cols.append(lagged_col.rename(col_name))

    df = pd.concat([df] + lagged_cols, axis=1)
    return df.dropna()


def get_numerical_categorical_preprocessed(
    full_df,
    dt_col,
    numerical_cols,
    categorical_cols,
    merge_cols,
    target_col,
    scaler=None,
    embedder=None
):
    numerical, scaler = process_numerical(
        df=full_df,
        dt_cols=[dt_col],
        numerical_cols=numerical_cols,
        merge_cols=merge_cols,
        target_col=target_col,
        scaler=scaler
    )
    categorical, embedder = process_categorical(
        df=full_df,
        dt_cols=[dt_col],
        categorical_cols=categorical_cols,
        merge_cols=merge_cols,
        embedder=embedder
    )

    return numerical, categorical, scaler, embedder


def process_numerical(
    df,
    dt_cols,
    numerical_cols,
    merge_cols,
    target_col,
    scaler=None
):
    cols = dt_cols + numerical_cols + merge_cols
    scale_cols = numerical_cols + [target_col]
    if target_col is not None:
        cols.append(target_col)

    df_scaled = df.copy()[cols]

    if scaler is None:
        scaler = StandardScaler()
        df_scaled[scale_cols] = scaler.fit_transform(
            df_scaled[scale_cols]
        )
    else:
        df_scaled[scale_cols] = scaler.transform(
            df_scaled[scale_cols]
        )

    return df_scaled, scaler


def process_categorical(
    df,
    dt_cols,
    categorical_cols,
    merge_cols,
    embedder=None
):
    cols = dt_cols + merge_cols + categorical_cols
    df_cat = df.copy()[cols]

    if embedder is None:
        embedder = CategoricalEmbeddingEncoder(categorical_cols, EMBED_DIM)
        embedder.fit(df_cat[categorical_cols])

    encoded = embedder.transform(df_cat[categorical_cols])
    embedded_df = pd.DataFrame(
        encoded,
        columns=[f"{col}_idx" for col in categorical_cols],
        index=df_cat.index
    )

    for col in dt_cols + merge_cols:
        embedded_df[col] = df_cat[col].values

    return embedded_df, embedder


def process_ts_weekly(df, dt_col, numerical_cols, groupby_cols, target_col):
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values([dt_col])

    def weekly_agg(group):
        group = group.set_index(dt_col)
        rolled = group[numerical_cols] \
            .rolling('7D') \
            .agg(['min', 'max', 'mean'])
        rolled.columns = ['_'.join(col) for col in rolled.columns]

        for col in groupby_cols:
            rolled[col] = group[col]
        rolled = rolled.reset_index().dropna()
        return rolled

    weekly_df = df.groupby(groupby_cols, group_keys=True).apply(
        weekly_agg
    )
    weekly_df.index = np.arange(weekly_df.shape[0])
    return weekly_df


class GRUDataset(Dataset):
    def __init__(
        self,
        df,
        dt_col,
        numerical_cols,
        groupby_cols,
        target_col,
        valid_idx,
        window_size=24
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
                idx_range = indices[i:i + window_size]
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
