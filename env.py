from enum import StrEnum, auto
import os

ROOT_PATH = "/home/surikadt/personal/ds/cw"
DATA_PATH = os.path.join(ROOT_PATH, "data")

CSV_PATH = os.path.join(DATA_PATH, "csv")
PT_DATA_PATH = os.path.join(DATA_PATH, "pt_data")

TRAIN_PATH = os.path.join(CSV_PATH, "train.csv")
GAS_PRICES_PATH = os.path.join(CSV_PATH, "gas_prices.csv")
CLIENT_PATH = os.path.join(CSV_PATH, "client.csv")
ELECTRICITY_PRICES_PATH = os.path.join(CSV_PATH, "electricity_prices.csv")
FORECAST_WEATHER_PATH = os.path.join(CSV_PATH, "forecast_weather.csv")
HISTORICAL_WEATHER_PATH = os.path.join(CSV_PATH, "historical_weather.csv")
WEATHER_STATION_MAPPING_PATH = os.path.join(
    CSV_PATH, "weather_station_to_county_mapping.csv"
)

LOG_DIR = "./runs"
MODEL_DIR = "./models/trained_models"

TRAIN_BATCH_SIZE = 6000
VAL_BATCH_SIZE = 2048
EMBED_DIM = 3


class DataSourceType(StrEnum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()


class Features:
    TARGET_LAGS = [1] + list(range(24, 15 * 24 + 1, 24))
    CATEGORICAL_FEATURES = [
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "hour",
        "day",
        "weekday",
        "month",
    ]
    NUMERICAL_FEATURES = [
        "eic_count",
        "installed_capacity",
        "euros_per_mwh",
        "fw_temperature",
        "fw_dewpoint",
        "fw_snowfall",
        "fw_total_precipitation",
        "fw_new_feature",
        "fw_cloudcover_low",
        "fw_cloudcover_mid",
        "fw_cloudcover_high",
        "fw_cloudcover_total",
        "fw_10_metre_u_wind_component",
        "fw_10_metre_v_wind_component",
        "fw_direct_solar_radiation",
        "fw_surface_solar_radiation_downwards",
        "hw_temperature",
        "hw_dewpoint",
        "hw_rain",
        "hw_snowfall",
        "hw_surface_pressure",
        "hw_cloudcover_low",
        "hw_cloudcover_mid",
        "hw_cloudcover_high",
        "hw_cloudcover_total",
        "hw_windspeed_10m",
        "hw_winddirection_10m",
        "hw_shortwave_radiation",
        "hw_direct_solar_radiation",
        "hw_diffuse_radiation",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ] + [f"target_lag_{lag}" for lag in TARGET_LAGS]

    FORECAST_WEATHER_COLS = [
        "temperature",
        "dewpoint",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_high",
        "cloudcover_total",
        "10_metre_u_wind_component",
        "10_metre_v_wind_component",
        "direct_solar_radiation",
        "surface_solar_radiation_downwards",
        "snowfall",
        "total_precipitation",
    ]
    HISTORICAL_WEATHER_COLS = [
        "temperature",
        "dewpoint",
        "rain",
        "snowfall",
        "surface_pressure",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_high",
        "cloudcover_total",
        "windspeed_10m",
        "winddirection_10m",
        "shortwave_radiation",
        "direct_solar_radiation",
        "diffuse_radiation",
    ]
    DATETIME_COL = "datetime"
    TARGET_COL = "target"
    GROUPBY_COLS = ["prediction_unit_id"]


class GruParams:
    EPOCHS = 30
    LR = 1e-3
    INPUT_SIZE = len(Features.NUMERICAL_FEATURES)
    HIDDEN_SIZE = 64
    LAYERS = 1
    OUTPUT_SIZE = 1


class AEParams:
    EPOCHS = 50
    LR = 1e-3
    LATENT_DIM = 32
    INPUT_DIM = (
        len(Features.NUMERICAL_FEATURES)
        + len(Features.FORECAST_WEATHER_COLS)
        + len(Features.HISTORICAL_WEATHER_COLS)
    )


LGBM_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 3000,
    "learning_rate": 1e-2,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "device_type": "gpu",
    "num_threads": 7,
    "seed": 42,
}
