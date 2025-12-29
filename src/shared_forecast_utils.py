from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, RandomWalkWithDrift, ARIMA, AutoARIMA

# Fixed split train data
SPLIT_YEARS = {
    "train": (pd.Timestamp("2007-10-01"), pd.Timestamp("2021-09-01")),
    "val":   (pd.Timestamp("2021-10-01"), pd.Timestamp("2023-09-01")),
    "test":  (pd.Timestamp("2023-10-01"), pd.Timestamp("2025-09-01")),
}

# Families for coloring & classification
FAMILY_MAP = {
    # baseline
    'naive': 'baseline',
    'seasonal_naive': 'baseline',
    'historic_average': 'baseline',
    # statistical
    'rw_drift': 'statistical',
    'arima': 'statistical',
    'auto_arima': 'statistical',
    # ml
    'linreg': 'ml',
    'ridge': 'ml',
    'rf': 'ml',
    'gbr': 'ml',
    # neural
    'rnn': 'neural',
    'nbeats': 'neural',
    'nhits': 'neural',
    'mlp': 'neural',
}

@dataclass
class ModelForecastResult:
    method: str
    val_pred: pd.Series
    test_pred: pd.Series
    val_true: pd.Series
    test_true: pd.Series
    family: str

def get_split_masks(dates: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    train_start, train_end = SPLIT_YEARS["train"]
    val_start,   val_end   = SPLIT_YEARS["val"]
    test_start,  test_end  = SPLIT_YEARS["test"]
    train_mask = (dates >= train_start) & (dates <= train_end)
    val_mask   = (dates >= val_start)   & (dates <= val_end)
    test_mask  = (dates >= test_start)  & (dates <= test_end)

    return train_mask, val_mask, test_mask

def make_future_dates(start: pd.Timestamp, h: int, freq: str = "MS") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=h, freq=freq)

def historic_avg_last_season(y: np.ndarray, h: int, season_length: int) -> np.ndarray:
    window = y[-season_length:] if y.size >= season_length else y
    mean_val = float(np.nanmean(window)) if window.size else np.nan
    return np.full(h, mean_val)

def get_friendly_model_name(key: str) -> Tuple[str, str]:
    key = key.lower().strip()
    return {
        "naive": ("Naive", "statsforecast"),
        "seasonal_naive": ("SeasonalNaive", "statsforecast"),
        "rw_drift": ("RandomWalkWithDrift", "statsforecast"),
        "arima": ("ARIMA", "statsforecast"),
        "auto_arima": ("AutoARIMA", "statsforecast"),
        "historic_average": ("HistoricAverage", "custom"),
        "rnn": ("RNNModel", "neuralforecast"),
        "nbeats": ("NBEATS", "neuralforecast"),
        "nhits": ("NHITS", "neuralforecast"),
        "mlp": ("MLP", "neuralforecast"),
        "linreg": ("LinearRegression", "mlforecast"),
        "ridge": ("Ridge", "mlforecast"),
        "rf": ("RandomForest", "mlforecast"),
        "gbr": ("GradientBoosting", "mlforecast"),
    }.get(key, (key, "custom"))

def run_statsforecast_model(
    df_uid_ds_y: pd.DataFrame,
    h: int,
    method: str,
    season_length: int,
    *,
    arima_order: tuple[int, int, int] = (1, 1, 1),
    freq: str = "MS",
) -> pd.Series:
    """
    Supports: 'naive', 'seasonal_naive', 'rw_drift', 'arima', 'auto_arima', 'rnn'.
    ARIMA in StatsForecast uses `order=(p,d,q)` and optional `season_length=m`.
    AutoARIMA uses `season_length=m` and auto-selects orders.
    """
    m = method.lower().strip()
    if m == "naive":
        model = Naive()
    elif m == "seasonal_naive":
        model = SeasonalNaive(season_length=season_length)
    elif m == "rw_drift":
        model = RandomWalkWithDrift()
    elif m == "arima":
        model = ARIMA(order=arima_order)
    elif m == "auto_arima":
        model = AutoARIMA(season_length=season_length)
    elif m == "rnn":
        # neuralforecast RNN model
        from neuralforecast.core import NeuralForecast
        from neuralforecast.models import RNN
        try:
            from neuralforecast.losses.pytorch import MSE
        except Exception:  # fallback for older versions
            from neuralforecast.losses import mse as MSE  # type: ignore

        # Minimal, sensible defaults; tune as needed
        model = RNN(
            h=h,
            input_size=season_length*2,
            loss=MSE(),
            inference_input_size=24,
            start_padding_enabled=True,
            scaler_type='robust',
            encoder_n_layers=2,
            encoder_hidden_size=128,
            decoder_layers=2,
            decoder_hidden_size=128,
            max_steps=500,
            learning_rate=1e-3,
        )

        nf = NeuralForecast(models=[model], freq=freq)
        nf.fit(df_uid_ds_y)
        fc = nf.predict()
        value_cols = [c for c in fc.columns if c not in ("unique_id", "ds")]
        return fc.set_index("ds")[value_cols[0]]

    else:
        raise ValueError(
            f"Unsupported method '{method}'. "
            "Use one of: 'naive', 'seasonal_naive', 'rw_drift', 'arima', 'auto_arima', 'rnn'."
        )

    sf = StatsForecast(models=[model], freq=freq, n_jobs=1, verbose=False)
    sf.fit(df_uid_ds_y)
    fc = sf.predict(h=h)
    value_cols = [c for c in fc.columns if c not in ("unique_id", "ds")]
    return fc.set_index("ds")[value_cols[0]]