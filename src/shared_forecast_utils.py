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


def run_mlforecast_model(
    df_uid_ds_y: pd.DataFrame,
    h: int,
    method: str,
    season_length: int,
    *,
    freq: str = "MS",
    params: dict | None = None,
) -> pd.Series:
    """Run an MLForecast model (leakage-safe if df is already appropriately truncated).

    Supported methods: 'linreg', 'ridge', 'rf', 'gbr'
    """
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import ExpandingMean

    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge

    m = method.lower().strip()
    params = params or {}

    if m == "linreg":
        model = LinearRegression(**params)
    elif m == "ridge":
        model = Ridge(**params)
    elif m == "rf":
        model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    elif m == "gbr":
        model = GradientBoostingRegressor(random_state=42, **params)
    else:
        raise ValueError(f"Unsupported ML method '{method}'.")

    # Minimal, broadly useful features. Keep deterministic to aid tuning.
    lags = sorted({1, 2, 3, 6, season_length})
    lag_transforms = {1: [ExpandingMean()]}
    date_features = ["month"]

    fcst = MLForecast(
        models={m: model},
        freq=freq,
        lags=lags,
        lag_transforms=lag_transforms,
        date_features=date_features,
    )
    fcst.fit(df_uid_ds_y, id_col="unique_id", time_col="ds", target_col="y")
    preds = fcst.predict(h=h)
    return preds.set_index("ds")[m]


def run_neuralforecast_model(
    df_uid_ds_y: pd.DataFrame,
    h: int,
    method: str,
    season_length: int,
    *,
    freq: str = "MS",
    params: dict | None = None,
) -> pd.Series:
    """Run a NeuralForecast model.

    Supported methods: 'rnn', 'nbeats', 'nhits', 'mlp'
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import MLP, NBEATS, NHITS, RNN

    try:
        from neuralforecast.losses.pytorch import MSE
        loss = MSE()
    except Exception:  # older versions
        loss = None

    m = method.lower().strip()
    params = params or {}
    input_size = int(params.pop("input_size", season_length * 2))
    max_steps = int(params.pop("max_steps", 300))
    learning_rate = float(params.pop("learning_rate", 1e-3))

    common = dict(
        h=h,
        input_size=input_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        scaler_type=params.pop("scaler_type", "robust"),
    )
    if loss is not None:
        common["loss"] = loss

    if m == "rnn":
        model = RNN(**common, **params)
    elif m == "nbeats":
        model = NBEATS(**common, **params)
    elif m == "nhits":
        model = NHITS(**common, **params)
    elif m == "mlp":
        model = MLP(**common, **params)
    else:
        raise ValueError(f"Unsupported neural method '{method}'.")

    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df_uid_ds_y)
    fc = nf.predict()
    value_cols = [c for c in fc.columns if c not in ("unique_id", "ds")]
    return fc.set_index("ds")[value_cols[0]]


def run_any_model(
    df_uid_ds_y: pd.DataFrame,
    h: int,
    method: str,
    season_length: int,
    *,
    freq: str = "MS",
    arima_order: tuple[int, int, int] = (1, 1, 1),
    params: dict | None = None,
) -> pd.Series:
    """Dispatch helper for any method in FAMILY_MAP."""
    m = method.lower().strip()
    family = FAMILY_MAP.get(m)
    if family in ("baseline", "statistical"):
        return run_statsforecast_model(
            df_uid_ds_y,
            h=h,
            method=m,
            season_length=season_length,
            arima_order=arima_order,
            freq=freq,
        )
    if family == "ml":
        return run_mlforecast_model(
            df_uid_ds_y,
            h=h,
            method=m,
            season_length=season_length,
            freq=freq,
            params=params,
        )
    if family == "neural":
        return run_neuralforecast_model(
            df_uid_ds_y,
            h=h,
            method=m,
            season_length=season_length,
            freq=freq,
            params=params,
        )
    raise ValueError(f"Unknown method '{method}' (family={family}).")