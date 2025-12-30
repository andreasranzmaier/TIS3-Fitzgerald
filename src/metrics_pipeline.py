# metrics_pipeline.py
from __future__ import annotations

import numpy as np
import pandas as pd

# Nixtla evaluators (used by StatsForecast docs)
import utilsforecast.losses as ufl               # MAE / RMSE / MAPE, etc.
from utilsforecast.evaluation import evaluate    # vectorized evaluator

# R²
from sklearn.metrics import r2_score             # standard definition


# ---------- OPE (Overall % Error) ----------
def _overall_percent_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Overall % Error (a.k.a. % bias): 100 * (sum(ŷ) - sum(y)) / sum(y).
    Returns NaN if sum(y) == 0 or any input missing after alignment.
    """
    a = pd.concat([y_true.rename("y"), y_pred.rename("yhat")], axis=1).dropna()
    denom = a["y"].sum()
    if denom == 0 or np.isnan(denom):
        return np.nan
    return (a["yhat"].sum() - a["y"].sum()) / denom * 100.0


# ---------- core metric computation using Nixtla evaluate ----------
def compute_metrics_with_utilsforecast(
    actuals_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    *,
    model_col: str
) -> pd.DataFrame:
    """
    actuals_df: columns [unique_id, ds, y]
    preds_df:   columns [unique_id, ds, <model_col>] (one column per model when repeated)
    model_col:  the name of the prediction column to score
    Returns tidy metrics rows for that model: metric, value
    """
    # evaluate expects [unique_id, ds, y, <model-col>]
    df = actuals_df.merge(preds_df[["unique_id", "ds", model_col]], on=["unique_id", "ds"], how="inner")

    # choose losses (Nixtla losses operate in raw units or percentages)
    losses = [ufl.mae, ufl.rmse, ufl.mape]

    ev = evaluate(df, metrics=losses)   # returns columns [unique_id, metric, <model_col>]
    # pivot to wide
    wide = ev.pivot(index="unique_id", columns="metric", values=model_col).reset_index()

    # Add R2 and OPE using the merged df
    # Align per unique_id
    add_rows = []
    for uid, g in df.groupby("unique_id"):
        r2 = r2_score(g["y"].values, g[model_col].values) if g["y"].var() > 0 else np.nan
        ope = _overall_percent_error(g["y"], g[model_col])
        add_rows.append({"unique_id": uid, "R2": r2, "OPE": ope})
    add_df = pd.DataFrame(add_rows)

    out = wide.merge(add_df, on="unique_id", how="left")
    # harmonize column names to upper-case metrics
    rename_map = {"mae": "MAE", "rmse": "RMSE", "mape": "MAPE"}
    out = out.rename(columns=rename_map)
    return out  # columns: unique_id, MAE, RMSE, MAPE, R2, OPE