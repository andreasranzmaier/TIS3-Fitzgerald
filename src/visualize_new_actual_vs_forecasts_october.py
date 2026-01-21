from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "Kaffee_Tee_Mate_und_Gewuerze_2007-2025_cleaned.csv"

FEATURES = {
    # From notebook: src/forecast_import_qty_kg.ipynb
    "import_qty_kg": {
        "label": "Import qty (kg)",
        "future_only": BASE_DIR / "results" / "forecast_results_import_qty_kg_future_only.csv",
        "best3": ["naive", "rw_drift", "historic_average"],
    },
    # From notebook: src/forecast_import_value_eur.ipynb
    "import_value_eur": {
        "label": "Import value (EUR)",
        "future_only": BASE_DIR / "results" / "forecast_results_import_value_eur_future_only.csv",
        "best3": ["auto_arima", "gbr", "ridge"],
    },
    # From notebook: src/forecast_import_value_eur_real_2025.ipynb
    "import_value_eur_real_2025": {
        "label": "Import value (real 2025 EUR)",
        "future_only": BASE_DIR
        / "results"
        / "forecast_results_import_value_eur_real_2025_future_only.csv",
        "best3": ["auto_arima", "rf", "ridge"],
    },
}


def _read_actual(data_path: Path, date_str: str, column: str) -> float:
    df = pd.read_csv(data_path)
    row = df[df["date"] == date_str]
    if row.empty:
        raise ValueError(f"No actual data for {date_str} in {data_path}")
    return float(row[column].iloc[0])


def _read_predictions(future_path: Path, date_str: str, models: list[str]) -> pd.Series:
    df = pd.read_csv(future_path)
    df = df[df["ds"] == date_str]
    df = df[df["model"].isin(models)]

    found = df.set_index("model")["yhat"]
    missing = [m for m in models if m not in found.index]
    if missing:
        raise ValueError(
            f"Missing forecast(s) for {date_str} in {future_path}: {', '.join(missing)}"
        )

    # keep the model order as requested
    return found.reindex(models).astype(float)


def build_comparison(date_str: str) -> pd.DataFrame:
    rows: list[dict] = []

    for feature_name, cfg in FEATURES.items():
        actual = _read_actual(DATA_FILE, date_str, feature_name)
        preds = _read_predictions(cfg["future_only"], date_str, cfg["best3"])

        for model_name, yhat in preds.items():
            err = float(yhat - actual)
            rows.append(
                {
                    "date": date_str,
                    "feature": feature_name,
                    "label": cfg["label"],
                    "model": model_name,
                    "predicted": float(yhat),
                    "actual": float(actual),
                    "error": err,
                    "abs_error": abs(err),
                    "pct_error": (err / actual * 100.0) if actual != 0 else np.nan,
                }
            )

    return pd.DataFrame(rows)


def plot_feature_bars(
    summary: pd.DataFrame, *, feature: str, label: str, date_str: str, out_path: Path
) -> None:
    g = summary[summary["feature"] == feature].copy()
    if g.empty:
        raise ValueError(f"No summary rows for feature={feature}")

    # stable order: Actual first, then model order from FEATURES
    models = FEATURES[feature]["best3"]
    g["model"] = pd.Categorical(g["model"], categories=models, ordered=True)
    g = g.sort_values("model")

    actual = float(g["actual"].iloc[0])
    labels = ["Actual"] + models
    values = [actual] + g["predicted"].tolist()

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#333333"] + [None] * len(models)  # matplotlib default cycle for model bars

    for i, (xi, v) in enumerate(zip(x, values)):
        ax.bar(xi, v, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(f"{label} — {date_str}: actual vs precomputed forecasts (top 3)")
    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.2)

    # annotate values for quick read
    for xi, v in zip(x, values):
        ax.annotate(
            f"{v:,.0f}",
            (float(xi), float(v)),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_timeseries_overlay(
    summary: pd.DataFrame,
    *,
    feature: str,
    label: str,
    date_str: str,
    out_path: Path,
) -> None:
    # Load the original time series
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    if feature not in df.columns:
        raise ValueError(f"Feature column '{feature}' not found in {DATA_FILE}")

    compare_date = pd.to_datetime(date_str)
    ts = df[["date", feature]].dropna().copy()

    # Zoom to the last year ending at the comparison date
    start_date = compare_date - pd.DateOffset(years=1)
    ts = ts[(ts["date"] >= start_date) & (ts["date"] <= compare_date)].copy()

    # Predictions + actual for the comparison date
    g = summary[summary["feature"] == feature].copy()
    if g.empty:
        raise ValueError(f"No summary rows for feature={feature}")

    models = FEATURES[feature]["best3"]
    g["model"] = pd.Categorical(g["model"], categories=models, ordered=True)
    g = g.sort_values("model")

    actual_value = float(g["actual"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 4.8))

    ax.plot(ts["date"], ts[feature].astype(float), label="Actual (time series)")

    # Mark the passed-in actual datapoint explicitly
    x_point = np.array([compare_date.to_datetime64()])
    ax.scatter(
        x_point,
        np.array([actual_value], dtype=float),
        color="#333333",
        s=45,
        zorder=5,
        label="Actual (passed value)",
    )

    # Plot each model prediction as a point at the comparison date
    for _, row in g.iterrows():
        ax.scatter(
            x_point,
            np.array([float(row["predicted"])], dtype=float),
            s=45,
            zorder=5,
            label=f"Predicted ({row['model']})",
        )

    ax.set_title(f"{label} — time series with {date_str} actual & forecasts")
    ax.set_ylabel(label)
    ax.set_xlabel("Date")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)


    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "No-retraining visualization: compare the new October actual datapoint "
            "against previously saved forecasts for the pre-selected top-3 models per feature."
        )
    )
    parser.add_argument(
        "--date",
        default="2025-10-01",
        help="Date to compare (YYYY-MM-01). Default: 2025-10-01",
    )
    parser.add_argument(
        "--out-dir",
        default=str(BASE_DIR / "docu" / "img"),
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--csv",
        default=str(BASE_DIR / "results" / "october_actual_vs_forecasts_top3.csv"),
        help="Path to save the comparison table CSV.",
    )
    args = parser.parse_args()

    summary = build_comparison(args.date)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.csv, index=False)

    out_dir = Path(args.out_dir)
    for feature_name, cfg in FEATURES.items():
        out_path = out_dir / f"{feature_name}_{args.date}_actual_vs_forecast_top3.png"
        plot_feature_bars(
            summary,
            feature=feature_name,
            label=cfg["label"],
            date_str=args.date,
            out_path=out_path,
        )

        ts_path = out_dir / f"{feature_name}_{args.date}_timeseries_with_forecasts_top3.png"
        plot_feature_timeseries_overlay(
            summary,
            feature=feature_name,
            label=cfg["label"],
            date_str=args.date,
            out_path=ts_path,
        )

    print(f"Saved comparison CSV: {args.csv}")
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
