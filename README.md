# Timeseries Lab

Exploratory notebooks and utilities for analyzing beverage sales from 2021 to 2025 using classical statistics, machine learning, and Nixtla forecasting libraries.

## Getting Started

1. Install dependencies with `uv sync`.
2. Launch the notebook in `src/visualalisation.ipynb` to experiment with the datasets under `data/`.

## Data

Source data lives in `data/Kaffee_Tee_Mate_und_Gewuerze_2021-2025.csv` and contains time-indexed beverage counts.

## Tooling

The project leans on numpy/pandas for manipulation, polars for fast tabular work, and Nixtla's `statsforecast`, `mlforecast`, and `neuralforecast` for modeling.

## Dokumentation

This project is documented in the `Docu.md`file.