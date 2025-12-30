# Capstone Project - Team Fitzgerald

Team-members: *Andreas Ranzmaier & Lea Treml*

## Initial Thoughts

After searching for a suitable dataset that allows us to analyze past trends and predict future developments, we decided to work with the Austrian Import and Export dataset. To make the analysis more focused and interesting, we narrowed the scope down to **Section 9 – Coffee, Tea, Mate, and Spices**, instead of looking at total imports and exports.

The dataset currently covers the period from **2007 onward** and is published at a **monthly frequency**, which makes it well suited for time series forecasting.

Ideally, since our task is to generate forecasts starting from **January 1st, 2026**, we would like to have complete data up to **December 2025**. However, at the moment, the most recent available data only goes up to **September 2025**. According to the publication schedule, preliminary data for **October 2025** will be released on **January 9th, 2026**, which will at least allow us to partially validate our forecasts once it becomes available. Because of this timing issue, we need to slightly bend the rules regarding data availability - but of course we also predicted the data for **January 2026**.

The core question we want to answer is whether observed increases in import values are driven by **higher quantities being imported**, or whether they are mainly caused by **rising prices** — so are we actually buying more, or is it just getting more expensive.

## Preprocessing

The data was downloaded via the **StatsCube link** provided on the official Statistics Austria website for international trade data:
[[https://www.statistik.at/statistiken/internationaler-handel/internationaler-warenhandel/importe-und-exporte-von-guetern](https://www.statistik.at/statistiken/internationaler-handel/internationaler-warenhandel/importe-und-exporte-von-guetern)]

Between downloading the raw `.csv` file and using it for preprocessing, some manual adjustments were necessary. Everything except the actual data and column names was removed. This included external metadata, empty columns, and empty rows. In addition, the monthly entries had to be reordered correctly, since this was not directly possible within StatsCube itself (at least to our knowledge).

For comparison purposes, we extracted **both quantity (kg)** and **value (EUR)** data for imports and exports from StatsCube.

All further preprocessing and cleaning steps were done in **Python**.

### Cleaning the Data

The first step was renaming the original **German column names** to English for better readability. The time column was renamed to `period`, and all quantity-related columns were shortened to `qty` in their names.

Next, the `period` column had to be converted into a proper date format. Originally, it was stored as a string containing abbreviated month names and two-digit years (e.g. `Feb.25`). To fix this:

* Unnecessary characters (such as dots or extra spaces) were removed.
* The month abbreviation was extracted and mapped to its corresponding numeric value (e.g. `Feb` → `02`).
* The two-digit year was extracted and expanded to a four-digit year.
* These components were combined into a standard date format (`YYYY-MM-01`) and converted into a proper date column.
* The original `period` column was dropped afterward.

This resulted in a clean and consistent monthly date column that could be used directly for time series analysis.

### Inflation Adjustment

After cleaning the raw data, we decided to additionally investigate whether changes in import and export values were influenced by **inflation effects**.

To do this, the Euro-based values were adjusted using official Austrian inflation data
(source: WKO inflation overview:
[[https://www.wko.at/statistik/prognose/inflation.pdf](https://www.wko.at/statistik/prognose/inflation.pdf)]).

The inflation adjustment was applied cumulatively, meaning that each year includes the inflation effects of all previous years. This allows for a more meaningful comparison of monetary values over time and helps differentiate between real growth and pure pricing effects.

## Seasonality and Time Series Diagnostics

Before fitting statistical forecasting models, we performed a set of **time series diagnostics** to better understand the structural properties of the data — in particular **seasonality, trend behavior, and stationarity**. This step is important because many statistical models (such as ARIMA) rely on assumptions about stationarity and seasonal structure.

### Time Series Construction

The selected variable was converted into a clean monthly time series by:

* keeping only the date and value columns,
* removing missing values,
* converting the date column into a proper datetime format,
* sorting the data chronologically,
* and setting the date as the time index.

For the diagnostics, we used **import quantities in kilograms (`import_qty_kg`)** as the default series, since it represents real *physical* demand and is not directly affected by inflation.

### Stationarity Tests

To assess stationarity, we applied two complementary statistical tests:

* **Augmented Dickey–Fuller (ADF) test**
  *Null hypothesis (H₀):* The series has a unit root (is non-stationary).

* **KPSS test**
  *Null hypothesis (H₀):* The series is stationary around a constant level.

The results are summarized below:

| Test | Null Hypothesis            | Result                  | Interpretation                             |
| ---- | -------------------------- | ----------------------- | ------------------------------------------ |
| ADF  | Unit root (non-stationary) | **Rejected (p < 0.01)** | No unit root → series is not a random walk |
| KPSS | Level-stationary           | **Rejected (p ≈ 0.04)** | Mean is not constant over time             |

This combination of results indicates that the series is **not a pure random walk**, but also **not strictly level-stationary**.

Such a pattern usually occurs when a time series is either:

* **trend-stationary** (stationary around a deterministic trend), or
* affected by **structural breaks** or gradual level shifts over time.

### Seasonal–Trend Decomposition (STL)

To further investigate the structure of the series, we applied **STL decomposition** with a monthly period (12) and robust fitting.

![](img/STL_decomposition.png)

The decomposition separates the series into:

* **Trend component**
  Shows slow-moving changes in the average level over time.

* **Seasonal component**
  Reveals a stable and recurring annual pattern, indicating strong monthly seasonality.

* **Remainder (residuals)**
  Contains short-term fluctuations and noise after removing trend and seasonality.

### Interpretation and Implications for Modeling

The diagnostics suggest that:

* The series exhibits **clear annual seasonality**
* There is **no evidence of a stochastic unit root**, but the mean is not constant
* The data is likely **trend-stationary with seasonality**, rather than difference-stationary

This justifies:

* the use of **seasonal baseline models** (e.g. seasonal naive),
* and **ARIMA-based models** with seasonal components and limited differencing

These findings informed our model selection and parameter choices in the forecasting stage.

### Autocorrelation

An autocorrelation scatter plot shows how a time series relates to itself when shifted by *k* time steps. The lag variable represents how far back in time the comparison is made.

![](img/autocorr_scatter.png)

The plot shows that **lag 1** has a higher correlation (≈ 0.59) compared to **lag 12** (≈ 0.35). Neither scatter plot shows a clear linear relationship; instead, the points form an elliptical cloud. This is typical for seasonal or cyclic behavior, which is consistent with our previous findings.

![](img/ACF_PACF-all.png)

![](img/ACF_PACF-120.png)

The ACF and PACF plots display the same underlying data, with the second figure zoomed in to focus on the most informative lags. The remaining lags provide little additional information for explaining short-term dynamics, especially for the most recent months.

---

### Visualisation

The following figures illustrate the development of import and export quantities as well as inflation-adjusted values over time.

![](img/import-export_quantities+inflation-adj_val.png)

![](img/real_inflation-adj_val-combarison.png)

To avoid misinterpretation, **“Real 2025 EUR”** refers to the **price level** in which all values are expressed, not the time span used to compute inflation.

To obtain inflation-adjusted values, we constructed a **CPI-like index** by compounding the officially published annual inflation rates for the years **2007–2025**. Each nominal value was rescaled as follows:

***Real Value_2025 = Nominal Value_year × (CPI_2025 / CPI_year)***

This means that a value observed in, for example, **2007** is adjusted to reflect the same purchasing power as in **2025**. As a result, all inflation-adjusted values are directly comparable across time and expressed in **2025 euros**, which explains why the corresponding columns are suffixed with `_real_2025`.

### Preprocessing thoughts

From the Preprocessing alone ARIMA or something like this seems to be a good choice for forecasting - of course our choice will only really fall when we have tested and compared all the different models and model types to give out a good, in our opinion, prediction for the followign months. 

## Predictions
In this section, we generate and evaluate forecasts for a single time series, focusing on monthly import values (in EUR). The goal is to compare models against each other from different families of models (baseline, statistical, maschine learning and neural), evaluate their out-of-sample performance, and produce short-term forecasts beyond the last available observation.

The forecasting pipeline follows a consistent train–validation–test setup and applies the same evaluation metrics across all models to ensure comparability.

### Baseline
Baseline models are used as simple reference points. They are computationally cheap, easy to interpret, and provide a lower bound for forecasting performance. If more complex models cannot clearly outperform these baselines, their added complexity is usually not justified.

The following baseline methods were implemented:

- **Naive**
    - Assumes that future values remain equal to the last observed value.

- **Seasonal Naive**
    - Repeats the value from the same month in the previous year, assuming strong seasonality.

- **Historic Average**
    - Computes the average of the most recent seasonal cycle and uses this average to forecast future periods.

Each baseline model is trained on the historical data and evaluated on both a validation split and a test split, using time-based splits to avoid data leakage.

### Statistical
In addition to baseline approaches, we apply more advanced statistical time series models using the statsforecast framework. These models are able to capture trends, seasonality, and autocorrelation structures in the data more explicitly.

The statistical models used are:
- **Random Walk with Drift**   
    - Extends the naive approach by adding a constant drift term estimated from historical data.

- **ARIMA**
    - A classical autoregressive integrated moving average model with fixed parameters.

- **Auto-ARIMA**
    - Automatically selects the best ARIMA configuration based on information criteria, reducing manual tuning.

The data is converted into the required (unique_id, ds, y) format.
Models are then trained first on the training set and then on the combined training + validation set.

Forecasts are generated for both the validation horizon, the test horizon and the prediction for the future.
### Maschine Learning

### Neural

### Evaluation and Forecast Generation

Model performance is evaluated using multiple metrics:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R²
- OPE (Overall Percentage Error)

These metrics are computed separately for the validation and test splits, allowing us to assess both model selection and generalization performance.

After evaluation, each model is retrained on the full available dataset and used to generate future forecasts for the next four months beyond the last observed date. These forecasts are visualized together with historical data (from 2020 onward) to highlight recent dynamics and model behavior.

### Output
All evaluations ar stored in a structured results table for further analysis, seperateed based on the family. 

## Analysis

## Closing Thoughts

## Useful Links
 https://www.statistik.at/fileadmin/shared/QM/Standarddokumentationen/U/std_u_itgs.pdf
