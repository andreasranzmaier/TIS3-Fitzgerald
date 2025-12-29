# Capstone Project - Team Fitzgerald

Team-members: *Andreas Ranzmaier & Lea Treml*

## Initial Thoughts

After searching for a suitable dataset that allows us to analyze past trends and predict future developments, we decided to work with the Austrian Import and Export dataset. To make the analysis more focused and interesting, we narrowed the scope down to **Section 9 – Coffee, Tea, Mate, and Spices**, instead of looking at total imports and exports.

The dataset currently covers the period from **2007 onward** and is published at a **monthly frequency**, which makes it well suited for time series forecasting.

Ideally, since our task is to generate forecasts starting from **January 1st, 2026**, we would like to have complete data up to **December 2025**. However, at the moment, the most recent available data only goes up to **September 2025**. According to the publication schedule, preliminary data for **October 2025** will be released on **January 9th, 2026**, which will at least allow us to partially validate our forecasts once it becomes available. Because of this timing issue, we need to slightly bend the rules regarding data availability.

The core question we want to answer is whether observed increases in import values are driven by **higher quantities being imported**, or whether they are mainly caused by **rising prices** (so yeah, are we actually buying more, or is it just getting more expensive).

## Preprocessing

The data was downloaded via the **StatsCube link** provided on the official Statistics Austria website for international trade data [https://www.statistik.at/statistiken/internationaler-handel/internationaler-warenhandel/importe-und-exporte-von-guetern].

Between downloading the raw `.csv` file and using it for preprocessing, some manual adjustments were necessary. Everything except the actual data and column names was removed. This included external metadata, empty columns, and empty rows. In addition, the monthly entries had to be reordered correctly, since this was not directly possible within StatsCube itself (at least to our knowledge).

For comparison purposes, we extracted **both quantity (kg)** and **value (EUR)** data for imports and exports from StatsCube.

All further preprocessing and cleaning steps were done in **Python**.

---

### Cleaning the Data

The first step was renaming the original **German column names** to English for better readability. The time column was renamed to `period`, and all quantity-related columns were shortened to `qty` in their names.

Next, the `period` column had to be converted into a proper date format. Originally, it was stored as a string containing abbreviated month names and two-digit years (e.g. `Feb.25`).
To fix this:

* Any unnecessary characters (such as dots or extra spaces) were removed.
* The month abbreviation was extracted and mapped to its corresponding numeric value (e.g. `Feb` → `02`).
* The two-digit year was extracted and expanded to a four-digit year.
* Finally, these components were combined into a standard date format (`YYYY-MM-01`) and converted into a proper date column.
* The original `period` column was dropped afterward.

This resulted in a clean and consistent monthly date column that could be used directly for time series analysis.

---

### Inflation Adjustment

After cleaning the raw data, we decided to additionally investigate whether changes in import and export values were influenced by **inflation effects**.

To do this, the Euro-based values were adjusted using official Austrian inflation data
(source: WKO inflation overview [https://www.wko.at/statistik/prognose/inflation.pdf]).

The inflation adjustment was applied cumulatively, meaning that each year includes the inflation effects of all previous years. This allows for a more meaningful comparison of monetary values over time and helps distinguish between real growth and price-level effects.

### Autocorrelation

### Seasonality - Time series diagnostics

### Visualisation

The following figures illustrate the development of import and export quantities as well as inflation-adjusted values over time.

![](img/import-export_quantities+inflation-adj_val.png)

![](img/real_inflation-adj_val-combarison.png)

To avoid misinterpretation, it is important to clarify that **“Real 2025 EUR”** refers to the **price level** in which all values are expressed, not the time period used to calculate inflation.

To obtain inflation-adjusted values, we constructed a **CPI-like index** by compounding the officially published annual inflation rates for the years **2007–2025**. Using this index, each nominal value was rescaled according to:

***Real Value_2025 = Nominal Value_year × (CPI_2025 / CPI_year)***

This means that a value originally observed in, for example, **2007** is adjusted to reflect the same purchasing power as in **2025**. Consequently, all inflation-adjusted monetary values are directly comparable across time and are expressed in **2025 euros**, which explains why the corresponding columns are suffixed with `_real_2025`.

### Preprocessing thoughts

Joa 

## Predictions
In this section, we generate and evaluate forecasts for a single time series, focusing on monthly import values (in EUR). The goal is to compare simple baseline models against more advanced statistical forecasting methods, evaluate their out-of-sample performance, and produce short-term forecasts beyond the last available observation.

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
