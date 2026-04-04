# Hong Kong Monthly Electricity Forecasting Workflow

This repository now uses a **monthly city-level forecasting** setup for Hong Kong electricity consumption.

The project focus is:

1. Collect official public exogenous data for Hong Kong.
2. Use the official C&SD monthly electricity table as the supervised target.
3. Build a monthly train-ready dataset for interpretable and reproducible forecasting.

## Main Outputs

- [config/sources.json](config/sources.json)
  - official public source catalog
- [hk_power_data/cli.py](hk_power_data/cli.py)
  - CLI for collection, transformation, regression validation, reporting, and evaluation
- [hk_power_data/transform.py](hk_power_data/transform.py)
  - monthly ETL and feature engineering
- [hk_power_data/regression_validation.py](hk_power_data/regression_validation.py)
  - simple and multiple linear regression baselines
- [hk_power_data/regression_report.py](hk_power_data/regression_report.py)
  - markdown report and plots for regression results
- [docs/workflow.md](docs/workflow.md)
  - monthly workflow explanation
- [result](result)
  - final result package with analysis, error handling, and figures

## Monthly Target

The supervised target is derived from the official C&SD table:

- [Monthly statistics on consumption of electricity and gas by type of users](https://data.gov.hk/en-data/dataset/hk-censtatd-tablechart-915-91201)

The project uses the monthly `ELEC_LOCAL` series and builds:

- `electricity_total_tj`
- `electricity_domestic_tj`
- `electricity_commercial_tj`
- `electricity_industrial_tj`
- `electricity_street_lighting_tj`

## Feature Families

Dynamic monthly features:

- weather aggregates from HKO daily history
- public holiday and business-day counts
- monthly immigration aggregates
- monthly public transport demand
- monthly cross-harbour passenger flow
- HK Electric renewable generation
- lag and rolling statistics of electricity consumption

Static or slow-moving features:

- city population and density
- EV charger stock
- building footprint summary

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Collect the public data:

```bash
python -m hk_power_data collect --group core_exogenous --group mobility --group energy_proxies --group spatial_static
```

Build the monthly silver and model tables:

```bash
python -m hk_power_data silver
```

Run the main monthly baselines:

```bash
python -m hk_power_data train
python -m hk_power_data evaluate
```

Run the simple and multiple linear regression validation:

```bash
python -m hk_power_data validate-regression
python -m hk_power_data report-regression
```

Build the final result package:

```bash
python -m hk_power_data build-results
```

## Core Generated Tables

Silver layer:

- `data/silver/weather_monthly.csv`
- `data/silver/calendar_monthly.csv`
- `data/silver/immigration_city_monthly.csv`
- `data/silver/transport_public_total_monthly.csv`
- `data/silver/transport_public_mode_wide_monthly.csv`
- `data/silver/transport_cross_harbour_monthly.csv`
- `data/silver/city_energy_monthly.csv`
- `data/silver/hkelectric_re_generation_monthly.csv`
- `data/silver/city_population_yearly.csv`
- `data/silver/ev_city_static.csv`
- `data/silver/building_footprints_city_static.csv`

Model layer:

- `data/model/city_monthly_model_features.csv`
- `data/model/city_monthly_train_ready.csv`

Regression validation:

- `data/model/city_monthly_regression_validation/metrics.json`
- `data/model/city_monthly_regression_validation/predictions.csv`
- `data/model/city_monthly_regression_validation/report.md`

## Current Monthly Regression Result

Using the current official monthly dataset and chronological split:

- simple linear regression with `electricity_lag_12m`
  - MAE `1155.41`
  - RMSE `1346.22`
  - R2 `0.7261`
- multiple linear regression with 13 monthly features
  - MAE `784.25`
  - RMSE `944.58`
  - R2 `0.8652`

This shows that monthly exogenous features add clear predictive value beyond a single seasonal lag.
