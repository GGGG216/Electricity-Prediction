# End-to-End Workflow

This document explains the full workflow for the Hong Kong weekly load forecasting dataset in plain project language.

## 1. Problem Framing

The project target is not just a scalar electricity number. It is a **weekly load curve**, which means:

- Forecast horizon: the next 168 hours
- Spatial grain: `HK_TOTAL`, `HK_ISLAND`, `KOWLOON`, `NEW_TERRITORIES`
- Time grain: hourly

That leads to three data layers:

1. `raw`
   - original files as downloaded from official sources
2. `silver`
   - cleaned, standardized, analysis-ready tables with stable keys and units
3. `model`
   - joined feature tables that are directly consumable by machine-learning models

## 2. Why The Workflow Is Split This Way

Public Hong Kong data gives you many useful **drivers** of demand:

- weather
- holidays
- passenger flow
- transport demand proxies
- static spatial context such as population and EV chargers

But it does **not** cleanly expose the target label you actually need for this project:

- hourly demand by `HK_TOTAL / HK_ISLAND / KOWLOON / NEW_TERRITORIES`

So the workflow intentionally separates:

- automated collection for public exogenous variables
- manual drop-in for the supervised learning label

This lets you keep moving on ETL and feature engineering even before the true label is finalized.

## 3. Raw Layer

Raw files are collected with:

```bash
python -m hk_power_data collect
python -m hk_power_data collect --group mobility --group energy_proxies
python -m hk_power_data collect --group spatial_static
```

They are stored under:

- `data/raw/<source_name>/`

Each download also writes metadata and run manifests so you can trace:

- when a file was collected
- where it came from
- which source URL produced it

## 4. Silver Layer

Build silver tables with:

```bash
python -m hk_power_data silver
```

This creates standardized tables in `data/silver/`.

### Key silver outputs

- `weather_daily.csv`
  - one row per day
  - cleaned HKO weather features such as mean/max/min temperature, humidity, rainfall
- `holiday_events.csv`
  - official public-holiday event dates
- `calendar_daily.csv`
  - derived calendar features such as weekday, weekend, week-of-year, holiday flag
- `immigration_control_point_daily.csv`
  - daily passenger traffic at each control point and direction
- `immigration_city_daily.csv`
  - city-level daily arrivals and departures, aggregated from control points
- `transport_public_mode_monthly.csv`
  - monthly transport activity by mode
- `censtatd_energy_stats_tidy.csv`
  - tidy low-frequency electricity statistics from C&SD
- `zone_population_yearly.csv`
  - zone-level annual population and density derived from district projections
- `ev_chargers_zone_static.csv`
  - zone-level EV charger counts derived from district-level point data
- `building_footprints_silver.csv`
  - cleaned building footprint attributes

### Why silver matters

Silver tables remove source-specific quirks:

- mixed date formats
- inconsistent district naming
- nested JSON
- duplicated units and verbose headers

After this step, your joins become simple and reproducible.

## 5. Model Layer

The model layer only appears after you place a real label file at:

- `data/manual/city_load_hourly.csv`

Expected schema:

- `ts_utc`
- `ts_local`
- `zone_id`
- `load_value`
- `load_unit`
- `measure_type`
- `quality_flag`
- `source_url`

After that, run:

```bash
python -m hk_power_data silver
```

Then the pipeline will also create:

- `data/model/hourly_zone_features.csv`
- `data/model/train_ready_direct_168h.csv`

### `hourly_zone_features.csv`

This is the full joined hourly table. Each row is a known observation time `t` for one zone.

It includes:

- current load
- time features
- weather on day `t`
- holiday flags
- previous-day immigration aggregates
- static zone context
- lag features
- rolling statistics
- future target columns

### `train_ready_direct_168h.csv`

This is the filtered version for direct week-ahead supervised learning.

Each row means:

- features are available at time `t`
- the label is `load_value` at `t + 168h`

So if you train on `target_load_t_plus_168h`, you are solving a legitimate direct week-ahead problem.

## 6. Meaning Of The Main Features

### Time features

- `hour_of_day`, `hour_of_week`
  - capture daily and weekly periodicity
- `week_of_year`, `month`, `quarter`
  - capture seasonal effects
- `sin_*` and `cos_*`
  - encode cyclic time smoothly for linear models

### Load history features

- `load_lag_1h`
  - very short-term inertia
- `load_lag_24h`
  - same hour yesterday
- `load_lag_168h`
  - same hour last week
- `load_roll_mean_*`, `load_roll_std_*`
  - recent baseline and volatility

These are usually the strongest predictors in load forecasting.

### Weather features

- `temp_mean_c`, `temp_max_c`, `temp_min_c`
  - proxies for cooling demand
- `rh_mean_pct`
  - humidity effect
- `rainfall_mm`
  - weather regime and mobility suppression effects

### Mobility features

- `prev_day_arrival_total_passengers`
- `prev_day_departure_total_passengers`
- resident/mainland/other splits

These act as demand-activity proxies. The pipeline uses **previous-day** values to avoid obvious timing leakage.

### Static zone features

- `population`
- `population_density_per_km2`
- EV charger counts

These represent structural demand capacity differences across zones.

## 7. What To Use For Training

For the course project, start with `data/model/train_ready_direct_168h.csv`.

Target:

- `target_load_t_plus_168h`

Recommended feature families:

- load history
- time/calendar
- weather
- mobility
- zone static features

Do not start with everything blindly. Build in stages:

1. naive baseline
2. load-history-only model
3. load-history + calendar
4. load-history + calendar + weather
5. full feature set

This gives you a clean ablation story for your final presentation.

## 8. How To Train And Test

Use the built-in training runner:

```bash
python -m hk_power_data train
```

It expects:

- dataset: `data/model/train_ready_direct_168h.csv`
- target: `target_load_t_plus_168h`
- split: chronological hold-out

The script trains:

- naive same-hour-last-week baseline
- ridge regression
- random forest

And writes:

- `data/model/training_runs/metrics.json`
- `data/model/training_runs/predictions.csv`

### Why chronological split matters

Never randomly shuffle this dataset.

Electricity demand is a time series. If future rows leak into training, evaluation becomes unrealistically optimistic.

The repo therefore splits by timestamp order, not random rows.

## 9. Recommended Evaluation Strategy For The Course

Use three levels of evaluation:

1. overall metrics
   - MAE
   - RMSE
   - R²
2. per-zone metrics
   - which zones are harder
3. curve-level inspection
   - compare predicted vs actual 168-hour curves for several weeks

In the final report, show:

- one easy week
- one holiday week
- one abnormal weather week

That makes the model behavior interpretable.

## 10. Important Caveats

### The city-level label is still the bottleneck

Without the true hourly target, silver is complete but supervised training is not meaningful.

### Weather realism

The current workflow uses realized historical weather.
For real operational week-ahead forecasting, you would replace or augment this with weather forecasts.

### Low-frequency economic tables

The C&SD electricity tables are useful for context and exploratory analysis, but they should be used carefully in predictive models because monthly or annual aggregates can introduce leakage if aligned incorrectly.

### Building footprints are not yet zone-joined

The current building layer is normalized into silver form, but not spatially overlaid into zones yet. That is a future enhancement once you decide whether to add a proper GIS overlay step.

