# Monthly Regression Validation

## Setup

- Dataset: `D:\= =\3544\project\data\model\city_monthly_train_ready.csv`
- Train rows: 125
- Test rows: 32
- Test window: 2023-06 to 2026-01
- Target: `target_electricity_total_t_plus_1m`

## Models

- Simple linear regression: one feature `electricity_lag_12m`
- Multiple linear regression: 18 features

## Metrics

| Model | MAE | RMSE | R2 | MAPE |
|---|---:|---:|---:|---:|
| Simple linear regression | 1155.41 | 1346.22 | 0.7261 | 0.0871 |
| Multiple linear regression | 684.98 | 857.50 | 0.8889 | 0.0507 |

## Interpretation

- The simple model already works reasonably well because `electricity_lag_12m` captures strong annual seasonality in Hong Kong monthly electricity consumption.
- The multiple model improves MAE from 1155.41 to 684.98 and improves R2 from 0.7261 to 0.8889.
- This means monthly weather, mobility, transport, and seasonality features add useful signal beyond a single seasonal lag.

## Most Influential Multiple-Regression Coefficients

| Feature | Coefficient |
|---|---:|
| `cos_month` | -3298.2077 |
| `sin_month` | -1183.1565 |
| `temp_max_c_peak` | -695.5468 |
| `cross_harbour_total_pax` | -459.2486 |
| `electricity_lag_12m` | -441.8400 |
| `public_transport_total_avg_daily_pax` | 343.2534 |
| `gas_total_tj` | 117.9301 |
| `temp_mean_c_avg` | 100.2369 |

## Caveats

- These are time-series regressions on a relatively small monthly test set, so coefficients should be interpreted cautiously.
- Linear regression is useful here as an interpretable baseline, not necessarily as the final best-performing model.
- The next step is to compare this against seasonal-naive and regularized models in the same monthly setup.

## Artifacts

- `actual_vs_pred.png`: test-window actual vs predicted monthly electricity
- `metric_comparison.png`: MAE / RMSE / R2 comparison
- `multiple_coefficients.png`: coefficient magnitudes for the multiple linear regression
