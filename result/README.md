# Result Package

## Project Scope

This result package summarizes the monthly Hong Kong electricity forecasting workflow built from official public data.

## Dataset Overview

| Item | Value |
|---|---:|
| train_ready_rows | 157 |
| train_ready_columns | 96 |
| start_month | 2013-01-01 |
| end_month | 2026-01-01 |
| target_mean_tj | 13348.172239490446 |
| target_std_tj | 2587.146336058456 |
| target_min_tj | 9396.0 |
| target_max_tj | 17948.0 |

## Silver Build Summary

| Table | Rows |
|---|---:|
| weather_monthly | 1622 |
| calendar_monthly | 1716 |
| immigration_city_monthly | 63 |
| transport_public_mode_monthly | 2638 |
| transport_public_total_monthly | 157 |
| transport_public_mode_wide_monthly | 157 |
| transport_cross_harbour_monthly | 157 |
| building_consent_monthly | 176 |
| occupation_permits_monthly | 176 |
| building_completion_monthly | 176 |
| building_activity_monthly | 176 |
| censtatd_energy_use_monthly | 6130 |
| censtatd_local_consumption_revenue_quarterly | 1134 |
| censtatd_peak_demand_annual | 96 |
| censtatd_energy_stats_tidy | 7360 |
| city_energy_monthly_detail | 5660 |
| city_energy_monthly | 566 |
| hkelectric_re_generation_monthly | 61 |
| city_population_yearly | 11 |
| ev_city_static | 1 |
| building_footprints_city_static | 1 |
| city_monthly_model_features | 566 |
| city_monthly_train_ready | 157 |

## Baseline Model Summary

| Model | MAE | RMSE | R2 | MAPE |
|---|---:|---:|---:|---:|
| Naive Same Month Last Year | 1210.57 | 1396.17 | 0.7054 | 0.0895 |
| Ridge | 1143.67 | 1728.64 | 0.5484 | 0.0861 |
| Random Forest | 454.19 | 614.60 | 0.9429 | 0.0327 |

## Regression Summary

| Model | MAE | RMSE | R2 | MAPE |
|---|---:|---:|---:|---:|
| Simple linear regression | 1155.41 | 1346.22 | 0.7261 | 0.0871 |
| Multiple linear regression | 684.98 | 857.50 | 0.8889 | 0.0507 |

## Interpretation

- The simple regression uses `electricity_lag_12m` and captures annual seasonality reasonably well.
- The multiple regression improves MAE by 470.43 TJ and raises R2 from 0.7261 to 0.8889.
- This indicates that monthly weather, mobility, transport, building activity, and seasonal features contribute meaningful signal beyond a single lag baseline.

## Top Correlated Features

| Feature | Correlation |
|---|---:|
| `cos_month` | -0.9441 |
| `electricity_industrial_tj` | 0.8324 |
| `electricity_commercial_tj` | 0.8296 |
| `electricity_total_tj` | 0.8182 |
| `temp_mean_c_avg` | 0.8182 |
| `electricity_lag_12m` | 0.8174 |
| `temp_max_c_avg` | 0.8159 |
| `temp_min_c_avg` | 0.8158 |
| `temp_max_c_peak` | 0.8142 |
| `hot_day_count` | 0.8054 |
| `electricity_lag_6m` | -0.7989 |
| `temp_min_c_low` | 0.7915 |

## Errors Encountered and Handling

| Issue | Handling |
|---|---|
| Mixed time granularity across sources | Daily sources are aggregated to month, annual sources are merged through feature_year, and static sources are broadcast to all months. |
| Target and feature coverage start at different dates | The train-ready monthly window starts at 2013-01 and availability flags are preserved for immigration, transport, renewable generation, and population features. |
| HK Electric renewable file is mislabeled as CSV | The ETL checks the file signature and switches to read_excel when the payload is actually an XLSX workbook. |
| C&SD period field mixes annual, quarterly, and monthly values | The ETL parses the period string explicitly and only keeps valid monthly rows for the city-level target table. |
| Population projection is annual and limited in time range | The merge clips feature_year to the available projection range before joining the monthly model table. |

## Visualizations

- `figures/monthly_target_overview.png`
- `figures/feature_availability.png`
- `figures/top_correlations.png`
- `figures/baseline_metric_comparison.png`
- `figures/baseline_test_window.png`
- `figures/actual_vs_pred.png`
- `figures/metric_comparison.png`
- `figures/multiple_coefficients.png`
