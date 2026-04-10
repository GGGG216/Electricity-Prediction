# Monthly Workflow

This document explains the monthly city-level forecasting workflow used in this repository.

## 1. Problem Definition

The current project target is:

- spatial grain: whole Hong Kong
- time grain: monthly
- target: monthly local electricity consumption
- horizon: next month

This is a better fit for the currently available official public data than hourly or weekly load-curve forecasting.

## 2. Data Layers

The workflow uses three layers:

1. `raw`
   - official source files as downloaded
2. `silver`
   - cleaned monthly or static tables with stable keys
3. `model`
   - joined train-ready monthly forecasting tables

## 3. Target Construction

The supervised target comes from C&SD table `915-91201`:

- monthly electricity and gas consumption by type of users

The model uses the monthly `ELEC_LOCAL` total as the main prediction target:

- `electricity_total_tj`

Additional user-type monthly series are kept as contextual explanatory features:

- `electricity_domestic_tj`
- `electricity_commercial_tj`
- `electricity_industrial_tj`
- `electricity_street_lighting_tj`

## 4. Feature Engineering Logic

### Weather

Daily HKO variables are aggregated to month:

- average mean temperature
- average and peak maximum temperature
- average and minimum minimum temperature
- average daily temperature range
- average humidity
- monthly rainfall total
- rainy-day count
- hot-day count
- cool-day count

### Calendar

Daily holiday information is rolled up to month:

- days in month
- weekend days
- public holiday days
- business days

### Mobility

Daily immigration data is aggregated to monthly totals:

- arrivals and departures by visitor type
- total cross-border passengers
- net arrivals

### Transport

Monthly public transport and cross-harbour traffic are used directly:

- total average daily public transport passengers
- mode-level passenger proxies
- cross-harbour passenger totals

### Building activity

Monthly Buildings Department tables are used directly:

- consents to commence work
- occupation permits issued
- completion gross floor area and domestic units

### Spatial and structural features

Slow-moving or static city features are added to every month:

- city population
- population density
- building footprint snapshot kept as reference context

### Time-series history

The pipeline derives monthly autoregressive features:

- `electricity_lag_1m`
- `electricity_lag_2m`
- `electricity_lag_3m`
- `electricity_lag_6m`
- `electricity_lag_12m`
- rolling means and rolling standard deviations over `3m`, `6m`, and `12m`
- year-over-year delta and growth

The target column is:

- `target_electricity_total_t_plus_1m`

## 5. Alignment Choices

Different source cadences are aligned as follows:

- daily sources -> aggregated to month
- monthly sources -> kept at month level
- annual sources -> forward-filled by year through `feature_year`
- static sources -> broadcast to all monthly rows

The train-ready modeling window starts at `2013-01` because:

- public transport monthly data starts in 2013
- the full set of monthly exogenous features is more stable from that point onward

## 6. Error Handling in the ETL

The current ETL explicitly handles several real-world issues:

1. Mixed period formats in official statistics
   - annual, quarterly, and monthly periods coexist in C&SD tables
   - the ETL parses only valid monthly periods for the target table
2. Cross-source start-date mismatch
   - target starts much earlier than immigration, transport, and population
   - the model table keeps availability flags and uses a modern train-ready window
3. File-format mismatch in HK Electric renewable generation
   - the downloaded file is named like a CSV but is actually an Excel workbook
   - the ETL checks the file signature and switches reader accordingly
4. Annual population projections with limited year range
   - the ETL clips the year to the available projection range before merging

## 7. Main Commands

Collect:

```bash
python -m hk_power_data collect --group core_exogenous --group mobility --group energy_proxies --group spatial_static
```

Build silver and model tables:

```bash
python -m hk_power_data silver
```

Run baseline training:

```bash
python -m hk_power_data train
python -m hk_power_data evaluate
```

Run interpretable regression validation:

```bash
python -m hk_power_data validate-regression
python -m hk_power_data report-regression
```

## 8. Result Packaging

The final human-readable outputs for the project are collected under:

- `result/`

This folder contains:

- dataset analysis
- error handling notes
- regression result summary
- generated visualizations
