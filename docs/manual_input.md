# Manual Input Specification

This project can automate the public exogenous variables, but it still needs one critical manual input:

- `data/manual/city_load_hourly.csv`

This is the supervised learning label for the forecasting task.

## 1. What This File Represents

Each row is one observed hourly electricity demand value for one zone.

Minimum supported zones:

- `HK_TOTAL`
- `HK_ISLAND`
- `KOWLOON`
- `NEW_TERRITORIES`

If you only have `HK_TOTAL`, the pipeline can still run, but you will only be able to train a city-total model.

## 2. Required File Format

File name:

- `data/manual/city_load_hourly.csv`

Encoding:

- UTF-8 or UTF-8 with BOM

Delimiter:

- comma-separated CSV

Required columns:

- `ts_utc`
- `ts_local`
- `zone_id`
- `load_value`
- `load_unit`
- `measure_type`
- `quality_flag`
- `source_url`

Use the template:

- `data/manual_templates/city_load_hourly_template.csv`

## 3. Required Column Meanings

### `ts_utc`

- Type: timestamp string
- Meaning: the hour in UTC
- Example: `2025-01-06T00:00:00Z`

This is the canonical time key used in the model layer.

### `ts_local`

- Type: timestamp string
- Meaning: the same hour in Hong Kong local time
- Time zone: HKT, which is UTC+8
- Example: `2025-01-06 08:00:00`

This is used to derive:

- hour of day
- hour of week
- weekday
- holiday alignment

### `zone_id`

- Type: string
- Allowed values:
  - `HK_TOTAL`
  - `HK_ISLAND`
  - `KOWLOON`
  - `NEW_TERRITORIES`

### `load_value`

- Type: numeric
- Meaning: observed load for that hour
- Example: `6800.5`

### `load_unit`

- Type: string
- Recommended values:
  - `MW`
  - `kW`
  - `kWh`

For this project, use `MW` whenever the source is system demand or zone demand.

Be consistent across the file.

### `measure_type`

- Type: string
- Recommended values:
  - `demand`
  - `energy`

For this project, you almost always want:

- `demand`

because the task is forecasting the load curve, not accumulated hourly energy.

### `quality_flag`

- Type: string
- Recommended values:
  - `ok`
  - `estimated`
  - `missing`
  - `partial`
  - `anomaly`

This gives you traceability later when you decide whether to:

- drop low-quality rows
- keep them with filtering
- mark them in EDA

### `source_url`

- Type: string
- Meaning: where this label came from
- Example:
  - utility portal export link
  - internal dataset reference
  - source landing page URL

If the data comes from a private file, put a descriptive source string such as:

- `private_partner_dataset_2024q4_export`

## 4. Example

```csv
ts_utc,ts_local,zone_id,load_value,load_unit,measure_type,quality_flag,source_url
2025-01-06T00:00:00Z,2025-01-06 08:00:00,HK_TOTAL,6800.5,MW,demand,ok,https://example.com/source
2025-01-06T00:00:00Z,2025-01-06 08:00:00,HK_ISLAND,1050.2,MW,demand,ok,https://example.com/source
2025-01-06T00:00:00Z,2025-01-06 08:00:00,KOWLOON,2140.3,MW,demand,ok,https://example.com/source
2025-01-06T00:00:00Z,2025-01-06 08:00:00,NEW_TERRITORIES,3610.0,MW,demand,ok,https://example.com/source
```

## 5. Practical Data Requirements

For useful week-ahead modeling, aim for at least:

- 6 months of hourly data for one zone

Better:

- 1 to 2 years of hourly data for all 4 zones

Why:

- `lag_168h` needs at least one full prior week
- rolling features become much more reliable with longer history
- seasonality and holiday patterns only show up with enough time span

## 6. Quality Checks Before You Drop The File In

Before placing the file in `data/manual/`, check:

1. Every hour exists at the intended cadence.
2. `ts_utc` and `ts_local` match exactly by UTC+8.
3. Units are consistent.
4. `zone_id` values are spelled exactly as expected.
5. There are no duplicate rows for `(ts_utc, zone_id)`.
6. Missing rows are explicitly marked rather than silently deleted.

## 7. Optional Manual Inputs

These are optional, not required by the current pipeline:

- HKUST Dryad smart-meter raw files for a benchmark task
- private utility metadata about substations or feeder-zone mapping
- private tariff or outage annotations

Those can be added later, but the only essential manual file for the city-level forecasting workflow is:

- `data/manual/city_load_hourly.csv`

