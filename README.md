# Hong Kong Weekly Load Data Workflow

This repo now contains a working end-to-end workflow for your group project on forecasting Hong Kong weekly electricity demand curves.

The key design choice is simple:

1. Automate every official exogenous source that is already public.
2. Separate the true target label (`city_load_hourly`) into a manual input, because public hourly zone-level load for `HK_TOTAL / HK_ISLAND / KOWLOON / NEW_TERRITORIES` is not cleanly exposed through an official open API.
3. Keep an optional HKUST smart-meter path as a benchmark so you can validate the modeling stack before the city-level label is ready.

## What Is Included

- `config/sources.json`
  - Source catalog mapped from your PDF into `core_exogenous`, `mobility`, `energy_proxies`, `spatial_static`, and `manual_targets`.
- `config/district_to_zone.csv`
  - A practical crosswalk from district names to your 3-zone setup, including a canonical name column for inconsistent spellings such as `Central & Western` vs `Central and Western`.
- `hk_power_data/`
  - A CLI that can:
    - collect official raw data
    - transform raw data into silver tables
    - build a train-ready weekly forecasting table once the real hourly label is provided
    - run baseline training and evaluation
- `docs/workflow.md`
  - Detailed explanation of the workflow, feature meaning, and train/test logic.
- `data/manual/`
  - Drop zone for the real target label file. This folder is git-ignored for CSVs so you do not accidentally publish private labels.
- `data/manual_templates/`
  - Templates for the load label that still needs to be supplied manually.

The collectors can download:
    - HKO historical weather CSVs
    - 1823 public holidays JSON
    - Immigration daily passenger traffic
    - Transport Department raw detector XML and monthly digest CSVs
    - C&SD energy tables through the official JSON API
    - ArcGIS-hosted spatial layers such as district population projections, EV chargers, and building footprints

## Quick Start

Install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

List the available sources:

```bash
python -m hk_power_data list
python -m hk_power_data list --include-manual
```

Dry-run the default selection:

```bash
python -m hk_power_data collect --dry-run
```

By default, `collect` pulls the `core_exogenous` group only. This keeps the first run light.

Collect the core exogenous data:

```bash
python -m hk_power_data collect
```

Collect mobility and energy proxies as well:

```bash
python -m hk_power_data collect --group mobility --group energy_proxies
```

Collect a mixed set using both explicit sources and groups:

```bash
python -m hk_power_data collect --group core_exogenous --group mobility --source district_population_projection --source ev_public_chargers
```

Smoke-test ArcGIS sources without pulling the full large layers:

```bash
python -m hk_power_data collect --source district_population_projection --source ev_public_chargers --source building_footprints --max-records 25
```

Pull the full building footprint layer only when you are ready for a large download:

```bash
python -m hk_power_data collect --source building_footprints
```

Build the silver layer:

```bash
python -m hk_power_data silver
```

Train the baseline models once the real label file exists:

```bash
python -m hk_power_data train
```

## Recommended Project Workflow

### Phase 1: Lock the label

Put your target load file into:

- `data/manual/city_load_hourly.csv`

Use the schema in:

- `data/manual_templates/city_load_hourly_template.csv`

Required columns:

- `ts_utc`
- `ts_local`
- `zone_id`
- `load_value`
- `load_unit`
- `measure_type`
- `quality_flag`
- `source_url`

If the city-level label is delayed, use the HKUST Dryad dataset as a benchmark track so the modeling pipeline still advances.

### Phase 2: Collect exogenous drivers

Run:

```bash
python -m hk_power_data collect --group core_exogenous --group mobility --group energy_proxies
```

This gives you:

- Weather
- Holiday calendar
- Cross-border passenger flow
- Road traffic and transport demand proxies
- Electricity-sector proxy tables such as monthly consumption and annual peak demand

### Phase 3: Collect spatial covariates

Run:

```bash
python -m hk_power_data collect --group spatial_static
```

Use these for:

- District population
- EV charger density
- Building stock / footprint area

Then aggregate district-level features into your three forecasting zones with `config/district_to_zone.csv`.

### Phase 4: Build the silver tables

Run:

```bash
python -m hk_power_data silver
```

This creates normalized source tables in `data/silver/`.

Core canonical keys are:

- `ts_utc` or `date_local`
- `zone_id`
- `source_name`

The silver step also standardizes:

- date parsing
- units and numeric columns
- district name harmonization
- city-level mobility aggregates
- zone-level static features

Then, once the label is present, it derives:

- lag features: `lag_1h`, `lag_24h`, `lag_168h`
- rolling features: `mean_24h`, `std_24h`, `mean_168h`, `std_168h`
- calendar features: hour-of-week, holiday flags
- zone summaries: district-to-zone rollups for population and EV chargers
- future labels such as `target_load_t_plus_168h`

### Phase 5: Train and evaluate

After `data/manual/city_load_hourly.csv` is available and `silver` has been rebuilt, the pipeline writes:

- `data/model/hourly_zone_features.csv`
- `data/model/train_ready_direct_168h.csv`

Then run:

```bash
python -m hk_power_data train
```

This performs:

- chronological train/test split
- naive same-hour-last-week baseline
- ridge regression
- random forest

And writes:

- `data/model/training_runs/metrics.json`
- `data/model/training_runs/predictions.csv`

## Notes

- Collected raw files are written to `data/raw/<source_name>/`.
- Each download writes a sidecar metadata file and a run manifest under `data/raw/_runs/`.
- `building_footprints` is intentionally marked `P2` because it is large.
- `building_footprints_silver.csv` reflects whatever raw building layer you have collected. If your latest raw file came from a smoke test with `--max-records`, that silver table will also be partial.
- C&SD API calls use an explicit `Referer` header because the endpoint rejects anonymous bot-like requests otherwise.
- `data/model/` and `data/manual/*.csv` are git-ignored, because they are either generated artifacts or likely to contain sensitive/private labels.
- The current model workflow is designed for a **direct 168-hour forecasting setup**. Each row at time `t` predicts the load at `t + 168h`.

## Recommended Reading

- Full workflow explanation: `docs/workflow.md`

## Official Sources Used In This Workflow

- HKO daily climate downloads
- 1823 public holidays JSON
- Immigration Department daily passenger traffic CSV
- Transport Department strategic road detector feed and monthly digest CSVs
- C&SD web-table API for electricity statistics
- HK Electric renewable generation CSV
- ArcGIS-hosted public geospatial layers for district population, EV chargers, and building footprints
