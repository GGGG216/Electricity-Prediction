# Hong Kong Weekly Load Data Workflow

This repo now contains a working data-collection skeleton for your group project on forecasting Hong Kong weekly electricity demand curves.

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
  - A small CLI that can download:
    - HKO historical weather CSVs
    - 1823 public holidays JSON
    - Immigration daily passenger traffic
    - Transport Department raw detector XML and monthly digest CSVs
    - C&SD energy tables through the official JSON API
    - ArcGIS-hosted spatial layers such as district population projections, EV chargers, and building footprints
- `data/manual_templates/`
  - Templates for the load label that still needs to be supplied manually.

## Quick Start

Install the only runtime dependency:

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

Smoke-test ArcGIS sources without pulling the full large layers:

```bash
python -m hk_power_data collect --source district_population_projection --source ev_public_chargers --source building_footprints --max-records 25
```

Pull the full building footprint layer only when you are ready for a large download:

```bash
python -m hk_power_data collect --source building_footprints
```

## Recommended Project Workflow

### Phase 1: Lock the label

Put your target load file into the schema in `data/manual_templates/city_load_hourly_template.csv`.

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

After download, the next ETL step should normalize each source into these canonical keys:

- `ts_utc` or `date_local`
- `zone_id`
- `source_name`

Then derive:

- lag features: `lag_1h`, `lag_24h`, `lag_168h`
- rolling features: `mean_24h`, `max_168h`, `std_168h`
- calendar features: hour-of-week, holiday flags
- zone summaries: district-to-zone rollups for population, EV chargers, building area

## Notes

- Collected raw files are written to `data/raw/<source_name>/`.
- Each download writes a sidecar metadata file and a run manifest under `data/raw/_runs/`.
- `building_footprints` is intentionally marked `P2` because it is large.
- C&SD API calls use an explicit `Referer` header because the endpoint rejects anonymous bot-like requests otherwise.

## Official Sources Used In This Workflow

- HKO daily climate downloads
- 1823 public holidays JSON
- Immigration Department daily passenger traffic CSV
- Transport Department strategic road detector feed and monthly digest CSVs
- C&SD web-table API for electricity statistics
- HK Electric renewable generation CSV
- ArcGIS-hosted public geospatial layers for district population, EV chargers, and building footprints
