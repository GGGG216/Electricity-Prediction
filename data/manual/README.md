# Manual Label Drop Zone

Put your real target label file here:

- `city_load_hourly.csv`

Required schema:

- `ts_utc`
- `ts_local`
- `zone_id`
- `load_value`
- `load_unit`
- `measure_type`
- `quality_flag`
- `source_url`

The repo ignores `data/manual/*.csv` on purpose so you do not accidentally push the actual label file to GitHub.

