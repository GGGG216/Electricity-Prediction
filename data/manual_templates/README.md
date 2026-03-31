# Manual Inputs

Use this folder for sources that are not cleanly downloadable from an official public API.

Minimum manual files for the city-level weekly load project:

1. `city_load_hourly.csv`
   - Use the template in this folder.
   - Required if you want to forecast `HK_TOTAL`, `HK_ISLAND`, `KOWLOON`, or `NEW_TERRITORIES` weekly load curves.
   - Typical upstream options are utility exports, a partner dataset, or your own scrape that you verify against the source terms.

2. `hkust_meter_data/`
   - Optional benchmark path if you want to validate the modeling stack before the city-level label is ready.
   - Store the Dryad download here and aggregate it to building or pseudo-zone level.

The automated collectors in this repo can already fetch the exogenous features and static spatial covariates around these manual labels.

