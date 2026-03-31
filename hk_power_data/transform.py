from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .catalog import ROOT
from .fetchers import write_json


RAW_ROOT = ROOT / "data" / "raw"
SILVER_ROOT = ROOT / "data" / "silver"
MODEL_ROOT = ROOT / "data" / "model"
MANUAL_ROOT = ROOT / "data" / "manual"
ZONE_CROSSWALK_PATH = ROOT / "config" / "district_to_zone.csv"

WEATHER_SOURCE_COLUMNS = {
    "hko_temp_mean_daily": "temp_mean_c",
    "hko_temp_max_daily": "temp_max_c",
    "hko_temp_min_daily": "temp_min_c",
    "hko_relative_humidity_daily": "rh_mean_pct",
    "hko_rainfall_daily": "rainfall_mm",
}

TRAINING_LAG_COLUMNS = ["load_value", "load_lag_1h", "load_lag_24h", "load_lag_168h"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _latest_payload(source_name: str, raw_root: Path = RAW_ROOT) -> Path | None:
    source_dir = raw_root / source_name
    if not source_dir.exists():
        return None
    candidates = [
        path
        for path in source_dir.iterdir()
        if path.is_file() and ".meta." not in path.name and not path.name.endswith(".meta.json")
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def _clean_column_name(name: str) -> str:
    text = str(name).strip().replace("/", "")
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def _normalize_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [_clean_column_name(column) for column in frame.columns]
    return frame


def _load_zone_crosswalk() -> pd.DataFrame:
    crosswalk = pd.read_csv(ZONE_CROSSWALK_PATH)
    crosswalk["source_district_name"] = crosswalk["source_district_name"].astype(str).str.strip()
    crosswalk["canonical_district_name"] = crosswalk["canonical_district_name"].astype(str).str.strip()
    crosswalk["zone_id"] = crosswalk["zone_id"].astype(str).str.strip()
    return crosswalk


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _parse_hko_daily(source_name: str, value_column: str) -> pd.DataFrame | None:
    payload = _latest_payload(source_name)
    if payload is None:
        return None
    frame = pd.read_csv(payload, skiprows=2, encoding="utf-8-sig")
    frame = _normalize_frame_columns(frame)
    frame["date_local"] = pd.to_datetime(
        dict(year=frame["year"], month=frame["month"], day=frame["day"]),
        errors="coerce",
    )
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    completeness_col = next((col for col in frame.columns if "completeness" in col), None)
    keep = pd.DataFrame({"date_local": frame["date_local"], value_column: frame["value"]})
    if completeness_col:
        keep[f"{value_column}_completeness"] = frame[completeness_col].astype("string")
    keep["source_file"] = payload.name
    return keep.dropna(subset=["date_local"])


def build_weather_daily(raw_root: Path = RAW_ROOT) -> pd.DataFrame | None:
    merged: pd.DataFrame | None = None
    for source_name, value_column in WEATHER_SOURCE_COLUMNS.items():
        frame = _parse_hko_daily(source_name, value_column)
        if frame is None:
            continue
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame.drop(columns=["source_file"]), on="date_local", how="outer")
    if merged is None:
        return None
    merged = merged.sort_values("date_local").reset_index(drop=True)
    return merged


def build_holiday_events(raw_root: Path = RAW_ROOT) -> pd.DataFrame | None:
    payload = _latest_payload("hk_public_holidays_json", raw_root=raw_root)
    if payload is None:
        return None
    data = json.loads(payload.read_text(encoding="utf-8"))
    events = data["vcalendar"][0]["vevent"]
    rows: list[dict[str, Any]] = []
    for event in events:
        rows.append(
            {
                "date_local": pd.to_datetime(event["dtstart"][0], format="%Y%m%d", errors="coerce"),
                "holiday_name_en": event.get("summary"),
                "holiday_uid": event.get("uid"),
            }
        )
    frame = pd.DataFrame(rows).dropna(subset=["date_local"]).sort_values("date_local").reset_index(drop=True)
    return frame


def build_calendar_daily(weather_daily: pd.DataFrame | None, holiday_events: pd.DataFrame | None) -> pd.DataFrame | None:
    min_dates = []
    max_dates = []
    if weather_daily is not None and not weather_daily.empty:
        min_dates.append(weather_daily["date_local"].min())
        max_dates.append(weather_daily["date_local"].max())
    if holiday_events is not None and not holiday_events.empty:
        min_dates.append(holiday_events["date_local"].min())
        max_dates.append(holiday_events["date_local"].max())
    if not min_dates:
        return None
    calendar = pd.DataFrame({"date_local": pd.date_range(min(min_dates), max(max_dates), freq="D")})
    calendar["day_of_week"] = calendar["date_local"].dt.dayofweek + 1
    calendar["is_weekend"] = calendar["day_of_week"].isin([6, 7]).astype(int)
    calendar["week_of_year"] = calendar["date_local"].dt.isocalendar().week.astype(int)
    calendar["month"] = calendar["date_local"].dt.month.astype(int)
    calendar["quarter"] = calendar["date_local"].dt.quarter.astype(int)
    calendar["year"] = calendar["date_local"].dt.year.astype(int)
    if holiday_events is not None:
        holidays = holiday_events.copy()
        holidays["is_public_holiday"] = 1
        calendar = calendar.merge(holidays, on="date_local", how="left")
        calendar["is_public_holiday"] = calendar["is_public_holiday"].fillna(0).astype(int)
    else:
        calendar["holiday_name_en"] = pd.NA
        calendar["holiday_uid"] = pd.NA
        calendar["is_public_holiday"] = 0
    return calendar


def build_immigration_tables(raw_root: Path = RAW_ROOT) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    payload = _latest_payload("immd_daily_passenger_traffic", raw_root=raw_root)
    if payload is None:
        return None, None
    frame = pd.read_csv(payload, encoding="utf-8-sig")
    frame = _normalize_frame_columns(frame)
    frame = frame.rename(
        columns={
            "date": "date_local",
            "control_point": "control_point_name",
            "arrival_departure": "direction",
            "hong_kong_residents": "hk_residents",
            "mainland_visitors": "mainland_visitors",
            "other_visitors": "other_visitors",
            "total": "total_passengers",
        }
    )
    frame["date_local"] = pd.to_datetime(frame["date_local"], format="%d-%m-%Y", errors="coerce")
    frame["direction"] = frame["direction"].astype("string").str.strip().str.lower()
    frame["control_point_id"] = (
        frame["control_point_name"].astype("string").str.upper().str.replace(r"[^A-Z0-9]+", "_", regex=True).str.strip("_")
    )
    for column in ["hk_residents", "mainland_visitors", "other_visitors", "total_passengers"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    detail = frame[
        [
            "date_local",
            "control_point_id",
            "control_point_name",
            "direction",
            "hk_residents",
            "mainland_visitors",
            "other_visitors",
            "total_passengers",
        ]
    ].dropna(subset=["date_local"])

    pivot = (
        detail.groupby(["date_local", "direction"], as_index=False)[
            ["hk_residents", "mainland_visitors", "other_visitors", "total_passengers"]
        ]
        .sum()
        .pivot(index="date_local", columns="direction")
    )
    if pivot.empty:
        return detail, None
    pivot.columns = [f"{direction}_{metric}" for metric, direction in pivot.columns]
    city = pivot.reset_index().sort_values("date_local").reset_index(drop=True)
    city.columns = [_clean_column_name(column) for column in city.columns]
    return detail, city


def build_transport_monthly(raw_root: Path = RAW_ROOT) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}

    table21_path = _latest_payload("td_monthly_public_transport", raw_root=raw_root)
    if table21_path is not None:
        frame = pd.read_csv(table21_path, encoding="utf-8-sig")
        frame = _normalize_frame_columns(frame)
        frame["period_month"] = pd.to_datetime(frame["yr_mth"].astype(str) + "01", format="%Y%m%d", errors="coerce")
        frame["avg_daily_pax"] = pd.to_numeric(frame["avg_daily_pax"], errors="coerce")
        outputs["transport_public_mode_monthly"] = frame
        agg = frame.groupby("period_month", as_index=False)["avg_daily_pax"].sum().rename(
            columns={"avg_daily_pax": "avg_daily_public_transport_pax"}
        )
        outputs["transport_public_total_monthly"] = agg

    table82_path = _latest_payload("td_cross_harbour_traffic", raw_root=raw_root)
    if table82_path is not None:
        frame = pd.read_csv(table82_path, encoding="utf-8-sig")
        frame = _normalize_frame_columns(frame)
        outputs["transport_cross_harbour_monthly"] = frame

    return outputs


def build_censtatd_tidy(raw_root: Path = RAW_ROOT) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for source_name in [
        "censtatd_energy_use_monthly",
        "censtatd_local_consumption_revenue_quarterly",
        "censtatd_peak_demand_annual",
    ]:
        payload = _latest_payload(source_name, raw_root=raw_root)
        if payload is None:
            continue
        obj = json.loads(payload.read_text(encoding="utf-8"))
        frame = pd.DataFrame(obj.get("dataSet", []))
        if frame.empty:
            continue
        frame = _normalize_frame_columns(frame)
        frame["table_title"] = obj.get("header", {}).get("title")
        frame["source_name"] = source_name
        outputs[source_name] = frame

    if outputs:
        combined = pd.concat(outputs.values(), ignore_index=True, sort=False)
        outputs["censtatd_energy_stats_tidy"] = combined
    return outputs


def build_hkelectric_re_generation(raw_root: Path = RAW_ROOT) -> pd.DataFrame | None:
    payload = _latest_payload("hkelectric_re_generation_by_type", raw_root=raw_root)
    if payload is None:
        return None
    with payload.open("rb") as handle:
        signature = handle.read(4)
    if signature == b"PK\x03\x04":
        frame = pd.read_excel(payload)
    else:
        frame = pd.read_csv(payload, encoding="utf-8-sig")
    frame = _normalize_frame_columns(frame)
    return frame


def build_zone_population_yearly(raw_root: Path = RAW_ROOT) -> pd.DataFrame | None:
    payload = _latest_payload("district_population_projection", raw_root=raw_root)
    if payload is None:
        return None
    data = json.loads(payload.read_text(encoding="utf-8"))
    crosswalk = _load_zone_crosswalk()[["source_district_name", "canonical_district_name", "zone_id"]]
    rows: list[dict[str, Any]] = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        district = props.get("District_Council_District")
        for key, value in props.items():
            if re.fullmatch(r"F\d{4}", str(key)):
                rows.append(
                    {
                        "district_name": district,
                        "year": int(str(key)[1:]),
                        "population": pd.to_numeric(value, errors="coerce"),
                        "district_area_m2": pd.to_numeric(props.get("Shape__Area"), errors="coerce"),
                    }
                )
    frame = pd.DataFrame(rows).dropna(subset=["district_name", "year", "population"])
    frame = frame.merge(crosswalk, left_on="district_name", right_on="source_district_name", how="left")
    zone = (
        frame.groupby(["zone_id", "year"], as_index=False)[["population", "district_area_m2"]]
        .sum()
        .sort_values(["zone_id", "year"])
        .reset_index(drop=True)
    )
    zone["population_density_per_km2"] = zone["population"] / (zone["district_area_m2"] / 1_000_000.0)
    return zone


def build_ev_zone_static(raw_root: Path = RAW_ROOT) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    payload = _latest_payload("ev_public_chargers", raw_root=raw_root)
    if payload is None:
        return None, None
    data = json.loads(payload.read_text(encoding="utf-8"))
    crosswalk = _load_zone_crosswalk()[["source_district_name", "canonical_district_name", "zone_id"]]
    rows = [feature.get("properties", {}) for feature in data.get("features", [])]
    if not rows:
        return None, None
    frame = pd.DataFrame(rows)
    frame = _normalize_frame_columns(frame)
    district_column = "name_of_district_council_distri"
    count_columns = [column for column in frame.columns if column.endswith("_no")]
    for column in count_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame["total_chargers"] = frame[count_columns].sum(axis=1)
    frame["district_name"] = frame[district_column].astype("string").str.strip()
    frame = frame.merge(crosswalk, left_on="district_name", right_on="source_district_name", how="left")

    zone = (
        frame.groupby("zone_id", as_index=False)[count_columns + ["total_chargers"]]
        .sum()
        .sort_values("zone_id")
        .reset_index(drop=True)
    )
    zone = zone.rename(
        columns={
            "standard_bs1363_no": "ev_standard_chargers",
            "medium_iec62196_no": "ev_medium_iec62196_chargers",
            "quick_chademo_no": "ev_quick_chademo_chargers",
            "quick_ccs_dc_combo_no": "ev_quick_ccs_combo_chargers",
            "quick_iec62196_no": "ev_quick_iec62196_chargers",
            "quick_gb_t20234_3_dc_no": "ev_quick_gbt_dc_chargers",
        }
    )
    return frame, zone


def build_building_footprints(raw_root: Path = RAW_ROOT) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    payload = _latest_payload("building_footprints", raw_root=raw_root)
    if payload is None:
        return None, None
    data = json.loads(payload.read_text(encoding="utf-8"))
    rows = [feature.get("properties", {}) for feature in data.get("features", [])]
    if not rows:
        return None, None
    frame = pd.DataFrame(rows)
    frame = _normalize_frame_columns(frame)
    frame["building_height_m"] = pd.to_numeric(frame.get("rooflevel"), errors="coerce") - pd.to_numeric(
        frame.get("baselevel"), errors="coerce"
    )
    frame["footprint_area_m2"] = pd.to_numeric(frame.get("shape_area"), errors="coerce")
    detail = frame[
        [
            "buildingid",
            "typeofbuildingblock",
            "buildingstatus",
            "certainty",
            "baselevel",
            "rooflevel",
            "building_height_m",
            "footprint_area_m2",
        ]
    ].rename(
        columns={
            "buildingid": "building_id",
            "typeofbuildingblock": "building_block_type",
            "buildingstatus": "building_status",
        }
    )
    summary = pd.DataFrame(
        [
            {
                "building_count": int(detail["building_id"].nunique()),
                "footprint_area_m2_sum": float(detail["footprint_area_m2"].sum(skipna=True)),
                "footprint_area_m2_mean": float(detail["footprint_area_m2"].mean(skipna=True)),
                "building_height_m_mean": float(detail["building_height_m"].mean(skipna=True)),
            }
        ]
    )
    return detail, summary


def build_city_load_hourly(manual_root: Path = MANUAL_ROOT) -> pd.DataFrame | None:
    payload = manual_root / "city_load_hourly.csv"
    if not payload.exists():
        return None
    frame = pd.read_csv(payload, encoding="utf-8-sig")
    frame = _normalize_frame_columns(frame)
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    frame["ts_local"] = pd.to_datetime(frame["ts_local"], errors="coerce")
    frame["date_local"] = frame["ts_local"].dt.normalize()
    frame["load_value"] = pd.to_numeric(frame["load_value"], errors="coerce")
    frame = frame.sort_values(["zone_id", "ts_utc"]).reset_index(drop=True)
    return frame


def _merge_zone_population(
    frame: pd.DataFrame,
    zone_population_yearly: pd.DataFrame | None,
) -> pd.DataFrame:
    if zone_population_yearly is None or zone_population_yearly.empty:
        return frame
    out = frame.copy()
    year_min = int(zone_population_yearly["year"].min())
    year_max = int(zone_population_yearly["year"].max())
    out["feature_year"] = out["ts_local"].dt.year.clip(lower=year_min, upper=year_max)
    return out.merge(
        zone_population_yearly.rename(columns={"year": "feature_year"}),
        on=["zone_id", "feature_year"],
        how="left",
    )


def _add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["zone_id", "ts_utc"]).copy()
    grouped = out.groupby("zone_id")["load_value"]
    for lag in [1, 2, 24, 168]:
        out[f"load_lag_{lag}h"] = grouped.shift(lag)
    for window in [24, 168]:
        out[f"load_roll_mean_{window}h"] = grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=max(1, window // 4)).mean()
        )
        out[f"load_roll_std_{window}h"] = grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=max(1, window // 4)).std()
        )
    for horizon in [1, 24, 168]:
        out[f"target_load_t_plus_{horizon}h"] = grouped.shift(-horizon)
    return out


def build_model_tables(
    city_load_hourly: pd.DataFrame | None,
    weather_daily: pd.DataFrame | None,
    calendar_daily: pd.DataFrame | None,
    immigration_city_daily: pd.DataFrame | None,
    zone_population_yearly: pd.DataFrame | None,
    ev_zone_static: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if city_load_hourly is None or city_load_hourly.empty:
        return None, None

    frame = city_load_hourly.copy()
    frame["hour_of_day"] = frame["ts_local"].dt.hour.astype(int)
    frame["day_of_week"] = frame["ts_local"].dt.dayofweek + 1
    frame["hour_of_week"] = (frame["day_of_week"] - 1) * 24 + frame["hour_of_day"]
    frame["week_of_year"] = frame["ts_local"].dt.isocalendar().week.astype(int)
    frame["month"] = frame["ts_local"].dt.month.astype(int)
    frame["quarter"] = frame["ts_local"].dt.quarter.astype(int)
    frame["year"] = frame["ts_local"].dt.year.astype(int)
    frame["is_weekend"] = frame["day_of_week"].isin([6, 7]).astype(int)
    frame["sin_hour_of_day"] = frame["hour_of_day"].map(lambda x: math.sin(2 * math.pi * x / 24))
    frame["cos_hour_of_day"] = frame["hour_of_day"].map(lambda x: math.cos(2 * math.pi * x / 24))
    frame["sin_hour_of_week"] = frame["hour_of_week"].map(lambda x: math.sin(2 * math.pi * x / 168))
    frame["cos_hour_of_week"] = frame["hour_of_week"].map(lambda x: math.cos(2 * math.pi * x / 168))

    if weather_daily is not None:
        frame = frame.merge(weather_daily.drop_duplicates("date_local"), on="date_local", how="left")

    if calendar_daily is not None:
        frame = frame.merge(
            calendar_daily[["date_local", "holiday_name_en", "is_public_holiday"]],
            on="date_local",
            how="left",
        )
        frame["is_public_holiday"] = frame["is_public_holiday"].fillna(0).astype(int)
    else:
        frame["holiday_name_en"] = pd.NA
        frame["is_public_holiday"] = 0

    if immigration_city_daily is not None:
        shifted = immigration_city_daily.copy()
        shifted["date_local"] = shifted["date_local"] + pd.Timedelta(days=1)
        rename = {column: f"prev_day_{column}" for column in shifted.columns if column != "date_local"}
        shifted = shifted.rename(columns=rename)
        frame = frame.merge(shifted, on="date_local", how="left")

    frame = _merge_zone_population(frame, zone_population_yearly)

    if ev_zone_static is not None:
        frame = frame.merge(ev_zone_static, on="zone_id", how="left")

    frame = _add_lag_features(frame)
    frame = frame.sort_values(["ts_utc", "zone_id"]).reset_index(drop=True)

    train_ready = frame.copy()
    required = TRAINING_LAG_COLUMNS + ["target_load_t_plus_168h"]
    train_ready = train_ready.dropna(subset=required).reset_index(drop=True)
    return frame, train_ready


def build_silver_tables(
    raw_root: Path = RAW_ROOT,
    silver_root: Path = SILVER_ROOT,
    model_root: Path = MODEL_ROOT,
    manual_root: Path = MANUAL_ROOT,
) -> dict[str, Any]:
    _ensure_dir(silver_root)
    _ensure_dir(model_root)

    outputs: dict[str, Any] = {"built": {}, "skipped": []}

    weather_daily = build_weather_daily(raw_root=raw_root)
    if weather_daily is not None:
        _write_csv(weather_daily, silver_root / "weather_daily.csv")
        outputs["built"]["weather_daily"] = len(weather_daily)
    else:
        outputs["skipped"].append("weather_daily")

    holiday_events = build_holiday_events(raw_root=raw_root)
    if holiday_events is not None:
        _write_csv(holiday_events, silver_root / "holiday_events.csv")
        outputs["built"]["holiday_events"] = len(holiday_events)
    else:
        outputs["skipped"].append("holiday_events")

    calendar_daily = build_calendar_daily(weather_daily=weather_daily, holiday_events=holiday_events)
    if calendar_daily is not None:
        _write_csv(calendar_daily, silver_root / "calendar_daily.csv")
        outputs["built"]["calendar_daily"] = len(calendar_daily)
    else:
        outputs["skipped"].append("calendar_daily")

    immigration_detail, immigration_city = build_immigration_tables(raw_root=raw_root)
    if immigration_detail is not None:
        _write_csv(immigration_detail, silver_root / "immigration_control_point_daily.csv")
        outputs["built"]["immigration_control_point_daily"] = len(immigration_detail)
    else:
        outputs["skipped"].append("immigration_control_point_daily")
    if immigration_city is not None:
        _write_csv(immigration_city, silver_root / "immigration_city_daily.csv")
        outputs["built"]["immigration_city_daily"] = len(immigration_city)
    else:
        outputs["skipped"].append("immigration_city_daily")

    transport_tables = build_transport_monthly(raw_root=raw_root)
    for name, frame in transport_tables.items():
        _write_csv(frame, silver_root / f"{name}.csv")
        outputs["built"][name] = len(frame)
    if not transport_tables:
        outputs["skipped"].append("transport_monthly")

    censtatd_tables = build_censtatd_tidy(raw_root=raw_root)
    for name, frame in censtatd_tables.items():
        _write_csv(frame, silver_root / f"{name}.csv")
        outputs["built"][name] = len(frame)
    if not censtatd_tables:
        outputs["skipped"].append("censtatd_energy_stats")

    re_generation = build_hkelectric_re_generation(raw_root=raw_root)
    if re_generation is not None:
        _write_csv(re_generation, silver_root / "hkelectric_re_generation_by_type.csv")
        outputs["built"]["hkelectric_re_generation_by_type"] = len(re_generation)
    else:
        outputs["skipped"].append("hkelectric_re_generation_by_type")

    zone_population_yearly = build_zone_population_yearly(raw_root=raw_root)
    if zone_population_yearly is not None:
        _write_csv(zone_population_yearly, silver_root / "zone_population_yearly.csv")
        outputs["built"]["zone_population_yearly"] = len(zone_population_yearly)
    else:
        outputs["skipped"].append("zone_population_yearly")

    ev_detail, ev_zone = build_ev_zone_static(raw_root=raw_root)
    if ev_detail is not None:
        _write_csv(ev_detail, silver_root / "ev_public_chargers_detail.csv")
        outputs["built"]["ev_public_chargers_detail"] = len(ev_detail)
    else:
        outputs["skipped"].append("ev_public_chargers_detail")
    if ev_zone is not None:
        _write_csv(ev_zone, silver_root / "ev_chargers_zone_static.csv")
        outputs["built"]["ev_chargers_zone_static"] = len(ev_zone)
    else:
        outputs["skipped"].append("ev_chargers_zone_static")

    building_detail, building_summary = build_building_footprints(raw_root=raw_root)
    if building_detail is not None:
        _write_csv(building_detail, silver_root / "building_footprints_silver.csv")
        outputs["built"]["building_footprints_silver"] = len(building_detail)
    else:
        outputs["skipped"].append("building_footprints_silver")
    if building_summary is not None:
        _write_csv(building_summary, silver_root / "building_footprints_global_summary.csv")
        outputs["built"]["building_footprints_global_summary"] = len(building_summary)
    else:
        outputs["skipped"].append("building_footprints_global_summary")

    city_load_hourly = build_city_load_hourly(manual_root=manual_root)
    if city_load_hourly is not None:
        _write_csv(city_load_hourly, silver_root / "city_load_hourly.csv")
        outputs["built"]["city_load_hourly"] = len(city_load_hourly)
    else:
        _remove_if_exists(silver_root / "city_load_hourly.csv")
        outputs["skipped"].append("city_load_hourly")

    hourly_features, train_ready = build_model_tables(
        city_load_hourly=city_load_hourly,
        weather_daily=weather_daily,
        calendar_daily=calendar_daily,
        immigration_city_daily=immigration_city,
        zone_population_yearly=zone_population_yearly,
        ev_zone_static=ev_zone,
    )
    if hourly_features is not None:
        _write_csv(hourly_features, model_root / "hourly_zone_features.csv")
        outputs["built"]["hourly_zone_features"] = len(hourly_features)
    else:
        _remove_if_exists(model_root / "hourly_zone_features.csv")
        outputs["skipped"].append("hourly_zone_features")
    if train_ready is not None:
        _write_csv(train_ready, model_root / "train_ready_direct_168h.csv")
        outputs["built"]["train_ready_direct_168h"] = len(train_ready)
    else:
        _remove_if_exists(model_root / "train_ready_direct_168h.csv")
        outputs["skipped"].append("train_ready_direct_168h")

    manifest_path = silver_root / "silver_manifest.json"
    write_json(manifest_path, outputs)
    return outputs
