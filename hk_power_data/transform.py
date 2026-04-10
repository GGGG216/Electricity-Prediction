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

MODEL_START_MONTH = pd.Timestamp("2013-01-01")
MONTHLY_TARGET_COL = "target_electricity_total_t_plus_1m"
MONTHLY_REQUIRED_COLUMNS = ["electricity_total_tj", "electricity_lag_1m", "electricity_lag_12m", MONTHLY_TARGET_COL]

OBSOLETE_SILVER_FILES = [
    "weather_daily.csv",
    "holiday_events.csv",
    "calendar_daily.csv",
    "immigration_control_point_daily.csv",
    "immigration_city_daily.csv",
    "zone_population_yearly.csv",
    "ev_public_chargers_detail.csv",
    "ev_chargers_zone_static.csv",
    "building_footprints_silver.csv",
    "building_footprints_global_summary.csv",
    "city_load_hourly.csv",
    "hkust_meter_metadata.csv",
    "hkust_meter_hourly.csv",
    "hkust_meter_hourly_summary.csv",
    "hkust_building_static.csv",
    "hkust_building_hourly.csv",
]

OBSOLETE_MODEL_FILES = [
    "hourly_zone_features.csv",
    "train_ready_direct_168h.csv",
    "hkust_building_hourly_features.csv",
    "hkust_building_train_ready_direct_168h.csv",
]

OBSOLETE_MODEL_DIRS = [
    "training_runs",
    "evaluation",
    "hkust_training_runs",
    "hkust_evaluation",
]


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
    text = str(name).strip().replace("/", " ")
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def _normalize_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [_clean_column_name(column) for column in frame.columns]
    return frame


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _parse_period_month(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.extract(r"(\d{6})")[0]
    return pd.to_datetime(values + "01", format="%Y%m%d", errors="coerce")


def _add_constant_features(frame: pd.DataFrame, constant_frame: pd.DataFrame | None) -> pd.DataFrame:
    if constant_frame is None or constant_frame.empty:
        return frame
    out = frame.copy()
    record = constant_frame.iloc[0].to_dict()
    for key, value in record.items():
        out[key] = value
    return out


def _cleanup_obsolete_outputs(silver_root: Path, model_root: Path) -> None:
    for name in OBSOLETE_SILVER_FILES:
        _remove_if_exists(silver_root / name)
    for name in OBSOLETE_MODEL_FILES:
        _remove_if_exists(model_root / name)
    for name in OBSOLETE_MODEL_DIRS:
        directory = model_root / name
        if directory.exists():
            for child in directory.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(directory.rglob("*"), reverse=True):
                if child.is_dir():
                    child.rmdir()
            directory.rmdir()


def _parse_hko_daily(source_name: str, value_column: str, raw_root: Path = RAW_ROOT) -> pd.DataFrame | None:
    payload = _latest_payload(source_name, raw_root=raw_root)
    if payload is None:
        return None
    frame = pd.read_csv(payload, skiprows=2, encoding="utf-8-sig")
    frame = _normalize_frame_columns(frame)
    frame["date_local"] = pd.to_datetime(
        dict(year=frame["year"], month=frame["month"], day=frame["day"]),
        errors="coerce",
    )
    frame[value_column] = pd.to_numeric(frame["value"], errors="coerce")
    keep_cols = ["date_local", value_column]
    completeness_col = next((column for column in frame.columns if "completeness" in column), None)
    if completeness_col:
        frame[f"{value_column}_completeness"] = frame[completeness_col].astype("string")
        keep_cols.append(f"{value_column}_completeness")
    keep = frame[keep_cols].dropna(subset=["date_local"]).copy()
    return keep


def build_weather_daily(raw_root: Path = RAW_ROOT) -> pd.DataFrame | None:
    merged: pd.DataFrame | None = None
    for source_name, value_column in WEATHER_SOURCE_COLUMNS.items():
        frame = _parse_hko_daily(source_name, value_column, raw_root=raw_root)
        if frame is None:
            continue
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="date_local", how="outer")
    if merged is None:
        return None
    return merged.sort_values("date_local").reset_index(drop=True)


def build_weather_monthly(weather_daily: pd.DataFrame | None) -> pd.DataFrame | None:
    if weather_daily is None or weather_daily.empty:
        return None
    frame = weather_daily.copy()
    frame["period_month"] = frame["date_local"].dt.to_period("M").dt.to_timestamp()
    frame["temp_range_c"] = frame["temp_max_c"] - frame["temp_min_c"]
    monthly = (
        frame.groupby("period_month", as_index=False)
        .agg(
            temp_mean_c_avg=("temp_mean_c", "mean"),
            temp_max_c_avg=("temp_max_c", "mean"),
            temp_max_c_peak=("temp_max_c", "max"),
            temp_min_c_avg=("temp_min_c", "mean"),
            temp_min_c_low=("temp_min_c", "min"),
            temp_range_c_avg=("temp_range_c", "mean"),
            rh_mean_pct_avg=("rh_mean_pct", "mean"),
            rainfall_mm_sum=("rainfall_mm", "sum"),
            rainfall_day_count=("rainfall_mm", lambda series: int(pd.to_numeric(series, errors="coerce").fillna(0).gt(0).sum())),
            hot_day_count=("temp_max_c", lambda series: int(pd.to_numeric(series, errors="coerce").ge(30).sum())),
            cool_day_count=("temp_min_c", lambda series: int(pd.to_numeric(series, errors="coerce").lt(12).sum())),
        )
        .sort_values("period_month")
        .reset_index(drop=True)
    )
    return monthly


def build_holiday_events(raw_root: Path = RAW_ROOT) -> pd.DataFrame | None:
    payload = _latest_payload("hk_public_holidays_json", raw_root=raw_root)
    if payload is None:
        return None
    data = json.loads(payload.read_text(encoding="utf-8"))
    events = data.get("vcalendar", [{}])[0].get("vevent", [])
    rows: list[dict[str, Any]] = []
    for event in events:
        rows.append(
            {
                "date_local": pd.to_datetime(event.get("dtstart", [None])[0], format="%Y%m%d", errors="coerce"),
                "holiday_name_en": event.get("summary"),
            }
        )
    frame = pd.DataFrame(rows).dropna(subset=["date_local"]).sort_values("date_local").reset_index(drop=True)
    return frame


def build_calendar_monthly(weather_daily: pd.DataFrame | None, holiday_events: pd.DataFrame | None) -> pd.DataFrame | None:
    min_dates: list[pd.Timestamp] = []
    max_dates: list[pd.Timestamp] = []
    if weather_daily is not None and not weather_daily.empty:
        min_dates.append(weather_daily["date_local"].min())
        max_dates.append(weather_daily["date_local"].max())
    if holiday_events is not None and not holiday_events.empty:
        min_dates.append(holiday_events["date_local"].min())
        max_dates.append(holiday_events["date_local"].max())
    if not min_dates:
        return None

    calendar = pd.DataFrame({"date_local": pd.date_range(min(min_dates), max(max_dates), freq="D")})
    calendar["period_month"] = calendar["date_local"].dt.to_period("M").dt.to_timestamp()
    calendar["is_weekend"] = calendar["date_local"].dt.dayofweek.isin([5, 6]).astype(int)
    holiday_lookup = set()
    if holiday_events is not None and not holiday_events.empty:
        holiday_lookup = set(holiday_events["date_local"].dt.normalize())
    calendar["is_public_holiday"] = calendar["date_local"].dt.normalize().isin(holiday_lookup).astype(int)
    calendar["is_business_day"] = ((calendar["is_weekend"] == 0) & (calendar["is_public_holiday"] == 0)).astype(int)
    monthly = (
        calendar.groupby("period_month", as_index=False)
        .agg(
            days_in_month=("date_local", "size"),
            weekend_days=("is_weekend", "sum"),
            public_holiday_days=("is_public_holiday", "sum"),
            business_days=("is_business_day", "sum"),
        )
        .sort_values("period_month")
        .reset_index(drop=True)
    )
    return monthly


def build_immigration_monthly(raw_root: Path = RAW_ROOT) -> pd.DataFrame | None:
    payload = _latest_payload("immd_daily_passenger_traffic", raw_root=raw_root)
    if payload is None:
        return None
    frame = pd.read_csv(payload, encoding="utf-8-sig")
    frame = _normalize_frame_columns(frame)
    frame = frame.rename(
        columns={
            "date": "date_local",
            "arrival_departure": "direction",
            "hong_kong_residents": "hk_residents",
            "mainland_visitors": "mainland_visitors",
            "other_visitors": "other_visitors",
            "total": "total_passengers",
        }
    )
    frame["date_local"] = pd.to_datetime(frame["date_local"], format="%d-%m-%Y", errors="coerce")
    frame["period_month"] = frame["date_local"].dt.to_period("M").dt.to_timestamp()
    frame["direction"] = frame["direction"].astype("string").str.strip().str.lower()
    for column in ["hk_residents", "mainland_visitors", "other_visitors", "total_passengers"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = (
        frame.groupby(["period_month", "direction"], as_index=False)[
            ["hk_residents", "mainland_visitors", "other_visitors", "total_passengers"]
        ]
        .sum()
    )
    pivot = grouped.pivot(index="period_month", columns="direction")
    if pivot.empty:
        return None
    pivot.columns = [f"immigration_{direction}_{metric}" for metric, direction in pivot.columns]
    city = pivot.reset_index().sort_values("period_month").reset_index(drop=True)
    arrivals = city.get("immigration_arrival_total_passengers")
    departures = city.get("immigration_departure_total_passengers")
    if arrivals is not None and departures is not None:
        city["immigration_total_cross_border_passengers"] = arrivals.fillna(0) + departures.fillna(0)
        city["immigration_net_arrivals_total_passengers"] = arrivals.fillna(0) - departures.fillna(0)
    return city


def build_transport_monthly(raw_root: Path = RAW_ROOT) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}

    public_path = _latest_payload("td_monthly_public_transport", raw_root=raw_root)
    if public_path is not None:
        frame = pd.read_csv(public_path, encoding="utf-8-sig")
        frame = _normalize_frame_columns(frame)
        frame["period_month"] = _parse_period_month(frame["yr_mth"])
        frame["avg_daily_pax"] = pd.to_numeric(frame["avg_daily_pax"], errors="coerce")
        frame["mode"] = frame["mode"].astype("string").str.strip()
        detail = frame[["period_month", "mode", "ttd_pto_code", "avg_daily_pax"]].dropna(subset=["period_month"])
        outputs["transport_public_mode_monthly"] = detail

        total = (
            detail.groupby("period_month", as_index=False)["avg_daily_pax"]
            .sum()
            .rename(columns={"avg_daily_pax": "public_transport_total_avg_daily_pax"})
        )
        outputs["transport_public_total_monthly"] = total

        wide = detail.pivot_table(index="period_month", columns="mode", values="avg_daily_pax", aggfunc="sum")
        wide.columns = [f"public_transport_{_clean_column_name(column)}_avg_daily_pax" for column in wide.columns]
        outputs["transport_public_mode_wide_monthly"] = wide.reset_index().sort_values("period_month").reset_index(drop=True)

    cross_path = _latest_payload("td_cross_harbour_traffic", raw_root=raw_root)
    if cross_path is not None:
        frame = pd.read_csv(cross_path, encoding="utf-8-sig")
        frame = _normalize_frame_columns(frame)
        frame["period_month"] = _parse_period_month(frame["yr_mth"])
        frame["no_pax"] = pd.to_numeric(frame["no_pax"], errors="coerce")
        frame["arrival_depart"] = frame["arrival_depart"].astype("string").str.strip().str.lower()
        detail = frame.groupby(["period_month", "arrival_depart"], as_index=False)["no_pax"].sum()
        wide = detail.pivot(index="period_month", columns="arrival_depart", values="no_pax")
        wide.columns = [f"cross_harbour_{_clean_column_name(column)}_pax" for column in wide.columns]
        monthly = wide.reset_index().sort_values("period_month").reset_index(drop=True)
        value_columns = [column for column in monthly.columns if column != "period_month"]
        monthly["cross_harbour_total_pax"] = monthly[value_columns].sum(axis=1, skipna=True)
        outputs["transport_cross_harbour_monthly"] = monthly

    return outputs


def _build_bd_monthly_from_source(
    source_name: str,
    rename_map: dict[str, str],
    raw_root: Path = RAW_ROOT,
) -> pd.DataFrame | None:
    payload = _latest_payload(source_name, raw_root=raw_root)
    if payload is None:
        return None
    frame = pd.read_csv(payload, encoding="utf-8-sig")
    frame = _normalize_frame_columns(frame)
    if "year" not in frame.columns or "month" not in frame.columns:
        return None
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce")
    frame["period_month"] = pd.to_datetime(
        dict(year=frame["year"], month=frame["month"], day=1),
        errors="coerce",
    )
    keep_columns = ["period_month"]
    for source_column, target_column in rename_map.items():
        if source_column in frame.columns:
            frame[target_column] = pd.to_numeric(frame[source_column], errors="coerce")
            keep_columns.append(target_column)
    out = frame[keep_columns].dropna(subset=["period_month"]).sort_values("period_month").reset_index(drop=True)
    return out if len(out.columns) > 1 else None


def build_building_activity_monthly(raw_root: Path = RAW_ROOT) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}

    consent_monthly = _build_bd_monthly_from_source(
        "bd_consent_commence_work_monthly",
        {
            "demolition_md_5_2": "building_consent_demolition_count",
            "site_formation": "building_consent_site_formation_count",
            "foundation": "building_consent_foundation_count",
            "general_building_superstructure_md5_4": "building_consent_superstructure_count",
            "total": "building_consent_total_count",
        },
        raw_root=raw_root,
    )
    if consent_monthly is not None:
        outputs["building_consent_monthly"] = consent_monthly

    occupation_permits_monthly = _build_bd_monthly_from_source(
        "bd_occupation_permits_monthly",
        {
            "domestic_op_issued": "occupation_permits_domestic_count",
            "non_domestic_op_issued": "occupation_permits_non_domestic_count",
            "composite_domestic_non_domestic_op_issued": "occupation_permits_composite_count",
            "total_op_issued": "occupation_permits_total_count",
            "total_no_of_domestic_units": "occupation_permits_domestic_units",
        },
        raw_root=raw_root,
    )
    if occupation_permits_monthly is not None:
        outputs["occupation_permits_monthly"] = occupation_permits_monthly

    building_completion_monthly = _build_bd_monthly_from_source(
        "bd_building_completion_monthly",
        {
            "domestic_gfa": "building_completion_domestic_gfa",
            "non_domestic_gfa": "building_completion_non_domestic_gfa",
            "total_gfa": "building_completion_total_gfa",
            "domestic_ufa": "building_completion_domestic_ufa",
            "non_domestic_ufa": "building_completion_non_domestic_ufa",
            "total_ufa": "building_completion_total_ufa",
            "total_no_of_domestic_units": "building_completion_domestic_units",
            "total_declared_building_costs_hk": "building_completion_declared_cost_hkd",
            "total_declared_building_costs_hk_alteration_and_addition_work": "building_completion_declared_cost_aa_hkd",
        },
        raw_root=raw_root,
    )
    if building_completion_monthly is not None:
        outputs["building_completion_monthly"] = building_completion_monthly

    if outputs:
        merged: pd.DataFrame | None = None
        for name in ["building_consent_monthly", "occupation_permits_monthly", "building_completion_monthly"]:
            frame = outputs.get(name)
            if frame is None:
                continue
            merged = frame if merged is None else merged.merge(frame, on="period_month", how="outer")
        if merged is not None:
            outputs["building_activity_monthly"] = merged.sort_values("period_month").reset_index(drop=True)
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
        outputs["censtatd_energy_stats_tidy"] = pd.concat(outputs.values(), ignore_index=True, sort=False)
    return outputs


def build_city_energy_monthly(raw_root: Path = RAW_ROOT) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    payload = _latest_payload("censtatd_energy_use_monthly", raw_root=raw_root)
    if payload is None:
        return None, None
    obj = json.loads(payload.read_text(encoding="utf-8"))
    frame = pd.DataFrame(obj.get("dataSet", []))
    if frame.empty:
        return None, None
    frame = _normalize_frame_columns(frame)
    frame["figure"] = pd.to_numeric(frame["figure"], errors="coerce")
    frame["freq"] = frame["freq"].astype("string").str.upper()
    frame["period_month"] = _parse_period_month(frame["period"])
    frame["user_typedesc"] = frame["user_typedesc"].astype("string").str.strip().replace({"": "Total"})
    detail = frame[frame["freq"].eq("M")].dropna(subset=["period_month"]).copy()
    if detail.empty:
        return None, None

    electricity = detail[detail["sv"].eq("ELEC_LOCAL")].copy()
    electricity_wide = electricity.pivot_table(
        index="period_month",
        columns="user_typedesc",
        values="figure",
        aggfunc="sum",
    )
    electricity_wide.columns = [f"electricity_{_clean_column_name(column)}_tj" for column in electricity_wide.columns]
    city = electricity_wide.reset_index().sort_values("period_month").reset_index(drop=True)

    gas_total = (
        detail[(detail["sv"].eq("GASC_LOCAL")) & (detail["user_typedesc"].eq("Total"))][["period_month", "figure"]]
        .rename(columns={"figure": "gas_total_tj"})
        .drop_duplicates(subset=["period_month"])
    )
    elec_export = (
        detail[(detail["sv"].eq("ELE_EX_CHN")) & (detail["user_typedesc"].eq("Total"))][["period_month", "figure"]]
        .rename(columns={"figure": "electricity_export_to_mainland_tj"})
        .drop_duplicates(subset=["period_month"])
    )
    city = city.merge(gas_total, on="period_month", how="left")
    city = city.merge(elec_export, on="period_month", how="left")
    return detail.sort_values(["period_month", "sv", "user_typedesc"]).reset_index(drop=True), city


def build_hkelectric_re_generation_monthly(raw_root: Path = RAW_ROOT) -> pd.DataFrame | None:
    payload = _latest_payload("hkelectric_re_generation_by_type", raw_root=raw_root)
    if payload is None:
        return None
    with payload.open("rb") as handle:
        signature = handle.read(4)
    frame = pd.read_excel(payload) if signature == b"PK\x03\x04" else pd.read_csv(payload, encoding="utf-8-sig")
    frame = _normalize_frame_columns(frame)
    if "year" not in frame.columns or "month" not in frame.columns:
        return None
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce")
    frame["period_month"] = pd.to_datetime(
        dict(year=frame["year"], month=frame["month"], day=1),
        errors="coerce",
    )
    value_columns = [column for column in frame.columns if column not in {"year", "month", "period_month"}]
    for column in value_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rename_map = {column: f"hkelectric_re_generation_{column}" for column in value_columns}
    out = frame.rename(columns=rename_map)[["period_month"] + list(rename_map.values())]
    return out.dropna(subset=["period_month"]).sort_values("period_month").reset_index(drop=True)


def build_city_population_yearly(raw_root: Path = RAW_ROOT) -> pd.DataFrame | None:
    payload = _latest_payload("district_population_projection", raw_root=raw_root)
    if payload is None:
        return None
    data = json.loads(payload.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        for key, value in props.items():
            if re.fullmatch(r"F\d{4}", str(key)):
                rows.append(
                    {
                        "year": int(str(key)[1:]),
                        "population": pd.to_numeric(value, errors="coerce"),
                        "district_area_m2": pd.to_numeric(props.get("Shape__Area"), errors="coerce"),
                    }
                )
    frame = pd.DataFrame(rows).dropna(subset=["year", "population"])
    if frame.empty:
        return None
    city = (
        frame.groupby("year", as_index=False)[["population", "district_area_m2"]]
        .sum()
        .sort_values("year")
        .reset_index(drop=True)
    )
    city["population_density_per_km2"] = city["population"] / (city["district_area_m2"] / 1_000_000.0)
    return city


def build_ev_city_static(raw_root: Path = RAW_ROOT) -> pd.DataFrame | None:
    payload = _latest_payload("ev_public_chargers", raw_root=raw_root)
    if payload is None:
        return None
    data = json.loads(payload.read_text(encoding="utf-8"))
    rows = [feature.get("properties", {}) for feature in data.get("features", [])]
    if not rows:
        return None
    frame = _normalize_frame_columns(pd.DataFrame(rows))
    count_columns = [column for column in frame.columns if column.endswith("_no")]
    for column in count_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame["total_chargers"] = frame[count_columns].sum(axis=1)
    summary = {
        "ev_standard_chargers": float(frame.get("standard_bs1363_no", pd.Series(dtype=float)).sum()),
        "ev_medium_iec62196_chargers": float(frame.get("medium_iec62196_no", pd.Series(dtype=float)).sum()),
        "ev_quick_chademo_chargers": float(frame.get("quick_chademo_no", pd.Series(dtype=float)).sum()),
        "ev_quick_ccs_combo_chargers": float(frame.get("quick_ccs_dc_combo_no", pd.Series(dtype=float)).sum()),
        "ev_quick_iec62196_chargers": float(frame.get("quick_iec62196_no", pd.Series(dtype=float)).sum()),
        "ev_quick_gbt_dc_chargers": float(frame.get("quick_gb_t20234_3_dc_no", pd.Series(dtype=float)).sum()),
        "ev_total_chargers": float(frame["total_chargers"].sum()),
    }
    return pd.DataFrame([summary])


def build_building_footprints_city_static(raw_root: Path = RAW_ROOT) -> pd.DataFrame | None:
    payload = _latest_payload("building_footprints", raw_root=raw_root)
    if payload is None:
        return None
    data = json.loads(payload.read_text(encoding="utf-8"))
    rows = [feature.get("properties", {}) for feature in data.get("features", [])]
    if not rows:
        return None
    frame = _normalize_frame_columns(pd.DataFrame(rows))
    frame["building_height_m"] = pd.to_numeric(frame.get("rooflevel"), errors="coerce") - pd.to_numeric(
        frame.get("baselevel"),
        errors="coerce",
    )
    frame["footprint_area_m2"] = pd.to_numeric(frame.get("shape_area"), errors="coerce")
    summary = {
        "building_count": int(frame.get("buildingid", pd.Series(dtype="string")).nunique()),
        "building_block_type_count": int(frame.get("typeofbuildingblock", pd.Series(dtype="string")).nunique()),
        "footprint_area_m2_sum": float(frame["footprint_area_m2"].sum(skipna=True)),
        "footprint_area_m2_mean": float(frame["footprint_area_m2"].mean(skipna=True)),
        "building_height_m_mean": float(frame["building_height_m"].mean(skipna=True)),
        "completed_building_count": int(frame.get("buildingstatus", pd.Series(dtype="string")).astype("string").str.lower().eq("existing").sum()),
    }
    return pd.DataFrame([summary])


def _merge_yearly_features(frame: pd.DataFrame, yearly: pd.DataFrame | None) -> pd.DataFrame:
    if yearly is None or yearly.empty:
        return frame
    out = frame.copy()
    year_min = int(yearly["year"].min())
    year_max = int(yearly["year"].max())
    out["feature_year"] = out["year"].clip(lower=year_min, upper=year_max)
    merged = out.merge(yearly.rename(columns={"year": "feature_year"}), on="feature_year", how="left")
    return merged


def _add_monthly_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values("period_month").reset_index(drop=True).copy()
    series = out["electricity_total_tj"]
    for lag in [1, 2, 3, 6, 12]:
        out[f"electricity_lag_{lag}m"] = series.shift(lag)
    for window in [3, 6, 12]:
        out[f"electricity_roll_mean_{window}m"] = series.shift(1).rolling(window, min_periods=max(1, window // 2)).mean()
        out[f"electricity_roll_std_{window}m"] = series.shift(1).rolling(window, min_periods=max(1, window // 2)).std()
    out["electricity_yoy_delta_tj"] = out["electricity_total_tj"] - out["electricity_lag_12m"]
    denominator = out["electricity_lag_12m"].replace({0: pd.NA})
    out["electricity_yoy_growth"] = out["electricity_total_tj"] / denominator - 1
    out[MONTHLY_TARGET_COL] = series.shift(-1)
    out["target_period_month"] = out["period_month"] + pd.offsets.MonthBegin(1)
    return out


def build_city_monthly_model_tables(
    city_energy_monthly: pd.DataFrame | None,
    weather_monthly: pd.DataFrame | None,
    calendar_monthly: pd.DataFrame | None,
    immigration_monthly: pd.DataFrame | None,
    transport_tables: dict[str, pd.DataFrame],
    building_activity_tables: dict[str, pd.DataFrame],
    hkelectric_re_generation_monthly: pd.DataFrame | None,
    city_population_yearly: pd.DataFrame | None,
    ev_city_static: pd.DataFrame | None,
    building_city_static: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if city_energy_monthly is None or city_energy_monthly.empty:
        return None, None

    frame = city_energy_monthly.copy().sort_values("period_month").reset_index(drop=True)
    frame["year"] = frame["period_month"].dt.year.astype(int)
    frame["month"] = frame["period_month"].dt.month.astype(int)
    frame["quarter"] = frame["period_month"].dt.quarter.astype(int)
    frame["sin_month"] = frame["month"].map(lambda value: math.sin(2 * math.pi * value / 12))
    frame["cos_month"] = frame["month"].map(lambda value: math.cos(2 * math.pi * value / 12))
    frame["source_dataset"] = "hong_kong_city_monthly"

    for monthly_frame in [
        weather_monthly,
        calendar_monthly,
        immigration_monthly,
        transport_tables.get("transport_public_total_monthly"),
        transport_tables.get("transport_public_mode_wide_monthly"),
        transport_tables.get("transport_cross_harbour_monthly"),
        building_activity_tables.get("building_activity_monthly"),
        hkelectric_re_generation_monthly,
    ]:
        if monthly_frame is not None and not monthly_frame.empty:
            frame = frame.merge(monthly_frame, on="period_month", how="left")

    frame = _merge_yearly_features(frame, city_population_yearly)

    frame["immigration_features_available"] = frame.filter(like="immigration_").notna().any(axis=1).astype(int)
    frame["transport_features_available"] = frame.filter(like="public_transport_").notna().any(axis=1).astype(int)
    frame["building_activity_features_available"] = frame.filter(like="building_").notna().any(axis=1).astype(int) | frame.filter(like="occupation_permits_").notna().any(axis=1).astype(int)
    frame["re_generation_features_available"] = frame.filter(like="hkelectric_re_generation_").notna().any(axis=1).astype(int)
    frame["population_features_available"] = frame[["population", "population_density_per_km2"]].notna().any(axis=1).astype(int)
    frame["building_activity_features_available"] = frame["building_activity_features_available"].astype(int)

    frame = _add_monthly_lag_features(frame)
    frame = frame.sort_values("period_month").reset_index(drop=True)

    train_ready = frame[frame["period_month"] >= MODEL_START_MONTH].copy()
    train_ready = train_ready.dropna(subset=MONTHLY_REQUIRED_COLUMNS).reset_index(drop=True)
    return frame, train_ready


def build_silver_tables(
    raw_root: Path = RAW_ROOT,
    silver_root: Path = SILVER_ROOT,
    model_root: Path = MODEL_ROOT,
    manual_root: Path = MANUAL_ROOT,
) -> dict[str, Any]:
    del manual_root
    _ensure_dir(silver_root)
    _ensure_dir(model_root)
    _cleanup_obsolete_outputs(silver_root=silver_root, model_root=model_root)

    outputs: dict[str, Any] = {"built": {}, "skipped": []}

    weather_daily = build_weather_daily(raw_root=raw_root)
    weather_monthly = build_weather_monthly(weather_daily)
    if weather_monthly is not None:
        _write_csv(weather_monthly, silver_root / "weather_monthly.csv")
        outputs["built"]["weather_monthly"] = int(len(weather_monthly))
    else:
        outputs["skipped"].append("weather_monthly")

    holiday_events = build_holiday_events(raw_root=raw_root)
    calendar_monthly = build_calendar_monthly(weather_daily=weather_daily, holiday_events=holiday_events)
    if calendar_monthly is not None:
        _write_csv(calendar_monthly, silver_root / "calendar_monthly.csv")
        outputs["built"]["calendar_monthly"] = int(len(calendar_monthly))
    else:
        outputs["skipped"].append("calendar_monthly")

    immigration_monthly = build_immigration_monthly(raw_root=raw_root)
    if immigration_monthly is not None:
        _write_csv(immigration_monthly, silver_root / "immigration_city_monthly.csv")
        outputs["built"]["immigration_city_monthly"] = int(len(immigration_monthly))
    else:
        outputs["skipped"].append("immigration_city_monthly")

    transport_tables = build_transport_monthly(raw_root=raw_root)
    for name, frame in transport_tables.items():
        _write_csv(frame, silver_root / f"{name}.csv")
        outputs["built"][name] = int(len(frame))
    if not transport_tables:
        outputs["skipped"].append("transport_monthly")

    building_activity_tables = build_building_activity_monthly(raw_root=raw_root)
    for name, frame in building_activity_tables.items():
        _write_csv(frame, silver_root / f"{name}.csv")
        outputs["built"][name] = int(len(frame))
    if not building_activity_tables:
        outputs["skipped"].append("building_activity_monthly")

    censtatd_tables = build_censtatd_tidy(raw_root=raw_root)
    for name, frame in censtatd_tables.items():
        _write_csv(frame, silver_root / f"{name}.csv")
        outputs["built"][name] = int(len(frame))
    if not censtatd_tables:
        outputs["skipped"].append("censtatd_energy_stats")

    city_energy_detail, city_energy_monthly = build_city_energy_monthly(raw_root=raw_root)
    if city_energy_detail is not None:
        _write_csv(city_energy_detail, silver_root / "city_energy_monthly_detail.csv")
        outputs["built"]["city_energy_monthly_detail"] = int(len(city_energy_detail))
    else:
        outputs["skipped"].append("city_energy_monthly_detail")
    if city_energy_monthly is not None:
        _write_csv(city_energy_monthly, silver_root / "city_energy_monthly.csv")
        outputs["built"]["city_energy_monthly"] = int(len(city_energy_monthly))
    else:
        outputs["skipped"].append("city_energy_monthly")

    hkelectric_re_generation_monthly = build_hkelectric_re_generation_monthly(raw_root=raw_root)
    if hkelectric_re_generation_monthly is not None:
        _write_csv(hkelectric_re_generation_monthly, silver_root / "hkelectric_re_generation_monthly.csv")
        outputs["built"]["hkelectric_re_generation_monthly"] = int(len(hkelectric_re_generation_monthly))
    else:
        outputs["skipped"].append("hkelectric_re_generation_monthly")

    city_population_yearly = build_city_population_yearly(raw_root=raw_root)
    if city_population_yearly is not None:
        _write_csv(city_population_yearly, silver_root / "city_population_yearly.csv")
        outputs["built"]["city_population_yearly"] = int(len(city_population_yearly))
    else:
        outputs["skipped"].append("city_population_yearly")

    ev_city_static = build_ev_city_static(raw_root=raw_root)
    if ev_city_static is not None:
        _write_csv(ev_city_static, silver_root / "ev_city_static.csv")
        outputs["built"]["ev_city_static"] = int(len(ev_city_static))
    else:
        outputs["skipped"].append("ev_city_static")

    building_city_static = build_building_footprints_city_static(raw_root=raw_root)
    if building_city_static is not None:
        _write_csv(building_city_static, silver_root / "building_footprints_city_static.csv")
        outputs["built"]["building_footprints_city_static"] = int(len(building_city_static))
    else:
        outputs["skipped"].append("building_footprints_city_static")

    monthly_features, train_ready = build_city_monthly_model_tables(
        city_energy_monthly=city_energy_monthly,
        weather_monthly=weather_monthly,
        calendar_monthly=calendar_monthly,
        immigration_monthly=immigration_monthly,
        transport_tables=transport_tables,
        building_activity_tables=building_activity_tables,
        hkelectric_re_generation_monthly=hkelectric_re_generation_monthly,
        city_population_yearly=city_population_yearly,
        ev_city_static=ev_city_static,
        building_city_static=building_city_static,
    )
    if monthly_features is not None:
        _write_csv(monthly_features, model_root / "city_monthly_model_features.csv")
        outputs["built"]["city_monthly_model_features"] = int(len(monthly_features))
    else:
        outputs["skipped"].append("city_monthly_model_features")
    if train_ready is not None:
        _write_csv(train_ready, model_root / "city_monthly_train_ready.csv")
        outputs["built"]["city_monthly_train_ready"] = int(len(train_ready))
    else:
        outputs["skipped"].append("city_monthly_train_ready")

    manifest_path = silver_root / "silver_manifest.json"
    write_json(manifest_path, outputs)
    return outputs
