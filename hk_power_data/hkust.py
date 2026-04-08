from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from rdflib import Graph, Namespace
    from rdflib.namespace import RDF
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Graph = None
    Namespace = None
    RDF = None


HKUST_TZ = "Asia/Hong_Kong"
HKUST_MIN_VALID_TS = pd.Timestamp("2020-01-01 00:00:00")
HKUST_ROOT_CANDIDATES = [
    Path("All_Data") / "All Data" / "Clean Dataset",
    Path("All_Data") / "Clean Dataset",
    Path("All Data") / "Clean Dataset",
]
HKUST_MODEL_REQUIRED_COLUMNS = ["load_value", "load_lag_1h", "load_lag_24h", "load_lag_168h", "target_load_t_plus_168h"]

if Namespace is not None:
    BRICK = Namespace("https://brickschema.org/schema/Brick#")
    REF = Namespace("https://brickschema.org/schema/Brick/ref#")
    EXT = Namespace("https://www.HKUST_Electric_Meter.com/schema/BrickExtension#")
    HKUST_PARENT_PREDICATES = {BRICK.hasPart, BRICK.isLocationOf, BRICK.isMeteredBy}
else:  # pragma: no cover - exercised only when rdflib is unavailable
    BRICK = None
    REF = None
    EXT = None
    HKUST_PARENT_PREDICATES = set()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _append_csv(frame: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    if path.exists():
        frame.to_csv(path, index=False, encoding="utf-8", mode="a", header=False)
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def _extract_node_id(node: Any) -> str | None:
    if node is None:
        return None
    text = str(node).strip()
    if not text:
        return None
    return text.split("#")[-1]


def _resolve_reference_value(graph: Graph, node: Any) -> str | None:
    if node is None:
        return None
    for predicate in [BRICK.value, RDF.value]:
        values = list(graph.objects(node, predicate))
        if values:
            return str(values[0]).strip()
    text = str(node).strip()
    return text or None


def _find_hkust_clean_root(raw_root: Path) -> Path | None:
    for relative_path in HKUST_ROOT_CANDIDATES:
        candidate = raw_root / relative_path
        if candidate.exists():
            return candidate
    return None


def _walk_ancestors(graph: Graph, start_node: Any) -> list[tuple[Any, int, Any]]:
    seen = {start_node}
    queue = deque([(start_node, 0)])
    ancestors: list[tuple[Any, int, Any]] = []
    while queue:
        node, depth = queue.popleft()
        for subject, predicate in graph.subject_predicates(node):
            if predicate not in HKUST_PARENT_PREDICATES or subject in seen:
                continue
            next_depth = depth + 1
            seen.add(subject)
            queue.append((subject, next_depth))
            ancestors.append((subject, next_depth, predicate))
    return ancestors


def _nearest_ancestor_by_type(graph: Graph, ancestors: list[tuple[Any, int, Any]], target_type: Any) -> tuple[str | None, int | None]:
    ranked = sorted(ancestors, key=lambda item: item[1])
    for node, depth, _predicate in ranked:
        node_types = set(graph.objects(node, RDF.type))
        if target_type in node_types:
            return _extract_node_id(node), depth
    return None, None


def _build_hkust_meter_metadata(clean_root: Path, local_files: list[Path]) -> pd.DataFrame:
    if Graph is None or RDF is None or Namespace is None:
        raise ImportError("rdflib is required to parse the HKUST Brick metadata.")

    ttl_path = clean_root / "HKUST_Meter_Metadata.ttl"
    graph = Graph()
    graph.parse(ttl_path)

    local_file_map = {path.stem.split(".")[-1].upper(): path.name for path in local_files}
    rows: list[dict[str, Any]] = []

    for meter in set(graph.subjects(RDF.type, BRICK.Electrical_Meter)):
        series_name = None
        for ext_ref in graph.objects(meter, REF.hasExternalReference):
            for timeseries_name in graph.objects(ext_ref, REF.hasTimeseriesData):
                series_name = str(timeseries_name).strip()
                break
            if series_name:
                break
        if not series_name:
            continue

        meter_id = Path(series_name).stem.upper()
        if meter_id not in local_file_map:
            continue

        usage_type = None
        for usage_ref in graph.objects(meter, EXT.usageType):
            usage_type = _resolve_reference_value(graph, usage_ref)
            if usage_type:
                break

        direct_parents = [
            (subject, predicate)
            for subject, predicate in graph.subject_predicates(meter)
            if predicate in HKUST_PARENT_PREDICATES
        ]
        direct_parent = direct_parents[0][0] if direct_parents else None
        direct_parent_type = None
        if direct_parent is not None:
            direct_parent_types = list(graph.objects(direct_parent, RDF.type))
            direct_parent_type = _extract_node_id(direct_parent_types[0]) if direct_parent_types else None

        ancestors = _walk_ancestors(graph, meter)
        zone_id, zone_depth = _nearest_ancestor_by_type(graph, ancestors, BRICK.Zone)
        building_id, building_depth = _nearest_ancestor_by_type(graph, ancestors, BRICK.Building)
        site_id, site_depth = _nearest_ancestor_by_type(graph, ancestors, BRICK.Site)

        unit = None
        for unit_node in graph.objects(meter, BRICK.hasUnit):
            unit = _extract_node_id(unit_node)
            if unit:
                break

        meter_node_id = _extract_node_id(meter)
        rows.append(
            {
                "meter_id": meter_id,
                "meter_node_id": meter_node_id,
                "meter_name": meter_node_id.replace("Meter_", "", 1) if meter_node_id else meter_id,
                "source_timeseries_file": local_file_map[meter_id],
                "metadata_timeseries_file": series_name,
                "usage_type": usage_type or "Undefined",
                "meter_unit": unit,
                "direct_parent_id": _extract_node_id(direct_parent),
                "direct_parent_type": direct_parent_type,
                "zone_id": zone_id,
                "zone_name": zone_id,
                "zone_depth": zone_depth,
                "building_id": building_id,
                "building_name": building_id,
                "building_depth": building_depth,
                "site_id": site_id,
                "site_name": site_id,
                "site_depth": site_depth,
            }
        )

    metadata = pd.DataFrame(rows).drop_duplicates(subset=["meter_id"]).sort_values("meter_id").reset_index(drop=True)
    if metadata.empty:
        raise ValueError("No HKUST meter metadata could be matched to the local T60 files.")
    return metadata


def _fallback_metadata_row(path: Path) -> dict[str, Any]:
    meter_id = path.stem.split(".")[-1].upper()
    return {
        "meter_id": meter_id,
        "meter_node_id": f"Meter_{meter_id}",
        "meter_name": meter_id,
        "source_timeseries_file": path.name,
        "metadata_timeseries_file": f"{meter_id}.xlsx",
        "usage_type": "Undefined",
        "meter_unit": "KiloW-HR",
        "direct_parent_id": None,
        "direct_parent_type": None,
        "zone_id": None,
        "zone_name": None,
        "zone_depth": None,
        "building_id": f"UNMAPPED_{meter_id}",
        "building_name": f"UNMAPPED_{meter_id}",
        "building_depth": None,
        "site_id": "UNKNOWN_SITE",
        "site_name": "UNKNOWN_SITE",
        "site_depth": None,
    }


def _process_hkust_meter_file(path: Path, metadata_row: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    meter_id = metadata_row.get("meter_id", path.stem.split(".")[-1].upper())
    frame = pd.read_excel(path, usecols=["time", "number"])
    frame = frame.rename(columns={"time": "ts_local", "number": "cumulative_kwh"})
    frame["ts_local"] = pd.to_datetime(frame["ts_local"], errors="coerce")
    frame["cumulative_kwh"] = pd.to_numeric(frame["cumulative_kwh"], errors="coerce")
    frame = frame[frame["ts_local"].ge(HKUST_MIN_VALID_TS) | frame["ts_local"].isna()]
    frame = frame.dropna(subset=["ts_local"]).sort_values("ts_local").drop_duplicates(subset=["ts_local"], keep="last")
    frame = frame.reset_index(drop=True)

    frame["prev_ts_local"] = frame["ts_local"].shift(1)
    frame["prev_cumulative_kwh"] = frame["cumulative_kwh"].shift(1)
    frame["gap_hours"] = (frame["ts_local"] - frame["prev_ts_local"]).dt.total_seconds().div(3600)
    frame["load_value"] = frame["cumulative_kwh"] - frame["prev_cumulative_kwh"]

    first_row_mask = frame["prev_ts_local"].isna()
    missing_cumulative_mask = frame["cumulative_kwh"].isna() | frame["prev_cumulative_kwh"].isna()
    irregular_gap_mask = frame["gap_hours"].ne(1) & frame["prev_ts_local"].notna()
    negative_diff_mask = frame["load_value"] < 0
    valid_increment_mask = ~(first_row_mask | missing_cumulative_mask | irregular_gap_mask | negative_diff_mask)

    frame.loc[~valid_increment_mask, "load_value"] = np.nan
    frame["quality_flag"] = np.select(
        [first_row_mask, negative_diff_mask, irregular_gap_mask, missing_cumulative_mask],
        ["initial_observation", "negative_diff", "irregular_gap", "missing_cumulative"],
        default="ok",
    )

    ts_utc = pd.DatetimeIndex(frame["ts_local"]).tz_localize(HKUST_TZ).tz_convert("UTC")
    frame["ts_utc"] = ts_utc
    frame["date_local"] = frame["ts_local"].dt.normalize()
    frame["meter_id"] = meter_id
    frame["building_id"] = metadata_row.get("building_id")
    frame["site_id"] = metadata_row.get("site_id")
    frame["source_file"] = path.name

    detail = frame[
        [
            "meter_id",
            "building_id",
            "site_id",
            "ts_local",
            "ts_utc",
            "date_local",
            "cumulative_kwh",
            "load_value",
            "quality_flag",
            "source_file",
        ]
    ].copy()

    summary = {
        "meter_id": meter_id,
        "building_id": metadata_row.get("building_id"),
        "building_name": metadata_row.get("building_name"),
        "site_id": metadata_row.get("site_id"),
        "site_name": metadata_row.get("site_name"),
        "usage_type": metadata_row.get("usage_type"),
        "source_file": path.name,
        "row_count": int(len(detail)),
        "start_ts_local": detail["ts_local"].min(),
        "end_ts_local": detail["ts_local"].max(),
        "missing_cumulative_count": int(frame["cumulative_kwh"].isna().sum()),
        "valid_increment_count": int(valid_increment_mask.sum()),
        "negative_diff_count": int(negative_diff_mask.sum()),
        "irregular_gap_count": int(irregular_gap_mask.sum()),
        "mean_hourly_load_kwh": float(detail["load_value"].mean(skipna=True)) if detail["load_value"].notna().any() else np.nan,
        "max_hourly_load_kwh": float(detail["load_value"].max(skipna=True)) if detail["load_value"].notna().any() else np.nan,
    }
    return detail, summary


def _build_hkust_building_static(meter_metadata: pd.DataFrame) -> pd.DataFrame:
    building_static = (
        meter_metadata.groupby(["building_id", "building_name", "site_id", "site_name"], dropna=False)
        .agg(
            meter_count_total=("meter_id", "nunique"),
            zone_count=("zone_id", lambda series: int(series.dropna().nunique())),
            usage_type_count=("usage_type", lambda series: int(series.dropna().nunique())),
            undefined_usage_meter_count=("usage_type", lambda series: int(series.fillna("Undefined").eq("Undefined").sum())),
        )
        .reset_index()
        .sort_values(["site_id", "building_id"])
        .reset_index(drop=True)
    )
    return building_static


def _add_group_lag_features(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = frame.sort_values([group_col, "ts_utc"]).copy()
    grouped = out.groupby(group_col)["load_value"]
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


def _build_hkust_model_tables(
    building_hourly: pd.DataFrame | None,
    weather_daily: pd.DataFrame | None,
    calendar_daily: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if building_hourly is None or building_hourly.empty:
        return None, None

    frame = building_hourly.copy()
    frame["hour_of_day"] = frame["ts_local"].dt.hour.astype(int)
    frame["day_of_week"] = frame["ts_local"].dt.dayofweek + 1
    frame["hour_of_week"] = (frame["day_of_week"] - 1) * 24 + frame["hour_of_day"]
    frame["week_of_year"] = frame["ts_local"].dt.isocalendar().week.astype(int)
    frame["month"] = frame["ts_local"].dt.month.astype(int)
    frame["quarter"] = frame["ts_local"].dt.quarter.astype(int)
    frame["year"] = frame["ts_local"].dt.year.astype(int)
    frame["is_weekend"] = frame["day_of_week"].isin([6, 7]).astype(int)
    frame["sin_hour_of_day"] = frame["hour_of_day"].map(lambda value: np.sin(2 * np.pi * value / 24))
    frame["cos_hour_of_day"] = frame["hour_of_day"].map(lambda value: np.cos(2 * np.pi * value / 24))
    frame["sin_hour_of_week"] = frame["hour_of_week"].map(lambda value: np.sin(2 * np.pi * value / 168))
    frame["cos_hour_of_week"] = frame["hour_of_week"].map(lambda value: np.cos(2 * np.pi * value / 168))

    if weather_daily is not None and not weather_daily.empty:
        frame = frame.merge(weather_daily.drop_duplicates("date_local"), on="date_local", how="left")

    if calendar_daily is not None and not calendar_daily.empty:
        frame = frame.merge(
            calendar_daily[["date_local", "holiday_name_en", "is_public_holiday"]],
            on="date_local",
            how="left",
        )
        frame["is_public_holiday"] = frame["is_public_holiday"].fillna(0).astype(int)
    else:
        frame["holiday_name_en"] = pd.NA
        frame["is_public_holiday"] = 0

    frame["source_dataset"] = "hkust_dryad"
    frame = _add_group_lag_features(frame, group_col="building_id")
    frame = frame.sort_values(["ts_utc", "building_id"]).reset_index(drop=True)

    train_ready = frame.dropna(subset=HKUST_MODEL_REQUIRED_COLUMNS).reset_index(drop=True)
    return frame, train_ready


def build_hkust_dryad_outputs(
    raw_root: Path,
    silver_root: Path,
    model_root: Path,
    weather_daily: pd.DataFrame | None,
    calendar_daily: pd.DataFrame | None,
) -> dict[str, Any]:
    clean_root = _find_hkust_clean_root(raw_root)
    if clean_root is None:
        return {"built": {}, "skipped": ["hkust_meter_metadata", "hkust_meter_hourly", "hkust_building_hourly", "hkust_building_train_ready_direct_168h"]}

    t60_dir = clean_root / "Resappled data" / "T60"
    ttl_path = clean_root / "HKUST_Meter_Metadata.ttl"
    if not t60_dir.exists() or not ttl_path.exists():
        return {"built": {}, "skipped": ["hkust_meter_metadata", "hkust_meter_hourly", "hkust_building_hourly", "hkust_building_train_ready_direct_168h"]}

    local_files = sorted(t60_dir.glob("*.xlsx"))
    if not local_files:
        return {"built": {}, "skipped": ["hkust_meter_metadata", "hkust_meter_hourly", "hkust_building_hourly", "hkust_building_train_ready_direct_168h"]}

    meter_metadata = _build_hkust_meter_metadata(clean_root=clean_root, local_files=local_files)
    meter_metadata_path = silver_root / "hkust_meter_metadata.csv"
    meter_hourly_path = silver_root / "hkust_meter_hourly.csv"
    meter_summary_path = silver_root / "hkust_meter_hourly_summary.csv"
    building_static_path = silver_root / "hkust_building_static.csv"
    building_hourly_path = silver_root / "hkust_building_hourly.csv"

    if meter_hourly_path.exists():
        meter_hourly_path.unlink()

    metadata_by_meter = meter_metadata.set_index("meter_id").to_dict(orient="index")
    building_static = _build_hkust_building_static(meter_metadata)
    building_static_map = building_static.set_index("building_id").to_dict(orient="index")

    building_sums: dict[str, pd.Series] = {}
    building_reporting_counts: dict[str, pd.Series] = {}
    summary_rows: list[dict[str, Any]] = []

    for path in local_files:
        meter_id = path.stem.split(".")[-1].upper()
        metadata_row = {"meter_id": meter_id, **metadata_by_meter.get(meter_id, _fallback_metadata_row(path))}
        detail, summary = _process_hkust_meter_file(path, metadata_row)
        _append_csv(detail, meter_hourly_path)
        summary_rows.append(summary)

        load_series = detail.set_index("ts_local")["load_value"]
        building_id = metadata_row["building_id"]
        building_sums[building_id] = building_sums.get(building_id, pd.Series(dtype="float64")).add(
            load_series.fillna(0.0),
            fill_value=0.0,
        )
        building_reporting_counts[building_id] = building_reporting_counts.get(building_id, pd.Series(dtype="float64")).add(
            load_series.notna().astype("float64"),
            fill_value=0.0,
        )

    meter_summary = pd.DataFrame(summary_rows).sort_values("meter_id").reset_index(drop=True)
    building_frames: list[pd.DataFrame] = []

    for building_id, sum_series in building_sums.items():
        count_series = building_reporting_counts[building_id].sort_index()
        load_series = sum_series.sort_index().where(count_series > 0, np.nan)
        static_row = building_static_map.get(building_id, {})
        meter_count_total = int(static_row.get("meter_count_total", 0))
        ts_local = pd.Series(load_series.index)
        ts_utc = pd.DatetimeIndex(load_series.index).tz_localize(HKUST_TZ).tz_convert("UTC")
        coverage_ratio = count_series.astype(float) / meter_count_total if meter_count_total > 0 else pd.Series(np.nan, index=count_series.index)
        building_frames.append(
            pd.DataFrame(
                {
                    "building_id": building_id,
                    "building_name": static_row.get("building_name", building_id),
                    "site_id": static_row.get("site_id"),
                    "site_name": static_row.get("site_name"),
                    "ts_local": ts_local.to_numpy(),
                    "ts_utc": ts_utc.to_numpy(),
                    "date_local": ts_local.dt.normalize().to_numpy(),
                    "load_value": load_series.to_numpy(),
                    "meter_count_reporting": count_series.astype(int).to_numpy(),
                    "meter_count_total": meter_count_total,
                    "meter_coverage_ratio": coverage_ratio.to_numpy(),
                }
            )
        )

    building_hourly = pd.concat(building_frames, ignore_index=True) if building_frames else pd.DataFrame()
    if not building_hourly.empty:
        building_hourly = building_hourly.sort_values(["ts_utc", "building_id"]).reset_index(drop=True)

    building_features, train_ready = _build_hkust_model_tables(
        building_hourly=building_hourly,
        weather_daily=weather_daily,
        calendar_daily=calendar_daily,
    )

    meter_metadata.to_csv(meter_metadata_path, index=False, encoding="utf-8-sig")
    meter_summary.to_csv(meter_summary_path, index=False, encoding="utf-8-sig")
    building_static.to_csv(building_static_path, index=False, encoding="utf-8-sig")

    if building_hourly.empty:
        if building_hourly_path.exists():
            building_hourly_path.unlink()
    else:
        building_hourly.to_csv(building_hourly_path, index=False, encoding="utf-8-sig")

    feature_path = model_root / "hkust_building_hourly_features.csv"
    train_ready_path = model_root / "hkust_building_train_ready_direct_168h.csv"
    if building_features is not None and not building_features.empty:
        building_features.to_csv(feature_path, index=False, encoding="utf-8-sig")
    elif feature_path.exists():
        feature_path.unlink()
    if train_ready is not None and not train_ready.empty:
        train_ready.to_csv(train_ready_path, index=False, encoding="utf-8-sig")
    elif train_ready_path.exists():
        train_ready_path.unlink()

    return {
        "built": {
            "hkust_meter_metadata": int(len(meter_metadata)),
            "hkust_meter_hourly": int(sum(row["row_count"] for row in summary_rows)),
            "hkust_meter_hourly_summary": int(len(meter_summary)),
            "hkust_building_static": int(len(building_static)),
            "hkust_building_hourly": int(len(building_hourly)),
            "hkust_building_hourly_features": int(len(building_features)) if building_features is not None else 0,
            "hkust_building_train_ready_direct_168h": int(len(train_ready)) if train_ready is not None else 0,
        },
        "skipped": [],
    }
