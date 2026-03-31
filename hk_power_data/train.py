from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .catalog import ROOT
from .fetchers import write_json


DEFAULT_DATASET = ROOT / "data" / "model" / "train_ready_direct_168h.csv"
DEFAULT_OUTPUT = ROOT / "data" / "model" / "training_runs"


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    denominator = np.maximum(np.abs(y_true.to_numpy()), 1e-6)
    mape = float(np.mean(np.abs((y_true.to_numpy() - y_pred) / denominator)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape,
    }


def _chronological_split(
    frame: pd.DataFrame,
    time_col: str,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_times = pd.Series(pd.to_datetime(frame[time_col], utc=True)).sort_values().drop_duplicates().reset_index(drop=True)
    if len(ordered_times) < 2:
        raise ValueError("Need at least two distinct timestamps for chronological split.")
    cutoff_index = max(1, int(len(ordered_times) * (1 - test_fraction)))
    cutoff_index = min(cutoff_index, len(ordered_times) - 1)
    cutoff_time = ordered_times.iloc[cutoff_index]
    train = frame[pd.to_datetime(frame[time_col], utc=True) < cutoff_time].copy()
    test = frame[pd.to_datetime(frame[time_col], utc=True) >= cutoff_time].copy()
    return train, test


def _build_feature_lists(frame: pd.DataFrame, target_col: str) -> tuple[list[str], list[str], list[str]]:
    exclude = {
        "ts_utc",
        "ts_local",
        "date_local",
        "source_url",
        "source_file",
        "quality_flag",
        "load_unit",
        "measure_type",
        "holiday_uid",
        "feature_year",
        target_col,
    }
    exclude.update(column for column in frame.columns if column.startswith("target_load_t_plus_") and column != target_col)
    exclude.update(column for column in frame.columns if column.endswith("_completeness"))
    feature_cols = [column for column in frame.columns if column not in exclude]
    categorical = [column for column in feature_cols if frame[column].dtype == "object" or str(frame[column].dtype).startswith("string")]
    numeric = [column for column in feature_cols if column not in categorical]
    return feature_cols, numeric, categorical


def run_training(
    dataset_path: Path = DEFAULT_DATASET,
    output_dir: Path = DEFAULT_OUTPUT,
    target_col: str = "target_load_t_plus_168h",
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path, encoding="utf-8-sig")
    for column in ["ts_utc", "ts_local", "date_local"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=(column == "ts_utc"))

    frame = frame.dropna(subset=[target_col]).sort_values(["ts_utc", "zone_id"]).reset_index(drop=True)
    train_frame, test_frame = _chronological_split(frame, time_col="ts_utc", test_fraction=test_fraction)
    feature_cols, numeric_cols, categorical_cols = _build_feature_lists(frame, target_col=target_col)

    x_train = train_frame[feature_cols]
    y_train = train_frame[target_col]
    x_test = test_frame[feature_cols]
    y_test = test_frame[target_col]

    numeric_pre = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pre = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    linear_preprocessor = ColumnTransformer(
        [("num", numeric_pre, numeric_cols), ("cat", categorical_pre, categorical_cols)],
        remainder="drop",
    )
    tree_preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ],
        remainder="drop",
    )

    models = {
        "ridge": Pipeline([("pre", linear_preprocessor), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline(
            [
                ("pre", tree_preprocessor),
                ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2)),
            ]
        ),
    }

    metrics: dict[str, Any] = {}
    predictions = test_frame[["ts_utc", "zone_id", target_col, "load_value"]].copy()

    naive_pred = test_frame["load_value"].to_numpy()
    metrics["naive_same_hour_last_week"] = _metrics(y_test, naive_pred)
    predictions["pred_naive_same_hour_last_week"] = naive_pred

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        metrics[model_name] = _metrics(y_test, pred)
        predictions[f"pred_{model_name}"] = pred

    zone_metrics: dict[str, Any] = {}
    for zone_id, zone_frame in predictions.groupby("zone_id"):
        zone_metrics[str(zone_id)] = {
            "naive_same_hour_last_week": _metrics(zone_frame[target_col], zone_frame["pred_naive_same_hour_last_week"]),
            "ridge": _metrics(zone_frame[target_col], zone_frame["pred_ridge"]),
            "random_forest": _metrics(zone_frame[target_col], zone_frame["pred_random_forest"]),
        }

    summary = {
        "dataset_path": str(dataset_path),
        "target_col": target_col,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "metrics": metrics,
        "per_zone_metrics": zone_metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", summary)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    return summary
