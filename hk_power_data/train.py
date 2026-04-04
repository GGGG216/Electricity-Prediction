from __future__ import annotations

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


DEFAULT_DATASET = ROOT / "data" / "model" / "city_monthly_train_ready.csv"
DEFAULT_OUTPUT = ROOT / "data" / "model" / "city_monthly_training_runs"


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


def _resolve_time_col(frame: pd.DataFrame) -> str:
    for column in ["period_month", "ts_utc", "ts_local", "date_local"]:
        if column in frame.columns:
            return column
    raise KeyError("Could not find a time column. Expected one of: period_month, ts_utc, ts_local, date_local.")


def _resolve_group_col(frame: pd.DataFrame) -> str | None:
    for column in ["zone_id", "building_id", "site_id", "meter_id"]:
        if column in frame.columns:
            return column
    return None


def _chronological_split(
    frame: pd.DataFrame,
    time_col: str,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_times = pd.Series(pd.to_datetime(frame[time_col], utc=False)).sort_values().drop_duplicates().reset_index(drop=True)
    if len(ordered_times) < 2:
        raise ValueError("Need at least two distinct timestamps for chronological split.")
    cutoff_index = max(1, int(len(ordered_times) * (1 - test_fraction)))
    cutoff_index = min(cutoff_index, len(ordered_times) - 1)
    cutoff_time = ordered_times.iloc[cutoff_index]
    train = frame[pd.to_datetime(frame[time_col], utc=False) < cutoff_time].copy()
    test = frame[pd.to_datetime(frame[time_col], utc=False) >= cutoff_time].copy()
    return train, test


def _build_feature_lists(frame: pd.DataFrame, target_col: str, time_col: str) -> tuple[list[str], list[str], list[str]]:
    exclude = {
        time_col,
        "target_period_month",
        "date_local",
        "ts_utc",
        "ts_local",
        "source_url",
        "source_file",
        "quality_flag",
        "load_unit",
        "measure_type",
        "holiday_name_en",
        "holiday_uid",
        "feature_year",
        target_col,
    }
    exclude.update(column for column in frame.columns if column.startswith("target_") and column != target_col)
    exclude.update(column for column in frame.columns if column.endswith("_completeness"))
    feature_cols = [column for column in frame.columns if column not in exclude]
    feature_cols = [column for column in feature_cols if not frame[column].isna().all()]
    feature_cols = [column for column in feature_cols if frame[column].nunique(dropna=False) > 1]
    categorical = [column for column in feature_cols if frame[column].dtype == "object" or str(frame[column].dtype).startswith("string")]
    numeric = [column for column in feature_cols if column not in categorical]
    return feature_cols, numeric, categorical


def _resolve_naive_baseline(frame: pd.DataFrame) -> tuple[str, np.ndarray]:
    baseline_candidates = [
        ("naive_same_month_last_year", "electricity_lag_12m"),
        ("naive_previous_month", "electricity_lag_1m"),
        ("naive_same_hour_last_week", "load_lag_168h"),
        ("naive_current_value", "electricity_total_tj"),
        ("naive_current_value", "load_value"),
    ]
    for label, column in baseline_candidates:
        if column in frame.columns:
            return label, frame[column].to_numpy()
    raise KeyError("Could not resolve a naive baseline column for the dataset.")


def run_training(
    dataset_path: Path = DEFAULT_DATASET,
    output_dir: Path = DEFAULT_OUTPUT,
    target_col: str = "target_electricity_total_t_plus_1m",
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path, encoding="utf-8-sig", low_memory=False)
    for column in ["period_month", "target_period_month", "ts_utc", "ts_local", "date_local"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    time_col = _resolve_time_col(frame)
    group_col = _resolve_group_col(frame)
    sort_cols = [time_col] + ([group_col] if group_col else [])
    frame = frame.dropna(subset=[target_col]).sort_values(sort_cols).reset_index(drop=True)
    train_frame, test_frame = _chronological_split(frame, time_col=time_col, test_fraction=test_fraction)
    feature_cols, numeric_cols, categorical_cols = _build_feature_lists(frame, target_col=target_col, time_col=time_col)

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
    prediction_cols = [time_col]
    if group_col:
        prediction_cols.append(group_col)
    prediction_cols.append(target_col)
    for column in ["electricity_total_tj", "electricity_lag_12m", "electricity_lag_1m", "load_value", "load_lag_168h"]:
        if column in test_frame.columns and column not in prediction_cols:
            prediction_cols.append(column)
    predictions = test_frame[prediction_cols].copy()

    baseline_name, naive_pred = _resolve_naive_baseline(test_frame)
    metrics[baseline_name] = _metrics(y_test, naive_pred)
    predictions[f"pred_{baseline_name}"] = naive_pred

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        metrics[model_name] = _metrics(y_test, pred)
        predictions[f"pred_{model_name}"] = pred

    summary: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "target_col": target_col,
        "time_col": time_col,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "metrics": metrics,
    }

    if group_col:
        group_metrics: dict[str, Any] = {}
        for group_value, group_frame in predictions.groupby(group_col):
            group_metrics[str(group_value)] = {}
            for pred_col in [column for column in predictions.columns if column.startswith("pred_")]:
                group_metrics[str(group_value)][pred_col] = _metrics(group_frame[target_col], group_frame[pred_col])
        summary["group_col"] = group_col
        summary["per_group_metrics"] = group_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", summary)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    return summary
