from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .catalog import ROOT
from .fetchers import write_json


DEFAULT_DATASET = ROOT / "data" / "model" / "city_monthly_train_ready.csv"
DEFAULT_OUTPUT = ROOT / "data" / "model" / "city_monthly_regression_validation"
DEFAULT_TARGET = "target_electricity_total_t_plus_1m"

SIMPLE_FEATURE_CANDIDATES = [
    "electricity_lag_12m",
    "electricity_lag_1m",
    "electricity_total_tj",
]

MULTIPLE_FEATURE_CANDIDATES = [
    "electricity_lag_1m",
    "electricity_lag_12m",
    "temp_mean_c_avg",
    "temp_max_c_peak",
    "rainfall_mm_sum",
    "public_holiday_days",
    "business_days",
    "immigration_total_cross_border_passengers",
    "public_transport_total_avg_daily_pax",
    "cross_harbour_total_pax",
    "building_consent_total_count",
    "occupation_permits_total_count",
    "occupation_permits_domestic_units",
    "building_completion_total_gfa",
    "building_completion_domestic_units",
    "gas_total_tj",
    "sin_month",
    "cos_month",
]


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    denominator = np.maximum(np.abs(y_true.to_numpy()), 1e-6)
    mape = float(np.mean(np.abs((y_true.to_numpy() - y_pred) / denominator)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape,
    }


def _chronological_split(frame: pd.DataFrame, time_col: str, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_times = pd.Series(pd.to_datetime(frame[time_col], errors="coerce")).sort_values().drop_duplicates().reset_index(drop=True)
    if len(ordered_times) < 2:
        raise ValueError("Need at least two distinct timestamps for regression validation.")
    cutoff_index = max(1, int(len(ordered_times) * (1 - test_fraction)))
    cutoff_index = min(cutoff_index, len(ordered_times) - 1)
    cutoff_time = ordered_times.iloc[cutoff_index]
    train = frame[pd.to_datetime(frame[time_col], errors="coerce") < cutoff_time].copy()
    test = frame[pd.to_datetime(frame[time_col], errors="coerce") >= cutoff_time].copy()
    return train, test


def _resolve_simple_feature(frame: pd.DataFrame) -> str:
    for column in SIMPLE_FEATURE_CANDIDATES:
        if column in frame.columns and frame[column].notna().any():
            return column
    raise KeyError(f"Could not find any simple-regression feature from: {', '.join(SIMPLE_FEATURE_CANDIDATES)}")


def _resolve_multiple_features(frame: pd.DataFrame) -> list[str]:
    features = [column for column in MULTIPLE_FEATURE_CANDIDATES if column in frame.columns and frame[column].notna().any()]
    if len(features) < 3:
        raise ValueError("Need at least three usable features for multiple linear regression.")
    return features


def _fit_linear_regression(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> tuple[Pipeline, np.ndarray]:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    return model, pred


def run_regression_validation(
    dataset_path: Path = DEFAULT_DATASET,
    output_dir: Path = DEFAULT_OUTPUT,
    target_col: str = DEFAULT_TARGET,
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path, encoding="utf-8-sig", low_memory=False)
    if "period_month" not in frame.columns:
        raise KeyError("Expected period_month in the monthly train-ready dataset.")
    frame["period_month"] = pd.to_datetime(frame["period_month"], errors="coerce")
    frame = frame.dropna(subset=["period_month", target_col]).sort_values("period_month").reset_index(drop=True)

    train_frame, test_frame = _chronological_split(frame, time_col="period_month", test_fraction=test_fraction)
    y_train = train_frame[target_col]
    y_test = test_frame[target_col]

    simple_feature = _resolve_simple_feature(frame)
    simple_model, simple_pred = _fit_linear_regression(
        train_frame[[simple_feature]],
        y_train,
        test_frame[[simple_feature]],
    )

    multiple_features = _resolve_multiple_features(frame)
    multiple_model, multiple_pred = _fit_linear_regression(
        train_frame[multiple_features],
        y_train,
        test_frame[multiple_features],
    )

    simple_lr = simple_model.named_steps["model"]
    multiple_lr = multiple_model.named_steps["model"]

    predictions = test_frame[["period_month", target_col]].copy()
    predictions["pred_simple_linear_regression"] = simple_pred
    predictions["pred_multiple_linear_regression"] = multiple_pred

    summary: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "target_col": target_col,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "simple_linear_regression": {
            "feature": simple_feature,
            "intercept": float(simple_lr.intercept_),
            "coefficient": float(simple_lr.coef_[0]),
            "metrics": _metrics(y_test, simple_pred),
        },
        "multiple_linear_regression": {
            "features": multiple_features,
            "intercept": float(multiple_lr.intercept_),
            "coefficients": {
                feature: float(coef)
                for feature, coef in zip(multiple_features, multiple_lr.coef_, strict=True)
            },
            "metrics": _metrics(y_test, multiple_pred),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", summary)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    return summary
