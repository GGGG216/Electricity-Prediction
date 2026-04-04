from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .catalog import ROOT
from .fetchers import write_json


DEFAULT_INPUT = ROOT / "data" / "model" / "city_monthly_training_runs" / "predictions.csv"
DEFAULT_OUTPUT = ROOT / "data" / "model" / "city_monthly_evaluation"


def compute_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    actual = pd.to_numeric(y_true, errors="coerce")
    pred = pd.to_numeric(y_pred, errors="coerce")
    valid = ~(actual.isna() | pred.isna())
    actual = actual[valid].astype(float)
    pred = pred[valid].astype(float)
    if actual.empty:
        raise ValueError("No valid rows available for evaluation.")

    error = actual - pred
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    actual_mean = float(np.mean(actual))
    ss_res = float(np.sum(np.square(error)))
    ss_tot = float(np.sum(np.square(actual - actual_mean)))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "row_count": int(len(actual)),
    }


def evaluate_predictions(
    input_path: Path = DEFAULT_INPUT,
    actual_col: str = "target_electricity_total_t_plus_1m",
    prediction_cols: list[str] | None = None,
    group_col: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {input_path}")

    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    if actual_col not in frame.columns:
        raise KeyError(f"Actual column not found: {actual_col}")

    if prediction_cols is None:
        prediction_cols = [column for column in frame.columns if column.startswith("pred_")]
    if not prediction_cols:
        raise ValueError("No prediction columns provided or detected.")

    for column in prediction_cols:
        if column not in frame.columns:
            raise KeyError(f"Prediction column not found: {column}")

    summary: dict[str, Any] = {
        "input_path": str(input_path),
        "actual_col": actual_col,
        "prediction_cols": prediction_cols,
        "overall": {},
        "by_group": {},
    }

    for pred_col in prediction_cols:
        summary["overall"][pred_col] = compute_regression_metrics(frame[actual_col], frame[pred_col])

    if group_col:
        if group_col not in frame.columns:
            raise KeyError(f"Group column not found: {group_col}")
        grouped_summary: dict[str, Any] = {}
        for group_value, group_frame in frame.groupby(group_col):
            grouped_summary[str(group_value)] = {}
            for pred_col in prediction_cols:
                grouped_summary[str(group_value)][pred_col] = compute_regression_metrics(
                    group_frame[actual_col],
                    group_frame[pred_col],
                )
        summary["group_col"] = group_col
        summary["by_group"] = grouped_summary

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", summary)
    return summary
