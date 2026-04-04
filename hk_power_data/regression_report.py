from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .catalog import ROOT


DEFAULT_METRICS = ROOT / "data" / "model" / "city_monthly_regression_validation" / "metrics.json"
DEFAULT_PREDICTIONS = ROOT / "data" / "model" / "city_monthly_regression_validation" / "predictions.csv"
DEFAULT_OUTPUT = ROOT / "data" / "model" / "city_monthly_regression_validation"


def _load_inputs(metrics_path: Path, predictions_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path, encoding="utf-8-sig", low_memory=False)
    predictions["period_month"] = pd.to_datetime(predictions["period_month"], errors="coerce")
    return metrics, predictions


def _build_report_markdown(metrics: dict[str, Any], predictions: pd.DataFrame) -> str:
    simple = metrics["simple_linear_regression"]
    multiple = metrics["multiple_linear_regression"]
    test_start = predictions["period_month"].min()
    test_end = predictions["period_month"].max()

    coefficient_items = sorted(
        multiple["coefficients"].items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:8]

    lines = [
        "# Monthly Regression Validation",
        "",
        "## Setup",
        "",
        f"- Dataset: `{metrics['dataset_path']}`",
        f"- Train rows: {metrics['train_rows']}",
        f"- Test rows: {metrics['test_rows']}",
        f"- Test window: {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}",
        f"- Target: `{metrics['target_col']}`",
        "",
        "## Models",
        "",
        f"- Simple linear regression: one feature `{simple['feature']}`",
        f"- Multiple linear regression: {len(multiple['features'])} features",
        "",
        "## Metrics",
        "",
        "| Model | MAE | RMSE | R2 | MAPE |",
        "|---|---:|---:|---:|---:|",
        f"| Simple linear regression | {simple['metrics']['mae']:.2f} | {simple['metrics']['rmse']:.2f} | {simple['metrics']['r2']:.4f} | {simple['metrics']['mape']:.4f} |",
        f"| Multiple linear regression | {multiple['metrics']['mae']:.2f} | {multiple['metrics']['rmse']:.2f} | {multiple['metrics']['r2']:.4f} | {multiple['metrics']['mape']:.4f} |",
        "",
        "## Interpretation",
        "",
        f"- The simple model already works reasonably well because `electricity_lag_12m` captures strong annual seasonality in Hong Kong monthly electricity consumption.",
        f"- The multiple model improves MAE from {simple['metrics']['mae']:.2f} to {multiple['metrics']['mae']:.2f} and improves R2 from {simple['metrics']['r2']:.4f} to {multiple['metrics']['r2']:.4f}.",
        "- This means monthly weather, mobility, transport, and seasonality features add useful signal beyond a single seasonal lag.",
        "",
        "## Most Influential Multiple-Regression Coefficients",
        "",
        "| Feature | Coefficient |",
        "|---|---:|",
    ]

    for feature, coefficient in coefficient_items:
        lines.append(f"| `{feature}` | {coefficient:.4f} |")

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- These are time-series regressions on a relatively small monthly test set, so coefficients should be interpreted cautiously.",
            "- Linear regression is useful here as an interpretable baseline, not necessarily as the final best-performing model.",
            "- The next step is to compare this against seasonal-naive and regularized models in the same monthly setup.",
            "",
            "## Artifacts",
            "",
            "- `actual_vs_pred.png`: test-window actual vs predicted monthly electricity",
            "- `metric_comparison.png`: MAE / RMSE / R2 comparison",
            "- `multiple_coefficients.png`: coefficient magnitudes for the multiple linear regression",
            "",
        ]
    )
    return "\n".join(lines)


def _plot_actual_vs_pred(predictions: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(predictions["period_month"], predictions["target_electricity_total_t_plus_1m"], label="Actual", color="#1f2937", linewidth=2.5)
    ax.plot(predictions["period_month"], predictions["pred_simple_linear_regression"], label="Simple Linear Regression", color="#2563eb", linewidth=2.0)
    ax.plot(predictions["period_month"], predictions["pred_multiple_linear_regression"], label="Multiple Linear Regression", color="#dc2626", linewidth=2.0)
    ax.set_title("Monthly Electricity Forecast on Test Window")
    ax.set_xlabel("Month")
    ax.set_ylabel("Electricity Consumption (TJ)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_metric_comparison(metrics: dict[str, Any], output_path: Path) -> None:
    simple_metrics = metrics["simple_linear_regression"]["metrics"]
    multiple_metrics = metrics["multiple_linear_regression"]["metrics"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metric_names = ["mae", "rmse", "r2"]
    labels = ["Simple", "Multiple"]
    colors = ["#2563eb", "#dc2626"]

    for axis, metric_name in zip(axes, metric_names, strict=True):
        values = [simple_metrics[metric_name], multiple_metrics[metric_name]]
        axis.bar(labels, values, color=colors, width=0.6)
        axis.set_title(metric_name.upper())
        axis.grid(axis="y", alpha=0.2)
        for idx, value in enumerate(values):
            axis.text(idx, value, f"{value:.2f}" if metric_name != "r2" else f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Regression Metric Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_multiple_coefficients(metrics: dict[str, Any], output_path: Path) -> None:
    coefficients = metrics["multiple_linear_regression"]["coefficients"]
    series = pd.Series(coefficients).sort_values(key=lambda values: values.abs(), ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#dc2626" if value < 0 else "#2563eb" for value in series]
    ax.barh(series.index, series.values, color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title("Multiple Linear Regression Coefficients")
    ax.set_xlabel("Coefficient")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_regression_report(
    metrics_path: Path = DEFAULT_METRICS,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, str]:
    metrics, predictions = _load_inputs(metrics_path=metrics_path, predictions_path=predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.md"
    actual_vs_pred_path = output_dir / "actual_vs_pred.png"
    metric_comparison_path = output_dir / "metric_comparison.png"
    coefficients_path = output_dir / "multiple_coefficients.png"

    report_path.write_text(_build_report_markdown(metrics, predictions), encoding="utf-8")
    _plot_actual_vs_pred(predictions, actual_vs_pred_path)
    _plot_metric_comparison(metrics, metric_comparison_path)
    _plot_multiple_coefficients(metrics, coefficients_path)

    return {
        "report_path": str(report_path),
        "actual_vs_pred_path": str(actual_vs_pred_path),
        "metric_comparison_path": str(metric_comparison_path),
        "multiple_coefficients_path": str(coefficients_path),
    }
