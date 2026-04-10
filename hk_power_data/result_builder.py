from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .catalog import ROOT


MODEL_ROOT = ROOT / "data" / "model"
SILVER_ROOT = ROOT / "data" / "silver"
RESULT_ROOT = ROOT / "result"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_monthly_target(frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frame["period_month"], frame["electricity_total_tj"], color="#1f2937", linewidth=2.4)
    ax.set_title("Hong Kong Monthly Electricity Consumption")
    ax.set_xlabel("Month")
    ax.set_ylabel("Electricity Consumption (TJ)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_feature_availability(frame: pd.DataFrame, output_path: Path) -> None:
    feature_cols = [
        "immigration_total_cross_border_passengers",
        "public_transport_total_avg_daily_pax",
        "cross_harbour_total_pax",
        "occupation_permits_total_count",
        "building_completion_total_gfa",
        "hkelectric_re_generation_grand_total",
        "population",
    ]
    available_cols = [column for column in feature_cols if column in frame.columns]
    availability = pd.DataFrame({"period_month": frame["period_month"]})
    for column in available_cols:
        availability[column] = frame[column].notna().astype(int)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for column in available_cols:
        ax.plot(availability["period_month"], availability[column], label=column, linewidth=1.8)
    ax.set_title("Monthly Feature Availability")
    ax.set_xlabel("Month")
    ax.set_ylabel("Available (1=yes)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_top_correlations(correlations: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#dc2626" if value < 0 else "#2563eb" for value in correlations]
    ax.barh(correlations.index, correlations.values, color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title("Top Correlations with Next-Month Electricity Target")
    ax.set_xlabel("Pearson Correlation")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_baseline_metric_comparison(metrics: dict[str, Any], output_path: Path) -> None:
    metric_names = ["mae", "rmse", "r2"]
    labels = list(metrics.keys())
    colors = ["#6b7280", "#2563eb", "#059669"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric_name in zip(axes, metric_names, strict=True):
        values = [metrics[label][metric_name] for label in labels]
        axis.bar(labels, values, color=colors[: len(labels)], width=0.65)
        axis.set_title(metric_name.upper())
        axis.grid(axis="y", alpha=0.2)
        for idx, value in enumerate(values):
            formatted = f"{value:.2f}" if metric_name != "r2" else f"{value:.3f}"
            axis.text(idx, value, formatted, ha="center", va="bottom", fontsize=9)
        axis.tick_params(axis="x", rotation=15)

    fig.suptitle("Baseline Model Comparison on Test Window")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_baseline_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        predictions["period_month"],
        predictions["target_electricity_total_t_plus_1m"],
        label="Actual",
        color="#111827",
        linewidth=2.5,
    )
    plot_specs = [
        ("pred_naive_same_month_last_year", "Seasonal Naive", "#6b7280"),
        ("pred_ridge", "Ridge", "#2563eb"),
        ("pred_random_forest", "Random Forest", "#059669"),
    ]
    for column, label, color in plot_specs:
        if column in predictions.columns:
            ax.plot(predictions["period_month"], predictions[column], label=label, color=color, linewidth=2.0)
    ax.set_title("Monthly Test-Window Forecasts")
    ax.set_xlabel("Month")
    ax.set_ylabel("Electricity Consumption (TJ)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _dataset_profile(train_ready: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"item": "train_ready_rows", "value": int(len(train_ready))},
        {"item": "train_ready_columns", "value": int(len(train_ready.columns))},
        {"item": "start_month", "value": str(train_ready["period_month"].min().date())},
        {"item": "end_month", "value": str(train_ready["period_month"].max().date())},
        {"item": "target_mean_tj", "value": float(train_ready["target_electricity_total_t_plus_1m"].mean())},
        {"item": "target_std_tj", "value": float(train_ready["target_electricity_total_t_plus_1m"].std())},
        {"item": "target_min_tj", "value": float(train_ready["target_electricity_total_t_plus_1m"].min())},
        {"item": "target_max_tj", "value": float(train_ready["target_electricity_total_t_plus_1m"].max())},
    ]
    return pd.DataFrame(rows)


def _top_correlations(train_ready: pd.DataFrame) -> pd.Series:
    candidate_cols = [
        column
        for column in train_ready.columns
        if column not in {"period_month", "target_period_month", "source_dataset", "target_electricity_total_t_plus_1m"}
    ]
    numeric = train_ready[candidate_cols].select_dtypes(include=["number"]).copy()
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    correlations = numeric.corrwith(train_ready["target_electricity_total_t_plus_1m"]).dropna()
    correlations = correlations.loc[correlations.abs().sort_values(ascending=False).index]
    return correlations.head(12)


def _error_handling_rows() -> list[dict[str, str]]:
    return [
        {
            "issue": "Mixed time granularity across sources",
            "handling": "Daily sources are aggregated to month, annual sources are merged through feature_year, and static sources are broadcast to all months.",
        },
        {
            "issue": "Target and feature coverage start at different dates",
            "handling": "The train-ready monthly window starts at 2013-01 and availability flags are preserved for immigration, transport, renewable generation, and population features.",
        },
        {
            "issue": "HK Electric renewable file is mislabeled as CSV",
            "handling": "The ETL checks the file signature and switches to read_excel when the payload is actually an XLSX workbook.",
        },
        {
            "issue": "C&SD period field mixes annual, quarterly, and monthly values",
            "handling": "The ETL parses the period string explicitly and only keeps valid monthly rows for the city-level target table.",
        },
        {
            "issue": "Population projection is annual and limited in time range",
            "handling": "The merge clips feature_year to the available projection range before joining the monthly model table.",
        },
    ]


def _build_markdown(
    dataset_profile: pd.DataFrame,
    top_correlations: pd.Series,
    regression_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    silver_manifest: dict[str, Any],
) -> str:
    simple = regression_metrics["simple_linear_regression"]
    multiple = regression_metrics["multiple_linear_regression"]
    baseline_rows = baseline_metrics.get("metrics", {})
    lines = [
        "# Result Package",
        "",
        "## Project Scope",
        "",
        "This result package summarizes the monthly Hong Kong electricity forecasting workflow built from official public data.",
        "",
        "## Dataset Overview",
        "",
        "| Item | Value |",
        "|---|---:|",
    ]

    for _, row in dataset_profile.iterrows():
        lines.append(f"| {row['item']} | {row['value']} |")

    lines.extend(
        [
            "",
            "## Silver Build Summary",
            "",
            "| Table | Rows |",
            "|---|---:|",
        ]
    )

    for name, count in silver_manifest.get("built", {}).items():
        lines.append(f"| {name} | {count} |")

    lines.extend(
        [
            "",
            "## Baseline Model Summary",
            "",
            "| Model | MAE | RMSE | R2 | MAPE |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for model_name, metric_values in baseline_rows.items():
        pretty_name = model_name.replace("_", " ").title()
        lines.append(
            f"| {pretty_name} | {metric_values['mae']:.2f} | {metric_values['rmse']:.2f} | {metric_values['r2']:.4f} | {metric_values['mape']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Regression Summary",
            "",
            "| Model | MAE | RMSE | R2 | MAPE |",
            "|---|---:|---:|---:|---:|",
            f"| Simple linear regression | {simple['metrics']['mae']:.2f} | {simple['metrics']['rmse']:.2f} | {simple['metrics']['r2']:.4f} | {simple['metrics']['mape']:.4f} |",
            f"| Multiple linear regression | {multiple['metrics']['mae']:.2f} | {multiple['metrics']['rmse']:.2f} | {multiple['metrics']['r2']:.4f} | {multiple['metrics']['mape']:.4f} |",
            "",
            "## Interpretation",
            "",
            f"- The simple regression uses `{simple['feature']}` and captures annual seasonality reasonably well.",
            f"- The multiple regression improves MAE by {simple['metrics']['mae'] - multiple['metrics']['mae']:.2f} TJ and raises R2 from {simple['metrics']['r2']:.4f} to {multiple['metrics']['r2']:.4f}.",
            "- This indicates that monthly weather, mobility, transport, building activity, and seasonal features contribute meaningful signal beyond a single lag baseline.",
            "",
            "## Top Correlated Features",
            "",
            "| Feature | Correlation |",
            "|---|---:|",
        ]
    )

    for feature, value in top_correlations.items():
        lines.append(f"| `{feature}` | {value:.4f} |")

    lines.extend(
        [
            "",
            "## Errors Encountered and Handling",
            "",
            "| Issue | Handling |",
            "|---|---|",
        ]
    )
    for row in _error_handling_rows():
        lines.append(f"| {row['issue']} | {row['handling']} |")

    lines.extend(
        [
            "",
            "## Visualizations",
            "",
            "- `figures/monthly_target_overview.png`",
            "- `figures/feature_availability.png`",
            "- `figures/top_correlations.png`",
            "- `figures/baseline_metric_comparison.png`",
            "- `figures/baseline_test_window.png`",
            "- `figures/actual_vs_pred.png`",
            "- `figures/metric_comparison.png`",
            "- `figures/multiple_coefficients.png`",
            "",
        ]
    )
    return "\n".join(lines)


def build_result_package(result_root: Path = RESULT_ROOT) -> dict[str, str]:
    train_ready_path = MODEL_ROOT / "city_monthly_train_ready.csv"
    silver_manifest_path = SILVER_ROOT / "silver_manifest.json"
    baseline_metrics_path = MODEL_ROOT / "city_monthly_training_runs" / "metrics.json"
    baseline_predictions_path = MODEL_ROOT / "city_monthly_training_runs" / "predictions.csv"
    regression_metrics_path = MODEL_ROOT / "city_monthly_regression_validation" / "metrics.json"
    regression_predictions_path = MODEL_ROOT / "city_monthly_regression_validation" / "predictions.csv"
    regression_dir = MODEL_ROOT / "city_monthly_regression_validation"

    if not train_ready_path.exists():
        raise FileNotFoundError(f"Missing train-ready dataset: {train_ready_path}")

    train_ready = pd.read_csv(train_ready_path, encoding="utf-8-sig", low_memory=False)
    train_ready["period_month"] = pd.to_datetime(train_ready["period_month"], errors="coerce")
    silver_manifest = _load_json(silver_manifest_path)
    baseline_metrics = _load_json(baseline_metrics_path)
    baseline_predictions = pd.read_csv(baseline_predictions_path, encoding="utf-8-sig", low_memory=False)
    baseline_predictions["period_month"] = pd.to_datetime(baseline_predictions["period_month"], errors="coerce")
    regression_metrics = _load_json(regression_metrics_path)
    _ = pd.read_csv(regression_predictions_path, encoding="utf-8-sig", low_memory=False)

    figures_dir = result_root / "figures"
    tables_dir = result_root / "tables"
    _ensure_dir(figures_dir)
    _ensure_dir(tables_dir)

    dataset_profile = _dataset_profile(train_ready)
    top_correlations = _top_correlations(train_ready)

    dataset_profile.to_csv(tables_dir / "dataset_profile.csv", index=False, encoding="utf-8-sig")
    top_correlations.rename("correlation").rename_axis("feature").reset_index().to_csv(
        tables_dir / "top_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(_error_handling_rows()).to_csv(tables_dir / "error_handling.csv", index=False, encoding="utf-8-sig")

    _plot_monthly_target(train_ready, figures_dir / "monthly_target_overview.png")
    _plot_feature_availability(train_ready, figures_dir / "feature_availability.png")
    _plot_top_correlations(top_correlations.sort_values(), figures_dir / "top_correlations.png")
    _plot_baseline_metric_comparison(baseline_metrics["metrics"], figures_dir / "baseline_metric_comparison.png")
    _plot_baseline_predictions(baseline_predictions, figures_dir / "baseline_test_window.png")

    for name in ["actual_vs_pred.png", "metric_comparison.png", "multiple_coefficients.png", "report.md", "metrics.json", "predictions.csv"]:
        source = regression_dir / name
        if source.exists():
            shutil.copy2(source, result_root / name if source.suffix in {".md", ".json", ".csv"} else figures_dir / name)

    shutil.copy2(baseline_metrics_path, result_root / "baseline_metrics.json")
    shutil.copy2(baseline_predictions_path, result_root / "baseline_predictions.csv")

    readme_path = result_root / "README.md"
    readme_path.write_text(
        _build_markdown(
            dataset_profile=dataset_profile,
            top_correlations=top_correlations,
            regression_metrics=regression_metrics,
            baseline_metrics=baseline_metrics,
            silver_manifest=silver_manifest,
        ),
        encoding="utf-8",
    )

    return {
        "result_root": str(result_root),
        "readme_path": str(readme_path),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
    }
