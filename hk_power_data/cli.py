from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .catalog import ROOT, list_groups, load_catalog, select_sources
from .evaluate import evaluate_predictions
from .fetchers import collect_source, timestamp_slug, write_json
from .regression_validation import run_regression_validation
from .regression_report import build_regression_report
from .result_builder import build_result_package
from .train import run_training
from .transform import MODEL_ROOT, RAW_ROOT, SILVER_ROOT, build_silver_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hk_power_data",
        description="Collect and model official monthly electricity data for Hong Kong.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available data sources.")
    list_parser.add_argument("--include-manual", action="store_true")

    collect_parser = subparsers.add_parser("collect", help="Download a subset of sources.")
    collect_parser.add_argument("--source", action="append", dest="sources", help="Specific source name.")
    collect_parser.add_argument("--group", action="append", dest="groups", help="Specific source group.")
    collect_parser.add_argument("--include-manual", action="store_true", help="Include manual placeholders.")
    collect_parser.add_argument(
        "--output-root",
        default=str(ROOT / "data" / "raw"),
        help="Directory for collected files.",
    )
    collect_parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for ArcGIS feature downloads. Useful for smoke tests.",
    )
    collect_parser.add_argument("--dry-run", action="store_true", help="Resolve selection only.")

    silver_parser = subparsers.add_parser("silver", help="Transform latest raw payloads into monthly silver/model tables.")
    silver_parser.add_argument("--raw-root", default=str(RAW_ROOT))
    silver_parser.add_argument("--silver-root", default=str(SILVER_ROOT))
    silver_parser.add_argument("--model-root", default=str(MODEL_ROOT))

    train_parser = subparsers.add_parser("train", help="Train baseline models on the train-ready dataset.")
    train_parser.add_argument(
        "--dataset",
        default=str(MODEL_ROOT / "city_monthly_train_ready.csv"),
        help="Input dataset CSV.",
    )
    train_parser.add_argument(
        "--output-dir",
        default=str(MODEL_ROOT / "city_monthly_training_runs"),
        help="Directory for metrics and predictions.",
    )
    train_parser.add_argument("--target-col", default="target_electricity_total_t_plus_1m")
    train_parser.add_argument("--test-fraction", type=float, default=0.2)

    regression_parser = subparsers.add_parser(
        "validate-regression",
        help="Run simple linear regression and multiple linear regression on the monthly dataset.",
    )
    regression_parser.add_argument(
        "--dataset",
        default=str(MODEL_ROOT / "city_monthly_train_ready.csv"),
        help="Input dataset CSV.",
    )
    regression_parser.add_argument(
        "--output-dir",
        default=str(MODEL_ROOT / "city_monthly_regression_validation"),
        help="Directory for regression metrics and predictions.",
    )
    regression_parser.add_argument("--target-col", default="target_electricity_total_t_plus_1m")
    regression_parser.add_argument("--test-fraction", type=float, default=0.2)

    report_parser = subparsers.add_parser(
        "report-regression",
        help="Generate a markdown report and plots for the monthly regression validation results.",
    )
    report_parser.add_argument(
        "--metrics",
        default=str(MODEL_ROOT / "city_monthly_regression_validation" / "metrics.json"),
        help="Regression metrics JSON.",
    )
    report_parser.add_argument(
        "--predictions",
        default=str(MODEL_ROOT / "city_monthly_regression_validation" / "predictions.csv"),
        help="Regression predictions CSV.",
    )
    report_parser.add_argument(
        "--output-dir",
        default=str(MODEL_ROOT / "city_monthly_regression_validation"),
        help="Directory for the markdown report and figures.",
    )

    result_parser = subparsers.add_parser(
        "build-results",
        help="Build the final result folder with analysis, error handling, and figures.",
    )
    result_parser.add_argument(
        "--output-dir",
        default=str(ROOT / "result"),
        help="Directory for the final result package.",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate prediction files with MAE / RMSE / R².")
    evaluate_parser.add_argument(
        "--input",
        default=str(MODEL_ROOT / "city_monthly_training_runs" / "predictions.csv"),
        help="CSV containing actual and prediction columns.",
    )
    evaluate_parser.add_argument("--actual-col", default="target_electricity_total_t_plus_1m")
    evaluate_parser.add_argument(
        "--pred-col",
        action="append",
        dest="prediction_cols",
        help="Prediction column to evaluate. Repeatable. Defaults to all columns starting with pred_.",
    )
    evaluate_parser.add_argument(
        "--group-col",
        default="",
        help="Optional grouping column for per-group metrics. Use empty string to disable.",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        default=str(MODEL_ROOT / "city_monthly_evaluation"),
        help="Directory for evaluation outputs.",
    )

    return parser


def handle_list(include_manual: bool) -> int:
    catalog = load_catalog()
    print("Groups:", ", ".join(list_groups(catalog)))
    for item in catalog.values():
        if item["kind"] == "manual" and not include_manual:
            continue
        print(
            json.dumps(
                {
                    "name": item["name"],
                    "group": item["group"],
                    "kind": item["kind"],
                    "priority": item["priority"],
                    "cadence": item["cadence"],
                },
                ensure_ascii=False,
            )
        )
    return 0


def handle_collect(args: argparse.Namespace) -> int:
    catalog = load_catalog()
    chosen = select_sources(
        catalog=catalog,
        source_names=args.sources,
        groups=args.groups,
        include_manual=args.include_manual,
    )

    if args.dry_run:
        for item in chosen:
            print(item["name"])
        return 0

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{timestamp_slug()}"
    results: list[dict[str, Any]] = []

    for item in chosen:
        print(f"[collect] {item['name']}")
        result = collect_source(item, output_root=output_root, max_records=args.max_records)
        results.append(result)

    run_manifest = {
        "run_id": run_id,
        "output_root": str(output_root),
        "selected_sources": [item["name"] for item in chosen],
        "results": results,
    }
    write_json(output_root / "_runs" / f"{run_id}.json", run_manifest)
    print(f"[done] wrote manifest for {len(results)} sources")
    return 0


def handle_silver(args: argparse.Namespace) -> int:
    summary = build_silver_tables(
        raw_root=Path(args.raw_root),
        silver_root=Path(args.silver_root),
        model_root=Path(args.model_root),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def handle_train(args: argparse.Namespace) -> int:
    summary = run_training(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        target_col=args.target_col,
        test_fraction=args.test_fraction,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def handle_validate_regression(args: argparse.Namespace) -> int:
    summary = run_regression_validation(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        target_col=args.target_col,
        test_fraction=args.test_fraction,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def handle_report_regression(args: argparse.Namespace) -> int:
    outputs = build_regression_report(
        metrics_path=Path(args.metrics),
        predictions_path=Path(args.predictions),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


def handle_build_results(args: argparse.Namespace) -> int:
    outputs = build_result_package(result_root=Path(args.output_dir))
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


def handle_evaluate(args: argparse.Namespace) -> int:
    summary = evaluate_predictions(
        input_path=Path(args.input),
        actual_col=args.actual_col,
        prediction_cols=args.prediction_cols,
        group_col=args.group_col or None,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        raise SystemExit(handle_list(args.include_manual))
    if args.command == "collect":
        raise SystemExit(handle_collect(args))
    if args.command == "silver":
        raise SystemExit(handle_silver(args))
    if args.command == "train":
        raise SystemExit(handle_train(args))
    if args.command == "validate-regression":
        raise SystemExit(handle_validate_regression(args))
    if args.command == "report-regression":
        raise SystemExit(handle_report_regression(args))
    if args.command == "build-results":
        raise SystemExit(handle_build_results(args))
    if args.command == "evaluate":
        raise SystemExit(handle_evaluate(args))

    parser.error(f"Unknown command: {args.command}")
