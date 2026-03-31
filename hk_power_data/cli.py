from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .catalog import ROOT, list_groups, load_catalog, select_sources
from .fetchers import collect_source, timestamp_slug, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hk_power_data",
        description="Collect official data sources for the Hong Kong weekly load curve project.",
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        raise SystemExit(handle_list(args.include_manual))
    if args.command == "collect":
        raise SystemExit(handle_collect(args))

    parser.error(f"Unknown command: {args.command}")

