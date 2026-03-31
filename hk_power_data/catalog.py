from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "config" / "sources.json"


def load_catalog() -> dict[str, dict[str, Any]]:
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {item["name"]: item for item in raw}


def list_groups(catalog: dict[str, dict[str, Any]]) -> list[str]:
    return sorted({item["group"] for item in catalog.values()})


def select_sources(
    catalog: dict[str, dict[str, Any]],
    source_names: list[str] | None = None,
    groups: list[str] | None = None,
    include_manual: bool = False,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    name_filter = set(source_names or [])
    group_filter = set(groups or [])

    for item in catalog.values():
        if item["kind"] == "manual" and not include_manual:
            continue
        if name_filter and item["name"] not in name_filter:
            continue
        if group_filter and item["group"] not in group_filter:
            continue
        if not name_filter and not group_filter and item["group"] != "core_exogenous":
            continue
        selected.append(item)

    missing = sorted(name_filter - {item["name"] for item in selected})
    if missing:
        raise KeyError(f"Unknown source names: {', '.join(missing)}")

    return selected

