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
    unknown_names = sorted(name_filter - set(catalog))
    if unknown_names:
        raise KeyError(f"Unknown source names: {', '.join(unknown_names)}")

    for item in catalog.values():
        if item["kind"] == "manual" and not include_manual:
            continue
        if not name_filter and not group_filter and item["group"] != "core_exogenous":
            continue
        if name_filter or group_filter:
            if item["name"] not in name_filter and item["group"] not in group_filter:
                continue
        selected.append(item)

    return selected
