from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import requests


DEFAULT_HEADERS = {
    "User-Agent": "hk-power-data-collector/0.1 (+https://data.gov.hk/)",
    "Accept": "*/*",
}


def batched(items: Iterable[int], batch_size: int) -> Iterable[list[int]]:
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def request_json_with_retry(
    method: str,
    url: str,
    *,
    attempts: int = 5,
    backoff_seconds: float = 2.0,
    **kwargs: Any,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.request(method=method, url=url, **kwargs)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(backoff_seconds * attempt)
    assert last_error is not None
    raise last_error


def fetch_static_url(source: dict[str, Any], destination: Path) -> dict[str, Any]:
    response = requests.get(source["url"], headers=DEFAULT_HEADERS, timeout=120)
    response.raise_for_status()
    ensure_parent(destination)
    destination.write_bytes(response.content)
    return {
        "bytes_written": destination.stat().st_size,
        "content_type": response.headers.get("content-type"),
        "source_url": source["url"],
    }


def fetch_censtatd_api(source: dict[str, Any], destination: Path) -> dict[str, Any]:
    headers = {
        **DEFAULT_HEADERS,
        "Referer": "https://data.gov.hk/",
        "Accept": "application/json, text/plain, */*",
    }
    response = requests.get(
        "https://www.censtatd.gov.hk/api/get.php",
        params={"id": source["table_id"], "lang": source.get("lang", "en"), "full_series": "1"},
        headers=headers,
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    write_json(destination, payload)
    return {
        "source_url": "https://www.censtatd.gov.hk/api/get.php",
        "table_id": source["table_id"],
        "top_level_keys": sorted(payload.keys()),
    }


def _arcgis_object_ids(service_url: str) -> tuple[dict[str, Any], list[int]]:
    metadata = request_json_with_retry(
        "GET",
        service_url,
        params={"f": "json"},
        headers=DEFAULT_HEADERS,
        timeout=120,
    )
    ids_payload = request_json_with_retry(
        "GET",
        f"{service_url}/query",
        params={"f": "json", "where": "1=1", "returnIdsOnly": "true"},
        headers=DEFAULT_HEADERS,
        timeout=120,
    )
    object_ids = sorted(ids_payload.get("objectIds", []))
    return metadata, object_ids


def fetch_arcgis_layer(
    source: dict[str, Any],
    destination: Path,
    max_records: int | None = None,
) -> dict[str, Any]:
    service_url = source["service_url"]
    metadata, object_ids = _arcgis_object_ids(service_url)
    if max_records is not None:
        object_ids = object_ids[:max_records]

    batch_size = int(source.get("batch_size") or metadata.get("maxRecordCount") or 1000)
    batch_size = max(1, min(batch_size, 2000))
    feature_count = 0
    temp_destination = destination.with_suffix(destination.suffix + ".tmp")
    ensure_parent(temp_destination)

    with temp_destination.open("w", encoding="utf-8") as handle:
        handle.write('{"type":"FeatureCollection","features":[')
        first_feature = True
        for chunk in batched(object_ids, batch_size):
            payload = request_json_with_retry(
                "POST",
                f"{service_url}/query",
                data={
                    "f": source.get("format", "geojson"),
                    "objectIds": ",".join(str(value) for value in chunk),
                    "outFields": "*",
                    "returnGeometry": "true",
                    "outSR": "4326",
                },
                headers=DEFAULT_HEADERS,
                timeout=180,
                attempts=6,
                backoff_seconds=3.0,
            )
            features = payload.get("features", [])
            for feature in features:
                if not first_feature:
                    handle.write(",")
                json.dump(feature, handle, ensure_ascii=False)
                first_feature = False
            feature_count += len(features)
        handle.write("]}")

    temp_destination.replace(destination)
    return {
        "source_url": service_url,
        "feature_count": feature_count,
        "service_name": metadata.get("name"),
    }


def collect_source(
    source: dict[str, Any],
    output_root: Path,
    max_records: int | None = None,
) -> dict[str, Any]:
    stamp = timestamp_slug()
    source_dir = output_root / source["name"]
    destination = source_dir / f"{stamp}_{source['file_name']}"

    if source["kind"] == "static_url":
        details = fetch_static_url(source, destination)
        status = "downloaded"
    elif source["kind"] == "censtatd_api":
        details = fetch_censtatd_api(source, destination)
        status = "downloaded"
    elif source["kind"] == "arcgis_layer":
        details = fetch_arcgis_layer(source, destination, max_records=max_records)
        status = "downloaded"
    elif source["kind"] == "manual":
        details = {
            "status_reason": "manual source",
            "expected_headers": source.get("expected_headers", []),
        }
        status = "manual"
    else:
        raise ValueError(f"Unsupported source kind: {source['kind']}")

    metadata = {
        "source_name": source["name"],
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "priority": source["priority"],
        "group": source["group"],
        "description": source["description"],
        "destination": str(destination) if status == "downloaded" else None,
        "details": details,
    }

    meta_path = source_dir / f"{stamp}_{Path(source['file_name']).stem}.meta.json"
    write_json(meta_path, metadata)
    return metadata
