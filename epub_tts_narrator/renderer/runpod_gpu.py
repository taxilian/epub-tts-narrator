"""Managed GPU selection policy for Runpod deployments.

Pure catalog logic with no network or saved-state access. Each call ranks
one cloud (Secure or Community); the caller walks runpod.cloud_types in order. The lifecycle caller
persists the chosen identity before creating any Pod, so an ambiguous create
can never re-resolve to a different GPU. Unknown or malformed catalog data
never selects a GPU: preferred H100 candidates are skipped in favor of the
configured fallback, and a fallback with unknown availability or unreadable
pricing fails closed before any resource is created.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Exact catalog IDs from the public Runpod GPU type list. The fallback is the
# default gpu_type_id; it is never guessed from similar names.
DEFAULT_GPU_TYPE_ID = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
H100_CANDIDATE_IDS = frozenset({
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 NVL",
})
# The preferred H100 must beat this ceiling outright, independent of the
# configured per-hour cap (which may be higher or lower).
PREFERRED_H100_HOURLY_LIMIT = 2.60
_AVAILABLE = frozenset({"HIGH", "MEDIUM", "LOW"})
# Runpod's two Pod clouds, as the v2 API names them, and their catalog price keys.
CLOUD_PRICE_KEYS = {"SECURE": "secure", "COMMUNITY": "community"}
CLOUD_LABELS = {"SECURE": "Secure Cloud", "COMMUNITY": "Community Cloud"}


def catalog_price(gpu, cloud="SECURE"):
    """Return the cloud's hourly price when a positive finite number, else None."""
    price = gpu.get("price")
    price = price.get(CLOUD_PRICE_KEYS[cloud]) if isinstance(price, dict) else None
    if (isinstance(price, bool) or not isinstance(price, (int, float))
            or not math.isfinite(price) or price <= 0):
        return None
    return price


def available_in(gpu, data_center_id):
    """True when overall and configured-datacenter availability are known, not NONE."""
    availability = gpu.get("availability")
    centers = gpu.get("dataCenters")
    if not isinstance(availability, str) or availability not in _AVAILABLE or not isinstance(centers, list):
        return False
    return any(isinstance(center, dict) and center.get("id") == data_center_id
               and isinstance(center.get("availability"), str)
               and center.get("availability") in _AVAILABLE
               for center in centers)


def select_gpu(rows, *, fallback_id, prefer_h100, data_center_id, max_hourly_price,
               h100_hourly_limit=PREFERRED_H100_HOURLY_LIMIT, cloud="SECURE"):
    """Choose (gpu_id, price) for one managed allocation, or raise ValueError.

    When prefer_h100 is set, any available H100 candidate strictly under
    h100_hourly_limit and within max_hourly_price wins; the cheapest is taken,
    breaking ties by catalog ID. Otherwise the exact fallback GPU must be
    available in the configured datacenter at or under max_hourly_price.
    """
    if not data_center_id:
        raise ValueError("Set runpod.data_center_id to an available datacenter for the configured GPU")
    rows = [row for row in rows or [] if isinstance(row, dict) and isinstance(row.get("id"), str)]
    if prefer_h100:
        candidates = []
        for gpu in rows:
            if gpu["id"] not in H100_CANDIDATE_IDS or not available_in(gpu, data_center_id):
                continue
            price = catalog_price(gpu, cloud)
            if price is not None and price < h100_hourly_limit and price <= max_hourly_price:
                candidates.append((price, gpu["id"]))
        if candidates:
            price, gpu_id = min(candidates)
            return gpu_id, price
    gpu = next((row for row in rows if row["id"] == fallback_id), None)
    price = catalog_price(gpu, cloud) if gpu is not None else None
    if gpu is None or not available_in(gpu, data_center_id):
        raise ValueError(f"Configured GPU {fallback_id} is unavailable; no resources were created")
    if (price is None or price > max_hourly_price
            or (prefer_h100 and fallback_id in H100_CANDIDATE_IDS and price >= h100_hourly_limit)):
        raise ValueError("Current GPU catalog price exceeds runpod.max_hourly_price or is unavailable")
    return fallback_id, price


@dataclass(frozen=True, slots=True)
class Placement:
    gpu_type_id: str
    data_center_id: str
    hourly_price: float
    cloud: str = "SECURE"


def select_placement(rows, *, fallback_id, prefer_h100, max_hourly_price, cloud="SECURE",
                     preferred_data_center=""):
    """Choose US compute in one cloud; the location hint only breaks ties.

    Workers keep /workspace on a host-local Pod disk, so no storage type
    constrains the location: every US datacenter that the cloud's catalog
    reports for an eligible GPU is a candidate.
    """
    if cloud not in CLOUD_PRICE_KEYS:
        raise ValueError("Unknown Runpod cloud type; no allocation is safe")
    if not isinstance(rows, list):
        raise ValueError("Runpod returned an invalid GPU catalog; no allocation is safe")
    identifiers = sorted({center["id"] for row in rows if isinstance(row, dict)
                          and isinstance(row.get("dataCenters"), list)
                          for center in row["dataCenters"] if isinstance(center, dict)
                          and isinstance(center.get("id"), str) and center["id"].startswith("US-")})
    candidates = []
    for identifier in identifiers:
        try:
            gpu_id, price = select_gpu(rows, fallback_id=fallback_id, prefer_h100=prefer_h100,
                                       data_center_id=identifier, max_hourly_price=max_hourly_price,
                                       cloud=cloud)
        except ValueError:
            continue
        gpu = next(row for row in rows if isinstance(row, dict) and row.get("id") == gpu_id)
        stock = min({"HIGH": 0, "MEDIUM": 1, "LOW": 2}[row["availability"]]
                    for row in gpu["dataCenters"] if isinstance(row, dict)
                    and row.get("id") == identifier and isinstance(row.get("availability"), str)
                    and row["availability"] in _AVAILABLE)
        gpu_class = 0 if prefer_h100 and gpu_id in H100_CANDIDATE_IDS else 1
        rank = (gpu_class, price, stock, identifier != preferred_data_center, identifier, gpu_id)
        candidates.append((rank, Placement(gpu_id, identifier, float(price), cloud)))
    if not candidates:
        raise ValueError(f"No US datacenter on Runpod {CLOUD_LABELS[cloud]} has available compute within "
                         "the configured GPU policy and hourly price cap; no resources were created")
    return min(candidates, key=lambda candidate: candidate[0])[1]
