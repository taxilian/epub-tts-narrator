"""Managed GPU selection policy for Runpod deployments.

Pure catalog logic with no network or saved-state access. Each call ranks
one cloud (Secure or Community); the caller walks runpod.cloud_types in order.
The lifecycle caller persists the chosen identity before creating any Pod, so
an ambiguous create can never re-resolve to a different GPU.

Any NVIDIA GPU whose catalog memory is at least ``runpod.min_gpu_memory_gb``
is eligible (``runpod.gpu_type_id`` optionally pins one exact catalog ID).
The catalog says nothing about speed, so GPUs with a measured Drama 3
throughput rank first by measured cost per generated audio hour, and
unmeasured GPUs follow by hourly price. Unknown or malformed catalog data
(memory, manufacturer, availability, price) never makes a GPU eligible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Default memory floor: one Drama 3 copy uses about 48 GB while generating,
# plus 2.5-4.5 GB for CUDA Whisper verification, so 48 GB cards cannot hold it.
MIN_GPU_MEMORY_GB = 56
# Measured generated audio seconds per second for one single-request Drama 3
# copy (deploy/runpod/README.md "Inference replicas"). Add a GPU only from a
# benchmark of the same corpus; never estimate one from its name.
MEASURED_REALTIME = {
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 2.10,
}
_AVAILABLE = frozenset({"HIGH", "MEDIUM", "LOW"})
_STOCK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
# Runpod's two Pod clouds, as the v2 API names them, and their catalog price keys.
CLOUD_PRICE_KEYS = {"SECURE": "secure", "COMMUNITY": "community"}
CLOUD_LABELS = {"SECURE": "Secure Cloud", "COMMUNITY": "Community Cloud"}


def catalog_price(gpu, cloud="SECURE"):
    """Return the cloud's hourly price when a positive finite number, else None."""
    price = gpu.get("price") if isinstance(gpu, dict) else None
    price = price.get(CLOUD_PRICE_KEYS[cloud]) if isinstance(price, dict) else None
    if (isinstance(price, bool) or not isinstance(price, (int, float))
            or not math.isfinite(price) or price <= 0):
        return None
    return price


def catalog_memory(gpu):
    """The GPU's catalog memory in GB when a positive finite number, else None."""
    memory = gpu.get("memory") if isinstance(gpu, dict) else None
    if (isinstance(memory, bool) or not isinstance(memory, (int, float))
            or not math.isfinite(memory) or memory <= 0):
        return None
    return memory


def stock_in(gpu, data_center_id):
    """Stock rank (0 best) when overall and datacenter availability are known, else None."""
    availability = gpu.get("availability")
    centers = gpu.get("dataCenters")
    if not isinstance(availability, str) or availability not in _AVAILABLE or not isinstance(centers, list):
        return None
    ranks = [_STOCK[center["availability"]] for center in centers
             if isinstance(center, dict) and center.get("id") == data_center_id
             and isinstance(center.get("availability"), str) and center["availability"] in _AVAILABLE]
    return min(ranks, default=None)


def eligible(gpu, *, gpu_type_id="", min_memory_gb=MIN_GPU_MEMORY_GB):
    """True for an NVIDIA GPU with enough known memory (and the pinned ID, if any)."""
    if not isinstance(gpu, dict) or not isinstance(gpu.get("id"), str):
        return False
    if gpu_type_id and gpu["id"] != gpu_type_id:
        return False
    memory = catalog_memory(gpu)
    return gpu.get("manufacturer") == "NVIDIA" and memory is not None and memory >= min_memory_gb


def cost_rank(gpu_id, price):
    """(measured first, cost): measured cost per generated audio hour, else the hourly price."""
    realtime = MEASURED_REALTIME.get(gpu_id)
    return (0, price / realtime) if realtime else (1, price)


@dataclass(frozen=True, slots=True)
class Placement:
    gpu_type_id: str
    data_center_id: str
    hourly_price: float
    cloud: str = "SECURE"
    memory_gb: float = 0.0


def select_placement(rows, *, max_hourly_price, cloud="SECURE", gpu_type_id="",
                     min_memory_gb=MIN_GPU_MEMORY_GB, preferred_data_center=""):
    """Choose a GPU and US datacenter in one cloud; the location hint only breaks ties.

    Workers keep /workspace on a host-local Pod disk, so no storage type
    constrains the location: every US datacenter that the cloud's catalog
    reports for an eligible GPU is a candidate. Ranking: measured GPUs by
    measured cost per generated audio hour, then unmeasured GPUs by price;
    then stock, the location hint, and catalog IDs for a stable choice.
    """
    if cloud not in CLOUD_PRICE_KEYS:
        raise ValueError("Unknown Runpod cloud type; no allocation is safe")
    if not isinstance(rows, list):
        raise ValueError("Runpod returned an invalid GPU catalog; no allocation is safe")
    candidates = []
    for gpu in rows:
        if not eligible(gpu, gpu_type_id=gpu_type_id, min_memory_gb=min_memory_gb):
            continue
        price = catalog_price(gpu, cloud)
        if price is None or price > max_hourly_price or not isinstance(gpu.get("dataCenters"), list):
            continue
        for identifier in sorted({center["id"] for center in gpu["dataCenters"] if isinstance(center, dict)
                                  and isinstance(center.get("id"), str) and center["id"].startswith("US-")}):
            stock = stock_in(gpu, identifier)
            if stock is None:
                continue
            rank = (*cost_rank(gpu["id"], price), stock, identifier != preferred_data_center,
                    identifier, gpu["id"])
            candidates.append((rank, Placement(gpu["id"], identifier, float(price), cloud,
                                               float(catalog_memory(gpu)))))
    if not candidates:
        wanted = (f"GPU {gpu_type_id} with at least {min_memory_gb} GB" if gpu_type_id
                  else f"NVIDIA GPU with at least {min_memory_gb} GB")
        raise ValueError(f"No US datacenter on Runpod {CLOUD_LABELS[cloud]} has an available {wanted} "
                         "within runpod.max_hourly_price; no resources were created")
    return min(candidates, key=lambda candidate: candidate[0])[1]
