"""Runtime device selection helpers for cross-platform Torch execution."""
from __future__ import annotations

import platform
from typing import Iterable, Sequence


VALID_DEVICES = {"auto", "cpu", "mps", "cuda"}


def default_fallback_chain() -> list[str]:
    if platform.system() == "Darwin":
        return ["mps", "cpu"]
    return ["cuda", "cpu"]


def parse_fallback_chain(
    raw_value: str | None, *, default_chain: Iterable[str] | None = None
) -> list[str]:
    if default_chain is None:
        default_chain = default_fallback_chain()

    if raw_value is None or not raw_value.strip():
        chain = [dev.lower() for dev in default_chain]
    else:
        chain = [chunk.strip().lower() for chunk in raw_value.split(",") if chunk.strip()]

    normalized: list[str] = []
    seen: set[str] = set()
    for device in chain:
        if device not in VALID_DEVICES or device == "auto":
            continue
        if device in seen:
            continue
        seen.add(device)
        normalized.append(device)

    if "cpu" not in seen:
        normalized.append("cpu")
    return normalized


def _is_mps_available(torch_module) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    if mps_backend is None:
        return False
    is_available = getattr(mps_backend, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    return False


def _is_cuda_available(torch_module) -> bool:
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None:
        return False
    is_available = getattr(cuda_module, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    return False


def is_device_available(device: str, torch_module) -> bool:
    normalized = device.lower()
    if normalized == "cpu":
        return True
    if normalized == "mps":
        return _is_mps_available(torch_module)
    if normalized == "cuda":
        return _is_cuda_available(torch_module)
    return False


def resolve_torch_device(
    requested_device: str | None, fallback_chain: Sequence[str], torch_module
) -> str:
    requested = (requested_device or "auto").strip().lower()
    if requested not in VALID_DEVICES:
        requested = "auto"

    if requested == "auto":
        for device in fallback_chain:
            if is_device_available(device, torch_module):
                return device
        return "cpu"

    if is_device_available(requested, torch_module):
        return requested

    for device in fallback_chain:
        if device == requested:
            continue
        if is_device_available(device, torch_module):
            return device
    return "cpu"


def clear_torch_cache(device: str, torch_module) -> None:
    normalized = device.lower()
    if normalized == "cuda":
        cuda_module = getattr(torch_module, "cuda", None)
        if cuda_module is not None and hasattr(cuda_module, "empty_cache"):
            cuda_module.empty_cache()
        return

    if normalized == "mps":
        mps_module = getattr(torch_module, "mps", None)
        if mps_module is not None and hasattr(mps_module, "empty_cache"):
            mps_module.empty_cache()


def next_available_fallback(
    current_device: str, fallback_chain: Sequence[str], torch_module
) -> str | None:
    current = current_device.lower()
    try:
        start_index = fallback_chain.index(current) + 1
    except ValueError:
        start_index = 0

    for device in fallback_chain[start_index:]:
        if is_device_available(device, torch_module):
            return device
    return None


def should_retry_on_mps_error(exc: Exception, device: str) -> bool:
    if device.lower() != "mps" or not isinstance(exc, RuntimeError):
        return False

    message = str(exc).lower()
    indicators = (
        "mps",
        "not implemented",
        "not currently supported",
        "unsupported",
        "placeholder storage",
    )
    return any(token in message for token in indicators)
