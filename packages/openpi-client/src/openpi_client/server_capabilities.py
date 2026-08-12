"""Shared WebSocket server capability metadata for RTC preflight checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SERVER_CAPABILITY_KEY = "openpi_server"
RTC_PROTOCOL_VERSION = 1


def add_rtc_server_capability(
    metadata: Mapping[str, Any] | None,
    *,
    rtc_enabled: bool,
    execution_horizon: int,
    prefix_attention_schedule: str,
    max_guidance_weight: float,
) -> dict[str, Any]:
    """Return policy metadata augmented with machine-readable RTC capability."""
    if execution_horizon <= 0:
        raise ValueError("execution_horizon must be positive")
    if max_guidance_weight <= 0:
        raise ValueError("max_guidance_weight must be positive")
    result = dict(metadata or {})
    result[SERVER_CAPABILITY_KEY] = {
        "rtc_protocol_version": RTC_PROTOCOL_VERSION,
        "rtc": {
            "supported": True,
            "enabled": bool(rtc_enabled),
            "fixed_prefix_shape": True,
            "execution_horizon": int(execution_horizon),
            "prefix_attention_schedule": str(prefix_attention_schedule),
            "max_guidance_weight": float(max_guidance_weight),
        },
    }
    return result


def validate_rtc_server_capability(
    metadata: Mapping[str, Any] | None,
    *,
    rtc_requested: bool,
    fixed_prefix_shape_required: bool = False,
) -> None:
    """Allow baseline legacy servers but fail fast for an incompatible RTC server."""
    if not rtc_requested:
        return
    raw_server = (metadata or {}).get(SERVER_CAPABILITY_KEY, {})
    if not isinstance(raw_server, Mapping):
        raise RuntimeError("Policy server returned malformed RTC capability metadata")
    server = dict(raw_server)
    protocol_version = server.get("rtc_protocol_version")
    raw_rtc = server.get("rtc", {})
    if not isinstance(raw_rtc, Mapping):
        raise RuntimeError("Policy server returned malformed RTC configuration metadata")
    rtc = dict(raw_rtc)
    if protocol_version != RTC_PROTOCOL_VERSION:
        raise RuntimeError(
            f"RTC client requires server rtc_protocol_version={RTC_PROTOCOL_VERSION}; "
            f"received {protocol_version!r}"
        )
    if not rtc.get("supported", False):
        raise RuntimeError("Policy server does not advertise RTC support")
    if not rtc.get("enabled", False):
        raise RuntimeError("Policy server RTC processor is not enabled")
    if fixed_prefix_shape_required and not rtc.get("fixed_prefix_shape", False):
        raise RuntimeError("RTC client requires server fixed-prefix-shape support")
