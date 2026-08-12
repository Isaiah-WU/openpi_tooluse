"""Measure RTC request latency without an action queue or robot runtime."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import dataclasses
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from openpi_client.rtc_calibration import validate_rtc_runtime_parameters
from openpi_client.server_capabilities import validate_rtc_server_capability


@dataclasses.dataclass(frozen=True)
class RTCLatencySample:
    request_index: int
    rtc_total_ms: float
    observed_delay_policy_steps: float
    server_infer_ms: float | None
    policy_infer_ms: float | None

    def as_row(self) -> dict[str, int | float | str]:
        return {
            "request_index": self.request_index,
            "rtc_total_ms": self.rtc_total_ms,
            "observed_delay_policy_steps": self.observed_delay_policy_steps,
            "server_infer_ms": "" if self.server_infer_ms is None else self.server_infer_ms,
            "policy_infer_ms": "" if self.policy_infer_ms is None else self.policy_infer_ms,
        }


@dataclasses.dataclass(frozen=True)
class RTCLatencyRecommendation:
    sample_count: int
    delay_p50: float
    delay_p95: float
    delay_p99: float
    delay_max: float
    inference_delay_policy_steps: int
    execution_horizon_policy_steps: int
    query_remaining_policy_steps: int


def _optional_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _validated_actions(result: Mapping[str, Any], *, action_horizon: int, action_dim: int) -> np.ndarray:
    actions = np.asarray(result["actions"], dtype=np.float32)
    expected_shape = (action_horizon, action_dim)
    if actions.shape != expected_shape:
        raise RuntimeError(f"RTC latency benchmark expected actions {expected_shape}, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise RuntimeError("RTC latency benchmark received NaN or Inf actions")
    return actions


def run_rtc_latency_benchmark(
    policy: Any,
    raw_observation: dict[str, Any],
    *,
    sample_count: int = 120,
    policy_hz: float = 30.0,
    action_horizon: int = 50,
    action_dim: int = 7,
    prev_chunk_valid_steps: int = 14,
    inference_delay: int = 3,
    execution_horizon: int = 10,
    clock=time.perf_counter,
) -> list[RTCLatencySample]:
    """Run one unmeasured baseline request followed by serial RTC requests.

    Returned chunks are used only as the next request's prefix. This function
    has no action queue, delay tracker, control loop, or robot command path.
    """
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if not math.isfinite(policy_hz) or policy_hz <= 0:
        raise ValueError("policy_hz must be finite and positive")
    if not 0 < prev_chunk_valid_steps <= action_horizon:
        raise ValueError("prev_chunk_valid_steps must fit the action horizon")
    if not 0 <= inference_delay <= execution_horizon <= action_horizon:
        raise ValueError("expected 0 <= inference_delay <= execution_horizon <= action_horizon")

    validate_rtc_server_capability(
        policy.get_server_metadata(),
        rtc_requested=True,
        fixed_prefix_shape_required=True,
        warmup_complete_required=True,
    )

    baseline = policy.infer(raw_observation)
    previous_actions = _validated_actions(
        baseline,
        action_horizon=action_horizon,
        action_dim=action_dim,
    )

    samples: list[RTCLatencySample] = []
    for request_index in range(sample_count):
        started = clock()
        result = policy.infer(
            raw_observation,
            prev_chunk_left_over=previous_actions,
            prev_chunk_valid_steps=prev_chunk_valid_steps,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )
        elapsed_seconds = clock() - started
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise RuntimeError(f"RTC latency benchmark clock returned invalid elapsed time {elapsed_seconds}")

        previous_actions = _validated_actions(
            result,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        samples.append(
            RTCLatencySample(
                request_index=request_index,
                rtc_total_ms=elapsed_seconds * 1000.0,
                observed_delay_policy_steps=elapsed_seconds * policy_hz,
                server_infer_ms=_optional_finite_float(result.get("server_timing", {}).get("infer_ms")),
                policy_infer_ms=_optional_finite_float(result.get("policy_timing", {}).get("infer_ms")),
            )
        )

    return samples


def recommend_from_rtc_latency(
    samples: list[RTCLatencySample],
    *,
    action_horizon: int = 50,
    minimum_execution_horizon: int = 10,
    minimum_samples: int = 100,
) -> RTCLatencyRecommendation:
    """Derive D/S/Q from RTC P99 and fail closed if they do not fit H."""
    if len(samples) < minimum_samples:
        raise ValueError(f"need at least {minimum_samples} RTC latency samples, got {len(samples)}")
    delays = np.asarray([sample.observed_delay_policy_steps for sample in samples], dtype=np.float64)
    if delays.ndim != 1 or not np.isfinite(delays).all() or (delays < 0).any():
        raise ValueError("RTC latency samples must contain finite non-negative delays")

    p50, p95, p99 = np.percentile(delays, [50, 95, 99])
    inference_delay = int(math.ceil(float(p99)))
    execution_horizon = max(minimum_execution_horizon, inference_delay)
    query_remaining = inference_delay + execution_horizon + 1
    if query_remaining >= action_horizon:
        raise ValueError(
            "RTC P99 delay cannot fit the action horizon: "
            f"D={inference_delay}, S={execution_horizon}, Q={query_remaining}, H={action_horizon}"
        )

    validate_rtc_runtime_parameters(
        inference_delay_policy_steps=inference_delay,
        execution_horizon_policy_steps=execution_horizon,
        query_remaining_policy_steps=query_remaining,
        action_horizon_policy_steps=action_horizon,
    )

    return RTCLatencyRecommendation(
        sample_count=len(samples),
        delay_p50=float(p50),
        delay_p95=float(p95),
        delay_p99=float(p99),
        delay_max=float(delays.max()),
        inference_delay_policy_steps=inference_delay,
        execution_horizon_policy_steps=execution_horizon,
        query_remaining_policy_steps=query_remaining,
    )


def save_rtc_latency_samples(path: Path, samples: list[RTCLatencySample]) -> None:
    """Write standalone RTC measurements without runtime timing rows."""
    if not samples:
        raise ValueError("cannot save an empty RTC latency benchmark")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(samples[0].as_row()))
        writer.writeheader()
        writer.writerows(sample.as_row() for sample in samples)
