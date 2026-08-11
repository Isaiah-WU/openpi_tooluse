"""Calibrate safe UR10e RTC runtime parameters from measured timing CSV data."""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import math
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class RTCRecommendation:
    """Measured latency distribution and derived policy-rate RTC parameters."""

    sample_count: int
    delay_p50: float
    delay_p95: float
    delay_p99: float
    inference_delay_policy_steps: int
    execution_horizon_policy_steps: int
    query_remaining_policy_steps: int


class RTCDelayTracker:
    """Conservatively forecast delay from a bounded window of real requests."""

    def __init__(self, initial_delay_policy_steps: int, *, maxlen: int = 20) -> None:
        if (
            not isinstance(initial_delay_policy_steps, int)
            or isinstance(initial_delay_policy_steps, bool)
            or initial_delay_policy_steps < 0
        ):
            raise ValueError("initial_delay_policy_steps must be a non-negative integer")
        if not isinstance(maxlen, int) or isinstance(maxlen, bool) or maxlen <= 0:
            raise ValueError("maxlen must be a positive integer")

        self._initial_delay = initial_delay_policy_steps
        self._delays: collections.deque[int] = collections.deque(maxlen=maxlen)

    def add(self, observed_delay_policy_steps: int) -> None:
        """Add one completed request delay to the forecast window."""
        if (
            not isinstance(observed_delay_policy_steps, int)
            or isinstance(observed_delay_policy_steps, bool)
            or observed_delay_policy_steps < 0
        ):
            raise ValueError("observed_delay_policy_steps must be a non-negative integer")
        self._delays.append(observed_delay_policy_steps)

    def estimate(self) -> int:
        """Return the initial calibration or largest recent delay, whichever is larger."""
        return max(self._initial_delay, max(self._delays, default=0))


def load_baseline_policy_delays(path: Path) -> list[float]:
    """Load valid non-RTC delays from one CSV file or a directory of runs."""
    delays: list[float] = []
    path = Path(path)
    csv_paths = (
        sorted(path.glob("**/request_timing.csv"))
        if path.is_dir()
        else [path]
    )
    if not csv_paths:
        raise ValueError(f"no request_timing.csv files found under {path}")

    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "observed_delay_policy_steps" not in reader.fieldnames:
                raise ValueError(
                    f"{csv_path} is missing observed_delay_policy_steps"
                )

            for row in reader:
                rtc_applied = (row.get("rtc_applied") or "0").strip().lower()
                if rtc_applied in {"1", "true", "yes"}:
                    continue

                try:
                    delay = float(row["observed_delay_policy_steps"])
                except (TypeError, ValueError):
                    continue

                if math.isfinite(delay) and delay >= 0:
                    delays.append(delay)

    return delays


def validate_rtc_runtime_parameters(
    *,
    inference_delay_policy_steps: int,
    execution_horizon_policy_steps: int,
    query_remaining_policy_steps: int,
    action_horizon_policy_steps: int,
) -> None:
    """Reject RTC settings that cannot retain a safe previous-chunk overlap."""
    values = {
        "inference_delay_policy_steps": inference_delay_policy_steps,
        "execution_horizon_policy_steps": execution_horizon_policy_steps,
        "query_remaining_policy_steps": query_remaining_policy_steps,
        "action_horizon_policy_steps": action_horizon_policy_steps,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value}")

    if execution_horizon_policy_steps == 0:
        raise ValueError("execution_horizon_policy_steps must be positive")
    if action_horizon_policy_steps == 0:
        raise ValueError("action_horizon_policy_steps must be positive")
    if execution_horizon_policy_steps < inference_delay_policy_steps:
        raise ValueError(
            "execution_horizon_policy_steps must be at least the inference delay"
        )
    if query_remaining_policy_steps <= (
        inference_delay_policy_steps + execution_horizon_policy_steps
    ):
        raise ValueError(
            "query_remaining_policy_steps must be greater than "
            "inference delay + execution horizon"
        )
    if query_remaining_policy_steps >= action_horizon_policy_steps:
        raise ValueError(
            "query_remaining_policy_steps must be smaller than the action horizon"
        )


def recommend_rtc_parameters(
    delays: list[float],
    *,
    action_horizon_policy_steps: int = 50,
    minimum_execution_horizon_policy_steps: int = 10,
    policy_hz: float = 30.0,
    control_hz: float = 10.0,
    minimum_samples: int = 20,
) -> RTCRecommendation:
    """Derive an RTC delay and conservative request threshold from measurements."""
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    if len(delays) < minimum_samples:
        raise ValueError(
            f"need at least {minimum_samples} valid baseline samples, got {len(delays)}"
        )
    if policy_hz <= 0 or control_hz <= 0:
        raise ValueError("policy_hz and control_hz must be positive")
    if minimum_execution_horizon_policy_steps <= 0:
        raise ValueError("minimum_execution_horizon_policy_steps must be positive")

    array = np.asarray(delays, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("delays must contain finite non-negative values")

    p50, p95, p99 = np.percentile(array, [50, 95, 99])
    inference_delay = int(math.ceil(float(p95)))
    execution_horizon = max(
        minimum_execution_horizon_policy_steps,
        inference_delay,
    )

    # Crossing a threshold happens only once per control cycle. Add one full
    # control-cycle stride to the P99 delay so the remaining prefix still stays
    # above delay + execution_horizon after that discrete crossing.
    policy_steps_per_control = int(math.ceil(policy_hz / control_hz))
    query_remaining = (
        int(math.ceil(float(p99)))
        + execution_horizon
        + policy_steps_per_control
    )

    validate_rtc_runtime_parameters(
        inference_delay_policy_steps=inference_delay,
        execution_horizon_policy_steps=execution_horizon,
        query_remaining_policy_steps=query_remaining,
        action_horizon_policy_steps=action_horizon_policy_steps,
    )

    return RTCRecommendation(
        sample_count=len(array),
        delay_p50=float(p50),
        delay_p95=float(p95),
        delay_p99=float(p99),
        inference_delay_policy_steps=inference_delay,
        execution_horizon_policy_steps=execution_horizon,
        query_remaining_policy_steps=query_remaining,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend UR10e RTC parameters from request_timing.csv",
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="request_timing.csv or a directory containing multiple timing runs",
    )
    parser.add_argument("--minimum-samples", type=int, default=20)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--minimum-execution-horizon", type=int, default=10)
    parser.add_argument("--policy-hz", type=float, default=30.0)
    parser.add_argument("--control-hz", type=float, default=10.0)
    args = parser.parse_args()

    recommendation = recommend_rtc_parameters(
        load_baseline_policy_delays(args.csv_path),
        action_horizon_policy_steps=args.action_horizon,
        minimum_execution_horizon_policy_steps=args.minimum_execution_horizon,
        policy_hz=args.policy_hz,
        control_hz=args.control_hz,
        minimum_samples=args.minimum_samples,
    )

    print(f"samples={recommendation.sample_count}")
    print(
        "observed_delay_policy_steps: "
        f"p50={recommendation.delay_p50:.3f} "
        f"p95={recommendation.delay_p95:.3f} "
        f"p99={recommendation.delay_p99:.3f}"
    )
    print("Set these values in examples/ur10e/ur10e_client.py:")
    print("RTC_ENABLED = True")
    print(
        "RTC_INFERENCE_DELAY_POLICY_STEPS = "
        f"{recommendation.inference_delay_policy_steps}"
    )
    print(
        "RTC_EXECUTION_HORIZON_POLICY_STEPS = "
        f"{recommendation.execution_horizon_policy_steps}"
    )
    print(
        "RTC_QUERY_REMAINING_POLICY_STEPS = "
        f"{recommendation.query_remaining_policy_steps}"
    )


if __name__ == "__main__":
    main()
