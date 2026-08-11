"""Compare one baseline UR10e timing run with one RTC timing run."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class RTCRunSummary:
    """Key latency, control-rate, and chunk-boundary metrics for one run."""

    request_count: int
    rtc_request_count: int
    rejected_chunk_count: int
    actions_enabled: bool
    client_infer_p95_ms: float
    observed_delay_p95_steps: float
    control_cycle_p95_ms: float
    achieved_control_hz: float
    boundary_arm_delta_p95: float
    boundary_arm_delta_max: float


@dataclasses.dataclass(frozen=True)
class RTCComparison:
    """Direct changes from baseline to RTC; positive frequency gain is better."""

    control_frequency_change_percent: float
    boundary_delta_change_percent: float
    client_infer_change_percent: float


def _load_csv(run_dir: Path, filename: str) -> list[dict[str, str]]:
    path = Path(run_dir) / filename
    if not path.is_file():
        raise ValueError(f"missing {path}")
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _numeric_values(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if len(values) else float("nan")


def summarize_run(run_dir: Path) -> RTCRunSummary:
    """Summarize one timestamped runtime directory."""
    request_rows = _load_csv(run_dir, "request_timing.csv")
    cycle_rows = _load_csv(run_dir, "control_cycle_timing.csv")

    completed_requests = [
        row
        for row in request_rows
        if float(row.get("accept_step", -1)) >= 0
    ]
    rtc_request_count = sum(int(float(row.get("rtc_applied", 0))) for row in completed_requests)
    rejected_chunk_count = sum(int(float(row.get("chunk_rejected", 0))) for row in completed_requests)
    actions_enabled = any(int(float(row.get("actions_enabled", 0))) for row in cycle_rows)

    client_infer = _numeric_values(completed_requests, "client_infer_ms")
    observed_delay = _numeric_values(completed_requests, "observed_delay_policy_steps")
    control_cycle = _numeric_values(cycle_rows, "control_cycle_ms")
    boundary_delta = _numeric_values(
        [row for row in cycle_rows if int(float(row.get("chunk_boundary", 0))) == 1],
        "arm_action_delta_l2",
    )

    control_cycle_p50 = _percentile(control_cycle, 50)
    achieved_control_hz = (
        1000.0 / control_cycle_p50
        if math.isfinite(control_cycle_p50) and control_cycle_p50 > 0
        else float("nan")
    )

    return RTCRunSummary(
        request_count=len(completed_requests),
        rtc_request_count=rtc_request_count,
        rejected_chunk_count=rejected_chunk_count,
        actions_enabled=actions_enabled,
        client_infer_p95_ms=_percentile(client_infer, 95),
        observed_delay_p95_steps=_percentile(observed_delay, 95),
        control_cycle_p95_ms=_percentile(control_cycle, 95),
        achieved_control_hz=achieved_control_hz,
        boundary_arm_delta_p95=_percentile(boundary_delta, 95),
        boundary_arm_delta_max=(
            float(boundary_delta.max())
            if len(boundary_delta)
            else float("nan")
        ),
    )


def _percent_change(baseline: float, rtc: float) -> float:
    if not math.isfinite(baseline) or not math.isfinite(rtc) or baseline == 0:
        return float("nan")
    return (rtc - baseline) / baseline * 100.0


def compare_run_summaries(
    baseline: RTCRunSummary,
    rtc: RTCRunSummary,
) -> RTCComparison:
    """Validate comparable modes and calculate baseline-to-RTC changes."""
    if baseline.actions_enabled != rtc.actions_enabled:
        raise ValueError("baseline and RTC runs must use the same actions_enabled mode")
    if baseline.rtc_request_count != 0:
        raise ValueError("baseline run unexpectedly contains RTC requests")
    if rtc.rtc_request_count == 0:
        raise ValueError("RTC run contains no completed RTC requests")

    return RTCComparison(
        control_frequency_change_percent=_percent_change(
            baseline.achieved_control_hz,
            rtc.achieved_control_hz,
        ),
        boundary_delta_change_percent=_percent_change(
            baseline.boundary_arm_delta_p95,
            rtc.boundary_arm_delta_p95,
        ),
        client_infer_change_percent=_percent_change(
            baseline.client_infer_p95_ms,
            rtc.client_infer_p95_ms,
        ),
    )


def _print_summary(label: str, summary: RTCRunSummary) -> None:
    print(
        f"{label}: requests={summary.request_count} "
        f"rtc_requests={summary.rtc_request_count} "
        f"rejected={summary.rejected_chunk_count} "
        f"actions_enabled={summary.actions_enabled}"
    )
    print(
        f"  client_infer_p95_ms={summary.client_infer_p95_ms:.3f} "
        f"delay_p95_policy_steps={summary.observed_delay_p95_steps:.3f} "
        f"control_cycle_p95_ms={summary.control_cycle_p95_ms:.3f} "
        f"achieved_control_hz={summary.achieved_control_hz:.3f}"
    )
    print(
        f"  boundary_arm_delta_p95={summary.boundary_arm_delta_p95:.6f} "
        f"boundary_arm_delta_max={summary.boundary_arm_delta_max:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and RTC UR10e timing runs")
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("rtc_run", type=Path)
    args = parser.parse_args()

    baseline = summarize_run(args.baseline_run)
    rtc = summarize_run(args.rtc_run)
    comparison = compare_run_summaries(baseline, rtc)

    _print_summary("baseline", baseline)
    _print_summary("rtc", rtc)
    print(
        "change rtc_vs_baseline: "
        f"control_frequency={comparison.control_frequency_change_percent:+.3f}% "
        f"boundary_delta={comparison.boundary_delta_change_percent:+.3f}% "
        f"client_infer={comparison.client_infer_change_percent:+.3f}%"
    )


if __name__ == "__main__":
    main()
