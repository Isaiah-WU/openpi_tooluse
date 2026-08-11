"""Save and summarize UR10e runtime timing records."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from runtime_timing import ControlCycleTiming
from runtime_timing import RequestTiming

REQUEST_METRIC_NAMES = (
    "rtc_prefix_steps",
    "rtc_inference_delay_steps",
    "observed_delay_policy_steps",
    "rtc_skipped_policy_steps",
    "observation_ms",
    "queue_ms",
    "client_infer_ms",
    "poll_schedule_ms",
    "chunk_prepare_ms",
    "rtc_total_ms",
    "sensor_to_action_ms",
    "execute_action_ms",
    "observed_delay_steps",
)

CYCLE_METRIC_NAMES = (
    "execute_action_ms",
    "control_cycle_ms",
)

def _valid_numeric_values(
    rows: list[dict],
    metric_name: str,
) -> list[float]:
    """Collect valid numeric values for one metric."""
    values: list[float] = []

    for row in rows:
        value = row.get(metric_name)

        if not isinstance(value, (int, float)):
            continue

        value = float(value)

        if not math.isfinite(value):
            continue

        if metric_name in {
            "rtc_inference_delay_steps",
            "observed_delay_policy_steps",
            "observed_delay_steps",
        } and value < 0:
            continue

        values.append(value)

    return values


def _print_metric_summary(
    metric_name: str,
    values: list[float],
) -> None:
    """Print summary statistics for one timing metric."""
    if not values:
        print(f"{metric_name}: no valid samples")
        return

    array = np.asarray(values, dtype=np.float64)

    print(
        f"{metric_name}: "
        f"count={len(array)} "
        f"mean={array.mean():.3f} "
        f"p50={np.percentile(array, 50):.3f} "
        f"p95={np.percentile(array, 95):.3f} "
        f"p99={np.percentile(array, 99):.3f} "
        f"max={array.max():.3f}"
    )


def _write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """Write timing rows to one CSV file."""
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


class TimingRecorder:
    """Collect request-level and control-cycle timing records."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

        self._request_timings: dict[int, RequestTiming] = {}
        self._cycle_timings: list[ControlCycleTiming] = []

    def record_request(
        self,
        timing: RequestTiming,
    ) -> None:
        """Add or update one request timing record."""
        self._request_timings[timing.request_id] = timing

    def record_cycle(
        self,
        timing: ControlCycleTiming,
    ) -> None:
        """Append one completed control-cycle record."""
        self._cycle_timings.append(timing)

    def request_rows(self) -> list[dict]:
        """Return request metrics sorted by request ID."""
        return [
            self._request_timings[request_id].as_metrics()
            for request_id in sorted(self._request_timings)
        ]

    def cycle_rows(self) -> list[dict]:
        """Return control-cycle metrics in execution order."""
        return [
            timing.as_metrics()
            for timing in self._cycle_timings
        ]

    def save(self) -> None:
        """Save request and control-cycle metrics to CSV files."""
        _write_csv(
            self.output_dir / "request_timing.csv",
            self.request_rows(),
        )

        _write_csv(
            self.output_dir / "control_cycle_timing.csv",
            self.cycle_rows(),
        )

    def print_summary(self) -> None:
        """Print percentile summaries for all runtime metrics."""
        request_rows = self.request_rows()
        cycle_rows = self.cycle_rows()

        print("Request timing summary")

        for metric_name in REQUEST_METRIC_NAMES:
            values = _valid_numeric_values(
                request_rows,
                metric_name,
            )
            _print_metric_summary(
                metric_name,
                values,
            )

        print("Control cycle timing summary")

        for metric_name in CYCLE_METRIC_NAMES:
            values = _valid_numeric_values(
                cycle_rows,
                metric_name,
            )
            _print_metric_summary(
                metric_name,
                values,
            )
