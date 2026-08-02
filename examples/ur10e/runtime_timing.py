"""Runtime timing records for asynchronous UR10e policy inference."""

from __future__ import annotations

import dataclasses



def _duration_ms(start: float | None, end: float | None) -> float:
    """Return duration in milliseconds, or NaN when either endpoint is missing."""
    if start is None or end is None:
        return float("nan")
    return (end - start) * 1000.0

@dataclasses.dataclass
class RequestTiming:
    """Timing information for one asynchronous policy request."""

    request_id: int 

    observation_step: int
    submit_step: int

    observation_start: float
    observation_ready: float
    request_submit: float

    worker_infer_start: float | None = None
    worker_infer_end: float | None = None
    response_ready: float | None = None

    control_poll: float | None = None

    chunk_prepare_start: float | None = None
    chunk_prepare_end: float | None = None
    chunk_accept: float | None = None
    accept_step: int | None = None

    first_action_start: float | None = None
    first_action_end: float | None = None

    def observed_delay_steps(self) -> int | None:
        """Return how many control action elapsed while waiting for this request."""
        if self.accept_step is None:
            return None
        return self.accept_step - self.submit_step

    def as_metrics(self) -> dict[str, int | float]:
        """Convert timestamps into durations suitable for logging and statistics."""
        delay_steps = self.observed_delay_steps()

        return {
            "request_id": self.request_id,
            "observation_step": self.observation_step,
            "submit_step": self.submit_step,
            "accept_step": (
                self.accept_step
                if self.accept_step is not None
                else -1
            ),
            "observation_ms": _duration_ms(
                self.observation_start,
                self.observation_ready,
            ),
            "queue_ms": _duration_ms(
                self.request_submit,
                self.worker_infer_start,
            ),
            "client_infer_ms": _duration_ms(
                self.worker_infer_start,
                self.worker_infer_end,
            ),
            "poll_schedule_ms": _duration_ms(
                self.response_ready,
                self.control_poll,
            ),
            "chunk_prepare_ms": _duration_ms(
                self.chunk_prepare_start,
                self.chunk_prepare_end,
            ),
            "rtc_total_ms": _duration_ms(
                self.request_submit,
                self.chunk_accept,
            ),
            "sensor_to_action_ms": _duration_ms(
                self.observation_start,
                self.first_action_start,
            ),
            "execute_action_ms": _duration_ms(
                self.first_action_start,
                self.first_action_end,
            ),
            "observed_delay_steps": (
                delay_steps
                if delay_steps is not None
                else -1
            ),
        }


@dataclasses.dataclass
class ControlCycleTiming:
    """Timing information for one UR10e control cycle."""

    control_step: int
    cycle_start: float

    request_id: int | None = None
    action_index: int | None = None
    action_source: str = "unknown"

    execute_action_start: float | None = None
    execute_action_end: float | None = None
    cycle_end: float | None = None

    def as_metrics(self) -> dict[str, int | float | str]:
        """Convert one control cycle into values suitable for logging."""
        return {
            "control_step": self.control_step,
            "request_id": (
                self.request_id
                if self.request_id is not None
                else -1
            ),
            "action_index": (
                self.action_index
                if self.action_index is not None
                else -1
            ),
            "action_source": self.action_source,
            "execute_action_ms": _duration_ms(
                self.execute_action_start,
                self.execute_action_end,
            ),
            "control_cycle_ms": _duration_ms(
                self.cycle_start,
                self.cycle_end,
            ),
        }