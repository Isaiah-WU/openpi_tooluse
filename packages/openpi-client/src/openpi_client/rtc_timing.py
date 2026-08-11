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

    rtc_prefix_steps: int = 0
    rtc_inference_delay_steps: int | None = None
    rtc_applied: bool = False
    observed_delay_policy_steps: int | None = None
    rtc_skipped_policy_steps: int = 0
    chunk_rejected: bool = False

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
            "accept_step": (self.accept_step if self.accept_step is not None else -1),
            "rtc_prefix_steps": self.rtc_prefix_steps,
            "rtc_inference_delay_steps": (
                self.rtc_inference_delay_steps
                if self.rtc_inference_delay_steps is not None
                else -1
            ),
            "rtc_applied": int(self.rtc_applied),
            "observed_delay_policy_steps": (
                self.observed_delay_policy_steps
                if self.observed_delay_policy_steps is not None
                else -1
            ),
            "rtc_skipped_policy_steps": self.rtc_skipped_policy_steps,
            "chunk_rejected": int(self.chunk_rejected),
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
            "observed_delay_steps": (delay_steps if delay_steps is not None else -1),
        }


@dataclasses.dataclass
class ControlCycleTiming:
    """Timing information for one UR10e control cycle."""

    control_step: int
    cycle_start: float

    request_id: int | None = None
    action_index: int | None = None
    action_source: str = "unknown"
    actions_enabled: bool = False
    chunk_boundary: bool = False
    arm_action_delta_l2: float | None = None
    max_abs_arm_action_delta: float | None = None
    gripper_action_delta_abs: float | None = None

    execute_action_start: float | None = None
    execute_action_end: float | None = None
    cycle_end: float | None = None

    def as_metrics(self) -> dict[str, int | float | str]:
        """Convert one control cycle into values suitable for logging."""
        return {
            "control_step": self.control_step,
            "request_id": (self.request_id if self.request_id is not None else -1),
            "action_index": (
                self.action_index if self.action_index is not None else -1
            ),
            "action_source": self.action_source,
            "actions_enabled": int(self.actions_enabled),
            "chunk_boundary": int(self.chunk_boundary),
            "arm_action_delta_l2": (
                self.arm_action_delta_l2
                if self.arm_action_delta_l2 is not None
                else float("nan")
            ),
            "max_abs_arm_action_delta": (
                self.max_abs_arm_action_delta
                if self.max_abs_arm_action_delta is not None
                else float("nan")
            ),
            "gripper_action_delta_abs": (
                self.gripper_action_delta_abs
                if self.gripper_action_delta_abs is not None
                else float("nan")
            ),
            "execute_action_ms": _duration_ms(
                self.execute_action_start,
                self.execute_action_end,
            ),
            "control_cycle_ms": _duration_ms(
                self.cycle_start,
                self.cycle_end,
            ),
        }
