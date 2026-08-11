"""UR10e action chunk frequency conversion."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class PreparedActionChunk:
    """Policy-rate and control-rate views of one accepted action chunk."""

    policy_actions: np.ndarray
    control_actions: np.ndarray
    observed_delay_policy_steps: int
    skipped_policy_steps: int


def resample_action_chunk(
    actions: np.ndarray,
    *,
    source_hz: float,
    target_hz: float,
) -> np.ndarray:
    """Resample a policy action chunk to the robot control frequency."""
    actions = np.asarray(
        actions,
        dtype=np.float32,
    )

    if actions.ndim != 2:
        raise ValueError(
            "actions must be a 2D array, "
            f"got shape {actions.shape}"
        )

    if source_hz <= 0:
        raise ValueError(
            "source_hz must be positive"
        )

    if target_hz <= 0:
        raise ValueError(
            "target_hz must be positive"
        )

    if (
        len(actions) < 2
        or abs(source_hz - target_hz) < 1e-6
    ):
        return actions.copy()

    source_times = (
        np.arange(
            len(actions),
            dtype=np.float32,
        )
        / float(source_hz)
    )

    end_time = float(source_times[-1])

    target_length = int(
        np.floor(end_time * target_hz + 1e-6)
    ) + 1

    target_times = (
        np.arange(target_length, dtype=np.float32)
        / float(target_hz)
    )

    output = np.empty(
        (
            target_length,
            actions.shape[1],
        ),
        dtype=np.float32,
    )

    for dimension in range(actions.shape[1]):
        output[:, dimension] = np.interp(
            target_times,
            source_times,
            actions[:, dimension],
        ).astype(np.float32)

    return output


def control_steps_to_policy_steps(
    control_steps: int,
    *,
    control_hz: float,
    policy_hz: float,
) -> int:
    """Convert measured control-loop delay into policy action steps."""
    if control_steps < 0:
        raise ValueError(
            "control_steps must be non-negative"
        )

    if control_hz <= 0:
        raise ValueError(
            "control_hz must be positive"
        )

    if policy_hz <= 0:
        raise ValueError(
            "policy_hz must be positive"
        )

    policy_steps = (
        control_steps
        * policy_hz
        / control_hz
    )

    return int(np.ceil(policy_steps))


def get_policy_action_leftover(
    policy_actions: np.ndarray,
    *,
    consumed_control_steps: int,
    control_hz: float,
    policy_hz: float,
) -> np.ndarray:
    """Return the unexecuted policy-rate suffix at the current control index."""
    policy_actions = np.asarray(policy_actions, dtype=np.float32)
    if policy_actions.ndim != 2:
        raise ValueError(f"policy_actions must be 2D, got shape {policy_actions.shape}")

    consumed_policy_steps = control_steps_to_policy_steps(
        consumed_control_steps,
        control_hz=control_hz,
        policy_hz=policy_hz,
    )
    start = min(consumed_policy_steps, len(policy_actions))
    return policy_actions[start:].copy()


def should_request_action_chunk(
    *,
    rtc_enabled: bool,
    inflight: bool,
    control_step: int,
    next_query_step: int,
    prefix_policy_steps: int | None,
    rtc_query_remaining_policy_steps: int | None,
) -> bool:
    """Return whether baseline or RTC scheduling should submit a new request."""
    if inflight:
        return False
    if not rtc_enabled:
        return control_step >= next_query_step
    if prefix_policy_steps is None:
        return True
    if rtc_query_remaining_policy_steps is None:
        raise ValueError("RTC requires rtc_query_remaining_policy_steps")
    if rtc_query_remaining_policy_steps < 0:
        raise ValueError("rtc_query_remaining_policy_steps must be non-negative")
    return prefix_policy_steps <= rtc_query_remaining_policy_steps


def prepare_action_chunk(
    actions: np.ndarray,
    *,
    observed_delay_control_steps: int,
    apply_rtc_delay_crop: bool,
    policy_hz: float,
    control_hz: float,
) -> PreparedActionChunk:
    """Align a returned policy chunk to now and resample it for control.

    RTC predicts a chunk whose index zero corresponds to the observation time.
    If the robot executed old actions during inference, those elapsed policy
    positions must be discarded before the new chunk becomes executable.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must be 2D, got shape {actions.shape}")
    if len(actions) == 0:
        raise ValueError("actions must contain at least one step")

    observed_delay_policy_steps = control_steps_to_policy_steps(
        observed_delay_control_steps,
        control_hz=control_hz,
        policy_hz=policy_hz,
    )
    skipped_policy_steps = (
        observed_delay_policy_steps
        if apply_rtc_delay_crop
        else 0
    )

    start = min(skipped_policy_steps, len(actions))
    aligned_policy_actions = actions[start:].copy()
    if len(aligned_policy_actions) == 0:
        control_actions = np.empty((0, actions.shape[1]), dtype=np.float32)
    else:
        control_actions = resample_action_chunk(
            aligned_policy_actions,
            source_hz=policy_hz,
            target_hz=control_hz,
        )

    return PreparedActionChunk(
        policy_actions=aligned_policy_actions,
        control_actions=control_actions,
        observed_delay_policy_steps=observed_delay_policy_steps,
        skipped_policy_steps=skipped_policy_steps,
    )
