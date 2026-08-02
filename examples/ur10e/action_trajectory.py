"""UR10e action chunk frequency conversion."""

from __future__ import annotations

import numpy as np


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

    target_length = (
        int(round(end_time * target_hz))
        + 1
    )

    target_times = np.linspace(
        0.0,
        end_time,
        target_length,
        dtype=np.float32,
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