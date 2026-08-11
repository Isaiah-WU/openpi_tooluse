"""Explicit arming gate for UR10e policy action execution."""

from __future__ import annotations

from collections.abc import Callable

ARM_PHRASE = "ARM UR10E"


def request_execution_arm(
    actions_enabled: bool,
    *,
    input_fn: Callable[[str], str] = input,
) -> bool:
    """Return false for dry-run mode or require an exact phrase to enable motion."""
    if not actions_enabled:
        return False

    response = input_fn(
        "Robot actions are enabled. Verify the workspace and emergency stop, "
        f'then type "{ARM_PHRASE}" to continue: '
    )
    if response.strip() != ARM_PHRASE:
        raise RuntimeError("Robot execution was not armed; no actions were sent")
    return True
