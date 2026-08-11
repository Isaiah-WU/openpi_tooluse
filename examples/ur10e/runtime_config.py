"""Environment-backed runtime switches for reproducible UR10e rollouts."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping


def _read_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _read_optional_int(
    env: Mapping[str, str],
    name: str,
    default: int | None,
) -> int | None:
    value = env.get(name)
    if value is None:
        return default
    if value.strip().lower() in {"", "none"}:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer or none, got {value!r}") from exc


def _read_positive_int(
    env: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    value = env.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


@dataclasses.dataclass(frozen=True)
class UR10eRuntimeConfig:
    """Runtime-only switches; defaults preserve the checked-in safe behavior."""

    use_async_runtime: bool = False
    robot_actions_enabled: bool = False
    rtc_enabled: bool = False
    rtc_inference_delay_policy_steps: int | None = None
    rtc_execution_horizon_policy_steps: int = 10
    rtc_query_remaining_policy_steps: int | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> UR10eRuntimeConfig:
        values = os.environ if env is None else env
        return cls(
            use_async_runtime=_read_bool(
                values,
                "OPENPI_UR10E_ASYNC",
                False,
            ),
            robot_actions_enabled=_read_bool(
                values,
                "OPENPI_UR10E_EXECUTE_ACTIONS",
                False,
            ),
            rtc_enabled=_read_bool(
                values,
                "OPENPI_UR10E_RTC",
                False,
            ),
            rtc_inference_delay_policy_steps=_read_optional_int(
                values,
                "OPENPI_UR10E_RTC_DELAY",
                None,
            ),
            rtc_execution_horizon_policy_steps=_read_positive_int(
                values,
                "OPENPI_UR10E_RTC_EXECUTION_HORIZON",
                10,
            ),
            rtc_query_remaining_policy_steps=_read_optional_int(
                values,
                "OPENPI_UR10E_RTC_QUERY_REMAINING",
                None,
            ),
        )
