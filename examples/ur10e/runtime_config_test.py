from __future__ import annotations

import pytest

from runtime_config import UR10eRuntimeConfig


def test_defaults_preserve_safe_checked_in_mode() -> None:
    config = UR10eRuntimeConfig.from_env({})
    assert not config.use_async_runtime
    assert not config.robot_actions_enabled
    assert not config.rtc_enabled
    assert config.rtc_inference_delay_policy_steps is None


def test_environment_selects_reproducible_rtc_dry_run() -> None:
    config = UR10eRuntimeConfig.from_env(
        {
            "OPENPI_UR10E_ASYNC": "1",
            "OPENPI_UR10E_EXECUTE_ACTIONS": "0",
            "OPENPI_UR10E_RTC": "true",
            "OPENPI_UR10E_RTC_DELAY": "6",
            "OPENPI_UR10E_RTC_EXECUTION_HORIZON": "10",
            "OPENPI_UR10E_RTC_QUERY_REMAINING": "19",
        }
    )

    assert config.use_async_runtime
    assert not config.robot_actions_enabled
    assert config.rtc_enabled
    assert config.rtc_inference_delay_policy_steps == 6
    assert config.rtc_execution_horizon_policy_steps == 10
    assert config.rtc_query_remaining_policy_steps == 19


def test_invalid_boolean_fails_before_hardware_initialization() -> None:
    with pytest.raises(ValueError, match="OPENPI_UR10E_RTC"):
        UR10eRuntimeConfig.from_env({"OPENPI_UR10E_RTC": "maybe"})


def test_invalid_execution_horizon_fails_before_hardware_initialization() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        UR10eRuntimeConfig.from_env(
            {"OPENPI_UR10E_RTC_EXECUTION_HORIZON": "0"}
        )
