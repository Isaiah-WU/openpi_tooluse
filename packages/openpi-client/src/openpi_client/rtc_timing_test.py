import pytest

from openpi_client.rtc_timing import ControlCycleTiming


def test_control_cycle_exports_command_to_state_metrics() -> None:
    cycle = ControlCycleTiming(
        control_step=3,
        cycle_start=1.0,
        arm_target_error_l2=0.12,
        max_abs_arm_target_error=0.08,
        gripper_target_error_abs=0.25,
    )

    metrics = cycle.as_metrics()

    assert metrics["arm_target_error_l2"] == pytest.approx(0.12)
    assert metrics["max_abs_arm_target_error"] == pytest.approx(0.08)
    assert metrics["gripper_target_error_abs"] == pytest.approx(0.25)
