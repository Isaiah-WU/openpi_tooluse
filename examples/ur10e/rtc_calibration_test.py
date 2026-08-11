from __future__ import annotations

from pathlib import Path

import pytest

from action_trajectory import should_request_action_chunk
from rtc_calibration import RTCDelayTracker
from rtc_calibration import load_baseline_policy_delays
from rtc_calibration import recommend_rtc_parameters
from rtc_calibration import validate_rtc_runtime_parameters


def test_loads_only_valid_baseline_delays(tmp_path: Path) -> None:
    csv_path = tmp_path / "request_timing.csv"
    csv_path.write_text(
        "request_id,rtc_applied,observed_delay_policy_steps\n"
        "0,0,6\n"
        "1,1,9\n"
        "2,0,-1\n"
        "3,0,nan\n"
        "4,False,7\n",
        encoding="utf-8",
    )

    assert load_baseline_policy_delays(csv_path) == [6.0, 7.0]


def test_combines_multiple_timing_runs(tmp_path: Path) -> None:
    for run_name, delay in [("run_a", 6), ("run_b", 9)]:
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        (run_dir / "request_timing.csv").write_text(
            "request_id,rtc_applied,observed_delay_policy_steps\n"
            f"0,0,{delay}\n",
            encoding="utf-8",
        )

    assert load_baseline_policy_delays(tmp_path) == [6.0, 9.0]


def test_recommends_p95_delay_and_p99_queue_margin() -> None:
    recommendation = recommend_rtc_parameters([6.0] * 20)

    assert recommendation.sample_count == 20
    assert recommendation.inference_delay_policy_steps == 6
    assert recommendation.execution_horizon_policy_steps == 10
    assert recommendation.query_remaining_policy_steps == 19


def test_execution_horizon_grows_with_measured_delay() -> None:
    recommendation = recommend_rtc_parameters([12.0] * 20)

    assert recommendation.inference_delay_policy_steps == 12
    assert recommendation.execution_horizon_policy_steps == 12
    assert recommendation.query_remaining_policy_steps == 27


def test_delay_tracker_never_drops_below_initial_calibration() -> None:
    tracker = RTCDelayTracker(6, maxlen=2)
    tracker.add(4)
    tracker.add(9)
    assert tracker.estimate() == 9

    tracker.add(3)
    tracker.add(2)
    assert tracker.estimate() == 6


def test_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        recommend_rtc_parameters([6.0] * 19)


def test_rejects_queue_threshold_without_overlap_margin() -> None:
    with pytest.raises(ValueError, match="greater than"):
        validate_rtc_runtime_parameters(
            inference_delay_policy_steps=6,
            execution_horizon_policy_steps=10,
            query_remaining_policy_steps=16,
            action_horizon_policy_steps=50,
        )


def test_baseline_scheduler_keeps_fixed_interval() -> None:
    assert not should_request_action_chunk(
        rtc_enabled=False,
        inflight=False,
        control_step=9,
        next_query_step=10,
        prefix_policy_steps=None,
        rtc_query_remaining_policy_steps=None,
    )
    assert should_request_action_chunk(
        rtc_enabled=False,
        inflight=False,
        control_step=10,
        next_query_step=10,
        prefix_policy_steps=None,
        rtc_query_remaining_policy_steps=None,
    )


def test_rtc_scheduler_uses_remaining_policy_prefix() -> None:
    assert not should_request_action_chunk(
        rtc_enabled=True,
        inflight=False,
        control_step=100,
        next_query_step=0,
        prefix_policy_steps=20,
        rtc_query_remaining_policy_steps=19,
    )
    assert should_request_action_chunk(
        rtc_enabled=True,
        inflight=False,
        control_step=101,
        next_query_step=0,
        prefix_policy_steps=17,
        rtc_query_remaining_policy_steps=19,
    )


def test_scheduler_never_submits_while_inflight() -> None:
    assert not should_request_action_chunk(
        rtc_enabled=True,
        inflight=True,
        control_step=100,
        next_query_step=0,
        prefix_policy_steps=10,
        rtc_query_remaining_policy_steps=19,
    )
