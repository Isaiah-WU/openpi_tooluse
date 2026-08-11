from __future__ import annotations

import math
from pathlib import Path

import pytest

from rtc_evaluation import RTCRunSummary
from rtc_evaluation import compare_run_summaries
from rtc_evaluation import summarize_run


def _summary(*, rtc_requests: int, actions_enabled: bool = False) -> RTCRunSummary:
    return RTCRunSummary(
        request_count=20,
        rtc_request_count=rtc_requests,
        rejected_chunk_count=0,
        actions_enabled=actions_enabled,
        client_infer_p95_ms=200.0 if rtc_requests == 0 else 240.0,
        observed_delay_p95_steps=6.0,
        control_cycle_p95_ms=101.0,
        achieved_control_hz=10.0 if rtc_requests == 0 else 10.5,
        boundary_arm_delta_p95=0.2 if rtc_requests == 0 else 0.1,
        boundary_arm_delta_max=0.3,
    )


def test_comparison_reports_frequency_and_boundary_changes() -> None:
    comparison = compare_run_summaries(_summary(rtc_requests=0), _summary(rtc_requests=19))

    assert math.isclose(comparison.control_frequency_change_percent, 5.0)
    assert math.isclose(comparison.boundary_delta_change_percent, -50.0)
    assert math.isclose(comparison.client_infer_change_percent, 20.0)


def test_comparison_rejects_mixed_execution_modes() -> None:
    with pytest.raises(ValueError, match="same actions_enabled"):
        compare_run_summaries(
            _summary(rtc_requests=0, actions_enabled=False),
            _summary(rtc_requests=19, actions_enabled=True),
        )


def test_comparison_requires_real_rtc_requests() -> None:
    with pytest.raises(ValueError, match="no completed RTC"):
        compare_run_summaries(_summary(rtc_requests=0), _summary(rtc_requests=0))


def test_summarizes_timestamped_run_csvs(tmp_path: Path) -> None:
    (tmp_path / "request_timing.csv").write_text(
        "request_id,accept_step,rtc_applied,chunk_rejected,client_infer_ms,"
        "observed_delay_policy_steps\n"
        "0,2,0,0,200,6\n"
        "1,4,1,0,240,7\n",
        encoding="utf-8",
    )
    (tmp_path / "control_cycle_timing.csv").write_text(
        "control_step,actions_enabled,chunk_boundary,action_source,"
        "arm_action_delta_l2,control_cycle_ms\n"
        "0,0,0,current_chunk,0.05,100\n"
        "1,0,1,new_chunk,0.10,100\n",
        encoding="utf-8",
    )

    summary = summarize_run(tmp_path)
    assert summary.request_count == 2
    assert summary.rtc_request_count == 1
    assert summary.achieved_control_hz == 10.0
    assert summary.boundary_arm_delta_p95 == 0.1
