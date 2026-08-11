from __future__ import annotations

import numpy as np
import pytest

from action_trajectory import get_policy_action_leftover
from action_trajectory import prepare_action_chunk
from async_policy import _build_rtc_infer_kwargs
from runtime_timing import RequestTiming


def _actions(length: int) -> np.ndarray:
    values = np.arange(length, dtype=np.float32)
    return np.repeat(values[:, None], 7, axis=1)


def test_leftover_is_kept_at_policy_frequency() -> None:
    leftover = get_policy_action_leftover(
        _actions(12),
        consumed_control_steps=2,
        control_hz=10.0,
        policy_hz=30.0,
    )

    assert leftover.shape == (6, 7)
    np.testing.assert_array_equal(leftover[:, 0], np.arange(6, 12))


def test_rtc_response_is_cropped_by_actual_delay() -> None:
    prepared = prepare_action_chunk(
        _actions(12),
        observed_delay_control_steps=2,
        apply_rtc_delay_crop=True,
        policy_hz=30.0,
        control_hz=10.0,
    )

    assert prepared.observed_delay_policy_steps == 6
    assert prepared.skipped_policy_steps == 6
    np.testing.assert_array_equal(prepared.policy_actions[:, 0], np.arange(6, 12))
    np.testing.assert_allclose(prepared.control_actions[:, 0], [6.0, 9.0])


def test_baseline_response_measures_delay_without_cropping() -> None:
    prepared = prepare_action_chunk(
        _actions(12),
        observed_delay_control_steps=2,
        apply_rtc_delay_crop=False,
        policy_hz=30.0,
        control_hz=10.0,
    )

    assert prepared.observed_delay_policy_steps == 6
    assert prepared.skipped_policy_steps == 0
    np.testing.assert_array_equal(prepared.policy_actions, _actions(12))


def test_fully_stale_rtc_response_is_empty() -> None:
    prepared = prepare_action_chunk(
        _actions(6),
        observed_delay_control_steps=2,
        apply_rtc_delay_crop=True,
        policy_hz=30.0,
        control_hz=10.0,
    )

    assert prepared.policy_actions.shape == (0, 7)
    assert prepared.control_actions.shape == (0, 7)


def test_first_request_keeps_original_protocol() -> None:
    assert _build_rtc_infer_kwargs(
        None,
        rtc_enabled=True,
        inference_delay_steps=6,
        execution_horizon=10,
        action_dim=7,
    ) == {}


def test_rtc_request_uses_policy_rate_prefix() -> None:
    prefix = _actions(8)
    kwargs = _build_rtc_infer_kwargs(
        prefix,
        rtc_enabled=True,
        inference_delay_steps=6,
        execution_horizon=10,
        action_dim=7,
    )

    np.testing.assert_array_equal(kwargs["prev_chunk_left_over"], prefix)
    assert kwargs["inference_delay"] == 6
    assert kwargs["execution_horizon"] == 10


def test_rtc_request_requires_horizon_at_least_delay() -> None:
    with pytest.raises(ValueError, match="at least inference_delay_steps"):
        _build_rtc_infer_kwargs(
            _actions(8),
            rtc_enabled=True,
            inference_delay_steps=12,
            execution_horizon=10,
            action_dim=7,
        )


@pytest.mark.parametrize("delay", [-1, 1.5, True])
def test_rtc_request_rejects_invalid_delay(delay: object) -> None:
    with pytest.raises(ValueError, match="inference_delay_steps"):
        _build_rtc_infer_kwargs(
            _actions(2),
            rtc_enabled=True,
            inference_delay_steps=delay,  # type: ignore[arg-type]
            execution_horizon=10,
            action_dim=7,
        )


def test_timing_exports_prediction_and_actual_alignment() -> None:
    timing = RequestTiming(
        request_id=3,
        observation_step=10,
        submit_step=10,
        observation_start=1.0,
        observation_ready=1.01,
        request_submit=1.02,
        rtc_prefix_steps=20,
        rtc_inference_delay_steps=6,
        rtc_applied=True,
        observed_delay_policy_steps=9,
        rtc_skipped_policy_steps=9,
        accept_step=13,
    )

    metrics = timing.as_metrics()
    assert metrics["rtc_inference_delay_steps"] == 6
    assert metrics["observed_delay_steps"] == 3
    assert metrics["observed_delay_policy_steps"] == 9
    assert metrics["rtc_skipped_policy_steps"] == 9
