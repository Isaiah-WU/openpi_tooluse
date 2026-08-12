from pathlib import Path

import numpy as np
import pytest

from openpi_client.rtc_latency_benchmark import RTCLatencySample
from openpi_client.rtc_latency_benchmark import IncrementalRTCLatencyWriter
from openpi_client.rtc_latency_benchmark import recommend_from_rtc_latency
from openpi_client.rtc_latency_benchmark import run_rtc_latency_benchmark
from openpi_client.rtc_latency_benchmark import save_rtc_latency_samples
from openpi_client.server_capabilities import add_rtc_server_capability


class _FakePolicy:
    def __init__(self):
        self.calls = []
        self.returned_actions = []

    def get_server_metadata(self):
        return add_rtc_server_capability(
            {},
            rtc_enabled=True,
            execution_horizon=10,
            prefix_attention_schedule="exp",
            max_guidance_weight=10.0,
            warmup_complete=True,
            warmup_inferences=2,
        )

    def infer(self, observation, **kwargs):
        self.calls.append((observation, kwargs))
        actions = np.full((50, 7), len(self.calls), dtype=np.float32)
        self.returned_actions.append(actions)
        return {
            "actions": actions,
            "server_timing": {"infer_ms": 20.0},
            "policy_timing": {"infer_ms": 15.0},
        }


def _sample(delay: float, index: int = 0) -> RTCLatencySample:
    return RTCLatencySample(
        request_index=index,
        rtc_total_ms=delay / 30.0 * 1000.0,
        observed_delay_policy_steps=delay,
        server_infer_ms=20.0,
        policy_infer_ms=15.0,
    )


def test_benchmark_excludes_baseline_and_recursively_uses_rtc_chunks():
    policy = _FakePolicy()
    observation = {"observation/state": np.zeros(7, dtype=np.float32)}
    clock = iter([0.0, 0.1, 1.0, 1.2])

    reported = []
    samples = run_rtc_latency_benchmark(
        policy,
        observation,
        sample_count=2,
        clock=lambda: next(clock),
        on_sample=reported.append,
    )

    assert len(policy.calls) == 3
    assert policy.calls[0] == (observation, {})
    first_rtc = policy.calls[1][1]
    second_rtc = policy.calls[2][1]
    assert first_rtc["prev_chunk_left_over"] is policy.returned_actions[0]
    assert second_rtc["prev_chunk_left_over"] is policy.returned_actions[1]
    for kwargs in (first_rtc, second_rtc):
        assert kwargs["prev_chunk_left_over"].shape == (50, 7)
        assert kwargs["prev_chunk_valid_steps"] == 14
        assert kwargs["inference_delay"] == 3
        assert kwargs["execution_horizon"] == 10
        assert "noise" not in kwargs
    assert samples[0].rtc_total_ms == pytest.approx(100.0)
    assert samples[0].observed_delay_policy_steps == pytest.approx(3.0)
    assert samples[1].observed_delay_policy_steps == pytest.approx(6.0)
    assert reported == samples


def test_benchmark_requires_completed_fixed_shape_server_warmup():
    policy = _FakePolicy()
    metadata = add_rtc_server_capability(
        {},
        rtc_enabled=True,
        execution_horizon=10,
        prefix_attention_schedule="exp",
        max_guidance_weight=10.0,
    )
    policy.get_server_metadata = lambda: metadata

    with pytest.raises(RuntimeError, match="completed server warm-up"):
        run_rtc_latency_benchmark(policy, {}, sample_count=1)

    assert policy.calls == []


def test_recommendation_uses_rtc_p99_for_d_s_and_q():
    samples = [_sample(4.2, index) for index in range(119)] + [_sample(8.2, 119)]

    recommendation = recommend_from_rtc_latency(samples)

    assert recommendation.sample_count == 120
    assert recommendation.inference_delay_policy_steps == 5
    assert recommendation.execution_horizon_policy_steps == 10
    assert recommendation.query_remaining_policy_steps == 16


def test_recommendation_fails_closed_when_rtc_delay_cannot_fit_horizon():
    samples = [_sample(25.0, index) for index in range(120)]

    with pytest.raises(ValueError, match="cannot fit the action horizon"):
        recommend_from_rtc_latency(samples)


def test_saves_standalone_rtc_latency_csv(tmp_path: Path):
    path = tmp_path / "rtc_latency.csv"

    save_rtc_latency_samples(path, [_sample(3.0)])

    text = path.read_text(encoding="utf-8")
    assert "request_index,rtc_total_ms,observed_delay_policy_steps,server_infer_ms,policy_infer_ms" in text
    assert "3.0" in text


def test_incremental_writer_flushes_each_completed_sample(tmp_path: Path):
    path = tmp_path / "rtc_latency.csv"
    writer = IncrementalRTCLatencyWriter(path)
    try:
        writer.write(_sample(3.0))
        text_while_open = path.read_text(encoding="utf-8")
    finally:
        writer.close()

    assert "request_index,rtc_total_ms,observed_delay_policy_steps" in text_while_open
    assert "3.0" in text_while_open
