from types import SimpleNamespace

import pytest

import scripts.serve_policy as serve_policy
from openpi.models_pytorch.rtc_processor import RTCInferenceConfig


class _FakePolicy:
    metadata = {}

    def __init__(self, *, failure: Exception | None = None):
        self.failure = failure
        self.warmup_calls = []

    def warm_up_rtc(self, **kwargs):
        self.warmup_calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return [1.0, 2.0, 3.0]


def test_disabled_rtc_does_not_warm_up_policy():
    policy = _FakePolicy()

    timings = serve_policy.warm_up_policy_for_serving(
        policy,
        RTCInferenceConfig(enabled=False),
    )

    assert timings is None
    assert policy.warmup_calls == []


def test_enabled_rtc_warms_baseline_and_configured_rtc_paths():
    policy = _FakePolicy()

    timings = serve_policy.warm_up_policy_for_serving(
        policy,
        RTCInferenceConfig(enabled=True, execution_horizon=10, warmup_inferences=2),
    )

    assert timings == [1.0, 2.0, 3.0]
    assert policy.warmup_calls == [
        {
            "execution_horizon": 10,
            "warmup_inferences": 2,
        }
    ]


def test_warmup_failure_prevents_server_construction(monkeypatch):
    policy = _FakePolicy(failure=RuntimeError("warm-up failed"))
    args = serve_policy.Args(
        rtc=RTCInferenceConfig(enabled=True),
        policy=serve_policy.Default(),
    )
    monkeypatch.setattr(serve_policy, "create_policy", lambda _args: policy)
    monkeypatch.setattr(
        serve_policy.websocket_policy_server,
        "WebsocketPolicyServer",
        lambda **_kwargs: pytest.fail("server must not be constructed after warm-up failure"),
    )

    with pytest.raises(RuntimeError, match="warm-up failed"):
        serve_policy.main(args)
