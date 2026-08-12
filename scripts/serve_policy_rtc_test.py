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
    raw_observation = {"observation/state": object()}

    timings = serve_policy.warm_up_policy_for_serving(
        policy,
        RTCInferenceConfig(enabled=True, execution_horizon=12, warmup_inferences=2),
        raw_observation=raw_observation,
    )

    assert timings == [1.0, 2.0, 3.0]
    assert policy.warmup_calls == [
        {
            "raw_observation": raw_observation,
            "execution_horizon": 10,
            "warmup_inferences": 2,
            "prev_chunk_valid_steps": 14,
            "inference_delay": 3,
            "expected_action_dim": 7,
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
        serve_policy,
        "warm_up_policy_for_serving",
        lambda _policy, _rtc, **_kwargs: (_ for _ in ()).throw(RuntimeError("warm-up failed")),
    )
    monkeypatch.setattr(
        serve_policy.websocket_policy_server,
        "WebsocketPolicyServer",
        lambda **_kwargs: pytest.fail("server must not be constructed after warm-up failure"),
    )

    with pytest.raises(RuntimeError, match="warm-up failed"):
        serve_policy.main(args)


def test_enabled_rtc_requires_ur10e_checkpoint(monkeypatch):
    policy = _FakePolicy()
    args = serve_policy.Args(
        rtc=RTCInferenceConfig(enabled=True),
        policy=serve_policy.Default(),
    )
    monkeypatch.setattr(serve_policy, "create_policy", lambda _args: policy)
    monkeypatch.setattr(
        serve_policy.websocket_policy_server,
        "WebsocketPolicyServer",
        lambda **_kwargs: pytest.fail("server must not be constructed without a UR10e warm-up input"),
    )

    with pytest.raises(ValueError, match="deployment-specific warm-up observation"):
        serve_policy.main(args)


def test_disabled_rtc_constructs_server_without_warmup_or_config_lookup(monkeypatch):
    policy = _FakePolicy()
    args = serve_policy.Args(
        rtc=RTCInferenceConfig(enabled=False),
        policy=serve_policy.Default(),
    )
    constructed = []

    monkeypatch.setattr(serve_policy, "create_policy", lambda _args: policy)
    monkeypatch.setattr(
        serve_policy._config,
        "get_config",
        lambda _name: pytest.fail("disabled RTC must not resolve a warm-up config"),
    )

    class _Server:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def serve_forever(self):
            return None

    monkeypatch.setattr(serve_policy.websocket_policy_server, "WebsocketPolicyServer", _Server)

    serve_policy.main(args)

    assert policy.warmup_calls == []
    assert len(constructed) == 1
    rtc_metadata = constructed[0]["metadata"]["openpi_server"]["rtc"]
    assert rtc_metadata["enabled"] is False
    assert "warmup_complete" not in rtc_metadata


def test_server_is_constructed_only_after_successful_warmup(monkeypatch):
    events = []
    policy = _FakePolicy()
    args = serve_policy.Args(
        rtc=RTCInferenceConfig(enabled=True),
        policy=serve_policy.Checkpoint(config="pi05_ur10e_long_horizon_lora", dir="unused"),
    )
    train_config = SimpleNamespace(data=serve_policy._config.LeRobotUR10eDataConfig())

    monkeypatch.setattr(serve_policy, "create_policy", lambda _args: policy)
    monkeypatch.setattr(serve_policy._config, "get_config", lambda _name: train_config)

    original_warmup = serve_policy.warm_up_policy_for_serving

    def tracked_warmup(*warmup_args, **warmup_kwargs):
        events.append("warmup")
        return original_warmup(*warmup_args, **warmup_kwargs)

    class _Server:
        def __init__(self, **kwargs):
            events.append("server")
            rtc_metadata = kwargs["metadata"]["openpi_server"]["rtc"]
            assert rtc_metadata["warmup_complete"] is True
            assert rtc_metadata["warmup_inferences"] == 2

        def serve_forever(self):
            return None

    monkeypatch.setattr(serve_policy, "warm_up_policy_for_serving", tracked_warmup)
    monkeypatch.setattr(serve_policy.websocket_policy_server, "WebsocketPolicyServer", _Server)

    serve_policy.main(args)

    assert events == ["warmup", "server"]
