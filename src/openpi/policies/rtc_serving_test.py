import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openpi.models import model as _model
from openpi.models_pytorch.rtc_processor import RTCInferenceConfig
from openpi.models_pytorch.rtc_processor import RTCProcessor
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def test_disabled_rtc_does_not_create_processor():
    assert _policy_config._create_rtc_processor(None, is_pytorch=True) is None  # noqa: SLF001
    assert (
        _policy_config._create_rtc_processor(  # noqa: SLF001
            RTCInferenceConfig(enabled=False),
            is_pytorch=True,
        )
        is None
    )


def test_enabled_rtc_creates_processor_only_for_pytorch():
    config = RTCInferenceConfig(enabled=True)

    processor = _policy_config._create_rtc_processor(config, is_pytorch=True)  # noqa: SLF001
    assert isinstance(processor, RTCProcessor)
    assert processor.config is config

    with pytest.raises(ValueError, match="PyTorch checkpoints"):
        _policy_config._create_rtc_processor(config, is_pytorch=False)  # noqa: SLF001


def test_pytorch_checkpoint_loader_accepts_rtc_processor():
    parameters = inspect.signature(_model.BaseModelConfig.load_pytorch).parameters

    assert "rtc_processor" in parameters


class _FakePytorchModel:
    action_horizon = 2

    def __init__(self):
        self.sample_kwargs = None
        self.sample_kwargs_history = []
        self.sample_observations = []
        self.config = SimpleNamespace(
            action_horizon=2,
            action_dim=3,
            fake_obs=lambda batch_size: _model.Observation(
                images={
                    "base_0_rgb": np.zeros((batch_size, 4, 5, 3), dtype=np.float32),
                },
                image_masks={
                    "base_0_rgb": np.ones((batch_size,), dtype=bool),
                },
                state=np.zeros((batch_size, 3), dtype=np.float32),
            ),
        )

    def to(self, _device):
        return self

    def eval(self):
        return self

    def sample_actions(self, _device, observation, **kwargs):
        self.sample_kwargs = kwargs
        self.sample_kwargs_history.append(kwargs)
        self.sample_observations.append(observation)
        return torch.zeros(observation.state.shape[0], 2, 3)


def _normalize_and_pad_prefix(data):
    data = dict(data)
    if "actions" in data:
        normalized = np.asarray(data["actions"], dtype=np.float32) * 2
        data["actions"] = np.pad(normalized, ((0, 0), (0, 1)))
    return data


def test_policy_normalizes_and_pads_previous_chunk_before_sampling(monkeypatch):
    monkeypatch.setattr(
        _model.Observation,
        "from_dict",
        staticmethod(lambda inputs: SimpleNamespace(state=inputs["state"])),
    )
    model = _FakePytorchModel()
    policy = _policy.Policy(
        model,
        transforms=[_normalize_and_pad_prefix],
        is_pytorch=True,
        pytorch_device="cpu",
    )
    previous = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    policy.infer(
        {"state": np.array([0.0, 0.0], dtype=np.float32)},
        prev_chunk_left_over=previous,
        prev_chunk_valid_steps=2,
        inference_delay=1,
        execution_horizon=2,
    )

    assert model.sample_kwargs is not None
    transformed = model.sample_kwargs["prev_chunk_left_over"]
    assert transformed.shape == (1, 2, 3)
    torch.testing.assert_close(
        transformed,
        torch.tensor([[[2.0, 4.0, 0.0], [6.0, 8.0, 0.0]]]),
    )
    torch.testing.assert_close(model.sample_kwargs["inference_delay"], torch.tensor(1))
    torch.testing.assert_close(model.sample_kwargs["execution_horizon"], torch.tensor(2))
    torch.testing.assert_close(model.sample_kwargs["prev_chunk_valid_steps"], torch.tensor(2))


def test_policy_default_request_does_not_add_rtc_kwargs(monkeypatch):
    monkeypatch.setattr(
        _model.Observation,
        "from_dict",
        staticmethod(lambda inputs: SimpleNamespace(state=inputs["state"])),
    )
    model = _FakePytorchModel()
    policy = _policy.Policy(
        model,
        is_pytorch=True,
        pytorch_device="cpu",
    )

    policy.infer({"state": np.array([0.0, 0.0], dtype=np.float32)})

    assert model.sample_kwargs == {}


def test_policy_rtc_warmup_discards_baseline_and_varies_scalar_tensors():
    model = _FakePytorchModel()
    policy = _policy.Policy(
        model,
        is_pytorch=True,
        pytorch_device="cpu",
    )

    timings = policy.warm_up_rtc(execution_horizon=2, warmup_inferences=2)

    assert len(timings) == 3
    assert all(seconds >= 0 for seconds in timings)
    assert model.sample_kwargs_history[0] == {}
    assert model.sample_observations[0].images["base_0_rgb"].shape == (1, 3, 4, 5)
    first_rtc, second_rtc = model.sample_kwargs_history[1:]
    assert first_rtc["prev_chunk_left_over"].shape == (1, 2, 3)
    assert second_rtc["prev_chunk_left_over"].shape == (1, 2, 3)
    torch.testing.assert_close(first_rtc["inference_delay"], torch.tensor(1))
    torch.testing.assert_close(second_rtc["inference_delay"], torch.tensor(2))
    torch.testing.assert_close(first_rtc["prev_chunk_valid_steps"], torch.tensor(2))
    torch.testing.assert_close(second_rtc["prev_chunk_valid_steps"], torch.tensor(1))


def test_policy_rtc_warmup_rejects_invalid_configuration():
    policy = _policy.Policy(
        _FakePytorchModel(),
        is_pytorch=True,
        pytorch_device="cpu",
    )

    with pytest.raises(ValueError, match="at least two"):
        policy.warm_up_rtc(execution_horizon=2, warmup_inferences=1)
    with pytest.raises(ValueError, match="fit the action horizon"):
        policy.warm_up_rtc(execution_horizon=3, warmup_inferences=2)


class _FakeWebsocket:
    def __init__(self, response):
        self.response = response
        self.sent = None

    def send(self, data):
        self.sent = data

    def recv(self):
        return self.response


def _fake_websocket_client():
    client = object.__new__(WebsocketClientPolicy)
    client._packer = msgpack_numpy.Packer()  # noqa: SLF001
    client._api_key = None  # noqa: SLF001
    client._ws = _FakeWebsocket(  # noqa: SLF001
        client._packer.pack({"actions": np.zeros((2, 3))})  # noqa: SLF001
    )
    return client


def test_websocket_client_keeps_default_payload_unchanged():
    client = _fake_websocket_client()
    observation = {"state": np.array([1.0, 2.0])}

    client.infer(observation)

    sent = msgpack_numpy.unpackb(client._ws.sent)  # noqa: SLF001
    np.testing.assert_array_equal(sent["state"], observation["state"])
    assert "observation" not in sent
    assert "infer_kwargs" not in sent


def test_websocket_client_sends_rtc_sampling_arguments():
    client = _fake_websocket_client()
    observation = {"state": np.array([1.0, 2.0])}
    previous = np.ones((2, 3))

    client.infer(
        observation,
        prev_chunk_left_over=previous,
        prev_chunk_valid_steps=2,
        inference_delay=1,
        execution_horizon=2,
    )

    sent = msgpack_numpy.unpackb(client._ws.sent)  # noqa: SLF001
    np.testing.assert_array_equal(sent["observation"]["state"], observation["state"])
    np.testing.assert_array_equal(sent["infer_kwargs"]["prev_chunk_left_over"], previous)
    assert sent["infer_kwargs"]["prev_chunk_valid_steps"] == 2
    assert sent["infer_kwargs"]["inference_delay"] == 1
    assert sent["infer_kwargs"]["execution_horizon"] == 2
