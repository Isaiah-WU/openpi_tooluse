import inspect
from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import _denoise_with_optional_rtc
from openpi.models_pytorch.pi0_pytorch import _prepare_rtc_sampling
from openpi.models_pytorch.rtc_processor import RTCInferenceConfig
from openpi.models_pytorch.rtc_processor import RTCProcessor


def _processor(*, enabled=True):
    return RTCProcessor(
        RTCInferenceConfig(
            enabled=enabled,
            execution_horizon=4,
            prefix_attention_schedule="zeros",
        )
    )


def test_sample_actions_exposes_lerobot_compatible_rtc_arguments():
    assert "rtc_processor" in inspect.signature(PI0Pytorch).parameters

    parameters = inspect.signature(PI0Pytorch.sample_actions).parameters

    assert "prev_chunk_left_over" in parameters
    assert "prev_chunk_valid_steps" in parameters
    assert "inference_delay" in parameters
    assert "execution_horizon" in parameters


def test_first_chunk_keeps_original_sampler_even_if_rtc_is_enabled():
    active_processor, inference_delay = _prepare_rtc_sampling(_processor(), None, None)

    assert active_processor is None
    assert inference_delay == 0


@pytest.mark.parametrize("processor", [None, _processor(enabled=False)])
def test_previous_chunk_requires_enabled_processor(processor):
    previous = torch.ones(1, 4, 1)

    with pytest.raises(ValueError, match="enabled RTCProcessor"):
        _prepare_rtc_sampling(processor, previous, 2)


def test_previous_chunk_requires_measured_inference_delay():
    previous = torch.ones(1, 4, 1)

    with pytest.raises(ValueError, match="requires inference_delay"):
        _prepare_rtc_sampling(_processor(), previous, None)


@pytest.mark.parametrize("inference_delay", [-1, 1.5, True])
def test_inference_delay_must_be_non_negative_integer(inference_delay):
    previous = torch.ones(1, 4, 1)

    with pytest.raises(ValueError, match="non-negative integer"):
        _prepare_rtc_sampling(_processor(), previous, inference_delay)


def test_valid_rtc_inputs_activate_attached_processor():
    processor = _processor()
    previous = torch.ones(1, 4, 1)

    active_processor, inference_delay = _prepare_rtc_sampling(processor, previous, 2)

    assert active_processor is processor
    assert inference_delay == 2


def test_optional_rtc_wrapper_is_exact_passthrough_without_processor():
    x_t = torch.full((1, 4, 1), 0.25)

    result = _denoise_with_optional_rtc(
        None,
        lambda value: value.square(),
        x_t,
        prev_chunk_left_over=None,
        inference_delay=0,
        time=torch.tensor(0.5),
        execution_horizon=None,
    )

    torch.testing.assert_close(result, x_t.square())


def test_optional_rtc_wrapper_applies_prefix_guidance():
    processor = _processor()
    x_t = torch.zeros(1, 4, 1)
    previous = torch.ones(1, 4, 1)

    velocity = _denoise_with_optional_rtc(
        processor,
        lambda value: value * 0,
        x_t,
        prev_chunk_left_over=previous,
        inference_delay=2,
        time=torch.tensor(0.5),
        execution_horizon=4,
    )

    next_x = x_t - 0.1 * velocity
    assert torch.all(next_x[:, :2] > x_t[:, :2])
    torch.testing.assert_close(next_x[:, 2:], x_t[:, 2:])


class _FakePaliGemmaWithExpert:
    def __init__(self):
        language_model = SimpleNamespace(config=SimpleNamespace(_attn_implementation=None))
        self.paligemma = SimpleNamespace(language_model=language_model)

    def forward(self, **_kwargs):
        return (None, None), "fake_cache"


def _fake_model(rtc_processor, denoise_step):
    model = SimpleNamespace(
        config=SimpleNamespace(action_horizon=2, action_dim=1),
        rtc_processor=rtc_processor,
        paligemma_with_expert=_FakePaliGemmaWithExpert(),
    )
    model._preprocess_observation = lambda _observation, train: (
        [],
        [],
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1),
    )
    model.embed_prefix = lambda _images, _img_masks, _tokens, _masks: (
        torch.zeros(1, 1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.bool),
    )
    model._prepare_attention_masks_4d = lambda mask: mask[:, None]
    model.denoise_step = denoise_step
    return model


def test_sample_actions_default_path_keeps_original_euler_loop():
    calls = []

    def denoise_step(_state, _prefix_masks, _cache, x_t, timestep):
        calls.append(timestep.clone())
        return torch.ones_like(x_t)

    model = _fake_model(None, denoise_step)
    observation = SimpleNamespace(state=torch.zeros(1, 1))
    noise = torch.zeros(1, 2, 1)

    actions = PI0Pytorch.sample_actions(model, "cpu", observation, noise=noise, num_steps=2)

    torch.testing.assert_close(actions, torch.full_like(actions, -1.0))
    assert len(calls) == 2


def test_sample_actions_routes_previous_chunk_through_rtc():
    def denoise_step(_state, _prefix_masks, _cache, x_t, _timestep):
        return x_t * 0

    model = _fake_model(_processor(), denoise_step)
    observation = SimpleNamespace(state=torch.zeros(1, 1))
    noise = torch.zeros(1, 2, 1)
    previous = torch.ones(1, 2, 1)

    actions = PI0Pytorch.sample_actions(
        model,
        "cpu",
        observation,
        noise=noise,
        num_steps=2,
        prev_chunk_left_over=previous,
        inference_delay=1,
        execution_horizon=2,
    )

    torch.testing.assert_close(actions, torch.tensor([[[1.0], [0.0]]]))


def test_sample_actions_ignores_fixed_prefix_padding():
    def denoise_step(_state, _prefix_masks, _cache, x_t, _timestep):
        return x_t * 0

    model = _fake_model(_processor(), denoise_step)
    observation = SimpleNamespace(state=torch.zeros(1, 1))
    noise = torch.zeros(1, 2, 1)
    previous = torch.tensor([[[1.0], [99.0]]])

    actions = PI0Pytorch.sample_actions(
        model,
        "cpu",
        observation,
        noise=noise,
        num_steps=2,
        prev_chunk_left_over=previous,
        prev_chunk_valid_steps=1,
        inference_delay=1,
        execution_horizon=2,
    )

    torch.testing.assert_close(actions, torch.tensor([[[1.0], [0.0]]]))
