import math

import pytest
import torch

from openpi.models_pytorch.rtc_processor import RTCInferenceConfig
from openpi.models_pytorch.rtc_processor import RTCProcessor


def _processor(schedule="exp", *, enabled=True, execution_horizon=6, max_guidance_weight=10.0):
    return RTCProcessor(
        RTCInferenceConfig(
            enabled=enabled,
            execution_horizon=execution_horizon,
            prefix_attention_schedule=schedule,
            max_guidance_weight=max_guidance_weight,
        )
    )


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        ("zeros", [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("ones", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ("linear", [1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_get_prefix_weights_matches_pi_reference(schedule, expected):
    weights = _processor(schedule).get_prefix_weights(2, 6, 10)

    torch.testing.assert_close(weights, torch.tensor(expected))


def test_exponential_prefix_weights_transform_linear_schedule():
    linear = _processor("linear").get_prefix_weights(2, 6, 10)
    exponential = _processor("exp").get_prefix_weights(2, 6, 10)
    expected = linear * torch.expm1(linear) / (math.e - 1)

    torch.testing.assert_close(exponential, expected)


def test_tensor_runtime_scalars_match_integer_prefix_weights():
    processor = _processor("exp")
    integer_weights = processor.get_prefix_weights(2, 6, 10)
    tensor_weights = processor.get_prefix_weights(
        torch.tensor(2, dtype=torch.int64),
        torch.tensor(6, dtype=torch.int64),
        10,
    )

    torch.testing.assert_close(tensor_weights, integer_weights)


def test_tensor_runtime_scalars_reuse_compiled_prefix_weight_graph():
    processor = _processor("linear")
    compile_count = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    def weights(delay, horizon):
        return processor.get_prefix_weights(delay, horizon, 10)

    compiled_weights = torch.compile(weights, backend=counting_backend, fullgraph=True)
    first = compiled_weights(torch.tensor(2), torch.tensor(6))
    second = compiled_weights(torch.tensor(3), torch.tensor(7))

    torch.testing.assert_close(first, processor.get_prefix_weights(2, 6, 10))
    torch.testing.assert_close(second, processor.get_prefix_weights(3, 7, 10))
    assert compile_count == 1


def test_no_prefix_is_exact_passthrough_without_enabling_grad():
    processor = _processor()
    x_t = torch.randn(1, 8, 3)
    saw_grad_enabled = []

    def denoiser(value):
        saw_grad_enabled.append(torch.is_grad_enabled())
        return value * 2

    with torch.no_grad():
        result = processor.denoise_step(x_t, None, 2, 0.5, denoiser)

    torch.testing.assert_close(result, x_t * 2)
    assert saw_grad_enabled == [False]


def test_disabled_processor_is_exact_passthrough():
    processor = _processor(enabled=False)
    x_t = torch.randn(1, 8, 3)
    previous = torch.randn(1, 6, 3)

    result = processor.denoise_step(x_t, previous, 2, 0.5, lambda value: value + 1)

    torch.testing.assert_close(result, x_t + 1)


def test_guidance_moves_reverse_euler_step_toward_previous_chunk():
    processor = _processor("zeros", execution_horizon=4)
    x_t = torch.zeros(1, 6, 1)
    previous = torch.ones(1, 4, 1)

    velocity = processor.denoise_step(x_t, previous, 2, 0.5, lambda value: value * 0)
    next_x = x_t - 0.1 * velocity

    assert torch.all(next_x[:, :2] > x_t[:, :2])
    torch.testing.assert_close(next_x[:, 2:], x_t[:, 2:])


def test_vjp_includes_denoiser_input_jacobian():
    processor = _processor("ones", execution_horizon=1)
    x_t = torch.full((1, 1, 1), 0.2)
    previous = torch.full((1, 1, 1), 0.8)

    velocity = processor.denoise_step(x_t, previous, 0, 0.5, lambda value: value.square())

    # clean estimate = x - 0.5*x^2 = 0.18; error = 0.62;
    # d(clean estimate)/dx = 0.8, so correction = 0.496.
    # At t=0.5 the unclipped RTC guidance coefficient is 2.
    expected = torch.tensor([[[-0.952]]])
    torch.testing.assert_close(velocity, expected)


@pytest.mark.parametrize(
    "config",
    [
        {"execution_horizon": 0},
        {"prefix_attention_schedule": "invalid"},
        {"max_guidance_weight": 0.0},
        {"max_guidance_weight": math.inf},
        {"enabled": True, "warmup_inferences": 1},
        {"warmup_inferences": -1},
    ],
)
def test_config_rejects_invalid_values(config):
    with pytest.raises(ValueError):
        RTCInferenceConfig(**config)


def test_rejects_leftover_longer_than_action_horizon():
    processor = _processor()
    x_t = torch.zeros(1, 4, 2)
    previous = torch.zeros(1, 5, 2)

    with pytest.raises(ValueError, match="cannot be longer"):
        processor.denoise_step(x_t, previous, 1, 0.5, lambda value: value)


def test_fixed_prefix_padding_matches_variable_prefix_guidance():
    processor = _processor("zeros", execution_horizon=4)
    x_t = torch.zeros(1, 6, 1)
    variable = torch.ones(1, 3, 1)
    fixed = torch.full((1, 6, 1), 99.0)
    fixed[:, :3] = variable

    variable_velocity = processor.denoise_step(x_t, variable, 2, 0.5, lambda value: value * 0)
    fixed_velocity = processor.denoise_step(
        x_t,
        fixed,
        2,
        0.5,
        lambda value: value * 0,
        previous_chunk_valid_steps=3,
    )

    torch.testing.assert_close(fixed_velocity, variable_velocity)


def test_tensor_runtime_scalars_match_integer_guidance():
    processor = _processor("linear", execution_horizon=4)
    x_t = torch.zeros(1, 6, 1)
    previous = torch.ones(1, 6, 1)
    denoiser = lambda value: value * 0

    integer_velocity = processor.denoise_step(
        x_t,
        previous,
        2,
        0.5,
        denoiser,
        execution_horizon=4,
        previous_chunk_valid_steps=3,
    )
    tensor_velocity = processor.denoise_step(
        x_t,
        previous,
        torch.tensor(2),
        0.5,
        denoiser,
        execution_horizon=torch.tensor(4),
        previous_chunk_valid_steps=torch.tensor(3),
    )

    torch.testing.assert_close(tensor_velocity, integer_velocity)
