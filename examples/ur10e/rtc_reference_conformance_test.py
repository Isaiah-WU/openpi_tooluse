"""Differential checks against Physical Intelligence's published RTC equations."""

from __future__ import annotations

import math

import numpy as np
import torch

from action_trajectory import prepare_action_chunk
from openpi.models_pytorch.rtc_processor import RTCInferenceConfig
from openpi.models_pytorch.rtc_processor import RTCProcessor


def _pi_prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: str,
) -> np.ndarray:
    """Independent NumPy transcription of PI's JAX get_prefix_weights()."""
    start = min(start, end)
    indices = np.arange(total, dtype=np.float64)
    if schedule == "ones":
        weights = np.ones(total, dtype=np.float64)
    elif schedule == "zeros":
        weights = (indices < start).astype(np.float64)
    else:
        weights = np.clip(
            (start - 1 - indices) / (end - start + 1) + 1,
            0,
            1,
        )
        if schedule == "exp":
            weights = weights * np.expm1(weights) / (math.e - 1)
    return np.where(indices >= end, 0, weights)


def test_prefix_weights_match_pi_reference_across_boundaries() -> None:
    cases = [(2, 6, 10), (0, 6, 10), (7, 4, 10), (3, 10, 10)]
    for schedule in ("linear", "exp", "ones", "zeros"):
        processor = RTCProcessor(
            RTCInferenceConfig(
                enabled=True,
                execution_horizon=6,
                prefix_attention_schedule=schedule,
            )
        )
        for start, end, total in cases:
            actual = processor.get_prefix_weights(
                start,
                end,
                total,
                dtype=torch.float64,
            )
            expected = torch.from_numpy(
                _pi_prefix_weights(start, end, total, schedule)
            )
            torch.testing.assert_close(actual, expected)


def test_reverse_time_vjp_matches_finite_difference_oracle() -> None:
    processor = RTCProcessor(
        RTCInferenceConfig(
            enabled=True,
            execution_horizon=2,
            prefix_attention_schedule="ones",
            max_guidance_weight=10.0,
        )
    )
    x_t = torch.tensor([[[-0.4], [0.2]]], dtype=torch.float64)
    previous = torch.tensor([[[0.3], [0.7]]], dtype=torch.float64)
    time = 0.4

    def denoiser(value: torch.Tensor) -> torch.Tensor:
        return 0.3 * value.square() + torch.sin(value)

    actual = processor.denoise_step(
        x_t,
        previous,
        inference_delay=0,
        time=time,
        denoiser=denoiser,
    )

    x_np = x_t.numpy()
    previous_np = previous.numpy()

    def velocity(value: np.ndarray) -> np.ndarray:
        return 0.3 * value**2 + np.sin(value)

    def clean_estimate(value: np.ndarray) -> np.ndarray:
        return value - time * velocity(value)

    epsilon = 1e-6
    derivative = (
        clean_estimate(x_np + epsilon) - clean_estimate(x_np - epsilon)
    ) / (2 * epsilon)
    clean = clean_estimate(x_np)
    correction = derivative * (previous_np - clean)
    tau = 1 - time
    guidance_weight = min(
        (time / tau) * ((time**2 + tau**2) / time**2),
        10.0,
    )
    expected = velocity(x_np) - guidance_weight * correction

    np.testing.assert_allclose(
        actual.numpy(),
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_runtime_alignment_matches_pi_eval_flow_chunk_splice() -> None:
    delay = 6
    execution_horizon = 10
    previous = np.arange(100, 112, dtype=np.float32)[:, None]
    new = np.arange(12, dtype=np.float32)[:, None]

    prepared = prepare_action_chunk(
        new,
        observed_delay_control_steps=2,
        apply_rtc_delay_crop=True,
        policy_hz=30.0,
        control_hz=10.0,
    )
    runtime_actions = np.concatenate(
        [
            previous[:delay],
            prepared.policy_actions[: execution_horizon - delay],
        ],
        axis=0,
    )
    pi_reference_actions = np.concatenate(
        [previous[:delay], new[delay:execution_horizon]],
        axis=0,
    )

    np.testing.assert_array_equal(runtime_actions, pi_reference_actions)
