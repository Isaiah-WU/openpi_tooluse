"""Inference-time Real-Time Chunking guidance for PyTorch action samplers.

The implementation follows the RTC VJP correction from Physical Intelligence's
reference implementation, adapted to OpenPI's reverse-time convention where
``time`` runs from 1 (noise) to 0 (clean actions).

References:
    https://arxiv.org/abs/2506.07339
    https://github.com/Physical-Intelligence/real-time-chunking-kinetix
    https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/modeling_rtc.py
"""

from collections.abc import Callable
import dataclasses
import math
from typing import Literal, TypeAlias

import torch
from torch import Tensor


PrefixAttentionSchedule: TypeAlias = Literal["exp", "linear", "ones", "zeros"]
StepScalar: TypeAlias = int | Tensor


@dataclasses.dataclass(frozen=True)
class RTCInferenceConfig:
    """Configuration for inference-time RTC guidance.

    The numeric defaults follow the current LeRobot RTC implementation, while
    RTC stays disabled until the sampler is explicitly wired to use it. The
    exponential prefix schedule follows the PI reference. Runtime code should
    override ``execution_horizon`` using the measured UR10e latency.
    """

    enabled: bool = False
    execution_horizon: int = 10
    prefix_attention_schedule: PrefixAttentionSchedule = "exp"
    max_guidance_weight: float = 10.0

    def __post_init__(self) -> None:
        if self.execution_horizon <= 0:
            raise ValueError(f"execution_horizon must be positive, got {self.execution_horizon}")
        if self.prefix_attention_schedule not in {"exp", "linear", "ones", "zeros"}:
            raise ValueError(f"invalid prefix_attention_schedule: {self.prefix_attention_schedule}")
        if not math.isfinite(self.max_guidance_weight) or self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be finite and positive, got {self.max_guidance_weight}")


class RTCProcessor:
    """Apply RTC prefix guidance to one reverse-time flow denoising step."""

    def __init__(self, config: RTCInferenceConfig):
        self.config = config

    def get_prefix_weights(
        self,
        inference_delay: StepScalar,
        execution_horizon: StepScalar,
        action_horizon: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Return PI-style prefix weights with shape ``(action_horizon,)``.

        Actions before ``inference_delay`` are fixed. Actions from there to
        ``execution_horizon`` transition from the old chunk to free replanning,
        and all later actions receive no prefix guidance.
        """
        inference_delay = self._as_step_tensor(
            inference_delay,
            name="inference_delay",
            device=device,
            minimum=0,
        )
        execution_horizon = self._as_step_tensor(
            execution_horizon,
            name="execution_horizon",
            device=device,
            minimum=0,
            maximum=action_horizon,
        )
        # Policy.infer validates the public integer inputs before converting
        # them to tensors. Keep this compiled path free of tensor-valued Python
        # assertions: PyTorch 2.7 lowers those through _local_scalar_dense and
        # breaks full-graph compilation. Clamping is a branch-free safeguard
        # for direct tensor callers and is a no-op for validated requests.
        inference_delay = inference_delay.clamp(min=0)
        execution_horizon = execution_horizon.clamp(min=0, max=action_horizon)

        start = torch.minimum(inference_delay, execution_horizon)
        indices = torch.arange(action_horizon, device=device, dtype=dtype)

        if self.config.prefix_attention_schedule == "ones":
            weights = torch.ones(action_horizon, device=device, dtype=dtype)
        elif self.config.prefix_attention_schedule == "zeros":
            weights = (indices < start).to(dtype=dtype)
        else:
            denominator = (execution_horizon - start + 1).to(dtype=dtype)
            weights = ((start.to(dtype=dtype) - 1 - indices) / denominator + 1).clamp(0, 1)
            if self.config.prefix_attention_schedule == "exp":
                weights = weights * torch.expm1(weights) / (math.e - 1)

        return torch.where(indices < execution_horizon, weights, 0.0)

    def denoise_step(
        self,
        x_t: Tensor,
        previous_chunk_leftover: Tensor | None,
        inference_delay: StepScalar,
        time: float | Tensor,
        denoiser: Callable[[Tensor], Tensor],
        *,
        execution_horizon: StepScalar | None = None,
        previous_chunk_valid_steps: StepScalar | None = None,
    ) -> Tensor:
        """Return the base or RTC-guided reverse-time velocity.

        ``denoiser`` must accept and return ``(batch, horizon, action_dim)``.
        If RTC is disabled or there is no leftover prefix, this method is an
        exact pass-through and does not enable autograd.
        """
        if not self.config.enabled or previous_chunk_leftover is None:
            return denoiser(x_t)

        self._validate_inputs(x_t, previous_chunk_leftover)
        if previous_chunk_leftover.shape[1] == 0:
            return denoiser(x_t)
        local_time = torch.as_tensor(time, device=x_t.device, dtype=x_t.dtype)
        if local_time.numel() != 1:
            raise ValueError(f"time must be scalar, got shape {tuple(local_time.shape)}")
        local_time = local_time.clamp(0.0, 1.0)

        action_horizon = x_t.shape[1]
        prefix_length = previous_chunk_leftover.shape[1]
        valid_steps = self._as_step_tensor(
            prefix_length if previous_chunk_valid_steps is None else previous_chunk_valid_steps,
            name="previous_chunk_valid_steps",
            device=x_t.device,
            minimum=0,
            maximum=prefix_length,
        )
        valid_steps = valid_steps.clamp(min=0, max=prefix_length)
        requested_horizon = self.config.execution_horizon if execution_horizon is None else execution_horizon
        requested_horizon = self._as_step_tensor(
            requested_horizon,
            name="execution_horizon",
            device=x_t.device,
            minimum=1,
        )
        requested_horizon = requested_horizon.clamp(min=1, max=action_horizon)
        effective_horizon = torch.minimum(requested_horizon, valid_steps)

        previous_chunk = torch.zeros_like(x_t)
        previous_chunk[:, :prefix_length] = previous_chunk_leftover.to(device=x_t.device, dtype=x_t.dtype)
        weights = self.get_prefix_weights(
            inference_delay,
            effective_horizon,
            action_horizon,
            device=x_t.device,
            dtype=x_t.dtype,
        )[None, :, None]

        # sample_actions() is globally no_grad, but RTC needs a VJP through the
        # denoiser with respect to x_t. Requiring grad before the denoiser call
        # is essential: otherwise the VJP omits the denoiser's input Jacobian.
        with torch.enable_grad():
            guided_x_t = x_t.detach().clone().requires_grad_(True)
            base_velocity = denoiser(guided_x_t)
            if base_velocity.shape != guided_x_t.shape:
                raise ValueError(
                    f"denoiser output must match x_t shape {tuple(guided_x_t.shape)}, "
                    f"got {tuple(base_velocity.shape)}"
                )

            # This repository integrates from t=1 to t=0 with a negative Euler
            # step, so its clean-action estimate has the opposite sign from the
            # PI reference's forward-time expression.
            clean_action_estimate = guided_x_t - local_time * base_velocity
            error = (previous_chunk - clean_action_estimate) * weights
            correction = torch.autograd.grad(
                clean_action_estimate,
                guided_x_t,
                grad_outputs=error.detach(),
                retain_graph=False,
                create_graph=False,
            )[0]

            guidance_weight = self._guidance_weight(local_time)
            guided_velocity = base_velocity - guidance_weight * correction

        return guided_velocity.detach()

    def _guidance_weight(self, local_time: Tensor) -> Tensor:
        """Convert PI's forward-time guidance coefficient to local time."""
        tau = 1 - local_time
        squared_one_minus_tau = local_time.square()
        inverse_r_squared = (squared_one_minus_tau + tau.square()) / squared_one_minus_tau
        coefficient = local_time / tau
        raw_weight = coefficient * inverse_r_squared
        return torch.nan_to_num(
            raw_weight,
            nan=0.0,
            posinf=self.config.max_guidance_weight,
            neginf=0.0,
        ).clamp(min=0.0, max=self.config.max_guidance_weight)

    @staticmethod
    def _as_step_tensor(
        value: StepScalar,
        *,
        name: str,
        device: torch.device | str | None,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> Tensor:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer scalar")
        if isinstance(value, int):
            if minimum is not None and value < minimum:
                raise ValueError(f"{name} must be at least {minimum}, got {value}")
            if maximum is not None and value > maximum:
                raise ValueError(f"{name} must be at most {maximum}, got {value}")
            return torch.tensor(value, dtype=torch.int64, device=device)
        if not isinstance(value, Tensor) or value.numel() != 1 or value.dtype == torch.bool:
            raise ValueError(f"{name} must be an integer scalar tensor")
        if value.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError(f"{name} must use an integer dtype, got {value.dtype}")
        return value.reshape(()).to(device=device, dtype=torch.int64)

    @staticmethod
    def _validate_inputs(x_t: Tensor, previous_chunk_leftover: Tensor) -> None:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must have shape (batch, horizon, action_dim), got {tuple(x_t.shape)}")
        if not x_t.is_floating_point():
            raise ValueError(f"x_t must be floating point, got {x_t.dtype}")
        if previous_chunk_leftover.ndim != 3:
            raise ValueError(
                "previous_chunk_leftover must have shape (batch, prefix, action_dim), "
                f"got {tuple(previous_chunk_leftover.shape)}"
            )
        if previous_chunk_leftover.shape[0] != x_t.shape[0]:
            raise ValueError("previous_chunk_leftover and x_t must have the same batch size")
        if previous_chunk_leftover.shape[2] != x_t.shape[2]:
            raise ValueError("previous_chunk_leftover and x_t must have the same action dimension")
        if previous_chunk_leftover.shape[1] > x_t.shape[1]:
            raise ValueError("previous_chunk_leftover cannot be longer than the action horizon")
