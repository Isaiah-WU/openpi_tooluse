from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _rtc_step_tensor(
    value: int,
    *,
    name: str,
    device: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> torch.Tensor:
    """Validate one RTC runtime scalar before it crosses the compile boundary."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}")
    return torch.tensor(value, dtype=torch.int64, device=device)


def _make_rtc_blend_weight(action_horizon: int, num_committed: int, prefix_attention_horizon: int) -> np.ndarray:
    """Builds the per-step blend weight used for RTC-style soft-masked inpainting.

    weight[i] = 1 for i < num_committed (strongly committed: the new chunk must match the
    old trajectory here), decays from just below 1 to 0 over [num_committed,
    prefix_attention_horizon), and is 0 beyond (fully free replanning). The first
    position of the decay window is *not* committed, so it starts strictly below 1;
    positions in the window blend toward a smooth continuation of the committed
    trajectory (the repeated last committed action).
    """
    if not 0 <= num_committed <= prefix_attention_horizon <= action_horizon:
        raise ValueError(
            f"expected 0 <= num_committed ({num_committed}) <= prefix_attention_horizon ({prefix_attention_horizon})"
            f" <= action_horizon ({action_horizon})"
        )
    weight = np.zeros(action_horizon, dtype=np.float32)
    if num_committed == 0:
        # Nothing is committed, so there is nothing to anchor to.
        return weight
    weight[:num_committed] = 1.0
    window_len = prefix_attention_horizon - num_committed
    if window_len > 0:
        weight[num_committed:prefix_attention_horizon] = np.linspace(1.0, 0.0, window_len + 1)[1:]
    return weight


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        prefix_actions: np.ndarray | None = None,
        prefix_attention_horizon: int | None = None,
        prev_chunk_left_over: np.ndarray | None = None,
        prev_chunk_valid_steps: int | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
    ) -> dict:  # type: ignore[misc]
        if prefix_actions is not None and prev_chunk_left_over is not None:
            raise ValueError("prefix_actions and prev_chunk_left_over cannot be used together")

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        if prev_chunk_left_over is not None:
            # Client actions are in the robot/output space. Put them through the
            # same action transforms as training so RTC operates in normalized,
            # model-padded action space rather than raw UR10e units.
            if "actions" in inputs:
                raise ValueError("observation already contains actions; cannot inject prev_chunk_left_over")
            inputs["actions"] = np.asarray(prev_chunk_left_over)

        inputs = self._input_transform(inputs)
        if prev_chunk_left_over is not None:
            if "actions" not in inputs:
                raise ValueError("input transforms removed prev_chunk_left_over actions")
            prev_chunk_left_over = np.asarray(inputs.pop("actions"))

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        if prefix_actions is not None:
            if prefix_attention_horizon is None:
                raise ValueError("prefix_attention_horizon is required when prefix_actions is provided.")
            prefix_actions = np.asarray(prefix_actions)
            if prefix_actions.ndim == 2:
                num_committed = prefix_actions.shape[0]
            elif prefix_actions.ndim == 3:
                num_committed = prefix_actions.shape[1]
            else:
                raise ValueError(
                    f"prefix_actions must have shape (c, action_dim) or (1, c, action_dim), got {prefix_actions.shape}"
                )
            blend_weight = _make_rtc_blend_weight(
                self._model.action_horizon, num_committed, prefix_attention_horizon
            )
            if self._is_pytorch_model:
                # NOTE: pi0_pytorch does not yet implement the RTC blend args; passing them
                # to a PyTorch model will raise inside sample_actions until ported.
                prefix_actions = torch.from_numpy(prefix_actions).to(self._pytorch_device)[None, ...]
                blend_weight = torch.from_numpy(blend_weight).to(self._pytorch_device)
            else:
                prefix_actions = jnp.asarray(prefix_actions)[None, ...]
                blend_weight = jnp.asarray(blend_weight)
            sample_kwargs["prefix_actions"] = prefix_actions
            sample_kwargs["blend_weight"] = blend_weight

        if prev_chunk_left_over is not None:
            if not self._is_pytorch_model:
                raise ValueError("prev_chunk_left_over VJP guidance requires a PyTorch policy")
            if inference_delay is None:
                raise ValueError("prev_chunk_left_over requires inference_delay")
            if prev_chunk_left_over.ndim == 2:
                prev_chunk_left_over = prev_chunk_left_over[None, ...]
            elif prev_chunk_left_over.ndim != 3 or prev_chunk_left_over.shape[0] != 1:
                raise ValueError(
                    "prev_chunk_left_over must have shape (steps, action_dim) or (1, steps, action_dim), "
                    f"got {prev_chunk_left_over.shape}"
                )

            prefix_length = prev_chunk_left_over.shape[1]
            inference_delay_tensor = _rtc_step_tensor(
                inference_delay,
                name="inference_delay",
                device=self._pytorch_device,
                minimum=0,
            )
            if prev_chunk_valid_steps is not None:
                valid_steps_tensor = _rtc_step_tensor(
                    prev_chunk_valid_steps,
                    name="prev_chunk_valid_steps",
                    device=self._pytorch_device,
                    minimum=0,
                    maximum=prefix_length,
                )
            else:
                valid_steps_tensor = None
            if execution_horizon is not None:
                execution_horizon_tensor = _rtc_step_tensor(
                    execution_horizon,
                    name="execution_horizon",
                    device=self._pytorch_device,
                    minimum=1,
                )
            else:
                execution_horizon_tensor = None

            sample_kwargs["prev_chunk_left_over"] = torch.from_numpy(
                np.ascontiguousarray(prev_chunk_left_over)
            ).to(self._pytorch_device)
            if valid_steps_tensor is not None:
                sample_kwargs["prev_chunk_valid_steps"] = valid_steps_tensor
            sample_kwargs["inference_delay"] = inference_delay_tensor
            if execution_horizon_tensor is not None:
                sample_kwargs["execution_horizon"] = execution_horizon_tensor

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
