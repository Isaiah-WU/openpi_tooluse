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
        num_committed_actions: int | None = None,
        prefix_attention_horizon: int | None = None,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        # RTC prefixes cross the websocket boundary in the policy's external action
        # space (for UR10e: absolute, unnormalized 7-D actions).  Feed them through
        # the exact same input action transforms used during training before passing
        # them to the model (for Pi0.5 UR10e: normalized and padded to 32-D).
        # Sending the external prefix directly to sample_actions would both use the
        # wrong units and fail when the external and model action dimensions differ.
        model_prefix_actions: np.ndarray | None = None
        prefix_length: int | None = None
        num_committed: int | None = None
        if prefix_actions is not None:
            if self._is_pytorch_model:
                raise NotImplementedError("RTC prefix actions are not implemented for PyTorch policies")
            if prefix_attention_horizon is None:
                raise ValueError("prefix_attention_horizon is required when prefix_actions is provided.")

            external_prefix_actions = np.asarray(prefix_actions)
            if external_prefix_actions.ndim == 2:
                prefix_length = external_prefix_actions.shape[0]
            elif external_prefix_actions.ndim == 3:
                if external_prefix_actions.shape[0] != 1:
                    raise ValueError(
                        "batched prefix_actions must have a singleton batch dimension, "
                        f"got {external_prefix_actions.shape}"
                    )
                external_prefix_actions = external_prefix_actions[0]
                prefix_length = external_prefix_actions.shape[0]
            else:
                raise ValueError(
                    "prefix_actions must have shape (c, action_dim) or "
                    f"(1, c, action_dim), got {external_prefix_actions.shape}"
                )
            if prefix_length == 0:
                raise ValueError("prefix_actions must contain at least one action")
            if prefix_length > self._model.action_horizon:
                raise ValueError(
                    f"prefix_actions has {prefix_length} steps but model action_horizon is {self._model.action_horizon}"
                )
            if not np.isfinite(external_prefix_actions).all():
                raise ValueError("prefix_actions contains NaN or infinity")

            num_committed = prefix_length if num_committed_actions is None else num_committed_actions
            if not 0 <= num_committed <= prefix_length:
                raise ValueError(
                    f"num_committed_actions ({num_committed}) must be in [0, prefix length={prefix_length}]"
                )
            if not 0 <= prefix_attention_horizon <= prefix_length:
                raise ValueError(
                    f"prefix_attention_horizon ({prefix_attention_horizon}) must be in "
                    f"[0, prefix length={prefix_length}]"
                )

            # Copy so in-place action transforms such as DeltaActions cannot mutate
            # a client-owned array or the broker's cached trajectory.
            inputs["actions"] = np.array(external_prefix_actions, copy=True)

        inputs = self._input_transform(inputs)
        if prefix_actions is not None:
            if "actions" not in inputs:
                raise ValueError(
                    "RTC prefix_actions were removed by the policy input transforms; "
                    "the policy cannot apply RTC guidance safely"
                )
            model_prefix_actions = np.asarray(inputs.pop("actions"))
            if model_prefix_actions.ndim != 2:
                raise ValueError(
                    "transformed RTC prefix_actions must have shape (c, model_action_dim), "
                    f"got {model_prefix_actions.shape}"
                )
            assert prefix_length is not None
            if model_prefix_actions.shape[0] != prefix_length:
                raise ValueError(
                    "policy input transforms changed the RTC prefix time dimension from "
                    f"{prefix_length} to {model_prefix_actions.shape[0]}; this transform "
                    "requires an explicit RTC horizon mapping"
                )
            if model_prefix_actions.shape[1] != self._model.action_dim:
                raise ValueError(
                    "transformed RTC prefix action dimension does not match the model: "
                    f"got {model_prefix_actions.shape[1]}, expected {self._model.action_dim}"
                )
            if not np.isfinite(model_prefix_actions).all():
                raise ValueError("transformed RTC prefix_actions contains NaN or infinity")

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

        if model_prefix_actions is not None:
            assert prefix_attention_horizon is not None
            assert num_committed is not None
            blend_weight = _make_rtc_blend_weight(self._model.action_horizon, num_committed, prefix_attention_horizon)
            model_prefix_actions = jnp.asarray(model_prefix_actions)[None, ...]
            blend_weight = jnp.asarray(blend_weight)
            sample_kwargs["prefix_actions"] = model_prefix_actions
            sample_kwargs["blend_weight"] = blend_weight

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
