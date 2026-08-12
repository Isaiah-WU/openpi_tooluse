import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    # Training-time RTC (arXiv 2512.05964, "Training-Time Action Conditioning for Efficient
    # Real-Time Chunking"). If set, `compute_loss` simulates an inference delay by sampling an
    # integer `delay` in [0, train_time_rtc_max_delay) per training example and conditioning
    # the model on that many ground-truth "prefix" action steps (ground-truth actions, no
    # noise, timestep pinned to 0) instead of denoising them; the loss is only scored on the
    # remaining "postfix" steps. This is a drop-in alternative to the external
    # guidance/blend-weight inference-time RTC trick already implemented in `sample_actions`
    # (the `prefix_actions`/`blend_weight` args): once the model is trained this way,
    # `sample_actions`'s `action_prefix`/`delay` path can condition on a prefix with a single
    # ordinary forward pass per denoising step, no extra guidance computation needed.
    # Leave as None (the default) to keep the original single-timestep-per-chunk behavior.
    train_time_rtc_max_delay: int | None = None
    # Only relevant when `train_time_rtc_max_delay` is set *and* LoRA is enabled (see
    # `get_freeze_filter`). The per-token timestep is injected via the adaRMS modulation
    # `nn.Dense` inside every transformer block's RMSNorm (see `gemma.RMSNorm`); that layer is
    # not a LoRA param, so under a pure-LoRA freeze filter it stays frozen at its pretrained,
    # single-timestep-per-chunk values and can never learn to treat a pinned prefix step
    # differently from a step actively being denoised. When True (the default), also keep the
    # action expert's adaRMS modulation params trainable so a LoRA retrofit of an existing
    # checkpoint can actually learn this. The parameter count is tiny relative to full LoRA
    # fine-tuning, so leaving this on is low-cost; set it to False to A/B against leaving that
    # layer frozen. Verify the path regex actually matches before relying on it --
    # see test_pi0_train_time_rtc_lora_keeps_adarms_trainable in pi0_test.py.
    train_time_rtc_unfreeze_adarms: bool = True

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.train_time_rtc_max_delay is not None and not 0 < self.train_time_rtc_max_delay <= self.action_horizon:
            raise ValueError(
                f"train_time_rtc_max_delay ({self.train_time_rtc_max_delay}) must be in "
                f"(0, action_horizon={self.action_horizon}]"
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params. `PathRegex` takes a single
            # pattern, so we use one regex with an alternation instead of composing
            # multiple filter objects (keeps this independent of whichever boolean
            # filter-combinators this nnx version happens to export).
            keep_trainable_pattern = ".*lora.*"
            if self.train_time_rtc_max_delay is not None and self.train_time_rtc_unfreeze_adarms:
                # Also exclude the action expert's adaRMS modulation Dense (see
                # `train_time_rtc_unfreeze_adarms` docstring above). `_1` is this
                # codebase's suffix for the action (second) expert -- see `gemma._name`
                # and `action_expert_params_filter` above, whose "_1" convention this
                # mirrors deliberately so only the action expert's norms are affected,
                # not PaliGemma's.
                keep_trainable_pattern += "|.*(pre_attention_norm_1|pre_ffw_norm_1).*"
            filters.append(
                nnx.Not(nnx_utils.PathRegex(keep_trainable_pattern)),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
