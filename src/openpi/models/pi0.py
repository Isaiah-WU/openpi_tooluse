import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, "*b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "*b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions.

    `pos` may carry any number of leading dimensions -- e.g. `[b]` for a single position per
    batch element, or `[b, ah]` for a distinct position (such as a per-action-step
    flow-matching timestep, used for training-time RTC) at every one of the `ah` positions in
    a chunk. The embedding dimension is always appended as the last axis.
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "...,j->...j",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Training-time RTC (see pi0_config.Pi0Config.train_time_rtc_max_delay).
        self.train_time_rtc_max_delay = config.train_time_rtc_max_delay

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"] | at.Float[at.Array, "b ah"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b ah emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # `timestep` is either one flow-matching progress value shared by the whole chunk
        # (shape `b`: every action step is denoised in lockstep -- the original behavior),
        # or one value per action step (shape `b ah`: used for training-time RTC, where
        # prefix steps are pinned to a different timestep than the postfix steps still being
        # denoised, see `compute_loss`/`sample_actions`). Normalize to `b ah` once up front
        # so the rest of this function -- and both branches below -- don't need to care
        # which case they're in.
        if timestep.ndim == 1:
            timestep = einops.repeat(timestep, "b -> b ah", ah=self.action_horizon)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS). `time_emb` is `[b, ah, emb]`; nnx.Linear/nnx.swish both
            # apply independently over leading dims, so per-step and shared-timestep inputs
            # go through the exact same computation here.
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            action_time_tokens = jnp.concatenate([action_tokens, time_emb], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001

        action_prefix_mask = jnp.zeros((*batch_shape, self.action_horizon), dtype=jnp.bool_)
        delay = None
        if self.train_time_rtc_max_delay:
            # Training-time RTC (arXiv 2512.05964): simulate inference delay by randomly
            # designating the first `delay` action steps of the chunk as an already-known
            # "prefix" -- the tail of the previous chunk that's still valid to execute while
            # a new chunk is generated. The model is fed the ground-truth prefix actions
            # directly (no noise mixed in) and is only scored on denoising the remaining
            # "postfix" steps. This teaches the model in-weights to condition on a committed
            # prefix, so at inference time we no longer need the external guidance/blend
            # trick in `sample_actions`'s `prefix_actions`/`blend_weight` path.
            delay = jax.random.randint(delay_rng, batch_shape, 0, self.train_time_rtc_max_delay)
            step_idx = jnp.arange(self.action_horizon)
            action_prefix_mask = step_idx < delay[..., None]  # [*b, ah]

        # openpi's flow convention is the reverse of the training-time RTC paper's: here t=0
        # is the clean target and t=1 is pure noise (sample_actions integrates 1 -> 0). So a
        # "known, already-clean" prefix step gets timestep 0 here, not 1 as in the paper.
        per_step_time = jnp.where(action_prefix_mask, 0.0, time[..., None])
        time_expanded = per_step_time[..., None]
        # Prefix steps see the ground-truth action directly; postfix steps use the usual
        # flow-matching interpolation between noise and the target action.
        x_t = jnp.where(
            action_prefix_mask[..., None], actions, time_expanded * noise + (1 - time_expanded) * actions
        )
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, per_step_time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        per_step_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if delay is not None:
            # `train.py` reduces this function's output with a plain, unweighted
            # `jnp.mean` over the full `[*b, ah]` array. Zeroing the prefix steps without
            # correction would silently shrink an example's contribution to that mean in
            # proportion to its own delay, biasing training towards small-delay examples in
            # a way that isn't controlled by (and would confound) `train_time_rtc_max_delay`.
            # Rescale so every example instead contributes exactly the mean of *its own*
            # postfix loss, regardless of how many steps its delay masked out.
            postfix_count = jnp.maximum(self.action_horizon - delay, 1)[..., None]
            per_step_loss = jnp.where(
                action_prefix_mask, 0.0, per_step_loss * self.action_horizon / postfix_count
            )
        return per_step_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        prefix_actions: at.Float[at.Array, "b c ad"] | None = None,
        blend_weight: at.Float[at.Array, "ah"] | None = None,
        action_prefix: at.Float[at.Array, "b c ad"] | None = None,
        delay: at.Int[at.Array, " b"] | int | None = None,
    ) -> _model.Actions:
        # `action_prefix`/`delay` is the training-time RTC (arXiv 2512.05964) path: cheap,
        # no guidance computation, but only works if this model's `compute_loss` was actually
        # run with `train_time_rtc_max_delay` set. It's a separate code path from the
        # `prefix_actions`/`blend_weight` inference-time RTC guidance below -- the two are
        # mutually exclusive and are not interchangeable inputs to the same sampler.
        if action_prefix is not None:
            if prefix_actions is not None:
                raise ValueError(
                    "Pass either `prefix_actions` (inference-time RTC guidance) or "
                    "`action_prefix` (training-time RTC), not both."
                )
            if delay is None:
                raise ValueError("`delay` is required when `action_prefix` is provided.")
            return self._sample_actions_train_time_rtc(
                rng, observation, num_steps=num_steps, noise=noise, action_prefix=action_prefix, delay=delay
            )

        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # RTC-style soft-masked inpainting. When `prefix_actions` (already-committed
        # actions, e.g. the not-yet-executed tail of the previous chunk) is provided, we
        # anchor those steps to the committed actions during denoising instead of letting
        # the model regenerate them freely. `blend_weight[i]` is 1 for strongly committed
        # positions, decays to 0 over a transition window, and is 0 beyond it. The
        # straight-line flow-matching velocity toward the committed actions is
        # `u_t = noise - actions`, which is constant in `t`, so we precompute it once and
        # reuse it every step. Without both arguments this is a no-op (identical sampler).
        if prefix_actions is not None:
            if blend_weight is None:
                raise ValueError("prefix_actions requires blend_weight to be provided.")
            blend_weight = jnp.asarray(blend_weight)
            if blend_weight.shape[-1] != self.action_horizon:
                raise ValueError(
                    f"blend_weight must have length action_horizon ({self.action_horizon}), got {blend_weight.shape}"
                )
            if prefix_actions.shape[-2] > self.action_horizon:
                raise ValueError(
                    f"prefix_actions has {prefix_actions.shape[-2]} steps but action_horizon is {self.action_horizon}"
                )
            if prefix_actions.shape[-2] < self.action_horizon:
                # Pad by repeating the last committed action. Positions beyond the committed
                # region still carry a non-zero blend_weight inside the decay window, and they
                # must blend toward a smooth continuation of the committed trajectory rather
                # than toward the zero vector (which would shrink the sampled actions to zero).
                pad = self.action_horizon - prefix_actions.shape[-2]
                last = prefix_actions[..., -1:, :]
                prefix_actions = jnp.concatenate(
                    [prefix_actions, jnp.repeat(last, pad, axis=-2)], axis=-2
                )
            v_known = noise - prefix_actions
            blend_weight = blend_weight[None, :, None]  # (1, ah, 1) for broadcasting
        else:
            v_known = None
            blend_weight = None

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            if v_known is not None:
                # RTC soft-masked inpainting: blend the model's predicted velocity toward
                # the committed-action velocity. Where blend_weight=1 the steps are fully
                # anchored to already-promised actions; beyond the decay window the model
                # is free to replan.
                v_t = blend_weight * v_known + (1.0 - blend_weight) * v_t

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _sample_actions_train_time_rtc(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""],
        noise: at.Float[at.Array, "b ah ad"] | None,
        action_prefix: at.Float[at.Array, "b c ad"],
        delay: at.Int[at.Array, " b"] | int,
    ) -> _model.Actions:
        """Training-time RTC inference (arXiv 2512.05964).

        Requires a model whose `compute_loss` was actually trained with
        `train_time_rtc_max_delay` set -- this function does not itself teach the model
        anything about prefixes, it just exercises the behavior training taught it. Unlike
        the `prefix_actions`/`blend_weight` guidance path in `sample_actions`, this needs no
        extra per-step computation: we pin the prefix positions (and their timestep) exactly
        as `compute_loss` did during training, and let one ordinary forward pass per
        denoising step do the rest.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        delay = jnp.broadcast_to(jnp.asarray(delay), (batch_size,))
        if action_prefix.shape[-2] > self.action_horizon:
            raise ValueError(
                f"action_prefix has {action_prefix.shape[-2]} steps but action_horizon is {self.action_horizon}"
            )
        if action_prefix.shape[-2] < self.action_horizon:
            # Pad by repeating the last committed action so every position has *some* value
            # to fall back on; the pad only matters where `action_prefix_mask` is also True,
            # i.e. within the requested `delay`, so real padding values are never used.
            pad = self.action_horizon - action_prefix.shape[-2]
            last = action_prefix[..., -1:, :]
            action_prefix = jnp.concatenate([action_prefix, jnp.repeat(last, pad, axis=-2)], axis=-2)

        step_idx = jnp.arange(self.action_horizon)
        action_prefix_mask = step_idx[None, :] < delay[:, None]  # [b, ah]

        # first fill KV cache with a forward pass of the (image/language) prefix -- unrelated
        # to `action_prefix` above; this is the same vision-language prefix/suffix split used
        # everywhere else in this model.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            # Pin the prefix positions to the known actions and mark them "already clean"
            # (t=0) on every step, exactly like `compute_loss` did during training; postfix
            # positions keep integrating normally.
            x_t = jnp.where(action_prefix_mask[..., None], action_prefix, x_t)
            per_step_time = jnp.where(action_prefix_mask, 0.0, time)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(per_step_time, (batch_size, self.action_horizon))
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            full_prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([full_prefix_attn_mask, suffix_attn_mask], axis=-1)
            suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        # The last Euler step runs after the final pin, and `v_t` at prefix positions was
        # never scored during training (its loss was masked out), so it isn't trained to be
        # small there -- pin one more time so the output honors the requested prefix exactly,
        # rather than only approximately.
        return jnp.where(action_prefix_mask[..., None], action_prefix, x_0)
