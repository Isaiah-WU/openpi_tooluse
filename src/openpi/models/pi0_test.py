import flax.nnx as nnx
import jax

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def _norm_modulation_paths(state: nnx.State) -> set[tuple]:
    return {path for path in state if any("pre_attention_norm_1" in p or "pre_ffw_norm_1" in p for p in path)}


def test_pi0_train_time_rtc_lora_freezes_adarms_by_default_when_disabled():
    """Sanity check: without opting in, a LoRA + training-time RTC config freezes the action
    expert's adaRMS modulation exactly like plain LoRA does (i.e. this option actually does
    nothing unless `train_time_rtc_max_delay` is set)."""
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        train_time_rtc_max_delay=None,
        train_time_rtc_unfreeze_adarms=True,
    )
    state = _get_frozen_state(config)
    modulation_paths = _norm_modulation_paths(state)
    assert len(modulation_paths) > 0, (
        "expected the action expert's adaRMS modulation Dense to show up as frozen under plain "
        "LoRA (train_time_rtc_max_delay=None) -- if this fails, the 'pre_attention_norm_1'/"
        "'pre_ffw_norm_1' path assumption in get_freeze_filter is wrong and needs updating."
    )


def test_pi0_train_time_rtc_lora_keeps_adarms_trainable():
    """With training-time RTC enabled (and the unfreeze opted into, the default), the action
    expert's adaRMS modulation Dense must NOT be in the frozen set -- otherwise the one layer
    that turns a per-token timestep into per-token behavior can never learn to treat a pinned
    prefix step differently from a step actively being denoised (see
    Pi0Config.train_time_rtc_unfreeze_adarms)."""
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        train_time_rtc_max_delay=8,
        train_time_rtc_unfreeze_adarms=True,
    )
    state = _get_frozen_state(config)
    modulation_paths = _norm_modulation_paths(state)
    assert len(modulation_paths) == 0, (
        f"expected no frozen params under pre_attention_norm_1/pre_ffw_norm_1, found: {modulation_paths}"
    )
    # Everything else about the freeze set should be unchanged from plain LoRA.
    baseline = _get_frozen_state(
        _pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    )
    baseline_modulation_paths = _norm_modulation_paths(baseline)
    assert set(baseline) - baseline_modulation_paths == set(state)


def test_pi0_train_time_rtc_unfreeze_adarms_opt_out():
    """`train_time_rtc_unfreeze_adarms=False` should reproduce the plain-LoRA frozen set even
    with training-time RTC enabled, for A/B comparisons against the unfrozen default."""
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        train_time_rtc_max_delay=8,
        train_time_rtc_unfreeze_adarms=False,
    )
    state = _get_frozen_state(config)
    baseline = _get_frozen_state(
        _pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    )
    assert set(state) == set(baseline)
