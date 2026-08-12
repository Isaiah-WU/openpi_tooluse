from flax import nnx
import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_train_time_rtc_delay_zero_matches_baseline():
    """`train_time_rtc_max_delay=1` forces `delay=0` for every example (jax.random.randint's
    upper bound is exclusive), which should make `compute_loss` numerically identical to the
    original, non-RTC behavior: no prefix, every step denoised the same as before. This is the
    regression guard for the architecture change (per-token timestep plumbing) -- if this
    fails, the refactor changed behavior for models that aren't even using training-time RTC.
    """
    key = jax.random.key(0)
    baseline_model = pi0_config.Pi0Config(pi05=True).create(key)
    rtc_model = pi0_config.Pi0Config(pi05=True, train_time_rtc_max_delay=1).create(key)

    batch_size = 2
    config = pi0_config.Pi0Config(pi05=True)
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    loss_key = jax.random.key(1)

    baseline_loss = nnx_utils.module_jit(baseline_model.compute_loss)(loss_key, obs, act)
    rtc_loss = nnx_utils.module_jit(rtc_model.compute_loss)(loss_key, obs, act)
    assert baseline_loss.shape == rtc_loss.shape == (batch_size, config.action_horizon)
    np.testing.assert_allclose(rtc_loss, baseline_loss, rtol=1e-4, atol=1e-5)


def test_pi0_train_time_rtc_sample_actions_pins_prefix_exactly():
    """The `action_prefix`/`delay` sampling path must return the requested prefix steps
    exactly (not just approximately) -- gello's async broker splices chunks at this boundary,
    so any drift there would show up as a discontinuity."""
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(pi05=True, train_time_rtc_max_delay=6)
    model = config.create(key)

    batch_size = 2
    obs = config.fake_obs(batch_size)
    delay = 3
    action_prefix = jax.random.normal(jax.random.key(2), (batch_size, delay, config.action_dim))

    actions = nnx_utils.module_jit(model.sample_actions)(
        key, obs, num_steps=10, action_prefix=action_prefix, delay=delay
    )
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
    np.testing.assert_array_equal(actions[:, :delay], action_prefix)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
