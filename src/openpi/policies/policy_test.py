from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import ur10e_policy
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


def test_make_rtc_blend_weight_separates_frozen_and_overlap_regions():
    weight = _policy._make_rtc_blend_weight(  # noqa: SLF001
        action_horizon=6,
        num_committed=2,
        prefix_attention_horizon=4,
    )

    np.testing.assert_allclose(weight, np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32))


def test_rtc_prefix_uses_training_action_transforms_before_sampling():
    """External UR10e actions must be normalized and padded before RTC sampling."""

    action_mean = np.arange(7, dtype=np.float32) + 10.0
    action_std = np.full(7, 2.0, dtype=np.float32)
    norm_stats = {
        "state": _normalize.NormStats(
            mean=np.zeros(7, dtype=np.float32),
            std=np.ones(7, dtype=np.float32),
        ),
        "actions": _normalize.NormStats(mean=action_mean, std=action_std),
    }

    # Build a lightweight Policy instance without loading or compiling a real model.
    policy = object.__new__(_policy.Policy)
    policy._model = SimpleNamespace(action_horizon=4, action_dim=32)  # noqa: SLF001
    policy._input_transform = transforms.compose(  # noqa: SLF001
        [
            ur10e_policy.UR10eInputs(model_type=_model.ModelType.PI05),
            transforms.Normalize(norm_stats),
            transforms.PadStatesAndActions(32),
        ]
    )
    policy._output_transform = transforms.compose([ur10e_policy.UR10eOutputs()])  # noqa: SLF001
    policy._sample_kwargs = {}  # noqa: SLF001
    policy._is_pytorch_model = False  # noqa: SLF001
    policy._rng = jax.random.key(0)  # noqa: SLF001

    captured = {}

    def sample_actions(_rng, _observation, **kwargs):
        captured.update(kwargs)
        return jnp.zeros((1, 4, 32), dtype=jnp.float32)

    policy._sample_actions = sample_actions  # noqa: SLF001

    external_prefix = np.stack([action_mean, action_mean + action_std, action_mean + 2 * action_std])
    observation = {
        "observation/state": np.zeros(7, dtype=np.float32),
        "observation/image": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
    }

    result = policy.infer(
        observation,
        prefix_actions=external_prefix,
        num_committed_actions=1,
        prefix_attention_horizon=3,
    )

    transformed_prefix = np.asarray(captured["prefix_actions"])
    assert transformed_prefix.shape == (1, 3, 32)
    np.testing.assert_allclose(
        transformed_prefix[0, :, :7],
        (external_prefix - action_mean) / (action_std + 1e-6),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(transformed_prefix[0, :, 7:], 0.0)
    np.testing.assert_allclose(
        np.asarray(captured["blend_weight"]),
        np.array([1.0, 0.5, 0.0, 0.0], dtype=np.float32),
    )
    assert result["actions"].shape == (4, 7)
    assert "actions" not in observation


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
