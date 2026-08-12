import numpy as np

from openpi.policies import ur10e_policy


def test_rtc_warmup_observation_matches_raw_ur10e_api():
    observation = ur10e_policy.make_ur10e_rtc_warmup_observation()

    assert set(observation) == {
        "observation/state",
        "observation/image",
        "observation/wrist_image",
        "prompt",
    }
    assert observation["observation/state"].shape == (7,)
    assert observation["observation/state"].dtype == np.float32
    for key in ("observation/image", "observation/wrist_image"):
        assert observation[key].shape == (224, 224, 3)
        assert observation[key].dtype == np.uint8
    assert observation["prompt"] == ur10e_policy.LONG_HORIZON_PROMPT
