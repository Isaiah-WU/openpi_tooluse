import threading

import numpy as np
import pytest

from openpi_client import action_chunk_broker


def _chunk(start: int, horizon: int = 6) -> dict:
    actions = np.arange(start, start + horizon, dtype=np.float32)[:, None]
    return {"actions": actions, "metadata": "kept"}


class _DelayedPolicy:
    def __init__(self):
        self.calls = 0
        self.prefetch_started = threading.Event()
        self.release_prefetch = threading.Event()
        self.prefetch_kwargs = None

    def infer(self, obs, **kwargs):
        del obs
        call = self.calls
        self.calls += 1
        if call == 0:
            return _chunk(0)
        self.prefetch_kwargs = kwargs
        self.prefetch_started.set()
        if not self.release_prefetch.wait(timeout=2):
            raise TimeoutError("test did not release RTC prefetch")
        return _chunk(100)

    def reset(self):
        pass


def test_rtc_broker_skips_actions_consumed_during_prefetch():
    policy = _DelayedPolicy()
    broker = action_chunk_broker.RtcActionChunkBroker(
        policy,
        prediction_horizon=6,
        execution_horizon=2,
        initial_delay_steps=1,
    )

    assert broker.infer({})["actions"].item() == 0
    assert broker.infer({})["actions"].item() == 1

    # Replanning begins at old-chunk index 2. Two old actions are consumed while
    # inference is blocked, so the new chunk must start at index 2 (value 102), not 0.
    assert broker.infer({})["actions"].item() == 2
    assert policy.prefetch_started.wait(timeout=2)
    assert broker.infer({})["actions"].item() == 3

    policy.release_prefetch.set()
    broker._pending_thread.join(timeout=2)  # noqa: SLF001
    assert not broker._pending_thread.is_alive()  # noqa: SLF001
    assert broker.infer({})["actions"].item() == 102

    np.testing.assert_array_equal(
        policy.prefetch_kwargs["prefix_actions"],
        np.array([[2], [3], [4], [5]], dtype=np.float32),
    )
    assert policy.prefetch_kwargs["num_committed_actions"] == 1
    assert policy.prefetch_kwargs["prefix_attention_horizon"] == 4
    broker.reset()


def test_rtc_broker_validates_prediction_horizon():
    class ShortPolicy:
        def infer(self, obs, **kwargs):
            del obs, kwargs
            return _chunk(0, horizon=5)

        def reset(self):
            pass

    broker = action_chunk_broker.RtcActionChunkBroker(
        ShortPolicy(),
        prediction_horizon=6,
        execution_horizon=2,
    )

    with pytest.raises(ValueError, match="returned 5 actions"):
        broker.infer({})
