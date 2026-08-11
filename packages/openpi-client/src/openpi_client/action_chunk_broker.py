import logging
import threading
from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy

logger = logging.getLogger(__name__)


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0


class PipelinedActionChunkBroker(_base_policy.BasePolicy):
    """Like ActionChunkBroker, but hides inference/network latency behind chunk execution.

    ActionChunkBroker only calls the wrapped policy once the current chunk is fully
    exhausted, which means the caller blocks on a full inference round trip right when
    the chunk runs out -- on a real robot this shows up as the arm visibly pausing every
    `action_horizon` steps. This broker instead fires the next `infer()` call in a
    background thread once execution crosses `replan_trigger_step` (still inside the
    current chunk), so the round trip overlaps with executing the current chunk's tail.
    By the time the current chunk is exhausted, the next one is usually already ready.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        replan_trigger_step: int | None = None,
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        # Default: start prefetching the next chunk once we're halfway through this one.
        self._replan_trigger_step = replan_trigger_step if replan_trigger_step is not None else action_horizon // 2
        if not 0 <= self._replan_trigger_step < action_horizon:
            raise ValueError(
                f"replan_trigger_step ({self._replan_trigger_step}) must be in [0, action_horizon={action_horizon})"
            )

        self._cur_step: int = 0
        self._last_results: Dict[str, np.ndarray] | None = None

        self._lock = threading.Lock()
        self._next_results: Dict[str, np.ndarray] | None = None
        self._pending_thread: threading.Thread | None = None

    def _start_prefetch(self, obs: Dict) -> None:
        def worker():
            try:
                results = self._policy.infer(obs)
            except Exception:
                logger.exception("Prefetch inference call failed; will fall back to a blocking call.")
                return
            with self._lock:
                self._next_results = results

        self._pending_thread = threading.Thread(target=worker, daemon=True)
        self._pending_thread.start()

    def _await_prefetch(self, obs: Dict) -> None:
        """Blocks until the next chunk is available, starting a fresh call if none was in flight."""
        if self._pending_thread is not None:
            self._pending_thread.join()
            self._pending_thread = None
        with self._lock:
            self._last_results = self._next_results
            self._next_results = None
        self._cur_step = 0
        if self._last_results is None:
            # Prefetch either was never triggered (e.g. reset() cut the chunk short) or
            # failed -- fall back to a plain blocking call so we always make progress.
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        if self._cur_step == self._replan_trigger_step and self._pending_thread is None:
            self._start_prefetch(obs)

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._await_prefetch(obs)

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        if self._pending_thread is not None:
            self._pending_thread.join()
            self._pending_thread = None
        self._last_results = None
        self._next_results = None
        self._cur_step = 0


class RtcActionChunkBroker(PipelinedActionChunkBroker):
    """Pipelined chunk broker that smooths chunk switches via RTC-style inpainting.

    On top of PipelinedActionChunkBroker (which prefetches the next chunk in a background
    thread), this broker also hands the not-yet-executed tail of the current chunk to the
    policy as `prefix_actions` when prefetching. The policy anchors the head of the new
    chunk to that committed tail with a blend weight that is 1 over the committed region
    and decays to 0 over `prefix_attention_horizon`, so the switch is a smooth
    continuation instead of a hard jump. This is the real-time chunking (RTC) idea from
    the pi0.5 line of work, applied to the flow-matching sampler.

    Note: `prefix_attention_horizon` must land inside the wrapped policy's contract (it
    is forwarded to Policy.infer); it defaults to the full action horizon.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        replan_trigger_step: int | None = None,
        prefix_attention_horizon: int | None = None,
    ):
        super().__init__(policy, action_horizon, replan_trigger_step)
        # Where the blend weight decays to zero (fully free replanning). The strongly
        # committed length is implicit: how many steps of the current chunk remain at
        # prefetch time (action_horizon - replan_trigger_step).
        self._prefix_attention_horizon = (
            prefix_attention_horizon if prefix_attention_horizon is not None else action_horizon
        )
        # Last action of the previous chunk, used to log the joint-space L2 jump at each
        # switch (an objective proxy for how smooth the chunk transitions are).
        self._prev_last_action: np.ndarray | None = None

    def _start_prefetch(self, obs: Dict) -> None:
        # Snapshot the committed tail of the current chunk on the main thread before the
        # worker starts. This is read-only w.r.t. the main thread (which only slices the
        # same array), so there is no write race.
        committed_actions = self._last_results["actions"][self._replan_trigger_step :, ...]
        self._prev_last_action = self._last_results["actions"][self._action_horizon - 1, ...]

        def worker():
            try:
                results = self._policy.infer(
                    obs,
                    prefix_actions=committed_actions,
                    prefix_attention_horizon=self._prefix_attention_horizon,
                )
            except Exception:
                logger.exception("RTC prefetch inference call failed; will fall back to a blocking call.")
                return
            with self._lock:
                self._next_results = results

        self._pending_thread = threading.Thread(target=worker, daemon=True)
        self._pending_thread.start()

    def _await_prefetch(self, obs: Dict) -> None:
        if self._pending_thread is not None:
            self._pending_thread.join()
            self._pending_thread = None
        with self._lock:
            new_results = self._next_results
            self._next_results = None
        self._cur_step = 0
        if new_results is None:
            # Prefetch never ran or failed -- fall back to a plain blocking call. No RTC
            # anchor, so this switch is a plain hard switch.
            new_results = self._policy.infer(obs)
            self._cur_step = 0
        if self._prev_last_action is not None:
            first_new = new_results["actions"][0, ...]
            joint_l2 = float(
                np.linalg.norm(np.asarray(self._prev_last_action) - np.asarray(first_new))
            )
            logger.info(f"chunk switch joint L2 jump = {joint_l2:.4f}")
        self._last_results = new_results

    @override
    def reset(self) -> None:
        super().reset()
        self._prev_last_action = None
