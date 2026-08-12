import logging
import threading
from collections import deque
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


class RtcActionChunkBroker(_base_policy.BasePolicy):
    """Runs action-chunk inference asynchronously with RTC-style trajectory guidance.

    ``prediction_horizon`` is the number of actions returned by the model, while
    ``execution_horizon`` is the minimum number of controller steps between replans.
    Keeping them separate is important: a model may predict 50 actions while the robot
    starts replanning every 10 controller steps.

    A new chunk is aligned to the old chunk at the action that is current when prefetch
    starts. While inference runs, the broker continues consuming the old chunk. If
    inference takes ``d`` controller steps, the broker starts consuming the new chunk at
    index ``d``. Thus, the prefix used to guide inference is never replayed on the robot.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        prediction_horizon: int,
        execution_horizon: int,
        *,
        prefix_attention_horizon: int | None = None,
        initial_delay_steps: int = 1,
        delay_history_size: int = 5,
    ):
        if prediction_horizon <= 1:
            raise ValueError(f"prediction_horizon must be greater than 1, got {prediction_horizon}")
        if not 1 <= execution_horizon < prediction_horizon:
            raise ValueError(
                f"execution_horizon ({execution_horizon}) must be in "
                f"[1, prediction_horizon={prediction_horizon})"
            )
        if prefix_attention_horizon is not None and not 1 <= prefix_attention_horizon <= prediction_horizon:
            raise ValueError(
                f"prefix_attention_horizon ({prefix_attention_horizon}) must be in "
                f"[1, prediction_horizon={prediction_horizon}]"
            )
        if not 1 <= initial_delay_steps < prediction_horizon:
            raise ValueError(
                f"initial_delay_steps ({initial_delay_steps}) must be in "
                f"[1, prediction_horizon={prediction_horizon})"
            )
        if delay_history_size <= 0:
            raise ValueError(f"delay_history_size must be positive, got {delay_history_size}")

        self._policy = policy
        self._prediction_horizon = prediction_horizon
        self._execution_horizon = execution_horizon
        self._prefix_attention_horizon = prefix_attention_horizon
        self._delay_steps = deque([initial_delay_steps], maxlen=delay_history_size)

        self._current_results: Dict[str, np.ndarray] | None = None
        self._current_index = 0
        self._steps_since_switch = 0
        self._last_emitted_action: np.ndarray | None = None

        self._lock = threading.Lock()
        self._pending_thread: threading.Thread | None = None
        self._next_results: Dict[str, np.ndarray] | None = None
        self._prefetch_start_index: int | None = None
        self._prefetch_failed = False

    def _validate_chunk(self, results: Dict, *, source: str) -> None:
        if "actions" not in results:
            raise ValueError(f"{source} policy result does not contain an 'actions' field")
        actions = np.asarray(results["actions"])
        if actions.ndim < 2:
            raise ValueError(f"{source} actions must have shape (horizon, action_dim), got {actions.shape}")
        if actions.shape[0] != self._prediction_horizon:
            raise ValueError(
                f"{source} policy returned {actions.shape[0]} actions, but prediction_horizon is "
                f"{self._prediction_horizon}"
            )

    def _load_initial_chunk(self, obs: Dict) -> None:
        results = self._policy.infer(obs)
        self._validate_chunk(results, source="initial")
        self._current_results = results
        self._current_index = 0
        self._steps_since_switch = 0

    def _start_prefetch(self, obs: Dict) -> None:
        assert self._current_results is not None
        assert self._pending_thread is None

        self._prefetch_start_index = self._current_index
        prefix_actions = np.asarray(self._current_results["actions"])[self._prefetch_start_index :].copy()
        overlap_horizon = prefix_actions.shape[0]
        if overlap_horizon == 0:
            return

        attention_horizon = overlap_horizon
        if self._prefix_attention_horizon is not None:
            attention_horizon = min(attention_horizon, self._prefix_attention_horizon)
        estimated_delay_steps = min(max(self._delay_steps), attention_horizon)
        self._prefetch_failed = False

        def worker():
            try:
                results = self._policy.infer(
                    obs,
                    prefix_actions=prefix_actions,
                    num_committed_actions=estimated_delay_steps,
                    prefix_attention_horizon=attention_horizon,
                )
                self._validate_chunk(results, source="RTC prefetch")
            except Exception:
                logger.exception("RTC prefetch inference call failed; will fall back to a blocking call.")
                with self._lock:
                    self._prefetch_failed = True
                return
            with self._lock:
                self._next_results = results

        self._pending_thread = threading.Thread(target=worker, daemon=True)
        self._pending_thread.start()
        logger.info(
            "RTC prefetch started at chunk index %d (overlap=%d, estimated_delay=%d, attention_horizon=%d)",
            self._prefetch_start_index,
            overlap_horizon,
            estimated_delay_steps,
            attention_horizon,
        )

    def _take_ready_results(self) -> Dict[str, np.ndarray] | None:
        with self._lock:
            if self._next_results is None:
                return None
            results = self._next_results
            self._next_results = None
        return results

    def _switch_to_prefetched_chunk(self, new_results: Dict[str, np.ndarray]) -> None:
        assert self._prefetch_start_index is not None
        actual_delay_steps = self._current_index - self._prefetch_start_index
        if not 0 <= actual_delay_steps < self._prediction_horizon:
            raise RuntimeError(
                f"RTC time alignment produced invalid delay {actual_delay_steps}; "
                f"current_index={self._current_index}, prefetch_start_index={self._prefetch_start_index}"
            )

        next_action = np.asarray(new_results["actions"])[actual_delay_steps]
        if self._last_emitted_action is not None:
            joint_l2 = float(np.linalg.norm(self._last_emitted_action - next_action))
            logger.info(
                "RTC chunk switch: actual_delay_steps=%d, new_chunk_index=%d, joint L2 jump=%.4f",
                actual_delay_steps,
                actual_delay_steps,
                joint_l2,
            )

        self._delay_steps.append(max(1, actual_delay_steps))
        self._current_results = new_results
        # Indices [0, actual_delay_steps) correspond to actions that were consumed from
        # the old chunk while inference was running. Skip them instead of replaying them.
        self._current_index = actual_delay_steps
        self._steps_since_switch = 0
        self._pending_thread = None
        self._prefetch_start_index = None
        self._prefetch_failed = False

    def _finish_or_fallback(self, obs: Dict) -> None:
        if self._pending_thread is not None:
            self._pending_thread.join()
        new_results = self._take_ready_results()
        if new_results is not None:
            self._switch_to_prefetched_chunk(new_results)
            return

        # The current chunk is exhausted and RTC prefetch failed (or was never started).
        # Make a normal blocking request so control can continue without replaying stale
        # actions. This is deliberately visible in logs because this transition is not RTC.
        logger.warning("RTC chunk unavailable at exhaustion; using a blocking non-RTC inference call.")
        self._pending_thread = None
        self._prefetch_start_index = None
        self._prefetch_failed = False
        self._load_initial_chunk(obs)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._current_results is None:
            self._load_initial_chunk(obs)

        ready_results = self._take_ready_results()
        if ready_results is not None:
            self._switch_to_prefetched_chunk(ready_results)

        if self._current_index >= self._prediction_horizon:
            self._finish_or_fallback(obs)

        if (
            self._pending_thread is not None
            and not self._pending_thread.is_alive()
            and self._next_results is None
            and self._prefetch_failed
        ):
            # Preserve the current chunk until it is exhausted. A blocking fallback at
            # exhaustion is safer than repeatedly launching failed background requests.
            self._pending_thread = None

        if (
            self._pending_thread is None
            and not self._prefetch_failed
            and self._steps_since_switch >= self._execution_horizon
            and self._current_index < self._prediction_horizon
        ):
            self._start_prefetch(obs)

        assert self._current_results is not None
        index = self._current_index

        def slicer(value):
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == self._prediction_horizon:
                return value[index, ...]
            return value

        results = tree.map_structure(slicer, self._current_results)
        self._last_emitted_action = np.asarray(self._current_results["actions"])[index].copy()
        self._current_index += 1
        self._steps_since_switch += 1
        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        if self._pending_thread is not None:
            self._pending_thread.join()
        with self._lock:
            self._next_results = None
        self._current_results = None
        self._current_index = 0
        self._steps_since_switch = 0
        self._last_emitted_action = None
        self._pending_thread = None
        self._prefetch_start_index = None
        self._prefetch_failed = False
