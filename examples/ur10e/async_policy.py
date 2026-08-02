"""Asynchronous policy inference for the UR10e client."""

from __future__ import annotations
from runtime_timing import RequestTiming
from typing import Any

import time
import numpy as np
import queue
import traceback
import multiprocessing as mp




def _fake_infer(
    observation: dict[str, Any],
    *,
    delay_ms: float,
    action_horizon: int,
    action_dim: int,
) -> dict[str, Any]:
    """Return a hold-position action chunk after a simulated delay."""
    if delay_ms < 0:
        raise ValueError("delay_ms must be non-negative")

    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")

    if action_dim <= 0:
        raise ValueError("action_dim must be positive")

    time.sleep(delay_ms / 1000.0)

    state = np.asarray(
        observation.get(
            "observation/state",
            np.zeros(action_dim, dtype=np.float32),
        ),
        dtype=np.float32,
    )

    hold_action = np.zeros(
        action_dim,
        dtype=np.float32,
    )

    copy_dim = min(len(state), action_dim)
    hold_action[:copy_dim] = state[:copy_dim]

    actions = np.repeat(
        hold_action[None, :],
        action_horizon,
        axis=0,
    )

    return {
        "actions": actions,
        "fake_inference": True,
    }

def _policy_worker(
    request_queue,
    response_queue,
    stop_event,
    remote_host: str,
    remote_port: int,
    fake_delay_ms: float | None,
    action_horizon: int,
    action_dim: int,
) -> None:
    """Run blocking policy inference outside the robot control process."""
    policy_client = None

    try:
        if fake_delay_ms is None:
            from openpi_client import websocket_client_policy

            policy_client = (
                websocket_client_policy.WebsocketClientPolicy(
                    host=remote_host,
                    port=remote_port,
                )
            )

        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if request is None:
                break

            request_id = request["request_id"]
            observation = request["observation"]
            timing: RequestTiming = request["timing"]

            try:
                timing.worker_infer_start = time.perf_counter()

                if fake_delay_ms is not None:
                    result = _fake_infer(
                        observation,
                        delay_ms=fake_delay_ms,
                        action_horizon=action_horizon,
                        action_dim=action_dim,
                    )
                else:
                    result = policy_client.infer(observation)

                timing.worker_infer_end = time.perf_counter()

                response = {
                    "request_id": request_id,
                    "ok": True,
                    "actions": np.asarray(
                        result["actions"],
                        dtype=np.float32,
                    ),
                    "policy_timing": result.get(
                        "policy_timing",
                        {},
                    ),
                    "server_timing": result.get(
                        "server_timing",
                        {},
                    ),
                    "fake_inference": result.get(
                        "fake_inference",
                        False,
                    ),
                    "timing": timing,
                }

            except Exception:
                timing.worker_infer_end = time.perf_counter()

                response = {
                    "request_id": request_id,
                    "ok": False,
                    "error": traceback.format_exc(),
                    "timing": timing,
                }

            timing.response_ready = time.perf_counter()
            response_queue.put(response)

    except Exception:
        response_queue.put(
            {
                "request_id": -1,
                "ok": False,
                "error": traceback.format_exc(),
                "timing": None,
            }
        )

class AsyncPolicyProcess:
    """Manage one policy inference worker process."""

    def __init__(
        self,
        *,
        remote_host: str,
        remote_port: int,
        fake_delay_ms: float | None = None,
        action_horizon: int = 50,
        action_dim: int = 7,
    ) -> None:
        if fake_delay_ms is not None and fake_delay_ms < 0:
            raise ValueError("fake_delay_ms must be non-negative")

        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        self._ctx = mp.get_context("spawn")

        self._request_queue = self._ctx.Queue(
            maxsize=1,
        )
        self._response_queue = self._ctx.Queue(
            maxsize=4,
        )
        self._stop_event = self._ctx.Event()

        self._process = self._ctx.Process(
            target=_policy_worker,
            args=(
                self._request_queue,
                self._response_queue,
                self._stop_event,
                remote_host,
                remote_port,
                fake_delay_ms,
                action_horizon,
                action_dim,
            ),
            daemon=True,
        )

        self._next_request_id = 0
        self._latest_request_id = -1
        self._inflight = False

    def start(self) -> None:
        """Start the policy inference worker process."""
        if self._process.pid is not None:
            raise RuntimeError(
                "AsyncPolicyProcess can only be started once"
            )

        self._process.start()

    def stop(self) -> None:
        """Stop the policy inference worker process."""
        if self._process.pid is None:
            return

        self._stop_event.set()

        try:
            self._request_queue.put_nowait(None)
        except queue.Full:
            try:
                self._request_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self._request_queue.put_nowait(None)
            except queue.Full:
                pass

        self._process.join(timeout=2.0)

        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)

        self._inflight = False

    @property
    def inflight(self) -> bool:
        """Return whether one inference request is still running."""
        return self._inflight

    def submit(
        self,
        observation: dict[str, Any],
        *,
        observation_step: int,
        submit_step: int,
        observation_start: float,
        observation_ready: float,
    ) -> RequestTiming:
        """Submit one observation for asynchronous inference."""
        if not self._process.is_alive():
            raise RuntimeError(
                "AsyncPolicyProcess must be started before submit()"
            )

        if self._inflight:
            raise RuntimeError(
                "Cannot submit while another request is in flight"
            )

        request_id = self._next_request_id
        request_submit = time.perf_counter()

        timing = RequestTiming(
            request_id=request_id,
            observation_step=observation_step,
            submit_step=submit_step,
            observation_start=observation_start,
            observation_ready=observation_ready,
            request_submit=request_submit,
        )

        self._request_queue.put_nowait(
            {
                "request_id": request_id,
                "observation": observation,
                "timing": timing,
            }
        )

        self._next_request_id += 1
        self._latest_request_id = request_id
        self._inflight = True

        return timing

    def poll(self) -> dict[str, Any] | None:
        """Return the newest available response without blocking."""
        latest_response = None

        while True:
            try:
                response = self._response_queue.get_nowait()
            except queue.Empty:
                break

            if (
                latest_response is None
                or response["request_id"]
                >= latest_response["request_id"]
            ):
                latest_response = response

        if latest_response is None:
            return None

        timing = latest_response.get("timing")

        if timing is not None:
            timing.control_poll = time.perf_counter()

        request_id = latest_response["request_id"]

        if request_id == -1:
            self._inflight = False
            return latest_response

        if request_id < self._latest_request_id:
            return None

        self._inflight = False
        return latest_response