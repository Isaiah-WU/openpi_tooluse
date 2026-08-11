"""Asynchronous policy inference for the UR10e client."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from typing import Any

import numpy as np

from openpi_client.server_capabilities import add_rtc_server_capability
from openpi_client.server_capabilities import validate_rtc_server_capability
from openpi_client.rtc_timing import RequestTiming


def _build_rtc_infer_kwargs(
    prev_chunk_left_over: np.ndarray | None,
    *,
    rtc_enabled: bool,
    inference_delay_steps: int | None,
    execution_horizon: int | None,
    action_dim: int,
) -> dict[str, Any]:
    """Build one RTC request, leaving first/no-prefix requests unchanged."""
    if prev_chunk_left_over is None:
        return {}
    if not rtc_enabled:
        raise ValueError("prev_chunk_left_over was provided while RTC is disabled")
    if (
        not isinstance(inference_delay_steps, int)
        or isinstance(inference_delay_steps, bool)
        or inference_delay_steps < 0
    ):
        raise ValueError("RTC requires a non-negative inference_delay_steps estimate")
    if (
        not isinstance(execution_horizon, int)
        or isinstance(execution_horizon, bool)
        or execution_horizon <= 0
    ):
        raise ValueError("RTC requires a positive execution_horizon")
    if execution_horizon < inference_delay_steps:
        raise ValueError("RTC execution_horizon must be at least inference_delay_steps")

    prefix = np.asarray(prev_chunk_left_over, dtype=np.float32)
    if prefix.ndim != 2 or prefix.shape[1] != action_dim:
        raise ValueError(
            "prev_chunk_left_over must have "
            f"shape (steps, {action_dim}), got {prefix.shape}"
        )
    if len(prefix) == 0:
        return {}

    return {
        "prev_chunk_left_over": np.ascontiguousarray(prefix),
        "inference_delay": inference_delay_steps,
        "execution_horizon": execution_horizon,
    }


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
    startup_queue,
    stop_event,
    remote_host: str,
    remote_port: int,
    fake_delay_ms: float | None,
    action_horizon: int,
    action_dim: int,
    rtc_enabled: bool,
    rtc_execution_horizon: int | None,
) -> None:
    """Run blocking policy inference outside the robot control process."""
    policy_client = None

    try:
        if fake_delay_ms is None:
            from openpi_client import websocket_client_policy

            policy_client = websocket_client_policy.WebsocketClientPolicy(
                host=remote_host,
                port=remote_port,
            )
            server_metadata = policy_client.get_server_metadata()
        else:
            server_metadata = add_rtc_server_capability(
                {},
                rtc_enabled=rtc_enabled,
                execution_horizon=rtc_execution_horizon or 1,
                prefix_attention_schedule="exp",
                max_guidance_weight=10.0,
            )

        validate_rtc_server_capability(
            server_metadata,
            rtc_requested=rtc_enabled,
        )
        startup_queue.put(
            {
                "ok": True,
                "server_metadata": server_metadata,
            }
        )

    except Exception:
        startup_queue.put(
            {
                "ok": False,
                "error": traceback.format_exc(),
            }
        )
        return

    try:
        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if request is None:
                break

            request_id = request["request_id"]
            observation = request["observation"]
            infer_kwargs = request["infer_kwargs"]
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
                    result = policy_client.infer(
                        observation,
                        **infer_kwargs,
                    )

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
                    "rtc_applied": bool(infer_kwargs),
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
        rtc_enabled: bool = False,
        rtc_inference_delay_steps: int | None = None,
        rtc_execution_horizon: int | None = None,
    ) -> None:
        if fake_delay_ms is not None and fake_delay_ms < 0:
            raise ValueError("fake_delay_ms must be non-negative")

        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        if rtc_enabled:
            if (
                not isinstance(rtc_inference_delay_steps, int)
                or isinstance(rtc_inference_delay_steps, bool)
                or rtc_inference_delay_steps < 0
            ):
                raise ValueError(
                    "rtc_inference_delay_steps must be non-negative when RTC is enabled"
                )
            if (
                not isinstance(rtc_execution_horizon, int)
                or isinstance(rtc_execution_horizon, bool)
                or rtc_execution_horizon <= 0
            ):
                raise ValueError(
                    "rtc_execution_horizon must be positive when RTC is enabled"
                )
            if rtc_execution_horizon < rtc_inference_delay_steps:
                raise ValueError(
                    "rtc_execution_horizon must be at least rtc_inference_delay_steps"
                )

        self._rtc_enabled = rtc_enabled
        self._rtc_inference_delay_steps = rtc_inference_delay_steps
        self._rtc_execution_horizon = rtc_execution_horizon
        self._action_dim = action_dim

        self._ctx = mp.get_context("spawn")

        self._request_queue = self._ctx.Queue(
            maxsize=1,
        )
        self._response_queue = self._ctx.Queue(
            maxsize=4,
        )
        self._startup_queue = self._ctx.Queue(
            maxsize=1,
        )
        self._stop_event = self._ctx.Event()

        self._process = self._ctx.Process(
            target=_policy_worker,
            args=(
                self._request_queue,
                self._response_queue,
                self._startup_queue,
                self._stop_event,
                remote_host,
                remote_port,
                fake_delay_ms,
                action_horizon,
                action_dim,
                rtc_enabled,
                rtc_execution_horizon,
            ),
            daemon=True,
        )

        self._next_request_id = 0
        self._latest_request_id = -1
        self._inflight = False
        self._server_metadata: dict[str, Any] | None = None

    def start(self, *, timeout_s: float = 30.0) -> None:
        """Start the worker and finish the server capability handshake."""
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self._process.pid is not None:
            raise RuntimeError("AsyncPolicyProcess can only be started once")

        self._process.start()
        try:
            startup = self._startup_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            self.stop()
            raise TimeoutError(
                f"Timed out after {timeout_s:.1f}s waiting for policy server metadata"
            ) from exc

        if not startup["ok"]:
            self.stop()
            raise RuntimeError("Policy worker startup failed:\n" f"{startup['error']}")
        self._server_metadata = startup["server_metadata"]

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

    @property
    def rtc_enabled(self) -> bool:
        """Return whether requests may include RTC prefix guidance."""
        return self._rtc_enabled

    @property
    def server_metadata(self) -> dict[str, Any]:
        """Return metadata verified during worker startup."""
        if self._server_metadata is None:
            raise RuntimeError("Policy worker has not completed startup")
        return self._server_metadata

    def configure_rtc_timing(
        self,
        *,
        inference_delay_steps: int,
        execution_horizon: int,
    ) -> None:
        """Update timing-only RTC parameters used by subsequent requests."""
        if not self._rtc_enabled:
            raise RuntimeError("Cannot configure RTC timing while RTC is disabled")
        if (
            not isinstance(inference_delay_steps, int)
            or isinstance(inference_delay_steps, bool)
            or inference_delay_steps < 0
        ):
            raise ValueError("inference_delay_steps must be a non-negative integer")
        if (
            not isinstance(execution_horizon, int)
            or isinstance(execution_horizon, bool)
            or execution_horizon <= 0
            or execution_horizon < inference_delay_steps
        ):
            raise ValueError(
                "execution_horizon must be an integer at least as large as the delay"
            )

        self._rtc_inference_delay_steps = inference_delay_steps
        self._rtc_execution_horizon = execution_horizon

    def submit(
        self,
        observation: dict[str, Any],
        *,
        observation_step: int,
        submit_step: int,
        observation_start: float,
        observation_ready: float,
        prev_chunk_left_over: np.ndarray | None = None,
    ) -> RequestTiming:
        """Submit one observation for asynchronous inference."""
        if not self._process.is_alive():
            raise RuntimeError("AsyncPolicyProcess must be started before submit()")

        if self._inflight:
            raise RuntimeError("Cannot submit while another request is in flight")

        request_id = self._next_request_id
        request_submit = time.perf_counter()
        infer_kwargs = _build_rtc_infer_kwargs(
            prev_chunk_left_over,
            rtc_enabled=self._rtc_enabled,
            inference_delay_steps=self._rtc_inference_delay_steps,
            execution_horizon=self._rtc_execution_horizon,
            action_dim=self._action_dim,
        )

        timing = RequestTiming(
            request_id=request_id,
            observation_step=observation_step,
            submit_step=submit_step,
            observation_start=observation_start,
            observation_ready=observation_ready,
            request_submit=request_submit,
            rtc_prefix_steps=(
                len(infer_kwargs["prev_chunk_left_over"]) if infer_kwargs else 0
            ),
            rtc_inference_delay_steps=(
                infer_kwargs.get("inference_delay") if infer_kwargs else None
            ),
            rtc_applied=bool(infer_kwargs),
        )

        self._request_queue.put_nowait(
            {
                "request_id": request_id,
                "observation": observation,
                "infer_kwargs": infer_kwargs,
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
                or response["request_id"] >= latest_response["request_id"]
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
