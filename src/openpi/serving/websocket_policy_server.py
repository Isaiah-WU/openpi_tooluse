import asyncio
import concurrent.futures
import functools
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class PolicyInferenceExecutor:
    """Serialize policy work on one persistent thread for CUDA Graph affinity."""

    def __init__(self) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="openpi-policy-inference",
        )

    def run(self, function, /, *args, **kwargs):
        """Run blocking policy work on the persistent inference thread."""
        return self._executor.submit(function, *args, **kwargs).result()

    async def run_async(self, function, /, *args, **kwargs):
        """Await policy work on the same persistent inference thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(function, *args, **kwargs),
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        inference_executor: PolicyInferenceExecutor | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._inference_executor = inference_executor
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                msg = msgpack_numpy.unpackb(await websocket.recv())
                # The client may bundle extra sampling kwargs (e.g. RTC prefix_actions)
                # alongside the observation. Only unwrap when both marker keys are present
                # so a raw observation dict (the original protocol) is untouched.
                if isinstance(msg, dict) and "observation" in msg and "infer_kwargs" in msg:
                    obs = msg["observation"]
                    infer_kwargs = dict(msg["infer_kwargs"])
                else:
                    obs = msg
                    infer_kwargs = {}

                infer_time = time.monotonic()
                # Run inference in a worker thread so it doesn't block the event loop --
                # otherwise this single call would stall every other connection (and any
                # pipelined prefetch request from this same client) for the duration of
                # the GPU/model forward pass.
                if self._inference_executor is None:
                    action = await asyncio.to_thread(self._policy.infer, obs, **infer_kwargs)
                else:
                    action = await self._inference_executor.run_async(self._policy.infer, obs, **infer_kwargs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
