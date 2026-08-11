import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(  # noqa: UP006
        self,
        obs: Dict,
        *,
        prefix_actions=None,
        prefix_attention_horizon=None,
        prev_chunk_left_over=None,
        inference_delay=None,
        execution_horizon=None,
    ) -> Dict:  # noqa: UP006
        if prefix_actions is not None and prev_chunk_left_over is not None:
            raise ValueError("prefix_actions and prev_chunk_left_over cannot be used together")

        infer_kwargs = {}
        if prefix_actions is not None:
            infer_kwargs.update(
                {
                    "prefix_actions": prefix_actions,
                    "prefix_attention_horizon": prefix_attention_horizon,
                }
            )
        if prev_chunk_left_over is not None:
            if inference_delay is None:
                raise ValueError("prev_chunk_left_over requires inference_delay")
            infer_kwargs.update(
                {
                    "prev_chunk_left_over": prev_chunk_left_over,
                    "inference_delay": inference_delay,
                    "execution_horizon": execution_horizon,
                }
            )

        if not infer_kwargs:
            # Common path: send the observation as-is (byte-identical to the original
            # protocol, so older servers keep working).
            data = self._packer.pack(obs)
        else:
            # RTC path: bundle the observation together with the extra sampling kwargs.
            # The server unwraps "observation" and forwards "infer_kwargs" to Policy.infer.
            data = self._packer.pack(
                {
                    "observation": obs,
                    "infer_kwargs": infer_kwargs,
                }
            )
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
