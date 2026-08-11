from __future__ import annotations

import time

import numpy as np

from openpi_client.rtc_async_policy import AsyncPolicyProcess


def test_packaged_fake_worker_completes_capability_handshake_and_inference() -> None:
    policy = AsyncPolicyProcess(
        remote_host="unused",
        remote_port=8000,
        fake_delay_ms=5.0,
        action_horizon=50,
        action_dim=7,
        rtc_enabled=True,
        rtc_inference_delay_steps=2,
        rtc_execution_horizon=10,
    )
    policy.start(timeout_s=5.0)
    try:
        assert policy.server_metadata["openpi_server"]["rtc"]["enabled"] is True
        now = time.perf_counter()
        timing = policy.submit(
            {"observation/state": np.arange(7, dtype=np.float32)},
            observation_step=0,
            submit_step=0,
            observation_start=now,
            observation_ready=now,
        )
        deadline = time.perf_counter() + 5.0
        response = None
        while response is None and time.perf_counter() < deadline:
            response = policy.poll()
            time.sleep(0.001)
        assert response is not None
        assert response["ok"] is True
        assert response["actions"].shape == (50, 7)
        assert response["timing"].request_id == timing.request_id
    finally:
        policy.stop()
