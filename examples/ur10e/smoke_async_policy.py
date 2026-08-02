from __future__ import annotations

import time

import numpy as np

from async_policy import AsyncPolicyProcess


def main() -> None:
    policy = AsyncPolicyProcess(
        remote_host="unused",
        remote_port=0,
        fake_delay_ms=100.0,
        action_horizon=50,
        action_dim=7,
    )

    policy.start()

    try:
        observation_start = time.perf_counter()

        observation = {
            "observation/state": np.arange(
                7,
                dtype=np.float32,
            ),
        }

        observation_ready = time.perf_counter()

        submitted_timing = policy.submit(
            observation,
            observation_step=10,
            submit_step=10,
            observation_start=observation_start,
            observation_ready=observation_ready,
        )

        deadline = time.perf_counter() + 2.0
        response = None
        poll_count = 0

        while response is None:
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for fake inference"
                )

            response = policy.poll()
            poll_count += 1

            if response is None:
                time.sleep(0.01)

        if not response["ok"]:
            raise RuntimeError(response["error"])

        returned_timing = response["timing"]

        print("request_id:", response["request_id"])
        print("actions_shape:", response["actions"].shape)
        print("first_action:", response["actions"][0])
        print("fake_inference:", response["fake_inference"])
        print("poll_count:", poll_count)
        print("inflight:", policy.inflight)
        print("same_request:", returned_timing.request_id == submitted_timing.request_id)

        for name, value in returned_timing.as_metrics().items():
            print(f"{name}: {value}")

    finally:
        policy.stop()


if __name__ == "__main__":
    main()