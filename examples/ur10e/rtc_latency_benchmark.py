"""Measure warmed RTC request latency without executing robot actions."""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from openpi_client.rtc_latency_benchmark import recommend_from_rtc_latency
from openpi_client.rtc_latency_benchmark import run_rtc_latency_benchmark
from openpi_client.rtc_latency_benchmark import save_rtc_latency_samples
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi.policies.ur10e_policy import make_ur10e_rtc_warmup_observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--policy-hz", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 100:
        parser.error("--samples must be at least 100 for RTC P99 calibration")

    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or Path("rtc_latency_benchmarks") / timestamp / "rtc_latency.csv"
    policy = WebsocketClientPolicy(host=args.server_host, port=args.server_port)
    try:
        samples = run_rtc_latency_benchmark(
            policy,
            make_ur10e_rtc_warmup_observation(),
            sample_count=args.samples,
            policy_hz=args.policy_hz,
        )
        save_rtc_latency_samples(output_path, samples)
        recommendation = recommend_from_rtc_latency(samples)
    finally:
        policy.close()

    print(f"saved={output_path}")
    print(f"samples={recommendation.sample_count}")
    print(
        "observed_delay_policy_steps: "
        f"p50={recommendation.delay_p50:.3f} "
        f"p95={recommendation.delay_p95:.3f} "
        f"p99={recommendation.delay_p99:.3f} "
        f"max={recommendation.delay_max:.3f}"
    )
    print(f"D={recommendation.inference_delay_policy_steps}")
    print(f"S={recommendation.execution_horizon_policy_steps}")
    print(f"Q={recommendation.query_remaining_policy_steps}")


if __name__ == "__main__":
    main()
