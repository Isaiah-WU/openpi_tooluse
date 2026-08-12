import dataclasses
import enum
import logging
import socket

import tyro

from openpi_client.server_capabilities import add_rtc_server_capability
from openpi.models_pytorch.rtc_processor import RTCInferenceConfig
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import ur10e_policy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Inference-time RTC is opt-in. The execution horizon should be replaced
    # with the value selected from measured UR10e inference delay.
    rtc: RTCInferenceConfig = dataclasses.field(default_factory=RTCInferenceConfig)

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    rtc_config: RTCInferenceConfig | None = None,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            rtc_config=rtc_config,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                rtc_config=args.rtc,
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                rtc_config=args.rtc,
            )


def warm_up_policy_for_serving(
    policy: _policy.Policy,
    rtc_config: RTCInferenceConfig,
    *,
    raw_observation: dict | None = None,
    inference_executor: websocket_policy_server.PolicyInferenceExecutor | None = None,
) -> list[float] | None:
    """Warm compiled RTC paths before the listening socket can be created."""
    if not rtc_config.enabled:
        return None
    if raw_observation is None:
        raise ValueError("RTC serving requires a deployment-specific warm-up observation")
    logging.info(
        "Warming up baseline and %d RTC inference paths before opening the server port",
        rtc_config.warmup_inferences,
    )
    warmup_kwargs = {
        "raw_observation": raw_observation,
        "execution_horizon": ur10e_policy.RTC_WARMUP_EXECUTION_HORIZON,
        "warmup_inferences": rtc_config.warmup_inferences,
        "prev_chunk_valid_steps": ur10e_policy.RTC_WARMUP_INITIAL_VALID_STEPS,
        "inference_delay": ur10e_policy.RTC_WARMUP_INFERENCE_DELAY,
        "expected_action_dim": ur10e_policy.RTC_WARMUP_ACTION_DIM,
    }
    if inference_executor is None:
        timings = policy.warm_up_rtc(**warmup_kwargs)
    else:
        timings = inference_executor.run(policy.warm_up_rtc, **warmup_kwargs)
    logging.info(
        "RTC warm-up complete (baseline=%.3fs, rtc=%s)",
        timings[0],
        ", ".join(f"{seconds:.3f}s" for seconds in timings[1:]),
    )
    return timings


def main(args: Args) -> None:
    policy = create_policy(args)
    inference_executor = (
        websocket_policy_server.PolicyInferenceExecutor()
        if args.rtc.enabled
        else None
    )
    warmup_observation = None
    if args.rtc.enabled and isinstance(args.policy, Checkpoint):
        train_config = _config.get_config(args.policy.config)
        if isinstance(train_config.data, _config.LeRobotUR10eDataConfig):
            warmup_observation = ur10e_policy.make_ur10e_rtc_warmup_observation()
    try:
        warmup_timings = warm_up_policy_for_serving(
            policy,
            args.rtc,
            raw_observation=warmup_observation,
            inference_executor=inference_executor,
        )
    except BaseException:
        if inference_executor is not None:
            inference_executor.close()
        raise
    warmup_complete = True if warmup_timings is not None else None

    policy_metadata = add_rtc_server_capability(
        policy.metadata,
        rtc_enabled=args.rtc.enabled,
        execution_horizon=args.rtc.execution_horizon,
        prefix_attention_schedule=args.rtc.prefix_attention_schedule,
        max_guidance_weight=args.rtc.max_guidance_weight,
        warmup_complete=warmup_complete,
        warmup_inferences=args.rtc.warmup_inferences if args.rtc.enabled else None,
    )

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    try:
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
            inference_executor=inference_executor,
        )
        server.serve_forever()
    finally:
        if inference_executor is not None:
            inference_executor.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
