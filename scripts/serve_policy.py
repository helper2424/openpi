import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
import jax

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

    policies_dirs: list[str] = dataclasses.field(default_factory=list)
    policies_configs: list[str] = dataclasses.field(default_factory=list)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)

def clear_device_memory():
    """Clear GPU/TPU memory more aggressively."""
    
    # Clear all JAX state
    jax.clear_caches()
    
    # For GPU: Force CUDA to release memory
    import jax.lib.xla_bridge as xb
    backend = xb.get_backend()
    
    if backend.platform == 'gpu':
        # This forces synchronization and cleanup
        for device in jax.devices():
            device.synchronize_all_activity()
    
    # Garbage collect
    import gc
    gc.collect()


def main(args: Args) -> None:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    if len(args.policies_dirs) <= 0:
        raise ValueError("policies_dirs must be provided")
    if len(args.policies_configs) <= 0:
        raise ValueError("policies_configs must be provided")
    
    if len(args.policies_dirs) != len(args.policies_configs):
        raise ValueError("policies_dirs and policies_configs must have the same length")
    
    policies_configs = []
    for policy_dir, policy_config in zip(args.policies_dirs, args.policies_configs):
        policies_configs.append(Checkpoint(config=policy_config, dir=policy_dir))
    
    # Initialize the policies before using them
    for policy in policies_configs:
        clear_device_memory()
        create_policy(policy)

    server = websocket_policy_server.WebsocketPolicyServer(
        policies_configs=policies_configs,
        host="0.0.0.0",
        port=args.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
