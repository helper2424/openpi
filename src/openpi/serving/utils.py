import jax

import dataclasses

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    
def create_policy(config: Checkpoint, default_prompt: str | None = None) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match config:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(config.config), config.dir, default_prompt=default_prompt
            )

def clear_device_memory():
    """Clear GPU/TPU memory more aggressively."""
    
    # Clear all JAX state
    jax.clear_caches()
    
    # For GPU: Force CUDA to release memory
    import jax.lib.xla_bridge as xb
    backend = xb.get_backend()
    
    # if backend.platform == 'gpu':
    #     # This forces synchronization and cleanup
    #     for device in jax.devices():
    #         device.synchronize_all_activity()
    
    # Garbage collect
    import gc
    gc.collect()