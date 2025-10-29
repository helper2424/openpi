#!/usr/bin/env python

import logging
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tyro

from openpi.policies import policy_config
from openpi.training import config as train_config
import openpi.training.data_loader as _data_loader


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # JAX uses explicit PRNG keys, but we can set the default behavior
    key = jax.random.PRNGKey(seed)
    logging.info(f"Random seed set to: {seed}")
    logging.info(f"JAX PRNG key initialized with seed: {seed}")
    return key


class RTCDatasetEvaluator:
    """Evaluator for RTC on dataset samples."""

    def __init__(self, cfg: "Args"):
        self.cfg = cfg

        # Load the training config to get model configuration
        logging.info(f"Loading policy from {cfg.checkpoint_path}")
        self.train_cfg = train_config.get_config(cfg.train_config_name)

        # Load policy using the openpi policy_config
        self.policy = policy_config.create_trained_policy(
            train_config=self.train_cfg,
            checkpoint_dir=pathlib.Path(cfg.checkpoint_path),
            sample_kwargs=cfg.sample_kwargs or {},
        )

        logging.info(f"Policy loaded successfully")
        logging.info(f"Model config: {self.train_cfg.model}")

        # Create data loader using the training config
        self.data_loader = _data_loader.create_data_loader(
            self.train_cfg,
            shuffle=True,
            skip_norm_stats=True,  # Skip normalization for evaluation
        )

        logging.info(f"Dataloader created successfully")

    def run_evaluation(self) -> dict:
        """Run full evaluation on dataset.

        Returns:
            Dictionary with aggregated metrics and detailed results
        """
        logging.info(f"Collecting episodes from dataloader to select 2 random episodes")

        # Collect all episodes from the dataloader
        all_episodes = []
        for batch in self.data_loader:
            # Each batch may contain multiple episodes
            # Store the batch data for later random selection
            all_episodes.append(batch)

        if len(all_episodes) < 2:
            logging.error(f"Not enough episodes in dataloader. Found {len(all_episodes)}, need at least 2")
            return {}

        # Randomly select 2 episodes
        selected_indices = random.sample(range(len(all_episodes)), 2)
        logging.info(f"Selected episodes at indices: {selected_indices}")

        # Get the two selected episodes
        first_episode = all_episodes[selected_indices[0]]
        second_episode = all_episodes[selected_indices[1]]

        # Extract actions from first episode
        prev_chunk_left_over = first_episode.get("actions")
        if prev_chunk_left_over is None and "action" in first_episode:
            prev_chunk_left_over = np.array(first_episode["action"])

        if prev_chunk_left_over is not None:
            # Take only first half of actions for comparison
            prev_chunk_left_over = prev_chunk_left_over[:self.cfg.action_horizon // 2]
        else:
            logging.warning("No actions found in first episode, skipping evaluation")
            return {}

        # Convert second episode to proper format
        # Note: This assumes the dataset returns observations in a specific format
        # You may need to adjust this based on your dataset structure
        obs = {
            "observation/image": np.array(second_episode.get("observation.images.top", second_episode.get("observation/image"))),
            "observation/state": np.array(second_episode.get("observation.state", [])),
        }

        # Add prompt if available
        if "prompt" in second_episode or "task" in second_episode:
            obs["prompt"] = second_episode.get("prompt", second_episode.get("task", ""))

        # Generate noise for inference
        noise = np.random.randn(self.cfg.action_horizon, self.cfg.action_dim).astype(np.float32)

        # Inference using the policy
        # Note: The pi0 model's inference is handled through the Policy.infer method
        result = self.policy.infer(obs, noise=noise, inference_delay=self.cfg.inference_delay, prev_chunk_left_over=prev_chunk_left_over)
        actions = result["actions"]

        # Create visualization
        fig, axs = plt.subplots(min(6, self.cfg.action_dim), 1, figsize=(12, 12))
        if self.cfg.action_dim == 1:
            axs = [axs]
        fig.suptitle(f"Episodes {selected_indices[0]} & {selected_indices[1]} - Action Prediction", fontsize=16)

        # Plot actions
        self.axs = axs
        self.plot_waypoints(prev_chunk_left_over, label="Previous Actions (Episode 1)", color="green")
        self.plot_waypoints(actions, label="Predicted Actions (Episode 2)", color="blue")

        plt.tight_layout()
        plt.savefig(f"actions_episodes_{selected_indices[0]}_{selected_indices[1]}.png", dpi=150)
        logging.info(f"Saved actions comparison to actions_episodes_{selected_indices[0]}_{selected_indices[1]}.png")
        plt.close(fig)

        logging.info("Evaluation completed")
        return {}

    def plot_waypoints(self, chunk, start_from: int = 0, color: str | None = None, label: str | None = None):
        for j in range(chunk.shape[-1]):
            self.axs[j].plot(
                np.arange(start_from, start_from + chunk.shape[0]),
                chunk[:, j],
                color=color,
                label=label,
            )
            self.axs[j].set_ylabel("Joint angle", fontsize=14)
            self.axs[j].grid()
            plt.tick_params(labelsize=14)
            self.axs[j].legend(loc="upper right", fontsize=14)
            if j == 2:
                self.axs[j].set_xlabel("Step #", fontsize=16)

@dataclass
class Args:
    """Arguments for RTC dataset evaluation."""

    # Training config name (e.g., "pi0_libero", "pi05_droid", etc.)
    train_config_name: str = field(
        metadata={"help": "Name of the training config to use (from openpi.training.config)"}
    )

    # Path to the checkpoint directory
    checkpoint_path: str = field(
        metadata={"help": "Path to the checkpoint directory"}
    )

    # Dataset configuration
    dataset_repo_id: str = field(
        metadata={"help": "HuggingFace repo ID for the LeRobot dataset"}
    )

    action_horizon: int = field(
        default=50,
        metadata={"help": "Action horizon (chunk size)"},
    )

    # If provided, will be used as default prompt
    default_prompt: str | None = field(
        default=None,
        metadata={"help": "Default prompt to use if not present in the data"},
    )

    # Seed configuration
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"},
    )

    # Action dimension
    action_dim: int = field(
        default=7,
        metadata={"help": "Action dimension"},
    )

    # Additional sample kwargs to pass to the model
    sample_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={"help": "Additional kwargs to pass to sample_actions"},
    )

    # Inference delay
    inference_delay: int = 1
    
def main(args: Args):
    """Main entry point for RTC dataset evaluation."""
    # Set random seed for reproducibility
    rng_key = set_seed(args.seed)

    logging.info("=" * 80)
    logging.info("Pi0 Dataset Evaluation with JAX")
    logging.info("=" * 80)
    logging.info(f"Training config: {args.train_config_name}")
    logging.info(f"Checkpoint: {args.checkpoint_path}")
    logging.info(f"Dataset: {args.dataset_repo_id}")
    logging.info(f"Action horizon: {args.action_horizon}")
    logging.info(f"Seed: {args.seed}")
    logging.info("=" * 80)

    evaluator = RTCDatasetEvaluator(args)
    evaluator.run_evaluation()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
