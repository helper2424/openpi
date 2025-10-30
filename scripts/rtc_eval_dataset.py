#!/usr/bin/env python

import logging
import pathlib
import random
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from openpi.policies import rtc_processor
import tyro

from openpi.policies import policy_config
from openpi.training import config as train_config
import openpi.training.data_loader as _data_loader
from openpi.policies.rtc_processor import RTCConfig
from openpi.policies import policy_config


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

        if self.policy._model.config.rtc_config is None:
            self.policy._model.config.rtc_config = RTCConfig()
            self.policy._model.config.rtc_config.enabled = self.cfg.rtc_config.enabled
            self.policy._model.config.rtc_config.prefix_attention_schedule = self.cfg.rtc_config.prefix_attention_schedule
            self.policy._model.config.rtc_config.max_guidance_weight = self.cfg.rtc_config.max_guidance_weight
            self.policy._model.config.rtc_config.execution_horizon = self.cfg.rtc_config.execution_horizon

        self.policy._model.init_rtc_processor()

        logging.info(f"RTC config: {self.policy._model.config.rtc_config}")

        logging.info(f"Policy loaded successfully")
        logging.info(f"Model config: {self.train_cfg.model}")

        # Create raw dataset without transforms for inference
        # We'll use policy.infer() which applies transforms itself
        self.data_config = self.train_cfg.data.create(self.train_cfg.assets_dirs, self.train_cfg.model)
        self.raw_dataset = _data_loader.create_torch_dataset(
            self.data_config,
            self.train_cfg.model.action_horizon,
            self.train_cfg.model
        )

        # Apply only repack transforms to get the right key structure
        # Skip data_transforms and model_transforms since policy.infer() will apply them
        self.dataset = _data_loader.TransformedDataset(
            self.raw_dataset,
            list(self.data_config.repack_transforms.inputs)
        )

        logging.info(f"Dataloader created successfully")

    def run_evaluation(self) -> dict:
        """Run full evaluation on dataset.

        Returns:
            Dictionary with aggregated metrics and detailed results
        """
        # Randomly select 2 samples from the dataset
        dataset_size = len(self.dataset)
        logging.info(f"Dataset size: {dataset_size}")

        if dataset_size < 2:
            logging.error(f"Not enough samples in dataset. Found {dataset_size}, need at least 2")
            return {}

        selected_indices = random.sample(range(dataset_size), 2)
        logging.info(f"Selected samples at indices: {selected_indices}")

        # Get the two selected samples
        first_sample = self.dataset[selected_indices[0]]
        second_sample = self.dataset[selected_indices[1]]

        # Extract actions from first sample
        # Take only first half based on model's action_horizon, not cfg
        model_action_horizon = self.train_cfg.model.action_horizon
        prev_chunk_left_over = np.array(first_sample["actions"])
        if len(prev_chunk_left_over.shape) > 1:
            # Take first half based on model's action horizon
            prev_chunk_left_over = prev_chunk_left_over[:model_action_horizon // 2]
            # Add batch dimension to match observation batch size
            prev_chunk_left_over = prev_chunk_left_over[np.newaxis, ...]  # Shape: (1, time, action_dim)
            logging.info(f"prev_chunk_left_over shape after adding batch dim: {prev_chunk_left_over.shape}")
        else:
            logging.warning("Actions have unexpected shape, skipping evaluation")
            return {}

        # Prepare observation dict for policy.infer()
        # Remove actions key since policy.infer() doesn't need it
        obs = {k: v for k, v in second_sample.items() if k != "actions"}

        # Debug: print observation keys
        logging.info(f"Observation keys: {list(obs.keys())}")
        for key, value in obs.items():
            if isinstance(value, (np.ndarray, list)):
                logging.info(f"  {key}: shape={np.array(value).shape if hasattr(value, 'shape') or isinstance(value, list) else 'N/A'}, type={type(value)}")
            elif isinstance(value, dict):
                logging.info(f"  {key}: nested dict with keys {list(value.keys())}")
            else:
                logging.info(f"  {key}: type={type(value)}")

        # Generate noise for inference
        # Use model's action_horizon and action_dim, not cfg values
        model_action_horizon = self.train_cfg.model.action_horizon
        model_action_dim = self.train_cfg.model.action_dim
        noise = np.random.randn(model_action_horizon, model_action_dim).astype(np.float32)

        # Inference using the policy
        # Note: The pi0 model's inference is handled through the Policy.infer method
        # execution_horizon is required even if RTC is not enabled in the model
        result = self.policy.infer(
            obs,
            noise=noise,
            inference_delay=4,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=self.cfg.execution_horizon
        )
        actions = result["actions"]

        # Create visualization
        # Use min of 6 and model's action_dim for plots
        num_plots = min(6, model_action_dim)
        fig, axs = plt.subplots(num_plots, 1, figsize=(12, 12))
        if num_plots == 1:
            axs = [axs]
        fig.suptitle(f"Episodes {selected_indices[0]} & {selected_indices[1]} - Action Prediction", fontsize=16)

        # Plot actions
        self.axs = axs
        # Remove batch dimension for plotting
        prev_chunk_to_plot = prev_chunk_left_over[0] if prev_chunk_left_over.ndim == 3 else prev_chunk_left_over
        actions_to_plot = actions[0] if actions.ndim == 3 else actions
        self.plot_waypoints(prev_chunk_to_plot, label="Previous Actions (Episode 1)", color="green")
        self.plot_waypoints(actions_to_plot, label="Predicted Actions (Episode 2)", color="blue")

        plt.tight_layout()
        plt.savefig(f"actions_episodes_{selected_indices[0]}_{selected_indices[1]}.png", dpi=150)
        logging.info(f"Saved actions comparison to actions_episodes_{selected_indices[0]}_{selected_indices[1]}.png")
        plt.close(fig)

        logging.info("Evaluation completed")
        return {}

    def plot_waypoints(self, chunk, start_from: int = 0, color: str | None = None, label: str | None = None):
        # Only plot as many dimensions as we have subplots
        num_dims_to_plot = min(len(self.axs), chunk.shape[-1])
        for j in range(num_dims_to_plot):
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

    # Execution horizon for RTC
    execution_horizon: int = field(
        default=10,
        metadata={"help": "Execution horizon for RTC (number of timesteps for prefix weights)"},
    )

    rtc_config: RTCConfig = field(
        default_factory=RTCConfig,
        metadata={"help": "RTC configuration for real-time control"},
    )
    
def main(args: Args):
    """Main entry point for RTC dataset evaluation."""
    # Set random seed for reproducibility
    rng_key = set_seed(args.seed)

    # Ensure RTC is enabled when config is provided
    # tyro may not preserve the default enabled=True when parsing nested dataclasses
    if args.rtc_config is not None:
        args.rtc_config = replace(args.rtc_config, enabled=True)

    logging.info("=" * 80)
    logging.info("Pi0 Dataset Evaluation with JAX")
    logging.info("=" * 80)
    logging.info(f"Training config: {args.train_config_name}")
    logging.info(f"Checkpoint: {args.checkpoint_path}")
    logging.info(f"Dataset: {args.dataset_repo_id}")
    logging.info(f"Action horizon: {args.action_horizon}")
    logging.info(f"Seed: {args.seed}")
    logging.info(f"RTC config: {args.rtc_config}")
    logging.info("=" * 80)

    evaluator = RTCDatasetEvaluator(args)
    evaluator.run_evaluation()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Starting RTC dataset evaluation")
    main(tyro.cli(Args))
