#!/usr/bin/env python
"""Test script to verify RTC config parsing with tyro."""

import tyro
from dataclasses import dataclass, field
from openpi.policies.rtc_processor import RTCConfig, RTCAttentionSchedule

@dataclass
class TestArgs:
    train_config_name: str
    checkpoint_path: str
    dataset_repo_id: str
    seed: int = 42
    rtc_config: RTCConfig = field(default_factory=RTCConfig)

def main(args: TestArgs):
    print("=" * 60)
    print("Parsed arguments:")
    print(f"  train_config_name: {args.train_config_name}")
    print(f"  checkpoint_path: {args.checkpoint_path}")
    print(f"  dataset_repo_id: {args.dataset_repo_id}")
    print(f"  seed: {args.seed}")
    print(f"  rtc_config: {args.rtc_config}")
    print(f"    - enabled: {args.rtc_config.enabled}")
    print(f"    - prefix_attention_schedule: {args.rtc_config.prefix_attention_schedule}")
    print(f"    - max_guidance_weight: {args.rtc_config.max_guidance_weight}")
    print(f"    - execution_horizon: {args.rtc_config.execution_horizon}")
    print("=" * 60)

if __name__ == "__main__":
    main(tyro.cli(TestArgs))