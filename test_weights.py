#!/usr/bin/env python
"""Test script to verify RTC weights calculation."""

import jax.numpy as jnp
from openpi.policies.rtc_processor import RTCProcessor, RTCConfig, RTCAttentionSchedule

def test_weights():
    # Create RTC processor with different schedules
    for schedule in [RTCAttentionSchedule.LINEAR, RTCAttentionSchedule.EXP, RTCAttentionSchedule.ZEROS, RTCAttentionSchedule.ONES]:
        print(f"\n{'='*60}")
        print(f"Testing schedule: {schedule}")
        print(f"{'='*60}")

        config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=schedule,
            max_guidance_weight=5.0,
            execution_horizon=10
        )

        processor = RTCProcessor(config)

        # Test with typical values
        start = 1
        end = 10
        total = 16

        weights = processor.get_prefix_weights(start, end, total, schedule)

        print(f"Parameters: start={start}, end={end}, total={total}")
        print(f"Weights shape: {weights.shape}")
        print(f"Weights values: {weights}")
        print(f"Sum: {float(jnp.sum(weights)):.3f}")
        print(f"Max: {float(jnp.max(weights)):.3f}")
        print(f"Min: {float(jnp.min(weights)):.3f}")
        print(f"Non-zero count: {int(jnp.sum(weights > 0))}")

        # Detailed view of first 10 values
        print(f"First 10 weights: {weights[:10]}")

if __name__ == "__main__":
    test_weights()