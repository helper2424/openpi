#!/bin/bash
# Script to test RTC with tracking on remote machine

echo "Pulling latest changes from helper2424/openpi feat/add-rtc-support branch..."
git remote add helper2424 https://github.com/helper2424/openpi.git 2>/dev/null || true
git fetch helper2424 feat/add-rtc-support
git checkout feat/add-rtc-support
git pull helper2424 feat/add-rtc-support

echo "Running RTC evaluation with tracking..."
uv run python scripts/rtc_eval_dataset.py \
    --train-config-name cables10 \
    --checkpoint-path /home/ubuntu/rtc-check/cables10/ \
    --dataset-repo-id 1g0rrr/cables10 \
    --rtc-config.enabled \
    --rtc-config.prefix-attention-schedule EXP \
    --rtc-config.max-guidance-weight 5.0 \
    --rtc-config.execution-horizon 10 \
    --seed 42

echo "Check for output images:"
ls -la *.png