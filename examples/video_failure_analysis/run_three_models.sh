#!/usr/bin/env bash
set -euo pipefail

python examples/video_failure_analysis/run_benchmark.py \
  --stage benchmark \
  --models wan ltx cogvideox \
  --categories physics temporal_consistency instruction_following multi_object_interaction \
  --seeds 0 1 \
  --output_dir examples/video_failure_analysis/outputs \
  --resume \
  "$@"
