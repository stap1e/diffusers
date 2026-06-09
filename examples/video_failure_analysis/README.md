# Video Failure Analysis

This example provides a lightweight benchmark scaffold for finding failure modes in open-source video generation models supported by Diffusers.

This setup is configured for local-only model loading. The benchmark expects every model checkpoint to already exist on disk, and it will fail fast if any configured local path is missing.

It is designed around four first-pass categories:

- physical commonsense
- temporal consistency
- instruction following
- multi-object interaction

The goal is to produce a reproducible failure profile, not just a single aesthetic score.

## Files

```text
examples/video_failure_analysis/
  generate.py                         # generate videos and a manifest.jsonl
  run_benchmark.py                    # batch runner for dry run / benchmark / judge requests
  evaluate_llm_judge.py               # create judge requests or normalize judge annotations
  check_models.py                     # validate local model checkpoint paths
  configs/models.json                 # local model/runtime settings
  configs/models.example.json         # template with placeholder local paths
  prompts/*.json                      # first prompt set, 10 prompts per category
  judge_prompts/*.yaml                # extra judge templates for this benchmark
```

The suite also reuses existing judge prompts:

- `examples/cosmos/llm_judge_prompts/video_physics.yaml`
- `examples/cosmos/llm_judge_prompts/video_IF.yaml`

## Prompt schema

Each prompt record is a JSON object:

```json
{
  "id": "physics_001",
  "category": "physics",
  "prompt": "A glass ball falls from the edge of a wooden table...",
  "prompt_zh": "一个玻璃球从木桌边缘掉下...",
  "expected_objects": ["glass ball", "wooden table", "floor"],
  "expected_motion": "falling, impact, two bounces, rest",
  "known_failure_modes": ["floating", "no_collision_response"]
}
```

## Generate videos

Start with a small run before launching the full matrix:

```bash
python examples/video_failure_analysis/generate.py \
  --model wan \
  --prompt_file examples/video_failure_analysis/prompts/physics.json \
  --seeds 0 \
  --limit 2 \
  --output_dir examples/video_failure_analysis/outputs
```

Useful options:

```bash
--prompt_language zh       # use prompt_zh when present
--cpu_offload              # lower VRAM usage
--resume                   # skip videos already present
--height 480 --width 832   # override model defaults
--num_frames 81
--num_inference_steps 30
--guidance_scale 5.0
```

You can also use the batch runner to avoid writing shell loops by hand.

Before generating videos, validate that every configured checkpoint is a local Diffusers directory:

```bash
python examples/video_failure_analysis/check_models.py
```

To validate only selected models:

```bash
python examples/video_failure_analysis/check_models.py --models wan ltx cogvideox
```

All model paths are resolved from the `model_path` fields in `examples/video_failure_analysis/configs/models.json`, or from the `--model_path` override. `configs/models.example.json` provides placeholder values for new machines; copy the settings you need into `models.json` and replace every `model_path` with a local absolute path. Every model path must be an existing local absolute Diffusers checkpoint directory. `generate.py` passes `local_files_only=True` and rejects Hub repository IDs, relative paths, and missing paths, so no remote download fallback is used.

### Step 1: dry run

This matches the recommended first-pass validation setup:

- models: `wan`, `ltx`, `cogvideox`
- categories: all 4 benchmark categories
- prompts per category: 2
- seeds: `0`
- language: `en`

```bash
python examples/video_failure_analysis/run_benchmark.py \
  --stage dry_run \
  --resume
```

If VRAM is tight:

```bash
python examples/video_failure_analysis/run_benchmark.py \
  --stage dry_run \
  --cpu_offload \
  --resume
```

### Step 2: full mini benchmark

This produces the 240-video matrix:

```text
3 models * 4 categories * 10 prompts * 2 seeds = 240 videos
```

```bash
python examples/video_failure_analysis/run_benchmark.py \
  --stage benchmark \
  --resume
```

### Step 3: judge request JSONL

After generation completes:

```bash
python examples/video_failure_analysis/run_benchmark.py \
  --stage judge_requests
```

Or run the whole pipeline in one command:

```bash
python examples/video_failure_analysis/run_benchmark.py \
  --stage all \
  --resume
```

Outputs are written under:

```text
examples/video_failure_analysis/outputs/<model>/<prompt_file_stem>/
  videos/*.mp4
  manifest.jsonl
```

## Prepare judge requests

This script does not call an LLM provider directly. It writes JSONL requests containing `video_path`, `system_prompt`, and `user_prompt`, so you can run them through your preferred multimodal judge.

```bash
python examples/video_failure_analysis/evaluate_llm_judge.py \
  --manifest examples/video_failure_analysis/outputs/wan/physics/manifest.jsonl \
  --output examples/video_failure_analysis/results/wan_physics_requests.jsonl
```

Default judge prompts are selected by category:

| Category | Judge prompt |
|---|---|
| `physics` | `examples/cosmos/llm_judge_prompts/video_physics.yaml` |
| `instruction_following` | `examples/cosmos/llm_judge_prompts/video_IF.yaml` |
| `temporal_consistency` | `examples/video_failure_analysis/judge_prompts/video_temporal_consistency.yaml` |
| `multi_object_interaction` | `examples/video_failure_analysis/judge_prompts/video_interaction.yaml` |

Expected scored annotation rows should contain at least:

```json
{
  "request_id": "wan::physics_001::seed0::en",
  "score": 3,
  "reason": "The ball falls but the bounce is not physically plausible.",
  "failure_tags": ["no_collision_response"]
}
```

Normalize those annotations into result rows:

```bash
python examples/video_failure_analysis/evaluate_llm_judge.py \
  --manifest examples/video_failure_analysis/outputs/wan/physics/manifest.jsonl \
  --annotations examples/video_failure_analysis/results/wan_physics_annotations.jsonl \
  --output examples/video_failure_analysis/results/wan_physics_scores.jsonl
```

## Summarize results

```bash
python examples/video_failure_analysis/summarize.py \
  --results examples/video_failure_analysis/results/*_scores.jsonl \
  --output_dir examples/video_failure_analysis/reports
```

This writes:

- `summary.csv`
- `model_overview.csv`
- `seed_sensitivity.csv`
- `failure_profile.md`

Recommended interpretation:

- `summary.csv`: mean score and top failure tags for each `(model, category)`
- `model_overview.csv`: overall score plus per-category mean score for category-gap comparison
- `seed_sensitivity.csv`: per-prompt mean/std plus best and worst seed
- `failure_profile.md`: a readable report for failure-study style writeups

## Suggested first benchmark

A practical first pass is:

- models: `wan`, `ltx`, `cogvideox`
- categories: `physics`, `temporal_consistency`, `instruction_following`, `multi_object_interaction`
- prompts per category: 10
- seeds: 0 and 1

That produces:

```text
3 models * 4 categories * 10 prompts * 2 seeds = 240 videos
```

Use the generated failure tags and mean scores to identify which models need deeper follow-up experiments.

## Suggested execution order

This repository now supports the recommended workflow directly:

1. Run a dry run with `run_benchmark.py --stage dry_run`
2. Inspect generated videos and `manifest.jsonl`
3. Run the 240-video benchmark with `run_benchmark.py --stage benchmark`
4. Create judge requests with `run_benchmark.py --stage judge_requests`
5. Run your multimodal judge or human annotation system
6. Normalize annotations with `evaluate_llm_judge.py --annotations ...`
7. Summarize with `summarize.py`
8. Use `seed_sensitivity.csv` and `model_overview.csv` to select second-round ablations

## Controlled ablation suggestions

Keep the prompt set and seeds fixed, and change only one variable at a time:

- `--num_inference_steps 20 30 50` via repeated `generate.py` runs
- `--guidance_scale <default * 0.75 / default / default * 1.25>`
- `--num_frames 49 / 81 / 121 / 161` for long-video models such as `wan` and `ltx`
- `--prompt_language en` vs `--prompt_language zh`
- `--height` and `--width` for recommended / square / wide / portrait settings

For ablations, keep outputs in separate directories so the manifests remain easy to compare:

```bash
python examples/video_failure_analysis/generate.py \
  --model wan \
  --prompt_file examples/video_failure_analysis/prompts/physics.json \
  --seeds 0 1 \
  --limit 3 \
  --num_inference_steps 20 \
  --output_dir examples/video_failure_analysis/outputs_steps20 \
  --resume
```
