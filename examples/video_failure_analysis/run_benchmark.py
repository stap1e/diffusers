import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = ["wan", "ltx", "cogvideox"]
DEFAULT_CATEGORIES = [
    "physics",
    "temporal_consistency",
    "instruction_following",
    "multi_object_interaction",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the video failure analysis baseline pipeline.")
    parser.add_argument(
        "--stage",
        choices=["dry_run", "benchmark", "judge_requests", "all"],
        default="dry_run",
        help="Pipeline stage to run.",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--model_config", default="examples/video_failure_analysis/configs/models.json")
    parser.add_argument("--prompt_dir", default="examples/video_failure_analysis/prompts")
    parser.add_argument("--output_dir", default="examples/video_failure_analysis/outputs")
    parser.add_argument("--results_dir", default="examples/video_failure_analysis/results")
    parser.add_argument("--prompt_language", choices=["en", "zh"], default="en")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip_model_check",
        action="store_true",
        help="Skip local model_path validation before generation stages.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately when any subprocess fails.",
    )
    return parser.parse_args()


def run_command(command, fail_fast):
    print("$ " + " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError:
        if fail_fast:
            raise
        return False


def model_check_command(args):
    return [
        sys.executable,
        "examples/video_failure_analysis/check_models.py",
        "--model_config",
        args.model_config,
        "--models",
        *args.models,
    ]


def generation_overrides(args):
    if args.stage == "dry_run":
        return {"seeds": args.seeds or [0], "limit": args.limit if args.limit is not None else 2}
    return {"seeds": args.seeds or [0, 1], "limit": args.limit}


def generation_jobs(args):
    overrides = generation_overrides(args)
    for model in args.models:
        for category in args.categories:
            prompt_file = Path(args.prompt_dir) / f"{category}.json"
            command = [
                sys.executable,
                "examples/video_failure_analysis/generate.py",
                "--model",
                model,
                "--model_config",
                args.model_config,
                "--prompt_file",
                str(prompt_file),
                "--output_dir",
                args.output_dir,
                "--prompt_language",
                args.prompt_language,
                "--device",
                args.device,
                "--seeds",
                *[str(seed) for seed in overrides["seeds"]],
            ]
            if overrides["limit"] is not None:
                command.extend(["--limit", str(overrides["limit"])])
            if args.cpu_offload:
                command.append("--cpu_offload")
            if args.resume:
                command.append("--resume")
            yield command


def judge_request_jobs(args):
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for category in args.categories:
            manifest = Path(args.output_dir) / model / category / "manifest.jsonl"
            output = results_dir / f"{model}_{category}_requests.jsonl"
            yield [
                sys.executable,
                "examples/video_failure_analysis/evaluate_llm_judge.py",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
            ]


def main():
    args = parse_args()
    commands = []
    if args.stage in {"dry_run", "benchmark", "all"} and not args.skip_model_check:
        commands.append(model_check_command(args))
    if args.stage in {"dry_run", "benchmark", "all"}:
        commands.extend(generation_jobs(args))
    if args.stage in {"judge_requests", "all"}:
        commands.extend(judge_request_jobs(args))

    if not commands:
        return

    failures = []
    for command in commands:
        ok = run_command(command, args.fail_fast)
        if not ok:
            failures.append(command)

    if failures:
        print(f"{len(failures)} command(s) failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
