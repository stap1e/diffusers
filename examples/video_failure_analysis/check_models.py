import argparse
import json
import sys
from pathlib import Path


DEFAULT_MODEL_CONFIG = "examples/video_failure_analysis/configs/models.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate local model paths for video failure analysis.")
    parser.add_argument("--model_config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--models", nargs="+", help="Optional model keys to validate. Defaults to all models.")
    return parser.parse_args()


def load_configs(path):
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate_model(name, config):
    errors = []
    raw_path = config.get("model_path")
    if raw_path is None:
        return [f"{name}: missing required local model_path"]

    model_path = Path(raw_path).expanduser()
    if not model_path.is_absolute():
        errors.append(f"{name}: model_path must be an absolute local path, got '{raw_path}'")
        return errors
    if not model_path.exists():
        errors.append(f"{name}: local model path does not exist: {model_path}")
        return errors
    if not model_path.is_dir():
        errors.append(f"{name}: model_path must be a Diffusers checkpoint directory: {model_path}")
        return errors
    if not (model_path / "model_index.json").exists():
        errors.append(f"{name}: missing model_index.json in Diffusers checkpoint directory: {model_path}")
    return errors


def main():
    args = parse_args()
    configs = load_configs(args.model_config)
    names = args.models or sorted(configs)

    unknown = sorted(set(names) - set(configs))
    if unknown:
        print(f"Unknown model(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"Available models: {', '.join(sorted(configs))}", file=sys.stderr)
        sys.exit(1)

    failures = []
    for name in names:
        errors = validate_model(name, configs[name])
        if errors:
            failures.extend(errors)
            continue
        print(f"OK {name}: {Path(configs[name]['model_path']).expanduser()}")

    if failures:
        for failure in failures:
            print(f"ERROR {failure}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
