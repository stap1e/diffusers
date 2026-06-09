import argparse
import inspect
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from diffusers.utils import export_to_video


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate videos for video failure analysis prompts.")
    parser.add_argument("--model", required=True, help="Model key from --model_config, e.g. wan, ltx, cogvideox.")
    parser.add_argument("--model_config", default="examples/video_failure_analysis/configs/models.json")
    parser.add_argument("--prompt_file", required=True, help="JSON prompt file or JSONL prompt file.")
    parser.add_argument("--output_dir", default="examples/video_failure_analysis/outputs")
    parser.add_argument("--prompt_language", choices=["en", "zh"], default="en")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--model_path", help="Local absolute model directory. Overrides the path in --model_config.")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num_frames", type=int)
    parser.add_argument("--num_inference_steps", type=int)
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--torch_dtype", choices=sorted(DTYPES), default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="Skip videos that already exist.")
    return parser.parse_args()


def load_records(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def load_config(args):
    configs = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    if args.model not in configs:
        raise ValueError(f"Unknown model '{args.model}'. Available models: {', '.join(sorted(configs))}")

    config = dict(configs[args.model])
    for key in [
        "model_path",
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        "guidance_scale",
        "fps",
        "torch_dtype",
    ]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if "model_path" not in config:
        raise ValueError(f"Model '{args.model}' is missing required local model_path in {args.model_config}")
    return config


def get_installed_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def validate_environment(config):
    transformers_version = get_installed_version("transformers")
    required_transformers = Version("4.41.2")
    pipeline_class = config["pipeline_class"]

    if transformers_version is None:
        raise RuntimeError("transformers is not installed in the active environment.")

    if Version(transformers_version) < required_transformers:
        raise RuntimeError(
            f"{pipeline_class} requires transformers>={required_transformers}, but found {transformers_version}. "
            "Upgrade the active environment, for example:\n"
            "  /home/hongyu/miniconda3/envs/ecai/bin/python -m pip install -U 'transformers>=4.41.2'"
        )

    model_path = Path(config["model_path"]).expanduser()
    if not model_path.is_absolute():
        raise RuntimeError(
            f"{pipeline_class} requires a local absolute model path, but found '{config['model_path']}'. "
            "Hub repository IDs and relative paths are not allowed because this benchmark must not download weights."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Local model path does not exist: {model_path}")
    if not model_path.is_dir():
        raise NotADirectoryError(f"Local model path must be a Diffusers checkpoint directory: {model_path}")
    if not (model_path / "model_index.json").exists():
        raise FileNotFoundError(f"Missing model_index.json in Diffusers checkpoint directory: {model_path}")
    config["model_path"] = str(model_path)


def build_pipeline(config, device, cpu_offload):
    import diffusers

    pipeline_cls = getattr(diffusers, config["pipeline_class"])
    pipe = pipeline_cls.from_pretrained(
        config["model_path"],
        torch_dtype=DTYPES[config.get("torch_dtype", "bfloat16")],
        local_files_only=True,
    )

    if cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    return pipe


def get_prompt(record, language):
    if language == "zh" and record.get("prompt_zh"):
        return record["prompt_zh"]
    return record["prompt"]


def filtered_call_kwargs(pipe, kwargs):
    signature = inspect.signature(pipe.__call__)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def extract_frames(output):
    if hasattr(output, "frames"):
        frames = output.frames
    elif hasattr(output, "video"):
        frames = output.video
    else:
        frames = output[0]

    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu()
        if frames.ndim == 5:
            frames = frames[0]
        if frames.ndim == 4 and frames.shape[1] in (1, 3, 4):
            frames = frames.permute(0, 2, 3, 1)
        frames = frames.float().clamp(0, 1).numpy()

    if isinstance(frames, list) and frames and isinstance(frames[0], list):
        return frames[0]
    if not isinstance(frames, list) and hasattr(frames, "shape") and len(frames.shape) == 5:
        return frames[0]
    return frames


def write_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    config = load_config(args)
    validate_environment(config)
    records = load_records(args.prompt_file)
    if args.limit is not None:
        records = records[: args.limit]

    prompt_file = Path(args.prompt_file)
    output_root = Path(args.output_dir) / args.model / prompt_file.stem
    video_dir = output_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"

    pipe = build_pipeline(config, args.device, args.cpu_offload)
    fps = config.get("fps", 10)

    for record in records:
        prompt = get_prompt(record, args.prompt_language)
        for seed in args.seeds:
            video_name = f"{record['id']}_seed{seed}_{args.prompt_language}.mp4"
            video_path = video_dir / video_name
            if args.resume and video_path.exists():
                continue

            generator = torch.Generator(device=args.device if args.device != "cpu" else "cpu").manual_seed(seed)
            call_kwargs = filtered_call_kwargs(
                pipe,
                {
                    "prompt": prompt,
                    "height": config.get("height"),
                    "width": config.get("width"),
                    "num_frames": config.get("num_frames"),
                    "num_inference_steps": config.get("num_inference_steps"),
                    "guidance_scale": config.get("guidance_scale"),
                    "frame_rate": fps,
                    "output_type": config.get("output_type", "pil"),
                    "generator": generator,
                },
            )
            call_kwargs = {key: value for key, value in call_kwargs.items() if value is not None}

            output = pipe(**call_kwargs)
            frames = extract_frames(output)
            export_to_video(frames, str(video_path), fps=fps, macro_block_size=1)

            row = {
                "video_path": str(video_path),
                "model": args.model,
                "model_path": config["model_path"],
                "pipeline_class": config["pipeline_class"],
                "prompt_file": str(prompt_file),
                "prompt_id": record["id"],
                "category": record.get("category", prompt_file.stem),
                "prompt_language": args.prompt_language,
                "prompt": prompt,
                "seed": seed,
                "height": config.get("height"),
                "width": config.get("width"),
                "num_frames": config.get("num_frames"),
                "num_inference_steps": config.get("num_inference_steps"),
                "guidance_scale": config.get("guidance_scale"),
                "fps": fps,
                "expected_objects": record.get("expected_objects", []),
                "expected_motion": record.get("expected_motion", ""),
                "expected_constraints": record.get("expected_constraints", []),
                "known_failure_modes": record.get("known_failure_modes", []),
            }
            write_jsonl(manifest_path, row)
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
