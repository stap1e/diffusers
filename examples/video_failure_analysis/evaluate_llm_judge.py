import argparse
import json
from pathlib import Path

import yaml


DEFAULT_JUDGES = {
    "physics": "examples/cosmos/llm_judge_prompts/video_physics.yaml",
    "instruction_following": "examples/cosmos/llm_judge_prompts/video_IF.yaml",
    "temporal_consistency": "examples/video_failure_analysis/judge_prompts/video_temporal_consistency.yaml",
    "multi_object_interaction": "examples/video_failure_analysis/judge_prompts/video_interaction.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare video judge requests or normalize judge annotations.")
    parser.add_argument("--manifest", required=True, help="Manifest JSONL produced by generate.py.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--judge_prompt", help="YAML judge prompt. Defaults by category when omitted.")
    parser.add_argument("--annotations", help="Optional JSONL annotations to normalize into result rows.")
    return parser.parse_args()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_user_prompt(template, row):
    instruction = row.get("prompt", "")
    try:
        return template.format(instruction=instruction, **row)
    except KeyError:
        return template.replace("{instruction}", instruction)


def request_rows(manifest, judge_prompt):
    rows = []
    prompt_cache = {}
    for row in manifest:
        judge_path = judge_prompt or DEFAULT_JUDGES.get(row.get("category"))
        if judge_path is None:
            raise ValueError(f"No judge prompt configured for category '{row.get('category')}'")
        if judge_path not in prompt_cache:
            prompt_cache[judge_path] = load_yaml(judge_path)
        template = prompt_cache[judge_path]
        rows.append(
            {
                "request_id": f"{row['model']}::{row['prompt_id']}::seed{row['seed']}::{row.get('prompt_language', 'en')}",
                "video_path": row["video_path"],
                "model": row["model"],
                "model_path": row.get("model_path"),
                "prompt_id": row["prompt_id"],
                "category": row.get("category"),
                "seed": row["seed"],
                "prompt": row.get("prompt"),
                "judge_prompt": judge_path,
                "system_prompt": template.get("system_prompt", ""),
                "user_prompt": format_user_prompt(template.get("user_prompt", ""), row),
                "known_failure_modes": row.get("known_failure_modes", []),
            }
        )
    return rows


def normalize_annotations(manifest, annotations):
    by_key = {}
    for row in manifest:
        request_id = f"{row['model']}::{row['prompt_id']}::seed{row['seed']}::{row.get('prompt_language', 'en')}"
        by_key[request_id] = row

    rows = []
    for annotation in annotations:
        request_id = annotation.get("request_id")
        source = by_key.get(request_id, {})
        score = annotation.get("score")
        if score is not None:
            score = int(score)
        rows.append(
            {
                "request_id": request_id,
                "video_path": annotation.get("video_path", source.get("video_path")),
                "model": annotation.get("model", source.get("model")),
                "model_path": annotation.get("model_path", source.get("model_path")),
                "prompt_id": annotation.get("prompt_id", source.get("prompt_id")),
                "category": annotation.get("category", source.get("category")),
                "seed": annotation.get("seed", source.get("seed")),
                "score": score,
                "reason": annotation.get("reason", ""),
                "failure_tags": annotation.get("failure_tags", []),
                "raw_response": annotation.get("raw_response", annotation.get("response", "")),
            }
        )
    return rows


def main():
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    if args.annotations:
        rows = normalize_annotations(manifest, read_jsonl(args.annotations))
    else:
        rows = request_rows(manifest, args.judge_prompt)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
