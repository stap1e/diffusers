import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize video failure analysis judge results.")
    parser.add_argument("--results", nargs="+", required=True, help="Judge result JSONL files.")
    parser.add_argument("--output_dir", default="examples/video_failure_analysis/reports")
    return parser.parse_args()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values):
    return sum(values) / len(values) if values else None


def stddev(values):
    if not values:
        return None
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def main():
    args = parse_args()
    rows = []
    for path in args.results:
        rows.extend(read_jsonl(path))

    scored = [row for row in rows if row.get("score") is not None]
    by_model_category = defaultdict(list)
    by_model = defaultdict(list)
    by_prompt = defaultdict(list)
    tag_counts = defaultdict(Counter)
    tag_counts_by_model = defaultdict(Counter)
    categories = set()

    for row in scored:
        model = row.get("model", "unknown")
        category = row.get("category", "unknown")
        score = float(row["score"])
        by_model_category[(model, category)].append(score)
        by_model[model].append(score)
        by_prompt[(model, category, row.get("prompt_id", "unknown"))].append((row.get("seed"), score))
        categories.add(category)
        for tag in row.get("failure_tags", []):
            tag_counts[(model, category)][tag] += 1
            tag_counts_by_model[model][tag] += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    model_overview_path = output_dir / "model_overview.csv"
    seed_sensitivity_path = output_dir / "seed_sensitivity.csv"
    md_path = output_dir / "failure_profile.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "category", "num_samples", "mean_score", "top_failure_tags"])
        writer.writeheader()
        for model, category in sorted(by_model_category):
            scores = by_model_category[(model, category)]
            top_tags = tag_counts[(model, category)].most_common(5)
            writer.writerow(
                {
                    "model": model,
                    "category": category,
                    "num_samples": len(scores),
                    "mean_score": round(mean(scores), 4),
                    "top_failure_tags": "; ".join(f"{tag}:{count}" for tag, count in top_tags),
                }
            )

    category_columns = ["model", "num_samples", "overall_mean_score"] + [f"{category}_mean_score" for category in sorted(categories)]
    with model_overview_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=category_columns)
        writer.writeheader()
        for model in sorted(by_model):
            row = {
                "model": model,
                "num_samples": len(by_model[model]),
                "overall_mean_score": round(mean(by_model[model]), 4),
            }
            for category in sorted(categories):
                scores = by_model_category.get((model, category), [])
                row[f"{category}_mean_score"] = round(mean(scores), 4) if scores else ""
            writer.writerow(row)

    with seed_sensitivity_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "category",
                "prompt_id",
                "num_seeds",
                "mean_score",
                "std_score",
                "best_seed",
                "best_score",
                "worst_seed",
                "worst_score",
            ],
        )
        writer.writeheader()
        for model, category, prompt_id in sorted(by_prompt):
            seed_scores = by_prompt[(model, category, prompt_id)]
            ordered = sorted(seed_scores, key=lambda item: ((item[0] is None), item[0]))
            scores = [score for _, score in ordered]
            best_seed, best_score = max(ordered, key=lambda item: item[1])
            worst_seed, worst_score = min(ordered, key=lambda item: item[1])
            writer.writerow(
                {
                    "model": model,
                    "category": category,
                    "prompt_id": prompt_id,
                    "num_seeds": len(ordered),
                    "mean_score": round(mean(scores), 4),
                    "std_score": round(stddev(scores), 4),
                    "best_seed": best_seed,
                    "best_score": round(best_score, 4),
                    "worst_seed": worst_seed,
                    "worst_score": round(worst_score, 4),
                }
            )

    lines = ["# Video Failure Analysis Summary", "", f"Scored videos: {len(scored)}", ""]
    lines.extend(["## Overall model scores", "", "| Model | Samples | Mean score | Top failure tags |", "|---|---:|---:|---|"])
    for model in sorted(by_model):
        scores = by_model[model]
        top_tags = ", ".join(f"{tag} ({count})" for tag, count in tag_counts_by_model[model].most_common(5)) or "-"
        lines.append(f"| {model} | {len(scores)} | {mean(scores):.2f} | {top_tags} |")

    lines.extend(["", "## Category gap", "", "| Model | Category | Samples | Mean score | Top failure tags |", "|---|---|---:|---:|---|"])
    for model, category in sorted(by_model_category):
        scores = by_model_category[(model, category)]
        top_tags = tag_counts[(model, category)].most_common(5)
        tag_text = ", ".join(f"{tag} ({count})" for tag, count in top_tags) or "-"
        lines.append(f"| {model} | {category} | {len(scores)} | {mean(scores):.2f} | {tag_text} |")

    lines.extend(
        [
            "",
            "## Seed sensitivity",
            "",
            "| Model | Category | Prompt ID | Seeds | Mean score | Std score | Best seed | Worst seed |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model, category, prompt_id in sorted(by_prompt):
        seed_scores = by_prompt[(model, category, prompt_id)]
        scores = [score for _, score in seed_scores]
        best_seed, _ = max(seed_scores, key=lambda item: item[1])
        worst_seed, _ = min(seed_scores, key=lambda item: item[1])
        lines.append(
            f"| {model} | {category} | {prompt_id} | {len(seed_scores)} | {mean(scores):.2f} | {stddev(scores):.2f} | {best_seed} | {worst_seed} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {model_overview_path}")
    print(f"Wrote {seed_sensitivity_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
