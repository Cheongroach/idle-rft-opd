"""Freeze one common learning rate using every declared seed and both arms."""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics

from .common import atomic_json, digest, read_json
from .judge import paired

SEEDS = (42, 43, 44)
LRS = (5e-5, 1e-4, 2e-4)


def decide(base_path, report_paths, output):
    base = read_json(base_path)
    if base["protocol"]["split"] != "dev" or base["protocol"]["adapter"] is not None:
        raise ValueError("Require a base-model development evaluation")
    matrix = {}
    recipes = []
    for path in map(Path, report_paths):
        report = read_json(path)
        recipe = dict(report["protocol"]["training"])
        key = recipe["arm"], recipe["seed"], recipe["lr"]
        if key in matrix:
            raise ValueError("Repeated arm/seed/learning-rate report")
        comparison = paired(base, report)
        matrix[key] = {"path": str(path), "sha256": digest(path), "comparison": comparison,
                       "adapter": report["protocol"]["adapter"]}
        for name in ("arm", "seed", "lr", "examples"):
            recipe.pop(name)
        recipes.append(recipe)
    expected = {(arm, seed, lr) for arm in ("A", "B") for seed in SEEDS for lr in LRS}
    if set(matrix) != expected:
        raise ValueError("Require all 18 declared development configurations")
    if any(recipe != recipes[0] for recipe in recipes):
        raise ValueError("Configurations differ beyond arm/seed/learning-rate")
    scores = {lr: statistics.mean(matrix[arm, seed, lr]["comparison"]["adapter_pass_at_1"]
                                  for arm in ("A", "B") for seed in SEEDS) for lr in LRS}
    chosen_lr = max(LRS, key=lambda lr: (scores[lr], -lr))
    by_arm = {}
    for arm in ("A", "B"):
        comparisons = [matrix[arm, seed, chosen_lr]["comparison"] for seed in SEEDS]
        by_arm[arm] = {"mean_pass_at_1": statistics.mean(c["adapter_pass_at_1"] for c in comparisons),
                       "mean_delta": statistics.mean(c["delta"] for c in comparisons),
                       "seeds": {str(seed): matrix[arm, seed, chosen_lr] for seed in SEEDS}}
    chosen_arm = max(("A", "B"), key=lambda arm: by_arm[arm]["mean_pass_at_1"])
    result = {"base": base["protocol"]["base"], "data": base["protocol"]["data"],
              "max_new_tokens": base["protocol"]["max_new_tokens"], "chosen_lr": chosen_lr,
              "chosen_arm": chosen_arm,
              "chosen_adapters": [matrix[arm, seed, chosen_lr]["adapter"]
                                  for arm in ("A", "B") for seed in SEEDS],
              "proceed": by_arm[chosen_arm]["mean_delta"] > 0,
              "lr_mean_pass_at_1": {str(lr): score for lr, score in scores.items()},
              "comparisons": by_arm, "base_report": {"path": str(base_path), "sha256": digest(base_path)},
              "all_reports": [matrix[key] for key in sorted(matrix)],
              "selection_rule": "Common lr by mean of six arm/seed dev scores; ties choose lower lr. "
                                "Blind compares all three seeds of both arms at that lr; never choose a seed.",
              "stop_rule": "Both mean dev gains <=0 => no blind evaluation."}
    output = Path(output)
    if output.exists() and read_json(output) != result:
        raise ValueError("Development decision already frozen")
    if not output.exists():
        atomic_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(decide(args.base, args.report, args.output))
