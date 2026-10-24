"""CPU grading and a predeclared random-vs-low-success selection comparison."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import random

from .common import atomic_json, digest, identity, read_json
from .verifier import verify


def grade(data, candidates, output, workers=1):
    candidates, output = Path(candidates), Path(output)
    if not 1 <= workers <= 64:
        raise ValueError("Use 1–64 bounded CPU grading workers")
    protocol = read_json(candidates / "protocol.json")
    if protocol["data"] != digest(Path(data) / "manifest.json"):
        raise ValueError("Candidate dataset revision mismatch")
    oracles = read_json(Path(data) / "private" / f'{protocol["split"]}.oracles.json')
    by_id = {r["task_id"]: r for r in oracles}
    version = digest(Path(__file__).with_name("verifier.py"))
    def grade_one(path):
        record = read_json(path)
        row = by_id[record["task_id"]]
        if record["protocol"] != protocol:
            raise ValueError("Mixed candidate protocols")
        key = identity([digest(path), identity(row), version])
        cache = output / "grades" / f"{key}.json"
        if cache.exists():
            result = read_json(cache)["result"]
        else:
            result = ({"status": "truncated", "passed": False} if record["truncated"] else
                      verify(record["code"], row["test_list"]))
            atomic_json(cache, {"result": result, "verifier": version})
        return {"task_id": row["task_id"], "group_id": row["group_id"],
                "index": record["index"], "code": record["code"],
                "candidate_key": record["key"], "result": result,
                "mutation_killed": row["mutation_killed"],
                "mutation_total": row["mutation_total"]}
    paths = [p for p in sorted(candidates.glob("*.json")) if p.name != "protocol.json"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(grade_one, paths))
    report = {"protocol": protocol, "verifier": version, "records": records,
              "expected_tasks": len(oracles), "tasks": len({r["task_id"] for r in records}),
              "candidates": len(records), "passed": sum(r["result"]["passed"] for r in records)}
    atomic_json(output / "report.json", report)
    return report


def family(row):
    prompt = row["prompt"].lower()
    if any(k in prompt for k in ("string", "character", "word", "text")):
        return "string"
    if any(k in prompt for k in ("list", "tuple", "array", "dictionary")):
        return "container"
    return "numeric_other"


def select(data, report_path, output):
    report = read_json(report_path)
    if report["protocol"]["split"] != "train" or report["protocol"]["adapter"] is not None:
        raise ValueError("Selection must use only shared base-model train candidates")
    rows = read_json(Path(data) / "private" / "train.oracles.json")
    if report["protocol"]["data"] != digest(Path(data) / "manifest.json"):
        raise ValueError("Selection dataset mismatch")
    by_id = defaultdict(list)
    for record in report["records"]:
        by_id[record["task_id"]].append(record)
    for task, records in by_id.items():
        if sorted(r["index"] for r in records) != [0, 1, 2, 3]:
            raise ValueError("Selection requires all four candidates for every task")
    if set(by_id) != {r["task_id"] for r in rows}:
        raise ValueError("Finish the entire frozen training pool before selection")
    eligible = []
    for row in rows:
        good = [r for r in by_id[row["task_id"]] if r["result"]["passed"]]
        if good:
            eligible.append((row, good, len(good) / 4))
    if len(eligible) < 100:
        raise RuntimeError(f"Stop self-generated RFT: only {len(eligible)} eligible tasks (<100)")
    buckets = defaultdict(list)
    for item in eligible:
        buckets[family(item[0])].append(item)
    rng = random.Random(42)
    arms = {"A": [], "B": []}
    for category, items in sorted(buckets.items()):
        quota = max(1, math.floor(len(items) * .7))
        shuffled = list(items)
        rng.shuffle(shuffled)
        difficult = sorted(items, key=lambda item: (
            item[2], -item[0]["mutation_killed"], identity([item[0]["task_id"], 42])))
        for arm, selected in (("A", shuffled[:quota]), ("B", difficult[:quota])):
            for row, correct, success in selected:
                candidate = correct[rng.randrange(len(correct))]
                arms[arm].append({k: row[k] for k in ("task_id", "prompt", "interface", "group_id")})
                arms[arm][-1].update(code=candidate["code"], family=category,
                                     pool_success=success, candidate_key=candidate["candidate_key"])
    for arm, examples in arms.items():
        rng.shuffle(examples)
        atomic_json(Path(output) / f"{arm}.json", examples)
    atomic_json(Path(output) / "selection.json", {
        "base": report["protocol"]["base"], "data": report["protocol"]["data"],
        "candidate_report": digest(report_path), "eligible_tasks": len(eligible),
        "arms": {k: len(v) for k, v in arms.items()},
        "variable": "task selection: stratified random vs low base success, mutation-count tiebreak",
        "boundary_coverage_claim": False,
        "files": {f"{a}.json": digest(Path(output) / f"{a}.json") for a in arms}})


def paired(base, adapted, trials=5000):
    if base["tasks"] != base["expected_tasks"] or adapted["tasks"] != base["expected_tasks"]:
        raise ValueError("Incomplete evaluation")
    a = {r["task_id"]: r for r in base["records"]}
    b = {r["task_id"]: r for r in adapted["records"]}
    if len(a) != len(base["records"]) or len(b) != len(adapted["records"]) or set(a) != set(b):
        raise ValueError("Require one candidate per paired task")
    p, q = dict(base["protocol"]), dict(adapted["protocol"])
    p.pop("adapter")
    q.pop("adapter")
    p.pop("training", None)
    q.pop("training", None)
    if p != q or base["verifier"] != adapted["verifier"]:
        raise ValueError("Evaluation protocols differ")
    groups = defaultdict(list)
    wins = losses = 0
    for task, row in a.items():
        before, after = int(row["result"]["passed"]), int(b[task]["result"]["passed"])
        groups[row["group_id"]].append(after - before)
        wins += after > before
        losses += after < before
    clusters = list(groups.values())
    rng = random.Random(42)
    differences = []
    for _ in range(trials):
        chosen = [rng.choice(clusters) for _ in clusters]
        differences.append(sum(map(sum, chosen)) / sum(map(len, chosen)))
    differences.sort()
    return {"n": len(a), "base_pass_at_1": sum(r["result"]["passed"] for r in a.values()) / len(a),
            "adapter_pass_at_1": sum(r["result"]["passed"] for r in b.values()) / len(b),
            "delta": (wins - losses) / len(a), "wins": wins, "losses": losses,
            "ci95": [differences[int(trials * .025)], differences[int(trials * .975)]],
            "method": "paired bootstrap over frozen duplicate groups, seed42",
            "training_seeds": 1}


def decide(base_path, arm_a_path, arm_b_path, output):
    base, a, b = (read_json(p) for p in (base_path, arm_a_path, arm_b_path))
    if base["protocol"]["split"] != "dev" or base["protocol"]["adapter"] is not None:
        raise ValueError("Decision requires a complete base-model dev report")
    recipes = []
    for report, arm in ((a, "A"), (b, "B")):
        recipe = dict(report["protocol"]["training"])
        if recipe.pop("arm") != arm:
            raise ValueError("Wrong training arm")
        recipe.pop("examples")
        recipes.append(recipe)
    if recipes[0] != recipes[1]:
        raise ValueError("Training recipes or supervised token budgets differ")
    comparisons = {"A": paired(base, a), "B": paired(base, b)}
    chosen = max(("A", "B"), key=lambda arm: comparisons[arm]["delta"])
    gain = comparisons[chosen]["delta"]
    adapted = a if chosen == "A" else b
    decision = {"base": base["protocol"]["base"], "data": base["protocol"]["data"],
                        "max_new_tokens": base["protocol"]["max_new_tokens"],
                        "comparisons": comparisons, "chosen_arm": chosen,
                        "chosen_adapter": adapted["protocol"]["adapter"],
                        "proceed": gain > 0,
                        "reports": {str(p): digest(p) for p in (base_path, arm_a_path, arm_b_path)},
                "stop_rule": "Both dev gains <=0 => stop without blind tuning."}
    if Path(output).exists():
        if read_json(output) != decision:
            raise ValueError("Dev decision is already frozen")
        return
    atomic_json(output, decision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    grading = sub.add_parser("grade")
    grading.add_argument("--data", required=True)
    grading.add_argument("--candidates", required=True)
    grading.add_argument("--output", required=True)
    grading.add_argument("--workers", type=int, default=1)
    selecting = sub.add_parser("select")
    selecting.add_argument("--data", required=True)
    selecting.add_argument("--report", required=True)
    selecting.add_argument("--output", required=True)
    comparing = sub.add_parser("compare")
    comparing.add_argument("--base", required=True)
    comparing.add_argument("--adapter", required=True)
    comparing.add_argument("--output", required=True)
    deciding = sub.add_parser("decide")
    deciding.add_argument("--base", required=True)
    deciding.add_argument("--arm-a", required=True)
    deciding.add_argument("--arm-b", required=True)
    deciding.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "grade":
        result = grade(args.data, args.candidates, args.output, args.workers)
        print({k: result[k] for k in ("tasks", "candidates", "passed")})
    elif args.command == "select":
        select(args.data, args.report, args.output)
    elif args.command == "compare":
        result = paired(read_json(args.base), read_json(args.adapter))
        atomic_json(args.output, result)
        print(result)
    else:
        decide(args.base, args.arm_a, args.arm_b, args.output)
