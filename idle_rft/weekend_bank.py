"""New compositional code tasks with independently evaluated hidden examples."""
import argparse
import concurrent.futures
import random
from pathlib import Path

from .common import atomic_json, digest, identity
from .verifier import verify


def evaluate(spec, values):
    """Trusted test oracle; deliberately uses a different execution strategy."""
    current = list(values)
    for op in spec:
        kind, a, b = op
        if kind == "keep_mod":
            current = [x for x in current if x % a != b]
        elif kind == "affine":
            current = [a * x + b for x in current]
        elif kind == "clip":
            current = [max(a, min(b, x)) for x in current]
        elif kind == "window":
            current = [sum(current[i:i + a]) for i in range(max(0, len(current) - a + 1))]
        elif kind == "pairs":
            current = [current[i + 1] - current[i] for i in range(len(current) - 1)]
        elif kind == "prefix":
            current = [sum(current[:i + 1]) for i in range(len(current))]
        elif kind == "sort_unique":
            current = sorted(set(current), reverse=bool(a))
        elif kind == "threshold":
            current = [int(x >= a) * b for x in current]
        elif kind == "index_add":
            current = [x + a * i + b for i, x in enumerate(current)]
        else:
            raise ValueError(kind)
    return current


def describe(op):
    kind, a, b = op
    return {
        "keep_mod": f"Remove each integer x for which x % {a} equals {b}; Python modulo semantics.",
        "affine": f"Replace every x by ({a}) * x + ({b}).",
        "clip": f"Clamp every value to the inclusive interval [{a}, {b}].",
        "window": f"Replace the sequence by sums of every full consecutive window of length {a}; no padding.",
        "pairs": "Replace the sequence by consecutive differences: next value minus previous value.",
        "prefix": "Replace the sequence by its inclusive cumulative sums.",
        "sort_unique": "Remove duplicates and sort the remaining values " + ("descending." if a else "ascending."),
        "threshold": f"Replace each value x by {b} when x >= {a}, and by 0 otherwise.",
        "index_add": f"At each zero-based position i, replace x by x + ({a}) * i + ({b}).",
    }[kind]


def reference(spec):
    code = ["def solve(xs):", "    v = list(xs)"]
    for kind, a, b in spec:
        if kind == "keep_mod":
            code.append(f"    v = [x for x in v if x % {a} != {b}]")
        elif kind == "affine":
            code.append(f"    v = [({a}) * x + ({b}) for x in v]")
        elif kind == "clip":
            code.append(f"    v = [min({b}, max({a}, x)) for x in v]")
        elif kind == "window":
            code.extend(["    w = []", f"    for i in range(len(v) - {a} + 1):",
                         f"        w.append(sum(v[i:i + {a}]))", "    v = w"])
        elif kind == "pairs":
            code.append("    v = [b - a for a, b in zip(v, v[1:])]")
        elif kind == "prefix":
            code.extend(["    w = []", "    total = 0", "    for x in v:",
                         "        total += x", "        w.append(total)", "    v = w"])
        elif kind == "sort_unique":
            code.append(f"    v = sorted(dict.fromkeys(v), reverse={bool(a)})")
        elif kind == "threshold":
            code.append(f"    v = [{b} if x >= {a} else 0 for x in v]")
        elif kind == "index_add":
            code.append(f"    v = [x + ({a}) * i + ({b}) for i, x in enumerate(v)]")
    return "\n".join(code + ["    return v"])


def make_spec(rng, family):
    mod = rng.randrange(2, 13)
    ops = [
        ("keep_mod", mod, rng.randrange(mod)), ("affine", rng.choice([-3, -2, 2, 3, 4]), rng.randrange(-9, 10)),
        ("clip", -rng.randrange(1, 21), rng.randrange(1, 21)),
        ("window", rng.randrange(2, 6), 0), ("pairs", 0, 0), ("prefix", 0, 0),
        ("sort_unique", rng.randrange(2), 0), ("threshold", rng.randrange(-10, 11), rng.randrange(1, 8)),
        ("index_add", rng.choice([-3, -2, -1, 1, 2, 3]), rng.randrange(-5, 6))]
    # Every family anchors a different operator. Remaining stages and their
    # parameters vary. Splits test new specifications within these same families.
    anchor = ops[family]
    selected = rng.sample([op for i, op in enumerate(ops) if i != family], rng.randrange(2, 5))
    selected.insert(rng.randrange(len(selected) + 1), anchor)
    return selected


def task(spec, split):
    key = identity(spec)
    rng = random.Random(int(key[:16], 16))
    tests = [[], [0], [1], [-1], [0, 0], list(range(-12, 13)),
             [3, -3, 3, -3, 0], [100, -100, 100, 0]]
    for n in range(1, 49):
        tests.append([rng.randrange(-25, 26) for _ in range(n % 17)])
    expected = [evaluate(spec, xs) for xs in tests]
    # Exclude only vacuous tasks whose oracle is identical across all examples.
    if all(result == expected[0] for result in expected):
        return None
    prompt = ("Implement solve(xs). xs is a list of integers. Apply these steps in the exact order below "
              "and return the final list. Each step consumes the result of the previous step. "
              "An empty sequence remains valid; consecutive differences and full windows on too short "
              "a sequence produce an empty list. Do not modify xs.\n"
              + "\n".join(f"{i + 1}. {describe(op)}" for i, op in enumerate(spec)))
    return {"task_id": "wk-" + key[:24], "prompt": prompt, "code": reference(spec),
            "test_list": [f"assert solve({xs!r}) == {out!r}" for xs, out in zip(tests, expected)],
            "interface": ["def solve(xs):"], "split": split, "group_id": key,
            "family": "pipeline_" + spec[0][0], "specification": spec,
            "mutation_killed": 0, "mutation_total": 0}


def build(destination, train=6144, dev=768, blind=1280, workers=8):
    destination = Path(destination)
    if (destination / "manifest.json").exists():
        raise ValueError("Dataset already frozen; never overwrite an experiment dataset")
    rng = random.Random(20261024)
    split_rows = {"train": [], "dev": [], "blind": []}
    seen = set()
    for split, count in zip(split_rows, (train, dev, blind)):
        while len(split_rows[split]) < count:
            spec = make_spec(rng, len(seen) % 9)
            key = identity(spec)
            if key in seen:
                continue
            seen.add(key)
            row = task(spec, split)
            if row is not None:
                split_rows[split].append(row)
    rows = [r for group in split_rows.values() for r in group]
    def audit(row):
        path = destination / "audits" / (row["task_id"] + ".json")
        if path.exists():
            from .common import read_json
            result = read_json(path)
        else:
            result = verify(row["code"], row["test_list"])
            atomic_json(path, result)
        if not result["passed"]:
            raise ValueError("Reference/oracle disagreement: " + row["task_id"])
        return row["task_id"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(audit, rows)):
            if i % 256 == 0:
                print({"audited": i + 1, "total": len(rows)}, flush=True)
    for split, entries in split_rows.items():
        atomic_json(destination / "private" / (split + ".oracles.json"), entries)
        atomic_json(destination / (split + ".prompts.json"),
                    [{k: r[k] for k in ("task_id", "prompt", "interface", "split", "group_id", "family")}
                     for r in entries])
    atomic_json(destination / "manifest.json", {
        "source": "programmatically generated compositional integer-sequence specifications",
        "seed": 20261024, "counts": {s: len(r) for s, r in split_rows.items()},
        "oracle_passed": len(rows), "tests_per_task": 56, "groups": len(rows),
        "implementation": digest(__file__),
        "dedup": "exact full operator/parameter specification hash",
        "limitations": ["Shared operator families across splits; measures generalization to new specifications.",
                        "Hand-authored task space; not a public code benchmark or SOTA claim.",
                        "Oracle and emitted reference use different implementations but share task semantics."],
        "files": {str(p.relative_to(destination)): digest(p)
                  for p in destination.rglob("*.json") if "audits" not in p.parts}})
    print({"frozen": True, "counts": {s: len(r) for s, r in split_rows.items()}}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", type=int, default=6144)
    parser.add_argument("--dev", type=int, default=768)
    parser.add_argument("--blind", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    build(args.output, args.train, args.dev, args.blind, args.workers)
