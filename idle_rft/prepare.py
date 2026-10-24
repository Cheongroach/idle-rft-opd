"""Audit real MBPP before freezing splits; no generated synthetic ground truth."""
import argparse
import ast
from collections import Counter
import re
from pathlib import Path

from .common import atomic_json, digest, identity, read_json
from .verifier import InfrastructureError, mutations, test_calls, verify


class Rename(ast.NodeTransformer):
    def __init__(self):
        self.names = {}

    def name(self, old):
        return self.names.setdefault(old, f"n{len(self.names)}")

    def visit_Name(self, node):
        node.id = self.name(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self.name(node.arg)
        return node

    def visit_FunctionDef(self, node):
        node.name = self.name(node.name)
        self.generic_visit(node)
        return node


def signatures(row):
    calls = test_calls(row["test_list"])
    functions = {n.name: ast.unparse(n.args) for n in ast.parse(row["code"]).body
                 if isinstance(n, ast.FunctionDef)}
    names = sorted({c["name"] for c in calls})
    if any(name not in functions for name in names):
        raise ValueError("Reference interface missing")
    return [f"def {name}({functions[name]}):" for name in names]


def prepare(source, metadata, destination):
    destination = Path(destination)
    provenance = read_json(metadata)
    if digest(source) != provenance["sha256"]:
        raise ValueError("MBPP source checksum mismatch")
    rows = read_json(source)
    if len(rows) != provenance["rows"]:
        raise ValueError("MBPP row count mismatch")
    # Durable per-task audit permits interruption during CPU preparation.
    audits = destination / "audits"
    kept, rejected = [], []
    for row in rows:
        audit_path = audits / f'{row["task_id"]}.json'
        contract = identity(row)
        if audit_path.exists():
            audit = read_json(audit_path)
            if audit["row_hash"] != contract:
                raise ValueError("Audit cache has different input")
        else:
            try:
                interface = signatures(row)
                result = verify(row["code"], row["test_list"])
                mutation_results = []
                if result["passed"]:
                    for code in mutations(row["code"]):
                        mutation_results.append(verify(code, row["test_list"]))
                audit = {"row_hash": contract, "oracle": result,
                         "interface": interface, "mutations": mutation_results}
            except (SyntaxError, ValueError) as exc:
                audit = {"row_hash": contract, "oracle": {
                    "passed": False, "status": "unsupported_task"}, "reason": str(exc)}
            # InfrastructureError deliberately aborts; it is not model failure.
            atomic_json(audit_path, audit)
        if audit["oracle"]["passed"]:
            kept.append({**row, "interface": audit["interface"],
                         "mutation_killed": sum(not m["passed"] for m in audit["mutations"]),
                         "mutation_total": len(audit["mutations"])})
        else:
            rejected.append({"task_id": row["task_id"], **audit["oracle"]})
    # Group exact normalized prompts and alpha-renamed reference programs.
    parent = list(range(len(kept)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(kept):
        keys = [("prompt", re.sub(r"\W+", " ", row["prompt"].lower()).strip()),
                ("program", ast.dump(Rename().visit(ast.parse(row["code"])), include_attributes=False))]
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i, row in enumerate(kept):
        groups.setdefault(root(i), []).append(row)
    groups = sorted(groups.values(), key=lambda g: identity(sorted(r["task_id"] for r in g)))
    targets = {"train": min(200, int(len(kept) * .50)), "dev": min(50, int(len(kept) * .12))}
    partitions = {"train": [], "dev": [], "blind": []}
    for group in groups:
        split = ("train" if len(partitions["train"]) < targets["train"] else
                 "dev" if len(partitions["dev"]) < targets["dev"] else "blind")
        for row in group:
            partitions[split].append({**row, "split": split,
                                      "group_id": identity(sorted(r["task_id"] for r in group))})
    for split, split_rows in partitions.items():
        atomic_json(destination / f"{split}.prompts.json", [
            {k: r[k] for k in ("task_id", "prompt", "interface", "split", "group_id")}
            for r in split_rows])
        atomic_json(destination / "private" / f"{split}.oracles.json", split_rows)
    manifest = {
        "source": provenance, "counts": {k: len(v) for k, v in partitions.items()},
        "groups": len(groups), "raw_tasks": len(rows), "rejected": rejected,
        "oracle_passed": len(kept),
        "mutation_tests": sum(r["mutation_total"] for r in kept),
        "mutation_killed": sum(r["mutation_killed"] for r in kept),
        "dedup": "normalized exact prompt + alpha-renamed exact program; not semantic dedup",
        "limitations": ["Public historical MBPP may be pretraining-contaminated.",
                         "Source assertions are sparse and not independent exhaustive tests.",
                         "Mutations are a basic sanity audit; no boundary coverage proof.",
                         "This is a pipeline pilot, not a SOTA benchmark."],
        "files": {str(p.relative_to(destination)): digest(p) for p in sorted(destination.rglob("*.json"))
                  if "audits" not in p.parts and p.name != "manifest.json"}}
    atomic_json(destination / "manifest.json", manifest)
    print(manifest["counts"], "oracle passed:", len(kept), "rejected:", len(rejected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.source, args.metadata, args.output)
