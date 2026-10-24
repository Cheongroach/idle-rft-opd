"""Validate and merge disjoint or resumed generation shards on the CPU."""
import argparse
from pathlib import Path
import shutil
import statistics

from .common import atomic_json, digest, identity, read_json


def merge(data, sources, output, split="train", samples=4, require_complete=True):
    data, output = Path(data), Path(output)
    rows = read_json(data / (split + ".prompts.json"))
    ids = {row["task_id"] for row in rows}
    expected = {(task, index) for task in ids for index in range(samples)}
    protocol = None
    candidates = {}
    duplicate_files = 0
    for source in map(Path, sources):
        for path in sorted(source.rglob("candidates/*/*.json")):
            if path.name == "protocol.json":
                continue
            record = read_json(path)
            current = record["protocol"]
            if (current["data"] != digest(data / "manifest.json") or current["split"] != split
                    or (split == "train" and current["adapter"] is not None)):
                raise ValueError("Shard dataset/split/base-pool contract mismatch")
            if protocol is None:
                protocol = current
            if protocol != current:
                raise ValueError("Mixed generation protocols")
            key = record["task_id"], record["index"]
            if key not in expected:
                raise ValueError("Unexpected task or sample in shard")
            if record["key"] != identity([current, record["prompt_hash"], *key]):
                raise ValueError("Candidate key does not match its sampling contract")
            if record["seed"] != int(identity([*key, 42])[:8], 16):
                raise ValueError("Candidate sampling seed mismatch")
            if key in candidates:
                if digest(candidates[key]) != digest(path):
                    raise ValueError("Conflicting retry candidate")
                duplicate_files += 1
                continue
            candidates[key] = path
    if not candidates:
        raise ValueError("No committed candidates collected")
    missing = expected - set(candidates)
    if missing and require_complete:
        raise ValueError(f"Frozen pool incomplete: {len(candidates)}/{len(expected)} candidates")
    namespace = output / "candidates" / identity(protocol)
    atomic_json(namespace / "protocol.json", protocol)
    records = []
    for key, path in sorted(candidates.items()):
        target = namespace / f"{key[0]}.{key[1]}.json"
        if target.exists() and digest(target) != digest(path):
            raise ValueError("Merged candidate changed")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        records.append(read_json(path))
    durations = sorted(record["seconds"] for record in records)
    summary = {"complete": not missing, "expected_candidates": len(expected),
               "candidates": len(records), "duplicate_files": duplicate_files,
               "missing_candidates": len(missing), "truncated": sum(r["truncated"] for r in records),
               "output_tokens": sum(r["output_tokens"] for r in records),
               "candidate_seconds_median": statistics.median(durations),
               "candidate_seconds_p95": durations[int((len(durations) - 1) * .95)],
               "sampling_gpu_seconds": sum(durations), "protocol": protocol,
               "candidate_directory": str(namespace), "sources": list(map(str, sources))}
    atomic_json(output / "merge.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "dev", "blind"), default="train")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    print(merge(args.data, args.source, args.output, args.split, args.samples,
                not args.allow_incomplete))
