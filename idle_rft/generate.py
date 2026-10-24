"""One serial GPU owner; each complete candidate committed independently."""
import argparse
from pathlib import Path
import signal
import time

from .common import atomic_json, digest, gpu_session, identity, read_json
from .runtime import load, render


def generate(args):
    import torch
    rows = read_json(Path(args.data) / f"{args.split}.prompts.json")
    if args.split == "train" and args.adapter:
        raise ValueError("Shared training pool must come from the unadapted base")
    if args.split != "train" and args.samples != 1:
        raise ValueError("Dev/blind comparison is fixed single-candidate pass@1")
    if args.limit:
        if args.split != "train":
            raise ValueError("Do not run incomplete dev/blind evaluation")
        rows = rows[:args.limit]
    shard_index = getattr(args, "shard_index", 0)
    shard_count = getattr(args, "shard_count", 1)
    if not 0 <= shard_index < shard_count:
        raise ValueError("Invalid disjoint task shard")
    rows = rows[shard_index::shard_count]
    stage = "generate" if args.split == "train" else "evaluate"
    with gpu_session(args.run, stage, args.window_seconds) as window:
        signal.signal(signal.SIGTERM, window.stop)
        signal.signal(signal.SIGINT, window.stop)
        model, tokenizer, base_hash = load(args.model, args.model_manifest)
        adapter_hash = None
        training = None
        if args.adapter:
            from peft import PeftModel
            adapter = Path(args.adapter)
            contract = read_json(adapter.parent / "contract.json")
            if contract["base"] != base_hash:
                raise ValueError("Adapter has a different base model")
            posttraining = ((contract.get("arm"), contract.get("algorithm"))
                            in (("OPD", "opd"), ("REPLAY", "replay")))
            if contract.get("arm") not in ("A", "B") and not posttraining:
                raise ValueError("Evaluation requires an experimental adapter")
            files = read_json(adapter.parent / "files.json")
            for name, expected in files.items():
                checkpoint_file = (adapter.parent / name).resolve()
                if (not checkpoint_file.is_relative_to(adapter.parent.resolve())
                        or digest(checkpoint_file) != expected):
                    raise ValueError("Evaluation checkpoint checksum mismatch")
            progress = torch.load(adapter.parent / "state.pt", map_location="cpu",
                                  weights_only=True)["progress"]
            budget = contract["response_tokens"] if posttraining else contract["supervised_tokens"]
            if progress["tokens"] != budget:
                raise ValueError("Evaluate only a completed equal-token training arm")
            training = contract
            adapter_hash = identity({p.name: digest(p) for p in sorted(adapter.iterdir()) if p.is_file()})
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        protocol = {"base": base_hash, "adapter": adapter_hash, "split": args.split,
                    "training": training,
                    "data": digest(Path(args.data) / "manifest.json"),
                    "temperature": .7 if args.split == "train" else 0.,
                    "top_p": .95, "max_new_tokens": args.max_new_tokens,
                    "thinking": False, "prompt_version": 1}
        if args.split == "blind":
            if not args.selection:
                raise ValueError("Blind evaluation requires a frozen dev decision")
            decision = read_json(args.selection)
            approved = decision.get("chosen_adapters", [decision.get("chosen_adapter")])
            if (decision["data"] != protocol["data"] or decision["base"] != base_hash
                    or decision["max_new_tokens"] != args.max_new_tokens
                    or not decision["proceed"]
                    or adapter_hash not in (None, *approved)):
                raise ValueError("Blind run does not match the frozen dev decision")
            frozen = Path(args.run) / "blind-decision.json"
            if frozen.exists() and read_json(frozen) != decision:
                raise ValueError("A different blind decision is already frozen")
            if not frozen.exists():
                atomic_json(frozen, decision)
        namespace = Path(args.run) / "candidates" / identity(protocol)
        atomic_json(namespace / "protocol.json", protocol)
        model.eval()
        for row in rows:
            prompt = render(tokenizer, row)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            for index in range(args.samples):
                target = namespace / f'{row["task_id"]}.{index}.json'
                sample_key = identity([protocol, identity(prompt), row["task_id"], index])
                if target.exists():
                    if read_json(target)["key"] != sample_key:
                        raise ValueError("Candidate identity mismatch")
                    continue
                if not window.active:
                    atomic_json(Path(args.run) / "generation-status.json", {
                        "complete": False, "shard_index": shard_index, "shard_count": shard_count,
                        "expected_candidates": len(rows) * args.samples, "protocol": protocol})
                    return
                seed = int(identity([row["task_id"], index, 42])[:8], 16)
                torch.manual_seed(seed)
                started = time.monotonic()
                with torch.inference_mode():
                    sequence = model.generate(
                        **encoded, max_new_tokens=args.max_new_tokens,
                        do_sample=args.split == "train",
                        **({"temperature": .7, "top_p": .95} if args.split == "train" else {}),
                        use_cache=True, pad_token_id=tokenizer.pad_token_id)
                output_ids = sequence[0, encoded.input_ids.shape[1]:].tolist()
                raw = tokenizer.decode(output_ids, skip_special_tokens=True)
                code = extract(raw)
                eos = model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                record = {"key": sample_key, "task_id": row["task_id"], "index": index,
                          "protocol": protocol, "prompt_hash": identity(prompt),
                          "seed": seed, "raw": raw, "code": code,
                          "output_tokens": len(output_ids),
                          "truncated": len(output_ids) == args.max_new_tokens and output_ids[-1] not in eos,
                          "seconds": time.monotonic() - started}
                atomic_json(target, record)
                print({"task": row["task_id"], "candidate": index,
                       "tokens": len(output_ids), "truncated": record["truncated"]}, flush=True)
        atomic_json(Path(args.run) / "generation-status.json", {
            "complete": True, "shard_index": shard_index, "shard_count": shard_count,
            "expected_candidates": len(rows) * args.samples, "protocol": protocol})


def extract(text):
    text = text.strip()
    if text.startswith("```python\n") and text.endswith("\n```"):
        return text[len("```python\n"):-4].strip()
    if text.startswith("```\n") and text.endswith("\n```"):
        return text[4:-4].strip()
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=["train", "dev", "blind"], default="train")
    parser.add_argument("--samples", type=int, default=4, choices=[1, 2, 4])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--adapter")
    parser.add_argument("--selection")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--window-seconds", type=int, default=3600)
    generate(parser.parse_args())
