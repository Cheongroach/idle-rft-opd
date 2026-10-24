"""Single-GPU answer-only LoRA, exact supervised-token budgets and recovery."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import random
import signal
import time
import uuid

from .common import atomic_json, digest, gpu_session, identity, read_json
from .runtime import load, render


def encode(tokenizer, examples, max_length):
    encoded, excluded = [], []
    for row in examples:
        prompt = tokenizer.encode(render(tokenizer, row), add_special_tokens=False)
        answer = tokenizer.encode(row["code"], add_special_tokens=False)
        answer += [tokenizer.eos_token_id]
        # Explicit segmented protocol used by training; never truncate solutions.
        ids = prompt + answer
        if len(ids) > max_length:
            excluded.append(row["task_id"])
            continue
        labels = [-100] * len(prompt) + answer
        encoded.append({"ids": ids, "labels": labels, "task_id": row["task_id"],
                        "tokens": len(answer)})
    return encoded, excluded


def plan_batch(examples, cursor, tokens, budget, accumulation):
    batch, next_cursor, next_tokens = [], cursor, tokens
    while next_cursor < len(examples) and next_tokens < budget and len(batch) < accumulation:
        example = examples[next_cursor]
        take = min(example["tokens"], budget - next_tokens)
        labels = list(example["labels"])
        active = [i for i, value in enumerate(labels) if value != -100]
        for i in active[take:]:
            labels[i] = -100
        batch.append({**example, "labels": labels, "tokens": take})
        next_cursor += 1
        next_tokens += take
    return batch, next_cursor, next_tokens


def update(model, optimizer, scheduler, batch, device):
    import torch
    import torch.nn.functional as functional
    optimizer.zero_grad(set_to_none=True)
    denominator = sum(r["tokens"] for r in batch)
    loss_sum = 0.
    for row in batch:
        inputs = torch.tensor([row["ids"]], dtype=torch.long, device=device)
        labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
        logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
        loss = functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum") / denominator
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        loss_sum += loss.detach().item()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                  1., error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return loss_sum


def cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(v) for v in value)
    return value


def save(root, model, optimizer, scheduler, progress, contract):
    import torch
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f'checkpoint-{progress["step"]:06d}-{uuid.uuid4().hex[:8]}'
    temporary, destination = root / (name + ".partial"), root / name
    temporary.mkdir()
    model.save_pretrained(temporary / "adapter", safe_serialization=True)
    atomic_json(temporary / "contract.json", contract)
    torch.save({"optimizer": cpu(optimizer.state_dict()), "scheduler": scheduler.state_dict(),
                "progress": progress, "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               temporary / "state.pt")
    files = {str(p.relative_to(temporary)): digest(p)
             for p in sorted(temporary.rglob("*")) if p.is_file()}
    atomic_json(temporary / "files.json", files)
    for path in temporary.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    os.replace(temporary, destination)
    atomic_json(root / "latest.json", {"directory": name, "files": digest(destination / "files.json")})
    return destination


def restore(root, model, optimizer, scheduler, contract):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    pointer = read_json(Path(root) / "latest.json")
    checkpoint = (Path(root) / pointer["directory"]).resolve()
    if not checkpoint.is_relative_to(Path(root).resolve()):
        raise ValueError("Unsafe checkpoint path")
    if digest(checkpoint / "files.json") != pointer["files"]:
        raise ValueError("Checkpoint manifest mismatch")
    files = read_json(checkpoint / "files.json")
    for name, expected in files.items():
        path = (checkpoint / name).resolve()
        if not path.is_relative_to(checkpoint) or digest(path) != expected:
            raise ValueError("Checkpoint file checksum mismatch")
    if read_json(checkpoint / "contract.json") != contract:
        raise ValueError("Resume base/data/token/optimizer contract mismatch")
    adapter_state = load_file(checkpoint / "adapter" / "adapter_model.safetensors")
    if set(adapter_state) != set(get_peft_model_state_dict(model)):
        raise ValueError("Missing or extra adapter tensor names")
    result = set_peft_model_state_dict(model, adapter_state)
    if result.unexpected_keys:
        raise ValueError("Unexpected adapter keys")
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint cannot resume on CPU")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["progress"]


def train(args):
    import torch
    phase = "preflight" if args.preflight else "train"
    with gpu_session(args.run, phase, args.window_seconds) as window:
        signal.signal(signal.SIGTERM, window.stop)
        signal.signal(signal.SIGINT, window.stop)
        model, tokenizer, base = load(args.model, args.model_manifest, rank=args.rank, seed=args.seed)
        prepared = Path(args.prepared)
        selection = read_json(prepared / "selection.json")
        if selection["base"] != base:
            raise ValueError("Candidate generation used a different model")
        for name, expected in selection["files"].items():
            if digest(prepared / name) != expected:
                raise ValueError("Selected data checksum mismatch")
        arms = {}
        exclusions = {}
        for arm in ("A", "B"):
            arms[arm], exclusions[arm] = encode(tokenizer, read_json(prepared / f"{arm}.json"), args.max_length)
            expanded = []
            for epoch in range(args.epochs):
                examples = list(arms[arm])
                random.Random(args.seed + epoch).shuffle(examples)
                expanded.extend(examples)
            arms[arm] = expanded
        budget = min(args.token_budget, *(sum(r["tokens"] for r in arms[a]) for a in arms))
        if budget <= 0:
            raise ValueError("No complete solution fits the sequence budget")
        contract = {"base": base, "data": digest(prepared / "selection.json"), "arm": args.arm,
                    "rank": args.rank, "lr": args.lr, "seed": args.seed,
                    "epochs": args.epochs, "accumulation": args.accumulation,
                    "supervised_tokens": budget, "max_length": args.max_length,
                    "sequence_protocol": "segmented chat prefix + code + EOS; answer loss only",
                    "examples": identity(arms[args.arm]),
                    "implementation": {name: digest(Path(__file__).with_name(name))
                                       for name in ("train.py", "runtime.py")}}
        root = Path(args.run) / ("preflight" if args.preflight else "training") / args.arm
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=args.lr, weight_decay=0.01)
        steps = 0
        cursor = tokens = 0
        while tokens < budget:
            _, cursor, tokens = plan_batch(arms[args.arm], cursor, tokens, budget, args.accumulation)
            steps += 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1 - step / steps))
        progress = {"cursor": 0, "tokens": 0, "step": 0, "loss": None}
        if (root / "latest.json").exists():
            progress = restore(root, model, optimizer, scheduler, contract)
        model.train()
        saved_at = time.monotonic()
        checkpoint = None
        while progress["tokens"] < budget and window.active:
            batch, cursor, tokens = plan_batch(arms[args.arm], progress["cursor"], progress["tokens"],
                                               budget, args.accumulation)
            if not batch:
                raise RuntimeError("Training plan unexpectedly exhausted")
            loss = update(model, optimizer, scheduler, batch, "cuda")
            progress = {"cursor": cursor, "tokens": tokens, "step": progress["step"] + 1, "loss": loss}
            print(progress, flush=True)
            if time.monotonic() - saved_at >= 600 or args.preflight:
                checkpoint = save(root, model, optimizer, scheduler, progress, contract)
                saved_at = time.monotonic()
            if args.preflight and progress["step"] >= 2:
                break
        checkpoint = save(root, model, optimizer, scheduler, progress, contract)
        atomic_json(root / "status.json", {"progress": progress, "contract": contract,
                    "complete": progress["tokens"] == budget, "checkpoint": checkpoint.name,
                    "exclusions": exclusions, "peak_cuda_bytes": torch.cuda.max_memory_allocated()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--arm", required=True, choices=["A", "B"])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2560)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=100000)
    parser.add_argument("--window-seconds", type=int, default=3600)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 4:
        raise ValueError("Freeze an epoch count between 1 and 4")
    train(args)
