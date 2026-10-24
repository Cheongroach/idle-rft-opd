"""Synchronous student-prefix distillation and an equal-token replay control."""
import argparse
import copy
from pathlib import Path
import random
import signal

from .common import atomic_json, digest, gpu_session, identity, read_json
from .runtime import load, render
from .train import encode, restore, save


def checkpoint_info(adapter, base):
    """Require a completed, checksummed checkpoint before loading its adapter."""
    import torch
    root = Path(adapter).parent.resolve()
    contract = read_json(root / "contract.json")
    if contract["base"] != base:
        raise ValueError("Adapter base/tokenizer provenance mismatch")
    for name, expected in read_json(root / "files.json").items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError("Adapter checkpoint checksum mismatch")
    progress = torch.load(root / "state.pt", map_location="cpu", weights_only=True)["progress"]
    budget = contract.get("response_tokens", contract.get("supervised_tokens"))
    if budget is None or progress["tokens"] != budget:
        raise ValueError("Adapter checkpoint training is incomplete")
    fingerprint = identity({p.name: digest(p) for p in sorted(Path(adapter).iterdir())
                            if p.is_file()})
    return contract, fingerprint


def response_positions(prompt_length, count):
    if prompt_length < 1 or count < 1:
        raise ValueError("Need a nonempty prompt and response")
    # The logit immediately before each response token predicts that token.
    return slice(prompt_length - 1, prompt_length + count - 1)


def topk_reverse_divergence(student_logits, teacher_logps, token_ids):
    """Generalized reverse divergence on unnormalized teacher top-k support.

    Neither distribution is renormalized on the selected support. The -p+q
    correction makes every summand nonnegative, with zero gradient at p=q.
    This is a truncated approximation, not full-vocabulary reverse KL.
    """
    import torch
    logp = torch.log_softmax(student_logits.float(), dim=-1).gather(-1, token_ids)
    logq = teacher_logps.detach().float()
    p, q = logp.exp(), logq.exp()
    terms = p * (logp - logq) - p + q
    return terms.sum(-1), p.sum(-1), q.sum(-1)


def next_route(tokens, targets):
    """Serve the most underfilled route; all routes have exact final quotas."""
    remaining = [name for name in targets if tokens[name] < targets[name]]
    if not remaining:
        return None
    return min(remaining, key=lambda name: (tokens[name] / targets[name], name))


def row_at(rows, cursor, seed):
    epoch, index = divmod(cursor, len(rows))
    order = list(range(len(rows)))
    random.Random(seed + epoch).shuffle(order)
    return rows[order[index]]


def parameter_fingerprint(model):
    import hashlib
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            result.update(name.encode())
            result.update(parameter.detach().float().cpu().numpy().tobytes())
    return result.hexdigest()


def warm_student(args, path):
    import torch
    from peft import PeftModel
    base, tokenizer, fingerprint = load(args.model, args.model_manifest, seed=args.seed)
    contract, adapter_hash = checkpoint_info(path, fingerprint)
    if contract.get("rank") != 16:
        raise ValueError("This experiment requires the frozen rank-16 warm start")
    student = PeftModel.from_pretrained(base, path, is_trainable=True)
    for name, parameter in student.named_parameters():
        if parameter.requires_grad:
            if ".lora_" not in name:
                raise ValueError("Only LoRA parameters may be updated")
            parameter.data = parameter.data.float()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not any(p.requires_grad for p in student.parameters()):
        raise ValueError("No trainable student parameters")
    return student, tokenizer, fingerprint, adapter_hash


def teacher_model(args, routes, fingerprint, tokenizer):
    from peft import PeftModel
    base, other_tokenizer, other_hash = load(args.model, args.model_manifest, seed=args.seed)
    if other_hash != fingerprint or other_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("Direct token scoring requires identical model assets and vocabulary")
    teacher = None
    hashes = {}
    for route, path in routes.items():
        _, hashes[route] = checkpoint_info(path, fingerprint)
        if teacher is None:
            teacher = PeftModel.from_pretrained(base, path, adapter_name=route, is_trainable=False)
        else:
            teacher.load_adapter(path, adapter_name=route, is_trainable=False)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher, hashes


def batch_rows(args, student, tokenizer, rows, progress, targets):
    import torch
    planned = copy.deepcopy(progress)
    batch = []
    student.eval()
    while len(batch) < args.accumulation:
        route = next_route(planned["route_tokens"], targets)
        if route is None:
            break
        row = row_at(rows[route], planned["route_cursor"][route], args.seed)
        left = targets[route] - planned["route_tokens"][route]
        if args.algorithm == "opd":
            prompt = tokenizer.encode(render(tokenizer, row), add_special_tokens=False)
            limit = min(args.rollout_tokens, left, args.max_length - len(prompt))
            if limit < 1:
                raise ValueError("Frozen prompt leaves no room for student rollout")
            inputs = torch.tensor([prompt], device="cuda")
            with torch.no_grad():
                sequence = student.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    do_sample=True, temperature=1., top_p=1., top_k=0,
                    max_new_tokens=limit, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id)
            ids = sequence[0].tolist()
            count = len(ids) - len(prompt)
            if count < 1:
                raise ValueError("Student produced no response tokens")
            item = {"ids": ids, "prompt_length": len(prompt), "tokens": count,
                    "prefix_at_cap": count == limit}
        else:
            item = dict(row)
            count = min(item["tokens"], left)
            active = [i for i, token in enumerate(item["labels"]) if token != -100]
            item["labels"] = list(item["labels"])
            for position in active[count:]:
                item["labels"][position] = -100
            item["tokens"] = count
        batch.append({**item, "route": route})
        planned["route_tokens"][route] += count
        planned["route_cursor"][route] += 1
        planned["tokens"] += count
    return batch, planned


def update(args, student, teacher, optimizer, batch):
    import torch
    import torch.nn.functional as functional
    denominator = sum(row["tokens"] for row in batch)
    optimizer.zero_grad(set_to_none=True)
    metrics = {"loss": 0., "teacher_mass": 0., "student_mass": 0.,
               "sample_outside_topk": 0., "prefixes_at_cap": 0}
    student.train()
    for row in batch:
        inputs = torch.tensor([row["ids"]], device="cuda")
        mask = torch.ones_like(inputs)
        if args.algorithm == "opd":
            positions = response_positions(row["prompt_length"], row["tokens"])
            teacher.set_adapter(row["route"])
            teacher.requires_grad_(False)
            with torch.no_grad():
                logits = teacher(input_ids=inputs, attention_mask=mask, use_cache=False).logits
                logps = torch.log_softmax(logits[0, positions].float(), dim=-1)
                teacher_logps, token_ids = logps.topk(args.topk, dim=-1)
                del logits, logps
            logits = student(input_ids=inputs, attention_mask=mask, use_cache=False).logits
            terms, student_mass, teacher_mass = topk_reverse_divergence(
                logits[0, positions], teacher_logps, token_ids)
            loss = terms.sum() / denominator
            generated = inputs[0, row["prompt_length"]:]
            outside = ~(token_ids == generated.unsqueeze(-1)).any(-1)
            metrics["teacher_mass"] += teacher_mass.sum().detach().item() / denominator
            metrics["student_mass"] += student_mass.sum().detach().item() / denominator
            metrics["sample_outside_topk"] += outside.sum().item() / denominator
            metrics["prefixes_at_cap"] += int(row["prefix_at_cap"])
        else:
            labels = torch.tensor([row["labels"]], device="cuda")
            logits = student(input_ids=inputs, attention_mask=mask, use_cache=False).logits
            loss = functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum") / denominator
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite post-training loss")
        loss.backward()
        metrics["loss"] += loss.detach().item()
        del logits, loss
    parameters = [p for p in student.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    metrics["grad_norm"] = norm.item()
    optimizer.step()
    return metrics


def train(args):
    import torch
    config_path = Path(args.experiment_config)
    config = read_json(config_path)
    if not config["qualification"]["proceed"]:
        raise ValueError("Teacher qualification rejected distillation")
    targets = config["route_response_tokens"]
    if any(v <= 0 for v in targets.values()) or set(targets) != {"bank", "mbpp"}:
        raise ValueError("Freeze positive bank and MBPP response-token quotas")
    with gpu_session(args.run, "train", args.window_seconds) as window:
        signal.signal(signal.SIGTERM, window.stop)
        signal.signal(signal.SIGINT, window.stop)
        student, tokenizer, base, warm_hash = warm_student(args, config["student_adapter"])
        rows, exclusions = {}, {}
        for route in targets:
            path = config_path.parent / config["train_files"][route]
            if digest(path) != config["train_sha256"][route]:
                raise ValueError("Frozen training data checksum mismatch")
            rows[route] = read_json(path)
            if args.algorithm == "replay":
                rows[route], exclusions[route] = encode(tokenizer, rows[route], args.max_length)
            if not rows[route]:
                raise ValueError("Training route is empty")
        teacher, teacher_hashes = (None, {})
        if args.algorithm == "opd":
            teacher, teacher_hashes = teacher_model(
                args, config["teacher_adapters"], base, tokenizer)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        budget = sum(targets.values())
        contract = {
            "base": base, "warm_start": warm_hash, "teachers": teacher_hashes,
            "experiment": digest(config_path), "algorithm": args.algorithm,
            "arm": "OPD" if args.algorithm == "opd" else "REPLAY",
            "rank": 16, "seed": args.seed, "lr": args.lr,
            "response_tokens": budget, "route_response_tokens": targets,
            "rollout_tokens": args.rollout_tokens, "max_length": args.max_length,
            "accumulation": args.accumulation, "topk": args.topk,
            "sampling": "fresh student, T=1, top_p=1, top_k=0; no asynchronous replay",
            "objective": ("teacher-topk generalized reverse divergence, no support renormalization"
                          if args.algorithm == "opd" else "answer-only cross entropy"),
            "implementation": {n: digest(Path(__file__).with_name(n))
                               for n in ("opd.py", "train.py", "runtime.py")}}
        root = Path(args.run) / "posttraining" / contract["arm"]
        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                                      lr=args.lr, weight_decay=.01)
        # Identical token-based LR schedule in both arms, even with different EOS lengths.
        progress = {"tokens": 0, "step": 0,
                    "route_tokens": dict.fromkeys(targets, 0),
                    "route_cursor": dict.fromkeys(targets, 0)}
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: max(0., 1 - progress["tokens"] / budget))
        initial = parameter_fingerprint(student)
        if (root / "latest.json").exists():
            progress = restore(root, student, optimizer, scheduler, contract)
        while progress["tokens"] < budget and window.active:
            batch, planned = batch_rows(args, student, tokenizer, rows, progress, targets)
            metrics = update(args, student, teacher, optimizer, batch)
            progress.update(planned)
            progress["step"] += 1
            scheduler.step()
            progress["metrics"] = metrics
            atomic_json(root / "progress.json", progress)
            print(progress, flush=True)
            if progress["step"] % 8 == 0:
                save(root, student, optimizer, scheduler, progress, contract)
        checkpoint = save(root, student, optimizer, scheduler, progress, contract)
        final = parameter_fingerprint(student)
        complete = progress["tokens"] == budget
        atomic_json(root / "status.json", {
            "complete": complete, "progress": progress, "contract": contract,
            "checkpoint": checkpoint.name, "initial_parameters": initial,
            "final_parameters": final, "parameters_changed": initial != final,
            "optimizer_states": len(optimizer.state), "exclusions": exclusions,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated()})
        if not complete or initial == final or not optimizer.state:
            raise RuntimeError("Post-training did not complete a real parameter update")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--algorithm", choices=("opd", "replay"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--rollout-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=2560)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--window-seconds", type=int, default=7200)
    train(parser.parse_args())
