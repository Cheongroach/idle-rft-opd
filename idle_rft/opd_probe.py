"""No-update diagnostics on frozen student-generated development prefixes."""
import argparse
from pathlib import Path
import signal
import time

from .common import atomic_json, digest, gpu_session, identity, read_json
from .opd import (checkpoint_info, response_positions, teacher_model,
                  topk_reverse_divergence, warm_student)
from .runtime import render


def probe(args):
    import torch
    config_path = Path(args.experiment_config)
    config = read_json(config_path)
    prefix_path = config_path.parent / config["fixed_prefix_file"]
    if digest(prefix_path) != config["fixed_prefix_sha256"]:
        raise ValueError("Fixed-prefix file changed")
    with gpu_session(args.run, "evaluate", args.window_seconds) as window:
        signal.signal(signal.SIGTERM, window.stop)
        student, tokenizer, base, warm_hash = warm_student(args, config["student_adapter"])
        if args.adapter:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            _, adapter_hash = checkpoint_info(args.adapter, base)
            result = set_peft_model_state_dict(
                student, load_file(Path(args.adapter) / "adapter_model.safetensors"))
            if result.unexpected_keys:
                raise ValueError("Unexpected diagnostic adapter keys")
        else:
            adapter_hash = warm_hash
        student.requires_grad_(False)
        student.eval()
        teacher, hashes = teacher_model(args, config["teacher_adapters"], base, tokenizer)
        before = identity({name: p.detach().float().cpu().numpy().tobytes().hex()
                           for name, p in student.named_parameters() if ".lora_" in name})
        started, records = time.monotonic(), []
        with torch.inference_mode():
            for row in read_json(prefix_path):
                if not window.active:
                    raise TimeoutError("Fixed-state diagnostic incomplete")
                prompt = tokenizer.encode(render(tokenizer, row), add_special_tokens=False)
                answer = tokenizer.encode(row["code"], add_special_tokens=False)
                answer = (answer + [tokenizer.eos_token_id])[:min(256, args.max_length-len(prompt))]
                if not answer:
                    raise ValueError("Empty diagnostic response")
                inputs = torch.tensor([prompt + answer], device="cuda")
                positions = response_positions(len(prompt), len(answer))
                teacher.set_adapter(row["route"])
                teacher.requires_grad_(False)
                q = teacher(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            use_cache=False).logits[0, positions].float().log_softmax(-1)
                qk, ids = q.topk(64, -1)
                logits = student(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                 use_cache=False).logits[0, positions]
                terms, pmass, qmass = topk_reverse_divergence(logits, qk, ids)
                p = logits.float().log_softmax(-1)
                actions = inputs[0, len(prompt):].unsqueeze(-1)
                records.append({
                    "task_id": row["task_id"], "route": row["route"], "tokens": len(answer),
                    "top64_distance": terms.mean().item(), "student_mass": pmass.mean().item(),
                    "teacher_mass": qmass.mean().item(),
                    "sampled_logratio": (p.gather(-1, actions)-q.gather(-1, actions)).mean().item(),
                    "student_ce": -p.gather(-1, actions).mean().item(),
                    "teacher_ce": -q.gather(-1, actions).mean().item(),
                    "entropy": -(p.exp()*p).sum(-1).mean().item()})
                del logits, p, q
        after = identity({name: p.detach().float().cpu().numpy().tobytes().hex()
                          for name, p in student.named_parameters() if ".lora_" in name})
        if before != after:
            raise ValueError("Diagnostic modified model parameters")
        aggregate = {}
        for route in ("bank", "mbpp"):
            rows = [r for r in records if r["route"] == route]
            total = sum(r["tokens"] for r in rows)
            aggregate[route] = {"tokens": total, "prefixes": len(rows), **{
                key: sum(r[key]*r["tokens"] for r in rows)/total for key in
                ("top64_distance", "student_mass", "teacher_mass", "sampled_logratio",
                 "student_ce", "teacher_ce", "entropy")}}
        atomic_json(Path(args.run) / "fixed-prefix-diagnostics.json", {
            "complete": True, "parameters_unchanged": True, "adapter": adapter_hash,
            "base": base, "teachers": hashes, "prefix_sha256": digest(prefix_path),
            "records": records, "by_route": aggregate, "seconds": time.monotonic()-started,
            "scope": "Frozen conditional prefix scoring; no teacher continuation or gradient update"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=2560)
    parser.add_argument("--window-seconds", type=int, default=3600)
    probe(parser.parse_args())
