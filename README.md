# Bounded RFT and on-policy distillation

`main` contains code; `research` contains the manuscript, frozen protocol,
conversation excerpts, sanitized logs and measured results.

This implements single-GPU LoRA, executable CPU verification, frozen RFT
selection, synchronous student-prefix distillation and an equal-response-token
SFT replay control. The teacher-top-k objective is
`sum[p*(log(p)-log(q))-p+q]` with full-vocabulary-normalized probabilities.
It is a truncated approximation. The frozen teacher is separate from the student.
This same-size Qwen3.5-4B experiment covers two code task suites, not general
multidomain fusion or reward-based joint RL.

Use Python 3.11, CUDA PyTorch 2.9.1 and `requirements.txt`.
Run `python -m unittest discover -s tests`.
Supply cached Qwen3.5-4B text assets and a JSON manifest of every model/tokenizer
file: `{"files":{"config.json":{"bytes":123,"sha256":"<actual hash>"}}}`.
Actual file sizes and hashes are required. Downloads and remote code are disabled.

`prepare.py` reads local MBPP JSON; `weekend_bank.py` creates a deterministic
specification bank. Both separate prompts from private executable tests.
Generate an unadapted pool with `generate.py`, grade it on CPU with `judge.py`,
freeze arm selection, and run `train.py` with identical token budgets.
Use `--help` for each module. `weekend_evaluation.decide` selects a common
learning rate using all 18 bank development configurations.

For OPD, qualify a teacher against a completed SFT student on a complete
development set, then fill `examples/opd-experiment.json` with actual paths
and hashes. Training rows contain `task_id`, `prompt`, `interface`, `code`;
replay code must be a verified training answer.

```bash
CUDA_VISIBLE_DEVICES=0 python -m idle_rft.opd \
  --model /data/model --model-manifest /data/model-manifest.json \
  --experiment-config /data/opd/experiment.json --run /results/opd-42 \
  --algorithm opd --seed 42 --lr 0.00005 --topk 64 \
  --rollout-tokens 256 --max-length 2560 --accumulation 4 \
  --window-seconds 7200
```

Repeat with `--algorithm replay` and seeds 42/43/44.
Each run has exactly 16,384 response tokens, bank/MBPP=3:1.
Fresh student sampling uses temperature 1, top-p 1, top-k 0.
Logs report top-k probability coverage, support misses and capped prefixes.

An unchanged command and directory resume checksummed Adam, scheduler,
exact route counters, prompt cursors and RNG. Acceptance requires completed
quotas, finite metrics, changed parameter hashes and nonempty optimizer state.
Each process has a finite window and an owner lock.

Evaluate with `generate --split dev --samples 1 --adapter <checkpoint/adapter>`.
Blind runs also require a frozen `--selection` with base/dataset hashes,
approved adapter hashes, `max_new_tokens=2048`, `proceed=true`.
Report every declared variant; never select a seed on blind.

CPU tests cover divergence gradients, position alignment, route quotas,
resumable ordering and checkpoint integrity. Actual GPU evidence is on
`research`. Enforce a shared GPU allocation/budget in your external scheduler.
