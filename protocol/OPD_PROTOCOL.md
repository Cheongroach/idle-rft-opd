# OPD experiment protocol — frozen before teacher qualification

Frozen 2026-10-24. This adds a distillation comparison to the weekend RFT
experiment; its data, decisions and reports remain separately identifiable.

- Student: completed MBPP arm A, seed 42, rank-16 SFT adapter. This is a
  same-size expert fusion experiment on a cached Qwen3.5-4B checkpoint. The
  checkpoint's pure pretraining provenance is unknown.
- Bank teacher: original experiment's common learning rate and winning arm,
  selected using all 18 complete development reports. Teacher seed 42 is fixed
  in advance. Never select a teacher seed using development or blind outcomes.
- MBPP teacher: the student's initial MBPP adapter, to anchor its previous task.
- Qualification: compare the bank teacher with the MBPP student on all 768 bank
  development tasks. Require at least +1 percentage point and a positive lower
  endpoint of the paired 95% bootstrap interval. A failed gate is a scientific
  stopping result; do not silently replace the teacher.
- Algorithms: synchronous teacher top-64 generalized reverse divergence on
  freshly sampled student prefixes, and answer-only SFT replay. The latter uses
  the frozen, verified generated solutions from the original task.
- Three training seeds: 42, 43, 44. Warm start and teachers are identical.
  Learning rate 5e-5, AdamW weight decay .01, gradient norm cap 1, accumulation
  of four responses, token-based linear LR decay, rank 16, maximum length 2560.
- Exact response-token budget per configuration: 16,384, divided into 12,288
  bank and 4,096 MBPP tokens. Teacher scoring tokens and rollout computation
  are additional compute, not additional supervised tokens. All six runs use
  the same route quotas. No performance-based continuation or LR sweep.
- Student sampling: temperature 1, full softmax, top-p 1, top-k 0. Generate up
  to 256 tokens per response; prefixes at the cap are usable distillation
  contexts, are recorded, and are not presented as completed correct programs.
- Objective: sum over teacher top-k support of
  `p * (log(p) - log(q)) - p + q`, with full-vocabulary-normalized probabilities.
  This is an approximation with omitted tail terms. Report teacher/student
  probability mass and sampled-token support misses. Do not label it exact
  full-vocabulary KL or sampled-token policy-gradient OPD.
- No asynchronous rollout, importance weighting, RL reward, or RL teacher
  construction is included in this first implementation. Student prefixes are
  fixed during each update; no gradient through discrete generation.
- Acceptance: successful terminal snapshot, checksums, exact route budgets,
  finite loss and gradients, changed LoRA parameter hash, nonempty optimizer
  state. Submission or process startup does not count as training success.
- Evaluation: frozen single-candidate, greedy, thinking disabled, 2048 output
  tokens. Compare all six configurations, without selecting one on blind:
  1280 bank blind tasks and 170 MBPP blind tasks. Include warm student and both
  relevant experts. Reuse an existing result only when the complete evaluation
  protocol, adapter hash, dataset, verifier and task set match.
- Report cross-seed mean/std, paired intervals, per-task wins/losses, MBPP
  retention and bank expert gap. This measures two code task suites, not
  math/code/reasoning multidomain fusion or frontier-model capability.
- Every launch uses the existing global locked ledger, <=64 concurrent H20
  GPUs and <=3000 conservatively reserved GPU-hours. No automatic GPU retries.
  Private reference judging data stays on the CPU side.
