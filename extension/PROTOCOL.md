# OPD strengthening protocol — frozen before extension runs

Frozen 2026-10-24. This is a follow-up to the disclosed 16,384-response-token
pilot. Its published blind scores are historical evidence; they must not select
new recipes or serve as a new one-shot final evaluation.

The original authorization remains one shared allocation: [authorized resource group],
cloudnative-lf H20, <=64 high-priority concurrent GPUs and <=3000 conservatively
reserved GPU-hours across the pilot and this extension. Pilot reservations total
721 GPUh; no refunds. No automatic failed-job retries. Create timeouts may be
reconciled only to exactly one matching existing run, never blindly resubmitted.

## Training and comparisons

- Same cached Qwen3.5-4B assets, tokenizer, rank16 warm MBPP SFT adapter and
  frozen bank teacher as the pilot. Training data stays the pilot's vetted
  generated answers; all private reference programs/tests stay on CPU.
- Main curves: independent full runs at 65,536 / 262,144 / 524,288 response
  tokens, OPD and answer-only replay, seeds42/43/44. Bank/MBPP token share 3:1.
  These runs have independent token-based LR horizons, not checkpoints along
  one identical optimization path. Do not join them into one trajectory.
- Bank-only OPD controls match the bank token quota of each main run:
  49,152 / 196,608 / 393,216 bank tokens, zero MBPP. All three seeds.
  This measures extra interference at matched target-task tokens, not matched
  total compute, and does not independently diagnose a capacity limit.
- Retention-share ablation: balanced OPD at 524,288 total tokens, bank/MBPP=1:1,
  seeds42/43/44; bank-only 262,144-token controls match its bank quota.
- Prefix sensitivity: 65,536-token 3:1 OPD with 512-token prefixes, three seeds.
  Default rollout cap is256. Capped prefixes are usable conditional contexts,
  not completed correct programs.
- All runs start from the same SFT adapter; lr5e-5, AdamW weight decay.01,
  norm cap1, four-response accumulation, token-based linear decay. Teacher-top64
  generalized reverse divergence retains full-vocabulary-normalized p/q.
  No reward-based RL, asynchronous rollout or full-vocabulary-loss ablation.
- Quotas are exact. Loss/gradient finiteness, real parameter change, nonempty
  optimizer state and complete checkpoint/file hashes gate training completion.
- Each GPU process/node has a finite <=12h cap. Shared reservations include
  complete command duration, 1h platform preparation and30s kill grace.

## Data and selection

- Development combines the original development sets with256 newly generated
  bank specifications and approximately80 eligible full-MBPP tasks excluded
  from every original project task ID, normalized prompt and alpha-renamed
  reference program. Connected duplicate groups never cross dev/final.
- Final uses2048 new bank specifications and the remaining newly audited
  full-MBPP tasks. New bank specifications exclude all8192 old spec hashes.
  Empty train split ensures these prompts do not enter optimization.
- CPU reference execution must pass before freezing. Public MBPP has sparse
  source tests and potential model-pretraining contamination. Project-disjoint
  tasks do not prove pretraining cleanliness. Shared bank operator families
  measure new-spec generalization, not unseen-operator transfer.
- All declared36 training configurations are evaluated on the complete
  development set with identical greedy/no-thinking/2048-token generation.
- Choose one OPD token budget using three-seed dev means: retain budgets whose
  retention pass rate is no more than2pp below warm dev; among those maximize
  mean bank score, tie-break smaller budget. If none qualifies, report a failed
  selection and evaluate the highest macro-mean budget without deployment claim.
  Compare replay at the same selected total-token budget.
- All three seeds of selected OPD/replay, matched bank-only OPD, balanced OPD,
  balanced matched bank-only, 512-prefix OPD and warm/bank experts enter final.
  No final seed selection or final-dependent additional training.
- Report budget curves, same-bank-token single/mixed differences, per-seed
  paired bootstrap CIs, seed dispersion, specialist gap, truncations and
  exact response-token/teacher/process/Pod cost boundaries.
- A curve may be described as approximately plateaued only if the largest two
  main OPD budget means differ by<1pp on bank dev and paired per-seed intervals
  contain zero. Otherwise report remaining gain/uncertainty; training completion
  and decreasing changing-prefix loss do not prove convergence.
- Retention final non-inferiority target: student minus warm lower95% paired
  CI >=-2pp in every reported seed. Wide CIs fail to establish non-inferiority;
  never substitute "no significant loss" for passing this criterion.

## Fixed-state diagnostic and delivery

Freeze64 original bank-dev prefixes (32 formerly correct/32 incorrect) and16
MBPP-dev prefixes (8/8), selected by deterministic hash before new training.
Only prompt/interface/generated code enter GPU packages; correctness metadata
and tests stay on CPU. Score identical frozen response prefixes with warm and
completed students: top64 distance, teacher/student mass, sampled-token
logratio, response CE and entropy. No gradient, no updating on diagnostic rows.
This is conditional scoring, not teacher corrective continuation success.

Publish portable training code on GitHub main and manuscripts, actual dialogue,
sanitized metrics and validated results on research. Preserve the pilot report,
add a separate extension report; no numerical acceptance before terminal evidence.
