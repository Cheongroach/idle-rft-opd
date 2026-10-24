# RFT + OPD research record

Portable implementation: [main](https://github.com/Cheongroach/idle-rft-opd/tree/main).

Pilot complete; 26 original configurations complete and 10 interrupted configurations resumed from verified checkpoints. Full strengthening acceptance remains pending.

| Blind suite | Warm SFT | OPD, 3-seed mean | Replay, 3-seed mean |
|---|---:|---:|---:|
| bank | 85.39% | 94.66% | 90.13% |
| mbpp | 61.76% | 61.37% | 58.82% |

Observed H20 Pod time: 172.98 GPUh; peak allocation 64 GPUs; pilot terminal accounting. Resource observations are not billing.

[Final measured report](results/FINAL_REPORT.md) · [OPD design and interview manuscript](manuscript/OPD_DESIGN_AND_INTERVIEW.md) · [Frozen protocol](protocol/OPD_PROTOCOL.md) · [Conversation excerpts](conversations/excerpts.md)

[Strengthening protocol](extension/PROTOCOL.md) · [Interruption and recovery record](extension/INTERRUPTION_AND_RECOVERY.md) · [Verified recovery startup](extension/recovery-start.json)

This covers two code task suites with the same-size tokenizer-compatible teachers. OPD is synchronous, after completed MBPP SFT, without reward-based joint RL. Equal response-token budgets do not equal total compute budgets. Paired per-seed intervals are reported separately from cross-seed dispersion.

The manuscript distinguishes cited public methods, analysis and proposed work. Results include only completed, validated experiments. Raw platform payloads, credentials, private judging data and weights are excluded.
