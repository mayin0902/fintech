# Top2 candidate risk retest summary

Date: 2026-05-17
Device: single H20 via `CUDA_VISIBLE_DEVICES=0`
Run dir: `/root/autodl-tmp/fintech/track1/h20_runs/20260517_top2_risk_retest`

## Candidates

1. `/root/autodl-tmp/fintech/track1/official/solution_candidate_sprint_shared_add_inplace.py`
2. `/root/autodl-tmp/fintech/track1/official/solution_candidate_sprint_shared_add_inplace_adaptive_long_t327680_flat.py`

The second candidate is a flattened, self-contained file. It is not the earlier thin wrapper variant.

## Submission-shape and load-order risk

Both candidates passed:

- `python -m py_compile`
- strict `load_state_dict` against `MoEBlockBaseline`
- parameter-key order check against `MoEBlockBaseline`

Parameter order matched:

`experts.gate_up_proj, experts.down_proj, shared_expert.gate_proj.weight, shared_expert.up_proj.weight, shared_expert.down_proj.weight, gate.weight, post_norm.weight`

## Correctness stability risk

Both candidates passed all multi-seed correctness checks:

- seeds: `42`, `123`
- sequence lengths: `2048`, `8192`, `16384`
- dtype: `bf16`
- checks: forward output, input gradient, gate/expert/shared/post_norm parameter gradients

No `FAIL`, `Traceback`, `RuntimeError`, CUDA error, or OOM keyword was found in the correctness logs.

## Single-GPU benchmark

All benchmark numbers below are from the same single H20 run with `warmup=3`, `measure=5`, `bf16`.

| Candidate | SeqLen | Peak MB | Avg ms | Notes |
|---|---:|---:|---:|---|
| official baseline wrapper | 8192 | 4348.38 | 48.47 | visible baseline |
| official baseline wrapper | 131072 | 33639.01 | 560.33 | visible baseline |
| shared_add_inplace | 8192 | 4180.38 | 46.76 | pass |
| shared_add_inplace | 131072 | 21946.01 | 542.42 | pass |
| shared_add_inplace | 327680 | 55172.76 | 1364.79 | pass |
| shared_add_inplace | 393216 | 70653.02 | 1631.85 | pass |
| t327680_flat | 8192 | 4180.38 | 46.69 | pass |
| t327680_flat | 131072 | 21946.01 | 542.70 | pass |
| t327680_flat | 327680 | 54922.14 | 1368.44 | pass |
| t327680_flat | 393216 | 70141.02 | 1636.95 | pass |

## 6:4 proxy versus official baseline

Proxy definition used for reporting: visible seqs `8192` and `131072` are equally averaged; score ratio is:

`0.6 * avg(baseline_peak_mb / candidate_peak_mb) + 0.4 * avg(baseline_avg_ms / candidate_avg_ms)`

| Candidate | 8192 mem/time reduction | 131072 mem/time reduction | 6:4 proxy ratio |
|---|---:|---:|---:|
| shared_add_inplace | 3.86% / 3.53% | 34.76% / 3.20% | 1.1858 |
| t327680_flat | 3.86% / 3.67% | 34.76% / 3.15% | 1.1860 |

## Risk interpretation

`shared_add_inplace` remains the lower-risk final submission candidate. It has the same visible memory as `t327680_flat`, essentially tied visible speed, and a simpler always-on behavior without an adaptive long-sequence threshold branch.

`t327680_flat` is now valid as a single-file submission candidate and passed the same load/correctness tests. Its only concrete advantage is lower memory at very long lengths: about `250.62 MB` less at `327680` and `512.00 MB` less at `393216` in this run. The tradeoff is slightly slower long-sequence time in this run and additional branch logic, so it is best viewed as a long-sequence hedge rather than a clearly stronger visible-score candidate.

