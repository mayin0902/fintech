# H20 8-10h Final-Shape Results - 2026-05-16

Run directory:

`/root/autodl-tmp/fintech/track1/h20_runs/20260515_165329_8_10h_final_shape_interactive`

## Final Recommendation

Use `compile_shared_post_nocp_s262144_e262144` for the score-seeking H20 submission if hidden 393216 safety is not required.

Keep `default_ckpt_s65536_e196608` as the fallback if hidden long sequence length matters.

Do not use `compile_route_shared_post` as default; it is slightly lower memory but consistently slower, with only marginal memory savings.

Do not use checkpoint+compile as the main default; `max-autotune-no-cudagraphs` passes correctness and lowers long-seq memory, but fails the 131072 stable-default gate because it is slower and higher memory than the current checkpoint default at 131072.

## Key Tables

### Reward Long Confirmation

22 cycles, 528/528 benchmark rows passed.

| Seq len | `compile_shared_post` peak MB | `compile_shared_post` avg ms | `nocp` avg ms | Speedup vs `nocp` |
|---:|---:|---:|---:|---:|
| 8192 | 4180.38 | 46.750 | 47.645 | 1.878% |
| 32768 | 7151.01 | 146.684 | 149.975 | 2.195% |
| 65536 | 11977.76 | 278.101 | 284.681 | 2.311% |
| 131072 | 22594.02 | 542.333 | 555.290 | 2.333% |
| 196608 | 32922.25 | 798.980 | 818.265 | 2.357% |
| 262144 | 43537.51 | 1060.601 | 1086.288 | 2.365% |

### Capacity

| Seq len | `compile_shared_post` | `nocp` | `default_ckpt` |
|---:|---|---|---|
| 327680 | pass, 69307.00 MB, 1581.22 ms | pass, 82164.76 MB, 1397.07 ms | pass, 51444.76 MB, 1657.94 ms |
| 393216 | OOM | OOM | pass, 61476.01 MB, 1982.80 ms |

### Final Gate

Recommended candidate:

- 2048 correctness: pass.
- 8192 correctness: pass.
- Final no-env benchmark through 262144: pass.

Final no-env reward candidate benchmark:

| Seq len | Peak MB | Avg ms |
|---:|---:|---:|
| 8192 | 4180.38 | 46.720 |
| 32768 | 7151.01 | 146.710 |
| 65536 | 11977.76 | 278.190 |
| 131072 | 22594.02 | 542.640 |
| 262144 | 43754.52 | 1068.000 |

Current official `solution.py` no-env benchmark:

| Seq len | Peak MB | Avg ms |
|---:|---:|---:|
| 8192 | 4060.38 | 65.910 |
| 131072 | 21351.01 | 663.470 |
| 262144 | 41413.51 | 1330.100 |

### Swapped Reward Soak

10 cycles, 180/180 benchmark rows passed. Main ran on GPU1 and stress ran on GPU0.

| Seq len | `compile_shared_post` peak MB | `compile_shared_post` avg ms | `nocp` avg ms | Speedup vs `nocp` |
|---:|---:|---:|---:|---:|
| 131072 | 22594.02 | 538.679 | 551.698 | 2.360% |
| 262144 | 43537.51 | 1068.137 | 1093.627 | 2.331% |

## Artifacts

- Final decision: `final_decision.md`
- Control log: `control_log.md`
- Risk register: `risk_register.md`
- Final gate summary: `phase_007_final_gate_summary.md`
- Reward long confirmation: `phase_006_reward_long_confirm_summary.md`
- Capacity repeat: `phase_006_capacity_repeat_summary.md`
- Stable hedge confirmation: `phase_006_stable_hedge_long_summary.md`
- Swapped reward soak: `phase_007_swapped_reward_soak_summary.md`

## Final Action

Prepare a restore-backed `solution.py` replacement from:

`track1/official/solution_candidate_compile_shared_post.py`

Then rerun final correctness and benchmark before submission.
