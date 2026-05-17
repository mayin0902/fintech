# Submission validation summary

Date: 2026-05-17

## Submission artifacts

- Clean folder: `/root/autodl-tmp/fintech/track1/submission_shared_add_inplace_clean_20260517`
- Source file: `/root/autodl-tmp/fintech/track1/submission_shared_add_inplace_clean_20260517/solution.py`
- Zip package: `/root/autodl-tmp/fintech/track1/submission_shared_add_inplace_clean_20260517_code.zip`

The clean folder contains only `solution.py`.

The zip package contains exactly one root-level file:

- `solution.py`

SHA256 for the submitted `solution.py`:

`8e60f420fc081b738b758c84cdbd4be19a0b793d66f9bd4309d692656b87419e`

## Format checks

- `python -m py_compile solution.py`: pass
- `from solution import MoEBlockOptimized`: pass
- strict `load_state_dict` from `MoEBlockBaseline`: pass
- parameter order matches `MoEBlockBaseline`: pass
- zip extraction smoke import and strict load: pass

Matched parameter order:

`experts.gate_up_proj, experts.down_proj, shared_expert.gate_proj.weight, shared_expert.up_proj.weight, shared_expert.down_proj.weight, gate.weight, post_norm.weight`

## Correctness checks

Official local `correctness_check.py` was run against the clean folder's `solution.py`.

- seeds: `42`, `123`
- sequence lengths: `2048`, `8192`, `16384`
- dtype: `bf16`

All 6 correctness cases passed. No failure, traceback, CUDA error, or OOM keyword was found in the log.

Log:

`/root/autodl-tmp/fintech/track1/h20_runs/20260517_submission_shared_add_inplace_validation/correctness_clean_solution.log`

Additional non-`B=1` correctness smoke tests were run:

| Batch | SeqLen | Seed | Result |
|---:|---:|---:|---|
| 2 | 2048 | 42 | pass |
| 2 | 8192 | 42 | pass |
| 4 | 2048 | 42 | pass |
| 4 | 4096 | 123 | pass |

Log:

`/root/autodl-tmp/fintech/track1/h20_runs/20260517_submission_shared_add_inplace_validation_batchsize/correctness_batchsize_smoke.log`

## Benchmark smoke

Official local `benchmark.py` was run on one H20 with `warmup=3`, `measure=5`, `bf16`.

| SeqLen | Peak Memory MB | Avg ms |
|---:|---:|---:|
| 8192 | 4180.38 | 46.63 |
| 131072 | 21946.01 | 542.41 |

Log:

`/root/autodl-tmp/fintech/track1/h20_runs/20260517_submission_shared_add_inplace_validation/benchmark_clean_solution_8k_128k.log`
