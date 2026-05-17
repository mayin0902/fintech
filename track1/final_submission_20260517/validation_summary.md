# Final Validation Summary

## Package

- Code zip: `马胤+code.zip`
- Source file: `solution.py`
- Report: `马胤+report.md`

## Format Validation

- Zip root contains exactly `solution.py`
- `python -m py_compile solution.py`: pass
- `from solution import MoEBlockOptimized`: pass
- strict `load_state_dict` from `MoEBlockBaseline`: pass
- parameter order matches baseline: pass

## Correctness Validation

Official local `correctness_check.py` on the final submission source passed:

- seeds: `42`, `123`
- sequence lengths: `2048`, `8192`, `16384`
- dtype: `bf16`

Additional smoke tests passed:

- `B=2`, `T=2048`, seed `42`
- `B=2`, `T=8192`, seed `42`
- `B=4`, `T=2048`, seed `42`
- `B=4`, `T=4096`, seed `123`

## Benchmark Validation

Official local `benchmark.py`, single H20, `bf16`, `warmup=3`, `measure=5`:

| SeqLen | Peak MB | Avg ms |
|---:|---:|---:|
| 8192 | 4180.38 | 46.63 |
| 131072 | 21946.01 | 542.41 |

Direct rebenchmark from the final zip produced the same peak memory and only small timing noise:

| Run | 8192 Avg ms | 131072 Avg ms |
|---|---:|---:|
| Rebenchmark 1 | 46.81 | 542.99 |
| Rebenchmark 2 | 46.69 | 542.67 |

