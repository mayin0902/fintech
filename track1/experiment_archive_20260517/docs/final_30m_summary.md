# 30m Final Sprint Summary

Updated UTC: `2026-05-17T02:15:00Z`

## Scope

本轮只做最后窄范围复测和总文档整理：

- 不覆盖 `official/solution.py`
- 不打包
- 只比较当前主推 `shared_add_inplace` 和最接近的 `t327680` 近邻

## Correctness

Both candidates passed `16384` correctness:

| Candidate | 16384 correctness |
|---|---|
| `shared_add_inplace` | pass |
| `shared_add_inplace_adaptive_long_t327680_s131072_e262144` | pass |

`py_compile` also passed for both.

## Benchmark, warmup=5, measure=10

| Candidate/run | 8192 MB/ms | 131072 MB/ms | 262144 MB/ms | 327680 MB/ms | 393216 MB/ms |
|---|---|---|---|---|---|
| `shared_add_inplace` run1 | 4180.38 / 46.78 | 21946.01 / 542.77 | 42602.51 / 1068.23 | 55245.77 / 1365.00 | 70727.02 / 1631.79 |
| `t327680` run1 | 4180.38 / 46.57 | 21946.01 / 542.20 | 42602.51 / 1066.90 | 54995.14 / 1358.43 | 70215.02 / 1624.74 |
| `shared_add_inplace` swap | 4180.38 / 46.62 | 21946.01 / 542.25 | 42602.51 / 1060.45 | 55245.77 / 1355.43 | 70727.02 / 1620.81 |
| `t327680` swap | 4180.38 / 46.59 | 21946.01 / 542.32 | 42602.51 / 1067.92 | 54995.14 / 1368.12 | 70215.02 / 1636.17 |

## Interpretation

- On visible `8192/131072`, `t327680` and `shared_add_inplace` are effectively tied.
- `t327680` consistently reduces long-sequence peak memory:
  - `327680`: `55245.77 -> 54995.14 MB`
  - `393216`: `70727.02 -> 70215.02 MB`
- Long-sequence speed is mixed after role swap, so `t327680` is not a robust speed win.
- `t327680` is currently a wrapper candidate importing the adaptive-long base file. It is not submission-shape unless flattened into a single `solution.py`.

## Recommendation

Keep `shared_add_inplace` as primary final candidate because it is self-contained and has the broader soak history.

Keep `t327680` as a hidden-risk alternative only if there is time to flatten it and rerun final gates.

