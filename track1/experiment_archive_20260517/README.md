# Track 1 Experiment Archive - 2026-05-17

This directory is a curated archive for review and post-contest retrospective. It intentionally contains only the main candidate implementations, core experiment summaries, and runner scripts. Large raw caches and full `h20_runs/` outputs are not included.

## Project Goal

Track 1 optimizes the training-time memory footprint of a long-sequence MoE Block on a single H20 96GB GPU. The implementation must preserve the baseline mathematical definition and pass forward output, input gradient, and parameter gradient checks before memory and speed are scored.

The final selected method is **chunked computation + in-place merge**:

- flatten `[B, T, H]` to `[B*T, H]`;
- route tokens without materializing a large one-hot expert mask;
- compute routed experts and shared expert in token chunks;
- add shared expert output directly into routed output;
- keep baseline-compatible parameter names and strict load behavior.

## Candidate Implementations

| File | Role | Summary |
|---|---|---|
| `candidates/00_official_baseline_wrapper.py` | Baseline wrapper | Imports and exposes official baseline as `MoEBlockOptimized` for comparable benchmark runs. |
| `candidates/01_checkpoint_chunk_fallback.py` | Conservative fallback | Uses checkpoint-style recomputation and smaller chunks to lower long-sequence memory, at the cost of slower backward. |
| `candidates/02_large_chunk_compile_shared_post.py` | Speed-oriented middle stage | Uses larger chunks, disables checkpoint by default, and compiles shared expert/post norm paths. Strong 8K/128K result but OOM at 393216 in testing. |
| `candidates/03_ordered_large_chunk_compile.py` | Submission-shape repair | Keeps large-chunk behavior while aligning parameter registration order with the official baseline. |
| `candidates/04_final_chunked_inplace_merge.py` | Final selected code | Removes an extra full-size `combined` tensor by adding shared expert output into routed output in place. |
| `candidates/05_adaptive_long_sequence_hedge.py` | Hidden long-sequence hedge | Further reduces long-sequence memory with adaptive chunking, but did not robustly improve visible 8K/128K score. |

## Key Documents

| File | Contents |
|---|---|
| `docs/results_2026-05-16_h20_8_10h_final_shape.md` | Earlier H20 final-shape search and capacity confirmation. |
| `docs/results_2026-05-17_final_experiment_summary.md` | Main experiment timeline and final candidate ranking. |
| `docs/final_30m_summary.md` | Last sprint comparison between final candidate and adaptive long-sequence near-neighbor. |
| `docs/top2_risk_retest_summary.md` | Final top-2 risk retest with correctness, strict load, and 8K/128K/long benchmark data. |
| `docs/submission_validation_summary.md` | Final submission package validation summary. |

## Final 8K/128K Result

Single H20, `bf16`, official local `benchmark.py`, `warmup=3`, `measure=5`:

| Implementation | SeqLen | Peak MB | Avg ms |
|---|---:|---:|---:|
| Official baseline wrapper | 8192 | 4348.38 | 48.47 |
| Official baseline wrapper | 131072 | 33639.01 | 560.33 |
| Final solution | 8192 | 4180.38 | 46.63 |
| Final solution | 131072 | 21946.01 | 542.41 |

Approximate visible-score proxy using equal 8K/128K averaging and memory:speed = 6:4:

`1.186x` relative to the official baseline wrapper.

## Final Package

The competition-ready package is in:

`../final_submission_20260517/马胤+code.zip`

That zip contains exactly one root-level file:

`solution.py`

