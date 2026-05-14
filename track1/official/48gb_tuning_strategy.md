# 48GB GPU Tuning Strategy

This note records the next tuning plan after the 12-hour dual-GPU search run in
`track1/search_runs/20260513_1344_gpu0_gpu3_12h`.

## Current Baseline

The 24GB run found that the current implementation is correctness-stable and
can run long sequences with activation checkpointing:

| SeqLen | Best low-memory variant | Peak MB | Avg ms | Best fast variant | Peak MB | Avg ms |
|---:|---|---:|---:|---|---:|---:|
| 2048 | `ckpt_s8192_e8192` | 3432.23 | 57.97 | `nocp_s2048_e2048` | 3510.10 | 47.25 |
| 8192 | `ckpt_s8192_e8192` | 4012.63 | 96.31 | `nocp_s8192_e8192` | 4300.63 | 73.75 |
| 32768 | `ckpt_s8192_e8192` | 6814.85 | 387.94 | `nocp_s8192_e8192` | 9328.38 | 325.72 |
| 65536 | `ckpt_s2048_e2048` | 11272.01 | 2320.49 | `nocp_s8192_e8192` | 17416.01 | 698.84 |
| 131072 | `ckpt_s8192_e8192` | 21303.26 | 1880.07 | `ckpt_s8192_e8192` | 21303.26 | 1666.88 |

Key observation: on 24GB, checkpointing is needed for the longest sequence, but
it adds material recompute cost. A 48GB card should be used to test larger
chunks and no-checkpoint variants first, rather than repeating the full 24GB
search space.

## Objective

For 48GB hardware, optimize for the fastest correct variant that keeps peak
memory below roughly 44-45GB. Lowest memory is no longer the primary objective;
the extra memory budget should be spent on larger chunks and less recompute.

Ranking rule:

1. Correctness passes.
2. Benchmark does not OOM.
3. Peak memory is below 45GB.
4. Lowest `avg_ms` wins.
5. If speeds are close, choose the lower-memory variant.

## Priority Search Space

Start with these variants:

```text
nocp_s8192_e8192
nocp_s16384_e8192
nocp_s16384_e16384
nocp_s32768_e8192

ckpt_s8192_e8192
ckpt_s16384_e8192
ckpt_s16384_e16384
ckpt_s32768_e8192
ckpt_s32768_e16384
```

Do not spend much time on `1024` or `2048` chunk variants unless larger chunks
OOM. The previous run showed that small chunks can reduce memory but often cost
too much time.

## Sequence-Length Plan

Run in this order:

```text
131072 -> 196608 -> 262144
```

Recommended first pass:

1. Validate `131072` with all no-checkpoint variants above.
2. Keep only variants under 45GB and benchmark them with more than one measure.
3. Move the surviving no-checkpoint variants to `196608`.
4. Use checkpointed large-chunk variants for `262144`.

Expected candidates:

- `nocp_s8192_e8192` or `nocp_s16384_e8192` for `131072`.
- `ckpt_s16384_e8192` for `196608` and `262144`.
- `ckpt_s8192_e8192` as the stable fallback.

If `nocp_s16384_e16384` is already near 45GB or OOM at `131072`, stop testing
no-checkpoint variants on longer sequences and switch to checkpointed variants.

## Two-Card Execution

When two 48GB cards are available, keep work serial within each card but keep
both cards occupied:

- GPU A: correctness and stability runs for `131072`, `196608`, and `262144`.
- GPU B: benchmark runs for the surviving variants, prioritizing no-checkpoint
  and large-chunk candidates.

This avoids intra-card memory contention while still using both cards
continuously.

## Suggested Next Runner Changes

The current runner supports fixed variants through `VARIANTS` in
`search_runner.py`. For the next round, add a 48GB profile or a custom variant
list so the run does not waste hours on the old 24GB-oriented combinations.

Suggested profile behavior:

- `profile=48gb-correctness`: correctness at `2048`, then benchmarks at
  `131072`, `196608`, and `262144`.
- `profile=48gb-benchmark`: skip low-value small chunks and repeat the best
  surviving long-sequence variants with `measure > 1`.

Before any long run, do a smoke test with one no-checkpoint variant and one
checkpointed fallback at `131072`.
