# Track 1 Final Submission Artifacts

This directory contains the curated final artifacts for Track 1: long-sequence training MoE Block memory optimization.

## Files

| File | Purpose |
|---|---|
| `solution.py` | Final source implementation of `MoEBlockOptimized`. |
| `马胤+code.zip` | Competition code package. The zip root contains exactly one file: `solution.py`. |
| `马胤+report.md` | Markdown version of the solution report. Convert to `马胤+report.pdf` for formal submission if the platform requires PDF. |
| `validation_summary.md` | Concise validation and benchmark summary for the final package. |

## Final Implementation

The final implementation uses a "chunked computation + in-place merge" strategy:

- Keep parameter names, shapes, and load behavior compatible with `MoEBlockBaseline`.
- Flatten `[B, T, H]` into `[B*T, H]` for routing and expert computation.
- Avoid materializing a large one-hot expert mask.
- Chunk routed expert and shared expert computation along the token dimension.
- Add shared expert output directly into routed output, avoiding an extra full-size `combined` tensor.
- Use local `torch.compile` on shared expert and post RMSNorm paths.

## Main Results

Single H20, `bf16`, official local `benchmark.py`, `warmup=3`, `measure=5`:

| Implementation | SeqLen | Peak MB | Avg ms |
|---|---:|---:|---:|
| Official baseline wrapper | 8192 | 4348.38 | 48.47 |
| Official baseline wrapper | 131072 | 33639.01 | 560.33 |
| Final solution | 8192 | 4180.38 | 46.63 |
| Final solution | 131072 | 21946.01 | 542.41 |

The final code package was also rebenchmarked directly from `马胤+code.zip`; memory was unchanged and speed variation was within normal benchmark noise.

