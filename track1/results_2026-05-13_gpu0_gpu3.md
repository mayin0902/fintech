# Track 1 GPU0/GPU3 Reference Results

Date: 2026-05-13

## Environment

- Host GPU: NVIDIA GeForce RTX 4090, 24564 MiB each
- Driver: 575.57.08
- CUDA reported by `nvidia-smi`: 12.9
- Local Python environment: `/home/mayin/deepmd-kit/bin/python`
- Local PyTorch version observed before tests: 2.6.0
- Note: official target environment is Python 3.12 + PyTorch 2.8.0 + CUDA 12.8, so these are local 24GB reference results, not final official scores.

## Solution Variant

File: `track1/official/solution.py`

- Pure PyTorch implementation.
- Removes baseline one-hot expert mask.
- Processes routed experts by hit expert and token chunk.
- Processes shared expert by token chunk.
- Uses non-reentrant activation checkpointing by default.
- Default runtime knobs:
  - `MOE_SHARED_CHUNK_SIZE=2048`
  - `MOE_EXPERT_CHUNK_SIZE=2048`
  - `MOE_USE_CHECKPOINT=1`

## GPU0 Correctness

Command:

```bash
cd /home/mayin/fintech/track1/official
env CUDA_VISIBLE_DEVICES=0 python correctness_check.py --solution solution.py --seq-len 2048
```

Result: all checks passed.

- Device: cuda
- dtype: torch.bfloat16
- Shape: B=1, T=2048, H=2048
- Forward output: pass
- Input grad: pass
- Parameter grads: pass for `gate.weight`, expert weights, shared expert weights, and `post_norm.weight`

Command:

```bash
cd /home/mayin/fintech/track1/official
env CUDA_VISIBLE_DEVICES=0 python correctness_check.py --solution solution.py --seq-len 8192
```

Result: all checks passed.

Notable nonzero diffs remain inside thresholds:

| Check | max_abs_diff | rel_l2 | cosine |
|---|---:|---:|---:|
| input grad | 1.464844e-03 | 2.826862e-03 | 0.999996 |
| shared gate grad | 6.250000e-02 | 3.925469e-03 | 0.999992 |
| shared up grad | 6.250000e-02 | 3.925290e-03 | 0.999992 |
| shared down grad | 1.250000e-01 | 3.922495e-03 | 0.999992 |

## GPU3 Benchmark

Command:

```bash
cd /home/mayin/fintech/track1/official
env CUDA_VISIBLE_DEVICES=3 python benchmark.py --solution solution.py --seq-lens 2048,8192 --warmup 1 --measure 2
```

| SeqLen | Peak Memory MB | Avg ms | Min ms | Max ms |
|---:|---:|---:|---:|---:|
| 2048 | 3432.23 | 118.53 | 60.85 | 176.20 |
| 8192 | 4132.78 | 245.25 | 242.84 | 247.67 |

Command:

```bash
cd /home/mayin/fintech/track1/official
env CUDA_VISIBLE_DEVICES=3 python benchmark.py --solution solution.py --seq-lens 32768 --warmup 1 --measure 1
```

| SeqLen | Peak Memory MB | Avg ms | Min ms | Max ms |
|---:|---:|---:|---:|---:|
| 32768 | 6934.88 | 1072.51 | 1072.51 | 1072.51 |

## Next Validation

- Run `65536` and `131072` on 48GB/80GB/H20-96GB before treating the result as final.
- On 24GB cards, run each large seq length in a fresh process to avoid OOM fragmentation.
- If speed is too slow, first tune `MOE_SHARED_CHUNK_SIZE` and `MOE_EXPERT_CHUNK_SIZE`; then consider disabling checkpoint for shorter sequences with `MOE_USE_CHECKPOINT=0`.
