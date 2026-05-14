# Track 1 48GB GPU1 Tuning Results

Date: 2026-05-13

## Environment

- Host GPU: NVIDIA GeForce RTX 4090, 49140 MiB shown by `nvidia-smi`
- Tested GPU: physical GPU 1 via `CUDA_VISIBLE_DEVICES=1`
- Driver: 595.45.04
- CUDA reported by `nvidia-smi`: 13.2
- Python environment: `/home/caoxiangyu/miniconda3/envs/chemformer_bo/bin/python`
- Python: 3.10.20
- PyTorch: 2.6.0+cu124
- Note: official target environment is Python 3.12 + PyTorch 2.8.0 + CUDA 12.8, so these are local 48GB reference results.

## Strategy Readout

The 24GB search established that the current pure PyTorch implementation is correctness-stable and that `ckpt_s8192_e8192` can run 131072 tokens at about 21.3GB peak memory. The 48GB card was then used to spend memory on larger chunks and disable checkpointing for the official-scale 8K/128K cases.

The key result is that no-checkpoint full-chunk execution fits 131072 tokens on 48GB and is much faster than the 24GB checkpoint path.

## Correctness

Command:

```bash
CUDA_VISIBLE_DEVICES=1 \
MOE_SHARED_CHUNK_SIZE=131072 \
MOE_EXPERT_CHUNK_SIZE=131072 \
MOE_USE_CHECKPOINT=0 \
/home/caoxiangyu/miniconda3/envs/chemformer_bo/bin/python correctness_check.py \
  --solution solution.py \
  --seq-len 8192
```

Result: all checks passed.

- Forward output: pass
- Input grad: pass
- Parameter grads: pass for `gate.weight`, expert weights, shared expert weights, and `post_norm.weight`

## 131072 Variant Sweep

All variants below used `MOE_USE_CHECKPOINT=0`.

| Variant | Shared chunk | Expert chunk | Peak MB | Avg ms | Measures |
|---|---:|---:|---:|---:|---:|
| `nocp_s8192_e8192` | 8192 | 8192 | 33591.26 | 1553.73 | 2 |
| `nocp_s16384_e8192` | 16384 | 8192 | 33591.26 | 1520.96 | 2 |
| `nocp_s16384_e16384` | 16384 | 16384 | 33591.26 | 1057.46 | 2 |
| `nocp_s32768_e8192` | 32768 | 8192 | 33591.26 | 1507.33 | 2 |
| `nocp_s32768_e16384` | 32768 | 16384 | 33591.26 | 1044.44 | 2 |
| `nocp_s32768_e32768` | 32768 | 32768 | 33591.26 | 809.68 | 2 |
| `nocp_s65536_e32768` | 65536 | 32768 | 33591.26 | 802.66 | 2 |
| `nocp_s65536_e65536` | 65536 | 65536 | 33591.26 | 684.30 | 2 |
| `nocp_s131072_e65536` | 131072 | 65536 | 33591.26 | 679.30 | 2 |
| `nocp_s131072_e131072` | 131072 | 131072 | 33591.26 | 618.28 | 2 |

Best 131072 result:

```text
MOE_SHARED_CHUNK_SIZE=131072
MOE_EXPERT_CHUNK_SIZE=131072
MOE_USE_CHECKPOINT=0
```

## Multi-Length Retest With Best 48GB Default

Command:

```bash
CUDA_VISIBLE_DEVICES=1 \
MOE_SHARED_CHUNK_SIZE=131072 \
MOE_EXPERT_CHUNK_SIZE=131072 \
MOE_USE_CHECKPOINT=0 \
/home/caoxiangyu/miniconda3/envs/chemformer_bo/bin/python benchmark.py \
  --solution solution.py \
  --seq-lens 8192,32768,65536,131072 \
  --warmup 2 \
  --measure 5
```

| SeqLen | Peak MB | Avg ms | Min ms | Max ms |
|---:|---:|---:|---:|---:|
| 8192 | 4300.63 | 72.89 | 72.81 | 72.93 |
| 32768 | 9328.38 | 182.32 | 182.10 | 182.51 |
| 65536 | 17416.01 | 326.61 | 326.21 | 327.52 |
| 131072 | 33591.26 | 620.49 | 618.23 | 623.71 |

After updating `solution.py` defaults to the best 48GB path, a no-env 131072 retest produced:

| SeqLen | Peak MB | Avg ms | Min ms | Max ms |
|---:|---:|---:|---:|---:|
| 131072 | 33591.26 | 617.19 | 616.83 | 617.56 |

## Longer-Sequence Fallback

No-checkpoint full-chunk execution OOMed at 196608 tokens:

```text
MOE_SHARED_CHUNK_SIZE=131072
MOE_EXPERT_CHUNK_SIZE=131072
MOE_USE_CHECKPOINT=0
seq_len=196608 -> OOM
```

Checkpointed full-chunk execution ran longer sequences:

```bash
CUDA_VISIBLE_DEVICES=1 \
MOE_SHARED_CHUNK_SIZE=131072 \
MOE_EXPERT_CHUNK_SIZE=131072 \
MOE_USE_CHECKPOINT=1 \
/home/caoxiangyu/miniconda3/envs/chemformer_bo/bin/python benchmark.py \
  --solution solution.py \
  --seq-lens 196608,262144 \
  --warmup 1 \
  --measure 3
```

| SeqLen | Peak MB | Avg ms | Min ms | Max ms |
|---:|---:|---:|---:|---:|
| 196608 | 32943.38 | 1129.68 | 1127.38 | 1131.52 |
| 262144 | 41366.63 | 1485.82 | 1484.98 | 1486.89 |

## Implementation Change

The default runtime knobs in `solution.py` were changed from the 24GB-safe path:

```text
MOE_SHARED_CHUNK_SIZE=2048
MOE_EXPERT_CHUNK_SIZE=2048
MOE_USE_CHECKPOINT=1
```

to the 48GB/H20-oriented path:

```text
MOE_SHARED_CHUNK_SIZE=131072
MOE_EXPERT_CHUNK_SIZE=131072
MOE_USE_CHECKPOINT=0
```

Environment variables still override these defaults. For 24GB testing or very long hidden cases, set `MOE_USE_CHECKPOINT=1`.

## Recommendation

- For the official 8K/128K workload, keep the default no-checkpoint `131072/131072` path.
- For 24GB cards, override with `MOE_USE_CHECKPOINT=1` and smaller chunks if needed.
- For longer-than-official stress tests such as 196608/262144, use checkpointed full chunks on 48GB.
- Before final submission, rerun on the official Python 3.12 + Torch 2.8.0 + CUDA 12.8 environment.
