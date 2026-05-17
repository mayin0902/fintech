# H20 Track1 Final Experiment Summary - 2026-05-17

## Executive Conclusion

当前推荐的最终提交候选仍是：

`/root/autodl-tmp/fintech/track1/official/solution_candidate_sprint_shared_add_inplace.py`

推荐理由：

- 它是目前最稳妥的 visible-score 候选。
- `8192/131072` 的 6:4 proxy 相比 ordered reward 约 `+0.63%` 到 `+0.73%`。
- 已通过 `2048/8192/16384` correctness。
- strict load 和 baseline 参数注册顺序均通过。
- `262144/327680/393216` 均可运行。
- 它是自包含候选文件，且已有更充分的 soak 历史，更适合按 visible-score 优先作为最终打包候选。

最接近的隐藏风险备选现在是：

`solution_candidate_sprint_shared_add_inplace_adaptive_long_t327680_flat.py`

它已经由 wrapper 近邻 flatten 成单文件候选，并通过 `py_compile`、strict load/order、`2048/8192/16384` correctness 和双卡 benchmark 复测。它仍然不是主推，原因是 visible `8192/131072` 没有形成稳定胜出，只是隐藏长序列显存更低。

## Scoring Interpretation

赛题说明中明确显存:速度权重为 `6:4`，并写明固定输入规模包括 `8k` 和 `128k`，同时使用“包括但不限于”的措辞。因此最终判断采用：

```text
memory_score = reference_peak_mb / candidate_peak_mb
speed_score  = reference_avg_ms / candidate_avg_ms
proxy_6_4    = 0.6 * memory_score + 0.4 * speed_score
final_proxy  = mean(proxy_6_4 at 8192, proxy_6_4 at 131072)
```

实际取舍：

- 主评分优先：`8192/131072` 的 6:4 proxy。
- 安全检查：`262144/327680/393216` 不作为唯一主指标，但用来评估隐藏风险。
- 提交安全：最终 `solution.py` 必须是单文件、自包含、no-env 默认行为明确、strict load/order 通过。

## Experiment Timeline

### 1. 初始低显存路线：checkpoint fallback

早期确定的低显存稳定路线是：

`default_ckpt_s65536_e196608`

特点：

- 通过 `393216`。
- 128K/262K 显存低。
- 速度明显慢，visible 6:4 proxy 不如 reward 分支。

它保留为灾备 fallback，而不是主提交。

### 2. H20 reward 路线：compile shared + post

在 H20 96GB 上，关闭 checkpoint 并放大 chunk：

```text
shared_chunk = 262144
expert_chunk = 262144
checkpoint = off
compile = _shared_chunk + _post_norm
```

得到 `solution_candidate_compile_shared_post.py` / packaged reward `solution.py`。

关键历史结果：

| Seq len | Peak MB | Avg ms | Status |
|---:|---:|---:|---|
| 8192 | 4180.38 | 46.5-46.7 | pass |
| 131072 | 22594.02 | 538.7-542.4 | pass |
| 262144 | 43537.51 | 1060.6-1068.1 | pass |
| 327680 | 69307.00 | 1606.05 | pass |
| 393216 | OOM | -- | fail |

这一路线成为第一版高分候选，但有两个问题：

- `393216` OOM。
- 旧 packaged `solution.py` 的参数注册顺序不匹配 baseline 顺序，虽 strict load 可过，但提交风险不够干净。

### 3. Ordered reward 路线：修复顺序风险

随后创建：

`solution_candidate_reward_ordered.py`

作用：

- 保持 reward 行为。
- 修复 baseline 参数注册顺序。
- 作为后续所有 proxy 比较的 reference。

它本身仍然 OOM at `393216`，所以不是最终隐藏风险最优方案。

### 4. Aggressive visible 分支：shared add inplace

关键突破来自：

`solution_candidate_sprint_shared_add_inplace.py`

核心变化：

- 不再显式保留 `combined = routed_output + shared_output` 的额外 full tensor。
- 将 shared expert 输出直接累加到 routed output。
- 保持数学定义不变。

7h sprint 中的结果：

| Candidate | Final 6:4 proxy vs ordered reward | 393216 |
|---|---:|---|
| `shared_add_inplace` | `1.007306` | pass |
| `shared_add_inplace_adaptive_long` | `1.003598` | pass |
| `reward_ordered_solution_py` | `1.000000` | OOM |

1-2h final validation 中的结果：

| Candidate | Final 6:4 proxy vs ordered reward | 393216 |
|---|---:|---|
| `shared_add_inplace` | `1.006330` | pass, `70727.02 MB / 1632.08 ms` |
| `shared_add_inplace` repeat | `1.006841` | pass, `70727.02 MB / 1620.84 ms` |
| `shared_add_inplace_adaptive_long` | `1.006145` | pass, `65350.02 MB / 1625.64 ms` |
| `t327680_s131072_e262144` | `1.006831` | pass, `70215.02 MB / 1625.58 ms` |
| `t393216_s98304_e262144` | `1.006196` | pass, `65222.02 MB / 1637.92 ms` |

当时结论：`shared_add_inplace` 证据最充分，adaptive-long 家族作为隐藏风险备选。

### 5. 最后 30 分钟冲刺：`shared_add_inplace` vs `t327680`

本轮只做窄范围最终对照：

- `shared_add_inplace`
- `shared_add_inplace_adaptive_long_t327680_s131072_e262144`

新增 correctness：

| Candidate | 16384 correctness |
|---|---|
| `shared_add_inplace` | pass |
| `t327680` | pass |

warmup=5, measure=10 的两轮 benchmark：

| Candidate/run | 8192 MB/ms | 131072 MB/ms | 262144 MB/ms | 327680 MB/ms | 393216 MB/ms |
|---|---|---|---|---|---|
| `shared_add_inplace` run1 | 4180.38 / 46.78 | 21946.01 / 542.77 | 42602.51 / 1068.23 | 55245.77 / 1365.00 | 70727.02 / 1631.79 |
| `t327680` run1 | 4180.38 / 46.57 | 21946.01 / 542.20 | 42602.51 / 1066.90 | 54995.14 / 1358.43 | 70215.02 / 1624.74 |
| `shared_add_inplace` swap | 4180.38 / 46.62 | 21946.01 / 542.25 | 42602.51 / 1060.45 | 55245.77 / 1355.43 | 70727.02 / 1620.81 |
| `t327680` swap | 4180.38 / 46.59 | 21946.01 / 542.32 | 42602.51 / 1067.92 | 54995.14 / 1368.12 | 70215.02 / 1636.17 |

解读：

- `8192/131072` visible 指标上，二者基本打平。
- `t327680` 长序列峰值显存更低：
  - `327680`: `55245.77 -> 54995.14 MB`
  - `393216`: `70727.02 -> 70215.02 MB`
- `t327680` 长序列速度在 role swap 后不稳定，不能说稳定更快。
- 此阶段的 `t327680` 还是 wrapper 候选，不是最终单文件提交形态；后续第 6 节已单独复测 flatten 版本并解除该提交形态风险。

因此本轮没有足够理由替换主推荐。

### 6. 重要风险项复测：flatten `t327680`

随后针对前述重要风险项单独复测：

- 风险 1：`t327680` 原来是 wrapper，不是最终单文件提交形态。
- 风险 2：`t327680` 长序列速度有 role-swap 噪声。

新建单文件候选：

`solution_candidate_sprint_shared_add_inplace_adaptive_long_t327680_flat.py`

该文件从完整 adaptive-long 实现复制而来，只修改：

```text
long_threshold = 327680
long_shared_chunk_size = 131072
long_expert_chunk_size = 262144
```

门槛结果：

| Gate | Result |
|---|---|
| `py_compile` | pass |
| strict load | pass |
| state_dict order matches baseline | pass |
| correctness 2048 | pass |
| correctness 8192 | pass |
| correctness 16384 | pass |

双卡 benchmark：

| Run | 8192 | 131072 | 262144 | 327680 | 393216 |
|---|---|---|---|---|---|
| flat t327680 GPU0 | 4180.38 / 46.64 | 21946.01 / 542.42 | 42602.51 / 1067.90 | 54995.14 / 1367.90 | 70215.02 / 1636.39 |
| flat t327680 GPU1 | 4180.38 / 46.72 | 21946.01 / 542.64 | 42602.51 / 1068.10 | 54995.14 / 1358.55 | 70215.02 / 1625.23 |

复测结论：

- wrapper/submission-shape 风险已解除。
- flat 版 visible `8192/131072` 仍与 `shared_add_inplace` 基本打平，不是稳定可见分数突破。
- flat 版 long-seq 显存更低，但 long-seq 速度仍有运行噪声。
- 因此它从“wrapper 风险备选”升级为“可提交形态的 hidden-risk 备选”，但不替代主推。

## Final Candidate Ranking

### Primary

`solution_candidate_sprint_shared_add_inplace.py`

用途：

- 以 visible 8K/128K 6:4 得分为主的最终候选。
- 兼顾 `393216` 可运行。
- 自包含，最接近最终提交形态。

### Hidden-Risk Alternative

`solution_candidate_sprint_shared_add_inplace_adaptive_long_t327680_flat.py`

用途：

- 如果特别担心隐藏长序列显存峰值，可考虑。
- 已经是单文件候选，可复制成 `solution.py` 后进入 final gates。
- 仍需要最终 gates，因为它的 soak 历史少于主推。

### Conservative Fallback

`solution_candidate_reward_ordered.py`

用途：

- 保守回退。
- strict load/order 通过。
- 但 `393216` OOM，visible proxy 也低于 shared-add 系列。

## Current File Status

- 本轮没有覆盖 `official/solution.py`。
- 本轮没有打包。
- 本轮没有生成 `code.zip`。
- 当前官方 `solution.py` 仍不是最终推荐候选；最终提交前需要显式复制候选并重跑 gates。

## Final Packaging Recommendation

如果现在进入最终打包，建议路径：

1. 保存 restore：

```bash
cp /root/autodl-tmp/fintech/track1/official/solution.py /root/autodl-tmp/fintech/track1/official/restore_before_final_submit_solution.py
```

2. 复制主推荐：

```bash
cp /root/autodl-tmp/fintech/track1/official/solution_candidate_sprint_shared_add_inplace.py /root/autodl-tmp/fintech/track1/official/solution.py
```

3. 立刻重跑 final gates：

```bash
cd /root/autodl-tmp/fintech/track1/official
python -m py_compile solution.py
CUDA_VISIBLE_DEVICES=0 python correctness_check.py --solution solution.py --seq-len 2048
CUDA_VISIBLE_DEVICES=0 python correctness_check.py --solution solution.py --seq-len 8192
CUDA_VISIBLE_DEVICES=0 python correctness_check.py --solution solution.py --seq-len 16384
CUDA_VISIBLE_DEVICES=0 python benchmark.py --solution solution.py --seq-lens 8192,131072,262144,327680,393216 --warmup 5 --measure 10
```

4. 只有 gates 全部通过后再创建提交 zip。

## Evidence Map

Key documents:

- 8-10h final-shape: `/root/autodl-tmp/fintech/track1/results_2026-05-16_h20_8_10h_final_shape.md`
- 6h reward packaging: `/root/autodl-tmp/fintech/track1/results_2026-05-16_h20_6h_reward_packaging.md`
- 7h final sprint: `/root/autodl-tmp/fintech/track1/h20_runs/20260516_161644_7h_final_sprint_autonomous/final_recommendation.md`
- 1-2h validation: `/root/autodl-tmp/fintech/track1/h20_runs/20260517_1_2h_final_validation_interactive/final_recommendation.md`
- 30m final sprint: `/root/autodl-tmp/fintech/track1/h20_runs/20260517_30m_final_sprint_and_report/final_30m_summary.md`

Raw latest logs:

- `/root/autodl-tmp/fintech/track1/h20_runs/20260517_30m_final_sprint_and_report/benchmark_shared_add_inplace_w5_m10.log`
- `/root/autodl-tmp/fintech/track1/h20_runs/20260517_30m_final_sprint_and_report/benchmark_t327680_w5_m10.log`
- `/root/autodl-tmp/fintech/track1/h20_runs/20260517_30m_final_sprint_and_report/benchmark_shared_add_inplace_w5_m10_gpu1_swap.log`
- `/root/autodl-tmp/fintech/track1/h20_runs/20260517_30m_final_sprint_and_report/benchmark_t327680_w5_m10_gpu0_swap.log`
