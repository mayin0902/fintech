# 长序列训练 MoE Block 显存优化方案报告

参赛者：马胤

## 1. 任务理解

本赛题要求在不改变 MoE Block 数学定义的前提下，对训练过程中的峰值显存进行优化，并尽量控制单步训练耗时。评测入口为 `MoEBlockOptimized`，评测脚本会通过 `from solution import MoEBlockOptimized` 加载实现，并使用 `load_state_dict` 从基础实现加载权重。因此，本方案严格保持与基础实现一致的参数名称、参数形状和计算语义。

官方评分采用“正确性门槛 + 显存优化评分 + 速度评分”的方式。正确性全部通过后，再按显存与速度进行评分，其中显存与速度权重为 6:4。题面明确提到固定输入规模包括 8K 和 128K 长度，因此实验重点围绕这两个长度，同时额外检查更长序列和不同 batch size 下的稳定性。

## 2. 整体优化思路

基础实现的显存峰值主要来自三类张量：

1. 路由阶段构造 token 到 expert 的中间索引和 mask。
2. routed experts 和 shared expert 的中间激活，尤其是 shared expert 中 `gate_proj`、`up_proj`、`Silu(gate) * up` 对长序列较敏感。
3. 将 routed output 与 shared output 相加时，如果同时保留 `routed_output`、`shared_output` 和 `combined`，会额外产生接近 `[B*T, H]` 的 full-size 临时张量。

最终提交方案采用以下策略：

1. 将输入从 `[B, T, H]` 展平为 `[B*T, H]` 后统一处理，最后恢复形状。
2. 避免基础实现中较大的 one-hot expert mask，改为基于 `selected_experts` 查找每个 expert 命中的 token。
3. 对 expert 和 shared expert 按 token 维度分块计算，限制长序列下单次线性层激活的生命周期。
4. 将 shared expert 的输出直接原地累加到 routed output，避免额外保留一个完整的 `combined` 张量。
5. 对 shared expert 计算和 RMSNorm 使用 PyTorch `torch.compile` 进行局部编译，以减少 Python 调度和部分算子开销。

方案只使用 PyTorch 能力，不使用分布式并行，不修改模型结构、不修改路由定义、不修改输出定义，也不改变梯度数学定义。

## 3. 最终提交方案

最终提交实现可以概括为“分块计算 + 原地合并”的 MoE Block。

### 3.1 参数结构保持一致

提交文件中保留以下参数，与基础实现一致：

| 参数 | 形状 |
|---|---|
| `experts.gate_up_proj` | `[num_experts, 2*moe_intermediate_size, hidden_size]` |
| `experts.down_proj` | `[num_experts, hidden_size, moe_intermediate_size]` |
| `shared_expert.gate_proj.weight` | `[intermediate_size, hidden_size]` |
| `shared_expert.up_proj.weight` | `[intermediate_size, hidden_size]` |
| `shared_expert.down_proj.weight` | `[hidden_size, intermediate_size]` |
| `gate.weight` | `[num_experts, hidden_size]` |
| `post_norm.weight` | `[hidden_size]` |

最终提交包解压后，`solution.py` 可以通过 strict `load_state_dict` 加载基础实现权重，参数顺序也与基础实现一致。

### 3.2 路由计算

路由逻辑保持与基础实现一致：

1. 对展平后的 hidden states 计算 `router_logits = linear(hidden, gate.weight)`。
2. 对 logits 做 `softmax`。
3. 取 top-k expert 及其权重。
4. 在 `norm_topk_prob=True` 时，对 top-k 权重重新归一化。

该部分未改变数学定义，仅避免构造完整 one-hot mask，减少长序列下的临时张量。

### 3.3 Routed experts 分块计算

基础实现会根据 expert mask 遍历命中的 expert。最终方案改为：

1. 从 `selected_experts` 中找到实际被命中的 expert。
2. 对每个 expert，取出分配给它的 token 索引。
3. 按 token chunk 计算该 expert 的 `gate_up_proj`、激活乘法和 `down_proj`。
4. 乘以对应 top-k 权重后，通过 `index_add_` 累加回 routed output。

这样可以避免一次性保留过多 expert 中间激活，并降低路由 mask 的显存开销。

### 3.4 Shared expert 分块与原地合并

Shared expert 仍然计算：

```text
down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

优化点在于：

1. 对 token 维度分块，避免长序列下一次性生成过大的 shared expert 中间激活。
2. shared expert 的输出不再单独长期保留为 `shared_output`。
3. 计算完成后直接累加到 routed output：

```text
routed_output += shared_expert_output
```

基础数学定义中的 `combined = routed_output + shared_output` 没有改变，但实现上减少了一个 full-size 中间张量的生命周期，因此对长序列显存有直接帮助。

### 3.5 Post RMSNorm

RMSNorm 逻辑保持一致：

```text
x_float = x.to(float32)
variance = mean(x_float ** 2)
output = weight * x_float * rsqrt(variance + 1e-6)
```

最终再恢复为输入 dtype。该步骤与基础实现等价。

## 4. 探索过的主要方案

为了在显存、速度和隐藏长序列稳定性之间取舍，实验中主要比较了以下几类方案。

### 4.1 方案 A：Checkpoint 分块方案

该方案对 expert/shared expert 使用更小 chunk，并结合 activation checkpoint 降低长序列激活保存量。

代表性性能如下。该方案主要用于验证长序列容量，因此重点记录 327680 和 393216 长度：

| SeqLen | Peak Memory MB | Avg ms | 结果 |
|---:|---:|---:|---|
| 327680 | 51444.76 | 1657.94 | 通过 |
| 393216 | 61476.01 | 1982.80 | 通过 |

优点：

- 长序列显存低，能够覆盖更极端长度。
- 稳定性较好。

缺点：

- 反向阶段需要重算，速度明显变慢。
- 在 8K/128K 主要评分长度上，6:4 综合表现不如后续方案。

因此该方案仅作为保守 fallback，没有作为最终提交。

### 4.2 方案 B：大分块 + 局部编译方案

该方案关闭 checkpoint，使用较大的 expert/shared chunk，并对 shared expert 和 RMSNorm 使用 `torch.compile`。

该方案在中长序列上速度较好，但在更极端长度上显存余量不足：

| SeqLen | Peak Memory MB | Avg ms | 结果 |
|---:|---:|---:|---|
| 8192 | 4180.38 | 46.75 | 通过 |
| 32768 | 7151.01 | 146.68 | 通过 |
| 65536 | 11977.76 | 278.10 | 通过 |
| 131072 | 22594.02 | 542.33 | 通过 |
| 196608 | 32922.25 | 798.98 | 通过 |
| 262144 | 43537.51 | 1060.60 | 通过 |
| 327680 | 69307.00 | 1581.22 | 通过 |
| 393216 | -- | -- | OOM |

优点：

- 8K/128K 速度较好。
- 相比基础实现显著降低 128K 显存。

历史实验中，该方案在 8K 到 262K 范围内表现稳定，但在 393216 长度存在 OOM 风险。因此后续继续优化 full-size 中间张量生命周期。

### 4.3 方案 C：参数注册顺序修正

为了降低提交风险，单独验证了参数注册顺序问题。虽然 PyTorch strict load 主要按参数名匹配，但官方评测入口使用基础实现权重，因此最终方案要求参数名、参数形状和 `state_dict` 顺序均与基础实现一致。

该方案可以视为对方案 B 的提交安全修正，性能与方案 B 接近。一次验证中的代表性结果如下：

| SeqLen | Peak Memory MB | Avg ms | 结果 |
|---:|---:|---:|---|
| 8192 | 4180.38 | 46.61 | 通过 |
| 131072 | 22450.01 | 542.36 | 通过 |
| 262144 | 43610.51 | 1067.85 | 通过 |
| 327680 | 56786.77 | 1365.05 | 通过 |
| 393216 | -- | -- | OOM |

该方案本身不改变性能，但提高了提交格式和加载安全性。

### 4.4 方案 D：分块计算 + 原地合并方案

这是最终提交方案。它在方案 B 的基础上进一步减少 full-size 中间张量，尤其避免同时保留 routed output、shared output 和 combined。

该方案是最终选择。最后单卡复测结果如下：

| SeqLen | Peak Memory MB | Avg ms | 结果 |
|---:|---:|---:|---|
| 8192 | 4180.38 | 46.76 | 通过 |
| 131072 | 21946.01 | 542.42 | 通过 |
| 327680 | 55172.76 | 1364.79 | 通过 |
| 393216 | 70653.02 | 1631.85 | 通过 |

在另一轮 `warmup=5, measure=10` 的交叉复测中，该方案也保持稳定：

| SeqLen | Run1 Peak MB / Avg ms | Run2 Peak MB / Avg ms |
|---:|---:|---:|
| 8192 | 4180.38 / 46.78 | 4180.38 / 46.62 |
| 131072 | 21946.01 / 542.77 | 21946.01 / 542.25 |
| 262144 | 42602.51 / 1068.23 | 42602.51 / 1060.45 |
| 327680 | 55245.77 / 1365.00 | 55245.77 / 1355.43 |
| 393216 | 70727.02 / 1631.79 | 70727.02 / 1620.81 |

优点：

- 8K/128K 主要评分长度显存和速度均优于官方 baseline。
- 393216 长度可运行。
- 实现逻辑较简单，没有复杂的长序列阈值分支，提交风险较低。

该方案最终被选为提交版本。

### 4.5 方案 E：长序列自适应分块方案

该方案在超长序列时切换到更小 shared chunk，以进一步降低 327680/393216 长度的显存峰值。

代表性复测结果如下：

| SeqLen | Peak Memory MB | Avg ms | 结果 |
|---:|---:|---:|---|
| 8192 | 4180.38 | 46.69 | 通过 |
| 131072 | 21946.01 | 542.70 | 通过 |
| 327680 | 54922.14 | 1368.44 | 通过 |
| 393216 | 70141.02 | 1636.95 | 通过 |

与最终方案相比，该方案在 327680 和 393216 长度下显存分别约低 250 MB 和 512 MB，但速度略慢，且 8K/128K 主要评分长度没有形成稳定收益。

优点：

- 在 327680/393216 长度下显存略低。

缺点：

- 8K/128K 主要评分长度与最终方案基本打平，没有稳定可见收益。
- 超长序列速度存在运行噪声。
- 增加阈值分支，提交复杂度略高。

因此它保留为隐藏长序列风险备选，没有作为最终提交。

## 5. 正确性验证方法

正确性验证使用官方本地 `correctness_check.py`，对比基础实现和提交实现的：

1. 前向输出。
2. 输入梯度。
3. router、routed experts、shared expert、post_norm 的参数梯度。

判定阈值与题面一致：

| 项目 | 阈值 |
|---|---|
| 前向输出 | `rtol=2e-2, atol=1e-3` |
| 梯度 | `cosine_sim >= 0.995` 且 `relative_l2 <= 1e-2` |

最终提交文件额外做了格式与加载验证：

| 检查项 | 结果 |
|---|---|
| `python -m py_compile solution.py` | 通过 |
| `from solution import MoEBlockOptimized` | 通过 |
| strict `load_state_dict` | 通过 |
| 参数顺序与 baseline 一致 | 通过 |
| zip 解包后再次 import 和 strict load | 通过 |

### 5.1 最终提交正确性结果

最终提交版在 `bf16` 下通过以下正确性测试：

| Batch | SeqLen | Seed | 结果 |
|---:|---:|---:|---|
| 1 | 2048 | 42 | 通过 |
| 1 | 8192 | 42 | 通过 |
| 1 | 16384 | 42 | 通过 |
| 1 | 2048 | 123 | 通过 |
| 1 | 8192 | 123 | 通过 |
| 1 | 16384 | 123 | 通过 |

此外，为验证非 `B=1` 输入下的稳定性，补充测试：

| Batch | SeqLen | Seed | 结果 |
|---:|---:|---:|---|
| 2 | 2048 | 42 | 通过 |
| 2 | 8192 | 42 | 通过 |
| 4 | 2048 | 42 | 通过 |
| 4 | 4096 | 123 | 通过 |

所有测试均未发现 failure、traceback、CUDA error 或 OOM。

## 6. 性能实验结果

性能测试使用官方本地 `benchmark.py`，在单张 H20 上运行，dtype 为 `bf16`。除特别说明外，测试使用 `warmup=3`、`measure=5`。

### 6.1 与官方 baseline 对比

| 实现 | SeqLen | Peak Memory MB | Avg ms |
|---|---:|---:|---:|
| 官方 baseline | 8192 | 4348.38 | 48.47 |
| 官方 baseline | 131072 | 33639.01 | 560.33 |
| 最终提交方案 | 8192 | 4180.38 | 46.63 |
| 最终提交方案 | 131072 | 21946.01 | 542.41 |

相对官方 baseline：

| SeqLen | 显存降低 | 速度提升 |
|---:|---:|---:|
| 8192 | 约 3.86% | 约 3.80% |
| 131072 | 约 34.76% | 约 3.20% |

按 8K/128K 等权、显存:速度 = 6:4 的 proxy 估计，最终提交方案相对官方 baseline 的综合比值约为 `1.186`。

### 6.2 更长序列稳定性

为评估隐藏长序列风险，额外测试了 327680 和 393216 长度：

| 实现 | SeqLen | Peak Memory MB | Avg ms | 结果 |
|---|---:|---:|---:|---|
| 最终提交方案 | 327680 | 55172.76 | 1364.79 | 通过 |
| 最终提交方案 | 393216 | 70653.02 | 1631.85 | 通过 |

这说明最终方案在显存 96GB 的 H20 单卡环境下，对超出题面显式 8K/128K 的更长序列也有一定余量。

### 6.3 不同 batch size 的稳定性

由于最终实现将输入展平为 `[B*T, H]` 处理，因此理论上性能主要随总 token 数 `B*T` 变化，而不是由 batch 维度单独决定。实际测试结果如下：

| Batch | SeqLen | Total Tokens | Peak Memory MB | Avg ms |
|---:|---:|---:|---:|---:|
| 1 | 8192 | 8192 | 4180.38 | 46.74 |
| 2 | 4096 | 8192 | 4180.38 | 46.67 |
| 4 | 2048 | 8192 | 4180.38 | 46.69 |
| 1 | 16384 | 16384 | 5146.26 | 80.29 |
| 2 | 8192 | 16384 | 5146.26 | 80.24 |
| 4 | 4096 | 16384 | 5146.26 | 80.14 |

同样总 token 数下，不同 batch size 的显存一致，速度差异处于测量噪声范围内。

## 7. 速度与显存权衡分析

本题的主要矛盾是显存峰值与重算开销之间的权衡。

Checkpoint 分块方案能够进一步降低长序列显存，但反向阶段需要重算，使单步耗时明显增加。考虑到评分中速度占 40%，且题面明确关注 8K/128K，本方案没有采用 checkpoint 作为默认路径。

长序列自适应分块方案在 327680/393216 长度下显存略低，但主要评分长度 8K/128K 没有稳定提升，同时引入阈值分支和额外复杂度。最终没有选择该方案。

最终提交方案采用较保守的折中：

1. 通过分块限制 expert/shared expert 激活生命周期。
2. 通过原地合并去掉一个 full-size 中间张量。
3. 不启用 checkpoint，避免明显速度损失。
4. 不使用复杂自适应分支，降低隐藏输入形状下的行为风险。

该选择使 128K 显存下降幅度较大，同时保持 8K/128K 速度均优于 baseline。

## 8. 合规性说明

最终提交符合以下约束：

1. 单卡运行，不依赖多卡并行或分布式初始化。
2. 只使用 PyTorch，不依赖额外非标准运行时。
3. 不改变模型数学定义。
4. 不改变路由定义、top-k 权重归一化逻辑或 RMSNorm 逻辑。
5. 参数名称和形状与基础实现一致，可 strict load 基础实现权重。
6. 提交包 `马胤+code.zip` 根目录只包含 `solution.py`。

提交包校验结果：

| 项目 | 结果 |
|---|---|
| zip 根目录文件 | 仅 `solution.py` |
| `solution.py` SHA256 | `8e60f420fc081b738b758c84cdbd4be19a0b793d66f9bd4309d692656b87419e` |
| zip 解包后 import | 通过 |
| zip 解包后 strict load | 通过 |
| zip 解包后 2048 正确性 smoke | 通过 |

## 9. 主要结论

最终提交方案在保持数学等价和训练梯度正确的前提下，降低了 MoE Block 的训练峰值显存，并保持了较好的单步耗时。

在官方题面重点提到的 8K 和 128K 长度上，最终方案相对 baseline：

1. 8K 显存降低约 3.86%，速度提升约 3.80%。
2. 128K 显存降低约 34.76%，速度提升约 3.20%。
3. 6:4 综合 proxy 约为 baseline 的 1.186 倍。

实验表明，最终方案对不同 batch size 具有稳定行为，并且在 327680/393216 等更长序列下也能正常运行。综合正确性、显存收益、速度表现和提交风险，选择“分块计算 + 原地合并”方案作为最终提交。
