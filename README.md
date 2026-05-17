# Fintech Contest

本仓库用于整理招商银行金融科技竞赛资料、测试计划、实现代码和赛后复盘材料。

## 项目概览

当前主要内容是赛道 1：长序列训练 MoE Block 显存优化。目标是在不改变模型数学定义、不使用分布式并行的前提下，降低训练时峰值显存，并尽量保持单步训练速度。

最终提交方案采用“分块计算 + 原地合并”：

- 保持与官方 `MoEBlockBaseline` 一致的参数名称、形状和 strict load 行为。
- 将 `[B, T, H]` 展平为 `[B*T, H]` 后进行路由和 expert 计算。
- 避免构造大 one-hot expert mask。
- 对 routed experts 和 shared expert 沿 token 维度分块。
- 将 shared expert 输出直接累加到 routed output，减少一个 full-size 中间张量。

最终 8K/128K 单卡 H20 结果：

| 实现 | SeqLen | Peak MB | Avg ms |
|---|---:|---:|---:|
| 官方 baseline wrapper | 8192 | 4348.38 | 48.47 |
| 官方 baseline wrapper | 131072 | 33639.01 | 560.33 |
| 最终方案 | 8192 | 4180.38 | 46.63 |
| 最终方案 | 131072 | 21946.01 | 542.41 |

按 8K/128K 等权、显存:速度 = 6:4 的 proxy 估计，最终方案约为 baseline 的 `1.186x`。

## Track 1

赛道1是长序列训练 MoE Block 显存优化任务。当前已整理：

- `track1/赛道1.md`：赛题要求
- `track1/official/`：附件解压后的 baseline、solution 模板、correctness 和 benchmark 脚本
- `track1/final_submission_20260517/`：最终提交源码、代码 zip、报告 Markdown 和验证摘要
- `track1/experiment_archive_20260517/`：精选中间候选、关键实验结果和复盘材料
- `track1/赛道1_24G双卡测试方案与执行计划.md`：24GB 双卡测试策略与执行计划
- `track1/赛道1_文献资料与实验策略.md`：文献资料、全网资料依据和分显存实验策略
- `track1/results_2026-05-13_gpu0_gpu3.md`：GPU0/GPU3 本地参考测试结果
- `track1/results_2026-05-13_48gb_gpu1.md`：48GB GPU1 调试结果和默认参数更新依据
- `track1/results_2026-05-14_h20_96gb.md`：2 张 H20 96GB 最终调优结果和默认提交配置
- `track1/results_2026-05-14_h20_final_default.md`：H20 最终默认配置高重复复测和冻结建议
- `track1/赛道1_H20_A800最终实验策略_2026-05-17前.md`：H20/A800 官方尺度最终实验策略和截止日前执行计划
- `track1/H20_最终策略与5月15日前GoalPrompt.md`：H20 最终策略、5.15 前任务清单和可直接复制的 goal prompt

> 原始 `附件1.zip` 已在本地保留于 `/home/mayin/fintech/附件1.zip`；仓库中已上传其解压后的全部官方源码文件，便于直接 clone 后运行。
