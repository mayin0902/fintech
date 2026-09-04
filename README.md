# Fintech Contest

金融科技竞赛与工程实践作品集，包含显存优化和多轮 Text-to-SQL 后训练两条技术主线。

## Track 2 · Execution-Grounded Text-to-SQL Agent（简历项目）

> 构建 PostgreSQL 真执行环境作为 Text-to-SQL Agent 反馈闭环，设计 Schema grounding、错误归因与反思纠错 pipeline，将候选/训练标签 SQL 执行正确率由 **36.6% 提升至 97.32%（5484/5635）**。

核心链路：

```text
用户问题 → Schema 理解 → SQL 生成 → 数据库执行 → 错误反馈 → Reward 优化
                                  ↑          ↓
                                  └── 反思纠错 ┘
```

代码、复现实验、指标口径和面试讲义见 [`track2-text2sql/`](track2-text2sql/README.md)。

训练框架使用保留上游归属的 [`mayin0902/twinkle` fork](https://github.com/mayin0902/twinkle)，项目自有适配见 [`track2-text2sql/framework/`](track2-text2sql/framework/README.md)。

其中包含：

- PostgreSQL 只读/超时/时区一致的安全执行器；
- execution match、多重集合比对、稠密 reward 与 Agent retry loop；
- RFT、DPO、GRPO、PPO 原理及可读参考实现；
- vLLM、PagedAttention、KV Cache、prefix caching 与 FP8 KV 工程说明；
- 30 秒/2 分钟项目讲法与高频面试追问。

> 指标说明：97.32% 是候选/训练标签 SQL 的真库执行正确率，不是最终模型 held-out 推理准确率。公开内容已脱敏，不含原始数据、数据库 dump、账号凭据和模型权重。

## Track 1 · MoE Block 显存优化

赛道 1 是长序列训练 MoE Block 显存优化任务。当前已整理：

- `track1/赛道1.md`：赛题要求
- `track1/official/`：附件解压后的 baseline、solution 模板、correctness 和 benchmark 脚本
- `track1/赛道1_24G双卡测试方案与执行计划.md`：24GB 双卡测试策略与执行计划
- `track1/赛道1_文献资料与实验策略.md`：文献资料依据和分显存实验策略
- `track1/results_2026-05-13_gpu0_gpu3.md`：GPU0/GPU3 本地参考测试结果
- `track1/results_2026-05-13_48gb_gpu1.md`：48GB GPU1 调试结果和默认参数更新依据
- `track1/赛道1_H20_A800最终实验策略_2026-05-17前.md`：H20/A800 官方尺度最终实验策略

