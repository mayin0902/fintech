# Training framework integration

本项目的 NPU 训练实验基于开源框架 [ModelScope Twinkle](https://github.com/modelscope/twinkle)。为方便复现与阅读，已在个人 GitHub 建立保留上游关系和 Apache-2.0 许可证的 [mayin0902/twinkle fork](https://github.com/mayin0902/twinkle)。

框架原理、对象关系、GRPO/DPO/PPO 调用链、vLLM 权重同步和 NPU 排障，集中整理在 [Twinkle 框架源码级面试讲义](../docs/twinkle-framework-interview.md)。本适配已按上游 [`1040a02`](https://github.com/modelscope/twinkle/commit/1040a02d08c390031800093336718589160b52af) 的公开 API 校对；真实训练仍应固定依赖版本和镜像，避免直接跟随 `main` 漂移。

## 归属边界

- Twinkle 的核心框架、GRPO/DPO/PPO loss、vLLM sampler、分布式调度与文档属于 ModelScope Contributors；
- 本目录只放本项目编写的 Text-to-SQL 数据契约和 Qwen3.5/NPU 训练适配；
- 简历中应说“基于 Twinkle 适配”，不把上游框架写成个人原创。

## 值得读的上游实现

| 模块 | 个人 fork 中的源码 |
|---|---|
| GRPO 组内 advantage | [`src/twinkle/advantage/grpo.py`](https://github.com/mayin0902/twinkle/blob/main/src/twinkle/advantage/grpo.py) |
| GRPO/GSPO/SAPO loss | [`src/twinkle/loss/grpo.py`](https://github.com/mayin0902/twinkle/blob/main/src/twinkle/loss/grpo.py) |
| DPO/SimPO/CPO/ORPO loss | [`src/twinkle/loss/dpo.py`](https://github.com/mayin0902/twinkle/blob/main/src/twinkle/loss/dpo.py) |
| vLLM rollout sampler | [`src/twinkle/sampler/vllm_sampler/`](https://github.com/mayin0902/twinkle/tree/main/src/twinkle/sampler/vllm_sampler) |
| 完整 GRPO cookbook | [`cookbook/rl/grpo/grpo.py`](https://github.com/mayin0902/twinkle/blob/main/cookbook/rl/grpo/grpo.py) |
| LoRA DPO cookbook | [`cookbook/rl/dpo/dpo_lora.py`](https://github.com/mayin0902/twinkle/blob/main/cookbook/rl/dpo/dpo_lora.py) |
| PPO cookbook | [`cookbook/rl/ppo/ppo.py`](https://github.com/mayin0902/twinkle/blob/main/cookbook/rl/ppo/ppo.py) |

## 本项目适配

- `twinkle_sft_qwen35_npu.py`：verified-only Text-to-SQL 数据读取、格式硬闸门、Qwen3.5 LoRA、FSDP/DP 设备网格、断点保存；
- `run_twinkle_sft_npu.sh`：Ascend/CANN/HCCL 环境入口，不包含任何机器私有路径；
- `../src/text2sql_feedback/trl_adapter.py`：PostgreSQL reward 的并发训练适配，可迁移到 Twinkle reward 接口；
- `../scripts/train_grpo.py`、`../scripts/train_dpo.py`：较轻量的 TRL 对照实现，便于核对数据契约和算法行为。

## 运行

先按 [Twinkle 官方安装说明](https://github.com/modelscope/twinkle#install) 配置与 NPU/CANN 版本匹配的环境，然后：

```bash
export MODEL_ID=ms://Qwen/Qwen3.5-2B
export TRAIN_DATASET_ID=/path/to/verified_only.jsonl
export OUTPUT_DIR=artifacts/qwen35-text2sql-lora
export NUM_NPUS=2 FSDP_SIZE=2 DP_SIZE=1 BATCH_SIZE=2
bash framework/run_twinkle_sft_npu.sh
```

训练数据最后一条 message 必须是 assistant，并且内容必须是唯一的小写 ```` ```sql ... ``` ```` 代码块。脚本会在编码前执行格式过滤，避免坏格式样本静默进入训练。
