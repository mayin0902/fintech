# 面试讲法与追问准备

## 30 秒版本

“我负责的是多轮 Text-to-SQL 的数据与后训练闭环。核心不是再写一个 prompt，而是把 PostgreSQL 做成可验证环境：模型先做 Schema grounding 和 SQL 生成，执行器用只读事务、超时和统一时区运行，再把语法、表列、列数、行数或结果差异变成反思反馈。所有修复候选都必须回库复验，最终把候选/训练标签的执行正确率从 36.6% 提升到 97.32%，也把同一套 verifier 接成了后续 RFT、DPO 和 GRPO 的 reward。”

## 2 分钟版本

1. **问题**：没有可靠 gold SQL，只有 gold 执行结果；同一语义又存在大量等价 SQL，所以只做 SFT 容易把模型锁死在一条伪标签写法上。
2. **环境**：我把 PostgreSQL execution match 定义成统一裁判，固定时区、超时和只读权限，用列序加行多重集合做比对。
3. **闭环**：先生成候选；失败后按 Schema、投影、过滤/聚合、执行错误分类；规则能确定修的先修，结构性问题让强模型重写；每条都重新真库执行，未通过就隔离。
4. **结果**：同一批 5,635 turn 从 2,061 条正确提升到 5,484 条正确。最大工程洞察是时区口径错误制造了 345 条假失败，说明 RL 前必须先校准 reward 环境。
5. **后训练**：先用 verified-only 数据做 SFT/RFT 冷启动；再对每题采样 K 条，用执行 reward 做 GRPO；顽固模式把正确/错误候选组成 DPO 对。PPO 需要额外 critic，当前资源和可验证奖励场景下不优先。

## Twinkle 框架 60 秒版本

“Twinkle 把数据、模型、loss、采样器和分布式运行时做成解耦组件。同一份训练逻辑既能在 `torchrun` 本地进程组运行，也能通过 Ray 扩展，服务形态还可走 HTTP。SFT 路径是 Dataset 读取 verified-only JSONL，经 Preprocessor 和 Qwen 模板编码，由 DataLoader 按 DeviceMesh 切给各数据并行 rank，TransformersModel 在 FSDP/DP 下前后向，再由 optimizer group 做梯度累积、裁剪和保存。GRPO 则多了一条 rollout 环：vLLM 对同题采样一组 SQL，数据库执行器给 reward，GRPOAdvantage 做组内标准化，GRPOLoss 用 old/new policy ratio 做 clip；策略更新后必须把新权重同步给 sampler，并清掉由旧权重生成的 prefix/KV cache。我的工作是 Text-to-SQL 数据契约、NPU LoRA/SFT 适配和 execution reward 接口，不把 Twinkle 上游框架说成个人原创。”

这段话的对象关系、源码落点、白板图和追问答案见 [Twinkle 框架源码级面试讲义](twinkle-framework-interview.md)。

## 高频追问

### 97.32% 是最终模型准确率吗？

不是。它是候选/训练标签 SQL 的真库执行正确率。最终模型必须另报 held-out execution accuracy；我在简历里保留“候选/训练标签”限定，避免把数据质量成果包装成模型泛化成果。

### 为什么不直接比较 SQL 字符串？

SQL 是程序，同一语义有 JOIN 重写、子查询/CTE、谓词顺序等大量等价形式。字符串或 AST 相似度只能做 shaping，数据库执行结果才是任务目标。不过单快照 execution match 会有伪正确，因此关键样本要做多快照验证。

### 为什么不能用 `set(rows)` 比结果？

`set` 会删除重复行。模型漏写 `DISTINCT`、JOIN 粒度错误时，值集合可能一样但重复次数不同；多重集合 `Counter(rows)` 才能抓住这个错误。

### 为什么 GRPO 适合？

每个问题可以采样一组 SQL，并用数据库得到客观 reward。GRPO 用组内均值/方差构造相对 advantage，不需要 PPO 的单独 value model，显存和系统复杂度更低。代价是同组奖励全相同时 advantage 为 0；要通过难度课程、增大组大小或过滤过易/过难 prompt 保持 reward 方差。

### DPO 和 GRPO 的关系？

DPO 是离线偏好学习：从执行打分 rollout 中选 best/worst，训练模型提高 chosen 相对 rejected 的概率。它稳定、便宜，但受已有候选覆盖限制。GRPO 是 on-policy：持续从当前模型采样和执行，能探索新解，但吞吐、数据库并发和 reward 稳定性要求更高。项目里 GRPO 做主优化，DPO 对顽固错误做定向收尾。

### PPO 比 GRPO 多了什么？

PPO 通常训练 policy 和 value/critic，用 GAE 或 return 估计 advantage，再做 clipped policy/value update；还要维护 reference/KL、rollout、reward 与 value 四类组件。GRPO用同 prompt 多候选的相对 reward 代替 learned critic，因此省掉 value 模型，但其 baseline 质量依赖组内样本。

### Reward 为什么要稠密化？

2B 模型早期可能一组 K 个 SQL 全错，纯 0/1 reward 没有排序信号。我给格式、只读、可执行、列匹配、行数和部分结果重合少量分数，但所有非 exact match 都封顶，保证真正执行正确始终最高。

### 如何防 Reward Hacking？

剔除空结果/常量退化题；多快照复验；限制只读单语句和超时；监控空结果率、返回行数、SQL 长度和重复率；gold 与候选同会话执行；验证集绝不回流训练。

### 反思闭环在线上也能判“结果正确”吗？

通常不能，因为真实用户问题没有 gold 输出。线上可以利用执行错误、Schema 校验、业务约束和多候选一致性；gold execution match 是训练/评测环境的能力。把两者分清是这套系统可信的前提。

### 如果数据库执行很慢，GRPO 怎么做？

数据库侧用连接池、只读副本、statement timeout 和并发上限；训练侧把 rollout 与 reward 异步解耦，按 SQL 规范化 hash 缓存重复执行结果，先做静态拒绝，再批量提交可执行候选。还要把 DB reward latency 纳入训练吞吐监控，不能只看 GPU utilization。

### vLLM 与 KV Cache 怎么讲？

先讲公式和瓶颈，再讲开关：KV 每 token 的近似字节数是 `2 × 层数 × KV头数 × head_dim × dtype_bytes`；上下文越长、并发越大，KV 容量越先吃满。vLLM 用分页块按需分配，减少为最大长度预留产生的内部碎片，并用 continuous batching 提高 GPU 利用率。共享 system/schema 前缀时打开 prefix caching 可省 prefill；FP8 KV 大约降低一半缓存字节，但必须用真实 workload 复测精度、TTFT、TPOT 和吞吐。详见专项讲义。
