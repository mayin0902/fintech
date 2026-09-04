# Text-to-SQL Agent：后训练、反思纠错闭环与 vLLM / KV Cache 工程

> 用途：简历项目说明、技术面试复习、后续实现路线。
>
> 证据边界：本文把仓库中已经发生的工作标为“已落地”，把 GRPO / DPO / PPO 训练及多轮 Agent RL 标为“建议路线”。算法与系统原理只引用论文原文、官方文档或官方源码。

---

## 0. 先统一成果口径：97.32% 到底是什么

仓库的原始评测记录是 `2061 / 5635 = 36.57%`，最终正确集是 `5484 / 5635 = 97.32%`，聚合证据见 [`../results/summary.json`](../results/summary.json) 与 [`metrics.md`](metrics.md)。这组数字描述的是：**候选 SQL / 训练标签经过强模型生成、规则与模型修复、再由真库执行验收后的执行命中率**，不是“训练后的模型在独立测试集上的准确率”。

简历上建议写成：

> 构建 PostgreSQL 真库执行驱动的候选 SQL 验证与反思纠错闭环，将训练候选的执行命中率从 36.57%（2061/5635）提升至 97.32%（5484/5635）。

如果版面很窄，可写：

> 构建数据库执行反馈闭环，经分层纠错将候选 SQL 执行命中率由 36.6% 提升至 97.32%。

不建议写成“用 GRPO 将模型准确率从 36.6% 提升到 97.32%”：仓库证据不支持这句话。只有实际完成 RL 训练，并在冻结的独立测试集上复现结果后，才能把算法名和“模型准确率”绑定。

### 面试时一句话讲清主线

```text
用户问题
  ↓
Schema 理解 / 相关表列定位
  ↓
SQL 生成
  ↓
PostgreSQL 安全执行
  ↓
错误类型 / 结果不匹配反馈
  ↓
反思与重写；离线沉淀偏好对，在线形成可验证 Reward
```

核心不是“让大模型再生成一次”，而是把数据库从最终评测器前移为 Agent 的环境：每次动作是 SQL，环境返回语法、schema、超时或结果一致性信号，策略据此重写；成功轨迹又能反哺 SFT、DPO 或在线 RL。

---

## 1. 把 Text-to-SQL 写成一个 RL / Agent 问题

对于问题与 schema 上下文 `x`，模型生成 SQL 或包含多轮修复的完整轨迹 `y`。后训练的一般目标是：

$$
\max_\theta\; \mathbb{E}_{x\sim\mathcal D,\,y\sim\pi_\theta(\cdot|x)}[r(x,y)]
- \beta D_{KL}(\pi_\theta(\cdot|x)\|\pi_{ref}(\cdot|x)).
$$

第一项让模型追求数据库执行奖励；第二项约束策略不要一下偏离 SFT 参考模型太远。DPO 论文把标准 RLHF 目标明确写成这一形式，并解释了 KL 项用于限制分布漂移与模式坍缩，见 [DPO 论文 §3，式 (3)](https://arxiv.org/html/2305.18290#S3)。

这里有三种不同训练问题，不能混为一谈：

| 问题 | 数据从哪里来 | 是否训练时采样 | 典型算法 |
|---|---|---:|---|
| 模仿已验证的好 SQL | `(问题, 正确 SQL)` | 否 | SFT / RFT |
| 从一对 SQL 中学习偏好 | `(问题, chosen, rejected)` | 否 | DPO |
| 直接最大化执行奖励 | prompt + 在线 rollout + DB reward | 是 | PPO / GRPO |

SFT 学的是“像标签”；DPO 学的是“chosen 相对 rejected 更好”；PPO / GRPO 才是在当前策略分布上探索并直接优化环境奖励。

---

## 2. PPO：完整但最重的 Actor-Critic 路线

### 2.1 原理与目标函数

设 rollout 时的旧策略为 $\pi_{old}$，当前策略为 $\pi_\theta$，token 级概率比为：

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}.
$$

PPO 的 clipped surrogate objective 是：

$$
L^{clip}(\theta)=\mathbb E_t\left[
\min\left(\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)
\right].
$$

当更新会把概率比推得过远时，clip 截断收益，从而允许同一批 on-policy 样本做多轮 minibatch 更新，又避免一步走得太远。这是 [PPO 原论文](https://arxiv.org/abs/1707.06347) 的核心；LLM 场景的 token 级写法可直接对照 [DeepSeekMath §4.1.1，式 (1)](https://arxiv.org/html/2402.03300#S4.SS1.SSS1)。

PPO 的 $A_t$ 通常由 critic/value model $V_\psi(s_t)$ 和 GAE 估计。LLM RLHF 还常把相对参考模型的 KL 惩罚加入 token reward：

$$
r_t=r_{task,t}-\beta\log\frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}.
$$

这一结构及“critic 与 policy 同时训练”的原因见 [DeepSeekMath §4.1.1，式 (2)](https://arxiv.org/html/2402.03300#S4.SS1.SSS1)。InstructGPT 的经典流水线则是“示范数据 SFT → 人类比较训练 Reward Model → PPO 优化 SFT 模型”，见 [InstructGPT 论文 §2](https://arxiv.org/html/2203.02155#S2)。

### 2.2 工程上到底要维护什么

逻辑上至少有四个角色：

1. actor/current policy：被更新的 SQL 生成模型；
2. old policy：rollout 行为策略，提供 PPO 概率比分母；
3. reference policy：固定或低频更新，提供 KL 锚点；
4. critic/value model：估计每个状态的价值，供 GAE 算 advantage。

如果是传统 RLHF，还另有 reward model；本项目拥有 PostgreSQL 可验证奖励，可把神经 reward model 替换为程序化执行器，但 **critic 仍然存在**。工程可以通过冻结副本、参数共享、offload 或分片减少物理副本，不能简单理解为“显存一定放五份完整模型”。

训练循环：

```python
for prompts in loader:
    trajectories = rollout_engine.generate(prompts, policy=old_policy)
    task_rewards = postgres_executor.score(trajectories)
    old_logps = old_policy.logprobs(trajectories)
    ref_logps = ref_policy.logprobs(trajectories)
    values = critic(trajectories)
    advantages, returns = gae(task_rewards, values)

    for _ in range(ppo_epochs):
        new_logps = actor.logprobs(trajectories)
        optimize(
            clipped_policy_loss(new_logps, old_logps, advantages)
            + value_loss(critic, returns)
            + kl_penalty(new_logps, ref_logps)
        )
    old_policy.load_state_dict(actor.state_dict())
```

### 2.3 什么时候 Text-to-SQL 值得用 PPO

PPO 的优势是 critic 能对长、多步轨迹做更细的 credit assignment。例如：第一次 SQL 报列名错，第二次修正列名但聚合仍错，第三次才成功；value model 可以学习不同中间状态的价值。

它的代价也最明显：critic 的训练、actor/old/ref 的 logprob、rollout 与多轮更新都增加显存和系统复杂度。对当前 2B、单机 NPU、结果可验证的场景，先用 GRPO 通常更合理；只有当多轮 Agent 的中间步骤奖励很重要，且 GRPO 的整条轨迹 outcome reward 不够用时，再上 PPO。

---

## 3. DPO：把离线“好 SQL / 坏 SQL”直接变成分类损失

### 3.1 原理与目标函数

对偏好三元组 $(x,y_w,y_l)$，其中 $y_w$ 是 chosen SQL、$y_l$ 是 rejected SQL，DPO 的损失为：

$$
\mathcal L_{DPO}=-\mathbb E\left[
\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right)\right].
$$

DPO 从 KL 约束的 RLHF 最优策略推导出 reward 与策略 log-ratio 的映射，Bradley-Terry 偏好概率中的配分函数会在成对差值里抵消，因此无需显式 reward model，也无需在线 RL rollout；完整推导见 [DPO 论文 §4，式 (4)–(7)](https://arxiv.org/html/2305.18290#S4)。梯度直觉是提高 chosen 的相对 log probability、降低 rejected 的相对 log probability，并对当前排序最错的样本赋更大权重，见 [DPO 论文式 (8) 附近](https://arxiv.org/html/2305.18290#S4)。

注意：原始 DPO 使用完整序列概率，即 token logprob 的和；有些框架提供长度归一化或 reference-free 变体，它们不再等同于原论文目标。面试中应先说清“标准 DPO”，再讨论实现变体。

### 3.2 本项目如何构造可靠偏好对

数据库执行器天然能制造偏好：

```text
x        = 用户问题 + schema + 多轮历史
chosen   = 同一快照上执行结果正确且安全的 SQL
rejected = 当前模型生成的语法错 / schema 错 / 结果错 SQL
```

构造时必须遵守：

- 两条 SQL 都保留完整 prompt 上下文，尤其是多轮历史与 schema 版本；
- 若两条 SQL 执行结果等价，不应硬标 chosen/rejected；等价 SQL 正是 execution reward 相对 token-level SFT 的优势；
- chosen 与 rejected 不要长期存在明显长度差，否则模型可能学到长度捷径。DeepSeek-R1 在构造 helpfulness 偏好数据时专门控制两侧长度可比，见 [DeepSeek-R1 §3.1](https://arxiv.org/html/2501.12948#S3.SS1)；
- 同一问题的多个 rejected 可以按错误类型采样，但 train/val 必须按 episode 或模板族隔离，防止近重复泄漏；
- 时区、NULL、数值精度与行序比较口径先固定，否则偏好标签本身相互矛盾。

简化训练循环：

```python
for x, chosen, rejected in preference_loader:
    pi_w, pi_l = policy.sequence_logps(x, chosen, rejected)
    with no_grad():
        ref_w, ref_l = reference.sequence_logps(x, chosen, rejected)
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    loss = -logsigmoid(logits).mean()
    loss.backward()
    optimizer.step()
```

### 3.3 DPO 在本项目中的正确定位

DPO 适合处理“模型反复偏好的固定坏模式”，如漏 `DISTINCT`、把 `HAVING` 写成 `WHERE`、错误继承上一轮条件。它能复用已经落盘的执行日志，训练稳定、资源较省。

但 DPO **不是在线数据库反馈闭环**：训练时不会生成新 SQL、不会查询数据库，也不会自动发现当前模型的新错误。它的上限取决于离线 pair 的覆盖度。因此建议把 DPO 放在 RFT / GRPO 后，针对残留错误做定向收尾，而不是把它当作本项目唯一的 RL 方法。

---

## 4. GRPO：最贴合可验证 SQL 奖励的主路线

### 4.1 从 PPO 到 GRPO

GRPO 对同一个问题采样 $G$ 个输出，用组内 reward 的均值和标准差构造相对优势：

$$
\hat A_i=\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)}.
$$

然后像 PPO 一样使用 old-policy probability ratio 和 clip，并对 reference policy 加 KL 正则：

$$
J_{GRPO}=\mathbb E\left[
\frac1G\sum_i\frac1{|o_i|}\sum_t
\min(\rho_{i,t}\hat A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i)
-\beta D_{KL}(\pi_\theta\|\pi_{ref})
\right].
$$

原始 GRPO 用同 prompt 的组分数代替 learned critic baseline，从而省掉 value model；目标函数、KL 估计和 outcome supervision 的 token advantage 定义见 [DeepSeekMath §4.1](https://arxiv.org/html/2402.03300#S4.SS1)。DeepSeek-R1 继续采用 GRPO，并明确使用可程序验证的 accuracy reward 与 format reward，见 [DeepSeek-R1 §2.1–2.2](https://arxiv.org/html/2501.12948#S2.SS1)。

### 4.2 为什么它适合 Text-to-SQL

- 数据库执行结果是客观、可重复的 outcome verifier，类似代码任务中的编译器和测试用例；DeepSeek-R1 也把编译器 + 测试用例作为 rule-based accuracy reward 的例子，[论文 §2.2](https://arxiv.org/html/2501.12948#S2.SS2)。
- 一个问题可存在多条等价 SQL，GRPO 只要求结果 reward 高，不强迫模型模仿某一条 canonical SQL。
- 不训练 critic，显存与实现复杂度低于 PPO；代价是每个 prompt 要生成 $G$ 条 SQL，rollout 计算量较大。
- 同组相对比较特别适合“同题有对有错”的区域。

最后一点也揭示了 GRPO 最重要的工程坑：若某一组全对或全错，reward 方差为 0，原公式的相对优势没有有效学习信号。工程实现会加 epsilon、跳过或得到近零 advantage，但都不能凭空制造信息。因此必须监控 `zero_std_group_fraction`，并通过课程采样、增加 $G$、提高采样温度或引入严格受控的 shaped reward，让训练题落在“当前模型有时会、有时不会”的难度带。这个结论直接来自上面的组归一化公式。

### 4.3 Text-to-SQL 的 GRPO 训练循环

```python
for prompts in loader:
    # 每题 G 个候选；rollout 必须来自当前或足够新的 old policy
    groups = vllm_rollout.generate(prompts, n=G, temperature=1.0)

    rewards = []
    for prompt, candidates in zip(prompts, groups):
        rewards.append([safe_postgres_executor.score(prompt, sql) for sql in candidates])

    advantages = group_normalize(rewards)
    old_logps = rollout_logprobs(groups)
    ref_logps = reference_logprobs(groups)

    for minibatch in pack_by_token_budget(groups):
        new_logps = actor.logprobs(minibatch)
        loss = grpo_clipped_loss(
            new_logps,
            old_logps,
            ref_logps,
            advantages,
            clip_eps=eps,
            kl_beta=beta,
        )
        loss.backward()
        clip_grad_norm_()
        optimizer.step()

    sync_actor_weights_to_vllm()
    invalidate_old_kv_cache()
```

GRPO 实现需要的最小字段是：prompt/response token、response mask、reward/advantage、old logprob、reference logprob。仓库中的 [`../src/text2sql_feedback/post_training.py`](../src/text2sql_feedback/post_training.py) 展开了组内 advantage 与 clipped surrogate 的核心计算，[`../scripts/train_grpo.py`](../scripts/train_grpo.py) 展示执行 reward 的框架接入；它们是可运行参考实现，不能表述成项目已经完成 GRPO 训练并取得 97.32%。

### 4.4 Reward 设计：主奖励必须压住塑形分

建议把 reward 设计成“安全门控 + outcome 主奖励 + 低上限塑形”，避免一个语法漂亮但结果错误的 SQL 胜过真正正确的 SQL：

```text
unsafe / 非单条只读查询        → 直接最低分
超时 / 执行异常                → 负分，并保留错误类别
执行成功但结果不匹配           → 低分区间，只用结构信号排序
结果完全匹配                   → 满分
首次失败、根据反馈修复成功      → 满分 + 有上限的 recovery bonus
```

一个可执行的 reward 分解是：

$$
r = r_{exec\_match}
+ \lambda_1 r_{executable}
+ \lambda_2 r_{schema}
+ \lambda_3 r_{format}
+ \lambda_4 r_{recovery}
- \lambda_5 r_{timeout}
- \lambda_6 r_{unsafe}.
$$

约束应是：任何 `exec_match=1` 的总分都高于所有 `exec_match=0`；长度/查询代价只做 tie-breaker，不能诱导模型输出空 SQL、常量查询或故意提前结束。DeepSeek-R1 使用 accuracy + format 两类 rule reward，并说明神经 reward model 在大规模 RL 中可能被 reward hacking，[论文 §2.2](https://arxiv.org/html/2501.12948#S2.SS2)；Text-to-SQL 的真库 verifier 同样应让 correctness 保持最高优先级。

---

## 5. 三种算法怎么选

| 维度 | DPO | PPO | GRPO |
|---|---|---|---|
| 数据 | 固定 preference pairs | 在线 rollout + scalar reward | 同 prompt 的 $G$ 个在线 rollout + reward |
| 是否探索新 SQL | 否 | 是 | 是 |
| critic/value model | 无 | 有 | 无 |
| reference model | 标准 DPO 需要 | 常用 | 原始 GRPO 需要 |
| 主要计算成本 | chosen/rejected 的 policy/ref forward | rollout + actor/ref/critic + value update | $G$ 倍 rollout + actor/ref logprob |
| 最擅长 | 离线纠正固定偏好 | 长轨迹、细粒度 credit assignment | 可验证 outcome、资源受限的 reasoning/code/SQL |
| 本项目定位 | 残留模式收尾 | 多步 credit 不够时再用 | 主力在线 RL 路线 |

推荐顺序：

```text
verified-only SFT / RFT
    ↓
GRPO + PostgreSQL execution reward
    ↓
DPO 针对残留、稳定复现的错误模式
    ↓
只有多步状态价值确实重要时，再考虑 PPO
```

这不是说 GRPO 理论上总优于 PPO，而是当前约束下的工程选型：任务 reward 可验证、模型较小、算力有限，省掉 critic 的收益很实际。

---

## 6. “反思纠错闭环”应该如何真正落地

### 6.1 状态、动作、环境反馈

一次 episode 不应只是“采样 SQL → 打 0/1 分”，而应显式包含修复过程：

```text
state_0:
  question + schema + conversation history

action_0:
  SQL_0

observation_0:
  {status: syntax_error, sqlstate: 42703,
   message: "column ... does not exist"}
  或 {status: result_mismatch, mismatch_type: row_count/column/value}

action_1:
  reflection_1 + SQL_1

observation_1:
  execution result / error

terminal:
  success，或达到 max_attempts
```

数据库反馈只给修复所需的最小信息：错误类别、脱敏后的数据库消息、结果 mismatch 类型；不要把 gold rows 直接塞给模型，否则训练成“抄答案”而非纠错。

对多轮 Agent 训练，tool observation 属于环境 token，不应计算语言模型 loss；只对 assistant 的 reflection / SQL action token 建 response mask。若用 GRPO，可把完整多步 episode 视为一个 sampled output，用终局 reward 广播到该轨迹的 action token；若需要区分每次修复的价值，PPO + step reward/critic 更自然。

### 6.2 安全执行器

执行模型生成 SQL 时，应用数据库权限而不是只靠正则：

1. 使用只有 `SELECT` 权限的独立数据库角色；
2. 每条候选在独立事务中执行 `SET TRANSACTION READ ONLY`；PostgreSQL 的只读事务会禁止对非临时表的 `INSERT/UPDATE/DELETE/MERGE` 及 DDL，见 [PostgreSQL 17 `SET TRANSACTION`](https://www.postgresql.org/docs/17/sql-set-transaction.html)；
3. 在会话/事务局部设置 `statement_timeout`、`lock_timeout`；`statement_timeout` 会中止超过时限的语句，官方不建议在全局 `postgresql.conf` 中为所有会话统一设置，见 [PostgreSQL 16 Client Connection Defaults](https://www.postgresql.org/docs/16/runtime-config-client.html)；
4. SQL parser 只允许一条查询，并审计 `WITH` 中的数据修改语句——PostgreSQL 的 CTE 本身可以包含 `INSERT/UPDATE/DELETE/MERGE`，不能因为以 `WITH` 开头就认定安全，见 [PostgreSQL `WITH` 官方文档](https://www.postgresql.org/docs/17/queries-with.html)；
5. 限制结果行数、连接池并发、临时空间与执行超时，finally 中 rollback；
6. 记录 schema version、DB snapshot、timezone、collation 与 evaluator version，使 reward 可复现。

### 6.3 Execution-match 的比较契约

先定义任务语义，再写 comparator：

- 比较列数、列名还是只比较值；
- 无 `ORDER BY` 时按 multiset/bag 比较，有明确顺序语义时再按序列比较；
- `Decimal` 禁止直接转二进制 float；
- timestamp 在同一 timezone 下归一；
- 明确 `NULL`、NaN、大小写与尾随空格策略；
- gold 与 candidate 必须在同一快照、同一会话参数下运行；
- 区分 parse error、schema error、runtime error、timeout、结果 mismatch，既便于 reward shaping，也便于错误分析。

单快照 execution-match 只能证明“在当前数据上结果相同”，不能证明 SQL 语义等价。对重要样本可在保持约束的数据扰动快照上复验，减少 `WHERE false`、空集巧合或常量查询造成的伪正确。

### 6.4 专门衡量“反思纠错”的指标

只报最终 execution accuracy 会掩盖 Agent 是否真正会修：

| 指标 | 定义 | 解释 |
|---|---|---|
| first-attempt EA | 第一次 SQL 命中率 | 基础生成能力 |
| final-attempt EA | 尝试预算内最终命中率 | Agent 总体效果 |
| recovery rate | `P(final correct | first wrong)` | 反思纠错核心指标 |
| error-to-success | 各 error type 的修复成功率 | 能否利用具体反馈 |
| mean attempts on success | 成功样本平均尝试数 | 效率与是否乱试 |
| unsafe / timeout rate | 不安全或超时占比 | 环境与 reward 健康度 |

算法侧再监控：reward mean/std、GRPO 的零方差组比例、KL、entropy、clip fraction、response length、old/new/rollout logprob 差异。若 EA 不升但 KL 和长度持续上升，通常是在过优化代理 reward。

---

## 7. vLLM 为什么适合 rollout 与在线服务

### 7.1 Prefill 与 Decode 是两种不同负载

- **Prefill** 一次处理 prompt 的许多 token，计算所有层的 K/V 并写入 cache，矩阵较大，通常更偏 compute-bound；
- **Decode** 每条序列每步只生成一个新 token，却要读取此前所有可见 token 的 K/V，通常更偏 memory-bandwidth-bound；
- 连续批处理的核心是每个 engine step 动态接纳/退出请求，不让一个静态 batch 被最长序列拖住。

vLLM V1 的 chunked prefill 会先调度 decode，再用剩余 `max_num_batched_tokens` 预算装入 prefill；放不下的 prefill 被切块。官方说明这样能改善 ITL，并把偏计算的 prefill 与偏访存的 decode 放在同一批次以提高利用率，见 [vLLM 0.17 Optimization and Tuning](https://docs.vllm.ai/en/v0.17.0/configuration/optimization/#chunked-prefill)。

### 7.2 PagedAttention 解决的不是 Attention 数学，而是 KV 内存管理

传统做法为每个请求按最大长度预留连续 KV 空间，会产生：

- 未来 token 的预留浪费；
- 请求实际长度较短导致的内部碎片；
- 不同大小连续区间导致的外部碎片；
- beam / parallel sampling 难以共享 prompt KV。

PagedAttention 把每条序列的逻辑 KV cache 切成固定 token 数的 block，通过 block table 映射到不连续的物理 block；物理块按需增长，因此每条请求的浪费被限制在最后一个未填满 block。并行采样共享 prompt block，分叉写入时使用 copy-on-write。原论文给出逻辑/物理块、block table、按需分配与 COW 的完整过程，见 [PagedAttention 论文 §4](https://arxiv.org/html/2309.06180#S4)。

原论文中，传统系统只有 20.4%–38.2% 的 KV 空间真正存放 token state；vLLM 通过分页管理实现接近零浪费，并在论文工作负载上达到相对当时系统 2–4× 的吞吐。数字只代表论文实验，不能直接当成当前项目实测结果，见 [PagedAttention 论文 Introduction](https://arxiv.org/html/2309.06180#S1)。

### 7.3 KV Cache 的完整生命周期

```text
引擎启动
  → profile 可用显存
  → 预分配物理 block pool

新请求
  → 对 prompt token block 做 prefix hash 查询
  → 命中块 ref_cnt++ 并固定，缺失部分申请空闲块

prefill
  → 计算 K/V，写入 block；完整块可进入 prefix cache

decode
  → 每步追加一个 token；尾块满后再申请新块

请求结束
  → ref_cnt--；无人引用的块进入 free queue
  → 可复用的完整 prefix block 仍带 hash，等待命中或淘汰

内存压力
  → 从 free queue 头部淘汰 LRU cached block
  → 若运行中请求竞争仍过强，scheduler preempt 并稍后重算
```

vLLM V1 的官方设计文档说明：block pool 在初始化时一次创建，`KVCacheBlock` 带 `block_id`、`block_hash`、`ref_cnt` 和双向 free-queue 指针；分配会先 touch 命中块，再从 free queue 取块；请求结束后降引用并回队列，压力下淘汰 LRU block，见 [vLLM Prefix Caching 设计文档](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md#data-structure)。

当 KV cache 不足时，vLLM V1 默认通过 `RECOMPUTE` 抢占；频繁 preemption 会伤害端到端延迟。官方建议结合增加 `gpu_memory_utilization`、降低 `max_num_seqs/max_num_batched_tokens` 或调整并行度处理，并监控 preemption，见 [vLLM 0.17 Preemption](https://docs.vllm.ai/en/v0.17.0/configuration/optimization/#preemption)。

---

## 8. KV Cache 怎么估算内存

### 8.1 标准全注意力层

对每个 token、每个设备：

$$
B_{KV/token}=2\times L_{local}\times H_{KV,local}\times D_{head}\times s.
$$

- `2`：Key 和 Value；
- $L_{local}$：该设备负责的全注意力层数；
- $H_{KV,local}$：该设备实际保存的 KV heads；MHA 等于 attention heads，GQA/MQA 更少；
- $D_{head}$：每个 head 的维度；
- $s$：每元素字节数，FP16/BF16 通常 2，FP8 通常 1。

一批请求的粗略物理容量是：

$$
M_{KV}\approx B_{KV/token}\sum_i
\left\lceil\frac{T_i}{B}\right\rceil B,
$$

其中 $B$ 是 block token 数，$T_i$ 是当前已缓存 token 数。共享 prefix 应按物理 block 只计一次；滑动窗口、跨层 KV sharing、混合 attention/SSM 模型要按各自 cache spec 另算。

PagedAttention 论文给出一个可核算例子：OPT-13B 为 `2 × 5120 hidden × 40 layers × 2 bytes = 800 KiB/token`，2048 token 约 1.6 GiB/request，见 [论文 §3](https://arxiv.org/html/2309.06180#S3)。这也说明为什么 GQA/MQA 减少 KV heads 会显著降低 cache，而权重量化与 KV 量化是两件不同的事。

### 8.2 本项目的 Qwen3.5 混合结构不能套一个常数

本项目使用的 Qwen3.5-2B 是混合注意力结构；具体层型与层数应以固定模型版本的 [`config.json`](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json) 为准。因此不能把全部层机械代入标准 KV 公式；全注意力层保存随序列增长的 K/V，线性注意力层保存的是架构特定 recurrent/conv state。

vLLM 的 Hybrid KV Cache Manager 会按 attention type 分组，并要求各组使用统一 page size；某些混合结构需要 group/padding，实际物理容量以启动日志中的 cache spec 与 block 数为准，见 [vLLM Hybrid KV Cache Manager 官方设计](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md#allocation)。面试里说“按层类型求和，再以运行时分配日志校准”比机械套 Transformer 公式更准确。

### 8.3 FP8 KV 量化

BF16/FP16 → FP8 从表示宽度看可把 KV payload 约减半，从而容纳更多 token；实际吞吐是否增加取决于硬件和 attention kernel 是否原生支持，不能只凭容量推断。vLLM 0.17 支持 per-tensor 与 per-attention-head scale，并列出默认 scale、运行时计算和数据集校准三条路径；官方推荐用校准数据获得更好的 scale，见 [vLLM 0.17 Quantized KV Cache](https://docs.vllm.ai/en/v0.17.0/features/quantization/quantized_kvcache/)。

项目使用 vLLM-Ascend，不能把 CUDA 文档中的 FP8 支持直接当作 910B3 已支持。采用前必须同时核对固定版本 vLLM-Ascend 的量化矩阵与 kernel，并对 execution accuracy、TTFT、ITL、吞吐做 A/B。

---

## 9. Prefix Cache：什么时候真有用，什么时候没用

vLLM 的 Automatic Prefix Caching（APC）对每个完整 KV block 做链式 hash，key 包括 parent hash、当前 block tokens，以及 LoRA ID、多模态输入 hash、cache salt 等额外身份。只缓存完整 block；相同 token 前缀才能复用，见 [vLLM Prefix Caching 设计](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md#automatic-prefix-caching)。

因此 Text-to-SQL 很适合做 prompt 布局优化：

```text
[稳定 system prompt]
[稳定数据库说明 / schema]
[少量固定示例]
[变化的多轮历史与用户问题]
```

把稳定内容放前面，避免在时间戳、随机 request id、字段排序上制造无意义差异；同一 schema 的请求更可能命中 prefix cache。多轮会话也可复用既有历史。vLLM 官方列出的典型收益场景正是长文档多次查询与多轮对话，[APC 官方说明](https://docs.vllm.ai/en/v0.17.0/features/automatic_prefix_caching/)。

边界同样重要：APC 只跳过共享前缀的 prefill，不能加速新 token decode；若请求没有共同前缀或答案生成占主要时间，就几乎没有收益，[vLLM APC Limits](https://docs.vllm.ai/en/v0.17.0/features/automatic_prefix_caching/#limits)。所以随机 prompt benchmark 关闭 APC 可能更合理，而真实 Text-to-SQL 的固定 schema prompt 反而可能受益，必须按实际流量分别测试。

在多租户场景，cache hit 的时延差可能泄露另一个租户是否访问过相同前缀；vLLM 支持把 `cache_salt` 加入首块 hash，将复用限制在同一信任域，见 [官方 prefix cache isolation 设计](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md#cache-isolation-for-security)。

---

## 10. vLLM 的核心调优旋钮，应该怎么解释

| 参数 / 机制 | 真正控制什么 | 常见误区 |
|---|---|---|
| `gpu_memory_utilization` | 当前实例用于 model executor 的显存比例，间接决定 KV block pool | 不是“越接近 1 越快”；还要给运行时峰值留余量 |
| `max_num_seqs` | 单次 scheduler iteration 最大序列数 | 不是实际并发；过大可能制造 KV 压力 |
| `max_num_batched_tokens` | 每个 iteration 的 token 预算 | 小值偏 ITL，大值可提高 prefill 批量/TTFT与吞吐，需看 SLO |
| `block_size` | 每个 KV 物理块容纳的 token 数 | 大 block 可提高访问并行度，但加大尾块碎片；不要脱离后端支持盲调 |
| `max_model_len` | 单请求允许的最大上下文上限 | PagedAttention 按需分配，不会因为把上限设大就给每个请求预留整段 KV；它主要是安全与 admission 上限 |
| chunked prefill | 把长 prompt 切入 token budget，与 decode 混排 | 不是永远提高所有延迟；要在 TTFT/ITL/吞吐间取舍 |
| TP | 在卡间切分每层权重/计算 | 小模型能单卡放下时，通信可能抵消收益 |
| DP replicas | 多模型副本接不同请求 | 吞吐好但每副本都占权重显存，需要前端负载均衡 |

vLLM 0.17 的 cache config 把 `gpu_memory_utilization` 定义为 per-instance 限额，默认 0.9；`block_size` 是每个连续 cache block 的 token 数，见 [vLLM 0.17 CacheConfig](https://docs.vllm.ai/en/v0.17.0/api/vllm/config/cache/)。官方调优文档也明确警告 TP 增大会产生同步开销，并给出 `max_num_batched_tokens` 对 ITL / TTFT 的方向性影响，[vLLM 0.17 Optimization](https://docs.vllm.ai/en/v0.17.0/configuration/optimization/)。

### Ascend 图模式：降低调度开销，但要给捕获和兼容性留空间

在项目对应的 vLLM-Ascend `v0.17.0rc1` 中，图模式只用于 V1 Engine，ACLGraph 是默认路径，且官方仍将它标为 experimental；不兼容或运行失败时可用 `enforce_eager=True` 回退，[vLLM-Ascend v0.17.0rc1 Graph Mode Guide](https://github.com/vllm-project/vllm-ascend/blob/v0.17.0rc1/docs/source/user_guide/feature_guide/graph_mode.md)。工程上应分别记录首次编译/捕获成本和预热后的稳态延迟，并观察图捕获额外占用是否挤压 KV block pool；不能只看稳态 token/s，也不能假定所有动态 batch shape 都命中同一张图。最终生效模式以固定镜像的启动日志为准，不把新版本 `main` 分支的默认值套到 0.17。

### 结构化输出：先消灭格式错误，语义仍由数据库裁决

vLLM 0.17 的 structured outputs 可用 `choice`、`regex`、JSON Schema 与 context-free `grammar` 限制生成，官方还给出了简化 SQL 的 EBNF grammar 示例，[vLLM 0.17 Structured Outputs](https://docs.vllm.ai/en/v0.17.0/features/structured_outputs/)。它适合保证只输出 SQL、固定 JSON envelope 或约束语法形状，但 grammar 不知道当前数据库是否真的存在某张表/某列，也无法证明执行结果语义正确。因此它是生成侧 guardrail，不能替代 schema grounding、只读数据库执行和结果级 reward。

### 正确的 A/B 方法

1. 固定模型、tokenizer、输入/输出长度分布、采样参数和并发模型；
2. 预热后测量，不把编译/graph capture 算入稳态；
3. 同时报 TTFT、TPOT/ITL、E2E p50/p95/p99、request/s、input/output token/s；
4. 观察 `kv_cache_usage`、running/waiting requests、prefix hit、preemptions；vLLM 暴露这些 metrics，见 [vLLM 0.17 Production Metrics](https://docs.vllm.ai/en/v0.17.0/usage/metrics/)；
5. 从低 request rate 向上扫，找到满足 SLO 的最大稳定吞吐，而不是只跑一个并发点；
6. 精度敏感优化（权重/KV 量化、结构化解码）必须另外跑 execution accuracy 回归。

---

## 11. vLLM 如何接入 GRPO / PPO 训练

vLLM 负责高速 rollout，不负责反向传播。训练系统需要解决三件比“启动一个 server”更难的事。

### 11.1 资源布局

- **分离式**：训练 actor 与 rollout vLLM 各占 GPU/NPU，吞吐稳定但需要权重传输；vLLM 官方 RLHF 示例正是训练模型在一张 GPU、TP inference engine 在另外两张 GPU，并通过 collective RPC 更新权重，见 [vLLM 0.17 RLHF example](https://docs.vllm.ai/en/v0.17.0/examples/offline_inference/rlhf/)；
- **共置式**：训练与 rollout 轮流占同一设备，省卡但要处理权重、optimizer、KV cache 与 graph 的峰值竞争；
- **异步式**：rollout 和训练流水并行，吞吐高，但样本可能来自陈旧策略，逐渐变成 off-policy，必须记录 policy version 与 staleness。

当前两张 910B3 的保守起点应是同步或低 staleness：先验证 reward、logprob 与权重同步正确，再追求 fully async。

### 11.2 权重同步与 KV 失效

每次 actor update 后，rollout engine 必须加载新权重；否则采到的是旧策略。更隐蔽的是：旧权重计算出的 prefix KV cache 不能被新权重复用，因为 K/V 本身是模型参数的函数。verl 官方 vLLM adapter 在更新权重后显式 `clear_kv_cache`，见 [verl `vllm_rollout.py`](https://github.com/verl-project/verl/blob/main/verl/workers/rollout/vllm_rollout/vllm_rollout.py)。

同卡共置还要控制同步峰值。vLLM 的 sleep mode 可丢弃 KV、分阶段唤醒 weights 与 KV，以避免 weight update 时 OOM，但 0.17 官方页面注明其平台支持为 CUDA/ROCm；Ascend 是否可用必须查 vLLM-Ascend 实现，不能直接照搬，见 [vLLM 0.17 Sleep Mode](https://docs.vllm.ai/en/v0.17.0/features/sleep_mode/)。

### 11.3 Rollout logprob 与训练 logprob 一致性

vLLM 和训练框架可能使用不同 attention kernel、精度、padding/packing 或 logits processor。即使权重相同，recomputed logprob 也可能有细小差异；一旦 rollout engine 权重陈旧，差异会更大。应保存：

- rollout policy version；
- rollout token IDs 和 response mask，而不是重新 tokenize 文本；
- rollout old_logprob；
- trainer recomputed old/new logprob；
- per-token / per-sequence importance ratio、KL、被 clip 比例。

如果 response token 对不齐，先停止训练；这是数据契约错误，不是调 `epsilon` 能修的。若只是可控数值差异，再决定是否做 importance-sampling correction。不要在 correctness 尚未验证时直接 fully async。

---

## 12. 结合本仓库的可执行技术路线

### 阶段 A：冻结评测与执行器（先于 RL）

- 固化 `DB snapshot + timezone + comparator version`；
- 单元测试覆盖 NULL、Decimal、timestamp、无序 multiset、重复行、空结果、超时；
- 独立只读角色 + transaction timeout；
- 输出结构化 error taxonomy；
- 按 episode/schema/template family 切分训练、验证与最终测试。

验收：同一 SQL 多次 reward 一致；所有危险语句不可执行；现有 5635 条可复现 2061 与 5484 两个证据点。

### 阶段 B：SFT / RFT 冷启动

- 用 5484 条 verified-only SQL 训练；
- 当前 SFT 模型每题采样 $K$ 条，执行命中者进入 RFT；
- 去除重复等价 SQL 或控制同题权重；
- 只在冻结 validation 上选 checkpoint。

目标：格式稳定、可执行率高，并让后续 GRPO 组中出现一定比例正样本。

### 阶段 C：GRPO 主训练

- 每题先取 $G=4$ 或 8 做吞吐/信号验证，逐步扩大；
- rollout 使用温度采样，保证探索；
- execution match 是最高 reward，schema/format 只在错误候选之间塑形；
- 动态采样“组内有对有错”的题，监控零方差组；
- 小学习率、gradient clipping、KL 与 clip fraction 联合监控；
- 每次 actor→vLLM 权重同步后清 KV cache；
- 定期在冻结集跑 greedy first-attempt EA 与 Agent final-attempt EA。

### 阶段 D：把错误反馈加入 Agent rollout

- 先从最多 2 次尝试开始，避免 token 与 DB 成本失控；
- 反馈只给 error type + sanitized message / mismatch type；
- 记录 first wrong → final correct 的 recovery rate；
- 对“失败后未改变 SQL”“重复相同错误”“无意义拉长反思”加有上限惩罚；
- 对成功轨迹做 RFT；对 success/failure 同题配对做 DPO。

### 阶段 E：DPO 收尾，PPO 作为升级项

- 按残留错误族生成 chosen/rejected，控制长度与来源；
- DPO 后同时检查 chosen reward、rejected reward、validation EA 与 KL proxy；
- 若多步修复需要细粒度状态价值，而 outcome-only GRPO 明显不够，再引入 critic 做 PPO；不要为了简历关键词堆算法。

---

## 13. 高频面试追问与回答骨架

### 为什么 execution accuracy 可以当 reward？

因为 SQL 的最终语义是查询结果，数据库执行器是可程序验证的环境；它能同时容纳不同但等价的 SQL。局限是单一快照可能出现伪等价，所以要固定比较契约，并对关键样本做多快照/扰动验证。

### 为什么 GRPO 比 PPO 省显存？

PPO 用 learned value model 做 baseline；GRPO 用同 prompt 多个 completion 的组均值/方差做 baseline，因此省掉 critic。它不是“不要 reference/old policy”，原始目标仍有 old-policy ratio 与 reference KL。

### GRPO 全 0 reward 会发生什么？

同组标准差为 0，相对 advantage 没有有效信号。解决方向是调整题目难度、增加组大小和探索，或设计不会颠倒 correctness 排序的稠密 reward，而不是单纯多训几步。

### DPO 为什么不是真正的在线 RL？

它在固定的 `(chosen,rejected)` 数据上做二分类式策略优化，不在训练循环中与数据库交互，也不探索新错误。它适合消费执行日志，但不能替代 rollout。

### PagedAttention 与 FlashAttention 有什么区别？

FlashAttention 主要优化 attention 计算的 IO/tiling；PagedAttention 主要解决 serving 时动态 KV cache 的地址映射、分页分配与共享。二者可组合：一个管算子，一个管 KV 的物理组织与生命周期。

### Prefix cache 与普通 KV cache 有什么区别？

每个自回归请求都需要自己的 KV cache；prefix cache 是在请求之间复用“相同 token 前缀已经算出的 KV block”。它只省共享前缀的 prefill，不省新 token 的 decode。

### 调大 `max_model_len` 为什么不一定立刻占满显存？

PagedAttention 按已产生 token 动态分配物理 block，不为每个请求预留完整最大长度。`max_model_len` 是上限与 admission 约束；实际 KV 压力取决于同时存活 token 总数、block 尾部碎片和 cache sharing。

### 你们的 97.32% 是模型准确率吗？

不是。它是 5635 条训练候选经过真库执行验证后的正确集覆盖率，最终 5484 条命中；模型能力要单独报冻结验证/测试集的 execution accuracy。主动把口径说清，可信度反而更高。

---

## 14. 一手资料索引

- PPO 原论文：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- LLM PPO / RLHF 经典流程：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- DPO 原论文：[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- GRPO 原始系统化描述：[DeepSeekMath](https://arxiv.org/abs/2402.03300)
- Rule reward 与 GRPO 实践：[DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- PagedAttention / vLLM 原论文：[Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- vLLM 0.17 官方调优：[Optimization and Tuning](https://docs.vllm.ai/en/v0.17.0/configuration/optimization/)
- vLLM V1 prefix cache 官方设计：[Automatic Prefix Caching](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)
- vLLM hybrid cache 官方设计：[Hybrid KV Cache Manager](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md)
- vLLM 0.17 KV 量化：[Quantized KV Cache](https://docs.vllm.ai/en/v0.17.0/features/quantization/quantized_kvcache/)
- vLLM-Ascend 0.17 图模式：[Graph Mode Guide](https://github.com/vllm-project/vllm-ascend/blob/v0.17.0rc1/docs/source/user_guide/feature_guide/graph_mode.md)
- vLLM 0.17 结构化解码：[Structured Outputs](https://docs.vllm.ai/en/v0.17.0/features/structured_outputs/)
- vLLM 0.17 RLHF 权重同步示例：[RLHF example](https://docs.vllm.ai/en/v0.17.0/examples/offline_inference/rlhf/)
- PostgreSQL 只读事务：[SET TRANSACTION](https://www.postgresql.org/docs/17/sql-set-transaction.html)
- PostgreSQL 超时参数：[Client Connection Defaults](https://www.postgresql.org/docs/16/runtime-config-client.html)
