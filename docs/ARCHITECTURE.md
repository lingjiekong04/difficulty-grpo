# 架构、依赖与算法边界

## 一次更新如何执行

1. `DifficultySampler.__iter__` 按当前概率有放回选择 64 个位置；`num_workers=0`，不通过多进程预取滞后消费统计。
2. verl 在 repeat 之前为每次出现分配 UID，同一道题出现两次就是两个四回答组。vLLM TP2 从当前权重产生新轨迹。
3. `StrictNumericRewardManager` 使用最后 token 是否 EOS，把终止元数据放入隔离的 extra_info；奖励解析器只接受一个最终 `#### 数值`，正确且完整为 1。
4. verl 计算组内中心化并按标准差归一化的优势，FSDP2 actor 做一次 PPO epoch；KL 在 actor loss 中单独加入，token-mean 聚合。
5. `ResearchTrainer._update_actor` 在更新成功后按 UID 汇总反馈，更新统计、访问次数、混合组与 token 计数。下一批读取新分布，不重用旧回答。
6. actor 更新后的权重经上游 checkpoint manager 同步到 rollout；同步/更新边界可请求安全保存。

## 核心接口

|模块|接口|状态与约束|
|---|---|---|
|sampler|probabilities / feedback / state_dict / load_state_dict|p、visits、completed、position、独立 random.Random；恢复检查 ID/种子/组别/批量/步数|
|train|install_adapter / bind_trainer|重新绑定 Ray 序列化函数 globals，保证实际 worker 使用 ResearchTrainer|
|full_engine|trainability / optimizer_coverage / optimizer_step|禁止 PEFT、拒绝冻结；检查可训练张量恰好进入优化器一次；有限梯度、抽样权重变化|
|checkpoints|latest_complete / prune_full_checkpoints|模型/Adam/sidecar 完成才发布标记；只选完整 checkpoint；运行中按既定保留策略清理旧完整点|
|evaluate|score_completion / evaluate|TP1 参考生成；严格 EOS 和共享数值判分|
|test_plan|build / freeze|先固定角色与推理核心哈希，禁止重用改变过的计划|

重复出现同题的反馈顺序会影响后续 EMA，这与“每次出现独立组”的定义一致。题目采样无重要性校正，训练分布改变；不能把差异全部解释为无偏提高同一目标的估计效率。

## 组信号直觉

二元奖励四回答的混合概率在独立同分布 Bernoulli(p) 理想假设下为 `1 - p^4 - (1-p)^4`，在 p=0.5 最大。`p(1-p)` 是易计算的代理，并非该混合概率本身；同组回答也未必独立。全 0/全 1 组中心化奖励优势为零；不能因此声称整体梯度为零。

## 内存与吞吐

FP32 主参数、FP32 Adam 状态和原生 CPU offload，FSDP2 计算时 materialize BF16；CPU offload 不代表 LoRA 或冻结。Qwen 混合注意力中的 gated-delta 路径使用 FLA 0.6.0；remove-padding=false，额外 fused LM-head=false。SDPA/梯度检查点配合微批量控制峰值。

两卡训练与 TP2 rollout 采用同步共置，GPU 资源按阶段使用；离线评测 TP1 可以一张卡一个独立模型。未声称异步训练、解耦服务架构或持续服务。原短测整卡两秒采样峰值约 55GiB/卡，只是采样下界；无连续显存利用率证据，不能写“显存打满”。算子 11–13 倍实验不能当作端到端加速。

## 恢复边界

上游保存 actor 模型、优化器、调度/extra state、数据加载器；研究侧保存 sampler、累计 token、已用时间、最佳点、driver Python/NumPy/Torch RNG。恢复由 complete.json 原子标记保护，缺失标记不会作为自动续训来源。

历史第 2→3 步恢复有 driver RNG/采样访问与随机流、Adam 计数及权重更新记录。本交接未包含大 `.pt` 字节，不独立反序列化验真；vLLM/worker 的全部内部 RNG 没有逐 token 连续承诺。发布包不带检查点，所以本地 CPU 状态演示只验证接口语义。

## 依赖边界

标准库可运行 sampler/reward/data split 与部分编排测试。GPU 路径需要 torch、verl、vLLM、Ray、Hydra/OmegaConf、NumPy、datasets/HF Hub、transformers、pyarrow、torchdata、FLA 等。`pyproject.toml` 的空 dependencies 只表示没有自动拉取 CUDA 栈；完整训练依赖须按安装脚本和冻结环境准备。

上游 Git 工作区干净，固定基线加本目录覆盖层可重建，不分发完整 vendor。`tools/check_upstream.py` 验证 commit 及已私审文件哈希；这不是整个上游树逐文件审查。部署兼容 `compat/sitecustomize.py` 只针对 Ray dashboard agent 的 NVML 查询，不关闭训练 GPU；是否仍适用新驱动需要重新验证。
