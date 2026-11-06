# 五分钟展示提纲

1. 用README结果表先说明研究问题和负结果；指出主终点与历史选点方向不一致。
2. 展示架构：选题→当前策略rollout→严格奖励→GRPO→全参更新→反馈/同步→检查点。
3. 在项目根目录运行 `python -B examples/sampler_demo.py`：展示概率和JSON恢复完全一致；明确它使用人造奖励，未加载模型。
4. 运行 `python -B tools/summarize_results.py`，由公开评分文件重新统计1319题与3seed均值/样本SD；打开实验图和末端分布表。
5. 指出train.py按UID分组、full_engine优化器覆盖、v4评测与版本缺口，再用limitations结束。不在面试现场启动昂贵训练。

真实模型推理需要自己准备已导出权重和GPU；公开包没有在线服务、交互问答模型或API。演示截图/模拟输出不能作为真实推理验收。
