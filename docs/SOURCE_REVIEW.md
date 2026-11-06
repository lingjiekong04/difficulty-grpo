# 源码依赖链与实现核对

|入口/组件|依赖链与实现|审核结论|
|---|---|---|
|dagrpo.cli prepare|data→datasets/HF/tokenizer→split/export|实现真实数据准备；旧入口取latest，新prepare_locked固定revision|
|dagrpo.cli run|runner→train→verl main_ppo / TaskRunner→ResearchTrainer→full_engine / reward|完整训练调用链；外部重资源不随包打包；本轮仅静态/CPU检查|
|run_training/run_uniform|重exec项目venv→jobs→runtime→ready→runner|历史前台平台入口；依赖旧READY证明，不是新环境首个入口|
|run_repeats/run_ablations|platform→stages→jobs，锁/阶段先决条件/完成门槛|实现队列、中断停止和本组恢复；本轮mock测试而非真实GPU队列|
|sampler|独立Random、p/visits、EMA、概率和恢复|U/A/S/E均实际实现；未实现概率上限/新优化算法|
|train|UID反馈/Tracking副作用/完整checkpoint/恢复RNG|继承同步trainer；源码有真实worker绑定；不承诺全部worker/vLLM RNG|
|full_engine|EngineRegistry公开vendor hook→FSDPEngineWithLMHead|全参优化器与权重探针；上游tracked tree干净，无隐藏补丁假设|
|reward/evaluate|Decimal+regex→EOS共享判分；vLLM离线生成|数值/格式/EOS规则实现；局限是单位/分数等价不判|
|eval_audit/eval_queue/test_plan|逐题比较→资源隔离队列→冻结角色和源hash|可恢复评测实现；生成确定性依赖实测，不由flag保证|
|analysis/report/profile|账本、协议指纹、聚合/绘图、全参证明门槛|有实现；历史曲线含噪声，不能证明同预算优势|
|startup_retry/gpu_idle/locking|有限启动重试、等待指定设备、跨平台锁|不把失败当完成；测试使用模拟进程，实际GPU未运行|
|examples/tools新文件|人造奖励/公开评分/严格manifest/固定资源/文档/PDF/用户上传|发布辅助，不是新模型能力或业务功能|

全部Python已解析AST；本地相对模块存在性检查通过。包内无完整第三方代码，固定依赖检查工具提供重建边界；安装可行性和完整GPU链路只记历史证据，本轮未验证。未实现真实业务服务、多轮Agent、检索或生产API。
