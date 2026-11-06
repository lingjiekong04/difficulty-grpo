# 上游来源与贡献边界

|资源|固定来源/许可|本候选处理|
|---|---|---|
|verl|https://github.com/verl-project/verl/tree/1876b06d0a3e4e71e06230be10af14492ca8a75b ，Apache-2.0|不打包vendor；保留自有文件中的上游版权头；许可证副本|
|Qwen3.5-9B|https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a ，缓存model card与LICENSE标记Apache-2.0|不含权重；单独许可证副本；权重SHA未提供|
|GSM8K|https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866 ，缓存dataset card标记MIT|不打包题文/数据，公开自生成ID/评分和拆分哈希；使用者自行遵守原数据许可|
|vLLM / PyTorch / Ray / Transformers / FLA|见安装脚本与冻结版本，各自版权/许可证由原发行物提供|不打包完整实现与编译扩展，不把其创新归为项目原创|

当前没有发现上游tracked代码修改，因此不提供虚构补丁；`tools/check_upstream.py`检查已审关键源码，完整依赖边界见dependencies.lock.json。Python模块中的Apache版权头保持字节不变。Qwen/LICENSE副本与主仓库LICENSE是不同来源文件，不混称。

本候选根Apache许可是本次交付选择；原根LICENSE/NOTICE缺失，不追溯改写历史。未获作者姓名/归属信息，不添加猜测署名。本项目新增贡献集中于题目采样、训练适配/审计/恢复、评测控制、实验分析与发布工具。用户应仅提交其有权分发的内容。

以上URL来自已审源码/缓存卡片，发布整理阶段未联网验证可访问性。下载失败属于外部资源缺口，不能用未固定的新版本默默替代历史配置。
