# 分层复现说明

## 能复现到哪一层

|层级|所需资源|本候选与本轮情况|
|---|---|---|
|文件与统计|标准Python|源码哈希、公开评分重新统计、split ID检查|
|算法接口模拟|标准Python|EMA/有放回选题/奖励/JSON状态恢复；人造输入|
|CPU完整历史测试|标准库+matplotlib；训练集成另需torch/verl|本轮详见VALIDATION；跳过不等于通过|
|固定依赖重建|Linux、网络、Python3.12、CUDA兼容驱动|脚本与固定上游完整；本轮未安装或联网下载|
|真实推理|模型文件、一张足够显存的GPU|历史TP1参考；本轮未运行|
|训练/真实续训|同机两张80GB GPU、足够CPU/RAM/磁盘|历史记录；本轮没有checkpoint字节可重新加载|
|原成绩逐位重现|原权重/环境/硬件及全部随机状态|不保证；训练生成内部RNG边界及缺失中间权重有缺口|

## Windows CPU

在本地Windows PowerShell：

```powershell
$Project = Read-Host '项目绝对路径'
Set-Location -LiteralPath $Project
python -B tools/verify_release.py
if ($LASTEXITCODE -ne 0) { throw '保留原始校验错误' }
python -B examples/sampler_demo.py
if ($LASTEXITCODE -ne 0) { throw '保留原始演示错误' }
python -B tools/summarize_results.py
if ($LASTEXITCODE -ne 0) { throw '保留原始统计错误' }
python -B -X utf8 -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw '保留原始测试错误' }
```

继承测试的 `test_empty_report` 使用matplotlib；如本机没有该包，可单独先跑 `python -B -X utf8 -m unittest discover -s tests -p test_evaluation_eos.py -v`，或在自行选择的环境准备绘图库后再跑完整集合。发布CI安装单独的 `requirements-check.txt`，不拉CUDA。Windows不用于真实分布式GPU训练。

## Linux GPU：用户自行执行，不属于发布验收

所有下列命令在**新解压的候选项目根目录**执行，不对已有实验目录执行。需要可用网络和足够磁盘：9B FP32主参数加Adam/保留多个检查点可能占数百GB，CPU卸载还需要充分RAM；具体峰值未在本轮重测。不要直接对照抄旧机器路径。

```bash
python3 -B tools/verify_release.py
# 安装到本项目 .venv，下载CUDA/上游资源；脚本末尾含GPU可用性检查
bash scripts/setup_cu129.sh
.venv/bin/python -B tools/check_upstream.py
# 对齐实际模型与数据revision；不读取latest
.venv/bin/python -B scripts/prepare_locked.py --out data/gsm8k-qwen35-9b
```

原安装脚本使用公开wheel URL及版本约束；未来URL可用性和所有传递依赖解算未在本轮验证。公开freeze去掉两个本机editable路径，保留历史版本；它是记录，不承诺跨系统通用lock。安装脚本会重新记录环境freeze，须把新文件作为新运行证据，不能修改历史私有冻结原件。

### 两步短测与一步恢复

```bash
export CUDA_VISIBLE_DEVICES=0,1
.venv/bin/python -u -m dagrpo.cli run --phase pilot --variant U --seed 42 \
  --steps 200 --save-freq 2 --stop-after-step 2 --training-mode full \
  --batch-size 64 --micro-batch-size 8 --out runs/public-smoke-U42 \
  --ledger runs/public-smoke-budget.json
# 第2步安全停机，退出码2表示checkpoint可续训，不是任务失败/200步完成。
# 同一协议、同一目录，边界改3；检查STUDY_RESTORED、审计和complete.json。
.venv/bin/python -u -m dagrpo.cli run --phase pilot --variant U --seed 42 \
  --steps 200 --save-freq 2 --stop-after-step 3 --training-mode full \
  --batch-size 64 --micro-batch-size 8 --out runs/public-smoke-U42 \
  --ledger runs/public-smoke-budget.json
```

保留steps=200是为了恢复元数据兼容；stop-after-step控制真实只执行2步/再1步。不会短测200步。可另用verify_full_resume对比受信任的study_state.pt；该脚本会反序列化pickle，只对你自己生成的文件使用。本轮未运行它。

### 正式单组/恢复

短测成功后，用新正式目录起跑；不要把工程短测目录改成正式phase复用。以下为U42，A/S/E或种子只改对应变量，其他协议一致。

```bash
.venv/bin/python -u -m dagrpo.cli run --phase full --variant U --seed 42 \
  --steps 200 --save-freq 2 --training-mode full --batch-size 64 \
  --micro-batch-size 8 --out runs/public-U42 --ledger runs/public-full-budget.json
```

中断后重执行同一命令，自动选择最新完整检查点。没有GPU小时上限，账本保留；需要时可显式加 `--budget-gpu-hours` 或 `--session-hours`。不要启动重复任务；历史平台入口带互斥，直接CLI单组应自行保证独占两卡。

定期保存后仅保留最新两份和best-dev对应点，继承的prune函数会删除其识别为过期的完整全参点，未知/不完整目录不动。若研究需要全程参考曲线，应在新实验开始前设计不同保留策略和额外存储，不能用已清理权重补造曲线。

### 真实离线评测

使用pinned verl模型合并器导出你自己的actor checkpoint，再按参考协议运行：

```bash
.venv/bin/python -m verl.model_merger merge --backend fsdp \
  --local_dir runs/public-U42/checkpoints/global_step_200/actor \
  --target_dir exports/public-U42-step200
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m dagrpo.cli evaluate \
  --model exports/public-U42-step200 --data data/gsm8k-qwen35-9b/dev.jsonl \
  --out reports/public-U42-dev.jsonl
```

模型合并命令接口来自固定上游及历史源码调用；本轮未重新执行。先用dev验证冷启动一致性，最后冻结选择后才用test。CLI会设置hash/CUBLAS环境并重新exec，输出已存在会拒绝覆盖。训练权重不随包发布，没有公共自训练权重下载链接；不能宣称上述命令可直接复现原权重成绩。

## 下载与哈希

[dependencies.lock.json](../dependencies.lock.json)记载上游commit、模型/数据revision及已知prepared文件SHA。模型safetensors权重SHA未知，不能只凭revision把本轮未下载权重称为已核验。数据内容digest使用sort_keys/ensure_ascii=False的规范化JSON；文件SHA使用原字节，各保留原口径。

新 `prepare_locked.py` 只在空目录准备，使用相同prepare_records/export核心、固定revision并对比内容split digest。输出parquet字节可能受库版本影响，如字节不同需记录，不用更新历史hash掩盖。HF服务和下载URL未在发布整理阶段访问。

## 图表源

`tools/build_figures.py`从公开results和历史online-dev标量文件生成三张图，需要requirements-check.txt。脚本拒绝覆盖既有图；本轮已核验成图可直接复用。仅在新副本空assets目录重建并建立新manifest。PDF源码与报告依赖见VALIDATION。
