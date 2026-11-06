# 本轮检查记录（2026-11-06）

本轮只在本地审查用户交接包并制作独立候选。没有SSH、系统安装、GPU启动、训练、模型推理、创建远端仓库、commit或push。原实验、历史清单、私审包保持不变。

|检查|本轮结果|限制|
|---|---|---|
|私审包结构/hash|496文件，内部495条manifest全部匹配；外部SHA一致|安全解包至独立短目录，拒绝链接/特殊文件/越界；Windows长路径使用本地路径前缀|
|采集变化/缺失|已采文件与盘点无变化；4项明确缺失|根LICENSE/NOTICE、vendor NOTICE、generation_config；不伪造补齐|
|源码/依赖链|全部Python AST解析；相对模块存在；继承源hash核对|不等于CUDA依赖已安装或服务验收|
|固定上游+覆盖层|本地已存在上游HEAD/干净状态与远程一致；离线导出1229普通文件，六个审查源字节匹配，并叠加本项目模块|仅结构重建，不安装/导入上游；五个上游链接未跟随|
|全参历史|8组完成，双rank9.41B参数/760张量，历史proof核心hash匹配|未收checkpoint大字节，未独立加载Adam/模型|
|测试结果|17角色/15权重的1319题ID、正确数、token及格式/EOS计数匹配|基于采集器源hash和评分投影，不是本轮模型重跑|
|私有原行样本|68条可重评分样本一致|抽样不代替全部回答复审；原行不公开|
|拆分|manifest中3879题ID无交叉|完整题文未采；不证明基座无公开数据接触|
|基座重复|测试双冷进程评分及token/text哈希投影一致|其他test权重各一次；dev重复是历史记录|
|CPU测试|61项：53通过，8缺torch/torchdata/verl跳过|完整服务器集成未重跑；首次系统TEMP权限25错误日志保留，设置workspace TEMP/TMP后通过|
|新增示例/统计|人造奖励demo、JSON恢复和17角色重新统计通过|无模型/业务验收含义|
|源码/脚本语法|Python AST、PowerShell parser通过|上传脚本未联网、未真实commit/push；CI未在GitHub执行|
|PDF|10页PDFium渲染，逐页视觉检查；文本提取无替换乱码|无Poppler，使用已捆绑PDFium；检查图表、边距、页码、命令和表格|
|公开文件/链接/hash|本版严格manifest、来源映射、相对链接和ZIP一致性检查|下载URL未联网核验；新环境依赖安装未做|

## 历史测试与当前测试分开

私审包的评测参考日志记录55项通过，消融日志59项通过；原README曾称61项通过。本轮当前源码实际发现61项，53通过/8跳过。两份原始日志SHA和数量在historical-test-records.json，不用本轮skip替代历史服务器实测，也不借历史结果称本轮GPU通过。

## 发布差异与原件保护

核心dagrpo、tests、compat以及继承实验脚本保持字节。平台run_training.py仅修改含部署路径的错误提示，该文件LF规范化单独记载；环境freeze移除两个机器本地editable路径。新文档、prepare_locked、CPU示例、PDF与发布工具单列，不回写原工程。source-map记录每份继承源/候选hash及操作。

离线结构重建复用了已比对commit的本地上游。首次遇到上游agent-skill链接即停止，之后明确排除链接；默认Windows Git导出产生CRLF，与远程LF字节不符。最终用单次命令 `git -c core.autocrlf=false archive` 导出固定提交，六份上游源hash精确匹配。仅改变新导出的配置，未改继承核心换行/空白。两次不完整scratch保留，记录见evidence/offline-rebuild.json。此检查不证明完整依赖可安装或GPU可运行。

历史archive-manifest一处控制器脚本不匹配，详见AUDIT_FINDINGS；结果与冻结核心匹配，但该旧控制器源不可确认。历史清单不刷新。本版RELEASE_MANIFEST是全新的候选清单，不冒充原验收清单。

## 本轮日志在哪里

用户本机的独立release-audit-final目录保留两次测试原始输出、PRIVATE_AUDIT、PDF10页PNG、语法/链接/包检查记录；这些材料含本机路径，只交付用户私有查阅，不自动放GitHub。公开精简检查记录为evidence/release-checks.json。

## PDF重建

当前PDF已经逐页检查，可直接使用。tools/build_pdf.py提供源；需要requirements-report.txt和你有权使用的中文TTF（DAGRPO_PDF_FONT）。字体文件不随包分发。脚本拒绝覆盖已有PDF，请在新候选副本中生成新输出，并重新检查/建立新manifest，不能刷新历史清单冒充同版。
