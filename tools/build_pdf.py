"""Rebuild the Chinese report using reportlab and a user-provided CJK TTF.

Optional: set DAGRPO_PDF_FONT. Fonts are not distributed in this repository.
This script creates a document only; it never imports the training stack.
"""
from pathlib import Path
import os,json,html
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,Flowable
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/difficulty-grpo.zh-CN.pdf'
FONT=Path(os.environ.get('DAGRPO_PDF_FONT','C:/Windows/Fonts/simhei.ttf'))
if not FONT.is_file():raise SystemExit('Set DAGRPO_PDF_FONT to a licensed Chinese TrueType font.')
if OUT.exists():raise SystemExit('PDF output already exists; preserve it and choose a fresh candidate for revisions.')
pdfmetrics.registerFont(TTFont('CJK',str(FONT)))
pdfmetrics.registerFontFamily('CJK',normal='CJK',bold='CJK',italic='CJK',boldItalic='CJK')
OUT.parent.mkdir(parents=True,exist_ok=True)
BLUE=colors.HexColor('#185879');DARK=colors.HexColor('#243746');LIGHT=colors.HexColor('#EDF4F7')
W=A4[0]-88
styles={
 'title':ParagraphStyle('title',fontName='CJK',fontSize=27,leading=38,textColor=DARK,spaceAfter=15,wordWrap='CJK'),
 'sub':ParagraphStyle('sub',fontName='CJK',fontSize=15,leading=23,textColor=BLUE,spaceAfter=12,wordWrap='CJK'),
 'h':ParagraphStyle('h',fontName='CJK',fontSize=17,leading=24,textColor=BLUE,spaceAfter=12,wordWrap='CJK'),
 'h2':ParagraphStyle('h2',fontName='CJK',fontSize=12,leading=18,textColor=DARK,spaceBefore=11,spaceAfter=6,wordWrap='CJK'),
 'p':ParagraphStyle('p',fontName='CJK',fontSize=10.3,leading=16,spaceAfter=8,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='CJK',fontSize=8.6,leading=13,spaceAfter=6,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='CJK',fontSize=9,leading=13,wordWrap='CJK'),
 'code':ParagraphStyle('code',fontName='CJK',fontSize=8.6,leading=14,backColor=LIGHT,borderPadding=9,spaceBefore=5,spaceAfter=10,wordWrap='CJK')}
story=[]
def p(text,kind='p'):story.append(Paragraph(text,styles[kind]))
def h(text):p(text,'h2')
def title(n,text):p(f'{n:02d}  {text}','h')
def page():story.append(PageBreak())
def table(rows,widths):
 cells=[[Paragraph(html.escape(str(c)),styles['cell']) for c in row] for row in rows]
 t=Table(cells,colWidths=[W*x for x in widths],hAlign='LEFT',repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('TEXTCOLOR',(0,0),(-1,0),BLUE),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.6,BLUE),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#DCE5EA'))]));story.append(t);story.append(Spacer(1,8))
def image(name,height):story.append(Image(str(ROOT/'docs/assets'/name),width=W,height=height,kind='proportional'));story.append(Spacer(1,8))
def code(text):p(html.escape(text).replace('\n','<br/>'),'code')

class Architecture(Flowable):
 def __init__(self):Flowable.__init__(self);self.width=W;self.height=226
 def draw(self):
  c=self.canv;labels=[('固定训练池 / 2048题','U/A/S/E有放回采样'),('vLLM TP2当前权重','每题4个全新回答'),('严格数值 + EOS奖励','按独立UID计算组优势'),('FSDP2全参数更新','BF16计算 / FP32主权重'),('权重同步 / 完整检查点','反馈EMA后选择下一批')]
  for i,(a,b) in enumerate(labels):
   y=188-i*44;c.setFillColor(LIGHT);c.setStrokeColor(BLUE);c.roundRect(30,y,W-60,35,5,fill=1,stroke=0)
   c.setFillColor(DARK);c.setFont('CJK',10);c.drawString(42,y+20,a);c.setFont('CJK',8.5);c.drawString(42,y+7,b)
   if i<4:c.setStrokeColor(BLUE);c.line(W/2,y,W/2,y-8);c.line(W/2,y-8,W/2-3,y-4);c.line(W/2,y-8,W/2+3,y-4)

# Cover
story.append(Spacer(1,42))
p('Difficulty GRPO','sub')
p('面向有限算力的<br/>难度感知数学推理训练','title')
p('Qwen3.5-9B全参数强化学习<br/>两卡实现、评测核验与负结果分析','sub')
story.append(Spacer(1,16))
p('项目摘要','h2')
p('研究在线题目选择能否提高GRPO训练效率。基于verl与vLLM，在同机两张A100 80GB上完成均匀U、在线A、冻结S、零探索E共8组真实训练，并对评测生成漂移、完整检查点与成本进行核验。')
p('主要成果是可检查的实现与实验比较。固定200步测试终点，U为95.38±0.66%，A为95.25±0.52%；A额外消耗6.74%已记录卡时，未建立稳定泛化或效率优势。', 'p')
table([['展示层级','事实边界'],['历史真实模型实验','以当前远程交接证据核对；本轮未重新训练/推理'],['本轮本地验证','标准Python演示、统计复算与CPU测试'],['业务/生产验证','无线上业务、用户实验或生产负载证据']],[.25,.75])
p('整理日期：2026-11-06  |  作者与GitHub地址：未提供，不猜测署名','small')
p('本报告对应独立公开候选。完整协议、来源哈希与剩余缺口随仓库提供；原实验目录与历史验收清单保持不变。','small')
page()
title(2,'问题、目标与系统流程')
p('全对或全错的二元奖励组缺少组内相对任务信号。项目尝试把更多预算分给当前策略成功率处于中间的题目，但必须同时检验覆盖、泛化与实际成本。')
story.append(Architecture())
h('研究干预与对照')
p('仅改变题目选择分布，模型、优化器、生成、奖励、损失与算法批量保持一致。U/A分别3个训练种子，S/E分别seed42。正式每组200步，前20步统一均匀。')
h('同步训练与推理角色')
p('vLLM承担当前策略rollout和推理缓存/调度；verl负责训练流程和GRPO优势，FSDP2负责参数分片和更新。actor更新后同步权重，再生成新轨迹。训练rollout是TP2，离线参考评测是TP1。')
h('交付定位')
p('使用者可读核心代码、无需GPU演示采样和恢复、复算公开成绩；具备GPU和依赖后可准备固定数据并运行新实验。公开包不包含权重，因此不能开箱加载原训练模型。')
p('源码：dagrpo/sampler.py、train.py、full_engine.py、runner.py；完整架构见docs/ARCHITECTURE.md。','small')
page()
title(3,'在线难度采样的实现')
p('每题维护成功率p和访问次数，未访问题p=0.5。首次反馈直接取四回答平均正确率，此后EMA新观测权重0.3。', 'p')
code('s_i = p_i * (1 - p_i)\nA: q_i = 0.2/N + 0.8*s_i/sum(s)\nE: q_i = s_i/sum(s)\nsum(s)==0: uniform fallback\nS: freeze p_i after the 20-update warmup')
h('关键实现约束')
p('题目有放回抽取；同题重复出现仍是独立回答组。上游在repeat前创建UID，反馈按UID聚合，不能按题目ID合并，也不能依赖重排后的相邻行。data loader的num_workers=0，下一批使用已更新统计。')
p('只复用难度统计，不缓存旧回答。更新成功后反馈，再更新累计训练token、混合组与覆盖率。S冻结p但继续记访问数；E即便无探索也保留全零分数回退。')
h('为什么p(1-p)只是代理')
p('在独立Bernoulli回答假设下，4回答混合概率为1-p^4-(1-p)^4，在p=0.5最大。实际组内回答相关，历史EMA也有噪声/滞后；p(1-p)不等于真实混合概率。')
h('优势与目标分布')
p('全同奖励中心化后任务优势为0，KL loss仍可能有梯度。项目不做题目采样重要性校正，因此改变经验训练目标；不能称为对同一分布无偏提高效率。优势按组标准差归一化，loss按token-mean聚合，与原始按序列表述有实现差异。')
p('接口：probabilities、feedback、state_dict、load_state_dict。CPU示例展示人造奖励与JSON状态恢复，不代表真实模型推理。','small')
page()
title(4,'两卡全参数与可恢复工程')
table([['配置','采用值'],['模型与优化器覆盖','9,409,813,744可训练参数；760张量；LoRA rank=0'],['参数/计算','FP32主参数与Adam；BF16 GPU计算；FSDP2原生CPU卸载'],['算法/微批量','64题×4回答/更新；单卡微批量8；PPO epoch=1'],['训练','lr1e-6；clip0.2；KL loss0.001；梯度检查点'],['注意力','SDPA、FLA0.6.0；remove-padding=false；额外fused LM-head关闭']],[.25,.75])
h('全参证据')
p('引擎拒绝PEFT和冻结参数；检查全部可训练张量恰好进入优化器一次。两rank参数审计、有限梯度、嵌入/前中后层/LM head探针实际变化支持真实更新。不声称每个参数元素每一步都变化。')
h('完整检查点提交')
p('先保存研究sidecar，再由上游保存模型、Adam、数据/extra状态，最后发布complete.json。自动恢复仅选择完整点；研究状态包含采样器、访问次数、累计token、最佳点与driver Python/NumPy/Torch RNG。')
p('历史2→3步恢复核验了Adam计数、采样位置128→192、下一批访问与随机流。此次同时切换FLA，属于工程验证，不是算法对照。正式八组固定FLA。vLLM内部全部随机状态的逐token连续性未获证明。')
h('资源与保留策略')
p('CPU卸载需要足够主机内存/带宽与大磁盘。短测约55GiB/卡为两秒采样峰值下界，不称显存打满；FLA算子11-13倍不等于端到端加速。正式save2，保留最新两份与最佳点；旧完整点会按既定策略清理，因此部分曲线无法重建。')
page()
title(5,'数据、奖励与评测核验')
table([['数据/参数','口径'],['训练池 / 开发 / 测试','2048 / 512 / 1319；ID来自题目规范化哈希'],['拆分来源','官方train7473；固定seed42划dev再取训练池'],['prompt / response','512 / 1024 token；thinking=false'],['奖励','恰有一个最终####数值、正确且以EOS结束得1；其他0'],['参考解码','TP1/BF16/eager/greedy，128题分块；seed42']],[.28,.72])
p('manifest中3879 ID互斥。本轮未重新采完整数据；公开模型可能接触基准，不主张零污染。数值归一化支持千位分隔、小数和负数；不做单位换算，不接收分数表达。格式错和非EOS可重叠，不能相加当作错误数。')
h('greedy并不自动等于可复现')
p('默认三次正确481、481、479/512，首次与第三次有67题token变化、4题判分变化。v2关闭引擎多进程后冷启动仍漂移；v3关闭异步调度且TP2时仍有200题token、6题得分差异。')
p('v4改TP1、禁用自定义多卡归约、eager、确定性设置、关闭TF32，固定PYTHONHASHSEED与cuBLAS工作区，禁用prefix cache/引擎多进程/异步调度。基座2进程×3次一致；训练权重dev双次及CLI冷启动核对后才复评。')
h('冻结测试计划')
p('测试前固化主次终点、数据ID/哈希、四份核心评测源码与17角色。基座test两张卡各冷启动一次一致，其余14训练权重test各一次；不保证跨硬件确定性。选点来自历史在线dev，不根据test重选。')
page()
title(6,'主结果：收益随选点规则变化')
image('test-results.png',206)
table([['终点','U均值±样本SD','A均值±样本SD','A-U'],['固定step200','95.38±0.66%','95.25±0.52%','-0.13pp'],['原在线dev选点','94.47±0.42%','95.58±0.57%','+1.11pp']],[.25,.27,.27,.21])
p('官方测试划分1319题，本项目严格评分；n=3训练种子，SD单位为百分点。基座94.92%。不是置信区间，不声称显著；同题跨模型重复不能当独立样本。')
table([['seed','U200正确数','A200正确数','差值pp'],['42','1268/1319','1251/1319','-1.29'],['43','1253/1319','1264/1319','+0.83'],['44','1253/1319','1254/1319','+0.08']],[.15,.3,.3,.25])
h('开发集与测试集必须分开')
p('dev512参考固定终点：U93.82±0.11%，A94.86±0.11%；原在线选点参考复评U95.18±0.60%，A94.86±0.23%。不能将dev/test不同分母、不同协议的结果拼成提升曲线。')
p('当前证据不支持“稳定泛化提升”。次终点更好且主终点无优势说明结论对选点敏感，不能只展示最好结果。公开标量评分可复算统计，但不是本轮新模型输出。')
page()
title(7,'消融、成本与分布集中')
image('cost-and-signal.png',177)
table([['seed42','U','A','S','E'],['测试step200','96.13%','94.84%','94.69%','95.53%'],['末端零概率题','0','0','0','1892'],['最大单题概率','0.049%','62.55%','0.080%','99.68%'],['exp(熵)题数','2048','26.76','1514.88','1.04']],[.28,.18,.18,.18,.18])
p('A平均卡时比U多6.74%，训练回答token多17.94%；混合组均值5.10%→13.62%。增加代理信号不等于能力或效率提高。S/E只有单种子，不能外推一般优劣。')
h('末端状态不是全程采样分布')
p('快照位于200步反馈后、下一批选题前。20%探索只保证正概率，不阻止A剩余质量集中。E最后两步128次出现涉及79题，单题最多4次，不能由末端99.68%倒推出整段只练一道题。')
h('成本解释')
p('卡时包含已记账初始化、恢复、保存、在线评估与尝试；训练token只含回答。离线推理和控制器预留时间存在重叠，不能重复相加。重启次数不同也可能影响卡时。固定200步不能直接证明等卡时/等token效率。')
page()
title(8,'真实迭代：组件分别记录')
table([['组件/阶段','改变与问题','验证 / 最终状态'],['模型候选','Qwen2.5-3B候选→Qwen3.5-9B；历史LoRA调试→全参','3B无正式训练证据；LoRA不计正式成果；9B全参采用'],['数据/输入','固定2048/512/1319；thinking=false与严格最终数值','ID/哈希与边界测试；采用；后续数据升级无'],['执行层','Ray绑定、NVML隔离、端口、前台.py、save2','回归测试与真实记录；平台兼容需要新环境再验'],['计算层','Torch回退→FLA0.6.0；32题计划→64题正式','BF16对照及真实模块审计；正式各组一致'],['采样算法','U / EMA在线A / 冻结S / 零探索E','8组200步；没有后续新算法版本'],['评测协议','默认→v2→v3→TP1参考v4','冷启动漂移排查后采用v4；不能视为算法提升'],['发布整理','新候选、README/来源映射/CPU示例/PDF','本轮本地检查；不是新训练或算法升级']],[.19,.42,.39])
h('保留历史问题与失败')
p('实例中断前step8未有完整点，后以save2新目录重跑；旧消耗未抹去。默认和过渡评测结果只作历史探索，清理过的中间权重无法事后补造参考曲线。')
h('归档控制脚本差异')
p('历史archive-manifest与当前run_official_test.py哈希不同，未找到对应旧源码；68份可审核归档项中67匹配。四份冻结评分/推理核心与当前一致，结果文件也匹配。明确保留此版本缺口，不刷新历史清单，也不称控制器逐字可重放。')
p('详细的“阶段→变化→问题→验证→是否采用”表见docs/ITERATION_HISTORY.md。项目没有主Git HEAD，不能补造原提交历史。','small')
page()
title(9,'案例、展示与证据边界')
h('奖励边界：单位等价也可能判错')
p('历史dev案例中，模型正确得到1350分钟，再换算为22.5小时并以22.5作最终数值，会被严格答案1350判错。这是评分口径限制，不等于模型不会算时长。其他案例有折扣规则理解回退；不只挑成功回答。公开包不附原题/原回答，案例为历史报告的概括。')
h('五分钟展示')
code('python -B examples/sampler_demo.py\npython -B tools/summarize_results.py\npython -B tools/verify_release.py')
p('先说明研究问题与负结果，再展示EMA概率、独立UID组与JSON恢复，随后复算1319题成绩，最后讲v4评测及来源映射。示例为人造奖励；现场演示不是模型推理，不启动GPU。')
h('本轮验收与未运行项')
p('私审包496文件完整性通过；3879 ID互斥；17角色/15权重的评分投影与源hash一致；68条原行样本重评分。CPU测试61项中53通过、8因缺训练依赖跳过。初次临时目录权限错误保留，迁到工作区临时目录后通过。')
p('未执行SSH、依赖安装、下载、训练、GPU推理或重新加载大checkpoint。公开结果是派生摘要，不能代替独立重跑；权重SHA未知、跨硬件确定性未知、业务有效/生产可用未验证。')
h('未来研究')
p('可研究概率温度/上限、成功率先验和平滑、难基准及等预算试验，但本轮没有实现或验证。应另冻结开发/测试规则，不针对本次test反复调参。')
page()
title(10,'复现路径、来源与许可')
h('分层复现')
table([['层级','入口/条件'],['CPU','verify_release、sampler_demo、summarize_results；完整测试需matplotlib，训练集成按依赖跳过'],['环境','Linux；scripts/setup_cu129.sh；固定verl上游与CUDA版本；本轮未安装'],['数据','scripts/prepare_locked.py；固定模型/数据revision；空输出目录'],['真实训练','CLI pilot总steps保持200，但stop-after-step=2，续至3；成功再新目录phase full'],['评测','导出自有checkpoint；dev冷启动核验后冻结test计划；原权重不分发']],[.18,.82])
h('来源与公开范围')
p('远程私审包SHA256：<br/>71547702137d69407406f33ac4b807c72d91dc81f313cb3ac9603c4fbb744a88','small')
p('来源映射和提取规则位于provenance，严格文件清单位于RELEASE_MANIFEST.json。仅公开源码、配置、标量证据、图表和文档；不公开原始回答、部署地址/主机信息、凭据、模型/Adam、环境或完整数据。')
h('固定资源与参考资料')
p('verl v0.9.1：commit 1876b06d0a3e4e71e06230be10af14492ca8a75b；模型Qwen/Qwen3.5-9B：revision c202236235762e1c871ad0ccb60c8ee5ba337b9a；openai/gsm8k：revision 740312add88f781978c0658806c59bc2815b9866。完整可点击链接见dependencies.lock.json及third_party/THIRD_PARTY.md。','small')
p('参考：verl固定core_algos.py（组优势实现）；vLLM reproducibility文档（https://docs.vllm.ai/en/latest/usage/reproducibility/）；GSM8K dataset card；Qwen缓存model card。外链未在本轮联网验证，结论依据已交接版本，不外推当前最新版本。','small')
h('许可与贡献边界')
p('候选根新增Apache-2.0，保留上游版权头及verl/Qwen许可副本。GSM8K缓存卡片标记MIT，不打包原数据。GRPO、模型、verl/vLLM/FLA归上游；新增工作是采样、全参工程、恢复审计、评测控制与分析。作者/仓库地址未提供；上传由用户手动完成。')
p('完整命令与风险边界见README及docs/REPRODUCE.md；本PDF是发布说明，不替代原始私有证据。','small')

def footer(c,doc):
 c.setStrokeColor(colors.HexColor('#DCE5EA'));c.line(44,42,A4[0]-44,42)
 c.setFont('CJK',8);c.setFillColor(colors.HexColor('#5D7281'));c.drawString(44,28,'Difficulty GRPO | 公开候选 / 2026-11-06');c.drawRightString(A4[0]-44,28,str(doc.page))
doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=42,bottomMargin=55,title='难度感知全参数GRPO：两卡训练与实验分析',author='',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
