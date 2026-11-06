"""Create the final development-only evaluation audit and corrected comparison."""
from pathlib import Path
import json,sys,os
from statistics import mean,stdev
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from dagrpo.data import atomic_json
from dagrpo.stages import run_directory

def main():
    sync=os.environ.get('DAGRPO_EVAL_SYNC')=='1'
    reference_mode=os.environ.get('DAGRPO_EVAL_REFERENCE')=='1'
    suffix='-'+os.environ.get('DAGRPO_EVAL_TAG','sync') if sync else ''
    if not sync:raise SystemExit('Transitional asynchronous reports must not be presented as final')
    base=ROOT/'reports/evaluation-audit-20261103'
    stable=ROOT/f'reports/evaluation-stable-20261103{suffix}'
    revised=ROOT/f'reports/saved-model-dev-recheck-20261103{suffix}'
    assert json.loads((ROOT/f'logs/saved-model-eval{suffix}-job.json').read_text())['status']=='completed'
    a=json.loads((base/'comparison.json').read_text());b=json.loads((stable/'comparison.json').read_text())
    rows=json.loads((revised/'results.json').read_text());assert len(rows)==12
    assert len(b['sessions'])==6 and all(not ids for d in b['comparisons'].values() for ids in d.values())
    assert all(r['repeated_tokens_identical'] for r in rows)
    cold=json.loads((revised/'U-42-step200/public-entry-comparison.json').read_text())
    assert all(not ids for ids in cold.values()),'Public cold-entry consistency is required'
    originals={}
    for v in 'UA':
        for seed in (42,43,44):
            p=run_directory(ROOT,'full',v,seed,{'SAVE_FREQ':'2'})
            e=sorted({r['step']:r for r in map(json.loads,(p/'evaluations.jsonl').read_text().splitlines())}.values(),key=lambda r:r['step'])
            originals[f'{v}/{seed}']=e[-1]['accuracy']
    stats={}
    for role in ['final','historical_best']:
        stats[role]={}
        for v in 'UA':
            rs=[r for r in rows if r['variant']==v and r['role']==role]
            values=[r['accuracy'] for r in rs]
            stats[role][v]=dict(mean=mean(values),sample_sd=stdev(values),seeds=[r['seed'] for r in rs])
    reference=next(iter(b['sessions'].values()))['accuracy']
    base_rows=[json.loads(l) for l in (stable/'session-0/repeat-0.jsonl').read_text().splitlines()]
    cases=[]
    for row in rows:
        if row['role']!='final':continue
        output=[json.loads(l) for l in (Path(row['attempt'])/'repeat-0.jsonl').read_text().splitlines()]
        changes=dict(variant=row['variant'],seed=row['seed'],format_end_fixes=[],answer_fixes=[],format_end_regressions=[],answer_regressions=[])
        for old,new in zip(base_rows,output,strict=True):
            assert old['problem_id']==new['problem_id']
            if old['score']==0 and new['score']==1:
                key='format_end_fixes' if old['format_error'] or old['truncated'] else 'answer_fixes'
                changes[key].append(old['problem_id'])
            elif old['score']==1 and new['score']==0:
                key='format_end_regressions' if new['format_error'] or new['truncated'] else 'answer_regressions'
                changes[key].append(old['problem_id'])
        cases.append(changes)
    out=ROOT/f'reports/evaluation-review-20261103{suffix}';out.mkdir(exist_ok=True)
    validation=json.loads((out/'validation.json').read_text())
    assert validation['exit_code']==0,'Server validation must pass before the report is finalized'
    training_source=ROOT/'reports/UA-three-seeds-20261103/results.json'
    training=json.loads(training_source.read_text())
    assert len(training['runs'])==6 and all(r['complete'] for r in training['runs'])
    training_cost={v:{k:training['summary'][v][k] for k in ['gpu_hours','training_tokens','mixed_group_rate']} for v in 'UA'}
    atomic_json(out/'review.json',dict(default_audit=a,stable_audit=b,checkpoints=rows,statistics=stats,
        original_final=originals,base_dev_accuracy=reference,case_analysis=cases,
        training_cost=training_cost,training_cost_source=str(training_source),
        official_test_used=False,training_updates_this_audit=0))
    lines=['# 开发集评测核验与统一复评','',
        '本轮只进行推理、判分、模型格式导出与数值核对，没有训练更新，也没有使用官方测试集。原模型权重、训练检查点和历史 evaluations.jsonl 均保留。','',
        '## 核验结果','',
        '默认推理设置的三次同进程评估为 481、481、479 道正确题；第一次与第二次逐 token 一致，第三次 67 道题输出改变、4 道题判分改变（3 道正确变错、1 道错误变对），prompt token 未变。冷启动对照也出现输出变化。这直接证明 temperature=0 在此前默认调度设置下不足以保证可复现。','',
        f'使用所记录的参考流程后，两个全新进程各重复三次，六次的逐题 token、文本、结束原因和判分全部相同，均为 {int(reference*512)}/512（{reference:.2%}）。第一个训练检查点还通过了正式 CLI 的独立冷启动逐 token 对照。这是当前硬件、模型、软件与固定流程下的实测，不是跨硬件或所有模型的普遍保证。官方说明：https://docs.vllm.ai/en/latest/usage/reproducibility/ 。','',
        '判分统一为共享数值解析器和最后 token 的 EOS 判断。默认模式与稳定模式的基础模型对照中，两种判分方式没有分歧；本次已复现的变化来自生成输出。历史 verl 异步评测未保存逐题轨迹，因此不能对历史每条错误逐一归因。','',
        '## 第 200 步结果','',
        '|种子|原在线 U|统一复评 U|原在线 A|统一复评 A|复评 A−U（百分点）|','|---|---|---|---|---|---|']
    deltas=[]
    for seed in (42,43,44):
        u=next(r for r in rows if r['variant']=='U' and r['seed']==seed and r['role']=='final')
        ad=next(r for r in rows if r['variant']=='A' and r['seed']==seed and r['role']=='final')
        d=(ad['accuracy']-u['accuracy'])*100;deltas.append(d)
        lines.append(f"|{seed}|{originals[f'U/{seed}']:.2%}|{u['accuracy']:.2%}（{u['correct']}/512）|{originals[f'A/{seed}']:.2%}|{ad['accuracy']:.2%}（{ad['correct']}/512）|{d:+.2f}|")
    lines+=['','以下 ± 为跨三个训练种子的样本标准差，准确率标准差单位为百分点。','',
        '|角色|U 均值±标准差|A 均值±标准差|A−U|','|---|---|---|---|']
    for role in ['final','historical_best']:
        u=stats[role]['U'];ad=stats[role]['A']
        lines.append(f"|{role}|{u['mean']:.2%} ± {u['sample_sd']*100:.2f}|{ad['mean']:.2%} ± {ad['sample_sd']*100:.2f}|{(ad['mean']-u['mean'])*100:+.2f} 个百分点|")
    lines+=['',f"固定终点三种子中 A 高于 U 的次数为 {sum(d>0 for d in deltas)}/3。基座模型统一开发集准确率为 {reference:.2%}。n=3，且所有准确率来自同一 512 题开发集；不宣称统计显著、测试集提升或计算效率提升。",'',
        '## 原训练成本与学习信号', '',
        '|每组训练的跨种子均值|U|A|', '|---|---|---|',
        f"|已记录卡时|{training_cost['U']['gpu_hours']['mean']:.2f}|{training_cost['A']['gpu_hours']['mean']:.2f}|",
        f"|训练回答 token（百万）|{training_cost['U']['training_tokens']['mean']/1e6:.2f}|{training_cost['A']['training_tokens']['mean']/1e6:.2f}|",
        f"|混合奖励组比例|{training_cost['U']['mixed_group_rate']['mean']:.2%}|{training_cost['A']['mixed_group_rate']['mean']:.2%}|", '',
        '卡时包含各组账本已记录的初始化、恢复、评估与保存等尝试，重启次数不同；token 只含训练回答，不含开发集。本轮新增推理成本另记。A 消耗更多回答 token 和卡时，因此固定 200 步的结果不能直接证明训练效率提升。混合组比例提高只证明更多组产生非零任务奖励优势，不能代替独立能力评估。', '',
        '## 相对基座的逐题变化', '',
        '|组别 / 种子|格式或结束条件纠正|有效格式下答案纠正|新增格式/结束错误|新增答案错误|',
        '|---|---|---|---|---|']
    for c in cases:
        lines.append(f"|{c['variant']} / {c['seed']}|{len(c['format_end_fixes'])}|{len(c['answer_fixes'])}|{len(c['format_end_regressions'])}|{len(c['answer_regressions'])}|")
    lines+=['','分类互斥，使用 format_error 或 truncated 的并集，不重复计数；这些是结果类别，不是对模型内部推理机制的因果解释。逐题 ID 保存在 review.json，可与完整回答对照。','',
        'historical_best 是原在线开发集流程选出的最佳检查点；本次只对该已保留权重重新评估，没有重新挑选。由于中间权重已按原保留策略删除，不能将它称作新评测流程下全程真正最佳，也不能事后重建完整确定性学习曲线。旧曲线保留为探索性记录。','',
        '## 复评一致性与工程验证','',
        '12 个角色引用对应 11 个不同检查点，重复引用复用同一结果。每个不同检查点均重新启动独立推理进程并评估两次，要求逐 token 一致；prompt/tokenizer 输入也与基座核验逐题一致。首个导出的全部权重通过 pinned verl.model_merger 数值测试。服务器测试成功结束，包括 EOS 判分边界、评测入口门槛与 GPU 设备隔离；完整测试数量及跳过情况见 tests.log 和 validation.json。','',
        ('统一配置：TP=1、eager、禁用自定义多卡归约、torch 确定性算法、禁用 TF32、PYTHONHASHSEED=0、CUBLAS_WORKSPACE_CONFIG=:4096:8；' if reference_mode else '统一配置：TP=2；')+
        'BF16、seed=42、temperature=0、top_p=1、top_k=-1、128 题固定分块、prefix cache 关闭、引擎多进程关闭、async_scheduling=false、prompt≤512、response≤1024、enable_thinking=false。导出为推理 BF16，与原全参数 FP32 主权重训练/优化器保存用途分开；原检查点未改写。','',
        '## 仍需完成','',
        'S/E 消融与官方测试集评测尚未在本轮执行。后续对所有保留权重使用本报告记录的同一评测规则；历史在线记录不得与统一复评数值混作同一条确定性曲线。训练卡时保留，新增导出/核验/复评开销独立记账。未把开发集生成 token 混入原训练回答 token 曲线。','',
        f'逐题证据：evaluation-audit-20261103、{stable.name}、{revised.name} 下的 repeat-*.jsonl / attempt-*/repeat-*.jsonl。脚本与模型 SHA、输入 token、终止原因、原始回答、判分和比较文件均可追溯。']
    (out/'report.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,role,title in zip(axes,['final','historical_best'],['Final checkpoint (step 200)','Historically selected best checkpoint']):
        for v,shift,color in [('U',-.18,'tab:blue'),('A',.18,'tab:orange')]:
            values=[next(r['accuracy']*100 for r in rows if r['variant']==v and r['seed']==s and r['role']==role) for s in (42,43,44)]
            ax.bar([i+shift for i in range(3)],values,width=.34,label=v,color=color)
        ax.axhline(reference*100,ls='--',color='gray',label='Base model')
        ax.set_xticks(range(3),['42','43','44']);ax.set_xlabel('Training seed');ax.set_ylabel('Dev accuracy (%)')
        ax.set_ylim(90,98);ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Unified synchronous offline evaluation | 512 dev problems | two identical repeats / checkpoint')
    fig.tight_layout();fig.savefig(out/'corrected-comparison.png',dpi=170);plt.close(fig)
    print('EVALUATION_REVIEW_READY',json.dumps(stats))

if __name__=='__main__':main()
