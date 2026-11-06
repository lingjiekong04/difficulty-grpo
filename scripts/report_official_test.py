"""Report all prespecified test endpoints, without choosing a test winner."""
from pathlib import Path
import os,sys,json,tarfile
from statistics import mean,stdev
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from dagrpo.data import atomic_json,digest


def main():
    out=ROOT/'reports/official-test-20261105'
    assert json.loads((ROOT/'logs/official-test-job.json').read_text())['status']=='completed'
    plan=json.loads((out/'frozen-plan.json').read_text());rows=json.loads((out/'results.json').read_text())
    assert len(rows)==17 and all(r['plan_id']==digest(plan) and r['count']==1319 for r in rows)
    stats={}
    for role in ['final','historical_best']:
        stats[role]={}
        for v in 'UA':
            selected=[r['accuracy'] for r in rows if r['variant']==v and r['role']==role]
            assert len(selected)==3
            stats[role][v]=dict(mean=mean(selected),sample_sd=stdev(selected))
    base=next(r for r in rows if r['role']=='base')
    atomic_json(out/'statistics.json',dict(statistics=stats,base=base['accuracy'],scope='official GSM8K test 1319',
        primary='step200',secondary='original_online_dev_selection',selected_using_test=False))
    lines=['# Qwen3.5-9B 全参数GRPO：官方测试集结果','',
        '在首次读取测试预测前固定全部权重、报告角色、数据哈希和评测源码。第200步为主终点，原在线开发集选点为次终点；全部预定模型都报告，没有用测试集重新挑选检查点或调整训练。','',
        '“官方”指GSM8K官方测试划分；分数使用本项目预先固定的严格最终数值、格式和EOS判分口径，不声称使用官方榜单的全部解码与评分设置，也不直接与不同口径的排行榜数值比较。','',
        f"GSM8K官方测试集1319题。基座准确率{base['accuracy']:.2%}（{base['correct']}/1319）。开发集512题的结果单独保留，不能与此表混用。模型可能已接触公开基准，无法排除预训练接触。",'',
        '## U/A三训练种子','',
        '|报告位置|U均值±样本标准差|A均值±样本标准差|A−U（百分点）|','|---|---|---|---|']
    for role in ['final','historical_best']:
        u=stats[role]['U'];a=stats[role]['A']
        lines.append(f"|{role}|{u['mean']:.2%} ± {u['sample_sd']*100:.2f}|{a['mean']:.2%} ± {a['sample_sd']*100:.2f}|{(a['mean']-u['mean'])*100:+.2f}|")
    lines+=['','±的单位为百分点，n=3，不是题目层面的置信区间，不宣称统计显著。三个模型在同一测试集上评估，不能将1319×3当作独立题目样本量。','',
        '|种子|U第200步|A第200步|A−U（百分点）|','|---|---|---|---|']
    for seed in [42,43,44]:
        u=next(r for r in rows if r['variant']=='U' and r['seed']==seed and r['role']=='final')
        a=next(r for r in rows if r['variant']=='A' and r['seed']==seed and r['role']=='final')
        lines.append(f"|{seed}|{u['accuracy']:.2%}（{u['correct']}）|{a['accuracy']:.2%}（{a['correct']}）|{(a['accuracy']-u['accuracy'])*100:+.2f}|")
    lines+=['','## 单种子配对消融','',
        '|组别seed42|第200步|原在线选点（步数）|最终回答格式错误|最终未以EOS结束|','|---|---|---|---|---|']
    for v in 'UASE':
        f=next(r for r in rows if r['variant']==v and r['seed']==42 and r['role']=='final')
        b=next(r for r in rows if r['variant']==v and r['seed']==42 and r['role']=='historical_best')
        lines.append(f"|{v}|{f['accuracy']:.2%}（{f['correct']}/1319）|{b['accuracy']:.2%}（{b['step']}）|{f['format_error']}|{f['truncated']}|")
    lines+=['','S/E只有一个种子，与seed42的U/A配对；不代替U/A三种子均值，也不能单独证明方法普遍优劣。格式错误和非EOS结束可能重叠，不能相加当作错误题总数。','',
        '## 相对基座的逐题变化','',
        '|组别/种子|修正格式或结束错误|修正其他错误|新增格式或结束错误|新增其他错误|', '|---|---|---|---|---|']
    base_rows=[json.loads(l) for l in Path(base['output']).read_text().splitlines()]
    cases=[]
    for row in rows:
        if row['role']!='final':continue
        actual=[json.loads(l) for l in Path(row['output']).read_text().splitlines()]
        changes=dict(variant=row['variant'],seed=row['seed'],format_end_fixes=[],other_fixes=[],format_end_regressions=[],other_regressions=[])
        for b,a in zip(base_rows,actual,strict=True):
            assert b['problem_id']==a['problem_id']
            if b['score']==0 and a['score']==1:changes['format_end_fixes' if b['format_error'] or b['truncated'] else 'other_fixes'].append(b['problem_id'])
            if b['score']==1 and a['score']==0:changes['format_end_regressions' if a['format_error'] or a['truncated'] else 'other_regressions'].append(b['problem_id'])
        counts=[len(changes[k]) for k in ['format_end_fixes','other_fixes','format_end_regressions','other_regressions']]
        assert counts[0]+counts[1]-counts[2]-counts[3]==row['correct']-base['correct']
        cases.append(changes);lines.append(f"|{row['variant']}/{row['seed']}|"+'|'.join(map(str,counts))+'|')
    atomic_json(out/'failures-vs-base.json',cases)
    lines+=['','分类互斥，依据固定判分字段，不是对内部推理机制的因果解释。原先存在格式/结束错误的回答也可能同时有数值错误；不能把整类改进都算作纯格式收益。题目ID集合可在failures-vs-base.json中追溯。','',
        '## 训练末端采样分布','',
        '|seed42组别|零概率题目数|最大单题概率|熵对应有效题目数|累计单题最大访问次数|','|---|---|---|---|---|']
    sampling=json.loads((out/'sampling-endpoints.json').read_text())
    for r in sampling['results']:
        lines.append(f"|{r['variant']}|{r['zero_probability']}|{r['max_probability']:.2%}|{r['entropy_effective_questions']:.2f}|{r['max_visits']}|")
    lines+=['','这些是第200步反馈后、下一批选题前的状态快照，不是全程分布。20%均匀探索保证A每题有正概率，却不能阻止另外80%的质量集中。A的单题末端概率62.55%、有效题数26.76，提示仅提高混合组占比并不保证多样性或泛化。边界成功率、统计噪声与滞后可能使归一化分数集中，但没有完整概率轨迹，不能确定唯一因果机制。E最后两步实际涉及79道题，也不能由末端99.68%概率倒推出全程只训练一题。','',
        '## 成本与解释','',
        '原训练中A平均累计卡时30.71、U为28.77（A多6.74%）；训练回答token分别20.61M、17.47M（A多17.94%）。S42为28.97卡时/17.31M，E42为29.67卡时/19.23M。保留全部已记账的失败、短测和恢复；推理成本在每次预测对应的cost.json及official-test-job.json中独立记录，整段两卡预留卡时与单任务卡时重叠，不重复相加。','',
        '固定200步的准确率差异不等于等生成token或等卡时效率提升。部分中间权重已清理，无法重建完整的确定性测试学习曲线；本轮也不追补测试驱动的早停或检查点选择。历史选点来自原在线开发集流程，不是新参考评测流程下的全程真正最佳。','',
        '## 可复现范围','',
        '固定TP1、BF16、eager、seed42、greedy、128题分块、response≤1024、prompt≤512、thinking关闭、prefix cache关闭、同步调度、关闭引擎多进程、固定哈希种子及cuBLAS工作区，使用最后生成token的EOS判断和共享数值奖励解析器。基座在两个独立进程、两张卡各评一次，逐题token、文本、结束原因及得分完全相同，才允许后续模型评测。其余14份训练权重在测试集各评一次；它们此前均通过开发集双次逐token核验，但不声称其测试输出已逐模型重复核验或跨硬件普遍确定。','',
        'frozen-plan.json固定数据ID、哈希、评测源码及全部权重选择；results.json记录全部17个角色。15份不同权重中，共享权重的角色复用相同预测。每个独立输出保存逐题回答、token、真值、得分和终止原因，complete.json保存输出SHA256；中断可复用已完整校验的结果，不重置成本记录。']
    (out/'report.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,role in zip(axes,['final','historical_best']):
        for v,color in [('U','tab:blue'),('A','tab:orange')]:
            vals=[next(r['accuracy']*100 for r in rows if r['variant']==v and r['seed']==s and r['role']==role) for s in [42,43,44]]
            ax.plot([42,43,44],vals,'o-',label=v,color=color)
        ax.axhline(base['accuracy']*100,color='gray',ls='--',label='Base')
        ax.set_xticks([42,43,44]);ax.set_xlabel('Training seed');ax.set_ylabel('Test accuracy (%)');ax.set_title(role);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Official GSM8K test | 1319 questions | frozen checkpoint choices')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    atomic_json(out/'report-complete.json',dict(status='completed',plan_id=digest(plan)))
    print('OFFICIAL_TEST_REPORT_READY',json.dumps(stats))


if __name__=='__main__':main()
