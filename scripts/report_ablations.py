"""Summarize completed S/E training using the same reference DEV evaluator as U/A."""
from pathlib import Path
import json,sys,os
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from dagrpo.data import atomic_json


def main():
    out=ROOT/'reports/ablations-20261105'
    assert all(not ids for ids in json.loads((out/'cross-host-comparison.json').read_text()).values()),'Current-host U42 reference must match archived outputs'
    assert json.loads((ROOT/'logs/saved-model-eval-reference-SE-job.json').read_text())['status']=='completed'
    se=ROOT/'reports/saved-model-dev-recheck-20261105-reference-SE'
    old=ROOT/'reports/saved-model-dev-recheck-20261103-reference'
    rows=json.loads((se/'results.json').read_text());assert len(rows)==4
    assert all(r['repeated_tokens_identical'] for r in rows)
    assert all(not x for x in json.loads((se/'S-42-step200/public-entry-comparison.json').read_text()).values())
    rows += [r for r in json.loads((old/'results.json').read_text()) if r['seed']==42]
    training=json.loads((out/'training-audit.json').read_text())
    atomic_json(out/'results.json',dict(evaluations=rows,training=training,scope='DEV 512 questions, seed 42 only; official test unused'))
    lines=['# S/E 消融核验结果','',
        'S/E 各完整完成200步，退出码0，更新无缺失、无非有限指标；两卡全参数审计及模型、优化器、RNG检查点存在。协议与U/A相同。未启动新训练、未使用官方测试集。','',
        '下表全部使用同一参考流程：TP1、BF16、eager、固定哈希与cuBLAS工作区、关闭异步调度及引擎多进程、固定128题分块和EOS判分。S/E共3份不同权重，每份两次逐token一致；首个S检查点的正式CLI冷启动对照一致。','',
        '|组别（seed42）|统一复评第200步|原在线选点复评（步数）|累计训练卡时|训练回答token（百万）|混合组比例|','|---|---|---|---|---|---|']
    for v in 'UASE':
        f=next(r for r in rows if r['variant']==v and r['role']=='final')
        b=next(r for r in rows if r['variant']==v and r['role']=='historical_best')
        t=next(r for r in training if r['variant']==v)
        lines.append(f"|{v}|{f['accuracy']:.2%}（{f['correct']}/512）|{b['accuracy']:.2%}（{b['step']}）|{t['gpu_hours']:.2f}|{t['training_tokens']/1e6:.2f}|{t['mixed_mean']:.2%}|")
    lines+=['','这是单个训练种子的消融，与seed42的U/A配对；不能用其代替三种子均值或宣称显著性。U/A三种子的已核验结论单独保留。历史选点基于原在线开发集，不是本参考流程下的全程真正最佳；中间权重已被原策略清理，不能重建完整确定性曲线。卡时包含该运行账本记录的全部尝试，含失败、短测和恢复；复评开销单独记录。','',
        '## 采样状态核验','',
        'S在第75、200步的全部成功率估计完全一致，同时访问计数增加，证实冻结统计生效；本次S全程混合组比例低于均匀对照。当前实例的U42正式入口输出也与先前归档逐token一致，核对了迁移实例后的参考流程。本实验不能仅凭混合组比例推断能力提升。','']
    e=next(r for r in training if r['variant']=='E');s=e['sampler'];last=e['last_two_updates']
    lines += [f"E累计覆盖率为{e['coverage']:.2%}，但最终状态中{ s['zero_probability'] }/2048题采样概率为零，最大单题概率{s['max_probability']:.2%}，熵对应的有效题目数为{s['effective_questions']:.2f}，累计单题最大访问次数为{s['max_visits']}。这显示末端选题分布高度集中，累计覆盖率掩盖了后续探索不足。",'',
        f"限制：最终分布是第200步反馈后、下一批选题前的状态快照，不是整段训练分布。第199–200步的128次题目出现实际涉及{last['distinct_questions']}道不同题，单题最多{last['largest_question_count']}次。因此不能说整段训练或最后两步只训练了一道题；全零分数时的均匀回退也可能改变局部分布。没有逐步保存全部概率快照，不能量化集中状态持续了多久。",'',
        'E的开发集结果与上述分布现象应分别解释：没有均匀探索会产生零概率题目，但单种子、饱和的公开基准结果不能证明去掉探索更优或更差。公开基座可能已接触GSM8K，项目内划分不能排除预训练接触。','',
        '## 后续','',
        '训练矩阵U/A三种子与S/E单种子均已完成。接下来先固定最终模型及原在线选点两套报告规则，再在1319题官方测试集执行同一参考评测；只报告观测结果，测试结果不用于采样或调参。随后完成失败案例、复现文档、简历与面试材料。']
    (out/'report.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for k,role in enumerate(['final','historical_best']):
        values=[next(r['accuracy']*100 for r in rows if r['variant']==v and r['role']==role) for v in 'UASE']
        ax.plot(list('UASE'),values,'o-',label='Step 200' if k==0 else 'Original online selection')
    ax.set_ylabel('Reference dev accuracy (%)');ax.set_title('Ablations | seed 42 | 512 dev questions');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    print('ABLATIONS_REFERENCE_REPORT_READY')


if __name__=='__main__':main()
