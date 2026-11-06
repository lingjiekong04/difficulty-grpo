"""Observed development comparisons; never interpolate or peek beyond a budget."""
import argparse
import json
from pathlib import Path
from statistics import mean
from .data import atomic_json
from .report import protocol_id


def budget_snapshot(rows, key, budget):
    eligible = [r for r in rows if r[key] <= budget]
    if not eligible:
        return None
    row = max(eligible, key=lambda r: (r[key], r['step']))
    return dict(step=row['step'], accuracy=row['accuracy'], observed_cost=row[key],
                unused_budget=budget-row[key])


def read_run(path):
    path = Path(path)
    metadata = json.loads((path/'run.json').read_text(encoding='utf-8'))
    evaluations = sorted({r['step']: r for r in map(json.loads,
        (path/'evaluations.jsonl').read_text().splitlines())}.values(), key=lambda r:r['step'])
    steps = sorted({r['step']: r for r in map(json.loads,
        (path/'steps.jsonl').read_text().splitlines())
        if 'training/global_step' in r['metrics']}.values(), key=lambda r:r['step'])
    return metadata, evaluations, steps


def analyze(uniform, adaptive, out):
    runs = {v:read_run(p) for v,p in [('U',uniform),('A',adaptive)]}
    if protocol_id(runs['U'][0]) != protocol_id(runs['A'][0]):
        raise ValueError('Different model/data/environment/training protocols')
    if runs['U'][0]['study']['seed'] != runs['A'][0]['study']['seed']:
        raise ValueError('Compare the same seed')
    metrics = ['study/mixed_group_rate','critic/score/mean','actor/kl_loss',
               'actor/entropy','actor/grad_norm','response_length/mean',
               'response_length/clip_ratio','timing_s/step']
    diagnostics = {}
    for v, (_, _, steps) in runs.items():
        diagnostics[v] = {}
        for lo,hi in [(1,20),(21,75),(76,125),(126,200)]:
            rows = [r for r in steps if lo <= r['step'] <= hi]
            if rows:
                diagnostics[v][f'{lo}-{hi}'] = {k:mean(r['metrics'][k] for r in rows
                    if k in r['metrics']) for k in metrics}
    comparisons = {}
    for key, fixed in [('gpu_hours',[10,20,28]), ('tokens',[5e6,10e6,15e6])]:
        common = min(runs[v][1][-1][key] for v in runs)
        budgets = sorted(set([x for x in fixed if x<=common]+[common]))
        comparisons[key] = [dict(budget=b, **{v:budget_snapshot(runs[v][1],key,b)
                                             for v in runs}) for b in budgets]
    result = dict(protocol=protocol_id(runs['U'][0]),seed=runs['U'][0]['study']['seed'],
        budget_method='Latest observed development evaluation within budget, not best-so-far; no interpolation.',
        cost_limit='tokens counts training responses only; evaluation tokens were not recorded. gpu_hours includes observed process runtime.',
        comparisons=comparisons, diagnostics=diagnostics,
        evaluations={v:e for v,(_,e,_) in runs.items()})
    out = Path(out);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'budget-analysis.json',result)
    lines = ['# U/A seed-42 同预算与训练动态分析','',
        '所有准确率来自 512 道开发题。以下在预算内选择最近一次已观察评估，不选择历史最高分，不进行插值；展示实际观察成本，不能当成恰好用完同一预算的测量。', '',
        'token 仅统计训练回答；开发集生成 token 未记录，不能据此宣称全流程 token 效率。卡时包含已记录的初始化、评估和保存开销。','']
    for key,rows in comparisons.items():
        lines += [f'## {key}', '', '|预算|U 步数 / 准确率 / 实际成本|A 步数 / 准确率 / 实际成本|', '|---|---|---|']
        for row in rows:
            cells = [f"{row['budget']:,.3f}"]
            for v in ['U','A']:
                x=row[v]
                cells.append('无观察' if x is None else f"{x['step']} / {x['accuracy']*100:.2f}% / {x['observed_cost']:,.3f}")
            lines.append('| '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['## 训练动态', '', '|组别|步数窗口|混合组|奖励均值|KL loss|熵|梯度范数|回答长度|长度截断|秒/步|', '|---|---|---|---|---|---|---|---|---|---|']
    for v,windows in diagnostics.items():
        for window,m in windows.items():
            lines.append(f"|{v}|{window}|{m[metrics[0]]:.2%}|{m[metrics[1]]:.4f}|{m[metrics[2]]:.4f}|{m[metrics[3]]:.4f}|{m[metrics[4]]:.4f}|{m[metrics[5]]:.1f}|{m[metrics[6]]:.2%}|{m[metrics[7]]:.1f}|")
    lines += ['', '## 对 U 后期回落的判断', '',
        'U 的第 75 步与第 200 步相差 8 道开发题，两组最佳分相同。训练奖励、KL、长度等是诊断线索，单种子下不能据此认定过拟合或确定回落原因。', '',
        '|组别 / 步数|开发集准确率|格式错误率|截断率|', '|---|---|---|---|']
    for v,(_,evaluations,_) in runs.items():
        for r in evaluations:
            if r['step'] not in (0,75,175,200):continue
            m=r.get('metrics',{})
            values=[]
            for term in ['format_error','truncated']:
                found=[val for k,val in m.items() if f'/{term}/' in k]
                values.append(f'{found[0]:.2%}' if found else '未记录')
            lines.append(f"|{v} / {r['step']}|{r['accuracy']:.2%}|{values[0]}|{values[1]}|")
    lines += ['', '下一步保持协议不变，补 U/A 的 seed 43、44，报告每个种子与跨种子波动。暂不调整训练长度、学习率或数据，不使用官方测试集调参。']
    (out/'budget-analysis.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--uniform',required=True)
    p.add_argument('--adaptive',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();analyze(args.uniform,args.adaptive,args.out)
