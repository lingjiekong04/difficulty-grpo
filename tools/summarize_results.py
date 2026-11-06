"""Recompute public test statistics from scalar scores; never loads a model."""
from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[1]
def summarize():
    data=json.loads((ROOT/'evidence/results.json').read_text(encoding='utf-8'))
    for r in data['results']:
        p=ROOT/f"evidence/scores/{r['variant']}-{r['seed']}-step{r['step']}.jsonl"
        rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==r['count']==1319
        assert sum(x['score'] for x in rows)==r['correct']
        assert sum(x['tokens'] for x in rows)==r['generated_tokens']
        assert sum(x['format_error'] for x in rows)==r['format_error']
        assert sum(x['truncated'] for x in rows)==r['truncated']
    stats={}
    for role in ('final','historical_best'):
        stats[role]={}
        for v in ('U','A'):
            rows=[r for r in data['results'] if r['role']==role and r['variant']==v]
            assert {r['seed'] for r in rows}=={42,43,44}
            values=[r['correct']/r['count'] for r in rows]
            result=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values))
            assert result==data['statistics'][role][v]
            stats[role][v]=result
    return stats
if __name__=='__main__':
    print(json.dumps(dict(scope='Official GSM8K test split; project strict numeric/format/EOS scoring; 1319 items; sample SD over n=3 seeds',statistics=summarize()),indent=2))
