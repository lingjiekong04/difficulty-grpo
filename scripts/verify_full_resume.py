"""Verify the actual next sampled batch after exactly one resumed update."""
import argparse
import json
from collections import Counter
from pathlib import Path
import torch
from dagrpo.sampler import DifficultySampler

parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--before',required=True)
parser.add_argument('--after',required=True)
parser.add_argument('--out',required=True)
args=parser.parse_args()
before=torch.load(Path(args.before)/'study_state.pt',map_location='cpu',weights_only=False)['sampler']
after=torch.load(Path(args.after)/'study_state.pt',map_location='cpu',weights_only=False)['sampler']
sampler=DifficultySampler(before['ids'],before['variant'],before['seed'],before['batch_size'],before['steps'],before['warmup'])
sampler.load_state_dict(before)
iterator=iter(sampler)
next_ids=[sampler.ids[next(iterator)] for _ in range(sampler.batch_size)]
expected_visits=before['visits'].copy()
for pid,count in Counter(next_ids).items(): expected_visits[pid]+=count
checks=dict(position=after['position']==before['position']+sampler.batch_size,
    completed=after['completed']==before['completed']+1,
    next_batch_visits=after['visits']==expected_visits,
    sampler_rng=after['rng']==sampler.state_dict()['rng'])
result=dict(before=args.before,after=args.after,checks=checks,
    position_before=before['position'],position_after=after['position'],batch_size=sampler.batch_size)
from dagrpo.data import atomic_json
atomic_json(args.out,result)
print(json.dumps(result,indent=2))
if not all(checks.values()): raise SystemExit('Actual sampler continuation failed')
