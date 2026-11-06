"""Freeze checkpoint choices before reading any official test predictions."""
import hashlib,json
from pathlib import Path
from .data import atomic_json,digest


def validate_splits(splits,expected_test=1319):
    if len(splits['test'])!=expected_test:raise ValueError('Unexpected official test count')
    ids=set();questions=set()
    for split in ['train','dev','test']:
        for row in splits[split]:
            question=' '.join(row['question'].split())
            if row['problem_id'] in ids or question in questions:
                raise ValueError('Duplicate ID or question across/within project splits')
            ids.add(row['problem_id']);questions.add(question)


def freeze(path,plan):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=plan:raise ValueError('Frozen test protocol changed')
    else:atomic_json(path,plan)
    return digest(plan)


def build(root):
    root=Path(root)
    from .runner import verify_data
    from .stages import run_complete
    manifest=verify_data(root/'data/gsm8k-qwen35-9b')
    splits={s:[json.loads(l) for l in (root/f'data/gsm8k-qwen35-9b/{s}.jsonl').read_text().splitlines()] for s in ['train','dev','test']}
    validate_splits(splits)
    selected=[]
    for name in ['saved-model-dev-recheck-20261103-reference','saved-model-dev-recheck-20261105-reference-SE']:
        rows=json.loads((root/f'reports/{name}/results.json').read_text())
        for r in rows:
            assert r['repeated_tokens_identical'] and run_complete(r['run'],r['variant'],r['seed'])
            assert Path(r['export'],'export-complete.json').is_file()
            selected.append({k:r[k] for k in ['variant','seed','step','role','export','checkpoint']})
    assert len(selected)==16 and len({(r['variant'],r['seed'],r['step']) for r in selected})==14
    selected.insert(0,dict(variant='BASE',seed=42,step=0,role='base',export=manifest['model_path'],checkpoint=manifest['model_revision']))
    sources={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['dagrpo/evaluate.py','dagrpo/cli.py','dagrpo/reward.py','dagrpo/data.py']}
    return dict(schema=1,roles=selected,test_sha256=digest(splits['test']),test_ids=[r['problem_id'] for r in splits['test']],
        source_sha256=sources,protocol='offline-tp1-bf16-greedy1024-chunk128-strict-eos-reference-v4',
        primary='fixed_step_200',secondary='original_online_dev_selection',selection_uses_test=False,
        baseline_repetitions=2,other_model_repetitions=1,all_selected_models_reported=True)
