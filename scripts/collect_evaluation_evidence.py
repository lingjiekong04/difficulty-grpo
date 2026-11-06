"""Archive development review evidence and code, excluding all model weights."""
from pathlib import Path
import hashlib,json,sys,tarfile
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from dagrpo.data import atomic_json


def main():
    assert json.loads((ROOT/'logs/evaluation-review-pipeline-job.json').read_text())['status']=='completed'
    review=ROOT/'reports/evaluation-review-20261103-reference'
    costs=[];seen=set()
    candidates=list((ROOT/'logs').glob('*eval*.json'))
    candidates += list((ROOT/'reports').glob('evaluation-*/prior*.json'))
    for path in sorted(candidates):
        try:record=json.loads(path.read_text())
        except (ValueError,OSError):continue
        if not isinstance(record,dict) or 'status' not in record:continue
        key=(record.get('pid'),record.get('started'),record.get('finished'),record.get('status'))
        if key in seen:continue
        seen.add(key)
        costs.append(dict(path=str(path.relative_to(ROOT)),record=record,
            accounting='Envelope only; overlaps children, do not sum' if 'pipeline' in path.name else
                       'Recorded two-card wall-time upper bound including CPU export and idle waiting; not kernel-active GPU time'))
    atomic_json(review/'cost-records.json',dict(records=costs,
        note='Failures and interrupted attempts retained. Missing start/finish timestamps imply unknown cost, not zero. Do not sum envelope and child jobs. Training ledger is preserved separately.'))
    selected=set()
    for parent in (ROOT/'reports').iterdir():
        if parent.is_dir() and ('evaluation' in parent.name or 'saved-model-dev-recheck' in parent.name):
            selected.update(p for p in parent.rglob('*') if p.is_file() and not p.is_symlink())
    selected.update(p for p in (ROOT/'logs').glob('*eval*') if p.is_file())
    for directory in ['dagrpo','scripts','tests','compat']:
        selected.update((ROOT/directory).glob('*.py'))
    selected.update((ROOT/'scripts').glob('*.sh'))
    selected.update(p for p in (ROOT/'configs').rglob('*') if p.is_file())
    selected.update(ROOT.glob('run*.py'))
    selected.update(ROOT.glob('environment*.txt'))
    selected.update(ROOT.glob('environment*.json'))
    selected.update(p for p in [ROOT/'README.md',ROOT/'docs/EVALUATION_PROTOCOL.md',
        ROOT/'reports/UA-three-seeds-20261103/results.json',
        ROOT/'pyproject.toml',ROOT/'QUICKSTART.md',
        ROOT/'validation/evaluation-progress-20261103.json',
        ROOT/'runs/full-budget.json',ROOT/'runs/pilot-budget.json',
        ROOT/'data/gsm8k-qwen35-9b/manifest.json',ROOT/'data/gsm8k-qwen35-9b/dev.jsonl'] if p.is_file())
    files=sorted(p for p in selected if p!=review/'archive-manifest.json')
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
    target=review/'archive-manifest.json';atomic_json(target,manifest);files.append(target)
    archive=ROOT/'evaluation-review-evidence-20261103.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,
        files=len(files),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))


if __name__=='__main__':main()
