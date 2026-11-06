"""User-run GPU-environment data preparation at the recorded revisions.

Uses the inherited split/export implementation, never HfApi's current HEAD.
Requires model download and tokenizer loading; NOT run during release audit.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',required=True);args=p.parse_args()
    dest=Path(args.out)
    if dest.exists() and any(dest.iterdir()):raise ValueError('Output must be absent or empty; no overwrite')
    lock=json.loads((ROOT/'dependencies.lock.json').read_text(encoding='utf-8'))
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from dagrpo.data import prepare_records,export
    source=load_dataset(lock['dataset']['repo'],lock['dataset']['config'],revision=lock['dataset']['revision'])
    splits,manifest=prepare_records(list(source['train']),list(source['test']))
    expected=json.loads((ROOT/'evidence/data-manifest.json').read_text(encoding='utf-8'))
    assert manifest['splits']==expected['splits'],'Split content differs; preserve evidence, do not refresh old hashes'
    model=snapshot_download(lock['model']['repo'],revision=lock['model']['revision'])
    manifest.update(dataset_revision=lock['dataset']['revision'],model=lock['model']['repo'],model_revision=lock['model']['revision'],model_path=str(Path(model).resolve()),chat_template_kwargs={'enable_thinking':False})
    export(dest,splits,manifest,AutoTokenizer.from_pretrained(model))
    observed=json.loads((dest/'manifest.json').read_text(encoding='utf-8'))['files']
    diff={n:dict(expected=h,actual=observed.get(n)) for n,h in expected['files'].items() if observed.get(n)!=h}
    print(json.dumps(dict(status='prepared',byte_hash_differences=diff,split_content_matches=True),indent=2))
    if diff:raise SystemExit('Prepared byte hashes differ; retain output and investigate dependency/encoding differences.')
if __name__=='__main__':main()
