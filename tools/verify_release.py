"""Check a release manifest without importing any training dependencies."""
from pathlib import Path
import json,hashlib,os,sys
ROOT=Path(__file__).resolve().parents[1]
RUNTIME={'runs','exports','logs','reports','cache','data','vendor','python','bootstrap','wheelhouse','tmp','.venv'}
def verify():
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    files={}
    for parent,dirs,names in os.walk(ROOT,followlinks=False):
        base=Path(parent)
        dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','.pytest_cache') and not (base==ROOT and d in RUNTIME)]
        for n in names:
            p=base/n;rel=p.relative_to(ROOT).as_posix()
            if p.is_symlink():raise ValueError('Unapproved symlink: '+rel)
            if n.endswith(('.pyc','.pyo')):continue
            if rel!='RELEASE_MANIFEST.json':files[rel]=p
    assert set(files)==set(manifest['files']),dict(extra=sorted(set(files)-set(manifest['files'])),missing=sorted(set(manifest['files'])-set(files)))
    for rel,p in files.items():
        b=p.read_bytes();want=manifest['files'][rel]
        assert len(b)==want['bytes'] and hashlib.sha256(b).hexdigest()==want['sha256'],rel
    split=json.loads((ROOT/'evidence/data-manifest.json').read_text(encoding='utf-8'))['splits']
    ids=[i for s in split.values() for i in s['ids']]
    assert len(ids)==len(set(ids))==3879
    from summarize_results import summarize
    summarize()
    print(f'PASS: {len(files)} manifest files; hashes, split IDs and 17 test roles verified. No GPU/model invoked.')
if __name__=='__main__':verify()
