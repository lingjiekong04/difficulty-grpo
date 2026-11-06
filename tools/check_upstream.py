"""Verify an explicitly cloned pinned verl checkout, not the network."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
    lock=json.loads((ROOT/'dependencies.lock.json').read_text(encoding='utf-8'))['verl']
    vendor=ROOT/'vendor/verl'
    commit=subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()
    assert commit==lock['commit'],commit
    assert not subprocess.check_output(['git','-C',str(vendor),'status','--porcelain','--untracked-files=no'],text=True).strip()
    for n,h in lock['audited_source_sha256'].items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
    print('Pinned upstream and audited source subset match. No upstream patches required.')
if __name__=='__main__':main()
