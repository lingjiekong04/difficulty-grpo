"""Finish development-only review after the active reference audit succeeds."""
from pathlib import Path
import json, os, subprocess, sys, time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from dagrpo.data import atomic_json
from dagrpo.runtime import python_runtime


if __name__=='__main__':
    record=ROOT/'logs/evaluation-review-pipeline-job.json'
    source=ROOT/'logs/evaluation-stable-reference-job.json'
    started=time.time();code=1
    if record.exists():
        atomic_json(ROOT/f'logs/evaluation-review-pipeline-prior-{time.time_ns()}.json',json.loads(record.read_text()))
    atomic_json(record,dict(status='waiting_for_reference',pid=os.getpid(),started=started))
    try:
        while True:
            audit=json.loads(source.read_text())
            if audit['status']=='completed':break
            if audit['status']!='running':raise RuntimeError('Reference audit failed; no checkpoint review started')
            os.kill(audit['pid'],0)
            time.sleep(5)
        env={**os.environ,'DAGRPO_EVAL_SYNC':'1','DAGRPO_EVAL_REFERENCE':'1',
             'DAGRPO_EVAL_TAG':'reference','DAGRPO_EVAL_PARALLEL':'1',
             'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES','0,1'),'PYTHONHASHSEED':'0',
             'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        python=python_runtime(ROOT)
        atomic_json(record,dict(status='reevaluating_saved_models',pid=os.getpid(),started=started))
        subprocess.run([python,'-u','scripts/recheck_saved_models.py'],env=env,check=True)
        proof=ROOT/'reports/evaluation-review-20261103-reference'
        proof.mkdir(exist_ok=True)
        with (proof/'tests.log').open('w') as log:
            result=subprocess.run([python,'-m','unittest','discover','-s','tests','-v'],env=env,
                                  stdout=log,stderr=subprocess.STDOUT)
        atomic_json(proof/'validation.json',dict(command='python -m unittest discover -s tests -v',
                    exit_code=result.returncode,log=str(proof/'tests.log')))
        result.check_returncode()
        subprocess.run([python,'-u','scripts/report_evaluation_review.py'],env=env,check=True)
        code=0
    finally:
        atomic_json(record,dict(status='completed' if code==0 else 'failed',pid=os.getpid(),
                    started=started,finished=time.time(),exit_code=code,
                    note='Inference and tests only; envelope timing overlaps child jobs; do not add it to their cost'))
    raise SystemExit(code)
