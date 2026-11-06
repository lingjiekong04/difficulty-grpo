"""No training: base-model dev audit, same-process repeats then cold restarts."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from dagrpo.jobs import state
from dagrpo.runtime import python_runtime
from dagrpo.locking import exclusive_lock
from dagrpo.data import atomic_json
from dagrpo.runner import verify_data

if __name__=='__main__':
    out=ROOT/'reports/evaluation-audit-20261103'
    with exclusive_lock(ROOT/'logs/launch.lock'):
        assert all(state(m)['status']!='running' for m in ['train','matrix','pilot','prepare'])
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(ROOT/'logs/evaluation-audit-job.json',dict(status='running',pid=os.getpid(),started=time.time()))
        code=1
        try:
            manifest=verify_data(ROOT/'data/gsm8k-qwen35-9b')
            model=Path(manifest['model_path']);hashes={}
            for file in sorted(model.glob('*.safetensors')):
                h=hashlib.sha256()
                with file.open('rb') as f:
                    for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
                hashes[file.name]=h.hexdigest();print('MODEL_HASHED',file.name,flush=True)
            atomic_json(out/'model-manifest.json',dict(revision=manifest['model_revision'],sha256=hashes))
            python=python_runtime(ROOT)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
                HF_HOME=str(ROOT/'cache/huggingface'),TRITON_CACHE_DIR=str(ROOT/'cache/triton'),VLLM_CACHE_ROOT=str(ROOT/'cache/vllm'))
            for session,(seed,reps) in enumerate([(42,3),(42,1),(43,1),(44,1)]):
                print(f'AUDIT_SESSION_START session={session} seed={seed} repeats={reps}',flush=True)
                subprocess.run([python,'-u','-m','dagrpo.eval_audit','--model',str(model),'--data',str(ROOT/'data/gsm8k-qwen35-9b/dev.jsonl'),
                    '--out',str(out/f'session-{session}'),'--seed',str(seed),'--repeats',str(reps)],env=env,check=True)
            subprocess.run([python,'-m','dagrpo.eval_audit','--out',str(out),'--summarize'],env=env,check=True)
            code=0
        finally:
            atomic_json(ROOT/'logs/evaluation-audit-job.json',dict(status='completed' if code==0 else 'failed',pid=os.getpid(),exit_code=code,finished=time.time()))
        raise SystemExit(code)
