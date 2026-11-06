"""Test documented offline deterministic scheduling, after the baseline audit."""
from pathlib import Path
import sys,os,json,subprocess,time,hashlib
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from dagrpo.data import atomic_json
from dagrpo.jobs import state
from dagrpo.runtime import python_runtime
from dagrpo.locking import exclusive_lock
from dagrpo.gpu_idle import wait_gpu_idle

if __name__=='__main__':
    record=ROOT/'logs/evaluation-stable-job.json'
    atomic_json(record,dict(status='waiting_for_baseline_audit',pid=os.getpid()))
    with exclusive_lock(ROOT/'logs/launch.lock',blocking=True):
        assert all(state(m)['status']!='running' for m in ['train','matrix','pilot','prepare'])
        out=ROOT/'reports/evaluation-stable-20261103';out.mkdir(parents=True,exist_ok=True)
        if record.exists():
            atomic_json(out/f'prior-job-{time.time_ns()}.json',json.loads(record.read_text()))
        started=time.time();atomic_json(record,dict(status='running',pid=os.getpid(),started=started))
        code=1
        try:
            m=json.loads((ROOT/'data/gsm8k-qwen35-9b/manifest.json').read_text())
            python=python_runtime(ROOT)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',
                VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_BATCH_INVARIANT='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',
                HF_HOME=str(ROOT/'cache/huggingface'),TRITON_CACHE_DIR=str(ROOT/'cache/triton'),VLLM_CACHE_ROOT=str(ROOT/'cache/vllm'))
            atomic_json(out/'environment-controls.json',dict(VLLM_ENABLE_V1_MULTIPROCESSING='0',
                VLLM_BATCH_INVARIANT='0',seed=42,two_fresh_processes=True,repeats_each=3,
                worker_sha256=hashlib.sha256((ROOT/'dagrpo/eval_audit.py').read_bytes()).hexdigest(),
                source='https://docs.vllm.ai/en/latest/usage/reproducibility/',
                note='A100 has compute capability 8.0; installed batch-invariant kernels require >=9.0. Test scheduling control instead.'))
            for i in range(2):
                session=out/f'session-{i}'
                summary=session/'summary.json'
                if summary.exists() and len(json.loads(summary.read_text()))==3:
                    print('STABLE_SESSION_REUSED',i,flush=True);continue
                if session.exists():session.rename(out/f'failed-session-{i}-{time.time_ns()}')
                wait_gpu_idle()
                print('STABLE_SESSION_START',i,flush=True)
                subprocess.run([python,'-u','-m','dagrpo.eval_audit','--model',m['model_path'],
                    '--data',str(ROOT/'data/gsm8k-qwen35-9b/dev.jsonl'),'--out',str(out/f'session-{i}'),
                    '--seed','42','--repeats','3'],env=env,check=True)
            subprocess.run([python,'-m','dagrpo.eval_audit','--out',str(out),'--summarize'],env=env,check=True)
            code=0
        finally:
            atomic_json(record,dict(status='completed' if code==0 else 'failed',pid=os.getpid(),exit_code=code,
                started=started,finished=time.time(),gpu_hours=2*(time.time()-started)/3600))
        raise SystemExit(code)
