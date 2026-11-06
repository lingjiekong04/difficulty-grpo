"""Explicitly disable vLLM 0.24 asynchronous scheduling as well as engine MP."""
from pathlib import Path
import sys,os,json,time,subprocess
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from dagrpo.locking import exclusive_lock
from dagrpo.gpu_idle import wait_gpu_idle
from dagrpo.runtime import python_runtime
from dagrpo.data import atomic_json

if __name__=='__main__':
    with exclusive_lock(ROOT/'logs/launch.lock'):
        tag=os.environ.get('DAGRPO_EVAL_TAG','sync')
        out=ROOT/f'reports/evaluation-stable-20261103-{tag}';out.mkdir(exist_ok=True)
        record=ROOT/f'logs/evaluation-stable-{tag}-job.json';started=time.time();code=1
        if record.exists():atomic_json(out/f'prior-attempt-{time.time_ns()}.json',json.loads(record.read_text()))
        atomic_json(record,dict(status='running',started=started,pid=os.getpid()))
        try:
            m=json.loads((ROOT/'data/gsm8k-qwen35-9b/manifest.json').read_text());python=python_runtime(ROOT)
            env=os.environ.copy();env.update(DAGRPO_EVAL_SYNC='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',
                VLLM_BATCH_INVARIANT='0',CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','0,1'),VLLM_WORKER_MULTIPROC_METHOD='spawn',
                TRITON_CACHE_DIR=str(ROOT/'cache/triton'),VLLM_CACHE_ROOT=str(ROOT/'cache/vllm'))
            if env.get('DAGRPO_EVAL_REFERENCE')=='1':
                env.update(PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            atomic_json(out/'controls.json',dict(async_scheduling=False,engine_multiprocessing=False,seed=42,chunk_size=128,repeats_per_process=3,fresh_processes=2))
            for i in range(2):
                session=out/f'session-{i}'
                summary=session/'summary.json'
                if summary.exists() and len(json.loads(summary.read_text()))==3 and all(
                    (session/f'repeat-{j}.jsonl').exists() and len((session/f'repeat-{j}.jsonl').read_text().splitlines())==512 for j in range(3)):
                    print('SYNC_BASE_REUSED',i,flush=True);continue
                if session.exists():session.rename(out/f'interrupted-session-{i}-{time.time_ns()}')
                wait_gpu_idle();print('SYNC_BASE_START',i,flush=True)
                subprocess.run([python,'-u','-m','dagrpo.eval_audit','--model',m['model_path'],
                    '--data',str(ROOT/'data/gsm8k-qwen35-9b/dev.jsonl'),'--out',str(out/f'session-{i}'),
                    '--seed','42','--repeats','3'],env=env,check=True)
            subprocess.run([python,'-m','dagrpo.eval_audit','--out',str(out),'--summarize'],env=env,check=True)
            x=json.loads((out/'comparison.json').read_text());assert len(x['sessions'])==6
            assert all(not ids for d in x['comparisons'].values() for ids in d.values())
            code=0
        finally:
            atomic_json(record,dict(status='completed' if code==0 else 'failed',exit_code=code,started=started,
                finished=time.time(),gpu_hours=2*(time.time()-started)/3600,pid=os.getpid()))
        raise SystemExit(code)
