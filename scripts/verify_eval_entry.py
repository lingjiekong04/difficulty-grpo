"""Real-GPU check that the public evaluate CLI matches the audited protocol."""
from pathlib import Path
import sys,os,json,subprocess,time
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from dagrpo.data import atomic_json
from dagrpo.locking import exclusive_lock
from dagrpo.runtime import python_runtime
from dagrpo.gpu_idle import wait_gpu_idle

if __name__=='__main__':
    state=ROOT/'logs/evaluation-entry-job.json'
    atomic_json(state,dict(status='waiting_for_recheck',pid=os.getpid()))
    with exclusive_lock(ROOT/'logs/launch.lock',blocking=True):
        assert json.loads((ROOT/'logs/saved-model-eval-job.json').read_text())['status']=='completed'
        marker=json.loads((ROOT/'reports/saved-model-dev-recheck-20261103/U-42-step200/complete.json').read_text())
        started=time.time();code=1
        atomic_json(state,dict(status='running',pid=os.getpid(),started=started))
        try:
            wait_gpu_idle();python=python_runtime(ROOT)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
                VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_BATCH_INVARIANT='0',
                TRITON_CACHE_DIR=str(ROOT/'cache/triton'),VLLM_CACHE_ROOT=str(ROOT/'cache/vllm'))
            out=ROOT/'reports/evaluation-entry-validation-20261103.jsonl'
            if out.exists():
                out.rename(out.with_name(out.stem+f'.prior-{time.time_ns()}.jsonl'))
                summary=out.with_suffix('.summary.json')
                if summary.exists():summary.rename(summary.with_name(summary.stem+f'.prior-{time.time_ns()}.json'))
            subprocess.run([python,'-m','dagrpo.cli','evaluate','--model',marker['export'],
                '--data',str(ROOT/'data/gsm8k-qwen35-9b/dev.jsonl'),'--out',str(out)],env=env,check=True)
            a=[json.loads(l) for l in (Path(marker['attempt'])/'repeat-0.jsonl').read_text().splitlines()]
            b=[json.loads(l) for l in out.read_text().splitlines()]
            assert len(a)==len(b)==512
            fields=['problem_id','token_ids','text','score','finish_reason','stop_reason','truncated','format_error']
            differences={k:[r['problem_id'] for r,s in zip(a,b) if r[k]!=s[k]] for k in fields}
            assert all(not ids for ids in differences.values()),differences
            atomic_json(ROOT/'validation/evaluation-v2-entry.json',dict(passed=True,count=512,differences=differences,
                reference=marker['attempt'],public_cli_output=str(out)))
            code=0
        finally:
            atomic_json(state,dict(status='completed' if code==0 else 'failed',exit_code=code,pid=os.getpid(),
                started=started,finished=time.time(),gpu_hours=2*(time.time()-started)/3600))
        raise SystemExit(code)
