"""Frozen, resumable final GSM8K test. No training and no test-based selection."""
from pathlib import Path
import os,sys,json,time,subprocess,hashlib
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from dagrpo.data import atomic_json
from dagrpo.test_plan import build,freeze
from dagrpo.eval_queue import evaluate_plan
from dagrpo.gpu_idle import wait_gpu_idle
from dagrpo.locking import exclusive_lock
from dagrpo.runtime import python_runtime
from dagrpo.jobs import state
from dagrpo.reward import score


def main():
    out=ROOT/'reports/official-test-20261105';out.mkdir(exist_ok=True)
    record=ROOT/'logs/official-test-job.json';started=time.time();success=False
    with exclusive_lock(ROOT/'logs/launch.lock'):
        assert all(state(m)['status']!='running' for m in ['train','matrix','pilot','prepare'])
        plan=build(ROOT);plan_id=freeze(out/'frozen-plan.json',plan)
        if record.exists():atomic_json(out/f'controller-attempt-{time.time_ns()}.json',json.loads(record.read_text()))
        atomic_json(record,dict(status='running',pid=os.getpid(),started=started,plan_id=plan_id))
        python=python_runtime(ROOT)
        devices=['0','1']
        env={**os.environ,'PYTHONHASHSEED':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8',
             'VLLM_ENABLE_V1_MULTIPROCESSING':'0','VLLM_BATCH_INVARIANT':'0','PYTHONUNBUFFERED':'1'}
        def check(path):
            rows=[json.loads(l) for l in path.read_text().splitlines()]
            assert [r['problem_id'] for r in rows]==plan['test_ids']
            assert all(score(r['text'],r['truth'],not r['ended_with_eos'])['score']==r['score'] for r in rows)
            summary=json.loads(path.with_suffix('.summary.json').read_text())
            assert summary['data_sha256']==plan['test_sha256'] and summary['protocol']==plan['protocol']
            return rows,summary
        def execute(item,device,path):
            wait_gpu_idle(devices=device)
            t=time.time();print('OFFICIAL_TEST_START',item['variant'],item['seed'],item['step'],'GPU',device,flush=True)
            with path.with_suffix('.log').open('w') as log:
                subprocess.run([python,'-u','-m','dagrpo.cli','evaluate','--model',item['export'],
                    '--data',str(ROOT/'data/gsm8k-qwen35-9b/test.jsonl'),'--out',str(path)],
                    env={**env,'CUDA_VISIBLE_DEVICES':device},stdout=log,stderr=subprocess.STDOUT,check=True)
            rows,summary=check(path)
            atomic_json(path.with_suffix('.cost.json'),dict(seconds=time.time()-t,gpu_count=1,generated_tokens=summary['generated_tokens']))
            return rows,summary
        def model(item,device=None,verify_export=False,public_gate=False):
            key=f"{item['variant']}-{item['seed']}-step{item['step']}";cache=out/key;cache.mkdir(exist_ok=True)
            marker=cache/'complete.json'
            if marker.exists():
                saved=json.loads(marker.read_text());assert saved['plan_id']==plan_id
                assert hashlib.sha256(Path(saved['output']).read_bytes()).hexdigest()==saved['output_sha256']
                check(Path(saved['output']));return saved
            attempt=cache/f'attempt-{time.time_ns()}';attempt.mkdir()
            path=attempt/'predictions.jsonl'
            if public_gate:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    a=pool.submit(execute,item,'0',path)
                    b=pool.submit(execute,item,'1',attempt/'baseline-cold-repeat.jsonl')
                    rows,summary=a.result();other,_=b.result()
                fields=['problem_id','token_ids','text','finish_reason','stop_reason','score','truncated','format_error']
                differences={k:[r['problem_id'] for r,s in zip(rows,other,strict=True) if r[k]!=s[k]] for k in fields}
                atomic_json(attempt/'baseline-cold-comparison.json',differences)
                assert all(not ids for ids in differences.values()),'Test baseline cold reproducibility failed; other models withheld'
            else:rows,summary=execute(item,device,path)
            saved=dict(accuracy=summary['accuracy'],correct=int(sum(r['score'] for r in rows)),count=len(rows),
                output=str(path),output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),plan_id=plan_id,
                generated_tokens=summary['generated_tokens'],format_error=sum(r['format_error'] for r in rows),truncated=sum(r['truncated'] for r in rows))
            atomic_json(marker,saved);print('OFFICIAL_TEST_COMPLETE',key,saved['correct'],'/1319',flush=True)
            return saved
        try:
            print('FROZEN_TEST_PLAN',plan_id,'17 roles; 15 unique models; baseline 2 cold processes; other models once',flush=True)
            results=evaluate_plan(plan['roles'],model,devices,True,True,lambda rows:atomic_json(out/'results.json',rows))
            assert len(results)==17
            success=True
        finally:
            atomic_json(record,dict(status='completed' if success else 'failed',pid=os.getpid(),started=started,finished=time.time(),
                reserved_gpu_hours=2*(time.time()-started)/3600,plan_id=plan_id))
    if success:subprocess.run([python,'scripts/report_official_test.py'],env=env,check=True)


if __name__=='__main__':main()
