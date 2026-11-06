"""After reproducibility passes, reevaluate retained full checkpoints on DEV only."""
from pathlib import Path
import os,sys,json,subprocess,time,shutil
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from dagrpo.data import atomic_json,digest
from dagrpo.jobs import state
from dagrpo.runtime import python_runtime
from dagrpo.stages import run_directory,run_complete
from dagrpo.locking import exclusive_lock
from dagrpo.eval_audit import compare
from dagrpo.gpu_idle import wait_gpu_idle
from dagrpo.eval_queue import evaluate_plan

if __name__=='__main__':
    scope=os.environ.get('DAGRPO_EVAL_SCOPE','UA')
    if scope not in ('UA','SE'):raise ValueError('Evaluation scope must be UA or SE')
    sync=os.environ.get('DAGRPO_EVAL_SYNC')=='1';suffix='-'+os.environ.get('DAGRPO_EVAL_TAG','sync') if sync else ''
    scope_suffix='-SE' if scope=='SE' else ''
    record=ROOT/f'logs/saved-model-eval{suffix}{scope_suffix}-job.json'
    if record.exists():atomic_json(ROOT/f'logs/saved-model-eval{suffix}{scope_suffix}-prior-{time.time_ns()}.json',json.loads(record.read_text()))
    atomic_json(record,dict(status='waiting_for_reproducibility',pid=os.getpid()))
    with exclusive_lock(ROOT/'logs/launch.lock',blocking=True):
        stable=ROOT/f'reports/evaluation-stable-20261103{suffix}'
        evidence=json.loads((stable/'comparison.json').read_text())
        if len(evidence['sessions'])!=6 or any(ids for d in evidence['comparisons'].values() for ids in d.values()):
            atomic_json(record,dict(status='failed',pid=os.getpid(),reason='Reproducibility preflight did not pass'))
            raise RuntimeError('Reproducibility preflight did not pass; no checkpoint evaluated')
        assert all(state(m)['status']!='running' for m in ['train','matrix','pilot','prepare'])
        base_prompts=json.loads((stable/'session-0/prompts.json').read_text())['problems']
        series='saved-model-dev-recheck-20261105' if scope=='SE' else 'saved-model-dev-recheck-20261103'
        out=ROOT/f'reports/{series}{suffix}{scope_suffix}';out.mkdir(exist_ok=True)
        started=time.time();code=1
        atomic_json(record,dict(status='running',pid=os.getpid(),started=started))
        try:
            python=python_runtime(ROOT);m=json.loads((ROOT/'data/gsm8k-qwen35-9b/manifest.json').read_text())
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','0,1'),PYTHONUNBUFFERED='1',
                VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_BATCH_INVARIANT='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',
                HF_HOME=str(ROOT/'cache/huggingface'),TRITON_CACHE_DIR=str(ROOT/'cache/triton'),VLLM_CACHE_ROOT=str(ROOT/'cache/vllm'))
            if env.get('DAGRPO_EVAL_REFERENCE')=='1':env.update(PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            cpu_env={**env,'CUDA_VISIBLE_DEVICES':''}
            plan=[]
            for role in ['final','historical_best']:
                for v in scope:
                    for seed in ([42] if scope=='SE' else [42,43,44]):
                        r=run_directory(ROOT,'full',v,seed,{'SAVE_FREQ':'2'})
                        assert run_complete(r,v,seed)
                        step=200 if role=='final' else json.loads((r/'best-dev.json').read_text())['step']
                        plan.append(dict(variant=v,seed=seed,role=role,step=step,run=str(r)))
            atomic_json(out/'plan.json',plan)
            results=[]
            def evaluate_item(item,device=None,verify_export=False,public_gate=False):
                r=Path(item['run']);step=item['step'];source=r/f'checkpoints/global_step_{step}'
                assert (source/'complete.json').exists()
                key=f"{item['variant']}-{item['seed']}-step{step}"
                dest=ROOT/'exports/dev-recheck'/key;cache=out/key
                cache.mkdir(parents=True,exist_ok=True)
                marker=cache/'complete.json'
                if marker.exists():
                    if public_gate:
                        assert all(not ids for ids in json.loads((cache/'public-entry-comparison.json').read_text()).values())
                    return json.loads(marker.read_text())
                identity={str(p.name):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
                    for p in (source/'actor').glob('model_world_size_2_rank_*.pt')}
                assert len(identity)==2
                if not (dest/'export-complete.json').exists():
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    assert shutil.disk_usage(dest.parent).free>60*1024**3
                    temporary=dest.with_name(dest.name+f'.partial-{time.time_ns()}')
                    print('EXPORT_START',key,flush=True)
                    with (cache/'export.log').open('a') as log:
                        subprocess.run([python,'-m','verl.model_merger','merge','--backend','fsdp',
                            '--local_dir',str(source/'actor'),'--target_dir',str(temporary)],env=cpu_env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    base_gen=Path(m['model_path'])/'generation_config.json'
                    if base_gen.exists():shutil.copy2(base_gen,temporary/'generation_config.json')
                    atomic_json(temporary/'export-complete.json',dict(source=str(source),model_shards=identity,precision='bf16 via pinned verl merger'))
                    assert not dest.exists(),'Unexpected export without completion marker'
                    temporary.rename(dest)
                else:
                    assert json.loads((dest/'export-complete.json').read_text())['model_shards']==identity
                if verify_export:
                    print('EXPORT_NUMERICAL_TEST',key,flush=True)
                    with (cache/'export-test.log').open('a') as log:
                        subprocess.run([python,'-m','verl.model_merger','test','--backend','fsdp',
                            '--local_dir',str(source/'actor'),'--test_hf_dir',str(dest)],env=cpu_env,stdout=log,stderr=subprocess.STDOUT,check=True)
                attempt=cache/f'attempt-{time.time_ns()}'
                print('DEV_RECHECK_START',key,item['role'],flush=True)
                job_env=env if device is None else {**env,'CUDA_VISIBLE_DEVICES':device}
                wait_gpu_idle(timeout=int(os.environ.get('DAGRPO_EVAL_WAIT_SECONDS','120')),devices=device)
                subprocess.run([python,'-u','-m','dagrpo.eval_audit','--model',str(dest),
                    '--data',str(ROOT/'data/gsm8k-qwen35-9b/dev.jsonl'),'--out',str(attempt),
                    '--seed','42','--repeats','2'],env=job_env,check=True)
                assert json.loads((attempt/'prompts.json').read_text())['problems']==base_prompts
                a,b=([json.loads(l) for l in (attempt/f'repeat-{i}.jsonl').read_text().splitlines()] for i in [0,1])
                differences=compare(a,b)
                atomic_json(cache/'repeat-comparison.json',differences)
                assert all(not ids for ids in differences.values()),f'Checkpoint repeated output differs: {key}'
                if public_gate:
                    print('PUBLIC_COLD_ENTRY_GATE',key,flush=True)
                    wait_gpu_idle(devices=device);public=cache/'public-entry.jsonl'
                    if public.exists():
                        prior=cache/f'prior-public-entry-{time.time_ns()}'
                        prior.mkdir()
                        for old in [public,public.with_suffix('.summary.json'),cache/'public-entry-comparison.json']:
                            if old.exists():old.rename(prior/old.name)
                    subprocess.run([python,'-m','dagrpo.cli','evaluate','--model',str(dest),
                        '--data',str(ROOT/'data/gsm8k-qwen35-9b/dev.jsonl'),'--out',str(public)],env=job_env,check=True)
                    c=[json.loads(l) for l in public.read_text().splitlines()]
                    fields=['problem_id','token_ids','text','score','finish_reason','stop_reason','truncated','format_error']
                    cold={k:[r['problem_id'] for r,s in zip(a,c,strict=True) if r[k]!=s[k]] for k in fields}
                    atomic_json(cache/'public-entry-comparison.json',cold)
                    assert len(c)==512 and all(not ids for ids in cold.values()),'Public cold entry gate failed; no further models evaluated'
                summary=json.loads((attempt/'summary.json').read_text())[0]
                complete=dict(accuracy=summary['accuracy'],correct=int(sum(r['score'] for r in a)),
                    checkpoint=str(source),export=str(dest),attempt=str(attempt),repeated_tokens_identical=True)
                atomic_json(marker,complete)
                print('DEV_RECHECK_COMPLETE',key,json.dumps(complete),flush=True)
                return complete
            parallel=os.environ.get('DAGRPO_EVAL_PARALLEL')=='1'
            devices=[x.strip() for x in env['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]
            if parallel:
                assert env.get('DAGRPO_EVAL_REFERENCE')=='1','Parallel evaluation requires TP1 reference mode'
            results=evaluate_plan(plan,evaluate_item,devices,parallel,sync,
                lambda rows:atomic_json(out/'results.json',rows))
            atomic_json(out/'results.json',results);code=0
        finally:
            atomic_json(record,dict(status='completed' if code==0 else 'failed',pid=os.getpid(),exit_code=code,
                started=started,finished=time.time(),gpu_hours=2*(time.time()-started)/3600))
        raise SystemExit(code)
