"""Durable one-shot jobs; no SSH credentials and no recurring scheduler."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import math
from .data import atomic_json
from .locking import exclusive_lock
from .stages import planned_runs,run_directory,run_complete,validate_stage

ROOT=Path(__file__).resolve().parent.parent

def resource_limits(env):
    budget=float(env.get('GPU_HOURS','0'))
    session=float(env.get('SESSION_HOURS','0'))
    if math.isnan(budget) or budget<0 or not math.isfinite(session) or session<0:
        raise ValueError('Invalid budget/session duration')
    return (float('inf') if budget==0 else budget),session

def require_full_proof(env):
    ready=json.loads((ROOT/'READY.json').read_text())
    if not ready.get('prior_gpu_smoke_verified'):
        raise RuntimeError('Full-parameter GPU/resume validation required before formal training; run the full pilot first')
    proof=json.loads((ROOT/'GPU_FULL_SMOKE.json').read_text())
    expected={'data.train_batch_size':env.get('BATCH_SIZE','64'),
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':env.get('MICRO_BATCH_SIZE','8'),
        'actor_rollout_ref.model.use_fused_kernels':'true' if env.get('FUSED_KERNELS','0')=='1' else 'false'}
    if proof.get('training_config')!=expected:
        raise RuntimeError('Batch/kernel configuration differs from the verified full pilot; profile this configuration first')

def state(mode):
    path=ROOT/f'logs/{mode}-job.json'
    if not path.exists(): return {'status':'not_started'}
    result=json.loads(path.read_text())
    if result['status']=='running':
        try:
            os.kill(result['pid'],0)
            if sys.platform=='linux':
                cmd=Path(f'/proc/{result["pid"]}/cmdline').read_bytes()
                if b'dagrpo.jobs' not in cmd or b'worker' not in cmd: result['status']='interrupted'
        except (OSError,ProcessLookupError): result['status']='interrupted'
    return result

def start(mode):
    if sys.platform!='linux': raise RuntimeError('Launch the scripts on the Linux server')
    if mode not in ('prepare','train','matrix','pilot'): raise ValueError(mode)
    (ROOT/'logs').mkdir(exist_ok=True)
    with exclusive_lock(ROOT/'logs/launch.lock'):
        for other in ('prepare','train','matrix','pilot'):
            s=state(other)
            if s['status']=='running':
                print(f'{other} already running, PID={s["pid"]}; see logs/{other}.log')
                return
        if mode!='prepare' and not (ROOT/'READY.json').exists():
            raise RuntimeError('Preparation is not verified. Run bash scripts/prepare.sh first.')
        if mode=='matrix': (ROOT/'logs/matrix-stop-request.json').unlink(missing_ok=True)
        with (ROOT/f'logs/{mode}.log').open('a',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-m','dagrpo.jobs','worker',mode],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(ROOT/f'logs/{mode}-job.json',dict(status='running',pid=proc.pid,started=time.time()))
        print(f'{mode} started, PID={proc.pid}; log: {ROOT}/logs/{mode}.log')

def work(mode):
    job_started=time.monotonic()
    os.chdir(ROOT)
    # Parent publishes PID before the child can publish terminal status.
    with exclusive_lock(ROOT/'logs/launch.lock',blocking=True): pass
    env=os.environ.copy()
    if mode in ('train','matrix'): env.setdefault('SAVE_FREQ','2')
    env.update(HF_ENDPOINT=env.get('HF_ENDPOINT','https://hf-mirror.com'),
        HF_HOME=str(ROOT/'cache/huggingface'),HF_DATASETS_CACHE=str(ROOT/'cache/datasets'),
        HF_HUB_DISABLE_XET='1',CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',
        PYTHONFAULTHANDLER='1',
        # Use Ray's supported PID-based parent check in this container.
        # A dashboard-agent exit previously killed the entire local cluster.
        RAY_enable_pipe_based_agent_to_parent_health_check='0')
    python=str(ROOT/'.venv/bin/python')
    status='failed'; code=1
    print(f'JOB_STARTED: {mode}, pid={os.getpid()}, time={time.strftime("%Y-%m-%d %H:%M:%S")}',flush=True)
    try:
        if mode=='prepare':
            (ROOT/'READY.json').unlink(missing_ok=True)
            subprocess.run(['bash','scripts/setup_cu129.sh'],env=env,check=True)
            subprocess.run([python,'-m','dagrpo.cli','prepare'],env=env,check=True)
            subprocess.run([python,'-m','unittest','discover','-s','tests','-v'],env=env,check=True)
            subprocess.run([python,'-m','dagrpo.ready'],env=env,check=True)
            status='completed'; code=0
        else:
            from .runtime import python_runtime
            python=python_runtime(ROOT)
            print(f'PYTHON_RUNTIME: {python}',flush=True)
            subprocess.run([python,'-m','dagrpo.ready'],env=env,check=True)
            if mode in ('train','matrix'): require_full_proof(env)
            budget,session=resource_limits(env)
            runs=planned_runs(mode,env)
            if mode=='matrix' and env.get('EXPERIMENT_STAGE') in ('repeats','ablations'):
                print('STAGE_PREFLIGHT: '+json.dumps(validate_stage(ROOT,env['EXPERIMENT_STAGE'],env)),flush=True)
            for variant,seed in runs:
                if mode=='matrix' and (ROOT/'logs/matrix-stop-request.json').exists():
                    status='stopped_resume_available';code=2;break
                phase='pilot' if mode=='pilot' else 'full'
                batch=int(env.get('BATCH_SIZE','64')); micro=int(env.get('MICRO_BATCH_SIZE','8'))
                fused=env.get('FUSED_KERNELS','0')=='1'
                out=run_directory(ROOT,phase,variant,seed,env)
                if run_complete(out,variant,seed,30 if phase=='pilot' else 200):
                    print(f'RUN_SKIPPED_COMPLETE: {variant}/{seed}, {out}',flush=True)
                    continue
                print(f'RUN_STARTED: {variant}/{seed}, directory={out}',flush=True)
                if mode=='matrix':
                    atomic_json(ROOT/'logs/matrix-active.json',dict(variant=variant,seed=seed,run_dir=str(out)))
                remaining=max(0,session*3600-(time.monotonic()-job_started))/3600 if session else 0
                if session and remaining<=0:
                    status='stopped_resume_available'; code=2; break
                command=[python,'-m','dagrpo.cli','run','--phase',phase,'--variant',variant,'--seed',str(seed),
                    '--out',str(out),'--budget-gpu-hours',str(budget),'--session-hours',str(remaining),
                    '--training-mode','full','--batch-size',str(batch),'--micro-batch-size',str(micro),
                    '--pilot-budget-cap',env.get('PILOT_BUDGET_CAP',str(budget)),
                    '--ledger',str(ROOT/f'runs/{phase}-budget.json')]
                if fused: command.append('--fused-kernels')
                boundary=env.get('STOP_AFTER_STEP','2' if mode=='pilot' else '')
                if mode=='matrix' and env.get('QUEUE_SMOKE_STEPS'):
                    if env['QUEUE_SMOKE_STEPS']!='2':raise ValueError('Queue smoke boundary must be 2')
                    boundary='2'
                if boundary: command+=['--stop-after-step',boundary]
                if env.get('SAVE_FREQ'): command+=['--save-freq',env['SAVE_FREQ']]
                if env.get('INITIAL_CHECKPOINT'): command+=['--initial-checkpoint',env['INITIAL_CHECKPOINT']]
                if mode=='pilot' and env.get('SKIP_INITIAL_VALIDATION')=='1': command.append('--skip-initial-validation')
                code=subprocess.run(command,env=env).returncode
                if code==0 and not run_complete(out,variant,seed,30 if phase=='pilot' else 200):
                    raise RuntimeError(f'Process exited without complete final evidence: {out}')
                if code:
                    status='stopped_resume_available' if code==2 else 'failed'
                    break
            else: status='completed'; code=0
    except Exception as exc:
        status='failed';code=1
        print(f'JOB_FAILED: {exc}',flush=True)
    finally:
        atomic_json(ROOT/f'logs/{mode}-job.json',dict(status=status,pid=os.getpid(),exit_code=code,finished=time.time()))
    return code

def main():
    action=sys.argv[1]
    if action=='start': start(sys.argv[2])
    elif action=='worker': raise SystemExit(work(sys.argv[2]))
    elif action=='status':
        for mode in ('prepare','pilot','train','matrix'): print(mode,json.dumps(state(mode),ensure_ascii=False))
        for p in (ROOT/'runs').glob('*-budget.json'): print(p.name,p.read_text())
    else: raise ValueError(action)

if __name__=='__main__': main()
