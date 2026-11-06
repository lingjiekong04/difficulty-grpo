"""Foreground platform process for explicit, sequential experiment stages."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .data import atomic_json
from .jobs import state,require_full_proof
from .locking import exclusive_lock
from .stages import validate_stage

ROOT=Path(__file__).resolve().parent.parent


def main(stage):
    parser=argparse.ArgumentParser(description=f'Full-parameter GRPO stage: {stage}')
    controls=parser.add_mutually_exclusive_group()
    controls.add_argument('--check',action='store_true',help='CPU-only preflight; starts no GPU job')
    controls.add_argument('--smoke',action='store_true',help='Save and stop the first pending run at update 2; never continues the queue')
    args=parser.parse_args()
    os.chdir(ROOT)
    env=os.environ.copy()
    env.update(EXPERIMENT_STAGE=stage,GPU_HOURS='0',SESSION_HOURS='0',SAVE_FREQ='2',
               BATCH_SIZE='64',MICRO_BATCH_SIZE='8',FUSED_KERNELS='0',PYTHONUNBUFFERED='1')
    for key in ('STOP_AFTER_STEP','INITIAL_CHECKPOINT','SKIP_INITIAL_VALIDATION',
                'DAGRPO_STOP_AFTER_STEP','DAGRPO_STOP_FILE','QUEUE_SMOKE_STEPS'):
        env.pop(key,None)
    if args.smoke:env['QUEUE_SMOKE_STEPS']='2'
    require_full_proof(env)
    plan=validate_stage(ROOT,stage,env)
    if args.check:
        print(json.dumps(plan,indent=2),flush=True)
        return 0
    (ROOT/'logs').mkdir(exist_ok=True)
    with exclusive_lock(ROOT/'logs/launch.lock'):
        for mode in ('prepare','train','matrix','pilot'):
            current=state(mode)
            if current['status']=='running':
                raise RuntimeError(f'{mode} already running (PID {current["pid"]}); no duplicate launched')
        if all(r['status']=='completed' for r in plan['runs']):
            print(f'STAGE_ALREADY_COMPLETED: {stage}',flush=True)
            return 0
        (ROOT/'logs/matrix-stop-request.json').unlink(missing_ok=True)
        (ROOT/'logs/matrix-active.json').unlink(missing_ok=True)
        with (ROOT/'logs/matrix.log').open('a',encoding='utf-8') as log:
            worker=subprocess.Popen([sys.executable,'-u','-m','dagrpo.jobs','worker','matrix'],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
        atomic_json(ROOT/'logs/matrix-job.json',dict(status='running',pid=worker.pid,
            started=time.time(),stage=stage))
    print(f'STAGE_STARTED: {stage}; PID={worker.pid}',flush=True)
    print(' -> '.join(f"{r['variant']}/{r['seed']}" for r in plan['runs']),flush=True)
    print('200 updates each; full parameters; batch 64 x 4; microbatch 8/GPU; save every 2 updates.',flush=True)
    print('Completed runs skip; interrupted runs resume their own checkpoint. No time/GPU-hour cap.',flush=True)
    if args.smoke:print('SMOKE ONLY: save/stop at update 2 (or next update on a later resume); no automatic long run.',flush=True)
    print('Ctrl+C or SIGTERM requests checkpoint-stop; the next experiment will not start.',flush=True)
    stop_requested=False
    def request_stop(signum,frame):
        nonlocal stop_requested
        if not stop_requested:
            print('Checkpoint-stop requested; waiting for a safe update boundary.',flush=True)
        stop_requested=True
    previous={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    viewer=None
    try:
        paths=[str(ROOT/'logs/matrix.log')]+[str(Path(r['run_dir'])/'stdout.log') for r in plan['runs']]
        viewer=subprocess.Popen(['tail','-n','0','-F',*paths],stderr=subprocess.DEVNULL)
        while worker.poll() is None:
            if stop_requested:
                atomic_json(ROOT/'logs/matrix-stop-request.json',dict(reason='profile_stop'))
                active=ROOT/'logs/matrix-active.json'
                if active.exists():
                    current=json.loads(active.read_text())
                    run=Path(current['run_dir']).resolve()
                    if run.parent != (ROOT/'runs').resolve():raise RuntimeError('Active run escaped project runs')
                    atomic_json(run/'stop-request.json',dict(reason='profile_stop'))
            time.sleep(.5)
        code=worker.returncode
    finally:
        if viewer is not None:
            viewer.terminate()
            try:viewer.wait(timeout=5)
            except subprocess.TimeoutExpired:viewer.kill();viewer.wait()
        for sig,handler in previous.items():signal.signal(sig,handler)
    if code==0:print(f'STAGE_COMPLETED: {stage}',flush=True)
    elif code==2:print('STAGE_STOPPED: checkpoint available; rerun this same script to resume.',flush=True)
    else:print(f'STAGE_FAILED: exit={code}; inspect logs/matrix.log and the active run stdout.log.',flush=True)
    # Keep interrupted/incomplete stages distinguishable from successful completion.
    return code
