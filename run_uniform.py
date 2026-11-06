"""Platform entry: full-parameter uniform GRPO, seed 42, 200 updates."""
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
python = ROOT / '.venv/bin/python'
if Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
    if not python.exists():
        raise SystemExit(f'Project environment missing: {python}')
    os.execv(str(python), [str(python), '-u', str(Path(__file__).resolve())])

from dagrpo.data import atomic_json
from dagrpo.jobs import state
from dagrpo.locking import exclusive_lock

environment = os.environ.copy()
environment.update(GPU_HOURS='0', SESSION_HOURS='0', SAVE_FREQ='2',
                   VARIANT='U', SEED='42', BATCH_SIZE='64', MICRO_BATCH_SIZE='8',
                   FUSED_KERNELS='0', PYTHONUNBUFFERED='1')
for key in ('STOP_AFTER_STEP', 'INITIAL_CHECKPOINT', 'SKIP_INITIAL_VALIDATION',
            'DAGRPO_STOP_AFTER_STEP', 'DAGRPO_STOP_FILE'):
    environment.pop(key, None)
(ROOT / 'logs').mkdir(exist_ok=True)
main_log = ROOT / 'logs/train.log'
gpu_log = ROOT / 'runs/full-full-qwen35-9b-b64-m8-f0-fla060-U-42-save2/stdout.log'

with exclusive_lock(ROOT / 'logs/launch.lock'):
    for mode in ('prepare', 'train', 'matrix', 'pilot'):
        current = state(mode)
        if current['status'] == 'running':
            raise SystemExit(f'{mode} is already running (PID {current["pid"]}). Do not launch a duplicate.')
    with main_log.open('a', encoding='utf-8') as output:
        worker = subprocess.Popen(
            [sys.executable, '-u', '-m', 'dagrpo.jobs', 'worker', 'train'],
            cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True)
    atomic_json(ROOT / 'logs/train-job.json',
                dict(status='running', pid=worker.pid, started=time.time()))

print(f'Qwen3.5-9B FULL-PARAMETER GRPO | uniform U | seed 42 | 200 updates | PID={worker.pid}', flush=True)
print('Batch: 64 questions x 4 answers; microbatch: 8 per GPU; checkpoint every 2 updates.', flush=True)
print('Starts from the base model, or resumes this U run only. No wall-clock/GPU-hour limit.', flush=True)
print(f'Training log: {gpu_log}', flush=True)
print('This entry stays active. Ctrl+C requests a checkpoint and stop.', flush=True)
viewer = subprocess.Popen(['tail', '-n', '0', '-F', str(main_log), str(gpu_log)])
try:
    try:
        code = worker.wait()
    except KeyboardInterrupt:
        print('Checkpoint stop requested; waiting for the current update to finish.', flush=True)
        while worker.poll() is None:
            atomic_json(gpu_log.parent / 'stop-request.json', dict(reason='profile_stop'))
            try:
                worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        code = worker.returncode
finally:
    viewer.terminate()
    viewer.wait()
print(f'Training process exited with code {code}. See logs/train.log for final status.', flush=True)
if code == 2:
    print('Stopped at a checkpoint; the planned training may still be incomplete.', flush=True)
raise SystemExit(0 if code == 2 else code)
