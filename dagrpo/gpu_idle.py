"""Wait for previous evaluator workers to release GPUs; never kill other jobs."""
import subprocess
import time
import os

def wait_gpu_idle(timeout=120, devices=None):
    started=time.monotonic();empty=0
    visible=os.environ.get('CUDA_VISIBLE_DEVICES') if devices is None else devices
    selected=[] if visible is None else [x.strip() for x in visible.split(',') if x.strip()]
    if visible is not None and not selected:
        raise ValueError('No GPUs selected for evaluation')
    device_args=['--id='+','.join(selected)] if selected else []
    while time.monotonic()-started<timeout:
        apps=subprocess.check_output(['nvidia-smi',*device_args,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        memory=subprocess.check_output(['nvidia-smi',*device_args,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        idle=not apps and all(int(x.strip())<128 for x in memory.splitlines() if x.strip())
        empty=empty+1 if idle else 0
        if empty>=3:return
        time.sleep(2)
    raise RuntimeError(f'Selected GPUs {visible or "all"} still occupied or releasing memory; no evaluator started')
