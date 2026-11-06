"""One bounded retry for a known pre-update vLLM profiling race."""
import json
from pathlib import Path
import time
from .data import atomic_json
from .gpu_idle import wait_gpu_idle


def run_ablation_stage(entrypoint, root):
    root=Path(root)
    paths=[root/'runs'/f'full-full-qwen35-9b-b64-m8-f0-fla060-{v}-42-save2' for v in 'SE']
    offsets={p:(p/'stdout.log').stat().st_size if (p/'stdout.log').exists() else 0 for p in paths}
    def updates():
        result=[]
        for p in paths:
            f=p/'steps.jsonl'
            rows=[json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
            result.append(sum('training/global_step' in r.get('metrics',{}) for r in rows))
        return result
    before=updates()
    code=entrypoint('ablations')
    if code!=1 or updates()!=before:return code
    found=False
    for p in paths:
        f=p/'stdout.log'
        if not f.exists():continue
        with f.open('rb') as stream:
            stream.seek(max(offsets[p],f.stat().st_size-2_000_000))
            text=stream.read().decode('utf-8',errors='replace')
        found=found or ('Error in memory profiling' in text and 'current free memory' in text)
    if not found:return code
    # Never bypass the profiler assertion or terminate another GPU process.
    wait_gpu_idle(devices='0,1')
    atomic_json(root/'logs/ablation-startup-retry.json',dict(reason='vLLM initialization memory profiling race',
        timestamp=time.time(),retry=1,training_updates_before_retry=before))
    print('STARTUP_RETRY_ONCE: vLLM memory-profile race before any new update; prior logs and costs retained.',flush=True)
    return entrypoint('ablations')
