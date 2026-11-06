"""Summarize real full-parameter pilot evidence, never extrapolate from LoRA."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from .data import atomic_json

TRAINING_SOURCES=tuple('dagrpo/'+name for name in
    ('train.py','runner.py','sampler.py','reward.py','verl_reward.py','full_engine.py','checkpoints.py'))+(
        'compat/sitecustomize.py','scripts/check_fla.py','scripts/verify_full_resume.py')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def summarize(run):
    run=Path(run)
    meta=json.loads((run/'run.json').read_text())
    if meta.get('training_mode')!='full': raise ValueError('Full-parameter evidence required; historical LoRA cannot be used')
    # Each resumed attempt's first update can include kernel warmup. Exclude it
    # when measuring stable throughput; initialization remains in the cost ledger.
    trace=rows(run/'steps.jsonl')
    updates=[r for r in trace if 'timing_s/update_actor' in r['metrics']]
    seen=set(); steady=[]
    for record in updates:
        pid=record.get('pid','unknown')
        if pid in seen: steady.append(record)
        seen.add(pid)
    stage_keys=('gen','old_log_prob','ref','update_actor','update_weights','step')
    timings={k:median([r['metrics']['timing_s/'+k] for r in steady if 'timing_s/'+k in r['metrics']])
        for k in stage_keys if any('timing_s/'+k in r['metrics'] for r in steady)}
    memory={}
    if (run/'gpu-memory.csv').exists():
        for row in csv.reader((run/'gpu-memory.csv').open()):
            if len(row)!=5: continue
            try:
                gpu=row[1].strip(); used=float(row[2]); total=float(row[3])
            except ValueError: continue
            item=memory.setdefault(gpu,dict(peak_mib=0,total_mib=total,samples=0))
            item['peak_mib']=max(item['peak_mib'],used); item['samples']+=1
    audit=[]; changes=[]
    for rank in (0,1):
        path=run/f'audit/trainability-rank{rank}.json'
        if path.exists(): audit.append(json.loads(path.read_text()))
        changes+=rows(run/f'audit/updates-rank{rank}.jsonl')
    weights_changed=any(w['changed_values']>0 for change in changes for w in change['weights'])
    trainable_verified=len(audit)==2 and all(a['training_mode']=='full' and a['lora_rank']==0 and
        a['total_parameters']==a['trainable_parameters'] and a['total_parameters']>8_000_000_000 and
        a['parameter_tensors']==a['optimizer_parameter_tensors'] for a in audit)
    finite=bool(changes) and all(math.isfinite(change['gradient_norm']) for change in changes)
    stdout=(run/'stdout.log').read_text(errors='replace') if (run/'stdout.log').exists() else ''
    from .checkpoints import latest_complete
    complete=latest_complete(run/'checkpoints')
    resume_verified='sampler_and_driver_rng_exact=True' in stdout
    restored_steps=[]
    import re
    restored_steps=[int(x) for x in re.findall(r'STUDY_RESTORED step=(\d+)',stdout)]
    continued=any(r['step']>s for s in restored_steps for r in updates)
    checks=dict(full_trainability=trainable_verified,base_weights_changed=weights_changed,finite_gradients=finite,
        complete_checkpoint=complete is not None,resume_rng=resume_verified,update_after_resume=continued,
        optimizer_state_restored=any(c['process_optimizer_step']==1 and c.get('optimizer_step_before_max',0)>0 for c in changes),
        both_gpus_measured=len(memory)==2)
    continuation=run/'resume-check.json'
    continuation_checks=json.loads(continuation.read_text()).get('checks',{}) if continuation.exists() else {}
    checks['sampler_next_batch_verified']=bool(continuation_checks) and all(continuation_checks.values())
    if meta.get('kernel_versions'):
        checks['fla_kernel_active']=len(audit)==2 and all(any(name.startswith('fla.') for name in
            a.get('linear_attention_implementations',[])) for a in audit)
        parity=Path(__file__).resolve().parent.parent/'validation/fla-parity.json'
        evidence=json.loads(parity.read_text()) if parity.exists() else {}
        checks['fla_parity']=bool(evidence.get('passed') and evidence.get('fla_core')==meta['kernel_versions'].get('fla-core') and
            evidence.get('torch')==meta.get('torch'))
    result=dict(run=str(run),training_mode='full',updates_logged=len(updates),stable_updates=len(steady),
        median_stage_seconds=timings,gpu_memory=memory,checks=checks,
        checkpoint_timings=rows(run/'checkpoint-timings.jsonl'),
        note='Two-second sampled whole-card peaks are lower bounds. Stable-step estimates require several updates; '
             'initialization, failed attempts, validation and checkpoint overhead must be added from the ledgers.')
    if len(steady)>=3 and 'step' in timings:
        result['training_only_200_steps_wall_hours']=timings['step']*200/3600
        result['training_only_matrix_gpu_hours']=timings['step']*200*8*2/3600
    atomic_json(run/'full-profile.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--certify',action='store_true',help='Write readiness proof only after all actual GPU/resume checks pass')
    args=parser.parse_args()
    result=summarize(args.run)
    if args.certify:
        if not all(result['checks'].values()): raise RuntimeError(f"Full GPU validation incomplete: {result['checks']}")
        run=Path(args.run).resolve(); root=Path(__file__).resolve().parent.parent
        meta=json.loads((run/'run.json').read_text())
        cfg=dict(x.split('=',1) for x in meta['config_overrides'])
        proof=dict(training_mode='full',gpu_resume_verified=True,run=str(run),
            kernel_versions=meta.get('kernel_versions',{}),
            model_revision=meta['data']['model_revision'],data_files=meta['data']['files'],
            training_config={k:cfg[k] for k in ('data.train_batch_size',
                'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu','actor_rollout_ref.model.use_fused_kernels')},
            training_source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in TRAINING_SOURCES},
            environment_freeze_sha256=hashlib.sha256((root/'environment.cu129.freeze.txt').read_bytes()).hexdigest(),
            checks=result['checks'])
        atomic_json(root/'GPU_FULL_SMOKE.json',proof)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
