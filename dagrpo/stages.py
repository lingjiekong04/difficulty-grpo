"""Explicit experiment stages and fail-closed completion checks."""
import hashlib
import json
from pathlib import Path

REPEATS = [('U',43),('A',43),('U',44),('A',44)]
ABLATIONS = [('S',42),('E',42)]


def planned_runs(mode, env):
    if mode == 'pilot': return [('U',42)]
    if mode == 'train': return [(env.get('VARIANT','A'),int(env.get('SEED','42')))]
    if mode != 'matrix': raise ValueError(f'Unknown training mode: {mode}')
    stage = env.get('EXPERIMENT_STAGE','all')
    if stage == 'repeats': return list(REPEATS)
    if stage == 'ablations': return list(ABLATIONS)
    if stage == 'all': return [('U',42),('A',42)]+list(REPEATS)+list(ABLATIONS)
    raise ValueError(f'Unknown experiment stage: {stage}')


def run_directory(root, phase, variant, seed, env):
    batch=int(env.get('BATCH_SIZE','64')); micro=int(env.get('MICRO_BATCH_SIZE','8'))
    fused=int(env.get('FUSED_KERNELS','0')=='1')
    kernel=Path(root)/'environment.kernels.json'
    tag='-'+json.loads(kernel.read_text())['tag'] if kernel.exists() else ''
    name=f'{phase}-full-qwen35-9b-b{batch}-m{micro}-f{fused}{tag}-{variant}-{seed}'
    if env.get('SAVE_FREQ'):
        frequency=int(env['SAVE_FREQ'])
        if frequency<=0: raise ValueError('SAVE_FREQ must be positive')
        name+=f'-save{frequency}'
    return Path(root)/'runs'/name


def run_complete(path, variant, seed, steps=200):
    """A dev row alone must never cause a failed/partial run to be skipped."""
    path=Path(path)
    try:
        metadata=json.loads((path/'run.json').read_text())
        study=metadata['study']
        if metadata.get('training_mode')!='full': return False
        if any(study.get(k)!=v for k,v in dict(variant=variant,seed=seed,steps=steps).items()):return False
        execution=json.loads((path/'execution.json').read_text())
        if execution.get('status')!='completed' or execution.get('exit_code')!=0:return False
        rows=[json.loads(x) for x in (path/'evaluations.jsonl').read_text().splitlines() if x.strip()]
        if not any(r.get('step')==steps and r.get('variant')==variant and r.get('seed')==seed for r in rows):return False
        checkpoint=path/f'checkpoints/global_step_{steps}'
        marker=json.loads((checkpoint/'complete.json').read_text())
        if marker!=dict(step=steps,training_mode='full'):return False
        files=['data.pt','study_state.pt','actor/fsdp_config.json']
        files += [f'actor/{kind}_world_size_2_rank_{rank}.pt' for kind in ['model','optim','extra_state'] for rank in range(2)]
        return all((checkpoint/f).is_file() and (checkpoint/f).stat().st_size>0 for f in files)
    except (OSError,ValueError,KeyError,TypeError):
        return False


def validate_stage(root, stage, env):
    """CPU preflight: require existing evidence and match the seed-42 protocol."""
    from .runner import overrides,verify_data
    from .report import protocol_id
    root=Path(root).resolve()
    if stage not in ('repeats','ablations'):raise ValueError(stage)
    for k,v in dict(BATCH_SIZE='64',MICRO_BATCH_SIZE='8',FUSED_KERNELS='0',SAVE_FREQ='2').items():
        if env.get(k,v)!=v:raise ValueError(f'Stage requires {k}={v}')
    if any(env.get(k) for k in ('STOP_AFTER_STEP','INITIAL_CHECKPOINT','SKIP_INITIAL_VALIDATION','DAGRPO_STOP_AFTER_STEP')):
        raise ValueError('Formal queue cannot inherit smoke/stop/fork flags')
    reference_path=run_directory(root,'full','A',42,env)
    reference=json.loads((reference_path/'run.json').read_text())
    manifest=verify_data(root/'data/gsm8k-qwen35-9b')
    if manifest!=reference['data']:raise ValueError('Data/model differs from seed-42 baseline')
    freeze=hashlib.sha256((root/'environment.cu129.freeze.txt').read_bytes()).hexdigest()
    if freeze!=reference['environment_freeze_sha256']:raise ValueError('Environment freeze changed')
    prerequisites=[('U',42),('A',42)] + (REPEATS if stage=='ablations' else [])
    for variant,seed in prerequisites:
        path=run_directory(root,'full',variant,seed,env)
        if not run_complete(path,variant,seed):raise RuntimeError(f'Prerequisite incomplete: {path.name}')
        if protocol_id(json.loads((path/'run.json').read_text()))!=protocol_id(reference):
            raise ValueError(f'Prerequisite protocol differs: {path.name}')
    result=[]
    for variant,seed in planned_runs('matrix',{'EXPERIMENT_STAGE':stage}):
        path=run_directory(root,'full',variant,seed,env)
        settings=overrides(root/'data/gsm8k-qwen35-9b',path,seed,200,'full',
            batch_size=64,micro_batch_size=8,fused_kernels=False,save_freq=2,val_batch_size=128)
        expected={**reference,'config_overrides':settings}
        if protocol_id(expected)!=protocol_id(reference):raise ValueError('Planned training protocol changed')
        if (path/'run.json').exists():
            existing=json.loads((path/'run.json').read_text())
            if protocol_id(existing)!=protocol_id(reference) or existing['study']['variant']!=variant or existing['study']['seed']!=seed:
                raise ValueError(f'Existing run protocol/study mismatch: {path.name}')
        result.append(dict(variant=variant,seed=seed,steps=200,run_dir=str(path),
            status='completed' if run_complete(path,variant,seed) else 'pending_or_resumable'))
    return dict(stage=stage,protocol=protocol_id(reference),runs=result)
