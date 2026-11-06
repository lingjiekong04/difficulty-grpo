"""Argument-safe launch, immutable run metadata, and a shared GPU-hour ledger."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .data import atomic_json
from .locking import exclusive_lock

def recover_ledger(ledger):
    ledger=dict(ledger)
    active=ledger.pop('active',None)
    if active:
        ledger['runs']=list(ledger['runs'])+[{**active,'status':'interrupted','exit_code':None}]
    ledger.pop('active_run',None)
    return ledger

def overrides(data, run, seed, steps, training_mode="full", *, batch_size=64,
              micro_batch_size=8, fused_kernels=False, save_freq=10, val_batch_size=32,
              initial_validation=True):
    if training_mode != 'full':
        raise ValueError('This project entrypoint requires full-parameter training; LoRA runs are historical only')
    if batch_size < 2 or batch_size % 2 or micro_batch_size < 1 or (batch_size*4//2) % micro_batch_size:
        raise ValueError('Two ranks must receive an integral number of microbatches')
    from . import reward
    manifest = json.loads((Path(data)/"manifest.json").read_text(encoding="utf-8"))
    model = manifest["model_path"]
    values = {
        "model_engine":"dp", "trainer.use_v1":False,
        "algorithm.adv_estimator":"grpo", "algorithm.norm_adv_by_std_in_grpo":True,
        "algorithm.use_kl_in_reward":False, "algorithm.filter_groups.enable":False,
        "data.train_files":str(Path(data).resolve()/"train.parquet"),
        "data.val_files":str(Path(data).resolve()/"dev.parquet"),
        "data.train_batch_size":batch_size, "data.val_batch_size":val_batch_size,
        "data.max_prompt_length":512, "data.max_response_length":1024,
        "data.dataloader_num_workers":0, "data.shuffle":False, "data.seed":seed,
        "data.filter_overlong_prompts":False, "data.truncation":"error",
        "+data.apply_chat_template_kwargs.enable_thinking":False,
        "actor_rollout_ref.model.path":model,
        "actor_rollout_ref.model.enable_gradient_checkpointing":True,
        "actor_rollout_ref.model.use_remove_padding":False,
        "actor_rollout_ref.model.lora_rank":0,
        "actor_rollout_ref.model.lora_adapter_path":None,
        "actor_rollout_ref.model.external_lib":"dagrpo.full_engine",
        "actor_rollout_ref.model.use_fused_kernels":fused_kernels,
        "actor_rollout_ref.model.fused_kernel_options.impl_backend":"torch",
        "+actor_rollout_ref.model.override_config.attn_implementation":"sdpa",
        "actor_rollout_ref.actor.strategy":"fsdp2",
        "actor_rollout_ref.actor.optim.lr":1e-6,
        "actor_rollout_ref.actor.ppo_mini_batch_size":batch_size,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu":micro_batch_size,
        "actor_rollout_ref.actor.ppo_epochs":1,
        "actor_rollout_ref.actor.data_loader_seed":seed,
        "actor_rollout_ref.actor.fsdp_config.seed":seed,
        "actor_rollout_ref.actor.clip_ratio":.2,
        "actor_rollout_ref.actor.loss_agg_mode":"token-mean",
        "actor_rollout_ref.actor.entropy_coeff":0,
        "actor_rollout_ref.actor.use_kl_loss":True,
        "actor_rollout_ref.actor.kl_loss_coef":.001,
        "actor_rollout_ref.actor.kl_loss_type":"low_var_kl",
        "actor_rollout_ref.actor.fsdp_config.model_dtype":"fp32",
        "actor_rollout_ref.actor.fsdp_config.offload_policy":True,
        "actor_rollout_ref.actor.fsdp_config.param_offload":False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload":False,
        "actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation":False,
        "actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking":True,
        "actor_rollout_ref.actor.fsdp_config.entropy_checkpointing":False,
        "+actor_rollout_ref.actor.fsdp_config.mixed_precision":{"param_dtype":"bf16","reduce_dtype":"fp32"},
        "actor_rollout_ref.ref.strategy":"fsdp2",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu":micro_batch_size,
        "actor_rollout_ref.ref.fsdp_config.model_dtype":"bf16",
        "actor_rollout_ref.ref.fsdp_config.param_offload":True,
        "actor_rollout_ref.rollout.name":"vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size":2,
        "actor_rollout_ref.rollout.gpu_memory_utilization":.6,
        "actor_rollout_ref.rollout.max_num_seqs":128,
        "actor_rollout_ref.rollout.n":4,
        "actor_rollout_ref.rollout.seed":seed,
        "actor_rollout_ref.rollout.temperature":1.0,
        "actor_rollout_ref.rollout.top_p":1.0,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu":micro_batch_size,
        "actor_rollout_ref.rollout.val_kwargs.do_sample":False,
        "actor_rollout_ref.rollout.val_kwargs.n":1,
        "reward.custom_reward_function.path":str(Path(reward.__file__).resolve()),
        "reward.custom_reward_function.name":"compute_score",
        "reward.reward_manager.source":"importlib",
        "reward.reward_manager.name":"StrictNumericRewardManager",
        "reward.reward_manager.module.path":str(Path(__file__).with_name("verl_reward.py").resolve()),
        "reward.num_workers":2,
        "trainer.nnodes":1, "trainer.n_gpus_per_node":2,
        "trainer.total_training_steps":steps, "trainer.total_epochs":1,
        "trainer.save_freq":save_freq, "trainer.test_freq":25,
        "trainer.val_before_train":initial_validation, "trainer.logger":["console"],
        "trainer.project_name":"difficulty-grpo",
        "+trainer.study_master_port_range":[24000,25000],
        "trainer.experiment_name":Path(run).name,
        "trainer.default_local_dir":str(Path(run).resolve()/"checkpoints"),
        "trainer.resume_mode":"auto", "trainer.default_hdfs_dir":None,
        "ray_kwargs.ray_init.runtime_env.py_executable":None,
        "ray_kwargs.ray_init.num_cpus":24,
        "+ray_kwargs.ray_init.include_dashboard":False,
        "+ray_kwargs.ray_init.object_store_memory":1073741824,
        "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_ENGINE_VENDOR":"dagrpo_full",
        "+ray_kwargs.ray_init.runtime_env.env_vars.DAGRPO_AUDIT_DIR":str(Path(run).resolve()/"audit"),
        "+ray_kwargs.ray_init.runtime_env.env_vars.DAGRPO_CPU_THREADS":"8",
    }
    def literal(value):
        # Hydra's dict grammar accepts unquoted keys, unlike JSON object literals.
        if isinstance(value,dict): return '{'+','.join(k+':'+literal(v) for k,v in value.items())+'}'
        return json.dumps(value)
    return [k+"="+literal(v) for k,v in values.items()]

def verify_data(data):
    import hashlib
    manifest = json.loads((Path(data)/"manifest.json").read_text(encoding="utf-8"))
    for file, expected in manifest["files"].items():
        if hashlib.sha256((Path(data)/file).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Data hash mismatch: {file}")
    return manifest


def validate_initial_checkpoint(path, study, manifest):
    path=Path(path).resolve()
    marker=json.loads((path/'complete.json').read_text())
    parent=json.loads((path.parent.parent/'run.json').read_text())
    if marker.get('training_mode')!='full' or parent.get('training_mode')!='full':
        raise ValueError('Initial checkpoint must be a verified full-parameter run')
    for key in ('variant','seed','steps','batch_size'):
        if parent['study'].get(key)!=study[key]: raise ValueError(f'Initial checkpoint study mismatch: {key}')
    if parent['data']!=manifest: raise ValueError('Initial checkpoint model/data mismatch')
    for file in ('study_state.pt','data.pt'):
        if not (path/file).is_file(): raise ValueError('Incomplete initial checkpoint')
    if not (path/'actor').is_dir(): raise ValueError('Missing actor checkpoint')
    return str(path)

def supervise(command, run_dir, gpu_hours, ledger_path, env=None, gpus=2,
              session_seconds=None, grace_seconds=600):
    # A measured 9B update takes ~230s; leave room for evaluation and saving too.
    with exclusive_lock(Path(ledger_path).with_suffix('.lock')):
        return _supervise(command,run_dir,gpu_hours,ledger_path,env,gpus,session_seconds,grace_seconds)

def _supervise(command, run_dir, gpu_hours, ledger_path, env, gpus, session_seconds, grace_seconds):
    if gpu_hours <= 0:
        raise ValueError("Positive GPU-hour budget required")
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    proc = None
    memory_proc = None
    memory_log = None
    spent = 0
    try:
        ledger = recover_ledger(json.loads(ledger_path.read_text())) if ledger_path.exists() else {"gpu_hours":0, "runs":[]}
        spent = ledger["gpu_hours"]
        if spent >= gpu_hours:
            raise ValueError("Shared GPU-hour budget exhausted")
        allowed_seconds = (gpu_hours-spent)*3600/gpus
        session_limit = session_seconds if session_seconds is not None else float('inf')
        deadline = started + max(0,min(allowed_seconds,session_limit)-5)
        stop_status = 'budget_stop' if allowed_seconds <= session_limit else 'session_stop'
        stop_file=Path(run_dir)/'stop-request.json'
        ack_file=Path(run_dir)/'stop-ack.json'
        Path(run_dir).mkdir(parents=True,exist_ok=True)
        stop_file.unlink(missing_ok=True)
        ack_file.unlink(missing_ok=True)
        env=dict(os.environ if env is None else env)
        env['DAGRPO_STOP_FILE']=str(stop_file.resolve())
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        # Whole-card measurements include vLLM, reference, synchronization and saving.
        # A missing monitoring utility must not prevent checkpoint recovery.
        try:
            memory_log=(Path(run_dir)/'gpu-memory.csv').open('a',encoding='utf-8')
            memory_proc=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits','-l','2'],stdout=memory_log,stderr=subprocess.DEVNULL)
        except OSError:
            if memory_log: memory_log.close()
            memory_log=None
        with (Path(run_dir)/"stdout.log").open("a", encoding="utf-8") as log:
            print(f"TRAINING_LOG: {Path(run_dir)/'stdout.log'}", flush=True)
            log.write(f"\n=== ATTEMPT {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log.flush()
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=(os.name != "nt"))
            status = "completed"
            while proc.poll() is None:
                elapsed = time.monotonic()-started
                atomic_json(ledger_path, {**ledger, "gpu_hours":spent+elapsed*gpus/3600,
                    "active":dict(run=str(run_dir),seconds=elapsed,gpu_hours=elapsed*gpus/3600,pid=proc.pid)})
                if time.monotonic() >= deadline-grace_seconds and not stop_file.exists():
                    atomic_json(stop_file,{'reason':stop_status})
                if time.monotonic() >= deadline:
                    status = stop_status
                    if os.name == "nt":
                        proc.terminate()
                    else:
                        os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=min(5, max(0, allowed_seconds-elapsed)))
                    except subprocess.TimeoutExpired:
                        if os.name == "nt": proc.kill()
                        else: os.killpg(proc.pid, signal.SIGKILL)
                    break
                time.sleep(.2)
            proc.wait()
        if status == 'completed' and ack_file.exists():
            status=json.loads(ack_file.read_text())['reason']
        if proc.returncode and status == "completed": status = "failed"
        elapsed = time.monotonic()-started
        entry = dict(run=str(run_dir), status=status, exit_code=proc.returncode,
                     seconds=elapsed, gpu_hours=elapsed*gpus/3600, python=command[0])
        atomic_json(Path(run_dir)/"execution.json", entry)
        atomic_json(ledger_path, {"gpu_hours":spent+entry["gpu_hours"], "runs":ledger["runs"]+[entry]})
        print(f"TRAINING_EXIT: status={status}, exit_code={proc.returncode}, "
              f"cumulative_gpu_hours={spent+entry['gpu_hours']:.4f}", flush=True)
        if status == 'failed':
            print('TRAINING_ERROR_TAIL (full traceback in TRAINING_LOG):', flush=True)
            with (Path(run_dir)/'stdout.log').open(encoding='utf-8',errors='replace') as stream:
                from collections import deque
                for line in deque(stream, maxlen=24):
                    print(line.rstrip()[:500], flush=True)
        return 0 if status == "completed" else (2 if status in ('budget_stop','session_stop','profile_stop') else 1)
    finally:
        if proc is not None and proc.poll() is None:
            if os.name == "nt": proc.kill()
            else: os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        if memory_proc is not None:
            memory_proc.terminate()
            try: memory_proc.wait(timeout=5)
            except subprocess.TimeoutExpired: memory_proc.kill(); memory_proc.wait()
        if memory_log is not None: memory_log.close()

def launch(args):
    if sys.platform != "linux":
        raise RuntimeError("GPU training requires Linux and the pinned verl environment")
    import torch
    import verl
    if not str(verl.__version__).startswith("0.9.1"):
        raise RuntimeError("Expected verl 0.9.1")
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_properties(i).total_memory < 75*1024**3 for i in range(2)):
        raise RuntimeError("Expose exactly two 80GB GPUs using CUDA_VISIBLE_DEVICES")
    manifest = verify_data(args.data)
    run = Path(args.out).resolve()
    run.mkdir(parents=True, exist_ok=True)
    import shutil
    index=Path(manifest['model_path'])/'model.safetensors.index.json'
    if index.exists():
        size=json.loads(index.read_text()).get('metadata',{}).get('total_size',0)
        # BF16 source weights -> FP32 master + two FP32 Adam moments, plus margin.
        if shutil.disk_usage(run).free < int(size)*7:
            raise RuntimeError('Insufficient free disk for one complete full-parameter checkpoint (master weights and Adam states)')
    steps = args.steps or (30 if args.phase == "pilot" else 200)
    study = dict(variant=args.variant, seed=args.seed, steps=steps, gpus=2, run_dir=str(run),
                 training_mode='full', batch_size=args.batch_size)
    if args.initial_checkpoint:
        if args.phase!='pilot': raise ValueError('External checkpoint forks are for engineering pilots; formal runs start from the base model')
        study['initial_checkpoint']=validate_initial_checkpoint(args.initial_checkpoint,study,manifest)
    metadata = dict(study=study, data=manifest, python=sys.version, torch=torch.__version__,
                    verl=verl.__version__, phase=args.phase, training_mode=args.training_mode)
    from importlib.metadata import version,PackageNotFoundError
    kernels={}
    for name in ('fla-core','flash-linear-attention'):
        try: kernels[name]=version(name)
        except PackageNotFoundError: pass
    metadata['kernel_versions']=kernels
    freeze=Path(__file__).resolve().parent.parent/'environment.cu129.freeze.txt'
    if freeze.exists():
        import hashlib
        metadata['environment_freeze_sha256']=hashlib.sha256(freeze.read_bytes()).hexdigest()
    config_overrides = overrides(args.data, run, args.seed, steps, args.training_mode,
        batch_size=args.batch_size, micro_batch_size=args.micro_batch_size,
        fused_kernels=args.fused_kernels, save_freq=args.save_freq or (2 if args.phase=='pilot' else 10),
        val_batch_size=32 if args.phase=='pilot' else 128,initial_validation=not args.skip_initial_validation)
    metadata['config_overrides'] = config_overrides
    if (run/"run.json").exists() and json.loads((run/"run.json").read_text()) != metadata:
        raise ValueError("Existing run has a different protocol; use a new directory")
    atomic_json(run/"run.json", metadata)
    environment = os.environ.copy()
    environment['DAGRPO_DISABLE_RAY_GPU_METRICS']='1'
    root=Path(__file__).resolve().parent.parent
    environment.setdefault('TRITON_CACHE_DIR',str(root/'cache/triton'))
    environment.setdefault('TORCHINDUCTOR_CACHE_DIR',str(root/'cache/torchinductor'))
    environment.setdefault('VLLM_CACHE_ROOT',str(root/'cache/vllm'))
    if kernels: environment['DAGRPO_REQUIRE_FLA']='1'
    compat=str(Path(__file__).resolve().parent.parent/'compat')
    environment['PYTHONPATH']=compat+(os.pathsep+environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    ledger = recover_ledger(json.loads(Path(args.ledger).read_text())) if Path(args.ledger).exists() else {"runs":[]}
    prior = sum(r["seconds"] for r in ledger["runs"] if r["run"] == str(run))
    environment["DAGRPO_STUDY"] = json.dumps({**study,"started_unix":time.time(),"prior_seconds":prior})
    environment["PYTHONHASHSEED"] = str(args.seed)
    if args.stop_after_step:
        environment['DAGRPO_STOP_AFTER_STEP']=str(args.stop_after_step)
    command = [sys.executable, "-m", "dagrpo.train", *config_overrides]
    atomic_json(run/"command.json", command)
    return supervise(command, run, args.budget_gpu_hours, args.ledger, environment,
                     session_seconds=args.session_hours*3600 if args.session_hours else None,
                     grace_seconds=args.checkpoint_grace_minutes*60)
