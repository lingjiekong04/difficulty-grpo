"""Pinned legacy synchronous verl adapter. GPU imports are deliberately lazy."""
import json
import os
from pathlib import Path
import random
import time
from .sampler import DifficultySampler, group_feedback

class GracefulStop(Exception):
    """A checkpoint was committed at a synchronous update boundary."""

def bind_trainer(entrypoint, trainer_class):
    """Bind the trainer even when Ray serializes a detached globals dictionary.

    Changing the imported module's class does not affect that serialized method.
    Unwrap Ray's tracing decorator and bind the dependency in the method we call.
    """
    import inspect
    from types import FunctionType
    implementation = inspect.unwrap(entrypoint)
    if 'RayPPOTrainer' not in implementation.__code__.co_names:
        raise RuntimeError('Unsupported verl TaskRunner entrypoint: trainer binding missing')
    namespace = {**implementation.__globals__, 'RayPPOTrainer': trainer_class}
    bound = FunctionType(implementation.__code__, namespace, implementation.__name__,
                         implementation.__defaults__, implementation.__closure__)
    bound.__kwdefaults__ = implementation.__kwdefaults__
    return bound


def install_adapter(study):
    import numpy as np
    import torch
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from .data import atomic_json

    class ResearchTrainer(RayPPOTrainer):
        def init_workers(self):
            result=super().init_workers()
            original=self.checkpoint_manager.update_weights
            def synced(*args,**kwargs):
                value=original(*args,**kwargs)
                stop=Path(os.environ.get('DAGRPO_STOP_FILE',str(Path(study['run_dir'])/'stop-request.json')))
                # During training, stop after metrics are logged, so profiling retains
                # the final update. Initial synchronization has no update metrics.
                if stop.exists() and self.study_sampler.completed == 0:
                    if self.global_steps > 0:
                        self._save_checkpoint()
                    try: reason=json.loads(stop.read_text()).get('reason','session_stop')
                    except (ValueError,OSError): reason='session_stop'
                    atomic_json(Path(study['run_dir'])/'stop-ack.json',dict(reason=reason,step=self.global_steps))
                    raise GracefulStop(reason)
                return value
            self.checkpoint_manager.update_weights=synced
            return result

        def study_seconds(self):
            return study.get("prior_seconds", 0) + time.time()-study["started_unix"]

        def fit(self):
            from verl.utils.tracking import Tracking
            original_log = Tracking.log
            owner = self
            def log(logger, data, step, *args, **kwargs):
                record = dict(step=step, pid=os.getpid(), seconds=owner.study_seconds(),
                    gpu_hours=owner.study_seconds()*study["gpus"]/3600,
                    tokens=owner.study_tokens, metrics={k:float(v) for k,v in data.items()})
                with (Path(study["run_dir"])/"steps.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record)+"\n")
                result=original_log(logger, data, step, *args, **kwargs)
                stop=Path(os.environ.get('DAGRPO_STOP_FILE',str(Path(study['run_dir'])/'stop-request.json')))
                boundary=int(os.environ.get('DAGRPO_STOP_AFTER_STEP','0'))
                if 'training/global_step' in data and (stop.exists() or (boundary and step>=boundary)):
                    folder=Path(owner.config.trainer.default_local_dir)/f'global_step_{step}'
                    if not (folder/'complete.json').exists(): owner._save_checkpoint()
                    reason=json.loads(stop.read_text()).get('reason','session_stop') if stop.exists() else 'profile_stop'
                    atomic_json(Path(study['run_dir'])/'stop-ack.json',dict(reason=reason,step=step))
                    raise GracefulStop(reason)
                return result
            Tracking.log = log
            try:
                return super().fit()
            finally:
                Tracking.log = original_log

        def __init__(self, *args, **kwargs):
            from functools import partial
            # Separate rendezvous ports from Ray's ephemeral listener ports.
            kwargs['ray_worker_group_cls'] = partial(kwargs['ray_worker_group_cls'],
                master_port_range=list(kwargs['config'].trainer.study_master_port_range))
            dataset = kwargs["train_dataset"]
            ids = [dataset[i]["extra_info"]["problem_id"] for i in range(len(dataset))]
            self.study_sampler = DifficultySampler(ids, study["variant"], study["seed"],
                steps=study["steps"], batch_size=int(kwargs['config'].data.train_batch_size))
            kwargs["train_sampler"] = self.study_sampler
            self.study_tokens = 0
            self.study_started = time.monotonic()
            self.study_previous_seconds = 0
            self.study_metrics = {}
            self.study_checkpoint_seconds = 0
            self.study_best_accuracy = -1
            self.study_best_step = 0
            print(f"STUDY_ADAPTER_ACTIVE variant={study['variant']} seed={study['seed']} "
                  f"pool={len(ids)} sampler={type(self.study_sampler).__name__}", flush=True)
            super().__init__(*args, **kwargs)
            if self.config.data.dataloader_num_workers != 0:
                raise ValueError("Online sampling requires num_workers=0")

        def _update_actor(self, batch):
            # The superclass assigns a unique uid per sampled occurrence BEFORE repeat.
            # Group by uid, never by question ID or adjacent rows (balancing reorders them).
            ids = [info["problem_id"] for info in batch.non_tensor_batch["extra_info"]]
            rewards = batch.batch["token_level_scores"].sum(-1).detach().cpu().tolist()
            groups = group_feedback(ids, batch.non_tensor_batch["uid"], rewards)
            result = super()._update_actor(batch)
            self.study_sampler.feedback(groups)
            self.study_tokens += int(batch.batch["response_mask"].sum().item())
            means = [sum(rs)/len(rs) for _, rs in groups]
            self.study_metrics = {
                "mixed_group_rate": sum(0 < x < 1 for x in means)/len(means),
                "all_correct_rate": means.count(1)/len(means),
                "all_wrong_rate": means.count(0)/len(means),
                "coverage": sum(v > 0 for v in self.study_sampler.visits.values())/len(self.study_sampler.ids),
                "mean_response_tokens": float(batch.batch["response_mask"].sum(-1).float().mean().item())}
            result.meta_info["metrics"].update({"study/"+k:v for k,v in self.study_metrics.items()})
            return result

        def _save_checkpoint(self):
            folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
            folder.mkdir(parents=True, exist_ok=True)
            # Write sidecar before upstream publishes latest_checkpointed_iteration.txt.
            state = dict(sampler=self.study_sampler.state_dict(), tokens=self.study_tokens,
                seconds=self.study_previous_seconds + time.monotonic()-self.study_started,
                best_accuracy=self.study_best_accuracy,best_step=self.study_best_step,
                python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
            tmp = folder / "study_state.tmp"
            torch.save(state, tmp)
            tmp.replace(folder / "study_state.pt")
            if (folder/'complete.json').exists():
                # A dev-best update at a periodic save boundary only changes the
                # sidecar; model and Adam shards at this update are already durable.
                return None
            started=time.monotonic()
            result=super()._save_checkpoint()
            self.study_checkpoint_seconds=time.monotonic()-started
            # Resume only checkpoints whose model, optimizer and sampler all finished saving.
            atomic_json(folder/'complete.json',dict(step=self.global_steps,training_mode=study['training_mode']))
            with (Path(study['run_dir'])/'checkpoint-timings.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=self.global_steps,seconds=self.study_checkpoint_seconds))+"\n")
            from .checkpoints import prune_full_checkpoints
            prune_full_checkpoints(folder.parent,keep=2,best_step=self.study_best_step)
            return result

        def _load_checkpoint(self):
            if self.config.trainer.resume_mode == 'auto':
                from omegaconf import open_dict
                from .checkpoints import latest_complete
                root=Path(self.config.trainer.default_local_dir)
                complete=latest_complete(root)
                if complete is None and study.get('initial_checkpoint'):
                    complete=Path(study['initial_checkpoint'])
                with open_dict(self.config):
                    if complete:
                        self.config.trainer.resume_mode='resume_path'
                        self.config.trainer.resume_from_path=str(complete.resolve())
                    else:
                        self.config.trainer.resume_mode='disable'
            result = super()._load_checkpoint()
            if self.global_steps:
                folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
                if self.config.trainer.resume_mode == "resume_path":
                    folder = Path(self.config.trainer.resume_from_path)
                state = torch.load(folder / "study_state.pt", map_location="cpu", weights_only=False)
                self.study_sampler.load_state_dict(state["sampler"])
                self.study_tokens = state["tokens"]
                self.study_previous_seconds = state["seconds"]
                self.study_best_accuracy=state.get('best_accuracy',-1)
                self.study_best_step=state.get('best_step',0)
                random.setstate(state["python"])
                np.random.set_state(state["numpy"])
                torch.set_rng_state(state["torch"])
                if self.study_sampler.state_dict() != state['sampler']:
                    raise RuntimeError('Sampler state did not restore exactly')
                if random.getstate() != state['python'] or not torch.equal(torch.get_rng_state(),state['torch']):
                    raise RuntimeError('Driver RNG state did not restore exactly')
                restored_numpy=np.random.get_state()
                if not np.array_equal(restored_numpy[1],state['numpy'][1]) or restored_numpy[2:] != state['numpy'][2:]:
                    raise RuntimeError('NumPy RNG state did not restore exactly')
                print(f"STUDY_RESTORED step={self.global_steps} "
                      f"completed={self.study_sampler.completed} position={self.study_sampler.position} "
                      "sampler_and_driver_rng_exact=True",flush=True)
                # Initial dev evaluation already exists for this run. Avoid
                # repeating 512 greedy generations on every four-hour restart.
                if (Path(study['run_dir'])/'evaluations.jsonl').exists() or study.get('initial_checkpoint'):
                    from omegaconf import open_dict
                    with open_dict(self.config): self.config.trainer.val_before_train=False
                    print('RESUME_SKIPS_REPEATED_INITIAL_DEV',flush=True)
            return result

        def _validate(self):
            metrics = super()._validate()
            # Reward-loop output includes acc; fail rather than silently picking another metric.
            keys = [k for k in metrics if "acc" in k and "mean@1" in k]
            if not keys:
                raise RuntimeError(f"No greedy accuracy metric found: {list(metrics)}")
            accuracy = float(metrics[keys[0]])
            if accuracy > self.study_best_accuracy:
                self.study_best_accuracy=accuracy
                self.study_best_step=self.global_steps if self.study_sampler.completed else 0
                atomic_json(Path(study['run_dir'])/'best-dev.json',
                    dict(step=self.study_best_step,accuracy=accuracy,training_mode=study['training_mode']))
                if self.study_sampler.completed:
                    self._save_checkpoint()
            seconds = self.study_seconds()
            record = dict(step=self.global_steps, variant=study["variant"], seed=study["seed"],
                accuracy=accuracy, tokens=self.study_tokens, gpu_hours=seconds*study["gpus"]/3600,
                seconds=seconds, training_mode=study['training_mode'], batch_size=study['batch_size'],
                metrics={k:float(v) for k,v in metrics.items()},
                sampling=self.study_metrics, provenance="real_training")
            path = Path(study["run_dir"]) / "evaluations.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record)+"\n")
            return metrics

    return ResearchTrainer

def main():
    import hydra
    import ray
    import verl
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.main_ppo_v0 import TaskRunner
    study = json.loads(os.environ["DAGRPO_STUDY"])
    import signal
    def terminate(signum, frame):
        raise KeyboardInterrupt("GPU-hour deadline reached")
    signal.signal(signal.SIGTERM, terminate)
    # Install inside the Ray TaskRunner actor, not just the launcher process.
    original = TaskRunner.__ray_metadata__.modified_class
    @ray.remote
    class StudyRunner(original):
        def run(self, config):
            random.seed(study["seed"])
            import numpy as np
            import torch
            np.random.seed(study["seed"])
            torch.manual_seed(study["seed"])
            trainer_class = install_adapter(study)
            run = bind_trainer(original.run, trainer_class)
            try:
                return run(self, config)
            except GracefulStop as exc:
                return {'stopped':str(exc)}
    config_dir = str(Path(verl.__file__).resolve().parent / "trainer" / "config")
    @hydra.main(config_path=config_dir, config_name="ppo_trainer", version_base=None)
    def start(config):
        try:
            run_ppo(config, task_runner_class=StudyRunner)
        finally:
            ray.shutdown()
    start()

if __name__ == "__main__":
    main()
