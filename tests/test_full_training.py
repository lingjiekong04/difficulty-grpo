import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from dagrpo.data import atomic_json
from dagrpo.runner import overrides
from dagrpo.checkpoints import prune_full_checkpoints
from dagrpo.sampler import DifficultySampler


class FullConfigurationTests(unittest.TestCase):
    def test_nvml_workaround_only_affects_dashboard_agent(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        path=Path(__file__).resolve().parent.parent/'compat/sitecustomize.py'
        module=ModuleType('ray.dashboard.modules.reporter.gpu_providers')
        class Provider:
            def get_gpu_usage(self): return ['original']
        module.GpuMetricProvider=Provider
        with patch.dict(sys.modules,{module.__name__:module}),patch.dict('os.environ',{'DAGRPO_DISABLE_RAY_GPU_METRICS':'1'}):
            with patch.object(sys,'argv',['/ray/_private/workers/default_worker.py']): exec(compile(path.read_text(),str(path),'exec'),{})
            self.assertEqual(Provider().get_gpu_usage(),['original'])
            with patch.object(sys,'argv',['/site-packages/ray/dashboard/agent.py']): exec(compile(path.read_text(),str(path),'exec'),{})
            self.assertEqual(Provider().get_gpu_usage(),[])

    def settings(self,**kwargs):
        with tempfile.TemporaryDirectory() as folder:
            atomic_json(Path(folder)/'manifest.json',{'model_path':'/models/qwen35-9b'})
            return overrides(folder,Path(folder)/'run',42,30,**kwargs)

    def test_full_default_has_master_weights_and_native_offload(self):
        rows=dict(row.split('=',1) for row in self.settings())
        self.assertEqual(json.loads(rows['actor_rollout_ref.model.lora_rank']),0)
        self.assertEqual(rows['actor_rollout_ref.model.lora_adapter_path'],'null')
        self.assertEqual(json.loads(rows['actor_rollout_ref.actor.fsdp_config.model_dtype']),'fp32')
        self.assertEqual(rows['actor_rollout_ref.actor.fsdp_config.offload_policy'],'true')
        self.assertEqual(rows['actor_rollout_ref.actor.fsdp_config.optimizer_offload'],'false')
        self.assertEqual(rows['data.train_batch_size'],'64')
        self.assertEqual(rows['actor_rollout_ref.actor.ppo_mini_batch_size'],'64')
        self.assertEqual(rows['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu'],'8')
        self.assertIn('dagrpo.full_engine',rows['actor_rollout_ref.model.external_lib'])
        self.assertEqual(rows['+actor_rollout_ref.actor.fsdp_config.mixed_precision'],
            '{param_dtype:"bf16",reduce_dtype:"fp32"}')

    def test_lora_cannot_enter_new_training_path(self):
        with self.assertRaisesRegex(ValueError,'full-parameter'):
            self.settings(training_mode='lora')

    def test_larger_microbatch_preserves_algorithm_batch(self):
        a=dict(x.split('=',1) for x in self.settings(micro_batch_size=4))
        b=dict(x.split('=',1) for x in self.settings(micro_batch_size=16))
        for key in ('data.train_batch_size','actor_rollout_ref.actor.ppo_mini_batch_size',
                    'actor_rollout_ref.rollout.n','actor_rollout_ref.actor.optim.lr'):
            self.assertEqual(a[key],b[key])
        with self.assertRaises(ValueError): self.settings(micro_batch_size=3)

    def test_large_batch_sampler_resume(self):
        sampler=DifficultySampler([str(i) for i in range(100)],batch_size=64,steps=30)
        iterator=iter(sampler)
        batch=[next(iterator) for _ in range(64)]
        sampler.feedback([(str(i),[0,1,0,1]) for i in batch])
        state=sampler.state_dict()
        expected=[next(iterator) for _ in range(64)]
        fresh=DifficultySampler(sampler.ids,batch_size=64,steps=30)
        fresh.load_state_dict(state)
        fresh_iterator=iter(fresh)
        self.assertEqual(expected,[next(fresh_iterator) for _ in range(64)])
        self.assertEqual(len(fresh),1920)

    def test_checkpoint_retention_keeps_best_and_unknown_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            for step in (2,4,6,8,10):
                atomic_json(Path(folder)/f'global_step_{step}/complete.json',dict(step=step,training_mode='full'))
            atomic_json(Path(folder)/'global_step_1/complete.json',dict(step=1,training_mode='lora'))
            (Path(folder)/'global_step_12').mkdir()
            self.assertEqual(prune_full_checkpoints(folder,best_step=2),[4,6])
            self.assertTrue((Path(folder)/'global_step_2').exists())
            self.assertTrue((Path(folder)/'global_step_1').exists())
            self.assertTrue((Path(folder)/'global_step_12').exists())

    def test_historical_lora_cannot_certify_full_readiness(self):
        from unittest.mock import patch
        from dagrpo import jobs
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            atomic_json(root/'READY.json',{'prior_gpu_smoke_verified':False})
            atomic_json(root/'GPU_SMOKE.json',{'gpu_resume_verified':True})
            with patch.object(jobs,'ROOT',root), self.assertRaisesRegex(RuntimeError,'Full-parameter GPU'):
                jobs.require_full_proof({})

    def test_profile_rejects_lora_and_does_not_invent_timings(self):
        from dagrpo.profile import summarize
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            atomic_json(root/'run.json',{'training_mode':'lora'})
            with self.assertRaisesRegex(ValueError,'historical LoRA'): summarize(root)
            atomic_json(root/'run.json',{'training_mode':'full'})
            result=summarize(root)
            self.assertFalse(any(result['checks'].values()))
            self.assertNotIn('training_only_200_steps_wall_hours',result)

    def test_engineering_resume_requires_full_checkpoint_and_same_algorithm(self):
        from dagrpo.runner import validate_initial_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); checkpoint=root/'checkpoints/global_step_2'
            (checkpoint/'actor').mkdir(parents=True)
            for file in ('study_state.pt','data.pt'): (checkpoint/file).touch()
            study=dict(variant='U',seed=42,steps=30,batch_size=64)
            manifest={'model_revision':'fixed','files':{'train':'sha'}}
            atomic_json(root/'run.json',dict(training_mode='full',study=study,data=manifest))
            atomic_json(checkpoint/'complete.json',dict(training_mode='full',step=2))
            self.assertEqual(validate_initial_checkpoint(checkpoint,study,manifest),str(checkpoint.resolve()))
            with self.assertRaisesRegex(ValueError,'batch_size'):
                validate_initial_checkpoint(checkpoint,{**study,'batch_size':32},manifest)
            atomic_json(checkpoint/'complete.json',dict(training_mode='lora',step=2))
            with self.assertRaisesRegex(ValueError,'full-parameter'):
                validate_initial_checkpoint(checkpoint,study,manifest)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('verl'),
                     'requires server torch and verl')
class FullAuditTests(unittest.TestCase):
    def test_large_embedding_probe_indices_do_not_round_out_of_bounds(self):
        from dagrpo.full_engine import probe_positions
        for length in (1,1024,16_777_216,508_559_360,1_017_118_720):
            positions=probe_positions(length,'cpu')
            self.assertEqual(int(positions[0]),0)
            self.assertEqual(int(positions[-1]),length-1)
            self.assertTrue(bool((positions<length).all()))

    def test_real_base_weights_update_and_optimizer_restores(self):
        import io
        import torch
        from dagrpo.full_engine import trainability,optimizer_coverage,probes,local_tensor
        torch.manual_seed(42)
        model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Tanh(),torch.nn.Linear(8,2)).float()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        audit=trainability(model)
        self.assertEqual(audit['total_parameters'],audit['trainable_parameters'])
        self.assertEqual(optimizer_coverage(model,optimizer),4)
        model(torch.randn(5,4)).square().mean().backward()
        samples=probes(model)
        optimizer.step()
        self.assertTrue(any(torch.any(local_tensor(p.detach()).reshape(-1)[idx].cpu()!=before)
            for _,p,idx,before,_,_ in samples))
        saved=io.BytesIO(); torch.save(optimizer.state_dict(),saved); saved.seek(0)
        restored=torch.optim.AdamW(model.parameters(),lr=1e-3)
        restored.load_state_dict(torch.load(saved,weights_only=True))
        self.assertEqual(len(restored.state),4)
        for p in optimizer.state:
            torch.testing.assert_close(optimizer.state[p]['exp_avg'],restored.state[p]['exp_avg'])
        bad=torch.optim.AdamW(list(model.parameters())[:-1])
        with self.assertRaisesRegex(RuntimeError,'every trainable'): optimizer_coverage(model,bad)
        next(model.parameters()).requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError,'frozen'): trainability(model)


if __name__=='__main__': unittest.main()
