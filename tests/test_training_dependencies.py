"""CPU integration checks; enabled in the real training environment."""
import importlib.util
import unittest
from dagrpo.sampler import DifficultySampler

HAS_TORCHDATA=bool(importlib.util.find_spec('torch')) and bool(importlib.util.find_spec('torchdata'))
HAS_VERL=bool(importlib.util.find_spec('torch')) and bool(importlib.util.find_spec('verl'))

@unittest.skipUnless(HAS_TORCHDATA,'requires the server training environment')
class DataLoaderIntegrationTests(unittest.TestCase):
    def test_stateful_loader_restores_sampler_and_rng(self):
        from torchdata.stateful_dataloader import StatefulDataLoader
        ids=[str(i) for i in range(9)]
        sampler=DifficultySampler(ids,batch_size=4,steps=6,warmup=0)
        loader=StatefulDataLoader(list(range(9)),sampler=sampler,batch_size=4,num_workers=0)
        it=iter(loader)
        first=next(it).tolist()
        sampler.feedback([(ids[i],[1,0,1,0]) for i in first])
        next(it)
        state=loader.state_dict()
        expected=next(it).tolist()
        restored=DifficultySampler(ids,batch_size=4,steps=6,warmup=0)
        fresh=StatefulDataLoader(list(range(9)),sampler=restored,batch_size=4,num_workers=0)
        fresh.load_state_dict(state)
        self.assertEqual(next(iter(fresh)).tolist(),expected)
        self.assertEqual(restored.completed,1)

@unittest.skipUnless(HAS_VERL,'requires verl and torch')
class VerlIntegrationTests(unittest.TestCase):
    def test_full_configuration_composes_against_installed_verl(self):
        import json
        from pathlib import Path
        import tempfile
        import verl
        from hydra import initialize_config_dir,compose
        from dagrpo.runner import overrides
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'manifest.json').write_text(json.dumps({'model_path':'/model/Qwen3.5-9B'}))
            for micro in (4,8,16):
                with initialize_config_dir(config_dir=str(Path(verl.__file__).parent/'trainer/config'),version_base=None):
                    config=compose(config_name='ppo_trainer',overrides=overrides(folder,Path(folder)/'run',42,30,micro_batch_size=micro))
                self.assertTrue(config.actor_rollout_ref.actor.fsdp_config.offload_policy)
                self.assertEqual(config.actor_rollout_ref.model.lora_rank,0)
                self.assertEqual(config.actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype,'bf16')

    def test_concurrent_rewards_isolate_shared_question_metadata(self):
        import asyncio
        import threading
        import numpy as np
        import torch
        from verl import DataProto
        from dagrpo.verl_reward import StrictNumericRewardManager
        from dagrpo.reward import compute_score
        shared = {'problem_id': 'same-question'}
        barrier = threading.Barrier(2)

        class Tokenizer:
            eos_token_id = 99
            def decode(self, ids, **kwargs):
                barrier.wait(timeout=10)
                return '#### 42'

        async def run():
            manager = StrictNumericRewardManager.__new__(StrictNumericRewardManager)
            manager.tokenizer = Tokenizer()
            manager.loop = asyncio.get_running_loop()
            manager.compute_score = compute_score
            manager.is_async_reward_score = False
            manager.reward_router_address = None
            manager.reward_model_tokenizer = None
            data = DataProto.from_dict(tensors={
                'responses': torch.tensor([[42,99],[42,1]]),
                'attention_mask': torch.ones(2,4,dtype=torch.long)}, non_tensors={
                'data_source': np.array(['dagrpo/gsm8k']*2,dtype=object),
                'reward_model': np.array([{'ground_truth':'42'}]*2,dtype=object),
                'extra_info': np.array([shared,shared],dtype=object)})
            return await asyncio.gather(manager.run_single(data[:1]),manager.run_single(data[1:]))

        outputs = asyncio.run(run())
        self.assertEqual([o['reward_score'] for o in outputs],[1.,0.])
        self.assertEqual(shared,{'problem_id':'same-question'})

    def test_actual_ray_entrypoint_can_bind_trainer(self):
        from verl.trainer.main_ppo_v0 import TaskRunner
        from dagrpo.train import bind_trainer
        from unittest.mock import Mock
        sentinel = Mock()
        bound = bind_trainer(TaskRunner.__ray_metadata__.modified_class.run, sentinel)
        self.assertIs(bound.__globals__['RayPPOTrainer'], sentinel)
        self.assertIn('init_workers', bound.__code__.co_names)
        self.assertIn('fit', bound.__code__.co_names)

    def test_actual_grpo_constant_rewards(self):
        import numpy as np
        import torch
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
        for value in (0.,1.):
            adv,_=compute_grpo_outcome_advantage(torch.full((8,1),value),torch.ones(8,1),np.array(['a']*4+['b']*4))
            self.assertTrue(torch.equal(adv,torch.zeros_like(adv)))

    def test_actual_grpo_mixed_group_has_signal(self):
        import numpy as np
        import torch
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
        adv,_=compute_grpo_outcome_advantage(torch.tensor([[1.],[0.],[1.],[0.]]),torch.ones(4,1),np.array(['x']*4))
        self.assertGreater(float(adv[0,0]),0)
        self.assertLess(float(adv[1,0]),0)
