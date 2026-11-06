"""Ablation semantics at the real warmup boundary and across interruption."""
import itertools
import json
import unittest
from dagrpo.sampler import DifficultySampler


class AblationProtocolTests(unittest.TestCase):
    def sampler(self, variant):
        return DifficultySampler([str(i) for i in range(9)],variant,seed=42,
                                 batch_size=64,steps=200,warmup=20)

    def update(self, sampler, step):
        indices=list(itertools.islice(iter(sampler),64))
        rewards=([0]*4,[1]*4,[0,1,0,1])
        sampler.feedback([(sampler.ids[i],rewards[(i+step)%3]) for i in indices])
        return indices

    def test_real_warmup_is_identical_then_only_intended_statistics_diverge(self):
        samplers=[self.sampler(v) for v in 'ASE']
        for step in range(20):
            batches=[self.update(s,step) for s in samplers]
            self.assertEqual(batches[0],batches[1]);self.assertEqual(batches[0],batches[2])
            self.assertEqual(samplers[0].p,samplers[1].p)
            self.assertEqual(samplers[0].p,samplers[2].p)
        frozen=samplers[1].p.copy()
        for s in samplers:s.feedback([('0',[1]*4)])
        self.assertEqual(samplers[1].p,frozen)
        self.assertNotEqual(samplers[0].p,frozen)
        self.assertEqual(samplers[0].p,samplers[2].p)
        self.assertEqual(samplers[1].visits['0'],samplers[0].visits['0'])
        for s in samplers:s.p=dict.fromkeys(s.ids,0.0);s.p['1']=.5
        self.assertGreater(samplers[0].probabilities()[0],0)
        self.assertGreater(samplers[1].probabilities()[0],0)
        self.assertEqual(samplers[2].probabilities()[0],0)

    def test_both_ablations_restore_exactly_before_and_after_freezing(self):
        for variant in 'SE':
            for boundary in (19,20,21):
                with self.subTest(variant=variant,boundary=boundary):
                    original=self.sampler(variant)
                    for step in range(boundary):self.update(original,step)
                    checkpoint=json.loads(json.dumps(original.state_dict()))
                    restored=self.sampler(variant);restored.load_state_dict(checkpoint)
                    for step in range(boundary,boundary+3):
                        self.assertEqual(self.update(original,step),self.update(restored,step))
                        self.assertEqual(original.state_dict(),restored.state_dict())
