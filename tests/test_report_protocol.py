import json
from pathlib import Path
import tempfile
import unittest
from dagrpo.data import atomic_json
from dagrpo.report import generate,protocol_id

class ReportProtocolTests(unittest.TestCase):
    def fixture(self,root,name,mode='full',phase='full',seed=42,lr='1e-6'):
        run=Path(root)/name; run.mkdir()
        metadata=dict(training_mode=mode,phase=phase,data={'model_revision':'fixed'},
            config_overrides=[f'data.seed={seed}',f'trainer.experiment_name="{name}"',f'actor_rollout_ref.actor.optim.lr={lr}'])
        atomic_json(run/'run.json',metadata)
        (run/'evaluations.jsonl').write_text(json.dumps(dict(step=200,variant='A',seed=seed,
            accuracy=.5,tokens=100,gpu_hours=1))+'\n')
        return metadata

    def test_report_cannot_mix_lora_pilots_or_distinct_protocols(self):
        with tempfile.TemporaryDirectory() as root:
            a=self.fixture(root,'a',seed=42)
            b=self.fixture(root,'b',seed=43)
            c=self.fixture(root,'c',seed=44,lr='2e-6')
            self.fixture(root,'lora',mode='lora')
            self.fixture(root,'pilot',phase='pilot')
            self.assertEqual(protocol_id(a),protocol_id(b))
            self.assertNotEqual(protocol_id(a),protocol_id(c))
            out=Path(root)/'report'
            generate(root,out,plots=False)
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(len(summary),3)
            self.assertEqual(len(json.loads((out/'excluded-runs.json').read_text())),2)
            text=(out/'report.md').read_text()
            self.assertIn('n=2, mean=0.5000',text)
            self.assertIn('n=1, mean=0.5000',text)

    def test_duplicate_seeds_are_not_counted_as_independent_replicates(self):
        with tempfile.TemporaryDirectory() as root:
            self.fixture(root,'a',seed=42)
            self.fixture(root,'b',seed=42)
            out=Path(root)/'report'; generate(root,out,plots=False)
            self.assertIn('duplicate seeds; aggregate withheld',(out/'report.md').read_text())

if __name__=='__main__': unittest.main()
