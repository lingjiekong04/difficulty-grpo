import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from dagrpo.data import atomic_json
from dagrpo.stages import planned_runs,run_directory,run_complete
from dagrpo.analysis import budget_snapshot


def finished(root,variant,seed,env):
    r=run_directory(root,'full',variant,seed,env)
    atomic_json(r/'run.json',dict(training_mode='full',phase='full',
        study=dict(variant=variant,seed=seed,steps=200)))
    atomic_json(r/'execution.json',dict(status='completed',exit_code=0))
    (r/'evaluations.jsonl').write_text(json.dumps(dict(step=200,variant=variant,seed=seed))+'\n')
    c=r/'checkpoints/global_step_200'
    atomic_json(c/'complete.json',dict(step=200,training_mode='full'))
    for f in ['data.pt','study_state.pt','actor/fsdp_config.json']+[
        f'actor/{kind}_world_size_2_rank_{rank}.pt' for kind in ['model','optim','extra_state'] for rank in range(2)]:
        p=c/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'x')
    return r


class StageTests(unittest.TestCase):
    def test_repeat_order_and_ablations_are_separate(self):
        self.assertEqual(planned_runs('matrix',{'EXPERIMENT_STAGE':'repeats'}),[('U',43),('A',43),('U',44),('A',44)])
        self.assertEqual(planned_runs('matrix',{'EXPERIMENT_STAGE':'ablations'}),[('S',42),('E',42)])
        with self.assertRaises(ValueError):planned_runs('matrix',{'EXPERIMENT_STAGE':'typo'})

    def test_skip_requires_final_checkpoint_optimizer_and_successful_exit(self):
        with tempfile.TemporaryDirectory() as root:
            r=finished(root,'U',43,{'SAVE_FREQ':'2'})
            self.assertTrue(run_complete(r,'U',43))
            self.assertFalse(run_complete(r,'U',44))
            optimizer=r/'checkpoints/global_step_200/actor/optim_world_size_2_rank_1.pt'
            optimizer.unlink();self.assertFalse(run_complete(r,'U',43))
            optimizer.write_bytes(b'x')
            atomic_json(r/'execution.json',dict(status='failed',exit_code=1))
            self.assertFalse(run_complete(r,'U',43))
            atomic_json(r/'execution.json',dict(status='completed',exit_code=0))
            (r/'checkpoints/global_step_200/complete.json').write_text('{broken')
            self.assertFalse(run_complete(r,'U',43))

    def test_queue_stops_on_interruption_and_next_invocation_skips_only_completed(self):
        from dagrpo import jobs
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);env={'SAVE_FREQ':'2','EXPERIMENT_STAGE':'repeats'}
            (root/'logs').mkdir()
            finished(root,'U',43,env)
            attempts=[];interrupt=[True]
            def child(argv,**kwargs):
                if 'dagrpo.cli' not in argv:return SimpleNamespace(returncode=0)
                v=argv[argv.index('--variant')+1];s=int(argv[argv.index('--seed')+1])
                attempts.append((v,s))
                if interrupt[0]:interrupt[0]=False;return SimpleNamespace(returncode=2)
                finished(root,v,s,env)
                return SimpleNamespace(returncode=0)
            with patch.object(jobs,'ROOT',root),patch.dict(os.environ,env,clear=True),\
                 patch('dagrpo.runtime.python_runtime',return_value='python'),\
                 patch.object(jobs.os,'chdir'),patch.object(jobs,'require_full_proof'),patch.object(jobs,'validate_stage',return_value={}),\
                 patch.object(jobs.subprocess,'run',side_effect=child):
                self.assertEqual(jobs.work('matrix'),2)
                self.assertEqual(attempts,[('A',43)])
                self.assertEqual(jobs.work('matrix'),0)
                self.assertEqual(attempts,[('A',43),('A',43),('U',44),('A',44)])
                self.assertEqual(jobs.work('matrix'),0)
                self.assertEqual(len(attempts),4)

    def test_failed_run_cannot_fall_through_to_next_seed(self):
        from dagrpo import jobs
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);(root/'logs').mkdir();attempts=[]
            def child(argv,**kwargs):
                if 'dagrpo.cli' in argv:attempts.append(argv);return SimpleNamespace(returncode=1)
                return SimpleNamespace(returncode=0)
            with patch.object(jobs,'ROOT',root),patch.dict(os.environ,{'EXPERIMENT_STAGE':'repeats'},clear=True),\
                 patch('dagrpo.runtime.python_runtime',return_value='python'),\
                 patch.object(jobs.os,'chdir'),patch.object(jobs,'require_full_proof'),patch.object(jobs,'validate_stage',return_value={}),\
                 patch.object(jobs.subprocess,'run',side_effect=child):
                self.assertEqual(jobs.work('matrix'),1)
                self.assertEqual(len(attempts),1)

    def test_budget_snapshot_does_not_use_future_or_best_observation(self):
        rows=[dict(step=0,accuracy=.9,tokens=0),dict(step=25,accuracy=.6,tokens=100),dict(step=50,accuracy=1,tokens=201)]
        self.assertEqual(budget_snapshot(rows,'tokens',200)['step'],25)
        self.assertEqual(budget_snapshot(rows,'tokens',200)['accuracy'],.6)
        self.assertEqual(budget_snapshot(rows,'tokens',200)['unused_budget'],100)
        self.assertIsNone(budget_snapshot(rows,'tokens',-1))

    def test_successful_child_without_final_evidence_is_failure(self):
        from dagrpo import jobs
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);(root/'logs').mkdir();attempts=[]
            def child(argv,**kwargs):
                if 'dagrpo.cli' in argv:attempts.append(argv)
                return SimpleNamespace(returncode=0)
            with patch.object(jobs,'ROOT',root),patch.dict(os.environ,{'EXPERIMENT_STAGE':'repeats'},clear=True),\
                 patch('dagrpo.runtime.python_runtime',return_value='python'),\
                 patch.object(jobs.os,'chdir'),patch.object(jobs,'require_full_proof'),\
                 patch.object(jobs,'validate_stage',return_value={}),\
                 patch.object(jobs.subprocess,'run',side_effect=child):
                self.assertEqual(jobs.work('matrix'),1)
                self.assertEqual(len(attempts),1)
                self.assertEqual(json.loads((root/'logs/matrix-job.json').read_text())['status'],'failed')


if __name__=='__main__':unittest.main()
