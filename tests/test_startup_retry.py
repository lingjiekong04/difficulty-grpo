import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dagrpo.startup_retry import run_ablation_stage


class StartupRetryTests(unittest.TestCase):
    def setup_run(self,root):
        p=root/'runs/full-full-qwen35-9b-b64-m8-f0-fla060-S-42-save2'
        p.mkdir(parents=True);(root/'logs').mkdir()
        return p

    @patch('dagrpo.startup_retry.wait_gpu_idle')
    def test_fresh_profile_failure_retries_only_once(self,idle):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.setup_run(root);calls=[]
            def run(stage):
                calls.append(stage)
                with (p/'stdout.log').open('a') as f:f.write('Error in memory profiling: current free memory increased\n')
                return 1
            self.assertEqual(run_ablation_stage(run,root),1)
            self.assertEqual(calls,['ablations','ablations']);idle.assert_called_once()

    @patch('dagrpo.startup_retry.wait_gpu_idle')
    def test_old_error_and_post_update_error_do_not_retry(self,idle):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.setup_run(root)
            message='Error in memory profiling: current free memory increased\n'
            (p/'stdout.log').write_text(message)
            calls=[]
            def no_new_error(stage):calls.append(stage);return 1
            self.assertEqual(run_ablation_stage(no_new_error,root),1)
            self.assertEqual(len(calls),1)
            def post_update(stage):
                with (p/'stdout.log').open('a') as f:f.write(message)
                (p/'steps.jsonl').write_text(json.dumps({'metrics':{'training/global_step':1}})+'\n')
                return 1
            self.assertEqual(run_ablation_stage(post_update,root),1)
            idle.assert_not_called()
