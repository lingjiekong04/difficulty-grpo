import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from dagrpo.data import atomic_json
from dagrpo.locking import exclusive_lock
from dagrpo.runner import recover_ledger,supervise
from dagrpo.checkpoints import latest_complete

class RecoveryTests(unittest.TestCase):
    def test_unlimited_budget_retains_accounting_without_exhaustion(self):
        import math
        from dagrpo.jobs import resource_limits
        self.assertEqual(resource_limits({}),(float('inf'),0))
        self.assertEqual(resource_limits({'GPU_HOURS':'0','SESSION_HOURS':'3.5'}),(float('inf'),3.5))
        with tempfile.TemporaryDirectory() as d:
            ledger=Path(d)/'budget.json'
            atomic_json(ledger,{'gpu_hours':10000,'runs':[]})
            code=supervise([sys.executable,'-c','pass'],d,float('inf'),ledger)
            self.assertEqual(code,0)
            spent=json.loads(ledger.read_text())['gpu_hours']
            self.assertTrue(math.isfinite(spent))
            self.assertGreaterEqual(spent,10000)

    def test_child_waits_for_parent_publication(self):
        with tempfile.TemporaryDirectory() as d:
            lock=Path(d)/'launch.lock'; published=Path(d)/'job.json'
            code='''from dagrpo.locking import exclusive_lock
import sys,json,os
with exclusive_lock(sys.argv[1],blocking=True):
 assert json.load(open(sys.argv[2]))['pid']==os.getpid()
 print('PUBLISHED',flush=True)
'''
            with exclusive_lock(lock):
                child=subprocess.Popen([sys.executable,'-c',code,str(lock),str(published)],
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                atomic_json(published,{'pid':child.pid})
            stdout,stderr=child.communicate(timeout=15)
            self.assertEqual(child.returncode,0,stderr)
            self.assertEqual(stdout.strip(),'PUBLISHED')

    def test_live_lock_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'lock'
            with exclusive_lock(p):
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(p): pass
            with exclusive_lock(p): pass

    def test_killed_process_releases_lock(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'lock'
            code='from dagrpo.locking import exclusive_lock; import sys,time\nwith exclusive_lock(sys.argv[1]):\n print("LOCKED",flush=True)\n time.sleep(30)'
            child=subprocess.Popen([sys.executable,'-c',code,str(p)],stdout=subprocess.PIPE,text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(),'LOCKED')
                child.kill(); child.wait(timeout=5)
                with exclusive_lock(p): pass
            finally:
                if child.poll() is None: child.kill(); child.wait()
                child.stdout.close()

    def test_account_interrupted_attempt_once(self):
        ledger={'gpu_hours':1.2,'runs':[], 'active':{'run':'A','seconds':300,'gpu_hours':1/6,'pid':123}}
        fixed=recover_ledger(ledger)
        self.assertEqual(fixed['gpu_hours'],1.2)
        self.assertEqual(fixed['runs'][0]['status'],'interrupted')
        self.assertEqual(recover_ledger(fixed),fixed)

    def test_partial_checkpoint_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            for step in [25,50,75]:
                p=Path(d)/f'global_step_{step}'
                (p/'actor').mkdir(parents=True)
                (p/'study_state.pt').touch(); (p/'data.pt').touch()
                if step<75: atomic_json(p/'complete.json',{'step':step})
            self.assertEqual(latest_complete(d).name,'global_step_50')
            (Path(d)/'global_step_50/complete.json').write_text('corrupted')
            self.assertEqual(latest_complete(d).name,'global_step_25')

    def test_graceful_session_stop_acknowledged(self):
        with tempfile.TemporaryDirectory() as d:
            code='''import os,json,time,pathlib
p=pathlib.Path(os.environ['DAGRPO_STOP_FILE'])
while not p.exists(): time.sleep(.02)
(p.parent/'saved-checkpoint').touch()
(p.parent/'stop-ack.json').write_text(p.read_text())
'''
            code=supervise([sys.executable,'-c',code],d,1,Path(d)/'budget.json',session_seconds=7.2,grace_seconds=2)
            self.assertEqual(code,2)
            self.assertTrue((Path(d)/'saved-checkpoint').exists())
            self.assertEqual(json.loads((Path(d)/'execution.json').read_text())['status'],'session_stop')

if __name__=='__main__':unittest.main()
