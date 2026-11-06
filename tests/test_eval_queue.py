import threading
import time
import unittest
from dagrpo.eval_queue import evaluate_plan


class EvaluationQueueTests(unittest.TestCase):
    def plan(self):
        rows=[dict(variant='U',seed=s,step=200,role='final') for s in range(4)]
        return rows+[dict(rows[0],role='historical_best')]

    def test_cold_gate_precedes_parallel_jobs_and_shared_weights_reused(self):
        gate=threading.Event();lock=threading.Lock();active=set();calls=[]
        def evaluate(item, device=None, verify_export=False, public_gate=False):
            if public_gate:
                self.assertEqual(item['seed'],0)
                self.assertTrue(verify_export)
                gate.set()
            else:self.assertTrue(gate.is_set())
            with lock:
                self.assertNotIn(device,active);active.add(device);calls.append(item['seed'])
            time.sleep(.01)
            with lock:active.remove(device)
            return dict(correct=item['seed'])
        saved=[]
        rows=evaluate_plan(self.plan(),evaluate,['0','1'],True,True,saved.append)
        self.assertCountEqual(calls,[0,1,2,3])
        self.assertEqual([r['role'] for r in rows],['final']*4+['historical_best'])
        self.assertEqual(rows[0]['correct'],rows[-1]['correct'])
        self.assertEqual(len(saved[-1]),5)

    def test_failed_cold_gate_prevents_every_other_model(self):
        calls=[]
        def evaluate(item,*args,**kwargs):
            calls.append(item['seed']);raise RuntimeError('cold mismatch')
        with self.assertRaisesRegex(RuntimeError,'cold mismatch'):
            evaluate_plan(self.plan(),evaluate,['0','1'],True,True,lambda rows:None)
        self.assertEqual(calls,[0])
