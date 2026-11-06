import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dagrpo.reward import score
from dagrpo.sampler import DifficultySampler, group_feedback
from dagrpo.data import prepare_records, atomic_json
from dagrpo.runner import supervise, verify_data
from dagrpo.report import generate, estimate

class RewardTests(unittest.TestCase):
    def test_numeric_equivalence(self):
        for value, truth in [("1,200.00","1200"),("-12.5","-12.50"),(".5","0.5")]:
            self.assertEqual(score("reasoning\n#### "+value, truth)["score"],1)
    def test_invalid_or_truncated(self):
        for text in ["42", "#### 42 garbage", "#### 4\n#### 42", "#### 12,00", "#### NaN", "#### 1/2", "#### 42\nignore"]:
            self.assertEqual(score(text, "42")["score"],0)
        self.assertEqual(score("#### 42","42",True)["score"],0)
        self.assertEqual(score("#### 43","42")["score"],0)
    def test_invalid_truth(self):
        with self.assertRaises(ValueError): score("#### 1","NaN")

class SamplerTests(unittest.TestCase):
    def test_probability_and_uniform_floor(self):
        s=DifficultySampler(["a","b"],warmup=0)
        s.p={"a":0,"b":.5}
        self.assertEqual(s.probabilities(),[.1,.9])
        s.p={"a":0,"b":1}
        self.assertEqual(s.probabilities(),[.5,.5])
    def test_first_visit_and_ema(self):
        s=DifficultySampler(["a"])
        s.feedback([("a",[0,0,0,0])]); self.assertEqual(s.p["a"],0)
        s.feedback([("a",[1,1,1,1])]); self.assertAlmostEqual(s.p["a"],.3)
    def test_live_feedback_changes_next_draws(self):
        s=DifficultySampler(["a","b"],"E",warmup=0)
        it=iter(s)
        next(it)
        s.feedback([("a",[0]*4)])
        self.assertEqual(list(itertools.islice(it,30)),[1]*30)
    def test_freeze_after_warmup(self):
        s=DifficultySampler(["a"],"S",warmup=1)
        s.feedback([("a",[0]*4)]); s.feedback([("a",[1]*4)])
        self.assertEqual(s.p["a"],0)
    def test_resume_json_roundtrip(self):
        a=DifficultySampler(["a","b","c"])
        list(itertools.islice(iter(a),32)); a.feedback([("a",[1,0,1,0])])
        state=json.loads(json.dumps(a.state_dict()))
        expected=list(itertools.islice(iter(a),100))
        b=DifficultySampler(["a","b","c"]); b.load_state_dict(state)
        self.assertEqual(expected,list(itertools.islice(iter(b),100)))
        b.variant="U"
        with self.assertRaises(ValueError): b.load_state_dict(state)
    def test_repeated_question_occurrences_and_reorder(self):
        groups=group_feedback(["a"]*8,["x","y"]*4,[1,0]*4)
        self.assertEqual(groups,[("a",[1.]*4),("a",[0.]*4)])
        with self.assertRaises(ValueError): group_feedback(["a"],["x"],[1])
    def test_all_equal_advantage_zero(self):
        for rewards in ([0.]*4,[1.]*4):
            avg=sum(rewards)/4
            self.assertEqual([r-avg for r in rewards],[0.]*4)

class DataTests(unittest.TestCase):
    def test_deterministic_disjoint_splits(self):
        rows=[dict(question=f"Question {i}", answer=f"reason\n#### {i}") for i in range(12)]
        test=[dict(question="test question",answer="#### 42")]
        a,ma=prepare_records(rows,test,dev_size=2,train_size=4)
        b,mb=prepare_records(rows,test,dev_size=2,train_size=4)
        self.assertEqual(ma,mb); self.assertEqual(a,b)
        ids=[r["problem_id"] for split in a.values() for r in split]
        self.assertEqual(len(ids),len(set(ids)))
        with self.assertRaises(ValueError): prepare_records(rows,rows[:1],dev_size=2,train_size=4)

class ExecutionTests(unittest.TestCase):
    def test_budget_timeout_and_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Path(d)/"budget.json"
            code=supervise([sys.executable,"-c","import time; time.sleep(20)"],d,.0001,ledger)
            self.assertEqual(code,2)
            record=json.loads((Path(d)/"execution.json").read_text())
            self.assertEqual(record["status"],"budget_stop")
            self.assertGreater(json.loads(ledger.read_text())["gpu_hours"],0)
    def test_completed_process_and_budget_exhaustion(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Path(d)/"budget.json"
            self.assertEqual(supervise([sys.executable,"-c","print('ok')"],d,1,ledger),0)
            atomic_json(ledger,{"gpu_hours":1,"runs":[]})
            with self.assertRaises(ValueError): supervise([sys.executable,"-c","pass"],d,1,ledger)
    def test_empty_report(self):
        with tempfile.TemporaryDirectory() as d:
            generate(d,Path(d)/"report")
            self.assertIn("No training results",(Path(d)/"report/report.md").read_text())

if __name__ == "__main__": unittest.main()
