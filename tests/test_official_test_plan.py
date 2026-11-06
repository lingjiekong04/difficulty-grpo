import tempfile,unittest
from pathlib import Path
from dagrpo.test_plan import freeze,validate_splits


class OfficialTestPlanTests(unittest.TestCase):
    def test_cross_split_question_leak_rejected_even_with_distinct_ids(self):
        rows={'train':[dict(problem_id='a',question='same  question')],'dev':[],
              'test':[dict(problem_id='b',question='same question')]}
        with self.assertRaisesRegex(ValueError,'Duplicate'):validate_splits(rows,expected_test=1)

    def test_resumption_cannot_change_frozen_checkpoint_or_data(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan.json';original=dict(checkpoints=['step200'],test_sha256='original')
            first=freeze(p,original);self.assertEqual(freeze(p,original),first)
            for changed in [dict(original,checkpoints=['test_selected_winner']),dict(original,test_sha256='different')]:
                with self.assertRaisesRegex(ValueError,'Frozen'):freeze(p,changed)
