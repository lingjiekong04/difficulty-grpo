from types import SimpleNamespace
import unittest
from dagrpo.evaluate import score_completion

class Tokenizer:
    eos_token_id=99
    def decode(self,ids,skip_special_tokens=True):return '#### 42'

class EvaluationEOSTests(unittest.TestCase):
    def test_stop_reason_without_eos_must_not_receive_correctness_reward(self):
        c=SimpleNamespace(token_ids=[42],text='#### 42',finish_reason='stop')
        _,ended,r=score_completion(c,Tokenizer(),'42')
        self.assertFalse(ended);self.assertEqual(r['score'],0);self.assertEqual(r['truncated'],1)

    def test_complete_numeric_answer_with_eos_is_accepted(self):
        c=SimpleNamespace(token_ids=[42,99],text='#### 42',finish_reason='stop')
        text,ended,r=score_completion(c,Tokenizer(),'42')
        self.assertTrue(ended);self.assertEqual(text,'#### 42');self.assertEqual(r['score'],1)

if __name__=='__main__':unittest.main()
