"""CPU demonstration with SYNTHETIC rewards, no model or dataset loading."""
from pathlib import Path
import sys, json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dagrpo.sampler import DifficultySampler, group_feedback
from dagrpo.reward import score

def main():
    sampler=DifficultySampler(['easy','medium','hard'],seed=42,batch_size=4,steps=3,warmup=0)
    # Human-written outcomes, not predictions from a trained LLM.
    sampler.feedback([('easy',[1,1,1,1]),('medium',[1,0,1,0]),('hard',[0,0,0,0])])
    snapshot=json.loads(json.dumps(sampler.state_dict()))
    expected=[next(iter(sampler)) for _ in range(4)]
    restored=DifficultySampler(['easy','medium','hard'],seed=42,batch_size=4,steps=3,warmup=0)
    restored.load_state_dict(snapshot)
    observed=[next(iter(restored)) for _ in range(4)]
    assert expected==observed
    print(json.dumps(dict(scope='SYNTHETIC CPU demonstration; no model inference',probabilities=dict(zip(restored.ids,restored.probabilities())),next_indices=observed,resume_exact=expected==observed,numeric_reward=score('synthetic reasoning\n#### 1,200.00','1200'),duplicate_question_groups=group_feedback(['same']*8,['x','y']*4,[1,0]*4)),indent=2))

if __name__=='__main__':main()
