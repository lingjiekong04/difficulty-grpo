"""Development-only reproducibility audit, with raw tokens and termination evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import os
from .data import PROMPT,atomic_json,digest
from .reward import score,FINAL,number


def compare(a,b):
    if [r['problem_id'] for r in a]!=[r['problem_id'] for r in b]:
        raise ValueError('Different problem ordering')
    fields=['prompt_sha256','token_ids','text','finish_reason','stop_reason','score','format_error','truncated']
    return {k:[r['problem_id'] for r,s in zip(a,b) if r.get(k)!=s.get(k)] for k in fields}


def run(model,data,out,seed,repeats):
    from vllm import LLM,SamplingParams
    from importlib.metadata import version
    import torch
    reference=os.environ.get('DAGRPO_EVAL_REFERENCE')=='1'
    if reference:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(x) for x in Path(data).read_text().splitlines()]
    assert len(rows)==512,'Audit must use the fixed development set only'
    settings=dict(tensor_parallel_size=2,dtype='bfloat16',max_model_len=1536,
        gpu_memory_utilization=.6,max_num_seqs=128,seed=seed,enable_prefix_caching=False)
    if os.environ.get('DAGRPO_EVAL_SYNC')=='1':settings['async_scheduling']=False
    if reference:settings.update(tensor_parallel_size=1,enforce_eager=True,disable_custom_all_reduce=True)
    atomic_json(out/'protocol.json',dict(model=model,data_sha256=digest(rows),seed=seed,
        repeats=repeats,engine=settings,temperature=0,top_p=1,top_k=-1,max_tokens=1024,
        chunk_size=128,enable_thinking=False,torch=torch.__version__,vllm=version('vllm'),
        scope='Controlled offline diagnostic; differs from historical asynchronous verl validation.'))
    engine=LLM(model=model,**settings)
    tok=engine.get_tokenizer()
    eos=tok.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    prompts=[tok.apply_chat_template([{'role':'user','content':PROMPT+r['question']}],
        tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False) for r in rows]
    assert max(map(len,prompts))<=512
    atomic_json(out/'prompts.json',dict(eos_token_ids=sorted(eos),
        problems=[dict(problem_id=r['problem_id'],token_ids=p) for r,p in zip(rows,prompts)]))
    params=SamplingParams(temperature=0,top_p=1,top_k=-1,max_tokens=1024,n=1)
    summaries=[]
    for rep in range(repeats):
        records=[];started=time.monotonic();path=out/f'repeat-{rep}.jsonl'
        with path.open('x',encoding='utf-8') as f:
            for lo in range(0,len(rows),128):
                outputs=engine.generate([{'prompt_token_ids':p} for p in prompts[lo:lo+128]],params,use_tqdm=False)
                for row,prompt,output in zip(rows[lo:lo+128],prompts[lo:lo+128],outputs,strict=True):
                    c=output.outputs[0];ids=list(c.token_ids)
                    text=tok.decode(ids,skip_special_tokens=True)
                    ended=bool(ids) and ids[-1] in eos
                    strict=score(text,row['truth'],not ended)
                    legacy=score(c.text,row['truth'],c.finish_reason=='length')
                    match=FINAL.search(text.strip())
                    record=dict(problem_id=row['problem_id'],truth=row['truth'],
                        prompt_sha256=digest(prompt),token_ids=ids,text=text,engine_text=c.text,
                        finish_reason=c.finish_reason,stop_reason=c.stop_reason,ended_with_eos=ended,
                        parsed_answer=str(number(match.group(1))) if match else None,
                        legacy_score=legacy['score'],**strict)
                    assert score(text,row['truth'],not ended)==strict
                    records.append(record);f.write(json.dumps(record,ensure_ascii=False)+'\n')
                f.flush()
                print(f'AUDIT_PROGRESS seed={seed} repeat={rep} completed={len(records)}/512',flush=True)
        summary=dict(repeat=rep,count=len(records),accuracy=sum(r['score'] for r in records)/len(records),
            format_error=sum(r['format_error'] for r in records)/len(records),
            truncated=sum(r['truncated'] for r in records)/len(records),
            legacy_disagreements=sum(r['score']!=r['legacy_score'] for r in records),
            generated_tokens=sum(len(r['token_ids']) for r in records),seconds=time.monotonic()-started)
        summaries.append(summary);atomic_json(out/'summary.json',summaries)
        print('AUDIT_REPEAT_COMPLETE '+json.dumps(summary),flush=True)
    print('AUDIT_SESSION_COMPLETE',out,flush=True)


def summarize(root):
    root=Path(root);sessions={};reference=None;comparisons={}
    for p in sorted(root.glob('session-*/repeat-*.jsonl')):
        rows=[json.loads(l) for l in p.read_text().splitlines()]
        if len(rows)!=512:continue
        key=f'{p.parent.name}/{p.name}';sessions[key]=dict(accuracy=sum(r['score'] for r in rows)/512,
            correct=int(sum(r['score'] for r in rows)),legacy_disagreements=sum(r['score']!=r['legacy_score'] for r in rows))
        if reference is None:reference=rows
        comparisons[key]=compare(reference,rows)
    result=dict(sessions=sessions,reference=next(iter(sessions),None),comparisons=comparisons,
        note='Changed token sequences do not necessarily change correctness. Offline stability does not identify the historical asynchronous cause.')
    atomic_json(root/'comparison.json',result)
    print(json.dumps(dict(sessions=sessions,changed_counts={k:{f:len(ids) for f,ids in v.items()} for k,v in comparisons.items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--data');p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--repeats',type=int,default=3);p.add_argument('--summarize',action='store_true')
    a=p.parse_args()
    if a.summarize:summarize(a.out)
    else:run(a.model,a.data,a.out,a.seed,a.repeats)
