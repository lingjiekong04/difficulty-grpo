"""Final greedy evaluation of an exported HF checkpoint; does not select checkpoints."""
import json
import os
from pathlib import Path
from .data import PROMPT, atomic_json, digest
from .reward import score

def score_completion(completion, tokenizer, truth):
    ids=list(completion.token_ids)
    eos=tokenizer.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    ended=bool(ids) and ids[-1] in eos
    text=tokenizer.decode(ids,skip_special_tokens=True)
    return text,ended,score(text,truth,not ended)

def evaluate(model, data, out, revision=None):
    path=Path(out)
    if path.exists(): raise ValueError('Evaluation output exists')
    # Candidate reference controls; measured audit evidence determines stability.
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    os.environ['VLLM_BATCH_INVARIANT']='0'
    from vllm import LLM, SamplingParams
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    rows = [json.loads(line) for line in Path(data).read_text(encoding="utf-8").splitlines()]
    engine = LLM(model=model, revision=revision, tensor_parallel_size=1, dtype="bfloat16",
                 max_model_len=1536, gpu_memory_utilization=.6,max_num_seqs=128,
                 seed=42,enable_prefix_caching=False,async_scheduling=False,
                 enforce_eager=True,disable_custom_all_reduce=True)
    tokenizer = engine.get_tokenizer()
    prompts = [tokenizer.apply_chat_template([{"role":"user", "content":PROMPT+r["question"]}],
        tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False) for r in rows]
    if max(map(len,prompts))>512:raise ValueError('Prompt exceeds the shared protocol cap')
    outputs=[]
    params=SamplingParams(temperature=0,top_p=1,top_k=-1,max_tokens=1024,n=1)
    for lo in range(0,len(prompts),128):
        outputs.extend(engine.generate([{'prompt_token_ids':p} for p in prompts[lo:lo+128]],
            params,use_tqdm=False))
    records = []
    for row, output in zip(rows, outputs, strict=True):
        completion = output.outputs[0]
        text,ended,rewards=score_completion(completion,tokenizer,row['truth'])
        records.append(dict(problem_id=row["problem_id"], text=text,truth=row['truth'],
            token_ids=list(completion.token_ids),stop_reason=completion.stop_reason,ended_with_eos=ended,
            tokens=len(completion.token_ids), finish_reason=completion.finish_reason,
            **rewards))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf-8")
    atomic_json(path.with_suffix(".summary.json"), dict(model=str(model), revision=revision,
        data_sha256=digest(rows), count=len(records), accuracy=sum(r["score"] for r in records)/len(records),
        protocol="offline-tp1-bf16-greedy1024-chunk128-strict-eos-reference-v4", provenance="real_evaluation",
        engine_seed=42,VLLM_ENABLE_V1_MULTIPROCESSING='0',prefix_caching=False,async_scheduling=False,
        generated_tokens=sum(r['tokens'] for r in records)))
