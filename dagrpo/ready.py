"""CPU-side preparation validation. No model generation or training."""
import json
import hashlib
from pathlib import Path
from .data import atomic_json, PROMPT
from .runner import verify_data

def validate(root):
    from importlib.metadata import version
    from safetensors import safe_open
    from transformers import AutoTokenizer
    root=Path(root)
    manifest=verify_data(root/'data/gsm8k-qwen35-9b')
    if manifest['model']!='Qwen/Qwen3.5-9B': raise ValueError('Wrong model')
    versions={n:version(n) for n in ['torch','vllm','verl','transformers']}
    for n,prefix in [('torch','2.11.'),('vllm','0.24.'),('verl','0.9.1')]:
        if not versions[n].startswith(prefix): raise ValueError(f'Unexpected {n}: {versions[n]}')
    model=Path(manifest['model_path'])
    index=model/'model.safetensors.index.json'
    shards=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    keys=set()
    for shard in shards:
        with safe_open(model/shard,framework='pt',device='cpu') as handle:
            keys.update(handle.keys())
    if index.exists() and not set(json.loads(index.read_text())['weight_map']).issubset(keys):
        raise ValueError('Incomplete model shards')
    tokenizer=AutoTokenizer.from_pretrained(model)
    for split in ['train','dev','test']:
        for row in map(json.loads,(root/f'data/gsm8k-qwen35-9b/{split}.jsonl').read_text().splitlines()):
            ids=tokenizer.apply_chat_template([{'role':'user','content':PROMPT+row['question']}],
                tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
            if len(ids)>512: raise ValueError('Prompt exceeds 512 tokens')
    # Stamp the actual tokenization protocol after validating old partial preparations too.
    manifest['chat_template_kwargs']={'enable_thinking':False}
    atomic_json(root/'data/gsm8k-qwen35-9b/manifest.json',manifest)
    # Historical LoRA evidence is preserved, but cannot certify full-parameter training.
    proof_path=root/'GPU_FULL_SMOKE.json'
    proof=json.loads(proof_path.read_text()) if proof_path.exists() else {}
    from .profile import TRAINING_SOURCES
    source_hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in TRAINING_SOURCES}
    freeze=root/'environment.cu129.freeze.txt'
    prior_verified=bool(proof.get('training_mode')=='full' and proof.get('gpu_resume_verified') and
        proof.get('model_revision')==manifest['model_revision'] and
        proof.get('data_files')==manifest['files'] and
        proof.get('training_source_sha256')==source_hashes and freeze.exists() and
        proof.get('environment_freeze_sha256')==hashlib.sha256(freeze.read_bytes()).hexdigest())
    atomic_json(root/'READY.json',dict(model=manifest['model'],revision=manifest['model_revision'],
        versions=versions,shards=len(shards),training_mode='full',prior_gpu_smoke_verified=prior_verified,
        status='prepared_prior_gpu_smoke_verified' if prior_verified else 'prepared_not_gpu_training_verified'))
    print('PREPARATION_READY; prior matching GPU smoke and resume verified.' if prior_verified else
          'PREPARATION_READY; GPU training is not yet verified.',flush=True)

if __name__=='__main__':
    validate(Path.cwd())
