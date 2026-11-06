"""Deterministic splits and verl parquet export."""
import hashlib
import json
import random
from pathlib import Path
from .reward import FINAL, number

PROMPT = "Solve the problem step by step. End with exactly one final line: #### <number>\n\n"

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def prepare_records(train, test, seed=42, dev_size=512, train_size=2048):
    def convert(rows, source):
        result = []
        for idx, row in enumerate(rows):
            question = " ".join(row["question"].split())
            match = FINAL.search(row["answer"].strip())
            if match is None:
                raise ValueError("Malformed ground truth")
            truth = str(number(match.group(1)))
            result.append(dict(problem_id=digest(question), question=question, truth=truth,
                               source_index=idx, source=source))
        return result
    train, test = convert(train, "train"), convert(test, "test")
    all_ids = [r["problem_id"] for r in train + test]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate questions within or across official splits")
    if len(train) < dev_size + train_size:
        raise ValueError("Not enough training examples")
    random.Random(seed).shuffle(train)
    splits = dict(dev=train[:dev_size], train=train[dev_size:dev_size+train_size], test=test)
    manifest = dict(seed=seed, raw_hash=digest([train, test]), splits={
        key: dict(count=len(rows), ids=[r["problem_id"] for r in rows], sha256=digest(rows))
        for key, rows in splits.items()})
    return splits, manifest

def export(out, splits, manifest, tokenizer):
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Data output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        records = []
        for row in rows:
            prompt = [{"role": "user", "content": PROMPT + row["question"]}]
            tokens = tokenizer.apply_chat_template(prompt, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
            if len(tokens) > 512:
                raise ValueError("Prompt exceeds fixed cap; revise the shared protocol before training")
            records.append(dict(data_source="dagrpo/gsm8k", prompt=prompt, ability="math",
                reward_model=dict(style="rule", ground_truth=row["truth"]),
                extra_info=dict(problem_id=row["problem_id"], split=split)))
        pq.write_table(pa.Table.from_pylist(records), out / f"{split}.parquet")
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    manifest["files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    atomic_json(out / "manifest.json", manifest)

def prepare(out, model="Qwen/Qwen3.5-9B"):
    import tempfile
    import uuid
    out=Path(out)
    if (out/'manifest.json').exists():
        from .runner import verify_data
        m=verify_data(out)
        if m['model'] != model: raise ValueError('Prepared data uses another model')
        print('Existing prepared data verified; no download required.')
        return
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer
    api = HfApi()
    ds_rev = api.dataset_info("openai/gsm8k").sha
    model_rev = api.model_info(model).sha
    data = load_dataset("openai/gsm8k", "main", revision=ds_rev)
    splits, manifest = prepare_records(list(data["train"]), list(data["test"]))
    path = snapshot_download(model, revision=model_rev)
    manifest.update(dataset_revision=ds_rev, model=model, model_revision=model_rev, model_path=str(Path(path).resolve()), chat_template_kwargs={"enable_thinking":False})
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-',dir=out.parent) as staging:
        export(staging, splits, manifest, AutoTokenizer.from_pretrained(path))
        if out.exists():
            if any(out.iterdir()): out.rename(out.with_name(out.name+'.incomplete-'+uuid.uuid4().hex[:8]))
            else: out.rmdir()
        Path(staging).rename(out)
