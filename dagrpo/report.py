import csv
import json
from pathlib import Path
from statistics import mean, stdev
from .data import atomic_json,digest

def protocol_id(metadata):
    volatile={'data.seed','actor_rollout_ref.actor.data_loader_seed',
        'actor_rollout_ref.actor.fsdp_config.seed','actor_rollout_ref.rollout.seed',
        'trainer.experiment_name','trainer.default_local_dir',
        'ray_kwargs.ray_init.runtime_env.env_vars.DAGRPO_AUDIT_DIR'}
    settings={}
    for item in metadata.get('config_overrides',[]):
        key,value=item.split('=',1); key=key.lstrip('+')
        if key not in volatile: settings[key]=value
    return digest(dict(config=settings,data=metadata.get('data',{}),
        kernels=metadata.get('kernel_versions',{}),environment=metadata.get('environment_freeze_sha256')))


def generate(root, out, *, plots=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []; excluded=[]
    for path in sorted(Path(root).rglob("evaluations.jsonl")):
        metadata_path=path.parent/'run.json'
        metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if metadata.get('training_mode')!='full' or metadata.get('phase')!='full':
            excluded.append(dict(run=str(path.parent),reason='Not a formal full-parameter run'))
            continue
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        # Resumes may re-evaluate the same step. Retain its most recent evaluation.
        rows = sorted({r["step"]:r for r in rows}.values(), key=lambda r:r["step"])
        if not rows: continue
        best = max(rows, key=lambda r:(r["accuracy"], -r["step"]))
        runs.append(dict(run=str(path.parent), variant=rows[-1]["variant"], seed=rows[-1]["seed"],
            protocol=protocol_id(metadata),final=rows[-1], best_dev=best, curve=rows))
    atomic_json(out/"summary.json", runs)
    atomic_json(out/'excluded-runs.json',excluded)
    lines = ["# Experiment report", "", "All accuracies here are DEVELOPMENT results. Final test evaluation is separate.", "",
             "|Run|Variant|Seed|Final step|Dev accuracy|Best dev step|GPU hours|", "|---|---|---|---|---|---|---|"]
    for r in runs:
        f = r["final"]
        lines.append(f"|{Path(r['run']).name}|{r['variant']}|{r['seed']}|{f['step']}|{f['accuracy']:.4f}|{r['best_dev']['step']}|{f['gpu_hours']:.3f}|")
    if not runs: lines += ["", "No training results available. No improvement claim can be made."]
    for protocol in sorted({r['protocol'] for r in runs}):
        for variant in "UA":
            matched=[r for r in runs if r['protocol']==protocol and r['variant']==variant and r['final']['step']==200]
            values=[r['final']['accuracy'] for r in matched]
            if len({r['seed'] for r in matched})!=len(matched):
                lines += ['',f'{variant}, protocol {protocol[:12]}: duplicate seeds; aggregate withheld.']
            elif values:
                lines += ["",f"{variant}, protocol {protocol[:12]}: n={len(values)}, mean={mean(values):.4f}, sample SD={stdev(values) if len(values)>1 else 'unavailable'}"]
    if excluded: lines += ['',f'{len(excluded)} historical, pilot or unlabelled runs excluded; see excluded-runs.json.']
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if runs and plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2,figsize=(10,4))
        for run in runs:
            for ax, x in zip(axes, ("tokens", "gpu_hours")):
                ax.plot([r[x] for r in run["curve"]], [r["accuracy"] for r in run["curve"]], marker=".", label=f"{run['variant']}-{run['seed']}-{run['protocol'][:6]}")
                ax.set_xlabel(x); ax.set_ylabel("Dev greedy accuracy")
        axes[0].legend(); fig.tight_layout(); fig.savefig(out/"learning_curves.png", dpi=160); plt.close(fig)

def estimate(trace, out):
    rows = sorted({r["step"]:r for r in map(json.loads, Path(trace).read_text().splitlines())}.values(), key=lambda r:r["step"])
    if len(rows)<2 or rows[-1]["step"] == rows[0]["step"]:
        raise ValueError("Need two evaluation checkpoints for a measured estimate")
    per_step = (rows[-1]["gpu_hours"]-rows[0]["gpu_hours"])/(rows[-1]["step"]-rows[0]["step"])
    atomic_json(out, dict(measured_gpu_hours_per_step=per_step,
        eight_runs_200_steps_estimate=8*(rows[0]["gpu_hours"]+200*per_step),
        note="Includes observed evaluation overhead; excludes exports/final test and future failures. Re-profile if configuration changes."))
