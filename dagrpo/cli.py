import argparse
import json
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Difficulty-aware GRPO experiment toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--out", default="data/gsm8k-qwen35-9b")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p = sub.add_parser("run")
    p.add_argument("--phase", choices=["pilot","full"], default="pilot")
    p.add_argument("--variant", choices=list("UASE"), default="U")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default="data/gsm8k-qwen35-9b")
    p.add_argument("--training-mode", choices=["full"], default="full")
    p.add_argument("--batch-size",type=int,default=64,help="Questions per update; four fresh answers each")
    p.add_argument("--micro-batch-size",type=int,default=8,help="Sequences per GPU forward/backward")
    p.add_argument("--fused-kernels",action='store_true',help="Opt-in torch fused LM head; profile parity first")
    p.add_argument("--steps",type=int,default=None)
    p.add_argument("--save-freq",type=int,default=None)
    p.add_argument("--stop-after-step",type=int,default=None,help="Save and stop at an update boundary; resume with a larger boundary")
    p.add_argument("--initial-checkpoint",default=None,help="Full-parameter checkpoint for a separate engineering profile; never a LoRA checkpoint")
    p.add_argument("--skip-initial-validation",action='store_true',help="Engineering smoke only; formal runs retain the initial dev evaluation")
    p.add_argument("--checkpoint-grace-minutes",type=float,default=20)
    p.add_argument("--pilot-budget-cap",type=float,default=float('inf'),help="Optional pilot limit; disabled by default")
    p.add_argument("--out", required=True)
    p.add_argument("--budget-gpu-hours", type=float,default=float('inf'),help="Optional GPU-hour limit; inf disables it and retains accounting")
    p.add_argument("--ledger", default="runs/budget.json")
    p.add_argument("--session-hours",type=float,default=None,
                   help="Request a checkpoint before this wall-clock session limit")
    p = sub.add_parser("matrix"); p.add_argument("--out", default="matrix.json")
    p = sub.add_parser("report"); p.add_argument("--root", default="runs"); p.add_argument("--out", default="reports")
    p = sub.add_parser("estimate"); p.add_argument("--trace", required=True); p.add_argument("--out", default="cost-estimate.json")
    p = sub.add_parser("evaluate"); p.add_argument("--model", required=True); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cmd=='evaluate' and any(os.environ.get(k)!=v for k,v in
            {'VLLM_ENABLE_V1_MULTIPROCESSING':'0','PYTHONHASHSEED':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8'}.items()):
        environment=os.environ.copy()
        environment.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_BATCH_INVARIANT='0',
            PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        os.execvpe(sys.executable,[sys.executable,'-m','dagrpo.cli',*sys.argv[1:]],environment)
    if args.cmd == "prepare":
        from .data import prepare
        prepare(args.out,args.model)
    elif args.cmd == "run":
        if args.phase=='full' and args.skip_initial_validation:
            parser.error('Formal experiments require initial validation')
        if args.phase == "pilot" and args.budget_gpu_hours > args.pilot_budget_cap:
            parser.error("Pilot budget exceeds optional --pilot-budget-cap")
        if any(x is not None and x<=0 for x in (args.steps,args.save_freq,args.stop_after_step)) or args.checkpoint_grace_minutes<0:
            parser.error('Steps, save frequency and stop boundary must be positive; grace must be nonnegative')
        from .runner import launch
        raise SystemExit(launch(args))
    elif args.cmd == "matrix":
        from .data import atomic_json
        atomic_json(args.out, [dict(variant=v, seed=s, steps=200) for v in "UA" for s in (42,43,44)] +
                    [dict(variant=v, seed=42, steps=200) for v in "SE"])
    elif args.cmd == "report":
        from .report import generate
        generate(args.root, args.out)
    elif args.cmd == "estimate":
        from .report import estimate
        estimate(args.trace, args.out)
    elif args.cmd == "evaluate":
        from .evaluate import evaluate
        evaluate(args.model, args.data, args.out)

if __name__ == "__main__": main()
