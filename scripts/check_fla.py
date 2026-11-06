"""CUDA forward/backward parity and isolated timing against the HF torch fallback."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import time
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from dagrpo.data import atomic_json

parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--out',required=True)
args=parser.parse_args()
torch.manual_seed(42)

def evaluate(function,values):
    inputs=[x.detach().clone().requires_grad_(True) for x in values]
    output,_=function(*inputs,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=True)
    loss=output.float().square().mean()
    gradients=torch.autograd.grad(loss,inputs)
    return output.detach(),[g.detach() for g in gradients]

def relative_error(reference,actual):
    return float((reference.float()-actual.float()).norm()/(reference.float().norm()+1e-12))

results=[]
for length,heads in ((128,4),(512,32)):
    shape=(2,length,heads,128)
    values=[torch.randn(shape,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    values += [-F.softplus(torch.randn(shape[:3],device='cuda',dtype=torch.float32)),
               torch.rand(shape[:3],device='cuda',dtype=torch.bfloat16)]
    reference,reference_grads=evaluate(torch_chunk_gated_delta_rule,values)
    actual,actual_grads=evaluate(chunk_gated_delta_rule,values)
    errors=[relative_error(a,b) for a,b in zip(reference_grads,actual_grads)]
    error=relative_error(reference,actual)
    finite=bool(torch.isfinite(actual).all()) and all(bool(torch.isfinite(g).all()) for g in actual_grads)
    timings={}
    for name,function in (('torch_fallback',torch_chunk_gated_delta_rule),('fla',chunk_gated_delta_rule)):
        durations=[]
        for _ in range(3):
            torch.cuda.synchronize(); started=time.perf_counter()
            evaluate(function,values)
            torch.cuda.synchronize(); durations.append(time.perf_counter()-started)
        timings[name]=median(durations)
    record=dict(length=length,heads=heads,output_relative_error=error,
        gradient_relative_errors=errors,finite=finite,median_forward_backward_seconds=timings,
        isolated_speedup=timings['torch_fallback']/timings['fla'],
        passed=finite and error<0.03 and max(errors)<0.05)
    results.append(record); print(json.dumps(record),flush=True)
summary=dict(torch=torch.__version__,fla_core=version('fla-core'),flash_linear_attention=version('flash-linear-attention'),
    device=torch.cuda.get_device_name(),cases=results,passed=all(r['passed'] for r in results),
    note='BF16 relative norm tolerances: output 3%, gradients 5%. Isolated operator timings do not predict full training speedup.')
atomic_json(args.out,summary)
if not summary['passed']: raise SystemExit('FLA parity failed; do not start formal training')
