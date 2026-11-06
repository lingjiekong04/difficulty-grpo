"""Audited full-parameter FSDP2 engine, registered through verl's public vendor hook.

FP32 master shards and Adam states stay on CPU with native CPUOffloadPolicy;
FSDP2 materializes BF16 weights for GPU computation. No PEFT adapters are used.
"""
import json
import math
import os
from pathlib import Path
import time

import torch
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from .data import atomic_json


def trainability(module):
    named=list(module.named_parameters())
    if any('lora_' in name.lower() for name,_ in named) or hasattr(module,'peft_config'):
        raise RuntimeError('Full-parameter run contains PEFT/LoRA adapters')
    frozen=[name for name,p in named if not p.requires_grad]
    if frozen:
        raise RuntimeError(f'Full-parameter run contains frozen parameters: {frozen[:8]}')
    return dict(total_parameters=sum(p.numel() for _,p in named),
                trainable_parameters=sum(p.numel() for _,p in named if p.requires_grad),
                parameter_tensors=len(named), storage_dtypes=sorted({str(p.dtype) for _,p in named}))


def optimizer_coverage(module, optimizer):
    wanted={id(p) for p in module.parameters() if p.requires_grad}
    actual=[id(p) for group in optimizer.param_groups for p in group['params']]
    if len(actual)!=len(set(actual)) or set(actual)!=wanted:
        raise RuntimeError('Optimizer does not cover every trainable parameter exactly once')
    return len(actual)


def local_tensor(tensor):
    return tensor.to_local() if hasattr(tensor,'to_local') else tensor


def probe_positions(numel,device):
    # Float32 linspace rounds (numel-1) up to numel for large embedding shards.
    count=min(1024,numel)
    return torch.arange(count,device=device,dtype=torch.int64)*(numel-1)//max(1,count-1)


def probes(module):
    # Each rank can have zero local gradients for a vocabulary shard; record this,
    # and assess signal across ranks/steps rather than demanding every shard move.
    eligible=[(n,p) for n,p in module.named_parameters()
        if p.requires_grad and p.ndim>=2 and
        not any(x in n.lower() for x in ('visual','vision','projector'))]
    if not eligible: raise RuntimeError('No language-model base weights to audit')
    indices=sorted({0,len(eligible)//4,len(eligible)//2,3*len(eligible)//4,len(eligible)-1})
    output=[]
    for i in indices:
        name,p=eligible[i]
        value=local_tensor(p.detach()).reshape(-1)
        if not value.numel(): continue
        positions=probe_positions(value.numel(),value.device)
        before=value[positions].float().cpu().clone()
        grad=local_tensor(p.grad.detach()) if p.grad is not None else None
        output.append((name,p,positions,before,grad is not None,
            float(grad.float().norm().cpu()) if grad is not None else None))
    return output


@EngineRegistry.register(model_type='language_model',backend='fsdp2',device='cuda',vendor='dagrpo_full')
class FullParameterEngine(FSDPEngineWithLMHead):
    def _build_module(self):
        module=super()._build_module()
        if not self.engine_config.forward_only:
            if self.model_config.lora_rank or self.model_config.lora_adapter_path:
                raise RuntimeError('FullParameterEngine refuses adapters')
            if not self.engine_config.offload_policy:
                raise RuntimeError('This two-card full-parameter protocol requires native FSDP2 CPU offload')
            self._audit=trainability(module)
            if self._audit['storage_dtypes']!=['torch.float32']:
                raise RuntimeError('Full training requires FP32 master parameters with BF16 computation')
            available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
            threads=min(available,int(os.environ.get('DAGRPO_CPU_THREADS','8')))
            torch.set_num_threads(max(1,threads))
            self._audit.update(training_mode='full',lora_rank=0,cpu_threads=torch.get_num_threads(),
                native_cpu_offload=True,rank=self.rank,pid=os.getpid())
            implementations=sorted({layer.chunk_gated_delta_rule.__module__ for layer in module.modules()
                if hasattr(layer,'chunk_gated_delta_rule')})
            self._audit['linear_attention_implementations']=implementations
            if os.environ.get('DAGRPO_REQUIRE_FLA')=='1' and not any(x.startswith('fla.') for x in implementations):
                raise RuntimeError('Optimized protocol requires FLA; torch fallback is not accepted')
        return module

    def _build_optimizer(self,module):
        optimizer=super()._build_optimizer(module)
        self._audit['optimizer_parameter_tensors']=optimizer_coverage(module,optimizer)
        folder=Path(os.environ['DAGRPO_AUDIT_DIR']); folder.mkdir(parents=True,exist_ok=True)
        self._audit_folder=folder
        atomic_json(folder/f'trainability-rank{self.rank}.json',self._audit)
        print('FULL_PARAMETER_AUDIT '+json.dumps(self._audit),flush=True)
        self._audit_step=0
        return optimizer

    def optimizer_step(self):
        self._audit_step+=1
        previous_steps=[float(state['step'].item()) for state in self.optimizer.state.values()
            if 'step' in state] if self._audit_step<=3 else []
        # Audit the first three optimizer steps of EACH process, including resumes.
        samples=probes(self.module) if self._audit_step<=3 else []
        started=time.monotonic()
        norm=super().optimizer_step()
        elapsed=time.monotonic()-started
        if not math.isfinite(float(norm)):
            raise RuntimeError('Non-finite gradient norm in full-parameter training')
        if samples:
            records=[]
            for name,p,positions,before,has_grad,grad_norm in samples:
                after=local_tensor(p.detach()).reshape(-1)[positions].float().cpu()
                delta=(after-before).abs()
                records.append(dict(name=name,has_gradient=has_grad,local_gradient_norm=grad_norm,
                    sampled_values=len(before),changed_values=int((delta>0).sum()),max_delta=float(delta.max())))
            record=dict(pid=os.getpid(),rank=self.rank,process_optimizer_step=self._audit_step,
                optimizer_state_parameters_before=len(previous_steps),
                optimizer_step_before_max=max(previous_steps,default=0),
                gradient_norm=float(norm),optimizer_seconds=elapsed,weights=records)
            with (self._audit_folder/f'updates-rank{self.rank}.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
        return norm
