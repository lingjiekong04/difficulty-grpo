import json
from pathlib import Path

def latest_complete(root):
    candidates=[]
    for marker in Path(root).glob('global_step_*/complete.json'):
        folder=marker.parent
        try:
            step=int(folder.name.rsplit('_',1)[1])
            if json.loads(marker.read_text())['step']!=step: continue
        except (ValueError,KeyError,OSError): continue
        if (folder/'study_state.pt').is_file() and (folder/'data.pt').is_file() and (folder/'actor').is_dir():
            candidates.append((step,folder))
    return max(candidates)[1] if candidates else None


def prune_full_checkpoints(root, keep=2, best_step=None):
    """Keep latest complete full checkpoints plus best dev. Ignore unknown/partial data."""
    import shutil
    root=Path(root).resolve()
    complete=[]
    for marker in root.glob('global_step_*/complete.json'):
        try:
            info=json.loads(marker.read_text())
            step=int(marker.parent.name.rsplit('_',1)[1])
            if info.get('training_mode')=='full' and info['step']==step:
                complete.append((step,marker.parent.resolve()))
        except (OSError,ValueError,KeyError): continue
    preserve={step for step,_ in sorted(complete,reverse=True)[:keep]} | {best_step}
    removed=[]
    for step,folder in sorted(complete):
        if step not in preserve:
            if folder.parent != root: raise RuntimeError('Checkpoint path escaped run root')
            shutil.rmtree(folder)
            removed.append(step)
    return removed
