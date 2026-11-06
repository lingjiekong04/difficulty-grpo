"""Temporary local-disk copy of the persistent environment for GPFS deployments."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from .data import atomic_json
from .locking import exclusive_lock


def python_runtime(root):
    root = Path(root).resolve()
    primary = root / '.venv/bin/python'
    freeze = root / 'environment.cu129.freeze.txt'
    if not freeze.exists(): return str(primary)
    signature = hashlib.sha256(str(root).encode() + freeze.read_bytes()).hexdigest()
    cache = Path('/tmp/dagrpo-runtime') / signature[:20]
    with exclusive_lock(root / 'logs/runtime-cache.lock', blocking=True):
        marker = cache / 'READY.json'
        executable = cache / 'venv/bin/python'
        if marker.exists() and executable.exists():
            if json.loads(marker.read_text()).get('signature') == signature:
                return str(executable)
        print(f'RUNTIME_CACHE_BUILD: {cache}; persistent environment: {primary}', flush=True)
        started = time.monotonic()
        cache.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='building-', dir=cache.parent))
        archive = root / 'cache/runtime' / (signature[:20] + '.tar')
        if shutil.which('tar'):
            (staging/'venv').mkdir()
            if archive.exists():
                subprocess.run(['tar','-C',str(staging/'venv'),'-xf',str(archive)],check=True)
            else:
                # One metadata pass on GPFS; copy2 would stat each file repeatedly.
                producer=subprocess.Popen(['tar','-C',str(root/'.venv'),'-cf','-','.'],stdout=subprocess.PIPE)
                try:
                    consumer=subprocess.run(['tar','-C',str(staging/'venv'),'-xf','-'],stdin=producer.stdout)
                finally:
                    producer.stdout.close()
                if producer.wait() or consumer.returncode:
                    raise RuntimeError('Environment streaming copy failed')
        else:
            shutil.copytree(root / '.venv', staging / 'venv', symlinks=True)
        # The environment retains its original managed Python and editable source paths.
        check = ('import sys,importlib.metadata as m,json; '
                 'print(json.dumps({"prefix":sys.prefix,"versions":'
                 '{k:m.version(k) for k in ["torch","vllm","ray","verl"]}}))')
        result = json.loads(subprocess.check_output([str(staging/'venv/bin/python'), '-c', check], text=True))
        if Path(result['prefix']).resolve() != (staging/'venv').resolve():
            raise RuntimeError('Copied environment did not use its local site-packages')
        if cache.exists():
            cache.rename(cache.with_name(cache.name+'-incomplete-'+str(time.time_ns())))
        staging.rename(cache)
        atomic_json(marker, dict(signature=signature, source=str(primary), versions=result['versions']))
        print(f'RUNTIME_CACHE_READY: {executable}; copy_seconds={time.monotonic()-started:.1f}', flush=True)
        if not archive.exists() and shutil.which('tar'):
            archive.parent.mkdir(parents=True,exist_ok=True)
            pending=archive.with_name(archive.name+'.'+str(time.time_ns())+'.partial')
            subprocess.run(['tar','-C',str(cache/'venv'),'-cf',str(pending),'.'],check=True)
            pending.replace(archive)
            print(f'RUNTIME_ARCHIVE_READY: {archive}',flush=True)
        return str(executable)


if __name__ == '__main__':
    print(python_runtime(Path.cwd()), flush=True)
