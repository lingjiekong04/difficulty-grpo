"""Container-specific Ray dashboard/NVML workaround, isolated to its agent process.

Driver 550.54.14 in this platform segfaults in Ray's vendored
nvmlDeviceGetComputeRunningProcesses_v3. GPU resources and training workers
remain enabled; the project records card metrics through nvidia-smi instead.
"""
import os
import sys

if os.environ.get('DAGRPO_DISABLE_RAY_GPU_METRICS')=='1' and (
    sys.argv and sys.argv[0].replace('\\','/').endswith('/ray/dashboard/agent.py')
):
    from ray.dashboard.modules.reporter.gpu_providers import GpuMetricProvider
    GpuMetricProvider.get_gpu_usage=lambda self: []
    print('DAGRPO: Ray dashboard NVML process queries disabled; external card monitoring remains enabled.',
          file=sys.stderr,flush=True)
