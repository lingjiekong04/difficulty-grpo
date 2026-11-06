import os
import unittest
from unittest.mock import patch
from dagrpo.gpu_idle import wait_gpu_idle


class GpuIdleTests(unittest.TestCase):
    @patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'})
    @patch('dagrpo.gpu_idle.time.sleep')
    @patch('dagrpo.gpu_idle.subprocess.check_output')
    def test_only_selected_device_is_checked(self, output, sleep):
        output.side_effect=['', '0\n']*3
        wait_gpu_idle()
        self.assertEqual(output.call_count, 6)
        for call in output.call_args_list:
            self.assertIn('--id=1', call.args[0])

    @patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'})
    @patch('dagrpo.gpu_idle.time.monotonic', side_effect=[0,0,121])
    @patch('dagrpo.gpu_idle.time.sleep')
    @patch('dagrpo.gpu_idle.subprocess.check_output', side_effect=['123\n','1600\n'])
    def test_occupied_selected_device_cannot_start(self, output, sleep, monotonic):
        with self.assertRaisesRegex(RuntimeError, 'Selected GPUs 1'):
            wait_gpu_idle()

    @patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'})
    @patch('dagrpo.gpu_idle.time.sleep')
    @patch('dagrpo.gpu_idle.subprocess.check_output', side_effect=['','0\n']*3)
    def test_worker_device_overrides_parent_mask(self, output, sleep):
        wait_gpu_idle(devices='1')
        for call in output.call_args_list:
            self.assertIn('--id=1',call.args[0])
