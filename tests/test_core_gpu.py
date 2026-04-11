"""Tests for edgevox.core.gpu — hardware detection with subprocess mocking."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from edgevox.core.gpu import (
    get_nvidia_gpu_name,
    get_nvidia_used_mb,
    get_nvidia_vram_gb,
    get_ram_gb,
    has_cuda,
    has_metal,
)


class TestGetNvidiaVramGb:
    @patch("shutil.which", return_value=None)
    def test_no_nvidia_smi(self, _mock):
        assert get_nvidia_vram_gb() is None

    @patch("subprocess.check_output", return_value="8192\n")
    @patch("shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_parses_vram(self, _which, _sub):
        assert get_nvidia_vram_gb() == 8.0

    @patch("subprocess.check_output", side_effect=subprocess.SubprocessError("fail"))
    @patch("shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_subprocess_error(self, _which, _sub):
        assert get_nvidia_vram_gb() is None


class TestGetNvidiaGpuName:
    @patch("shutil.which", return_value=None)
    def test_no_nvidia_smi(self, _mock):
        assert get_nvidia_gpu_name() is None

    @patch("subprocess.check_output", return_value="NVIDIA RTX 4090\n")
    @patch("shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_returns_name(self, _which, _sub):
        assert get_nvidia_gpu_name() == "NVIDIA RTX 4090"


class TestGetNvidiaUsedMb:
    @patch("shutil.which", return_value=None)
    def test_no_nvidia_smi(self, _mock):
        assert get_nvidia_used_mb() is None

    @patch("subprocess.check_output", return_value="2048\n")
    @patch("shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_returns_mb(self, _which, _sub):
        assert get_nvidia_used_mb() == 2048.0


class TestHasCuda:
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=8.0)
    def test_true_with_vram(self, _mock):
        assert has_cuda() is True

    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_false_without_vram(self, _mock):
        assert has_cuda() is False


class TestHasMetal:
    @patch("platform.machine", return_value="arm64")
    @patch("platform.system", return_value="Darwin")
    def test_true_on_macos_arm(self, _sys, _mach):
        assert has_metal() is True

    @patch("platform.machine", return_value="x86_64")
    @patch("platform.system", return_value="Linux")
    def test_false_on_linux(self, _sys, _mach):
        assert has_metal() is False

    @patch("platform.machine", return_value="x86_64")
    @patch("platform.system", return_value="Darwin")
    def test_false_on_macos_intel(self, _sys, _mach):
        assert has_metal() is False


class TestGetRamGb:
    def test_with_psutil(self):
        mock_vm = MagicMock()
        mock_vm.total = 16 * (1024**3)
        with patch("psutil.virtual_memory", return_value=mock_vm):
            assert abs(get_ram_gb() - 16.0) < 0.1

    def test_fallback_default(self):
        with (
            patch.dict("sys.modules", {"psutil": None}),
            patch("builtins.open", side_effect=OSError),
        ):
            # When psutil import fails and /proc/meminfo fails, returns 8.0
            result = get_ram_gb()
            assert isinstance(result, float)
