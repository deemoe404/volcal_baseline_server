import os
import sys
import unittest


APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import runtime_device  # noqa: E402


class _FakeAvailability:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeCuda:
    def __init__(self, available: bool):
        self._available = available
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self._available

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeMps:
    def __init__(self, available: bool):
        self._available = available
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self._available

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeBackends:
    def __init__(self, mps_available: bool):
        self.mps = _FakeAvailability(mps_available)


class _FakeTorch:
    def __init__(self, *, cuda_available: bool, mps_available: bool):
        self.cuda = _FakeCuda(cuda_available)
        self.mps = _FakeMps(mps_available)
        self.backends = _FakeBackends(mps_available)


class RuntimeDeviceTests(unittest.TestCase):
    def test_auto_prefers_mps_when_available(self):
        fake_torch = _FakeTorch(cuda_available=False, mps_available=True)
        chosen = runtime_device.resolve_torch_device("auto", ["mps", "cpu"], fake_torch)
        self.assertEqual(chosen, "mps")

    def test_auto_falls_back_to_cpu_when_mps_unavailable(self):
        fake_torch = _FakeTorch(cuda_available=False, mps_available=False)
        chosen = runtime_device.resolve_torch_device("auto", ["mps", "cpu"], fake_torch)
        self.assertEqual(chosen, "cpu")

    def test_explicit_cpu_never_uses_accelerators(self):
        fake_torch = _FakeTorch(cuda_available=True, mps_available=True)
        chosen = runtime_device.resolve_torch_device("cpu", ["mps", "cpu"], fake_torch)
        self.assertEqual(chosen, "cpu")

    def test_parse_fallback_chain_normalizes_and_appends_cpu(self):
        parsed = runtime_device.parse_fallback_chain("mps,cuda,mps,invalid")
        self.assertEqual(parsed, ["mps", "cuda", "cpu"])

    def test_mps_runtime_error_is_retryable(self):
        error = RuntimeError("operator not currently supported on MPS")
        self.assertTrue(runtime_device.should_retry_on_mps_error(error, "mps"))

    def test_non_mps_error_is_not_retryable(self):
        error = RuntimeError("operator not currently supported on MPS")
        self.assertFalse(runtime_device.should_retry_on_mps_error(error, "cpu"))


if __name__ == "__main__":
    unittest.main()
