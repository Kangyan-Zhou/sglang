import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Kimi-K2.5 native INT4 — runs in the FP8 round only.
# The NVFP4 round uses nvidia/Kimi-K2.5-NVFP4 instead (test_kimi_k25_nvfp4_gb300.py).
register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300-fp8", nightly=True)

KIMI_K25_MODEL_PATH = "moonshotai/Kimi-K2.5"


class TestKimiK25GB300(unittest.TestCase):
    """Kimi-K2.5 (native INT4) performance on GB300 (4x B200 NVL4).

    Kimi-K2.5 is a 1T-parameter thinking/reasoning model with native INT4
    quantization (compressed-tensors). No separate FP8 variant exists, so this
    test covers both rounds.
    """

    def test_kimi_k25(self):
        base_args = [
            "--trust-remote-code",
            "--reasoning-parser=kimi_k2",
            "--tool-call-parser=kimi_k2",
            "--kv-cache-dtype=fp8_e4m3",
            "--chunked-prefill-size=32768",
            "--mem-fraction-static=0.8",
            "--max-running-requests=2048",
            "--enable-metrics",
        ]

        variants = [
            ModelLaunchSettings(
                KIMI_K25_MODEL_PATH,
                tp_size=4,
                extra_args=base_args,
                variant="INT4",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Kimi-K2.5-GB300",
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_kimi_k25_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
