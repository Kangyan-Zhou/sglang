import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300-nvfp4", nightly=True)

KIMI_K25_NVFP4_MODEL_PATH = "nvidia/Kimi-K2.5-NVFP4"


class TestKimiK25NvfpGB300(unittest.TestCase):
    """Kimi-K2.5 NVFP4 performance on GB300 (4x B200 NVL4).

    NVIDIA's NVFP4 quantized version of the 1T-parameter Kimi-K2.5 thinking model.
    Uses modelopt_fp4 quantization with trtllm_mla attention backend.
    """

    def test_kimi_k25_nvfp4(self):
        base_args = [
            "--trust-remote-code",
            "--reasoning-parser=kimi_k2",
            "--tool-call-parser=kimi_k2",
            "--quantization=modelopt_fp4",
            "--attention-backend=trtllm_mla",
            "--moe-runner-backend=flashinfer_trtllm",
            "--chunked-prefill-size=32768",
            "--mem-fraction-static=0.8",
            "--max-running-requests=2048",
            "--enable-metrics",
        ]

        variants = [
            ModelLaunchSettings(
                KIMI_K25_NVFP4_MODEL_PATH,
                tp_size=4,
                extra_args=base_args,
                variant="NVFP4",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Kimi-K2.5-NVFP4-GB300",
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_kimi_k25_nvfp4_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
