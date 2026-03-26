import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300-nvfp4", nightly=True)

DEEPSEEK_V32_NVFP4_MODEL_PATH = "nvidia/DeepSeek-V3.2-NVFP4"


class TestDeepseekV32NvfpGB300(unittest.TestCase):
    """DeepSeek V3.2 NVFP4 performance on GB300 (4x B200 NVL4).

    Thinking model with EAGLE speculative decoding and modelopt_fp4 quantization.
    Uses reasoning-parser deepseek-v3 to strip <think> tokens.
    """

    def test_deepseek_v32_nvfp4(self):
        base_args = [
            "--trust-remote-code",
            "--tool-call-parser=deepseekv32",
            "--reasoning-parser=deepseek-v3",
            "--quantization=modelopt_fp4",
            "--moe-runner-backend=flashinfer_trtllm",
            "--mem-fraction-static=0.8",
            "--max-running-requests=2048",
            "--kv-cache-dtype=bfloat16",
            "--cuda-graph-max-bs=1024",
            "--enable-metrics",
        ]
        mtp_args = [
            "--speculative-algorithm=EAGLE",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
        ]

        variants = [
            ModelLaunchSettings(
                DEEPSEEK_V32_NVFP4_MODEL_PATH,
                tp_size=4,
                extra_args=base_args + mtp_args,
                variant="NVFP4+MTP",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="DeepSeek-V3.2-NVFP4-GB300",
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_deepseek_v32_nvfp4_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
