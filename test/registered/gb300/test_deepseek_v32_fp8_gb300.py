import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300-fp8", nightly=True)

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2"


class TestDeepseekV32Fp8GB300(unittest.TestCase):
    """DeepSeek V3.2 FP8 performance on GB300 (4x B200 NVL4).

    Thinking model with EAGLE speculative decoding. Uses reasoning-parser
    deepseek-v3 to strip <think> tokens. DP=4 + EP=4 on 4 GPUs.
    """

    def test_deepseek_v32_fp8(self):
        base_args = [
            "--trust-remote-code",
            "--tool-call-parser=deepseekv32",
            "--reasoning-parser=deepseek-v3",
            "--enable-symm-mem",
            "--mem-fraction-static=0.8",
            "--max-running-requests=2048",
            "--kv-cache-dtype=bfloat16",
            "--enable-metrics",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]
        mtp_args = [
            "--speculative-algorithm=EAGLE",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
        ]

        variants = [
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=4,
                extra_args=base_args + mtp_args,
                variant="FP8+MTP",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="DeepSeek-V3.2-FP8-GB300",
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_deepseek_v32_fp8_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
