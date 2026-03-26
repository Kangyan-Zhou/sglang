import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300-fp8", nightly=True)

GLM5_FP8_MODEL_PATH = "zai-org/GLM-5-FP8"


class TestGlm5Fp8GB300(unittest.TestCase):
    """GLM-5 FP8 performance on GB300 (4x B200 NVL4).

    EAGLE speculative decoding with FP8 quantization.
    """

    def test_glm5_fp8(self):
        base_args = [
            "--trust-remote-code",
            "--reasoning-parser=glm45",
            "--tool-call-parser=glm47",
            "--kv-cache-dtype=bfloat16",
            "--cuda-graph-max-bs=1024",
            "--mem-fraction-static=0.8",
            "--max-running-requests=2048",
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
                GLM5_FP8_MODEL_PATH,
                tp_size=4,
                extra_args=base_args + mtp_args,
                variant="FP8+MTP",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="GLM-5-FP8-GB300",
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_glm5_fp8_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
