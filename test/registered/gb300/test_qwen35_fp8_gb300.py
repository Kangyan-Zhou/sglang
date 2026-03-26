import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300-fp8", nightly=True)

QWEN35_FP8_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B-FP8"


class TestQwen35Fp8GB300(unittest.TestCase):
    """Qwen3.5-397B FP8 performance on GB300 (4x B200 NVL4).

    Single variant with EAGLE speculative decoding and FP8 quantization.
    """

    def test_qwen35_fp8(self):
        base_args = [
            "--trust-remote-code",
            "--reasoning-parser=qwen3",
            "--tool-call-parser=qwen3_coder",
            "--fp8-gemm-backend=flashinfer_trtllm",
            "--attention-backend=trtllm_mha",
            "--mamba-scheduler-strategy=extra_buffer",
            "--max-mamba-cache-size=128",
            "--page-size=64",
            "--enable-flashinfer-allreduce-fusion",
            "--mem-fraction-static=0.8",
            "--max-running-requests=256",
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
                QWEN35_FP8_MODEL_PATH,
                tp_size=4,
                extra_args=base_args + mtp_args,
                variant="FP8+MTP",
                env={"SGLANG_ENABLE_SPEC_V2": "1"},
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Qwen3.5-397B-FP8-GB300",
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_qwen35_fp8_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
