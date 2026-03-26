import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300-nvfp4", nightly=True)

QWEN35_NVFP4_MODEL_PATH = "nvidia/Qwen3.5-397B-A17B-NVFP4"


class TestQwen35NvfpGB300(unittest.TestCase):
    """Qwen3.5-397B NVFP4 performance on GB300 (4x B200 NVL4).

    Single variant with EAGLE speculative decoding and modelopt_fp4 quantization.
    """

    def test_qwen35_nvfp4(self):
        base_args = [
            "--trust-remote-code",
            "--reasoning-parser=qwen3",
            "--tool-call-parser=qwen3_coder",
            "--quantization=modelopt_fp4",
            "--fp4-gemm-backend=flashinfer_cutlass",
            "--moe-runner-backend=flashinfer_trtllm",
            "--kv-cache-dtype=fp8_e4m3",
            "--attention-backend=trtllm_mha",
            "--mamba-scheduler-strategy=extra_buffer",
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
                QWEN35_NVFP4_MODEL_PATH,
                tp_size=4,
                extra_args=base_args + mtp_args,
                variant="NVFP4+MTP",
                env={"SGLANG_ENABLE_SPEC_V2": "1"},
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Qwen3.5-397B-NVFP4-GB300",
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_qwen35_nvfp4_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
