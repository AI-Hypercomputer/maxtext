# SPDX-License-Identifier: Apache-2.0

""" Smoke test for inference microbenchmark"""
import jax
import os.path
import pytest
import unittest
from absl.testing import absltest

from MaxText import pyconfig
from MaxText.globals import PKG_DIR
from MaxText.inference_microbenchmark import run_benchmarks


class Inference_Microbenchmark(unittest.TestCase):
  """integration test for inference microbenchmark"""

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test(self):
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    config = pyconfig.initialize(
        [
            None,
            os.path.join(PKG_DIR, "configs", "tpu_smoke_test.yml"),
            rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
            "ici_autoregressive_parallelism=-1",
            "ici_fsdp_parallelism=1",
            "max_prefill_predict_length=1024",
            "max_target_length=2048",
            "scan_layers=false",
            "weight_dtype=bfloat16",
            "attention=dot_product",
            "skip_jax_distributed_system=True",
        ]
    )
    run_benchmarks(config)


if __name__ == "__main__":
  absltest.main()
