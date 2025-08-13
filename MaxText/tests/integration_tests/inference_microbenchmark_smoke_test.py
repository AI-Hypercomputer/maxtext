"""Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

""" Smoke test for inference microbenchmark"""
import jax
import os.path
import pytest
import unittest
from absl.testing import absltest

from MaxText import pyconfig
from MaxText.globals import PKG_DIR, tpu_present
from MaxText.inference_microbenchmark import run_benchmarks


class Inference_Microbenchmark(unittest.TestCase):
  """integration test for inference microbenchmark"""

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  @unittest.skipIf(not tpu_present, "TPU only test")
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
