# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests that moe_accumulate_chunk_wgrad gives the same loss and gradients as with the flag off."""

import unittest

import jax
import numpy as np
import pytest

from tests.unit.moe_megatron_aux_loss_test import _run_routed_moe, _tiny_deepseek_config


@pytest.mark.tpu_only
class AccumulateChunkWgradParityTest(unittest.TestCase):
  """Chunked ring-of-experts MoE gives the same loss and gradients with and without the flag."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 2 or jax.device_count() % 2:
      self.skipTest("Needs an even number (>= 2) of devices for expert parallelism 2.")

  def _run(self, params=None, **overrides):
    """Runs the chunked ring-of-experts RoutedMoE on the tokamax gmm_v2 path."""
    cfg = _tiny_deepseek_config(
        "accum_wgrad_" + "_".join(f"{k}{v}" for k, v in sorted(overrides.items())),
        ici_expert_parallelism=2,
        use_ring_of_experts=True,
        num_moe_token_chunks=2,
        use_tokamax_gmm=True,
        use_gmm_v2=True,
        **overrides,
    )
    return _run_routed_moe(cfg, params)

  def _assert_parity(self, **overrides):
    """Asserts output, lb loss and all gradients match between flag off and on, with shared params."""
    (out_ref, lb_ref, _, _), grads_ref, params, _ = self._run(moe_accumulate_chunk_wgrad=False, **overrides)
    (out, lb, _, _), grads, _, _ = self._run(params, moe_accumulate_chunk_wgrad=True, **overrides)
    np.testing.assert_allclose(out, out_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lb, lb_ref, rtol=1e-6)
    for (path, g), g_ref in zip(
        jax.tree_util.tree_leaves_with_path(grads), jax.tree_util.tree_leaves(grads_ref), strict=True
    ):
      np.testing.assert_allclose(g, g_ref, rtol=1e-4, atol=1e-6, err_msg=jax.tree_util.keystr(path))

  def test_loss_and_grads_match(self):
    self._assert_parity()

  def test_loss_and_grads_match_fp8_fixed_calibration(self):
    """fp8_full with fixed calibration: chunk 0 quantizes the weights once and later chunks reuse them."""
    self._assert_parity(
        quantization="fp8_full",
        use_qwix_quantization=True,
        weight_quantization_calibration_method="fixed,-224,224",
        act_quantization_calibration_method="fixed,-224,224",
        bwd_quantization_calibration_method="fixed,-1,1",
    )


if __name__ == "__main__":
  unittest.main()
