# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Security tests for GDN attention."""

from absl.testing import absltest
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from maxtext.kernels.causal_conv1d_gated_delta_rule import wrapper
from tokamax._src.ops.experimental.utils.test_utils import poison_tpu_memory


class GDNSecurityTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    try:
      if not pltpu.get_tpu_info().generation >= 6:
        self.skipTest("Pallas TPU kernel requires TPU v6 or newer.")
    except Exception:
      self.skipTest("Failed to get TPU info.")

  def test_uninitialized_memory_robustness(self):
    poison_tpu_memory()
    seq_lens = jnp.array([128], dtype=jnp.int32)
    mixed_qkv = jnp.zeros((128, 1536), dtype=jnp.float32)
    new_states, output = wrapper.fused_conv1d_gdn(
        qkv=mixed_qkv,
        seq_lens=seq_lens,
        b=jnp.zeros((128, 8), dtype=jnp.float32),
        a=jnp.zeros((128, 8), dtype=jnp.float32),
        conv_state=jnp.zeros((2, 3, 1536), dtype=jnp.float32),
        recurrent_state=jnp.zeros((2, 8, 128, 128), dtype=jnp.float32),
        conv_weight=jnp.zeros((1536, 1, 4), dtype=jnp.float32),
        conv_bias=jnp.zeros((1536,), dtype=jnp.float32),
        a_log=jnp.zeros((8,), dtype=jnp.float32),
        dt_bias=jnp.zeros((8,), dtype=jnp.float32),
        query_start_loc=jnp.array([0, 128]),
        state_indices=jnp.array([1]),
        distribution=jnp.array([0, 3, 3], dtype=jnp.int32),
        n_kq=2,
        n_v=8,
        d_k=128,
        d_v=128,
        kernel_size=4,
    )
    for new_state in new_states:
      self.assertFalse(jnp.any(jnp.isnan(new_state)))
    self.assertFalse(jnp.any(jnp.isnan(output)))

  def test_security_isolation(self):
    # Configure two request sequences sharing the same local layer runner
    seq_lens = jnp.array([128, 128], dtype=jnp.int32)
    mixed_qkv = jnp.zeros((256, 1536), dtype=jnp.float32)

    # Baseline clean context evaluation
    _, output_clean = wrapper.fused_conv1d_gdn(
        qkv=mixed_qkv,
        seq_lens=seq_lens,
        b=jnp.zeros((256, 8), dtype=jnp.float32),
        a=jnp.zeros((256, 8), dtype=jnp.float32),
        conv_state=jnp.zeros((3, 3, 1536), dtype=jnp.float32),
        recurrent_state=jnp.zeros((3, 8, 128, 128), dtype=jnp.float32),
        conv_weight=jnp.zeros((1536, 1, 4), dtype=jnp.float32),
        conv_bias=jnp.zeros((1536,), dtype=jnp.float32),
        a_log=jnp.zeros((8,), dtype=jnp.float32),
        dt_bias=jnp.zeros((8,), dtype=jnp.float32),
        query_start_loc=jnp.array([0, 128, 256]),
        state_indices=jnp.array([1, 2]),
        distribution=jnp.array([0, 3, 3], dtype=jnp.int32),
        n_kq=2,
        n_v=8,
        d_k=128,
        d_v=128,
        kernel_size=4,
    )

    # Inject NaNs into the second request's allocated recurrent state buffer
    recurrent_state_malicious = (
        jnp.zeros((3, 8, 128, 128), dtype=jnp.float32).at[2].set(jnp.nan)
    )
    _, output_malicious = wrapper.fused_conv1d_gdn(
        qkv=mixed_qkv,
        seq_lens=seq_lens,
        b=jnp.zeros((256, 8), dtype=jnp.float32),
        a=jnp.zeros((256, 8), dtype=jnp.float32),
        conv_state=jnp.zeros((3, 3, 1536), dtype=jnp.float32),
        recurrent_state=recurrent_state_malicious,
        conv_weight=jnp.zeros((1536, 1, 4), dtype=jnp.float32),
        conv_bias=jnp.zeros((1536,), dtype=jnp.float32),
        a_log=jnp.zeros((8,), dtype=jnp.float32),
        dt_bias=jnp.zeros((8,), dtype=jnp.float32),
        query_start_loc=jnp.array([0, 128, 256]),
        state_indices=jnp.array([1, 2]),
        distribution=jnp.array([0, 3, 3], dtype=jnp.int32),
        n_kq=2,
        n_v=8,
        d_k=128,
        d_v=128,
        kernel_size=4,
    )

    # Assert strict sequence isolation for the first sequence
    np.testing.assert_allclose(
        np.array(output_malicious[:128]),
        np.array(output_clean[:128]),
        atol=0,
        rtol=0,
    )

if __name__ == "__main__":
  absltest.main()
