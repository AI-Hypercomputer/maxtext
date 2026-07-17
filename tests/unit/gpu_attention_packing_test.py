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

"""Tests packed-sequence masking of GPU flash attention kernels.

With sequence packing, causal attention must not cross packed segment
boundaries. Each kernel is compared against the dot_product reference on the
same packed batch for both the forward output and the gradient wrt query.
"""

import sys
import unittest
from unittest import mock

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import attention_op
from maxtext.layers.attention_op import AttentionOp
from maxtext.utils import maxtext_utils

from tests.utils.test_helpers import get_test_config_path

BATCH = 2
SEQ = 512
NUM_Q_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 128
# bf16 flash kernels differ from the unfused reference by ~1% of magnitude.
RELATIVE_TOL = 5e-2


class GpuAttentionPackingTest(parameterized.TestCase):
  """Compares GPU attention kernels against dot_product on packed batches."""

  def setUp(self):
    """Builds a packing-enabled config and shared bf16 query/key/value tensors."""
    super().setUp()
    self.config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="gpu_attention_packing_test",
        enable_checkpointing=False,
        max_target_length=SEQ,
        attention="dot_product",
        packing=True,
        max_segments_per_seq=4,
    )
    # Single-device mesh: this test checks kernel masking correctness against
    # a reference, not multi-device sharding, and must run on any GPU count.
    devices_array = maxtext_utils.create_device_mesh(self.config, devices=[jax.devices()[0]])
    self.mesh = Mesh(devices_array, self.config.mesh_axes)

    rng_q, rng_k, rng_v = jax.random.split(jax.random.PRNGKey(0), 3)
    # Queries are pre-scaled like the Attention layer does (kernels run with scale=1.0).
    self.query = jax.random.normal(rng_q, (BATCH, SEQ, NUM_Q_HEADS, HEAD_DIM), dtype=jnp.bfloat16) * (HEAD_DIM**-0.5)
    self.key = jax.random.normal(rng_k, (BATCH, SEQ, NUM_KV_HEADS, HEAD_DIM), dtype=jnp.bfloat16)
    self.value = jax.random.normal(rng_v, (BATCH, SEQ, NUM_KV_HEADS, HEAD_DIM), dtype=jnp.bfloat16)

  def build_op(self, kernel):
    return AttentionOp(
        config=self.config,
        num_query_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        max_target_length=SEQ,
        mesh=self.mesh,
        attention_kernel=kernel,
        dtype=jnp.bfloat16,
        dropout_rate=0.0,
    )

  def packed_inputs(self, boundary):
    """Two packed segments per row split at `boundary`, positions restarting per segment."""
    segment_ids = jnp.concatenate(
        [jnp.full((BATCH, boundary), 1, jnp.int32), jnp.full((BATCH, SEQ - boundary), 2, jnp.int32)], axis=1
    )
    positions = jnp.concatenate([jnp.arange(boundary, dtype=jnp.int32), jnp.arange(SEQ - boundary, dtype=jnp.int32)])[
        None, :
    ].repeat(BATCH, axis=0)
    return segment_ids, positions

  def unpacked_inputs(self):
    segment_ids = jnp.ones((BATCH, SEQ), jnp.int32)
    positions = jnp.arange(SEQ, dtype=jnp.int32)[None, :].repeat(BATCH, axis=0)
    return segment_ids, positions

  def fwd_and_grad(self, op, segment_ids, positions):
    def loss(query):
      out = op(query, self.key, self.value, segment_ids, positions, model_mode=MODEL_MODE_TRAIN)
      return jnp.sum(out.astype(jnp.float32) ** 2)

    out = op(self.query, self.key, self.value, segment_ids, positions, model_mode=MODEL_MODE_TRAIN)
    grad = jax.grad(loss)(self.query)
    return np.asarray(out.astype(jnp.float32)), np.asarray(grad.astype(jnp.float32))

  def assert_matches_reference(self, kernel, segment_ids, positions, boundary):
    """Asserts forward and grad outputs of `kernel` match dot_product on both sides of the boundary."""
    if jax.local_devices()[0].platform != "gpu":
      self.skipTest("Requires a GPU.")
    reference_op = self.build_op("dot_product")
    test_op = self.build_op(kernel)
    for name, ref, out in zip(
        ("forward", "grad_q"),
        self.fwd_and_grad(reference_op, segment_ids, positions),
        self.fwd_and_grad(test_op, segment_ids, positions),
    ):
      scale = max(np.abs(ref).max(), 1e-6)
      # Cross-segment leaks show up specifically after the boundary, so report both halves.
      first = np.abs(out[:, :boundary] - ref[:, :boundary]).max() / scale
      second = (np.abs(out[:, boundary:] - ref[:, boundary:]).max() / scale) if boundary < SEQ else 0.0
      self.assertLess(
          max(first, second),
          RELATIVE_TOL,
          f"{kernel} {name} deviates from dot_product: "
          f"rel diff before boundary={first:.4f}, after boundary={second:.4f}",
      )

  @pytest.mark.gpu_only
  @parameterized.named_parameters(
      ("unpacked", SEQ),
      ("packed", SEQ // 2),
      ("packed_unaligned", SEQ // 2 - 4),
  )
  def test_cudnn_flash_jax_matches_reference(self, boundary):
    if boundary == SEQ:
      segment_ids, positions = self.unpacked_inputs()
    else:
      segment_ids, positions = self.packed_inputs(boundary)
    self.assert_matches_reference("cudnn_flash_jax", segment_ids, positions, boundary)

  @pytest.mark.gpu_only
  @parameterized.named_parameters(
      ("unpacked", SEQ),
      ("packed", SEQ // 2),
      ("packed_unaligned", SEQ // 2 - 4),
  )
  def test_cutlass_flash_matches_reference(self, boundary):
    if attention_op.cutlass_flash_lib is None:
      self.skipTest("flash-attn-jax is not installed.")
    if boundary == SEQ:
      segment_ids, positions = self.unpacked_inputs()
    else:
      segment_ids, positions = self.packed_inputs(boundary)
    self.assert_matches_reference("cutlass_flash", segment_ids, positions, boundary)

  @pytest.mark.gpu_only
  @parameterized.named_parameters(
      ("unpacked", SEQ),
      ("packed", SEQ // 2),
      ("packed_unaligned", SEQ // 2 - 4),
  )
  def test_pallas_flash_matches_reference(self, boundary):
    if boundary == SEQ:
      segment_ids, positions = self.unpacked_inputs()
    else:
      segment_ids, positions = self.packed_inputs(boundary)
    # Force the pallas backend by hiding the cutlass library from the dispatch.
    with mock.patch.object(attention_op, "cutlass_flash_lib", None):
      self.assert_matches_reference("flash", segment_ids, positions, boundary)

  def test_segment_ids_to_seqlens_offsets(self):
    """Checks the cuDNN seqlens/offsets encoding of packed segment ids on the host."""
    op = self.build_op("cudnn_flash_jax")
    # Row 0: segments of 3 and 4 tokens plus one padding token; row 1: a single segment.
    segment_ids = jnp.asarray([[1, 1, 1, 2, 2, 2, 2, 0], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=jnp.int32)
    seqlens, offsets = op._segment_ids_to_seqlens_offsets(segment_ids)  # pylint: disable=protected-access
    np.testing.assert_array_equal(np.asarray(seqlens), [[3, 4, -1, -1], [8, -1, -1, -1]])
    np.testing.assert_array_equal(np.asarray(offsets), [[0, 3, -1, -1, -1], [0, -1, -1, -1, -1]])


if __name__ == "__main__":
  unittest.main()
