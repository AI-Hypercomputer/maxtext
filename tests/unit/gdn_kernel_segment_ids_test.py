# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests which segment IDs the GDN layer hands to the Pallas kernel."""

import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.kernels.gdn import gdn_bwd_pallas
from maxtext.kernels.gdn import model_runner
from maxtext.models import qwen3
from tests.utils.test_helpers import get_test_config_path

_BATCH, _SEQ_LEN = 2, 128


def _gdn_kernel_config(packing):
  """A minimal Qwen3-Next config on the GDN Pallas kernel path, sized for one CPU device."""
  return pyconfig.initialize(
      [
          None,
          get_test_config_path(),
          "run_name=gdn_kernel_segment_ids_test",
          "dtype=float32",
          "weight_dtype=float32",
          "decoder_block=qwen3_next",
          "attention=dot_product",
          "base_emb_dim=64",
          "base_num_query_heads=2",
          "base_num_kv_heads=2",
          "head_dim=32",
          "gdn_num_key_heads=1",
          "gdn_num_value_heads=2",
          "gdn_key_head_dim=128",
          "gdn_value_head_dim=128",
          "gdn_conv_kernel_dim=4",
          "gdn_chunk_size=64",
          f"max_target_length={_SEQ_LEN}",
          "use_gdn_kernel=True",
          f"packing={packing}",
      ],
      skip_jax_distributed_system=True,
  )


def _layer(packing, model_mode=MODEL_MODE_TRAIN):
  cfg = _gdn_kernel_config(packing)
  mesh = Mesh(np.array(jax.devices()[:1]).reshape([1] * len(cfg.mesh_axes)), cfg.mesh_axes)
  return qwen3.Qwen3NextGatedDeltaNet(
      config=cfg,
      mesh=mesh,
      model_mode=model_mode,
      inputs_shape=(_BATCH, _SEQ_LEN, cfg.emb_dim),
      rngs=nnx.Rngs(0),
  )


def _hidden_states(layer):
  return jax.random.normal(jax.random.PRNGKey(0), (_BATCH, _SEQ_LEN, layer.config.emb_dim), jnp.float32)


def _run(layer, segment_ids, model_mode=MODEL_MODE_TRAIN):
  """Runs the layer, returning its output and the segment IDs the kernel was called with."""
  with mock.patch.object(model_runner, "gdn_decoupled_conv1d", wraps=model_runner.gdn_decoupled_conv1d) as kernel:
    out, _ = layer(_hidden_states(layer), model_mode=model_mode, decoder_segment_ids=segment_ids)
  kernel.assert_called_once()
  return np.asarray(out), kernel.call_args.kwargs["segment_ids"]


def _unpacked_segment_ids():
  """One sequence per row: row 0 is left- and right-padded, row 1 right-padded."""
  seg = np.zeros((_BATCH, _SEQ_LEN), np.int32)
  seg[0, 8:108] = 1
  seg[1, :96] = 1
  return jnp.asarray(seg)


class GdnKernelSegmentIdsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    gdn_bwd_pallas.ensure_cpu_interpret_registered()

  def test_unpacked_training_skips_segment_path(self):
    seg = _unpacked_segment_ids()
    out_unpacked, kernel_seg_unpacked = _run(_layer(packing=False), seg)
    out_packed, kernel_seg_packed = _run(_layer(packing=True), seg)
    self.assertIsNone(kernel_seg_unpacked)
    self.assertIsNotNone(kernel_seg_packed)
    # With one sequence per row the segment path is pure overhead: same output either way, up to fp32
    # rounding (the two kernel variants order some arithmetic differently; ~6e-6 on CPU).
    np.testing.assert_allclose(out_unpacked, out_packed, rtol=1e-4, atol=1e-4)

  def test_packed_training_keeps_document_boundaries(self):
    seg = np.ones((_BATCH, _SEQ_LEN), np.int32)
    seg[:, 50:] = 2
    layer = _layer(packing=True)
    out_two_docs, kernel_seg = _run(layer, jnp.asarray(seg))
    out_one_doc, _ = _run(layer, jnp.ones_like(jnp.asarray(seg)))
    self.assertIsNotNone(kernel_seg)
    # The second document must start from a fresh state, so only tokens after the boundary change.
    np.testing.assert_allclose(out_two_docs[:, :50], out_one_doc[:, :50], rtol=1e-5, atol=1e-5)
    self.assertGreater(np.max(np.abs(out_two_docs[:, 50:] - out_one_doc[:, 50:])), 1e-3)

  def test_unpacked_prefill_keeps_segment_ids(self):
    # Prefill carries its states forward, and the segment path ends the conv state at the last valid token.
    _, kernel_seg = _run(_layer(packing=False, model_mode=MODEL_MODE_PREFILL), _unpacked_segment_ids(), MODEL_MODE_PREFILL)
    self.assertIsNotNone(kernel_seg)


if __name__ == "__main__":
  unittest.main()
