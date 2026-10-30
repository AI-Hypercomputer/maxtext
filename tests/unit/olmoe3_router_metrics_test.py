# Copyright 2023-2026 Google LLC
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

"""``olmoe3_router_metrics``: the per-layer rows come out in layer order, scanned or not.

    JAX_PLATFORMS=cpu pytest tests/unit/olmoe3_router_metrics_test.py
"""

import os
import unittest

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.layers import quantizations
from maxtext.trainers.pre_train.train import _collect_olmoe3_router_metrics, _olmoe3_router_metric_rows
from maxtext.utils import maxtext_utils, model_creation_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR

# olmoe3-30m at 15 layers: a dense layer 0, then 14 MoE layers in cycles of 5 (the scanned model unrolls the first
# cycle and scans the other two).
LAYERS, INTERVAL = 15, 5


def _row(layer):
  return jnp.full((1, 4), float(layer))


def _intermediates(scan_layers):
  """Router metrics sown by a 15-layer olmoe3-30m forward pass."""
  cfg = pyconfig.initialize(
      [
          "",
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          "model_name=olmoe3-30m",
          "run_name=olmoe3_router_metrics_test",
          "enable_checkpointing=False",
          f"scan_layers={scan_layers}",
          "per_device_batch_size=1",
          "max_target_length=64",
          "override_model_config=True",
          f"base_num_decoder_layers={LAYERS}",
          "olmoe3_router_metrics=True",
          "vocab_size=512",
          "emo_enabled=False",
      ]
  )
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = model_creation_utils.get_transformer_model(
      cfg, mesh, quantizations.configure_quantization(cfg), rngs=nnx.Rngs(0)
  )
  ids = jnp.ones((1, 64), jnp.int32)
  model(
      decoder_input_tokens=ids,
      decoder_positions=jnp.arange(64)[None],
      decoder_segment_ids=ids,
      enable_dropout=False,
      decoder_target_tokens=ids,
      decoder_target_mask=ids,
  )
  return nnx.pop(model, nnx.Intermediate).to_pure_dict()


class RouterMetricOrderTest(unittest.TestCase):
  """Rows sorted by global layer index: key order puts layers_10 before layers_2, and a scanned leaf holds a row per
  scanned cycle of one sublayer."""

  def test_unscanned_paths(self):
    tree = {"decoder": {f"layers_{i}": {"olmoe3_router_metrics": (_row(i),)} for i in range(1, LAYERS)}}
    rows = _collect_olmoe3_router_metrics(tree, INTERVAL)
    np.testing.assert_array_equal(np.asarray(rows[:, 0]), np.arange(1, LAYERS))

  def test_scanned_paths(self):
    # The unrolled first cycle (layer 0 dense, no metrics), then [cycles, 4] leaves for scanned sublayer j.
    first = {f"layer_{j}": {"olmoe3_router_metrics": (_row(j),)} for j in range(1, INTERVAL)}
    scanned = {
        f"layer_{j}": {"olmoe3_router_metrics": (jnp.concatenate([_row(INTERVAL + j), _row(2 * INTERVAL + j)]),)}
        for j in range(INTERVAL)
    }
    tree = {"decoder": {"layers_0": first, "scanned_blocks": scanned}}
    rows = _collect_olmoe3_router_metrics(tree, INTERVAL)
    np.testing.assert_array_equal(np.asarray(rows[:, 0]), np.arange(1, LAYERS))

  def test_none_without_metrics(self):
    self.assertIsNone(_collect_olmoe3_router_metrics({"decoder": {}}, INTERVAL))

  def test_model_paths_give_every_moe_layer_once(self):
    # The real sown paths, scanned and not: one row for each of layers 1..14, in order.
    for scan in (False, True):
      layers = [i for i, _ in _olmoe3_router_metric_rows(_intermediates(scan), INTERVAL)]
      self.assertEqual(layers, list(range(1, LAYERS)), f"scan_layers={scan}")


if __name__ == "__main__":
  unittest.main()
