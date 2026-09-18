# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for tid2eid hash-routing table sideloading and post-restore validation."""

import os
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import jax
from jax.sharding import Mesh
import numpy as np
import pytest
from safetensors.numpy import save_file

from maxtext.utils.model_creation_utils import _get_moe_block, _sideload_tid2eid, _validate_tid2eid

pytestmark = pytest.mark.cpu_only

_NUM_EXPERTS = 8


def _layer(table):
  """Builds a stand-in decoder layer whose MoE block holds `table`, or has no table at all."""
  tid2eid = None if table is None else SimpleNamespace(value=table)
  return SimpleNamespace(mlp=SimpleNamespace(MoeBlock_0=SimpleNamespace(tid2eid=tid2eid)))


def _model(*layers):
  return SimpleNamespace(decoder=SimpleNamespace(**{f"layers_{i}": layer for i, layer in enumerate(layers)}))


def _valid_table(seed):
  return np.arange(seed, seed + 4, dtype=np.float32) % _NUM_EXPERTS


class GetMoeBlockTest(unittest.TestCase):
  """Tests the unscanned-decoder-layer MoE block lookup."""

  def test_returns_the_moe_block_when_present(self):
    model = _model(_layer(_valid_table(1)))

    self.assertIs(_get_moe_block(model, 0), model.decoder.layers_0.mlp.MoeBlock_0)

  def test_returns_none_for_a_missing_layer(self):
    self.assertIsNone(_get_moe_block(_model(_layer(_valid_table(1))), 3))

  def test_returns_none_for_a_layer_without_an_mlp(self):
    model = SimpleNamespace(decoder=SimpleNamespace(layers_0=SimpleNamespace()))

    self.assertIsNone(_get_moe_block(model, 0))


class ValidateTid2EidTest(unittest.TestCase):
  """Tests the post-restore non-degeneracy check on the hash-routing tables."""

  def test_accepts_tables_with_in_range_nonzero_ids(self):
    model = _model(_layer(_valid_table(1)), _layer(_valid_table(2)))

    _validate_tid2eid(model, num_hash_layers=2, num_experts=_NUM_EXPERTS)

  def test_rejects_an_all_zero_table(self):
    model = _model(_layer(_valid_table(1)), _layer(np.zeros(4, dtype=np.float32)))

    with self.assertRaisesRegex(ValueError, "layer 1 has a missing or zero-initialized tid2eid table"):
      _validate_tid2eid(model, num_hash_layers=2, num_experts=_NUM_EXPERTS)

  def test_rejects_a_missing_table(self):
    model = _model(_layer(None))

    with self.assertRaisesRegex(ValueError, "layer 0 has a missing or zero-initialized tid2eid table"):
      _validate_tid2eid(model, num_hash_layers=1, num_experts=_NUM_EXPERTS)

  def test_rejects_a_missing_layer(self):
    model = _model(_layer(_valid_table(1)))

    with self.assertRaisesRegex(ValueError, "layer 1 has a missing or zero-initialized tid2eid table"):
      _validate_tid2eid(model, num_hash_layers=2, num_experts=_NUM_EXPERTS)

  def test_rejects_ids_at_or_above_num_experts(self):
    model = _model(_layer(np.array([0, 1, _NUM_EXPERTS], dtype=np.float32)))

    with self.assertRaisesRegex(ValueError, r"ids out of range \[0, 8\] for num_experts=8"):
      _validate_tid2eid(model, num_hash_layers=1, num_experts=_NUM_EXPERTS)

  def test_rejects_negative_ids(self):
    model = _model(_layer(np.array([-1, 5], dtype=np.float32)))

    with self.assertRaisesRegex(ValueError, r"ids out of range \[-1, 5\] for num_experts=8"):
      _validate_tid2eid(model, num_hash_layers=1, num_experts=_NUM_EXPERTS)


class SideloadTid2EidTest(unittest.TestCase):
  """Tests populating the hash-routing tables from an external safetensors file."""

  def setUp(self):
    super().setUp()
    self.mesh = Mesh(jax.devices(), ("data",))
    self.tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp_dir)

  def _write(self, tables):
    path = os.path.join(self.tmp_dir, "tid2eid.safetensors")
    save_file(tables, path)
    return path

  def test_populates_every_hash_routing_layer(self):
    tables = {"layers_0": _valid_table(1), "layers_1": _valid_table(2)}
    model = _model(_layer(np.zeros(4, dtype=np.float32)), _layer(np.zeros(4, dtype=np.float32)))

    _sideload_tid2eid(model, self.mesh, self._write(tables), num_hash_layers=2)

    for layer_idx in range(2):
      loaded = _get_moe_block(model, layer_idx).tid2eid.value
      self.assertEqual(loaded.dtype, np.float32)
      np.testing.assert_array_equal(np.asarray(loaded), tables[f"layers_{layer_idx}"])
    # The sideloaded tables must then satisfy the post-restore check.
    _validate_tid2eid(model, num_hash_layers=2, num_experts=_NUM_EXPERTS)

  def test_raises_when_the_file_is_missing_a_layer(self):
    model = _model(_layer(np.zeros(4, dtype=np.float32)), _layer(np.zeros(4, dtype=np.float32)))

    with self.assertRaisesRegex(ValueError, "expected 2 layer assignments, got 1"):
      _sideload_tid2eid(model, self.mesh, self._write({"layers_0": _valid_table(1)}), num_hash_layers=2)

  def test_raises_when_the_model_has_no_table_to_populate(self):
    model = _model(_layer(None))

    with self.assertRaisesRegex(ValueError, "expected 1 layer assignments, got 0"):
      _sideload_tid2eid(model, self.mesh, self._write({"layers_0": _valid_table(1)}), num_hash_layers=1)


if __name__ == "__main__":
  unittest.main()
