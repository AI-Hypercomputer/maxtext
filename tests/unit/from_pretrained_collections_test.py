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

"""Real save -> `from_pretrained` restore across on-disk collection layouts.

A weight typed as a custom `nnx.Variable` subclass (here DeepSeek's routed gate
bias, a `MoEBiasVar`) lives in its own Flax collection. Checkpoints store it
either in that collection (`params/MoEBiasVar/...`, as the DeepSeek-V4
converter writes) or, if they predate the promotion, inside `params/params`.
`from_pretrained` must restore it from both, not silently keep the initializer.
"""

import shutil
import sys
import tempfile
import unittest

from flax import nnx
import jax
import numpy as np
from jax.sharding import Mesh
from orbax import checkpoint as ocp

from maxtext.common import checkpointing
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path

_MOE_BIAS = "MoEBiasVar"
_TRANSIENT = (nnx.RngState, nnx.Cache, nnx.Intermediate, nnx.BatchStat)


def _make_config(**kwargs):
  """Tiny DeepSeek-3 MoE config: routed gate bias is a `MoEBiasVar`."""
  defaults = {
      "model_name": "deepseek3-tiny",
      "override_model_config": True,
      "run_name": "from_pretrained_collections_test",
      "per_device_batch_size": 1.0,
      "max_target_length": 16,
      "max_prefill_predict_length": 4,
      "attention": "dot_product",
      "scan_layers": False,
      "base_num_decoder_layers": 2,
      "first_num_dense_layers": 1,
      "vocab_size": 256,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "enable_checkpointing": False,
  }
  defaults.update(kwargs)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **defaults)


def _weights_by_collection(model):
  """`{collection: nested numpy tree}` of the model's persistent weights."""
  state = nnx.state(model).filter(lambda _, v: not isinstance(v, _TRANSIENT))
  grouped = checkpointing._group_leaves_by_collection(state)  # pylint: disable=protected-access
  return jax.tree.map(np.asarray, grouped)


def _flat(tree):
  return dict(nnx.traversals.flatten_mapping(tree))


class TestFromPretrainedCollections(unittest.TestCase):
  """`from_pretrained` restores custom collections from both on-disk layouts."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.config = _make_config()
    cls.mesh = Mesh(maxtext_utils.create_device_mesh(cls.config), cls.config.mesh_axes)
    fresh = _weights_by_collection(model_creation_utils.from_pretrained(cls.config, cls.mesh))
    assert _MOE_BIAS in fresh, f"test model has no {_MOE_BIAS} collection: {sorted(fresh)}"
    # Every saved value differs from a fresh initialization, so a weight that is never
    # restored is caught by value, not just by shape.
    cls.saved = {
        col: jax.tree.map(lambda x: (np.asarray(x, np.float32) * 2.0 + 0.25).astype(x.dtype), tree)
        for col, tree in fresh.items()
    }

  def setUp(self):
    self._dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)

  def _save(self, collections_payload):
    path = self._dir + "/ckpt"
    ocp.Checkpointer(ocp.PyTreeCheckpointHandler()).save(path, {"params": collections_payload})
    return path

  def _load(self, path):
    cfg = _make_config(enable_checkpointing=True, load_parameters_path=path)
    return _weights_by_collection(model_creation_utils.from_pretrained(cfg, self.mesh))

  def _assert_all_weights_restored(self, restored):
    for col, tree in self.saved.items():
      got, want = _flat(restored[col]), _flat(tree)
      self.assertEqual(set(got), set(want), f"collection {col}: leaf paths differ")
      for path, value in want.items():
        np.testing.assert_array_equal(got[path], value, err_msg=f"{col}/{'/'.join(map(str, path))}")

  def test_split_collection_layout_restores_custom_collection(self):
    """`params/MoEBiasVar/...` alongside `params/params/...` (DeepSeek-V4 converter layout)."""
    restored = self._load(self._save(self.saved))
    self._assert_all_weights_restored(restored)

  def test_legacy_layout_restores_promoted_weight_from_params(self):
    """Promoted weight stored inside `params/params` by a checkpoint that predates it."""
    legacy = {}
    for tree in self.saved.values():
      legacy = checkpointing._deep_merge_dicts(legacy, tree)  # pylint: disable=protected-access
    restored = self._load(self._save({"params": legacy}))
    self._assert_all_weights_restored(restored)

  def test_stale_legacy_copy_does_not_override_custom_collection(self):
    """With both a stale `params` copy and the custom collection on disk, the collection wins."""
    stale = jax.tree.map(np.zeros_like, self.saved[_MOE_BIAS])
    params_with_stale_copy = checkpointing._deep_merge_dicts(self.saved["params"], stale)  # pylint: disable=protected-access
    payload = dict(self.saved)
    payload["params"] = params_with_stale_copy
    restored = self._load(self._save(payload))
    self._assert_all_weights_restored(restored)


class TestCollectionHelpers(unittest.TestCase):
  """The shared checkpointing helpers `from_pretrained` and `load_params_from_path` use."""

  def test_merge_puts_params_first_so_custom_collections_win(self):
    merged = checkpointing._merge_restored_collections(  # pylint: disable=protected-access
        {"MoEBiasVar": {"gate": {"bias": 2}}, "params": {"gate": {"bias": 1, "kernel": 3}}}
    )
    self.assertEqual(merged, {"gate": {"bias": 2, "kernel": 3}})

  def test_merge_keeps_non_dict_collections(self):
    merged = checkpointing._merge_restored_collections({"params": {"a": 1}, "step": 7})  # pylint: disable=protected-access
    self.assertEqual(merged, {"a": 1, "step": 7})


if __name__ == "__main__":
  unittest.main()
