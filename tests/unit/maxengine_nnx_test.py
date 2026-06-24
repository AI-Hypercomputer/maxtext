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

"""Unit tests for NNX dispatch in maxengine (no jetstream / checkpoint needed)."""

import sys
import unittest

import jax

from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
from tests.utils.test_helpers import get_test_config_path


class SetEngineVarsNNXTest(unittest.TestCase):
  """set_engine_vars_from_base_engine must work on the NNX path."""

  def _nnx_config(self, **kwargs):
    init_kwargs = {
        "base_emb_dim": 32,
        "base_num_query_heads": 2,
        "base_num_kv_heads": 2,
        "base_num_decoder_layers": 2,
        "max_prefill_predict_length": 4,
        "max_target_length": 8,
        "per_device_batch_size": 1,
        "enable_checkpointing": False,
        "pure_nnx": True,
        "enable_nnx": True,
        "pure_nnx_decoder": True,
    } | kwargs
    return pyconfig.initialize([sys.argv[0], get_test_config_path()], **init_kwargs)

  def test_set_engine_vars_from_base_engine_nnx(self):
    """NNX dispatches to get_kv_cache_annotations_nnx; the Linen model.init() path AttributeErrors.

    state_mesh_annotations / abstract_params are merely copied from the base engine,
    so they're stubbed here — that lets the test exercise the kv-cache-annotations
    dispatch without loading a checkpoint.
    """
    cfg = self._nnx_config()
    engine = maxengine.MaxEngine(cfg, jax.devices())
    engine.state_mesh_annotations = None
    engine.abstract_params = None

    maxengine.set_engine_vars_from_base_engine(engine, engine, jax.random.PRNGKey(0))

    self.assertIsNotNone(engine.kv_cache_annotations)
    self.assertIsNotNone(engine.kv_cache_shardings)
    self.assertGreater(len(jax.tree_util.tree_leaves(engine.kv_cache_annotations)), 0)


if __name__ == "__main__":
  unittest.main()
