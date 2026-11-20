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

"""dual_pipe on a real (tiny) DeepSeek-V4 model matches serial gradient accumulation."""

# pylint: disable=missing-function-docstring,protected-access

import functools
import json
import sys
import unittest

from flax import nnx
from flax import traverse_util
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.experimental.dualpipe_nnx import _layer_groups, dualpipe_loss_and_grad
from maxtext.trainers.pre_train import train as pre_train
from maxtext.utils import maxtext_utils, model_creation_utils
from maxtext.utils.gradient_accumulation import gradient_accumulation_loss_and_grad
from tests.utils.test_helpers import get_test_config_path

pytestmark = pytest.mark.cpu_only

# Two unscanned hash-routed prefix layers (sliding-window + CSA) plus one scanned block of two layers (HCA + CSA),
# with hyper-connections, a routed bias, the MoE load-balance loss and the CSA indexer loss all on.
_V4_SMALL = {
    "model_name": "deepseek4-tiny",
    "override_model_config": True,
    "first_num_hash_layers": 2,
    "base_num_decoder_layers": 4,
    "compress_ratios": [0, 4, 128, 4],
    "mhc_expansion_rate": 2,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 128,
    "max_target_length": 16,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "scan_layers": True,
    "attention": "dot_product",
    "sparse_matmul": False,
    "megablox": False,
    "dtype": "float32",
    "weight_dtype": "float32",
    "matmul_precision": "highest",
    "enable_checkpointing": False,
    "enable_dropout": False,
    "remat_policy": "full",
    "indexer_sparse_training": True,
    "indexer_loss_scaling_factor": 0.5,
    "indexer_topk": 2,
    "load_balance_loss_weight": 0.05,
    "routed_bias_update_rate": 0.01,
}


def _params_shardings(model):
  _, params, _ = nnx.split(model, nnx.Param, ...)
  replicated = jax.sharding.NamedSharding(model.mesh, jax.sharding.PartitionSpec())
  return jax.tree.map(lambda _: replicated, params)


def _flat(tree):
  """{path: array} of a nested dict or NNX state; sown values are tuples, so take their single array."""
  tree = tree.to_pure_dict() if isinstance(tree, nnx.State) else tree
  return {
      path: np.asarray(jax.tree.leaves(leaf)[0]) if isinstance(leaf, (tuple, list)) else np.asarray(leaf)
      for path, leaf in traverse_util.flatten_dict(tree).items()
  }


def _run(overrides=None, serial=True):
  """(config, serial, dual_pipe) results, each (loss, aux, grads), on the same tiny V4 model and batch.

  Memoized: building the model and compiling both paths dominates the cost, and several tests share a configuration.
  """
  return _run_cached(json.dumps(overrides or {}, sort_keys=True), serial)


@functools.lru_cache(maxsize=None)
def _run_cached(overrides_json, serial):
  overrides = json.loads(overrides_json)
  cfg = pyconfig.initialize([sys.argv[0], get_test_config_path()], **{**_V4_SMALL, **overrides})
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
    model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    rows, seq = cfg.gradient_accumulation_steps * cfg.micro_batch_size_to_train_on, cfg.max_target_length
    ids = jax.random.randint(jax.random.PRNGKey(1), (rows, seq + 1), 1, cfg.vocab_size)
    # Unequal valid-token counts per microbatch: the tail of the second row is padding.
    segmentation = jnp.ones((rows, seq), jnp.int32).at[-1, seq // 2 :].set(0)
    data = {
        "inputs": ids[:, :-1],
        "inputs_position": jnp.tile(jnp.arange(seq)[None], (rows, 1)),
        "inputs_segmentation": segmentation,
        "targets": ids[:, 1:],
        "targets_segmentation": segmentation,
    }
    shardings = _params_shardings(model)
    serial = (
        gradient_accumulation_loss_and_grad(pre_train.loss_fn, cfg, nnx.clone(model), None, shardings, dict(data), None)
        if serial
        else None
    )
    dual = dualpipe_loss_and_grad(cfg, nnx.clone(model), shardings, dict(data), pre_train.loss_from_logits)
  return cfg, serial, dual


class DualPipeDeepSeek4Test(unittest.TestCase):

  @classmethod
  def tearDownClass(cls):
    _run_cached.cache_clear()
    super().tearDownClass()

  def assert_close(self, actual, expected, name, rtol=2e-4, atol=2e-6):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)

  def test_loss_aux_and_gradients_match_serial(self):
    cfg, (loss_s, aux_s, grads_s), (loss_d, aux_d, grads_d) = _run()
    self.assertGreater(float(aux_s["moe_lb_loss"]), 0.0)
    self.assertGreater(float(aux_s["indexer_loss"]), 0.0)
    self.assert_close(loss_d, loss_s, "loss")
    for key in ("xent_sum", "total_weights", "moe_lb_loss", "indexer_loss"):
      self.assert_close(aux_d[key], aux_s[key], key)
    got, want = _flat(grads_d), _flat(grads_s)
    self.assertEqual(set(got), set(want))
    for path in want:
      self.assert_close(got[path], want[path], "/".join(map(str, path)), rtol=1e-3, atol=1e-5)
    self.assertEqual(cfg.gradient_accumulation_steps, 2)

  def test_routed_bias_updates_match_serial(self):
    cfg, (_, aux_s, _), (_, aux_d, _) = _run()
    want = {p: v for p, v in _flat(aux_s["intermediate_outputs"]).items() if p[-1] == "moe_bias_updates"}
    got = {p: v for p, v in _flat(aux_d["intermediate_outputs"]).items() if p[-1] == "moe_bias_updates"}
    self.assertEqual(set(want), {("decoder", "scanned_blocks", f"layers_{i}", "moe_bias_updates") for i in range(2)})
    self.assertEqual(set(got), set(want))
    for path, serial in want.items():
      # Serial gradient accumulation leaves V4's summed expert counts unconverted here (its finalizer matches the
      # tuple-wrapped sown value's path by dict key and never fires), so train_step would add raw counts to the router
      # bias. dual_pipe returns the intended rate * sign(mean load - load) step; compare it to that.
      if np.issubdtype(serial.dtype, np.integer):
        serial = np.asarray(
            pre_train.moe.load_balance_updates_from_counts(serial, cfg.num_experts, cfg.routed_bias_update_rate)
        )
      self.assert_close(got[path], serial, "/".join(map(str, path)))
      self.assertEqual(got[path].dtype, np.float32)
      self.assertTrue(np.all(np.abs(got[path]) == np.float32(cfg.routed_bias_update_rate)))

  @pytest.mark.scheduled_only
  def test_auxiliary_losses_change_the_gradients(self):
    """The comparison above is not vacuous: the load-balance and indexer terms move the gradients."""
    _, _, (_, _, with_aux) = _run(serial=False)
    off = {"load_balance_loss_weight": 0.0, "indexer_loss_scaling_factor": 0.0}
    _, _, (_, aux, without_aux) = _run(off, serial=False)
    self.assertEqual(float(aux["moe_lb_loss"]), 0.0)
    self.assertEqual(float(aux["indexer_loss"]), 0.0)
    got, base = _flat(with_aux), _flat(without_aux)
    self.assertGreater(max(float(np.max(np.abs(got[path] - base[path]))) for path in got), 1e-5)

  @pytest.mark.scheduled_only
  def test_odd_remainder_ignores_the_trailing_layer_like_serial(self):
    """5 layers = 2 prefix + 3 remaining: the decoder scans floor(3 / 2) = 1 block and never builds the last layer."""
    overrides = {"base_num_decoder_layers": 5, "compress_ratios": [0, 4, 128, 4, 128]}
    cfg, (loss_s, aux_s, grads_s), (loss_d, aux_d, grads_d) = _run(overrides)
    self.assertEqual(_layer_groups(cfg), (("layers_0", 1), ("layers_1", 1), ("scanned_blocks", 1)))
    self.assert_close(loss_d, loss_s, "loss")
    for key in ("moe_lb_loss", "indexer_loss"):
      self.assert_close(aux_d[key], aux_s[key], key)
    got, want = _flat(grads_d), _flat(grads_s)
    self.assertEqual(set(got), set(want))
    for path in want:
      self.assert_close(got[path], want[path], "/".join(map(str, path)), rtol=1e-3, atol=1e-5)

  @pytest.mark.scheduled_only
  def test_prefix_only_model_without_scanned_blocks(self):
    """2 prefix layers + 1 remaining: floor(1 / 2) = 0 blocks, so there is no scanned stack and no routed bias."""
    overrides = {"base_num_decoder_layers": 3, "compress_ratios": [0, 4, 128]}
    cfg, (loss_s, aux_s, grads_s), (loss_d, aux_d, grads_d) = _run(overrides)
    self.assertEqual(_layer_groups(cfg), (("layers_0", 1), ("layers_1", 1)))
    self.assert_close(loss_d, loss_s, "loss")
    # Hash-routed layers have no load-balance loss or bias; the CSA prefix layer still has an indexer loss.
    self.assertEqual(float(aux_d["moe_lb_loss"]), 0.0)
    self.assert_close(aux_d["indexer_loss"], aux_s["indexer_loss"], "indexer_loss")
    self.assertEqual(aux_d["intermediate_outputs"], {})
    got, want = _flat(grads_d), _flat(grads_s)
    self.assertEqual(set(got), set(want))
    for path in want:
      self.assert_close(got[path], want[path], "/".join(map(str, path)), rtol=1e-3, atol=1e-5)

  @pytest.mark.scheduled_only
  def test_remat_policies_match_serial(self):
    """Without remat and with a name-saving policy (the scanned block also remats each layer internally)."""
    for policy in ("none", "minimal"):
      with self.subTest(remat_policy=policy):
        _, (loss_s, aux_s, grads_s), (loss_d, aux_d, grads_d) = _run({"remat_policy": policy})
        self.assert_close(loss_d, loss_s, "loss")
        for key in ("moe_lb_loss", "indexer_loss"):
          self.assert_close(aux_d[key], aux_s[key], key)
        got, want = _flat(grads_d), _flat(grads_s)
        self.assertEqual(set(got), set(want))
        for path in want:
          self.assert_close(got[path], want[path], "/".join(map(str, path)), rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
