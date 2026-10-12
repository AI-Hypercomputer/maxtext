# Copyright 2023-2026 Google LLC
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

"""DeepSeek-V4 train-mode rematerialization: same numerics, one forward recompute, policy honored."""

import functools
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils, model_creation_utils
from tests.utils.test_helpers import get_test_config_path

pytestmark = pytest.mark.cpu_only

_V4_SMALL = {
    "model_name": "deepseek4-tiny",
    "override_model_config": True,
    # Three unscanned prefix layers plus one scanned block of two layers.
    "base_num_decoder_layers": 5,
    "compress_ratios": [0, 0, 4, 128, 4],
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 128,
    "max_target_length": 32,
    "per_device_batch_size": 1,
    "scan_layers": True,
    "attention": "dot_product",
    "sparse_matmul": False,
    "megablox": False,
    "dtype": "float32",
    "weight_dtype": "float32",
    "matmul_precision": "highest",
    "enable_checkpointing": False,
    "enable_dropout": False,
    # The CSA indexer loss is created as a new variable during the layer call.
    "indexer_sparse_training": True,
    "indexer_loss_scaling_factor": 1.0,
    "indexer_topk": 4,
}


@functools.lru_cache(maxsize=None)
def _train_step(remat_policy, dropout=False):
  """Compiles one train-mode loss/grad step.

  Returns (loss, intermediates, grads, compiled FLOPs, RNG state after the step).
  """
  overrides = {**_V4_SMALL, "enable_dropout": True, "dropout_rate": 0.3} if dropout else _V4_SMALL
  cfg = pyconfig.initialize([sys.argv[0], get_test_config_path()], remat_policy=remat_policy, **overrides)
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
    model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
  batch, seq = cfg.micro_batch_size_to_train_on, cfg.max_target_length
  ids = jax.random.randint(jax.random.PRNGKey(1), (batch, seq), 1, cfg.vocab_size)
  positions = jnp.tile(jnp.arange(seq)[None], (batch, 1))
  segment_ids = jnp.ones((batch, seq), jnp.int32)
  graphdef, state = nnx.split(model)

  def loss_fn(m):
    logits = m(ids, positions, decoder_segment_ids=segment_ids, enable_dropout=dropout, model_mode=MODEL_MODE_TRAIN)
    logits = logits[0] if isinstance(logits, tuple) else logits
    ce = jnp.mean(jax.nn.logsumexp(logits, axis=-1) - jnp.take_along_axis(logits, ids[..., None], axis=-1)[..., 0])
    # Sown aux losses (MoE load balance, indexer) must survive the remat scan barrier.
    return ce, nnx.state(m, nnx.Intermediate)

  def step(state_in):
    m = nnx.merge(graphdef, state_in)
    (loss, inter), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m)
    return loss, inter, nnx.state(grads, nnx.Param), nnx.state(m, nnx.RngState)

  with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
    compiled = jax.jit(step).lower(state).compile()
    loss, inter, grads, rng_state = compiled(state)
  cost = compiled.cost_analysis()
  flops = float((cost[0] if isinstance(cost, (list, tuple)) else cost)["flops"])
  return (
      float(loss),
      [np.asarray(x) for x in jax.tree.leaves(inter)],
      [np.asarray(g) for g in jax.tree.leaves(grads)],
      flops,
      [np.asarray(x) for x in jax.tree.leaves(rng_state) if jnp.issubdtype(x.dtype, jnp.integer)],
  )


class DeepSeekV4RematTest(unittest.TestCase):

  def test_remat_preserves_loss_aux_and_grads(self):
    """remat_policy=full and none give the same loss, sown aux losses and gradients."""
    loss_full, inter_full, grads_full, _, _ = _train_step("full")
    loss_none, inter_none, grads_none, _, _ = _train_step("none")
    np.testing.assert_allclose(loss_full, loss_none, rtol=1e-6)
    self.assertGreater(len(inter_none), 0)
    self.assertEqual(len(inter_full), len(inter_none))
    for a, b in zip(inter_full, inter_none):
      np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
    self.assertEqual(len(grads_full), len(grads_none))
    for g_full, g_none in zip(grads_full, grads_none):
      np.testing.assert_allclose(g_full, g_none, rtol=1e-5, atol=1e-6)

  def test_remat_advances_dropout_rngs(self):
    """With dropout on, remat matches no remat in loss, gradients and the RNG counters it writes back."""
    loss_full, _, grads_full, _, rngs_full = _train_step("full", dropout=True)
    loss_none, _, grads_none, _, rngs_none = _train_step("none", dropout=True)
    np.testing.assert_allclose(loss_full, loss_none, rtol=1e-6)
    for g_full, g_none in zip(grads_full, grads_none):
      np.testing.assert_allclose(g_full, g_none, rtol=1e-5, atol=1e-6)
    # Dropout consumed keys, so the next step must not reuse the same masks.
    self.assertGreater(sum(int(c.sum()) for c in rngs_none), 0)
    self.assertEqual(len(rngs_full), len(rngs_none))
    for c_full, c_none in zip(rngs_full, rngs_none):
      np.testing.assert_array_equal(c_full, c_none)

  def test_full_remat_recomputes_the_forward_once(self):
    """Full remat adds one forward recompute: more FLOPs than no remat, fewer than two recomputes."""
    flops_none = _train_step("none")[3]
    flops_full = _train_step("full")[3]
    # Forward ~ 1/3 of a train step, so one recompute adds at most ~1/3 and two add ~2/3.
    self.assertGreater(flops_full, 1.05 * flops_none)
    self.assertLess(flops_full, 1.45 * flops_none)

  def test_remat_policy_is_honored(self):
    """A policy that saves named dots recomputes less than full remat."""
    self.assertLess(_train_step("minimal")[3], _train_step("full")[3])


if __name__ == "__main__":
  unittest.main()
