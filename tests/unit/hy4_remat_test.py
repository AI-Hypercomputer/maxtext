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

"""Unit tests for Hy4-preview custom rematerialization and Hy4Indexer backward DCE."""

from __future__ import annotations

import os

from absl.testing import absltest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import hy4
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

pytestmark = [pytest.mark.cpu_only]

BASE_CONFIG = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")

EMB, HEADS, HEAD_DIM = 32, 2, 16
Q_LORA, KV_LORA, NOPE, ROPE, V_DIM = 16, 8, 8, 8, 16
IDX_HEADS, IDX_DIM, IDX_TOPK = 4, 16, 3
MLP, MOE_MLP, EXPERTS, TOPK = 32, 16, 4, 2
HC_MULT = 4
VOCAB = 64
SWIGLU_LIMIT = 0.05


def _make_config(**kwargs):
  """Builds a minimal CPU MaxText hy4-preview config for remat tests."""
  overrides = dict(  # pylint: disable=use-dict-literal
      model_name="hy4-preview",
      override_model_config=True,
      base_emb_dim=EMB,
      base_num_query_heads=HEADS,
      base_num_kv_heads=HEADS,
      head_dim=HEAD_DIM,
      q_lora_rank=Q_LORA,
      kv_lora_rank=KV_LORA,
      qk_nope_head_dim=NOPE,
      qk_rope_head_dim=ROPE,
      v_head_dim=V_DIM,
      indexer_n_heads=IDX_HEADS,
      indexer_head_dim=IDX_DIM,
      indexer_topk=IDX_TOPK,
      use_indexer=True,
      mhc_expansion_rate=HC_MULT,
      base_mlp_dim=MLP,
      base_moe_mlp_dim=MOE_MLP,
      num_experts=EXPERTS,
      num_experts_per_tok=TOPK,
      shared_experts=1,
      routed_scaling_factor=2.827,
      routed_score_func="sigmoid",
      routed_bias=True,
      norm_topk_prob=True,
      n_routing_groups=-1,
      topk_routing_group=-1,
      mlp_activations_limit=SWIGLU_LIMIT,
      first_num_dense_layers=1,
      inhomogeneous_layer_cycle_interval=2,
      vocab_size=VOCAB,
      base_num_decoder_layers=3,
      per_device_batch_size=1,
      max_target_length=8,
      max_prefill_predict_length=4,
      max_position_embeddings=64,
      original_max_position_embeddings=64,
      scan_layers=False,
      attention="dot_product",
      attention_type="mla",
      attention_sink=True,
      sparse_matmul=False,
      capacity_factor=-1,
      matmul_precision="highest",
      dtype="float32",
      weight_dtype="float32",
      float32_logits=True,
      float32_qk_product=True,
      dropout_rate=0.0,
      skip_jax_distributed_system=True,
      hardware="cpu",
  )
  overrides.update(kwargs)
  return pyconfig.initialize(["", BASE_CONFIG], **overrides)


def _count_primitive(jaxpr, prim_name: str, **param_filters) -> int:
  """Recursively counts equations matching `prim_name` (and optional `eqn.params` filters) in `jaxpr`."""
  count = sum(
      (eqn.primitive.name == prim_name or eqn.params.get("name") == prim_name)
      and all(eqn.params.get(k) == v for k, v in param_filters.items())
      for eqn in jaxpr.eqns
  )
  for eqn in jaxpr.eqns:
    for sub in eqn.params.values():
      if isinstance(sub, jax.extend.core.ClosedJaxpr):
        count += _count_primitive(sub.jaxpr, prim_name, **param_filters)
      elif isinstance(sub, jax.extend.core.Jaxpr):
        count += _count_primitive(sub, prim_name, **param_filters)
  return count


class Hy4RematTest(absltest.TestCase):
  """Verifies Hy4Indexer topk_indices checkpoint tagging and backward VJP DCE."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.config = _make_config()
    cls.mesh = Mesh(maxtext_utils.create_device_mesh(cls.config), cls.config.mesh_axes)

  @classmethod
  def tearDownClass(cls):
    cls.config = None
    cls.mesh = None
    super().tearDownClass()

  def test_indexer_checkpoint_name_and_remat_dce(self):
    batch, seq = self.config.micro_batch_size_to_train_on, 6  # seq=6 > IDX_TOPK=3 triggers active top-k selection
    xj = jnp.asarray(np.random.RandomState(99).randn(batch, seq, EMB).astype(np.float32))
    pos_j = jnp.tile(jnp.arange(seq, dtype=jnp.int32)[None], (batch, 1))
    seg_j = jnp.ones((batch, seq), jnp.int32)

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.config.logical_axis_rules):
      layer = hy4.Hy4DecoderLayer(
          config=self.config,
          model_mode=MODEL_MODE_TRAIN,
          mesh=self.mesh,
          rngs=nnx.Rngs(params=0, dropout=1),
          layer_idx=0,
          is_full_indexer=True,
          is_moe=False,
      )
      graphdef, params, other_state = nnx.split(layer, nnx.Param, ...)

      def forward_fn(params_in, x_in):
        m = nnx.merge(graphdef, params_in, other_state)
        out, _, _ = m(
            x_in,
            seg_j,
            pos_j,
            deterministic=True,
            model_mode=MODEL_MODE_TRAIN,
        )
        return jnp.sum(out**2)

      # 1. Verify `"indexer_cutoff_threshold"` is tagged as an int32[B, S, IDX_TOPK] checkpoint residual.
      fwd_jaxpr = jax.make_jaxpr(forward_fn)(params, xj)
      idx_named_vars = [
          eqn.outvars[0].aval
          for eqn in fwd_jaxpr.jaxpr.eqns
          if eqn.primitive.name == "name" and eqn.params.get("name") == "indexer_cutoff_threshold"
      ]
      self.assertEqual(len(idx_named_vars), 1)
      self.assertEqual(idx_named_vars[0].shape, (batch, seq, IDX_TOPK))
      self.assertEqual(idx_named_vars[0].dtype, jnp.int32)

      ckpt_full_fn = jax.checkpoint(forward_fn, policy=jax.checkpoint_policies.nothing_saveable)
      ckpt_save_fn = jax.checkpoint(
          forward_fn,
          policy=jax.checkpoint_policies.save_only_these_names("indexer_cutoff_threshold"),
      )

      # 2. Verify backward VJP jaxpr eliminates the entire `Hy4Indexer` subgraph (`top_k` and `relu`)
      # when `"indexer_cutoff_threshold"` is saved, whereas `nothing_saveable` recomputes both.
      _, vjp_full = jax.vjp(ckpt_full_fn, params, xj)
      bwd_jaxpr_full = jax.make_jaxpr(vjp_full)(jnp.ones((), dtype=jnp.float32))
      self.assertEqual(_count_primitive(bwd_jaxpr_full.jaxpr, "top_k"), 1)
      self.assertEqual(_count_primitive(bwd_jaxpr_full.jaxpr, "relu"), 1)

      _, vjp_saved = jax.vjp(ckpt_save_fn, params, xj)
      bwd_jaxpr_saved = jax.make_jaxpr(vjp_saved)(jnp.ones((), dtype=jnp.float32))
      self.assertEqual(_count_primitive(bwd_jaxpr_saved.jaxpr, "top_k"), 0)
      self.assertEqual(_count_primitive(bwd_jaxpr_saved.jaxpr, "relu"), 0)

      # 3. Verify loss and gradients match between uncheckpointed, full-remat, and saved-indexer-remat execution.
      loss_ref, grads_ref = jax.jit(jax.value_and_grad(forward_fn, argnums=(0, 1)))(params, xj)
      loss_full, grads_full = jax.jit(jax.value_and_grad(ckpt_full_fn, argnums=(0, 1)))(params, xj)
      loss_ckpt, grads_ckpt = jax.jit(jax.value_and_grad(ckpt_save_fn, argnums=(0, 1)))(params, xj)
      np.testing.assert_allclose(np.asarray(loss_ckpt), np.asarray(loss_full), rtol=1e-6, atol=1e-6)
      jax.tree.map(
          lambda g_c, g_f: np.testing.assert_allclose(np.asarray(g_c), np.asarray(g_f), rtol=1e-6, atol=1e-6),
          grads_ckpt,
          grads_full,
      )
      np.testing.assert_allclose(np.asarray(loss_ckpt), np.asarray(loss_ref), rtol=1e-4, atol=1e-4)
      jax.tree.map(
          lambda g_c, g_r: np.testing.assert_allclose(np.asarray(g_c), np.asarray(g_r), rtol=1e-4, atol=1e-4),
          grads_ckpt,
          grads_ref,
      )

  def test_scannable_block_remat_dce(self):
    """Verifies Hy4ScannableBlock (full-indexer + shared-indexer MoE cycle) under jax.checkpoint."""
    batch, seq = self.config.micro_batch_size_to_train_on, 6
    xj = jnp.asarray(np.random.RandomState(101).randn(batch, seq, HC_MULT, EMB).astype(np.float32))
    pos_j = jnp.tile(jnp.arange(seq, dtype=jnp.int32)[None], (batch, 1))
    seg_j = jnp.ones((batch, seq), jnp.int32)

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.config.logical_axis_rules):
      block = hy4.Hy4ScannableBlock(
          config=self.config,
          mesh=self.mesh,
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(params=0, dropout=1),
      )
      self.assertTrue(block.layers_0.is_full_indexer)
      self.assertFalse(block.layers_1.is_full_indexer)
      graphdef, params, other_state = nnx.split(block, nnx.Param, ...)

      def block_forward_fn(params_in, x_in):
        m = nnx.merge(graphdef, params_in, other_state)
        out, _ = m(
            x_in,
            seg_j,
            pos_j,
            deterministic=True,
            model_mode=MODEL_MODE_TRAIN,
        )
        return jnp.sum(out**2)

      # 1. Exactly 1 `"indexer_cutoff_threshold"` residual (`layers_0` full indexer, shared by `layers_1`).
      fwd_jaxpr = jax.make_jaxpr(block_forward_fn)(params, xj)
      idx_named_vars = [
          eqn.outvars[0].aval
          for eqn in fwd_jaxpr.jaxpr.eqns
          if eqn.primitive.name == "name" and eqn.params.get("name") == "indexer_cutoff_threshold"
      ]
      self.assertEqual(len(idx_named_vars), 1)
      self.assertEqual(idx_named_vars[0].shape, (batch, seq, IDX_TOPK))

      ckpt_full_fn = jax.checkpoint(block_forward_fn, policy=jax.checkpoint_policies.nothing_saveable)
      ckpt_save_fn = jax.checkpoint(
          block_forward_fn,
          policy=jax.checkpoint_policies.save_only_these_names("indexer_cutoff_threshold"),
      )

      # 2. `nothing_saveable` recomputes `layers_0`'s Hy4Indexer (`top_k(k=IDX_TOPK)` and `relu`) plus 2 MoE router
      # `top_k(k=TOPK)` calls; saving `"indexer_cutoff_threshold"` eliminates Hy4Indexer from the backward VJP.
      _, vjp_full = jax.vjp(ckpt_full_fn, params, xj)
      bwd_jaxpr_full = jax.make_jaxpr(vjp_full)(jnp.ones((), dtype=jnp.float32))
      self.assertEqual(_count_primitive(bwd_jaxpr_full.jaxpr, "top_k", k=IDX_TOPK), 1)
      self.assertEqual(_count_primitive(bwd_jaxpr_full.jaxpr, "top_k", k=TOPK), 2)
      self.assertEqual(_count_primitive(bwd_jaxpr_full.jaxpr, "relu"), 1)

      _, vjp_saved = jax.vjp(ckpt_save_fn, params, xj)
      bwd_jaxpr_saved = jax.make_jaxpr(vjp_saved)(jnp.ones((), dtype=jnp.float32))
      self.assertEqual(_count_primitive(bwd_jaxpr_saved.jaxpr, "top_k", k=IDX_TOPK), 0)
      self.assertEqual(_count_primitive(bwd_jaxpr_saved.jaxpr, "top_k", k=TOPK), 2)
      self.assertEqual(_count_primitive(bwd_jaxpr_saved.jaxpr, "relu"), 0)

      # 3. Verify loss and gradients match across full-remat, saved-indexer-remat, and uncheckpointed execution.
      loss_ref, grads_ref = jax.jit(jax.value_and_grad(block_forward_fn, argnums=(0, 1)))(params, xj)
      loss_full, grads_full = jax.jit(jax.value_and_grad(ckpt_full_fn, argnums=(0, 1)))(params, xj)
      loss_ckpt, grads_ckpt = jax.jit(jax.value_and_grad(ckpt_save_fn, argnums=(0, 1)))(params, xj)
      np.testing.assert_allclose(np.asarray(loss_ckpt), np.asarray(loss_full), rtol=1e-6, atol=1e-6)
      jax.tree.map(
          lambda g_c, g_f: np.testing.assert_allclose(np.asarray(g_c), np.asarray(g_f), rtol=1e-6, atol=1e-6),
          grads_ckpt,
          grads_full,
      )
      np.testing.assert_allclose(np.asarray(loss_ckpt), np.asarray(loss_ref), rtol=1e-4, atol=1e-4)
      jax.tree.map(
          lambda g_c, g_r: np.testing.assert_allclose(np.asarray(g_c), np.asarray(g_r), rtol=1e-4, atol=1e-4),
          grads_ckpt,
          grads_ref,
      )


if __name__ == "__main__":
  absltest.main()
