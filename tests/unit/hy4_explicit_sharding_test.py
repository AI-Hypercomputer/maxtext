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

"""Unit tests for Hy4-preview explicit sharding and FSDP weight prefetch."""

from __future__ import annotations

import os

# Request 4 simulated host CPU devices before JAX initializes its backend so
# FSDP / TP meshes exercise real sharding under `shard_mode=explicit`.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

# pylint: disable=wrong-import-position,g-import-not-at-top
from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh
from jax.sharding import AxisType
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from maxtext.common.common_types import DecoderBlockType
from maxtext.common.common_types import MODEL_MODE_PREFILL
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.common.common_types import ShardMode
from maxtext.configs import pyconfig
from maxtext.layers import embeddings
from maxtext.layers import nnx_decoders
from maxtext.models import hy4
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
import numpy as np
import pytest
# pylint: enable=wrong-import-position,g-import-not-at-top

pytestmark = [pytest.mark.cpu_only]

BASE_CONFIG = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")

EMB, HEADS, HEAD_DIM = 32, 4, 16
Q_LORA, KV_LORA, NOPE, ROPE, V_DIM = 16, 8, 8, 8, 16
IDX_HEADS, IDX_DIM, IDX_TOPK = 4, 16, 3
MLP, MOE_MLP, EXPERTS, TOPK = 32, 16, 4, 2
HC_MULT = 4
VOCAB = 64


def _make_config(**kwargs):
  """Builds a minimal CPU MaxText hy4-preview config for sharding tests."""
  overrides = {
      "model_name": "hy4-preview",
      "override_model_config": True,
      "base_emb_dim": EMB,
      "base_num_query_heads": HEADS,
      "base_num_kv_heads": HEADS,
      "head_dim": HEAD_DIM,
      "q_lora_rank": Q_LORA,
      "kv_lora_rank": KV_LORA,
      "qk_nope_head_dim": NOPE,
      "qk_rope_head_dim": ROPE,
      "v_head_dim": V_DIM,
      "indexer_n_heads": IDX_HEADS,
      "indexer_head_dim": IDX_DIM,
      "indexer_topk": IDX_TOPK,
      "use_indexer": False,
      "mhc_expansion_rate": HC_MULT,
      "base_mlp_dim": MLP,
      "base_moe_mlp_dim": MOE_MLP,
      "num_experts": EXPERTS,
      "num_experts_per_tok": TOPK,
      "shared_experts": 1,
      "routed_scaling_factor": 2.827,
      "routed_score_func": "sigmoid",
      "routed_bias": True,
      "norm_topk_prob": True,
      "n_routing_groups": -1,
      "topk_routing_group": -1,
      "mlp_activations_limit": 10.0,
      "first_num_dense_layers": 1,
      "inhomogeneous_layer_cycle_interval": 1,
      "vocab_size": VOCAB,
      "base_num_decoder_layers": 4,
      "per_device_batch_size": 1,
      "max_target_length": 8,
      "max_prefill_predict_length": 4,
      "max_position_embeddings": 64,
      "original_max_position_embeddings": 64,
      "scan_layers": False,
      "attention": "dot_product",
      "attention_type": "mla",
      "attention_sink": True,
      "sparse_matmul": True,
      "megablox": False,
      "use_tokamax_gmm": False,
      "capacity_factor": -1,
      "matmul_precision": "highest",
      "dtype": "float32",
      "weight_dtype": "float32",
      "float32_logits": True,
      "float32_qk_product": True,
      "dropout_rate": 0.0,
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "hardware": "cpu",
  }
  overrides.update(kwargs)
  return pyconfig.initialize(["", BASE_CONFIG], **overrides)


class Hy4ExplicitShardingTest(parameterized.TestCase):
  """Tests `hy4` explicit-sharding config, layer parity, and prefetch parity."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest(
          "Needs 4 host devices to exercise FSDP/TP meshes; set XLA_FLAGS=--xla_force_host_platform_device_count=4"
      )

  def test_pyconfig_accepts_hy4_explicit_sharding(self):
    cfg = _make_config(shard_mode="explicit")
    self.assertEqual(cfg.shard_mode, ShardMode.EXPLICIT)
    self.assertEqual(cfg.decoder_block, DecoderBlockType.HY4)

    cfg_prefetch = _make_config(
        shard_mode="explicit",
        scan_layers=True,
        remat_policy="full",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
    )
    self.assertTrue(cfg_prefetch.prefetch_fsdp_weights)
    self.assertTrue(cfg_prefetch.prefetch_fsdp_defer_grad_reduce_scatter)

  def _run_layer(self, layer_cls, extra_config, ctor_kwargs, shard_mode):
    """Runs a single Hy4 layer forward and backward pass and returns NumPy arrays."""
    cfg = _make_config(**(extra_config | {"shard_mode": shard_mode}))
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
      layer = layer_cls(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(params=0, dropout=0),
          **ctor_kwargs,
      )
      nnx.pop(layer, nnx.RngState, nnx.RngCount)
      graphdef, params, state = nnx.split(layer, nnx.Param, ...)
      params = jax.tree.map(
          jax.device_put,
          params,
          sharding.nnx_construct_named_sharding(params, mesh),
      )
      state = jax.tree.map(
          jax.device_put,
          state,
          sharding.nnx_construct_named_sharding(state, mesh),
      )

      batch = cfg.micro_batch_size_to_train_on
      seq_len = cfg.max_target_length
      numel = batch * seq_len * cfg.emb_dim
      inputs = jnp.reshape(
          jnp.arange(numel, dtype=jnp.float32) / numel,
          (batch, seq_len, cfg.emb_dim),
      )
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))
      segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)
      if shard_mode == "explicit":
        act_sharding = sharding.create_sharding(
            mesh,
            ("activation_batch", "activation_norm_length", "activation_embed"),
        )
        pos_sharding = sharding.create_sharding(mesh, ("activation_batch", "activation_length"))
        inputs = jax.device_put(inputs, act_sharding)
        positions = jax.device_put(positions, pos_sharding)
        segment_ids = jax.device_put(segment_ids, pos_sharding)

      def _loss_and_out(p, s, x, seg, pos):
        m = nnx.merge(graphdef, p, s, copy=True)
        out = m(x, seg, pos, True, MODEL_MODE_TRAIN)
        y = out[0] if isinstance(out, tuple) else out
        return jnp.mean(y**2), y

      val_and_grad = jax.jit(jax.value_and_grad(_loss_and_out, argnums=(0, 2), has_aux=True))
      (_, y), (p_grads, x_grad) = val_and_grad(params, state, inputs, segment_ids, positions)
      return (
          np.asarray(jax.device_get(y)),
          np.asarray(jax.device_get(x_grad)),
          jax.tree.map(lambda g: np.asarray(jax.device_get(g)), p_grads),
      )

  @parameterized.named_parameters(
      (
          "dense_fsdp",
          hy4.Hy4DecoderLayer,
          {"use_indexer": False},
          {"layer_idx": 0},
      ),
      (
          "moe_fsdp",
          hy4.Hy4DecoderLayer,
          {"use_indexer": False},
          {"layer_idx": 1},
      ),
      (
          "dense_with_indexer_fsdp",
          hy4.Hy4DecoderLayer,
          {"use_indexer": True},
          {"layer_idx": 0},
      ),
      (
          "dense_with_indexer_fsdp_tp",
          hy4.Hy4DecoderLayer,
          {
              "use_indexer": True,
              "ici_fsdp_parallelism": 2,
              "ici_tensor_parallelism": 2,
          },
          {"layer_idx": 0},
      ),
      (
          "scannable_block_fsdp_cycle2_indexer",
          hy4.Hy4ScannableBlock,
          {
              "use_indexer": True,
              "inhomogeneous_layer_cycle_interval": 2,
          },
          {},
      ),
      (
          "moe_fsdp_tp",
          hy4.Hy4DecoderLayer,
          {
              "use_indexer": False,
              "ici_fsdp_parallelism": 2,
              "ici_tensor_parallelism": 2,
          },
          {"layer_idx": 1},
      ),
  )
  def test_layer_explicit_matches_auto(self, layer_cls, extra_config, ctor_kwargs):
    auto_out, auto_x_grad, auto_p_grads = self._run_layer(layer_cls, extra_config, ctor_kwargs, "auto")
    exp_out, exp_x_grad, exp_p_grads = self._run_layer(layer_cls, extra_config, ctor_kwargs, "explicit")
    self.assertEqual(auto_out.shape, exp_out.shape)
    np.testing.assert_allclose(exp_out, auto_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(exp_x_grad, auto_x_grad, rtol=1e-4, atol=1e-4)
    jax.tree.map(
        lambda g_exp, g_ref: np.testing.assert_allclose(g_exp, g_ref, rtol=1e-4, atol=1e-4),
        exp_p_grads,
        auto_p_grads,
    )

  def _run_hyper_head(self, shard_mode: str):
    """Runs Hy4HyperHead forward and backward on FSDP x TP and returns NumPy arrays."""
    cfg = _make_config(
        shard_mode=shard_mode,
        ici_fsdp_parallelism=2,
        ici_tensor_parallelism=2,
    )
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
      head = hy4.Hy4HyperHead(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=0))
      nnx.pop(head, nnx.RngState, nnx.RngCount)
      graphdef, params, state = nnx.split(head, nnx.Param, ...)
      params = jax.tree.map(
          jax.device_put,
          params,
          sharding.nnx_construct_named_sharding(params, mesh),
      )
      state = jax.tree.map(
          jax.device_put,
          state,
          sharding.nnx_construct_named_sharding(state, mesh),
      )

      batch = cfg.micro_batch_size_to_train_on
      seq_len = cfg.max_target_length
      numel = batch * seq_len * cfg.mhc_expansion_rate * cfg.emb_dim
      x = jnp.reshape(
          jnp.arange(numel, dtype=jnp.float32) / numel,
          (batch, seq_len, cfg.mhc_expansion_rate, cfg.emb_dim),
      )
      if shard_mode == "explicit":
        stream_sharding = sharding.create_sharding(
            mesh,
            (
                "activation_batch",
                "activation_norm_length",
                None,
                "activation_embed",
            ),
        )
        x = jax.device_put(x, stream_sharding)
        self.assertEqual(
            jax.typeof(x).sharding.spec.partitions[3],
            "tensor",
        )

      def _loss_and_out(p, s, inp):
        m = nnx.merge(graphdef, p, s, copy=True)
        y = m(inp)
        return jnp.mean(y**2), y

      val_and_grad = jax.jit(jax.value_and_grad(_loss_and_out, argnums=(0, 2), has_aux=True))
      (_, y), (p_grads, x_grad) = val_and_grad(params, state, x)
      return (
          np.asarray(jax.device_get(y)),
          np.asarray(jax.device_get(x_grad)),
          jax.tree.map(lambda g: np.asarray(jax.device_get(g)), p_grads),
      )

  def test_hyper_head_fsdp_tp_explicit_matches_auto(self):
    auto_out, auto_x_grad, auto_p_grads = self._run_hyper_head("auto")
    exp_out, exp_x_grad, exp_p_grads = self._run_hyper_head("explicit")
    np.testing.assert_allclose(exp_out, auto_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(exp_x_grad, auto_x_grad, rtol=1e-4, atol=1e-4)
    jax.tree.map(
        lambda g_exp, g_ref: np.testing.assert_allclose(g_exp, g_ref, rtol=1e-4, atol=1e-4),
        exp_p_grads,
        auto_p_grads,
    )

  def _run_indexer(
      self,
      *,
      shard_mode: str,
      extra_config: dict[str, int],
      mask_kind: str,
  ):
    """Runs Hy4Indexer under context parallelism and returns outputs and mask sharding."""
    cfg = _make_config(
        shard_mode=shard_mode,
        use_indexer=True,
        **extra_config,
    )
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
      layer = hy4.Hy4DecoderLayer(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(params=0, dropout=0),
          layer_idx=0,
      )
      indexer = layer.self_attention.indexer
      self.assertIsNotNone(indexer)
      graphdef, state = nnx.split(indexer)
      state = jax.tree.map(
          jax.device_put,
          state,
          sharding.nnx_construct_named_sharding(state, mesh),
      )

      batch = cfg.micro_batch_size_to_train_on
      seq_len = cfg.max_target_length
      rng = np.random.RandomState(7)
      inputs_q = jnp.asarray(
          rng.standard_normal((batch, seq_len, cfg.emb_dim)),
          dtype=jnp.float32,
      )
      low_rank_q = jnp.asarray(
          rng.standard_normal((batch, seq_len, cfg.q_lora_rank)),
          dtype=jnp.float32,
      )
      inputs_kv = inputs_q
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))
      causal = jnp.where(
          jnp.arange(seq_len)[None, :] <= jnp.arange(seq_len)[:, None],
          0.0,
          -1e10,
      ).astype(jnp.float32)
      if mask_kind == "segment":
        attention_mask = jnp.broadcast_to(causal[None, :, :], (batch, seq_len, seq_len))
      else:
        attention_mask = causal[None, :, :]

      if shard_mode == "explicit":
        q_sharding = sharding.create_sharding(
            mesh,
            (
                "activation_batch_attn",
                "activation_length_attn",
                "activation_embed",
            ),
        )
        lq_sharding = sharding.create_sharding(
            mesh,
            ("activation_batch_attn", "activation_length_attn", None),
        )
        pos_sharding = sharding.create_sharding(mesh, ("activation_batch", "activation_length"))
        inputs_q = jax.device_put(inputs_q, q_sharding)
        inputs_kv = jax.device_put(inputs_kv, q_sharding)
        low_rank_q = jax.device_put(low_rank_q, lq_sharding)
        positions = jax.device_put(positions, pos_sharding)
        self.assertEqual(
            jax.typeof(inputs_q).sharding.spec.partitions[1],
            "context",
        )
        if mask_kind == "segment":
          has_fsdp = extra_config.get("ici_fsdp_parallelism", 1) > 1
          mask_batch_axis = "fsdp" if has_fsdp else None
          mask_sharding = NamedSharding(mesh, PartitionSpec(mask_batch_axis, None, None))
        else:
          mask_sharding = NamedSharding(mesh, PartitionSpec(None, None, None))
        attention_mask = jax.device_put(attention_mask, mask_sharding)

      @jax.jit
      def _fwd(st, iq, lq, ikv, pos, mask):
        """Executes a jitted Hy4Indexer forward pass."""
        m = nnx.merge(graphdef, st)
        return m(
            inputs_q=iq,
            low_rank_q=lq,
            inputs_kv=ikv,
            inputs_positions=pos,
            attention_mask=mask,
            model_mode=MODEL_MODE_TRAIN,
        )

      idx_mask, topk_idx, idx_score = _fwd(state, inputs_q, low_rank_q, inputs_kv, positions, attention_mask)
      mask_parts = jax.typeof(idx_mask).sharding.spec.partitions if shard_mode == "explicit" else None
      return (
          np.asarray(jax.device_get(idx_mask)),
          np.asarray(jax.device_get(topk_idx)),
          np.asarray(jax.device_get(idx_score)),
          mask_parts,
      )

  @parameterized.named_parameters(
      (
          "fsdp2_cp2_segment_mask",
          {"ici_fsdp_parallelism": 2, "ici_context_parallelism": 2},
          "segment",
          ("fsdp", "context", None),
      ),
      (
          "fsdp2_cp2_causal_mask",
          {"ici_fsdp_parallelism": 2, "ici_context_parallelism": 2},
          "causal",
          ("fsdp", "context", None),
      ),
      (
          "cp2_tp2_segment_mask",
          {"ici_context_parallelism": 2, "ici_tensor_parallelism": 2},
          "segment",
          (None, "context", None),
      ),
  )
  def test_indexer_context_parallel_explicit_matches_auto(self, extra_config, mask_kind, expected_mask_parts):
    auto_mask, auto_topk, auto_score, _ = self._run_indexer(
        shard_mode="auto", extra_config=extra_config, mask_kind=mask_kind
    )
    exp_mask, exp_topk, exp_score, mask_parts = self._run_indexer(
        shard_mode="explicit", extra_config=extra_config, mask_kind=mask_kind
    )
    self.assertEqual(mask_parts, expected_mask_parts)
    np.testing.assert_allclose(exp_score, auto_score, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(exp_topk, auto_topk)
    np.testing.assert_allclose(exp_mask, auto_mask, rtol=1e-4, atol=1e-4)

  def _run_attention_prefill(self, shard_mode: str):
    """Runs Hy4Attention in prefill mode and returns NumPy outputs and gradients."""
    cfg = _make_config(
        shard_mode=shard_mode,
        use_indexer=False,
        ici_fsdp_parallelism=2,
        ici_context_parallelism=2,
    )
    # Diverge `prefill_activation_length` from `activation_length` so
    # `Hy4Attention.gate_proj` must use `out_logical_name` in prefill mode, and
    # omit size-1 mesh axes from `activation_batch_attn`/`activation_heads`/
    # `activation_kv` so `apply_attention_dot`'s `with_logical_constraint`
    # checks match on CPU.
    rule_overrides = {
        "activation_batch_attn": ("fsdp",),
        "activation_heads": (),
        "activation_kv": (),
        "activation_length": ("context",),
        "activation_length_attn": ("context",),
        "activation_prefill_kv_batch": ("fsdp",),
        "prefill_activation_length": (),
    }
    prefill_rules = [(name, rule_overrides.get(name, axes)) for name, axes in cfg.logical_axis_rules]
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    with nn_partitioning.axis_rules(prefill_rules), jax.set_mesh(mesh):
      layer = hy4.Hy4DecoderLayer(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_PREFILL,
          rngs=nnx.Rngs(params=0, dropout=0),
          layer_idx=0,
      )
      attn = layer.self_attention
      nnx.pop(attn, nnx.RngState, nnx.RngCount)
      graphdef, params, state = nnx.split(attn, nnx.Param, ...)
      params = jax.tree.map(
          jax.device_put,
          params,
          sharding.nnx_construct_named_sharding(params, mesh),
      )
      state = jax.tree.map(
          jax.device_put,
          state,
          sharding.nnx_construct_named_sharding(state, mesh),
      )

      batch = cfg.micro_batch_size_to_train_on
      seq_len = cfg.max_prefill_predict_length
      numel = batch * seq_len * cfg.emb_dim
      inputs = jnp.reshape(
          jnp.arange(numel, dtype=jnp.float32) / numel,
          (batch, seq_len, cfg.emb_dim),
      )
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))
      segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)
      if shard_mode == "explicit":
        in_sharding = sharding.create_sharding(
            mesh,
            (
                "activation_prefill_kv_batch",
                "prefill_activation_length",
                "activation_embed",
            ),
        )
        pos_sharding = sharding.create_sharding(
            mesh,
            ("activation_prefill_kv_batch", "prefill_activation_length"),
        )
        inputs = jax.device_put(inputs, in_sharding)
        positions = jax.device_put(positions, pos_sharding)
        segment_ids = jax.device_put(segment_ids, pos_sharding)

      def _loss_and_out(p, s, x, seg, pos):
        """Computes mean-squared loss and prefill attention output."""
        m = nnx.merge(graphdef, p, s, copy=True)
        out, _, _ = m(
            x,
            x,
            inputs_positions=pos,
            decoder_segment_ids=seg,
            deterministic=True,
            model_mode=MODEL_MODE_PREFILL,
        )
        return jnp.mean(out**2), out

      val_and_grad = jax.jit(jax.value_and_grad(_loss_and_out, argnums=(0, 2), has_aux=True))
      (_, y), (p_grads, x_grad) = val_and_grad(params, state, inputs, segment_ids, positions)
      return (
          np.asarray(jax.device_get(y)),
          np.asarray(jax.device_get(x_grad)),
          jax.tree.map(lambda g: np.asarray(jax.device_get(g)), p_grads),
      )

  def test_attention_prefill_explicit_matches_auto(self):
    auto_out, auto_x_grad, auto_p_grads = self._run_attention_prefill("auto")
    exp_out, exp_x_grad, exp_p_grads = self._run_attention_prefill("explicit")
    np.testing.assert_allclose(exp_out, auto_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(exp_x_grad, auto_x_grad, rtol=1e-4, atol=1e-4)
    jax.tree.map(
        lambda g_exp, g_ref: np.testing.assert_allclose(g_exp, g_ref, rtol=1e-4, atol=1e-4),
        exp_p_grads,
        auto_p_grads,
    )

  def test_mlp_intermediate_sharding_preserves_context_parallelism(self):
    mesh_cases = (
        ({"fsdp": 2, "context": 2}, PartitionSpec("fsdp", "context", None)),
        (
            {"fsdp": 2, "tensor_sequence": 2},
            PartitionSpec("fsdp", None, "tensor_sequence"),
        ),
        (
            {"fsdp": 2, "context": 2, "tensor_sequence": 2},
            PartitionSpec("fsdp", "context", "tensor_sequence"),
        ),
    )
    for model_mode in (MODEL_MODE_TRAIN, MODEL_MODE_PREFILL):
      cfg = _make_config(
          shard_mode="explicit",
          inhomogeneous_layer_cycle_interval=2,
          ici_fsdp_parallelism=2,
          ici_context_parallelism=2,
          base_num_decoder_layers=2,
      )
      for mesh_overrides, expected_spec in mesh_cases:
        axis_sizes = {a: 1 for a in cfg.mesh_axes} | mesh_overrides
        mesh = AbstractMesh(
            tuple(axis_sizes.values()),
            tuple(axis_sizes),
            axis_types=(AxisType.Explicit,) * len(axis_sizes),
        )
        with (
            jax.sharding.use_abstract_mesh(mesh),
            nn.logical_axis_rules(cfg.logical_axis_rules),
        ):
          decoder = nnx.eval_shape(
              lambda c=cfg, m=mesh, mode=model_mode: nnx_decoders.NNXDecoder(
                  config=c, mesh=m, model_mode=mode, rngs=nnx.Rngs(0)
              )
          )
        checked = 0
        for _, mod in nnx.iter_modules(decoder):
          mlp_sharding = getattr(mod, "mlp_intermediate_sharding", None) or getattr(mod, "mlp_sharding", None)
          if mlp_sharding is not None:
            self.assertEqual(mlp_sharding.spec, expected_spec)
            checked += 1
        self.assertGreater(checked, 0)

  def _run_full_decoder_loss_and_grads(
      self,
      *,
      shard_mode: str,
      prefetch: bool,
      defer_rs: bool,
      use_indexer: bool = False,
      inhomogeneous_layer_cycle_interval: int = 1,
      base_num_decoder_layers: int = 4,
  ):
    """Runs a scanned NNXDecoder loss and gradient pass with optional FSDP prefetch."""
    cfg = _make_config(
        shard_mode=shard_mode,
        scan_layers=True,
        remat_policy="full",
        use_indexer=use_indexer,
        inhomogeneous_layer_cycle_interval=inhomogeneous_layer_cycle_interval,
        base_num_decoder_layers=base_num_decoder_layers,
        moe_flat_fsdp_weights=False,
        ici_fsdp_parallelism=4,
        prefetch_fsdp_weights=prefetch,
        prefetch_fsdp_defer_grad_reduce_scatter=defer_rs,
    )
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
      rngs = nnx.Rngs(params=0, dropout=0)
      shared_emb = embeddings.Embed(
          num_embeddings=cfg.vocab_size,
          num_features=cfg.emb_dim,
          dtype=cfg.dtype,
          embedding_init=nn.initializers.normal(stddev=1.0),
          config=cfg,
          mesh=mesh,
          rngs=rngs,
      )
      decoder = nnx_decoders.NNXDecoder(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          rngs=rngs,
      )
      nnx.pop(shared_emb, nnx.RngState, nnx.RngCount)
      nnx.pop(decoder, nnx.RngState, nnx.RngCount)
      emb_graphdef, emb_params, emb_state = nnx.split(shared_emb, nnx.Param, ...)
      dec_graphdef, dec_params, dec_state = nnx.split(decoder, nnx.Param, ...)

      emb_params = jax.tree.map(
          jax.device_put,
          emb_params,
          sharding.nnx_construct_named_sharding(emb_params, mesh),
      )
      emb_state = jax.tree.map(
          jax.device_put,
          emb_state,
          sharding.nnx_construct_named_sharding(emb_state, mesh),
      )
      dec_params = jax.tree.map(
          jax.device_put,
          dec_params,
          sharding.nnx_construct_named_sharding(dec_params, mesh),
      )
      dec_state = jax.tree.map(
          jax.device_put,
          dec_state,
          sharding.nnx_construct_named_sharding(dec_state, mesh),
      )

      batch = cfg.global_batch_size_to_train_on
      seq_len = cfg.max_target_length
      ids_np = np.random.RandomState(42).randint(0, cfg.vocab_size, size=(batch, seq_len), dtype=np.int32)
      ids = jnp.asarray(ids_np)
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))
      segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)
      data_sharding = sharding.get_input_data_sharding(cfg, mesh)
      ids = jax.device_put(ids, data_sharding)
      positions = jax.device_put(positions, data_sharding)
      segment_ids = jax.device_put(segment_ids, data_sharding)

      def loss_fn(ep, dp, es, ds):
        """Computes mean-squared decoder logits loss."""
        emb_m = nnx.merge(emb_graphdef, ep, es, copy=True)
        dec_m = nnx.merge(dec_graphdef, dp, ds, copy=True)
        logits, _, _ = dec_m(
            emb_m,
            ids,
            positions,
            decoder_segment_ids=segment_ids,
            deterministic=True,
            model_mode=MODEL_MODE_TRAIN,
        )
        return jnp.mean(logits**2)

      val_and_grad = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))
      loss_val, (emb_grads, dec_grads) = val_and_grad(emb_params, dec_params, emb_state, dec_state)
      return (
          np.asarray(jax.device_get(loss_val)),
          jax.tree.map(lambda g: np.asarray(jax.device_get(g)), (emb_grads, dec_grads)),
      )

  @parameterized.named_parameters(
      {
          "testcase_name": "uniform_no_indexer",
          "use_indexer": False,
          "inhomogeneous_layer_cycle_interval": 1,
          "base_num_decoder_layers": 4,
      },
      {
          "testcase_name": "mixed_cycle_with_indexer",
          "use_indexer": True,
          "inhomogeneous_layer_cycle_interval": 2,
          "base_num_decoder_layers": 5,
      },
  )
  def test_full_hy4_decoder_explicit_prefetch_grad_parity(
      self,
      *,
      use_indexer: bool,
      inhomogeneous_layer_cycle_interval: int,
      base_num_decoder_layers: int,
  ):
    ref_loss, ref_grads = self._run_full_decoder_loss_and_grads(
        shard_mode="auto",
        prefetch=False,
        defer_rs=False,
        use_indexer=use_indexer,
        inhomogeneous_layer_cycle_interval=inhomogeneous_layer_cycle_interval,
        base_num_decoder_layers=base_num_decoder_layers,
    )
    exp_loss, exp_grads = self._run_full_decoder_loss_and_grads(
        shard_mode="explicit",
        prefetch=True,
        defer_rs=True,
        use_indexer=use_indexer,
        inhomogeneous_layer_cycle_interval=inhomogeneous_layer_cycle_interval,
        base_num_decoder_layers=base_num_decoder_layers,
    )
    np.testing.assert_allclose(exp_loss, ref_loss, rtol=1e-4, atol=1e-4)
    jax.tree.map(
        lambda g_exp, g_ref: np.testing.assert_allclose(g_exp, g_ref, rtol=1e-4, atol=1e-4),
        exp_grads,
        ref_grads,
    )


if __name__ == "__main__":
  absltest.main()
