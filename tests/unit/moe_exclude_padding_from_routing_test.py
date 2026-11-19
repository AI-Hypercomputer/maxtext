# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for `moe_exclude_padding_from_routing`."""

import os
import re
import sys
import types
import unittest

from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import yaml

from maxtext.common import common_types as ctypes
from maxtext.configs import pyconfig
from maxtext.configs import types as mt_types
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.models import qwen3_5
from maxtext.utils import globals as mt_globals
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_post_train_test_config_path, get_test_config_path


def _init_cfg(**overrides):
  """Initializes a shrunk MaxText config for unit tests."""
  kwargs = {
      "run_name": "moe_exclude_padding_test",
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


class MoeExcludePaddingConfigTest(parameterized.TestCase):
  """Config defaults, Tier-2 hygiene, routing_token_mask, and validator checks."""

  def test_tier2_default_false_and_not_in_base_yml(self):
    field = mt_types.MoEGeneral.model_fields["moe_exclude_padding_from_routing"]
    self.assertIs(field.annotation, bool)
    self.assertIs(field.default, False)
    base_yml_path = os.path.join(mt_globals.MAXTEXT_CONFIGS_DIR, "base.yml")
    with open(base_yml_path, encoding="utf-8") as f:
      base_yaml = yaml.safe_load(f)
    self.assertNotIn("moe_exclude_padding_from_routing", base_yaml)
    self.assertIs(_init_cfg().moe_exclude_padding_from_routing, False)
    self.assertIs(_init_cfg(moe_exclude_padding_from_routing=True).moe_exclude_padding_from_routing, True)

  def test_routing_token_mask_activation_conditions(self):
    seg = jnp.array([[1, 1, 0, 0], [2, 0, 3, 0]], dtype=jnp.int32)
    cfg_on = types.SimpleNamespace(
        moe_exclude_padding_from_routing=True,
        logical_global_batch_size_to_train_on=None,
    )
    cfg_off = types.SimpleNamespace(
        moe_exclude_padding_from_routing=False,
        logical_global_batch_size_to_train_on=None,
    )
    expected = np.array([[True, True, False, False], [True, False, True, False]])
    np.testing.assert_array_equal(
        moe.routing_token_mask(cfg_on, seg, ctypes.MODEL_MODE_TRAIN),
        expected,
    )
    self.assertIsNone(moe.routing_token_mask(cfg_on, None, ctypes.MODEL_MODE_TRAIN))
    self.assertIsNone(moe.routing_token_mask(cfg_on, seg, ctypes.MODEL_MODE_PREFILL))
    self.assertIsNone(moe.routing_token_mask(cfg_off, seg, ctypes.MODEL_MODE_TRAIN))

  @parameterized.named_parameters(
      ("te_moe_block", {"te_moe_block": True, "sparse_matmul": True}, "te_moe_block=False"),
      (
          "batch_split",
          {"use_batch_split_schedule": True, "decoder_block": "deepseek", "attention_type": "mla"},
          "use_batch_split_schedule=False",
      ),
      ("lineage", {"use_lineage": True}, "use_lineage=False"),
      ("m3", {"use_m3_model": True}, "use_m3_model=False"),
      ("kimi_linear", {"decoder_block": "kimi_linear", "packing": False}, "decoder_block != 'kimi_linear'"),
  )
  def test_validator_rejects_unsupported_moe_modes(self, overrides, expected_req):
    with self.assertRaisesRegex(
        ValueError,
        f"moe_exclude_padding_from_routing=True requires: .*{re.escape(expected_req)}",
    ):
      _init_cfg(
          moe_exclude_padding_from_routing=True,
          num_experts=4,
          num_experts_per_tok=2,
          base_mlp_dim=64,
          base_moe_mlp_dim=64,
          **overrides,
      )

  def test_rl_config_validator_rejects_unsupported_moe_modes(self):
    rl_argv = [
        os.path.join(mt_globals.MAXTEXT_PKG_DIR, "trainers", "post_train", "rl", "train_rl.py"),
        get_post_train_test_config_path(),
    ]
    for extra, req in (
        ({"te_moe_block": True, "sparse_matmul": True}, "te_moe_block=False"),
        ({"use_lineage": True}, "use_lineage=False"),
    ):
      with self.assertRaisesRegex(ValueError, f"moe_exclude_padding_from_routing=True requires: .*{re.escape(req)}"):
        pyconfig.initialize(
            rl_argv,
            run_name="moe_exclude_pad_rl_reject",
            enable_checkpointing=False,
            skip_jax_distributed_system=True,
            compile_topology="tpu7x-8",
            compile_topology_num_slices=1,
            moe_exclude_padding_from_routing=True,
            num_experts=4,
            num_experts_per_tok=2,
            base_mlp_dim=64,
            base_moe_mlp_dim=64,
            **extra,
        )


class MoeExcludePaddingRoutingAndCapacityTest(unittest.TestCase):
  """Ring-of-Experts capacity protection, routing counts, aux loss, and Qwen3.5 shared-gate masking."""

  def _roe_cfg(self, exclude_padding: bool, **overrides):
    """Builds a Ring-of-Experts test config with optional padding exclusion."""
    num_devices = jax.device_count()
    ep = min(4, num_devices) if num_devices >= 2 else 1
    use_roe = ep > 1
    kwargs = {
        "model_name": "deepseek3-671b",
        "override_model_config": True,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "base_emb_dim": 64,
        "base_mlp_dim": 64,
        "base_moe_mlp_dim": 64,
        "max_target_length": 16,
        "max_prefill_predict_length": 16,
        "per_device_batch_size": 1.0,
        "dtype": "bfloat16",
        "weight_dtype": "float32",
        "matmul_precision": "highest",
        "float32_gate_logits": True,
        "megablox": True,
        "sparse_matmul": True,
        "use_tokamax_gmm": False,
        "use_gmm_v2": False,
        "use_ring_of_experts": use_roe,
        "use_ragged_sort": use_roe,
        "use_custom_sort_vjp": False,
        "ragged_gather_fallback": True,
        "ragged_gather_reduce_fallback": True,
        "ragged_buffer_factor": -1.0,
        "ici_expert_parallelism": ep,
        "ici_fsdp_parallelism": num_devices // ep,
        "moe_exclude_padding_from_routing": exclude_padding,
    }
    kwargs.update(overrides)
    return _init_cfg(**kwargs)

  def _build_routed_moe(self, cfg, mesh):
    """Instantiates RoutedMoE under the given config and mesh."""
    return moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.moe_mlp_dim,
        weight_dtype=cfg.weight_dtype,
        dtype=cfg.dtype,
        rngs=nnx.Rngs(params=0),
    )

  def test_padding_does_not_steal_ragged_capacity_or_pollute_bias_and_grads(self):
    """Identical pad hidden states pile onto shard 0 and overflow when unmasked, but take 0 slots when masked."""
    if jax.device_count() < 2:
      self.skipTest("Requires >= 2 devices for Ring-of-Experts ragged buffer overflow check.")

    cfg_unmasked = self._roe_cfg(
        False,
        ragged_buffer_factor=1.5,
        num_moe_token_chunks=2,
        moe_dropless_fallback="step",
        routed_bias=True,
        routed_bias_update_rate=1e-3,
        load_balance_loss_weight=0.01,
    )
    cfg_masked = self._roe_cfg(
        True,
        ragged_buffer_factor=1.5,
        num_moe_token_chunks=2,
        moe_dropless_fallback="step",
        routed_bias=True,
        routed_bias_update_rate=1e-3,
        load_balance_loss_weight=0.01,
    )
    cfg_dropless_ref = self._roe_cfg(
        False,
        ragged_buffer_factor=-1.0,
        num_moe_token_chunks=2,
        routed_bias=True,
        routed_bias_update_rate=1e-3,
        load_balance_loss_weight=0.01,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_masked), cfg_masked.mesh_axes)
    batch = int(cfg_masked.per_device_batch_size * jax.device_count())
    seq_len = cfg_masked.max_target_length

    # First 4 tokens of each row are real with balanced forced routing across all 8 experts;
    # trailing 12 tokens are padding (segment_id == 0) whose identical hidden state routes to experts (0, 1) on shard 0.
    segment_ids = jnp.where(jnp.arange(seq_len)[None, :] < 4, 1, 0).repeat(batch, axis=0)
    real_x = jax.random.normal(jax.random.PRNGKey(7), (batch, seq_len, cfg_masked.base_emb_dim), jnp.float32)
    pad_vec = jnp.ones((1, 1, cfg_masked.base_emb_dim), jnp.float32) * 3.0
    x = jnp.where((segment_ids != 0)[..., None], real_x, pad_vec)

    # Balanced forced routing on real tokens; stale shard-0 expert ids (0, 1) on padding tokens to verify
    # moe_exclude_padding_from_routing overrides replayed ids at pad positions too.
    real_experts = jnp.stack(
        [
            (jnp.arange(batch * seq_len) * 2) % cfg_masked.num_experts,
            (jnp.arange(batch * seq_len) * 2 + 1) % cfg_masked.num_experts,
        ],
        axis=-1,
    ).reshape(batch, seq_len, 2)
    pad_experts = jnp.broadcast_to(jnp.array([0, 1], jnp.int32), (batch, seq_len, 2))
    forced = jnp.where((segment_ids != 0)[..., None], real_experts, pad_experts)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_masked.logical_axis_rules):
      base_layer = self._build_routed_moe(cfg_masked, mesh)
      _, params, _ = nnx.split(base_layer, nnx.Param, ...)

      def run_step(cfg, forced_arg):
        layer = self._build_routed_moe(cfg, mesh)
        gdef, _, rest = nnx.split(layer, nnx.Param, ...)
        tmask = moe.routing_token_mask(cfg, segment_ids, ctypes.MODEL_MODE_TRAIN)

        def loss_fn(p, inp):
          mdl = nnx.merge(gdef, p, rest)
          out, lb_loss, bias_upd = mdl(inp, forced_routed_experts=forced_arg, token_mask=tmask)
          overflow_var = getattr(mdl, "moe_has_overflow", None)
          overflow = (
              jnp.bool_(False)
              if overflow_var is None
              else jnp.any(jnp.stack([jnp.any(v) for v in jax.tree_util.tree_leaves(overflow_var.get_value())]))
          )
          # Loss over real tokens only, matching cross-entropy target masking in train.py.
          real_out = jnp.where((segment_ids != 0)[..., None], out, 0.0)
          loss = jnp.sum(real_out**2) / jnp.sum(segment_ids != 0)
          return loss, (out, lb_loss, bias_upd, overflow)

        (_, aux), (dparams, dx) = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(params, x)
        return aux, dparams, dx

      (out_unmasked, _, bias_unmasked, ov_unmasked), _, _ = run_step(cfg_unmasked, forced)
      (out_masked, lb_masked, bias_masked, ov_masked), dp_masked, dx_masked = run_step(cfg_masked, forced)

      # Reference on dropless buffer where padding slots are explicitly marked unused (-1) in forced_routed_experts.
      forced_ref = jnp.where((segment_ids != 0)[..., None], real_experts, -1)
      (out_ref, _, bias_ref, _), dp_ref, dx_ref = run_step(cfg_dropless_ref, forced_ref)

    # Without moe_exclude_padding_from_routing, padding tokens overflow shard 0's capped ragged buffer (1.5x balanced).
    self.assertTrue(bool(ov_unmasked), "Expected unmasked padding pile-up on shard 0 to overflow ragged buffer.")
    # With moe_exclude_padding_from_routing, padding tokens take 0 slots -> 0 overflow!
    self.assertFalse(bool(ov_masked), "Masked padding tokens must not consume ragged buffer slots or overflow.")

    is_real = np.asarray(segment_ids != 0)
    # Padding outputs and input gradients are identically zero.
    np.testing.assert_array_equal(np.asarray(out_masked)[~is_real], 0.0)
    np.testing.assert_array_equal(np.asarray(dx_masked)[~is_real], 0.0)
    self.assertGreater(float(np.max(np.abs(np.asarray(out_unmasked)[~is_real]))), 0.0)

    # Real-token outputs, gradients, and routed-bias updates match the dropless real-token reference.
    np.testing.assert_allclose(
        np.asarray(out_masked, np.float32)[is_real], np.asarray(out_ref, np.float32)[is_real], rtol=1e-2, atol=1e-2
    )
    np.testing.assert_allclose(
        np.asarray(dx_masked, np.float32)[is_real], np.asarray(dx_ref, np.float32)[is_real], rtol=1e-2, atol=1e-2
    )
    np.testing.assert_array_equal(np.asarray(bias_masked), np.asarray(bias_ref))
    self.assertFalse(np.array_equal(np.asarray(bias_unmasked), np.asarray(bias_ref)))
    self.assertTrue(bool(np.isfinite(lb_masked)))
    for (_, g_m), (_, g_r) in zip(
        jax.tree_util.tree_leaves_with_path(dp_masked),
        jax.tree_util.tree_leaves_with_path(dp_ref),
    ):
      np.testing.assert_allclose(np.asarray(g_m, np.float32), np.asarray(g_r, np.float32), rtol=1e-2, atol=1e-2)

  def test_qwen3_5_decoder_layer_zeroes_shared_gate_on_padding_and_preserves_default_jaxpr(self):
    """Qwen3.5 MoE block zeroes both routed and shared-expert terms on segment_id == 0 and is bit-identical when off."""
    base_kwargs = {
        "model_name": "qwen3.5-35b-a3b",
        "override_model_config": True,
        "attention": "dot_product",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "base_emb_dim": 64,
        "base_num_query_heads": 2,
        "base_num_kv_heads": 2,
        "head_dim": 32,
        "partial_rotary_factor": 0.25,
        "base_mlp_dim": 64,
        "base_moe_mlp_dim": 64,
        "shared_experts": 1,
        "gdn_key_head_dim": 16,
        "gdn_value_head_dim": 16,
        "gdn_num_key_heads": 2,
        "gdn_num_value_heads": 2,
        "gdn_chunk_size": 4,
        "use_mrope": False,
        "max_target_length": 8,
        "max_prefill_predict_length": 8,
        "per_device_batch_size": 1.0,
        "dtype": "float32",
        "weight_dtype": "float32",
        "matmul_precision": "highest",
        "float32_gate_logits": True,
        "float32_weight_sum": True,
        "sparse_matmul": False,
        "megablox": False,
    }
    cfg_off = _init_cfg(moe_exclude_padding_from_routing=False, **base_kwargs)
    cfg_on = _init_cfg(moe_exclude_padding_from_routing=True, **base_kwargs)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_on), cfg_on.mesh_axes)
    batch = int(cfg_on.per_device_batch_size * jax.device_count())
    x = jax.random.normal(jax.random.PRNGKey(3), (batch, 8, 64), jnp.float32)
    # Packed multi-segment row with mid-row gap (0) and trailing pad (0).
    seg = jnp.broadcast_to(jnp.array([[1, 1, 0, 2, 2, 2, 0, 0]], jnp.int32), (batch, 8))
    pos = jnp.broadcast_to(jnp.array([[0, 1, 0, 0, 1, 2, 0, 0]], jnp.int32), (batch, 8))

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_on.logical_axis_rules):
      moe_off = qwen3_5.Qwen3_5SparseMoEBlock(config=cfg_off, mesh=mesh, quant=None, rngs=nnx.Rngs(params=0))
      gdef_off, params_moe, rest_off = nnx.split(moe_off, nnx.Param, ...)
      moe_on = qwen3_5.Qwen3_5SparseMoEBlock(config=cfg_on, mesh=mesh, quant=None, rngs=nnx.Rngs(params=0))
      gdef_on, _, rest_on = nnx.split(moe_on, nnx.Param, ...)

      # Default False: token_mask=None produces zero extra ops.
      jaxpr_off = str(
          jax.make_jaxpr(lambda p, h: nnx.merge(gdef_off, p, rest_off)(h, deterministic=True, token_mask=None))(
              params_moe, x
          )
      )
      self.assertNotIn("not ", jaxpr_off)

      tmask = moe.routing_token_mask(cfg_on, seg, ctypes.MODEL_MODE_TRAIN)
      out_on, _ = jax.jit(lambda p, h, m: nnx.merge(gdef_on, p, rest_on)(h, deterministic=True, token_mask=m))(
          params_moe, x, tmask
      )
      out_off_unmasked, _ = jax.jit(
          lambda p, h: nnx.merge(gdef_off, p, rest_off)(h, deterministic=True, token_mask=None)
      )(params_moe, x)

      is_real = np.asarray(seg != 0)
      # Both routed and shared expert contributions are zeroed on pad tokens when moe_exclude_padding_from_routing=True.
      np.testing.assert_array_equal(np.asarray(out_on)[~is_real], 0.0)
      self.assertGreater(float(np.max(np.abs(np.asarray(out_off_unmasked)[~is_real]))), 0.0)
      np.testing.assert_allclose(np.asarray(out_on)[is_real], np.asarray(out_off_unmasked)[is_real], rtol=1e-5, atol=1e-5)

      # Full Qwen3_5DecoderLayer end-to-end forward + backward check.
      dec_layer = qwen3_5.Qwen3_5DecoderLayer(
          config=cfg_on,
          mesh=mesh,
          model_mode=ctypes.MODEL_MODE_TRAIN,
          quant=None,
          layer_idx=3,
          is_full_attention_layer=True,
          rngs=nnx.Rngs(params=1),
      )
      dec_gdef, dec_params, dec_rest = nnx.split(dec_layer, nnx.Param, ...)

      def dec_loss(p, inp):
        y, _ = nnx.merge(dec_gdef, p, dec_rest)(inp, seg, pos, True, ctypes.MODEL_MODE_TRAIN)
        real_y = jnp.where((seg != 0)[..., None], y, 0.0)
        return jnp.mean(real_y**2)

      loss_val, (d_dec_params, d_inp) = jax.jit(jax.value_and_grad(dec_loss, argnums=(0, 1)))(dec_params, x)
      self.assertTrue(bool(np.isfinite(loss_val)))
      np.testing.assert_array_equal(np.asarray(d_inp)[~is_real], 0.0)
      for leaf in jax.tree_util.tree_leaves(d_dec_params):
        self.assertTrue(bool(np.all(np.isfinite(np.asarray(leaf)))))

  def test_padding_exclusion_and_fallback_topk_with_rowwise_quantized_ra2a(self):
    """Padding exclusion and compacted fallback top-k preserve exact zero pad outputs/grads under FP8 ragged a2a."""
    if jax.device_count() < 2:
      self.skipTest("EP ragged all-to-all test requires >= 2 devices.")
    ep = min(4, jax.device_count())
    qa2a_kwargs = {
        "use_ring_of_experts": False,
        "use_ragged_sort": False,
        "ragged_buffer_factor": -1.0,
        "ici_expert_parallelism": ep,
        "ici_fsdp_parallelism": jax.device_count() // ep,
        "dtype": "float32",
        "load_balance_loss_weight": 0.01,
        "routed_bias": True,
        "routed_bias_update_rate": 1e-3,
        "moe_fallback_topk_capacity": 4,
        "moe_quantize_dispatch_fwd_method": "rowwise",
        "moe_quantize_combine_fwd_method": "rowwise",
        "moe_quantize_combine_bwd_method": "rowwise",
        "moe_quantize_dispatch_bwd_method": "rowwise",
    }
    cfg_masked = self._roe_cfg(exclude_padding=True, **qa2a_kwargs)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_masked), cfg_masked.mesh_axes)
    batch = int(cfg_masked.per_device_batch_size * jax.device_count())
    seq = cfg_masked.max_target_length
    emb = cfg_masked.base_emb_dim

    k_x, k_dy = jax.random.split(jax.random.PRNGKey(17))
    x = jax.random.normal(k_x, (batch, seq, emb), jnp.float32)
    # Non-zero upstream cotangent on ALL tokens (including padded positions) to verify RoutedMoE zeroes pad dx too.
    dy = jax.random.normal(k_dy, (batch, seq, emb), jnp.float32)
    # Include an entire shard row of padding when batch >= 2, plus mixed real/pad tokens on other rows.
    seg_np = np.ones((batch, seq), dtype=np.int32)
    seg_np[:, seq // 2 :] = 0
    if batch >= 2:
      seg_np[-1, :] = 0
    seg = jnp.asarray(seg_np)
    is_real = seg_np != 0

    # Forced routing: mix valid distinct expert pairs ([0, 1]) and fallback-triggering duplicate pairs ([2, 2]).
    forced_np = np.tile(np.array([0, 1], dtype=np.int32), (batch, seq, 1))
    forced_np[:, 1::3, :] = np.array([2, 2], dtype=np.int32)
    forced = jnp.asarray(forced_np)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_masked.logical_axis_rules):
      layer = self._build_routed_moe(cfg_masked, mesh)
      gdef, params, rest = nnx.split(layer, nnx.Param, ...)
      tmask = moe.routing_token_mask(cfg_masked, seg, ctypes.MODEL_MODE_TRAIN)

      def run_fwd_bwd(p, inp):
        out, lb_loss, bias_upd = nnx.merge(gdef, p, rest)(inp, forced_routed_experts=forced, token_mask=tmask)
        loss = jnp.sum(out * dy) + lb_loss
        return loss, (out, lb_loss, bias_upd)

      grad_fn = jax.value_and_grad(run_fwd_bwd, argnums=(0, 1), has_aux=True)
      # Trace full forward+backward JAXpr (including shard_map check_vma=True and all 4 FP8 + scale ra2a legs) on CPU/TPU.
      jaxpr_text = str(jax.make_jaxpr(grad_fn)(params, x))
      self.assertEqual(jaxpr_text.count("ragged_all_to_all["), 8)
      if jax.devices()[0].platform != "tpu":
        return

      (_, (out_m, lb_m, bias_m)), (dp_m, dx_m) = jax.jit(grad_fn)(params, x)

      # Mutating padded token activations to huge values must not affect real-token outputs, grads, lb_loss, or bias!
      x_perturbed_pad = jnp.where(jnp.asarray(is_real)[..., None], x, 100.0)
      (_, (out_p, lb_p, bias_p)), (dp_p, dx_p) = jax.jit(grad_fn)(params, x_perturbed_pad)

    np.testing.assert_array_equal(np.asarray(out_m)[~is_real], 0.0)
    np.testing.assert_array_equal(np.asarray(dx_m)[~is_real], 0.0)
    np.testing.assert_allclose(np.asarray(out_m)[is_real], np.asarray(out_p)[is_real], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(dx_m)[is_real], np.asarray(dx_p)[is_real], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(lb_m), np.asarray(lb_p), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(bias_m), np.asarray(bias_p))
    for g1, g2 in zip(jax.tree_util.tree_leaves(dp_m), jax.tree_util.tree_leaves(dp_p), strict=True):
      np.testing.assert_allclose(np.asarray(g1), np.asarray(g2), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
