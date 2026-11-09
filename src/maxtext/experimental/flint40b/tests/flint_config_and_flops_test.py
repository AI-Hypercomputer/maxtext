# Copyright 2026 Google LLC
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

"""Unit tests for Flint config, FLOPs, 3D sharding, ops, and PyTorch HF ref."""

from __future__ import annotations

import functools
import json
import math
import os
import subprocess
from typing import Any, cast

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
import numpy as np
import torch

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_gmm_v2
from maxtext.experimental.flint40b import flint_loss
from maxtext.experimental.flint40b import flint_model
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import flint_types
from maxtext.experimental.flint40b import ops
from maxtext.experimental.flint40b.tests import hf_reference


class FlintConfigAndFlopsTest(parameterized.TestCase):

  def test_layer_schedule_and_scan_layout(self):
    cfg40b = flint_config.flint_40b_config()
    global_indices = [
        idx for idx in range(cfg40b.model.num_hidden_layers) if flint_config.is_global_layer(idx, cfg40b.model)
    ]
    self.assertEqual(
        global_indices,
        [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48],
    )
    self.assertLen(global_indices, 14)
    self.assertEqual(cfg40b.model.num_hidden_layers - len(global_indices), 35)
    self.assertFalse(flint_config.is_moe_layer(0, cfg40b.model))
    self.assertFalse(flint_config.is_moe_layer(1, cfg40b.model))
    for idx in range(2, cfg40b.model.num_hidden_layers):
      self.assertTrue(flint_config.is_moe_layer(idx, cfg40b.model))

    self.assertEqual(flint_config.flint_scan_layout(cfg40b.model), (2, 11, 46))
    mini_cfg = flint_config.mini_flint_config()
    self.assertEqual(flint_config.flint_scan_layout(mini_cfg.model), (2, 1, 6))

  def test_exact_40b_param_counts_match_abstract_trees(self):
    cfg_scanned = flint_config.flint_40b_config(scan_layers=True)
    cfg_unscanned = flint_config.flint_40b_config(scan_layers=False)
    counts = flint_config.count_flint_params(cfg_scanned.model)
    self.assertEqual(
        counts,
        (39_785_287_456, 2_515_488_544, 2_515_271_680, 216_864),
    )

    for cfg in (cfg_scanned, cfg_unscanned):
      shapes = flint_sharding.abstract_param_shapes(cfg)
      pspecs = flint_sharding.param_pspecs(cfg)
      leaves = jax.tree.leaves(shapes)
      specs = jax.tree.leaves(pspecs)
      total_from_tree = sum(math.prod(s.shape) for s in leaves)
      vec_from_tree = sum(
          math.prod(s.shape) for s, sp in zip(leaves, specs, strict=True) if sp == jax.sharding.PartitionSpec()
      )
      self.assertEqual(total_from_tree, 39_785_287_456)
      self.assertEqual(vec_from_tree, 216_864)
      if cfg.model.scan_layers:
        self.assertLen(shapes.decoder.tail_moe, 3)
        self.assertEqual(shapes.decoder.tail_moe[0].self_attn.sinks.dtype, jnp.float32)
      else:
        self.assertEmpty(shapes.decoder.tail_moe)
        self.assertEqual(shapes.decoder.unscanned_moe[0].self_attn.sinks.dtype, jnp.float32)

  def test_exact_tflops_across_8k_and_all_20_recipes(self):
    cfg_8k = flint_config.flint_40b_config(
        seq_len=8192,
        global_batch_size=128,
        fsdp_size=128,
        cp_size=1,
        ep_size=1,
    )
    total_tf, weight_tf, attn_tf = flint_config.calculate_tflops_per_device(cfg_8k)
    np.testing.assert_allclose(
        (total_tf, weight_tf, attn_tf),
        (153.724089532, 123.630633615, 30.093455917),
        rtol=1e-9,
        atol=0.0,
    )

    recipe_matrix = (
        (8192, 128, (128, 1, 1), 153.724089532416),
        (8192, 256, (128, 1, 1), 307.448179064832),
        (8192, 256, (16, 1, 8), 307.448179064832),
        (8192, 512, (128, 1, 1), 614.896358129664),
        (8192, 1024, (128, 1, 1), 1229.792716259328),
        (16384, 64, (64, 2, 1), 176.926796808192),
        (16384, 128, (128, 1, 1), 353.853593616384),
        (16384, 256, (128, 1, 1), 707.707187232768),
        (16384, 512, (128, 1, 1), 1415.414374465536),
        (32768, 64, (32, 4, 1), 446.325533442048),
        (32768, 128, (128, 1, 1), 892.651066884096),
        (32768, 256, (128, 1, 1), 1785.302133768192),
        (65536, 32, (16, 8, 1), 631.099968454656),
        (65536, 64, (16, 8, 1), 1262.199936909312),
        (65536, 128, (16, 8, 1), 2524.399873818624),
        (131072, 16, (16, 8, 1), 1000.564116160512),
        (131072, 32, (32, 4, 1), 2001.128232321024),
        (131072, 64, (64, 2, 1), 4002.256464642048),
        (262144, 16, (16, 8, 1), 3478.900100825088),
        (262144, 32, (32, 4, 1), 6957.800201650176),
    )
    for seq_len, gbs, (fsdp, cp, ep), expected_total in recipe_matrix:
      rcfg = flint_config.flint_40b_config(
          seq_len=seq_len,
          global_batch_size=gbs,
          fsdp_size=fsdp,
          cp_size=cp,
          ep_size=ep,
      )
      got_total, _, _ = flint_config.calculate_tflops_per_device(rcfg)
      np.testing.assert_allclose(got_total, expected_total, rtol=1e-12)

  def test_fail_fast_validation_errors(self):
    with self.assertRaisesRegex(ValueError, "splash_dq_reduction_steps"):
      flint_config.flint_40b_config(use_splash=True, splash_dq_reduction_steps=None)
    with self.assertRaisesRegex(ValueError, "global_batch_size"):
      flint_config.flint_40b_config(global_batch_size=130)
    with self.assertRaisesRegex(ValueError, "2 \\* cp_size"):
      flint_config.flint_40b_config(
          use_splash=False,
          fsdp_size=64,
          cp_size=2,
          ep_size=1,
          global_batch_size=64,
          seq_len=8194,
      )
    with self.assertRaisesRegex(ValueError, "num_vocab_chunks"):
      flint_config.flint_40b_config(num_vocab_chunks=13)
    with self.assertRaisesRegex(ValueError, "scan_layers requires"):
      flint_config.flint_40b_config(num_hidden_layers=48, scan_layers=True)
    with self.assertRaisesRegex(ValueError, "Unknown FlintConfig override"):
      flint_config.flint_40b_config(nonexistent_knob=123)
    with self.assertRaisesRegex(ValueError, "Unknown FlintConfig override"):
      flint_config.mini_flint_config(bad_override=True)

  @parameterized.parameters(
      (128, 1, 1),
      (64, 2, 1),
      (32, 4, 1),
      (16, 1, 8),
      (16, 8, 1),
      (8, 4, 4),
  )
  def test_aot_tpu7x_4x4x4_mesh_and_persistent_state(self, fsdp_size: int, cp_size: int, ep_size: int):
    gbs = max(128, fsdp_size * ep_size)
    seq_len = 8192 if cp_size == 1 else 8192 * cp_size
    cfg = flint_config.flint_40b_config(
        fsdp_size=fsdp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        global_batch_size=gbs,
        seq_len=seq_len,
        param_dtype="float32",
    )
    mesh = flint_sharding.create_mesh(cfg.sharding, aot_topology="tpu7x:4x4x4")
    self.assertEqual(mesh.devices.size, 128)
    self.assertEqual(mesh.axis_names, ("fsdp", "cp", "ep"))
    self.assertEqual(mesh.devices.shape, (fsdp_size, cp_size, ep_size))

    named = flint_sharding.named_param_shardings(mesh, cfg)
    shapes = flint_sharding.abstract_param_shapes(cfg)
    for sds, sh in zip(jax.tree.leaves(shapes), jax.tree.leaves(named), strict=True):
      shard_shape = sh.shard_shape(sds.shape)
      self.assertGreater(math.prod(shard_shape), 0)

    param_bytes, adam_bytes = flint_sharding.estimate_persistent_state_bytes_per_device(cfg)
    self.assertEqual(param_bytes, 1_244_150_912)
    self.assertEqual(adam_bytes, 3_732_452_736)
    adam_gib = adam_bytes / (1024.0**3)
    self.assertAlmostEqual(adam_gib, 3.4761, places=3)

  def test_init_random_params_fan_in_ignores_scan_and_expert_dims(self):
    cfg_scan = flint_config.mini_flint_config(
        scan_layers=True,
        hidden_size=128,
        moe_intermediate_size=64,
        num_experts=16,
        fsdp_size=2,
        cp_size=2,
        ep_size=2,
        global_batch_size=8,
        seq_len=16,
    )
    cfg_unscan = flint_config.mini_flint_config(
        scan_layers=False,
        hidden_size=128,
        moe_intermediate_size=64,
        num_experts=16,
        fsdp_size=2,
        cp_size=2,
        ep_size=2,
        global_batch_size=8,
        seq_len=16,
    )
    mesh = flint_sharding.create_mesh(cfg_scan.sharding)
    self.assertEqual(mesh.devices.size, 8)
    key = jax.random.PRNGKey(42)
    params_scan = flint_sharding.init_random_params(key, mesh, cfg_scan)
    params_unscan = flint_sharding.init_random_params(key, mesh, cfg_unscan)

    for leaf in jax.tree.leaves(params_scan):
      self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))

    sb = params_scan.decoder.scanned_blocks
    self.assertIsNotNone(sb)
    assert sb is not None
    gu_scan = np.asarray(sb.sliding_0.moe.routed_experts.gate_up_proj)
    down_scan = np.asarray(sb.sliding_0.moe.routed_experts.down_proj)
    gu_unscan = np.asarray(params_unscan.decoder.unscanned_moe[0].moe.routed_experts.gate_up_proj)

    expected_gu_std = 1.0 / math.sqrt(128.0)
    expected_down_std = 1.0 / math.sqrt(64.0)
    self.assertLess(abs(float(np.asarray(params_scan.embed).std()) - 1.0), 0.05)
    self.assertAlmostEqual(float(gu_scan.std()), expected_gu_std, delta=0.01)
    self.assertAlmostEqual(float(gu_unscan.std()), expected_gu_std, delta=0.01)
    self.assertAlmostEqual(float(down_scan.std()), expected_down_std, delta=0.01)
    # Verify gate and up halves are distinct (initialized with separate keys).
    half = cfg_scan.model.moe_intermediate_size
    self.assertGreater(
        float(np.max(np.abs(gu_scan[..., :half] - gu_scan[..., half:]))),
        1e-2,
    )

  def test_rms_norm_and_split_half_rope_match_pytorch(self):
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((2, 16, 128)).astype(np.float32) * 3.0
    w_np = np.linspace(-12.0, 54.0, 128, dtype=np.float32)

    x_torch = torch.tensor(x_np, dtype=torch.bfloat16)
    norm_torch = hf_reference.TorchFlintRMSNorm(128, eps=1e-6)
    with torch.no_grad():
      norm_torch.weight.copy_(torch.tensor(w_np, dtype=torch.bfloat16))
    out_torch = norm_torch(x_torch).detach().to(torch.float32).numpy()

    x_jax = jnp.asarray(x_np, dtype=jnp.bfloat16)
    w_jax = jnp.asarray(w_np, dtype=jnp.bfloat16)
    out_jax = np.asarray(ops.flint_rms_norm(x_jax, w_jax, eps=1e-6).astype(jnp.float32))
    np.testing.assert_array_equal(out_jax, out_torch)

    q_np = rng.standard_normal((2, 16, 8, 32)).astype(np.float32)
    k_np = rng.standard_normal((2, 16, 2, 32)).astype(np.float32)
    pos_np = np.broadcast_to(np.arange(16, dtype=np.int32), (2, 16)).copy()

    q_t32, k_t32 = hf_reference.torch_apply_rotary_pos_emb(
        torch.tensor(q_np),
        torch.tensor(k_np),
        torch.tensor(pos_np, dtype=torch.int64),
        head_dim=32,
        rope_theta=10000.0,
    )
    q_j32, k_j32 = ops.apply_split_half_rope(
        jnp.asarray(q_np),
        jnp.asarray(k_np),
        jnp.asarray(pos_np),
        head_dim=32,
        rope_theta=10000.0,
    )
    np.testing.assert_allclose(np.asarray(q_j32), q_t32.numpy(), atol=1e-6)
    np.testing.assert_allclose(np.asarray(k_j32), k_t32.numpy(), atol=1e-6)

    q_t16, k_t16 = hf_reference.torch_apply_rotary_pos_emb(
        torch.tensor(q_np, dtype=torch.bfloat16),
        torch.tensor(k_np, dtype=torch.bfloat16),
        torch.tensor(pos_np, dtype=torch.int64),
        head_dim=32,
        rope_theta=10000.0,
    )
    q_j16, k_j16 = ops.apply_split_half_rope(
        jnp.asarray(q_np, dtype=jnp.bfloat16),
        jnp.asarray(k_np, dtype=jnp.bfloat16),
        jnp.asarray(pos_np),
        head_dim=32,
        rope_theta=10000.0,
    )
    np.testing.assert_allclose(
        np.asarray(q_j16.astype(jnp.float32)),
        q_t16.to(torch.float32).numpy(),
        atol=2e-2,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        np.asarray(k_j16.astype(jnp.float32)),
        k_t16.to(torch.float32).numpy(),
        atol=2e-2,
        rtol=1e-2,
    )

  @parameterized.parameters(
      (8, 1, 1),
      (2, 4, 1),
      (2, 2, 2),
  )
  def test_cp_bijection_and_all_gather_on_8_cpu_devices(self, fsdp_size: int, cp_size: int, ep_size: int):
    sh_cfg = flint_config.FlintShardingConfig(fsdp_size=fsdp_size, cp_size=cp_size, ep_size=ep_size)
    mesh = flint_sharding.create_mesh(sh_cfg)
    bsz = sh_cfg.dp_ways * 2
    seq_len = 32
    x_global = jnp.arange(bsz * seq_len * 4, dtype=jnp.float32).reshape(bsz, seq_len, 4)
    act_spec = flint_sharding.activation_pspec()
    full_seq_spec = jax.sharding.PartitionSpec(("fsdp", "ep"), None, None)

    def _roundtrip_and_gather(x_nat):
      x_lb = ops.cp_natural_to_lb(x_nat, cp_size, axis=1)
      x_back = ops.cp_lb_to_natural(x_lb, cp_size, axis=1)
      full_from_lb = ops.cp_all_gather_lb_to_natural(x_lb, cp_size, axis=1)
      full_from_nat = ops.cp_all_gather_natural(x_nat, cp_size, axis=1)
      pos_lb = ops.cp_lb_global_positions(seq_len, cp_size)
      return x_back, full_from_lb, full_from_nat, pos_lb

    sharded_fn = jax.shard_map(
        _roundtrip_and_gather,
        mesh=mesh,
        in_specs=(act_spec,),
        out_specs=(
            act_spec,
            full_seq_spec,
            full_seq_spec,
            jax.sharding.PartitionSpec("cp"),
        ),
        check_vma=False,
    )
    x_sharded = jax.device_put(x_global, NamedSharding(mesh, act_spec))
    x_back, full_lb, full_nat, pos_lb = sharded_fn(x_sharded)

    np.testing.assert_array_equal(np.asarray(x_back), np.asarray(x_global))
    np.testing.assert_array_equal(np.asarray(full_lb), np.asarray(x_global))
    np.testing.assert_array_equal(np.asarray(full_nat), np.asarray(x_global))
    np.testing.assert_array_equal(np.asarray(pos_lb), ops.lb_q_sequence_np(seq_len, cp_size))

  def test_check_vma_false_exact_grad_scaling_across_3d_meshes(self):
    base_kwargs = dict(
        scan_layers=False,
        num_hidden_layers=3,
        num_dense_prefix_layers=1,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=8,
        num_experts_per_tok=2,
        global_batch_size=8,
        seq_len=16,
        compute_dtype="float32",
        param_dtype="float32",
    )
    cfg_ref = flint_config.mini_flint_config(fsdp_size=1, cp_size=1, ep_size=1, **base_kwargs)
    ref_devs = jax.devices()[:1]
    mesh_ref = flint_sharding.create_mesh(cfg_ref.sharding, devices=ref_devs)
    raw_params = flint_sharding.init_random_params(jax.random.PRNGKey(99), mesh_ref, cfg_ref)
    host_params = jax.tree.map(np.asarray, raw_params)

    rng = np.random.default_rng(123)
    tokens_np = rng.integers(0, 64, size=(8, 16), dtype=np.int32)
    mask_np = (rng.uniform(size=(8, 16)) > 0.25).astype(np.int32)
    mask_np[0, 0] = 1

    def _run_toy_vg(fsdp_size: int, cp_size: int, ep_size: int):
      cfg = flint_config.mini_flint_config(
          fsdp_size=fsdp_size,
          cp_size=cp_size,
          ep_size=ep_size,
          **base_kwargs,
      )
      devs = jax.devices()[: cfg.sharding.num_devices]
      mesh = flint_sharding.create_mesh(cfg.sharding, devices=devs)
      pspecs = flint_sharding.param_pspecs(cfg)
      shardings = flint_sharding.named_param_shardings(mesh, cfg)
      params = jax.tree.map(jax.device_put, host_params, shardings)
      b_spec = flint_sharding.batch_pspec()
      tokens = jax.device_put(tokens_np, NamedSharding(mesh, b_spec))
      loss_mask = jax.device_put(mask_np, NamedSharding(mesh, b_spec))

      def _local_loss(p_l, tok_l, mask_l):
        embed_full = ops.gather_dense_weight(p_l.embed, cfg, sharded_axis=1, dtype=jnp.float32)
        x = embed_full[tok_l]
        sp0 = p_l.decoder.unscanned_moe[0]
        x = ops.flint_rms_norm(x, sp0.input_layernorm, eps=1e-6)

        q_w = ops.gather_dense_weight(sp0.self_attn.q_proj, cfg, sharded_axis=0)
        o_w = ops.gather_dense_weight(sp0.self_attn.o_proj, cfg, sharded_axis=1)
        sink_scale = jnp.mean(sp0.self_attn.sinks)
        x = x + ((x @ q_w) @ o_w) * (1.0 + sink_scale)

        # Exercise routed experts with EP token all_gather + psum_scatter.
        gu_local = ops.gather_routed_weight(sp0.moe.routed_experts.gate_up_proj, cfg, sharded_axis=1)
        dn_local = ops.gather_routed_weight(sp0.moe.routed_experts.down_proj, cfg, sharded_axis=2)
        flat_x = x.reshape(-1, cfg.model.hidden_size)
        if cfg.sharding.ep_size > 1:
          flat_ep = jax.lax.all_gather(flat_x, "ep", axis=0, tiled=True)
        else:
          flat_ep = flat_x
        moe_inter = cfg.model.moe_intermediate_size
        gate_w = gu_local[..., :moe_inter]
        up_w = gu_local[..., moe_inter:]
        h_exp = jax.nn.silu(jnp.einsum("td,edm->tem", flat_ep, gate_w)) * (jnp.einsum("td,edm->tem", flat_ep, up_w))
        y_partial = jnp.sum(jnp.einsum("tem,emd->ted", h_exp, dn_local), axis=1) / float(cfg.model.num_experts)
        if cfg.sharding.ep_size > 1:
          y_routed = jax.lax.psum_scatter(y_partial, "ep", scatter_dimension=0, tiled=True)
        else:
          y_routed = y_partial
        x = x + y_routed.reshape(x.shape)

        x = ops.flint_rms_norm(x, p_l.final_norm, eps=1e-6)
        head_w = ops.gather_dense_weight(p_l.lm_head, cfg, sharded_axis=0)
        logits = x @ head_w
        tok_loss = jnp.mean(jnp.square(logits), axis=-1) * mask_l.astype(jnp.float32)
        local_sum = jnp.sum(tok_loss)
        local_valid = jnp.sum(mask_l.astype(jnp.float32))
        if cfg.sharding.num_devices > 1:
          global_valid = jax.lax.psum(local_valid, ("fsdp", "cp", "ep"))
        else:
          global_valid = local_valid
        return local_sum / jax.lax.stop_gradient(jnp.maximum(global_valid, 1.0))

      def _sharded_vg(p_l, tok_l, mask_l):
        local_contrib, grads_l = jax.value_and_grad(_local_loss)(p_l, tok_l, mask_l)
        if cfg.sharding.num_devices > 1:
          loss = jax.lax.psum(local_contrib, ("fsdp", "cp", "ep"))
        else:
          loss = local_contrib
        grads_l = ops.psum_replicated_grads(grads_l, pspecs, cfg.sharding)
        return loss, grads_l

      vg_fn = jax.jit(
          jax.shard_map(
              _sharded_vg,
              mesh=mesh,
              in_specs=(pspecs, b_spec, b_spec),
              out_specs=(jax.sharding.PartitionSpec(), pspecs),
              check_vma=False,
          )
      )
      loss, grads = vg_fn(params, tokens, loss_mask)
      return float(np.asarray(loss)), jax.tree.map(np.asarray, grads)

    ref_loss, ref_grads = _run_toy_vg(1, 1, 1)
    for mesh_dims in ((4, 2, 1), (4, 1, 2), (2, 2, 2)):
      got_loss, got_grads = _run_toy_vg(*mesh_dims)
      self.assertAlmostEqual(got_loss, ref_loss, places=5)
      for g_got, g_ref in zip(jax.tree.leaves(got_grads), jax.tree.leaves(ref_grads), strict=True):
        np.testing.assert_allclose(g_got, g_ref, atol=1e-6, rtol=1e-5)

  def test_torch_hf_reference_keys_and_state_dict_conversion(self):
    self.assertLen(
        hf_reference.expected_hf_state_dict_keys(flint_config.flint_40b_config().model),
        36_734,
    )

    mini_cfg = flint_config.mini_flint_config()
    torch_model = hf_reference.TorchFlintForCausalLM(mini_cfg.model)
    hf_reference.randomize_torch_flint(torch_model, seed=11)
    sd = torch_model.state_dict()
    self.assertLen(sd, 454)
    self.assertIn("model.layers.2.mlp.shared_expert.gate_proj.weight", sd)
    self.assertIn("model.layers.0.self_attn.softmax_offset", sd)
    self.assertGreater(
        float(sd["model.layers.0.self_attn.softmax_offset"].abs().max()),
        0.0,
    )

    tokens = torch.randint(0, mini_cfg.model.vocab_size, (2, 16))
    targets = torch.randint(0, mini_cfg.model.vocab_size, (2, 16))
    mask = torch.ones((2, 16), dtype=torch.int32)
    logits, hiddens, router_outs = torch_model(tokens)
    self.assertEqual(tuple(logits.shape), (2, 16, mini_cfg.model.vocab_size))
    self.assertLen(hiddens, 9)
    self.assertIsNone(router_outs[0])
    self.assertIsNotNone(router_outs[2])
    loss = hf_reference.compute_torch_ce_loss(logits, targets, mask)
    loss.backward()
    self.assertIsNotNone(torch_model.model.layers[0].self_attn.softmax_offset.grad)

    for scan_layers in (True, False):
      cfg_1 = flint_config.mini_flint_config(
          scan_layers=scan_layers,
          fsdp_size=1,
          cp_size=1,
          ep_size=1,
          global_batch_size=8,
      )
      cfg_8 = flint_config.mini_flint_config(
          scan_layers=scan_layers,
          fsdp_size=2,
          cp_size=2,
          ep_size=2,
          global_batch_size=8,
      )
      mesh_1 = flint_sharding.create_mesh(cfg_1.sharding, devices=jax.devices()[:1])
      mesh_8 = flint_sharding.create_mesh(cfg_8.sharding)
      p1 = hf_reference.torch_state_dict_to_flint_params(sd, cfg_1, mesh_1)
      p8 = hf_reference.torch_state_dict_to_flint_params(sd, cfg_8, mesh_8)
      for a1, a8 in zip(jax.tree.leaves(p1), jax.tree.leaves(p8), strict=True):
        np.testing.assert_array_equal(np.asarray(a1), np.asarray(a8))

  def test_round1_kernel_defaults_and_validation(self):
    for cfg in (
        flint_config.flint_40b_config(),
        flint_config.mini_flint_config(),
    ):
      k = cfg.kernels
      self.assertEqual(k.moe_impl, "xla_ragged_dot")
      self.assertIsNone(k.gmm_tile_m)
      self.assertIsNone(k.tgmm_tile_m)
      self.assertEqual(k.sc_collective_mode, "off")
      self.assertFalse(k.routed_w_tiled)
      self.assertEqual(k.sc_min_collective_bytes, 2_000_000)
      self.assertFalse(k.moe_bijective_permute)
      self.assertFalse(k.fuse_layer_projections)
      self.assertIsNone(k.splash_global_block_q)
      self.assertIsNone(k.splash_global_block_kv)
      self.assertIsNone(k.splash_global_block_kv_compute)
      self.assertIsNone(k.splash_global_block_q_dkv)
      self.assertIsNone(k.splash_global_block_kv_dkv)
      self.assertIsNone(k.splash_global_block_kv_dkv_compute)
      self.assertFalse(k.splash_qk_diag_skip)
      self.assertFalse(k.splash_sv_diag_skip)
      self.assertEqual(cfg.train.remat_policy, "layer")

    self.assertEqual(
        flint_config.SELECTIVE_SAVED_CHECKPOINT_NAMES,
        (
            "x_norm1",
            "post_attn",
            "x_norm2",
            "attn_qk_pre_norm",
            "attn_q_res",
            "attn_global_kv",
            "attn_local_kv",
            "attn_global_out_lse",
            "attn_local_out_lse",
            "mlp_gate_up",
            "moe_router_pack",
            "moe_router_maps",
            "moe_gate_up",
        ),
    )
    self.assertEqual(
        flint_config.GLOBAL_SELECTIVE_SAVED_CHECKPOINT_NAMES,
        (
            "x_norm1",
            "attn_global_kv",
            "attn_global_out_lse",
            "moe_router_maps",
        ),
    )
    self.assertEqual(
        flint_config.LOCAL_SELECTIVE_SAVED_CHECKPOINT_NAMES,
        ("moe_router_maps",),
    )

    cfg_opt = flint_config.flint_40b_config(
        moe_impl="pallas_gmm_v2",
        gmm_tile_m=256,
        tgmm_tile_m=256,
        sc_collective_mode="dual_sc",
        routed_w_tiled=True,
        moe_bijective_permute=True,
        fuse_layer_projections=True,
        splash_global_block_q=1024,
        splash_global_block_kv=1024,
        splash_global_block_kv_compute=512,
        splash_global_block_q_dkv=1024,
        splash_global_block_kv_dkv=1024,
        splash_global_block_kv_dkv_compute=512,
        splash_qk_diag_skip=True,
        remat_policy="selective",
    )
    self.assertEqual(cfg_opt.kernels.moe_impl, "pallas_gmm_v2")
    self.assertEqual(cfg_opt.train.remat_policy, "selective")

    with self.assertRaisesRegex(ValueError, "moe_impl"):
      flint_config.flint_40b_config(moe_impl="bad_moe")
    with self.assertRaisesRegex(ValueError, "moe_intermediate_size"):
      flint_config.mini_flint_config(moe_impl="pallas_gmm_v2")
    with self.assertRaisesRegex(ValueError, "gmm_tile_m"):
      flint_config.flint_40b_config(gmm_tile_m=64)
    with self.assertRaisesRegex(ValueError, "tgmm_tile_m"):
      flint_config.flint_40b_config(tgmm_tile_m=192)
    with self.assertRaisesRegex(ValueError, "sc_collective_mode"):
      flint_config.flint_40b_config(sc_collective_mode="bad_sc")
    with self.assertRaisesRegex(ValueError, "sc_min_collective_bytes"):
      flint_config.flint_40b_config(sc_min_collective_bytes=-1)
    with self.assertRaisesRegex(ValueError, "routed_w_tiled"):
      flint_config.mini_flint_config(
          hidden_size=128,
          moe_intermediate_size=64,
          fsdp_size=8,
          cp_size=4,
          ep_size=1,
          global_batch_size=8,
          seq_len=64,
          routed_w_tiled=True,
      )
    with self.assertRaisesRegex(ValueError, "splash_global_block_q"):
      flint_config.flint_40b_config(splash_global_block_q=192)
    with self.assertRaisesRegex(ValueError, "splash_qk_diag_skip"):
      flint_config.flint_40b_config(
          splash_global_block_q=128,
          splash_global_block_kv=128,
          splash_qk_diag_skip=True,
      )
    with self.assertRaisesRegex(ValueError, "splash_qk_diag_skip"):
      flint_config.flint_40b_config(
          splash_global_block_q=512,
          splash_global_block_kv=256,
          splash_qk_diag_skip=True,
      )
    with self.assertRaisesRegex(ValueError, "splash_sv_diag_skip"):
      flint_config.flint_40b_config(
          splash_global_block_q=512,
          splash_global_block_kv=512,
          splash_global_block_kv_compute=256,
          splash_sv_diag_skip=True,
      )
    with self.assertRaisesRegex(ValueError, "remat_policy"):
      flint_config.flint_40b_config(remat_policy="bad_remat")
    with self.assertRaisesRegex(ValueError, "routed_w_tiled=False"):
      flint_config.flint_40b_config(shard_routed_on_expert_axis=True, routed_w_tiled=True)
    with self.assertRaisesRegex(ValueError, "ep_size == 1"):
      flint_config.flint_40b_config(
          fsdp_size=64,
          cp_size=1,
          ep_size=2,
          global_batch_size=128,
          shard_routed_on_expert_axis=True,
      )
    with self.assertRaisesRegex(ValueError, "divisible by fsdp_size"):
      flint_config.mini_flint_config(
          num_experts=12,
          fsdp_size=8,
          cp_size=1,
          ep_size=1,
          global_batch_size=8,
          shard_routed_on_expert_axis=True,
      )

    cfg_r2 = flint_config.flint_40b_config(
        replicate_router_gate=True,
        shard_routed_on_expert_axis=True,
        moe_slot_major_pack=True,
        save_extra_selective_residuals=True,
        prefetch_routed_weights=True,
        fp8_routed_all_gather=True,
    )
    self.assertTrue(cfg_r2.kernels.replicate_router_gate)
    self.assertTrue(cfg_r2.kernels.shard_routed_on_expert_axis)
    pspecs_r2 = flint_sharding.param_pspecs(cfg_r2)
    assert pspecs_r2.decoder.scanned_blocks is not None
    self.assertEqual(
        pspecs_r2.decoder.scanned_blocks.sliding_0.moe.router.gate,
        jax.sharding.PartitionSpec(),
    )
    self.assertEqual(
        pspecs_r2.decoder.scanned_blocks.sliding_0.moe.routed_experts.gate_up_proj,
        jax.sharding.PartitionSpec(None, ("fsdp", "cp"), None, None),
    )
    self.assertEqual(
        pspecs_r2.decoder.tail_moe[0].moe.routed_experts.down_proj,
        jax.sharding.PartitionSpec(("fsdp", "cp"), None, None),
    )

    for cfg_def in (
        flint_config.flint_40b_config(),
        flint_config.mini_flint_config(),
    ):
      kr = cfg_def.kernels
      self.assertFalse(kr.transpose_routed_down_ag)
      self.assertFalse(kr.transpose_dense_rhs_ag)
      self.assertFalse(kr.moe_zero_scatter_router)
      self.assertFalse(kr.prefetch_routed_down)
      self.assertFalse(kr.precompute_rope_tables)

    cfg_r3 = flint_config.flint_40b_config(
        transpose_routed_down_ag=True,
        transpose_dense_rhs_ag=True,
        moe_zero_scatter_router=True,
        prefetch_routed_down=True,
        precompute_rope_tables=True,
        splash_dq_reduction_steps=1,
    )
    self.assertTrue(cfg_r3.kernels.transpose_routed_down_ag)
    self.assertTrue(cfg_r3.kernels.transpose_dense_rhs_ag)
    self.assertTrue(cfg_r3.kernels.moe_zero_scatter_router)
    self.assertTrue(cfg_r3.kernels.prefetch_routed_down)
    self.assertTrue(cfg_r3.kernels.precompute_rope_tables)
    self.assertEqual(cfg_r3.kernels.splash_dq_reduction_steps, 1)

    with self.assertRaisesRegex(ValueError, "shard_routed_on_expert_axis"):
      flint_config.flint_40b_config(
          transpose_routed_down_ag=True,
          shard_routed_on_expert_axis=True,
      )
    with self.assertRaisesRegex(ValueError, "splash_dq_reduction_steps"):
      flint_config.flint_40b_config(splash_dq_reduction_steps=True)
    with self.assertRaisesRegex(ValueError, "splash_dq_reduction_steps"):
      flint_config.flint_40b_config(splash_dq_reduction_steps=2)
    with self.assertRaisesRegex(ValueError, "cp_size > 1"):
      flint_config.flint_40b_config(
          fsdp_size=64,
          cp_size=2,
          ep_size=1,
          seq_len=16384,
          global_batch_size=64,
          splash_dq_reduction_steps=1,
      )
    with self.assertRaisesRegex(ValueError, "transpose_routed_down_ag"):
      flint_config.flint_40b_config(transpose_routed_down_ag=1)  # pytype: disable=wrong-arg-types

  def test_pallas_ragged_dot_fails_fast_on_cpu_and_ops_helpers(self):
    lhs = jnp.ones((128, 128), dtype=jnp.float32)
    rhs = jnp.ones((4, 128, 128), dtype=jnp.float32)
    gs = jnp.array([32, 32, 32, 32], dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, "TPU backend"):
      flint_gmm_v2.pallas_ragged_dot(lhs, rhs, gs, jnp.dtype(jnp.float32))
    with self.assertRaisesRegex(ValueError, "TPU backend"):
      flint_gmm_v2.pallas_ragged_dot_transposed_rhs(lhs, rhs, gs, jnp.dtype(jnp.float32))

    rng_rope = np.random.default_rng(77)
    q_r = jnp.asarray(rng_rope.standard_normal((2, 8, 4, 16)), jnp.bfloat16)
    k_r = jnp.asarray(rng_rope.standard_normal((2, 8, 2, 16)), jnp.bfloat16)
    pos_r = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (2, 8))
    cos_r, sin_r = ops.compute_split_half_rope_cos_sin(pos_r, 16, rope_theta=10000.0, dtype=jnp.bfloat16)
    q_a, k_a = ops.apply_split_half_rope(q_r, k_r, pos_r, 16, 10000.0)
    q_b, k_b = ops.apply_precomputed_split_half_rope(q_r, k_r, cos_r, sin_r)
    np.testing.assert_array_equal(np.asarray(q_a), np.asarray(q_b))
    np.testing.assert_array_equal(np.asarray(k_a), np.asarray(k_b))

    rng = np.random.default_rng(1234)
    x_np = rng.standard_normal((16, 8)).astype(np.float32)
    cot_np = rng.standard_normal((16, 8)).astype(np.float32)
    perm_np = rng.permutation(16).astype(np.int32)
    inv_perm_np = np.argsort(perm_np).astype(np.int32)

    x_jax = jnp.asarray(x_np)
    cot_jax = jnp.asarray(cot_np)
    perm_jax = jnp.asarray(perm_np)
    inv_perm_jax = jnp.asarray(inv_perm_np)

    def _perm_loss(a):
      y = ops.bijective_permute(a, perm_jax, inv_perm_jax)
      return jnp.sum(y * cot_jax), y

    (_, y_got), dx_got = jax.value_and_grad(_perm_loss, has_aux=True)(x_jax)
    np.testing.assert_array_equal(np.asarray(y_got), x_np[perm_np])
    np.testing.assert_array_equal(np.asarray(dx_got), cot_np[inv_perm_np])

    w_np = rng.standard_normal((4, 32, 128)).astype(np.float32)
    w_tiled = ops.routed_w_to_tiles(jnp.asarray(w_np))
    self.assertEqual(w_tiled.shape, (4, 4, 8, 128))
    w_restored = ops.routed_w_from_tiles(w_tiled)
    np.testing.assert_array_equal(np.asarray(w_restored), w_np)

    with self.assertRaisesRegex(ValueError, "routed_w_to_tiles"):
      ops.routed_w_to_tiles(jnp.ones((4, 30, 128), dtype=jnp.float32))
    with self.assertRaisesRegex(ValueError, "routed_w_from_tiles"):
      ops.routed_w_from_tiles(jnp.ones((4, 4, 4, 128), dtype=jnp.float32))

  def test_all_64c_shell_recipes_and_optimized_helper(self):
    recipes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recipes")
    expected_recipes = (
        (
            "flint_64c_base.sh",
            "flint-40b-64c-8k",
            8192,
            1024,
            (128, 1, 1),
            "global_selective",
        ),
        (
            "flint_64c_8k.sh",
            "flint-40b-64c-8k",
            8192,
            1024,
            (128, 1, 1),
            "global_selective",
        ),
        (
            "flint_64c_16k.sh",
            "flint-40b-64c-16k",
            16384,
            512,
            (128, 1, 1),
            "global_selective",
        ),
        (
            "flint_64c_32k.sh",
            "flint-40b-64c-32k",
            32768,
            256,
            (128, 1, 1),
            "global_selective",
        ),
        (
            "flint_64c_64k.sh",
            "flint-40b-64c-64k",
            65536,
            128,
            (16, 8, 1),
            "global_selective",
        ),
        (
            "flint_64c_128k.sh",
            "flint-40b-64c-128k",
            131072,
            64,
            (64, 2, 1),
            "global_selective",
        ),
        (
            "flint_64c_256k.sh",
            "flint-40b-64c-256k",
            262144,
            32,
            (32, 4, 1),
            "global_selective",
        ),
        (
            "flint_64c_8k_pdb1.sh",
            "flint-40b-64c-8k-pdb1",
            8192,
            128,
            (128, 1, 1),
            "selective",
        ),
        (
            "flint_64c_8k_pdb2.sh",
            "flint-40b-64c-8k-pdb2",
            8192,
            256,
            (128, 1, 1),
            "selective",
        ),
        (
            "flint_64c_8k_pdb2_ep8.sh",
            "flint-40b-64c-8k-pdb2-ep8",
            8192,
            256,
            (16, 1, 8),
            "global_selective",
        ),
        (
            "flint_64c_8k_pdb4.sh",
            "flint-40b-64c-8k-pdb4",
            8192,
            512,
            (128, 1, 1),
            "global_selective",
        ),
        (
            "flint_64c_8k_pdb8.sh",
            "flint-40b-64c-8k-pdb8",
            8192,
            1024,
            (128, 1, 1),
            "global_selective",
        ),
        (
            "flint_64c_16k_pdb05.sh",
            "flint-40b-64c-16k-pdb05",
            16384,
            64,
            (64, 2, 1),
            "selective",
        ),
        (
            "flint_64c_16k_pdb1.sh",
            "flint-40b-64c-16k-pdb1",
            16384,
            128,
            (128, 1, 1),
            "selective",
        ),
        (
            "flint_64c_16k_pdb2.sh",
            "flint-40b-64c-16k-pdb2",
            16384,
            256,
            (128, 1, 1),
            "global_selective",
        ),
        (
            "flint_64c_16k_pdb4.sh",
            "flint-40b-64c-16k-pdb4",
            16384,
            512,
            (128, 1, 1),
            "global_selective",
        ),
        (
            "flint_64c_32k_pdb05.sh",
            "flint-40b-64c-32k-pdb05",
            32768,
            64,
            (32, 4, 1),
            "global_selective",
        ),
        (
            "flint_64c_32k_pdb1.sh",
            "flint-40b-64c-32k-pdb1",
            32768,
            128,
            (128, 1, 1),
            "global_selective",
        ),
        (
            "flint_64c_32k_pdb2.sh",
            "flint-40b-64c-32k-pdb2",
            32768,
            256,
            (128, 1, 1),
            "global_selective",
        ),
        (
            "flint_64c_64k_pdb025.sh",
            "flint-40b-64c-64k-pdb025",
            65536,
            32,
            (16, 8, 1),
            "global_selective",
        ),
        (
            "flint_64c_64k_pdb05.sh",
            "flint-40b-64c-64k-pdb05",
            65536,
            64,
            (16, 8, 1),
            "global_selective",
        ),
        (
            "flint_64c_64k_pdb1.sh",
            "flint-40b-64c-64k-pdb1",
            65536,
            128,
            (16, 8, 1),
            "global_selective",
        ),
        (
            "flint_64c_128k_pdb0125.sh",
            "flint-40b-64c-128k-pdb0125",
            131072,
            16,
            (16, 8, 1),
            "global_selective",
        ),
        (
            "flint_64c_128k_pdb025.sh",
            "flint-40b-64c-128k-pdb025",
            131072,
            32,
            (32, 4, 1),
            "global_selective",
        ),
        (
            "flint_64c_128k_pdb05.sh",
            "flint-40b-64c-128k-pdb05",
            131072,
            64,
            (64, 2, 1),
            "global_selective",
        ),
        (
            "flint_64c_256k_pdb0125.sh",
            "flint-40b-64c-256k-pdb0125",
            262144,
            16,
            (16, 8, 1),
            "global_selective",
        ),
        (
            "flint_64c_256k_pdb025.sh",
            "flint-40b-64c-256k-pdb025",
            262144,
            32,
            (32, 4, 1),
            "global_selective",
        ),
    )

    def _run_dry(script_name: str, extra_args=(), extra_env=None):
      clean_env = {"PATH": "/usr/bin:/bin", "USER": "testuser"}
      if extra_env:
        clean_env.update(extra_env)
      proc = subprocess.run(
          [os.path.join(recipes_dir, script_name), "--dry-run", *extra_args],
          check=False,
          capture_output=True,
          text=True,
          env=clean_env,
      )
      self.assertEqual(
          proc.returncode,
          0,
          f"{script_name} failed (rc={proc.returncode}): {proc.stderr}",
      )
      parsed: dict[str, str] = {}
      for line in proc.stdout.strip().splitlines():
        key, val = line.split("=", 1)
        parsed[key] = val
      return parsed

    for (
        script_name,
        job_name,
        seq_len,
        gbs,
        (fsdp, cp, ep),
        remat,
    ) in expected_recipes:
      out = _run_dry(script_name)
      self.assertEqual(out["JOB_NAME"], job_name)
      self.assertEqual(out["DATASET_TYPE"], "synthetic")
      self.assertEqual(out["STEPS"], "18")
      self.assertEqual(out["EXTRA_LIBTPU_ARGS"], "")
      self.assertEqual(out["OPTIMIZED_LIBTPU_ARGS"], "true")
      ov = json.loads(out["CONFIG_OVERRIDES_JSON"])
      expected_ov = flint_config.optimized_64c_recipe_overrides(
          seq_len=seq_len,
          global_batch_size=gbs,
          fsdp_size=fsdp,
          cp_size=cp,
          ep_size=ep,
      )
      self.assertEqual(ov, expected_ov)
      if script_name == "flint_64c_256k.sh":
        self.assertEqual(job_name, flint_config.DEFAULT_RECIPE_NAME)
        self.assertEqual(ov, flint_config.default_recipe_overrides())
      self.assertEqual(ov["remat_policy"], remat)
      cfg = flint_config.flint_40b_config(**ov)
      self.assertEqual(cfg.train.seq_len, seq_len)
      self.assertEqual(cfg.train.global_batch_size, gbs)
      self.assertEqual(cfg.train.remat_policy, remat)
      self.assertEqual(cfg.kernels.moe_impl, "pallas_gmm_v2")
      self.assertEqual(cfg.kernels.sc_collective_mode, "dual_sc")
      self.assertEqual(cfg.kernels.use_fp8, remat in ("selective", "global_selective"))
      self.assertTrue(cfg.kernels.splash_use_local_halo_cp)
      self.assertTrue(cfg.kernels.splash_ignore_segment_ids)

    bf16_override_out = _run_dry("flint_64c_8k_pdb1.sh", extra_env={"USE_FP8": "false"})
    bf16_override_cfg = flint_config.flint_40b_config(**json.loads(bf16_override_out["CONFIG_OVERRIDES_JSON"]))
    self.assertFalse(bf16_override_cfg.kernels.use_fp8)

    c4_out = _run_dry("flint_64c_16k.sh", extra_args=("--data=c4",))
    self.assertEqual(c4_out["DATASET_TYPE"], "c4")
    self.assertEqual(c4_out["STEPS"], "60")
    self.assertEqual(c4_out["C4_SHUFFLE_SEED"], "42")
    self.assertEqual(c4_out["LEARNING_RATE"], "3e-5")
    self.assertTrue(json.loads(c4_out["CONFIG_OVERRIDES_JSON"])["splash_ignore_segment_ids"])

    unopt = _run_dry("flint_64c_32k.sh", extra_env={"ENABLE_OPTIMIZED_STACK": "false"})
    self.assertEqual(unopt["EXTRA_LIBTPU_ARGS"], "")
    self.assertEqual(unopt["OPTIMIZED_LIBTPU_ARGS"], "false")
    unopt_cfg = flint_config.flint_40b_config(**json.loads(unopt["CONFIG_OVERRIDES_JSON"]))
    self.assertEqual(unopt_cfg.kernels.moe_impl, "xla_ragged_dot")
    self.assertEqual(unopt_cfg.kernels.sc_collective_mode, "off")
    self.assertEqual(unopt_cfg.train.seq_len, 32768)
    self.assertEqual(unopt_cfg.train.global_batch_size, 256)

    knob_out = _run_dry(
        "flint_64c_8k.sh",
        extra_env={
            "NUM_VOCAB_CHUNKS": "4",
            "USE_FP8": "true",
            "EXTRA_CONFIG_OVERRIDES_JSON": '{"vocab_token_chunk_size": 4096}',
        },
    )
    knob_cfg = flint_config.flint_40b_config(**json.loads(knob_out["CONFIG_OVERRIDES_JSON"]))
    self.assertEqual(knob_cfg.kernels.num_vocab_chunks, 4)
    self.assertEqual(knob_cfg.kernels.vocab_token_chunk_size, 4096)
    self.assertTrue(knob_cfg.kernels.use_fp8)

  def test_fp8_quantize_pmax_exactness_and_mini_model_fwd_bwd(self):
    self.assertFalse(flint_config.flint_40b_config().kernels.use_fp8)
    self.assertTrue(flint_config.flint_40b_config(use_fp8=True).kernels.use_fp8)
    with self.assertRaisesRegex(ValueError, "use_fp8"):
      flint_config.flint_40b_config(use_fp8=cast(Any, 1))
    with self.assertRaisesRegex(ValueError, "fp8_routed_all_gather"):
      flint_config.flint_40b_config(use_fp8=True, fp8_routed_all_gather=True)

    # Zero-tensor quantization safety (zero NaNs / Infs).
    q_zero, sc_zero = ops.quantize_fp8_axis(jnp.zeros((16, 32), dtype=jnp.float32), reduce_axis=1)
    self.assertEqual(q_zero.dtype, jnp.float8_e4m3fn)
    self.assertTrue(np.all(np.isfinite(np.asarray(q_zero, dtype=np.float32))))
    np.testing.assert_array_equal(np.asarray(q_zero, dtype=np.float32), np.zeros((16, 32), np.float32))
    np.testing.assert_array_equal(np.asarray(sc_zero), np.ones((16, 1), np.float32))

    # Bit-exact `lax.pmax` sharded quantization parity vs single-device.
    rng = np.random.default_rng(2026)
    w2d_np = (rng.standard_normal((64, 64)) * 1.5).astype(np.float32)
    w2d_np[0, :] = 0.0
    w2d_np[:, 0] = 0.0
    w3d_np = (rng.standard_normal((4, 64, 64)) * 1.5).astype(np.float32)
    w3d_np[0, :, :] = 0.0

    pspec = jax.sharding.PartitionSpec
    devs4 = jax.devices()[:4]
    for mesh_dims in ((4, 1, 1), (2, 2, 1)):
      cfg4 = flint_config.mini_flint_config(
          hidden_size=64,
          moe_intermediate_size=64,
          num_experts=4,
          num_experts_per_tok=2,
          fsdp_size=mesh_dims[0],
          cp_size=mesh_dims[1],
          ep_size=mesh_dims[2],
          global_batch_size=4,
          seq_len=16,
          routed_w_tiled=True,
          use_fp8=True,
      )
      mesh4 = flint_sharding.create_mesh(cfg4.sharding, devices=devs4)

      for sh_ax in (0, 1):
        w_spec = pspec(("fsdp", "cp", "ep"), None) if sh_ax == 0 else pspec(None, ("fsdp", "cp", "ep"))
        w2d_sh = jax.device_put(w2d_np, NamedSharding(mesh4, w_spec))
        for rhs_t in (False, True):
          q_ref, sc_ref = ops.quantize_fp8_axis(jnp.asarray(w2d_np), reduce_axis=1 if rhs_t else 0)
          sc_exp = jnp.swapaxes(sc_ref, 0, 1) if rhs_t else sc_ref

          fn_dense = jax.jit(
              jax.shard_map(
                  lambda ws, c=cfg4, sa=sh_ax, rt=rhs_t: (
                      ops.prefetch_dense_weight_fp8(ws, c, sharded_axis=sa, rhs_is_transposed=rt)
                  ),
                  mesh=mesh4,
                  in_specs=(w_spec,),
                  out_specs=(pspec(), pspec()),
                  check_vma=False,
              )
          )
          q_got, sc_got = fn_dense(w2d_sh)
          np.testing.assert_array_equal(
              np.asarray(q_got, dtype=np.float32),
              np.asarray(q_ref, dtype=np.float32),
          )
          np.testing.assert_allclose(np.asarray(sc_got), np.asarray(sc_exp), rtol=2e-7, atol=1e-9)

      routed_cases = (
          (1, "gate_up", False, pspec("ep", ("fsdp", "cp"), None)),
          (2, "down", False, pspec("ep", None, ("fsdp", "cp"))),
          (1, "down", True, pspec("ep", ("fsdp", "cp"), None)),
      )
      for sh_ax, role, rhs_t, r_spec in routed_cases:
        w3d_sh = jax.device_put(w3d_np, NamedSharding(mesh4, r_spec))
        q_ref3, sc_ref3 = ops.quantize_fp8_axis(jnp.asarray(w3d_np), reduce_axis=2 if rhs_t else 1)
        sc_exp3 = jnp.swapaxes(sc_ref3, 1, 2) if rhs_t else sc_ref3
        fn_routed = jax.jit(
            jax.shard_map(
                lambda ws, c=cfg4, sa=sh_ax, rl=role, rt=rhs_t: (
                    ops.prefetch_routed_weight_fp8(ws, c, sharded_axis=sa, role=rl, rhs_is_transposed=rt)
                ),
                mesh=mesh4,
                in_specs=(r_spec,),
                out_specs=(
                    pspec("ep", None, None),
                    pspec("ep", None, None),
                ),
                check_vma=False,
            )
        )
        q_got3, sc_got3 = fn_routed(w3d_sh)
        np.testing.assert_array_equal(
            np.asarray(q_got3, dtype=np.float32),
            np.asarray(q_ref3, dtype=np.float32),
        )
        np.testing.assert_allclose(np.asarray(sc_got3), np.asarray(sc_exp3), rtol=2e-7, atol=1e-9)

    # End-to-end 9-layer `mini_flint_config` fwd+bwd on `(4, 2, 1)` CPU mesh.
    common_kw = dict(
        use_splash=False,
        fsdp_size=4,
        cp_size=2,
        ep_size=1,
        global_batch_size=8,
        seq_len=16,
        compute_dtype="bfloat16",
        param_dtype="float32",
        routed_w_tiled=True,
        moe_bijective_permute=True,
        moe_slot_major_pack=True,
        fuse_layer_projections=True,
        replicate_router_gate=True,
        prefetch_routed_weights=True,
        transpose_routed_down_ag=True,
        transpose_dense_rhs_ag=True,
        moe_zero_scatter_router=True,
    )
    cfg_bf16 = flint_config.mini_flint_config(use_fp8=False, **common_kw)
    cfg_fp8 = flint_config.mini_flint_config(use_fp8=True, **common_kw)
    mesh8 = flint_sharding.create_mesh(cfg_bf16.sharding)
    params = flint_sharding.init_random_params(jax.random.key(99), mesh8, cfg_bf16)
    b_sh = flint_model.batch_named_shardings(mesh8)
    tok_np = rng.integers(0, cfg_bf16.model.vocab_size, (8, 16), np.int32)
    tgt_np = rng.integers(0, cfg_bf16.model.vocab_size, (8, 16), np.int32)
    pos_np = np.broadcast_to(np.arange(16, dtype=np.int32), (8, 16)).copy()
    seg_np = np.ones((8, 16), dtype=np.int32)
    msk_np = np.ones((8, 16), dtype=np.int32)
    batch = flint_types.FlintBatch(
        tokens=jax.device_put(tok_np, b_sh.tokens),
        targets=jax.device_put(tgt_np, b_sh.targets),
        positions=jax.device_put(pos_np, b_sh.positions),
        segment_ids=jax.device_put(seg_np, b_sh.segment_ids),
        loss_mask=jax.device_put(msk_np, b_sh.loss_mask),
    )

    (loss_bf16, _, _), grads_bf16 = flint_model.shard_mapped_value_and_grad(
        params, batch, cfg_bf16, mesh8, fp32_grads=True
    )
    (loss_fp8, ce_fp8, aux_fp8), grads_fp8 = flint_model.shard_mapped_value_and_grad(
        params, batch, cfg_fp8, mesh8, fp32_grads=True
    )
    for scalar in (loss_fp8, ce_fp8, aux_fp8):
      self.assertTrue(np.isfinite(float(np.asarray(scalar))))
    self.assertLess(abs(float(loss_fp8) - float(loss_bf16)), 0.08)

    flat_bf16 = np.concatenate([np.asarray(g, dtype=np.float64).reshape(-1) for g in jax.tree.leaves(grads_bf16)])
    flat_fp8 = np.concatenate([np.asarray(g, dtype=np.float64).reshape(-1) for g in jax.tree.leaves(grads_fp8)])
    self.assertTrue(np.all(np.isfinite(flat_fp8)))
    cos_all = float(np.dot(flat_bf16, flat_fp8) / (np.linalg.norm(flat_bf16) * np.linalg.norm(flat_fp8)))
    self.assertGreater(cos_all, 0.95)

  def test_rms_norm_custom_vjp_and_global_selective_remat_parity(self):
    rng = np.random.default_rng(777)

    def _ref_rms_norm(x: jax.Array, w: jax.Array, eps: float) -> jax.Array:
      x_f32 = x.astype(jnp.float32)
      inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)
      return (x_f32 * inv_rms * w.astype(jnp.float32)).astype(x.dtype)

    for shape in ((2, 8, 64), (2, 8, 4, 32)):
      for dtype in (jnp.float32, jnp.bfloat16):
        x_np = rng.standard_normal(shape).astype(np.float32)
        w_np = rng.standard_normal((shape[-1],)).astype(np.float32)
        dy_np = rng.standard_normal(shape).astype(np.float32)
        x_j = jnp.asarray(x_np, dtype=dtype)
        w_j = jnp.asarray(w_np, dtype=jnp.float32)
        dy_j = jnp.asarray(dy_np, dtype=dtype)

        y_got, vjp_got = jax.vjp(lambda x, w: ops.flint_rms_norm(x, w, 1e-6), x_j, w_j)
        y_ref, vjp_ref = jax.vjp(lambda x, w: _ref_rms_norm(x, w, 1e-6), x_j, w_j)
        dx_got, dw_got = vjp_got(dy_j)
        dx_ref, dw_ref = vjp_ref(dy_j)
        np.testing.assert_allclose(
            np.asarray(y_got, dtype=np.float32),
            np.asarray(y_ref, dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(dx_got, dtype=np.float32),
            np.asarray(dx_ref, dtype=np.float32),
            rtol=1e-4 if dtype == jnp.bfloat16 else 1e-5,
            atol=1e-4 if dtype == jnp.bfloat16 else 1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(dw_got, dtype=np.float32),
            np.asarray(dw_ref, dtype=np.float32),
            rtol=1e-4 if dtype == jnp.bfloat16 else 1e-5,
            atol=1e-4 if dtype == jnp.bfloat16 else 1e-5,
        )

    # Verify global_selective remat matches layer and none on mini_flint_config.
    base_kw = dict(
        use_splash=False,
        fsdp_size=4,
        cp_size=2,
        ep_size=1,
        global_batch_size=8,
        seq_len=16,
        compute_dtype="float32",
        param_dtype="float32",
    )
    cfg_none = flint_config.mini_flint_config(remat_policy="none", **base_kw)
    cfg_gsel = flint_config.mini_flint_config(remat_policy="global_selective", **base_kw)
    cfg_sel = flint_config.mini_flint_config(remat_policy="selective", **base_kw)
    mesh8 = flint_sharding.create_mesh(cfg_none.sharding)
    params = flint_sharding.init_random_params(jax.random.key(123), mesh8, cfg_none)
    b_sh = flint_model.batch_named_shardings(mesh8)
    tok_np = rng.integers(0, cfg_none.model.vocab_size, (8, 16), np.int32)
    tgt_np = rng.integers(0, cfg_none.model.vocab_size, (8, 16), np.int32)
    pos_np = np.broadcast_to(np.arange(16, dtype=np.int32), (8, 16)).copy()
    seg_np = np.ones((8, 16), dtype=np.int32)
    msk_np = np.ones((8, 16), dtype=np.int32)
    batch = flint_types.FlintBatch(
        tokens=jax.device_put(tok_np, b_sh.tokens),
        targets=jax.device_put(tgt_np, b_sh.targets),
        positions=jax.device_put(pos_np, b_sh.positions),
        segment_ids=jax.device_put(seg_np, b_sh.segment_ids),
        loss_mask=jax.device_put(msk_np, b_sh.loss_mask),
    )
    (loss_ref, _, _), grads_ref = flint_model.shard_mapped_value_and_grad(params, batch, cfg_none, mesh8, fp32_grads=True)
    for cand_cfg in (cfg_gsel, cfg_sel):
      (loss_got, _, _), grads_got = flint_model.shard_mapped_value_and_grad(
          params, batch, cand_cfg, mesh8, fp32_grads=True
      )
      np.testing.assert_allclose(float(loss_got), float(loss_ref), rtol=1e-5, atol=1e-5)
      for g_got, g_ref in zip(jax.tree.leaves(grads_got), jax.tree.leaves(grads_ref), strict=True):
        np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_ref), rtol=1e-5, atol=1e-5)

  def test_v4_lb_cp_layout_and_fused_vocab_ce_fp32_parity(self):
    rng = np.random.default_rng(2026)
    base_kw = dict(
        use_splash=False,
        fsdp_size=4,
        cp_size=2,
        ep_size=1,
        global_batch_size=8,
        seq_len=32,
        compute_dtype="float32",
        param_dtype="float32",
        num_vocab_chunks=1,
        vocab_token_chunk_size=128,
    )
    cfg_ref = flint_config.mini_flint_config(
        use_lb_cp_layout=False,
        use_fused_vocab_vjp_scan=False,
        **base_kw,
    )
    cfg_v4 = flint_config.mini_flint_config(
        use_lb_cp_layout=True,
        early_global_kv_proj=True,
        use_pallas_lb_kv_reorder=True,
        use_fused_qk_prologue=True,
        compact_router_maps=True,
        skip_router_aux_stats_when_zero_coef=True,
        use_fused_vocab_vjp_scan=True,
        **base_kw,
    )
    mesh8 = flint_sharding.create_mesh(cfg_ref.sharding)
    params = flint_sharding.init_random_params(jax.random.key(314), mesh8, cfg_ref)

    # Direct single-scan fused vocab CE parity against dense reference.
    h_np = rng.standard_normal((8, 32, cfg_ref.model.hidden_size)).astype(np.float32)
    tgt_np = rng.integers(0, cfg_ref.model.vocab_size, (8, 32), np.int32)
    msk_np = (rng.random((8, 32)) > 0.2).astype(np.int32)
    msk_np[:, -1] = 0
    tgt_np[msk_np == 0] = -1
    act_sh = NamedSharding(mesh8, flint_sharding.activation_pspec())
    b_sh = flint_model.batch_named_shardings(mesh8)
    h_sh = jax.device_put(h_np, act_sh)
    tgt_sh = jax.device_put(tgt_np, b_sh.targets)
    msk_sh = jax.device_put(msk_np, b_sh.loss_mask)

    def _make_ce_runner(ce_fn, cfg_ce):
      pspec = jax.sharding.PartitionSpec

      @jax.jit
      @functools.partial(
          jax.shard_map,
          mesh=mesh8,
          in_specs=(
              flint_sharding.activation_pspec(),
              pspec(("fsdp", "cp", "ep"), None),
              flint_sharding.batch_pspec(),
              flint_sharding.batch_pspec(),
          ),
          out_specs=(
              pspec(),
              flint_sharding.activation_pspec(),
              pspec(("fsdp", "cp", "ep"), None),
          ),
          check_vma=False,
      )
      def _inner(x_l, w_l, t_l, m_l):
        val, vjp_fn = jax.vjp(
            lambda xl, wl: ce_fn(xl, wl, t_l, m_l, cfg_ce),
            x_l,
            w_l,
        )
        dxl, dwl = vjp_fn(jnp.float32(1.75))
        return jax.lax.psum(val, ("fsdp", "cp", "ep")), dxl, dwl

      return _inner

    loss_dense, dh_dense, dw_dense = _make_ce_runner(flint_loss.dense_ce_loss_sum, cfg_ref)(
        h_sh, params.lm_head, tgt_sh, msk_sh
    )
    loss_got_ce, dh_got_ce, dw_got_ce = _make_ce_runner(flint_loss.chunked_lm_head_ce_sum, cfg_v4)(
        h_sh, params.lm_head, tgt_sh, msk_sh
    )
    np.testing.assert_allclose(float(loss_got_ce), float(loss_dense), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(dh_got_ce), np.asarray(dh_dense), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(dw_got_ce), np.asarray(dw_dense), rtol=1e-5, atol=1e-5)

    # Full-model forward + backward parity across 2-stripe LB CP layout.
    tok_np = rng.integers(0, cfg_ref.model.vocab_size, (8, 32), np.int32)
    pos_np = np.zeros((8, 32), dtype=np.int32)
    seg_np = np.ones((8, 32), dtype=np.int32)
    for b in range(8):
      cut = 12 + (b % 5)
      pos_np[b, :cut] = np.arange(cut, dtype=np.int32)
      pos_np[b, cut:] = np.arange(32 - cut, dtype=np.int32)
      seg_np[b, cut:] = 2
    batch = flint_types.FlintBatch(
        tokens=jax.device_put(tok_np, b_sh.tokens),
        targets=tgt_sh,
        positions=jax.device_put(pos_np, b_sh.positions),
        segment_ids=jax.device_put(seg_np, b_sh.segment_ids),
        loss_mask=msk_sh,
    )

    logits_ref = flint_model.forward_logits(params, batch, cfg_ref, mesh8)
    logits_v4 = flint_model.forward_logits(params, batch, cfg_v4, mesh8)
    np.testing.assert_allclose(np.asarray(logits_v4), np.asarray(logits_ref), rtol=1e-5, atol=1e-5)

    logits_i_ref, hiddens_ref, _ = flint_model.forward_with_intermediates(params, batch, cfg_ref, mesh8)
    logits_i_v4, hiddens_v4, _ = flint_model.forward_with_intermediates(params, batch, cfg_v4, mesh8)
    np.testing.assert_allclose(np.asarray(logits_i_v4), np.asarray(logits_i_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(hiddens_v4[-1]),
        np.asarray(hiddens_ref[-1]),
        rtol=1e-5,
        atol=1e-5,
    )

    (loss_ref, ce_ref, _), grads_ref = flint_model.shard_mapped_value_and_grad(
        params, batch, cfg_ref, mesh8, fp32_grads=True
    )
    (loss_v4, ce_v4, aux_v4), grads_v4 = flint_model.shard_mapped_value_and_grad(
        params, batch, cfg_v4, mesh8, fp32_grads=True
    )
    self.assertEqual(float(aux_v4), 0.0)
    np.testing.assert_allclose(float(loss_v4), float(loss_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(ce_v4), float(ce_ref), rtol=1e-5, atol=1e-5)
    for g_got, g_ref in zip(jax.tree.leaves(grads_v4), jax.tree.leaves(grads_ref), strict=True):
      np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_ref), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
