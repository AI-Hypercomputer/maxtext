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

"""TPU v7x tests for Flint hybrid GQA attention + Splash + CP."""

from __future__ import annotations

from typing import Any, cast

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np
from tokamax._src.ops.splash_attention import splash_attention_mask as tokamax_splash_mask
from tokamax._src.ops.splash_attention import splash_attention_mask_info as tokamax_splash_mask_info
import torch

from maxtext.experimental.flint40b import flint_attention
from maxtext.experimental.flint40b import flint_attention_prologue
from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import flint_types
from maxtext.experimental.flint40b import ops
from maxtext.experimental.flint40b.tests import hf_reference

P = PartitionSpec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
  a_f64 = a.astype(np.float64).reshape(-1)
  b_f64 = b.astype(np.float64).reshape(-1)
  denom = np.linalg.norm(a_f64) * np.linalg.norm(b_f64)
  if denom == 0.0:
    return 1.0 if np.allclose(a_f64, b_f64) else 0.0
  return float(np.dot(a_f64, b_f64) / denom)


def _make_test_config(
    *,
    fsdp_size: int,
    cp_size: int,
    ep_size: int,
    use_splash: bool,
    compute_dtype: str = "float32",
    param_dtype: str = "float32",
    seq_len: int = 2048,
    global_batch_size: int = 8,
    hidden_size: int = 512,
    use_base2_exp: bool = False,
    fuse_layer_projections: bool = False,
    sc_collective_mode: str = "off",
    splash_global_block_size: int | None = None,
    splash_qk_diag_skip: bool = False,
    splash_sv_diag_skip: bool = False,
    splash_dq_reduction_steps: int = 3,
    splash_sliding_block_size: int = 512,
    transpose_dense_rhs_ag: bool = False,
    precompute_rope_tables: bool = False,
) -> flint_config.FlintConfig:
  model_cfg = flint_config.FlintModelConfig(
      vocab_size=2560,
      hidden_size=hidden_size,
      intermediate_size=1024,
      moe_intermediate_size=256,
      shared_expert_intermediate_size=512,
      num_hidden_layers=9,
      num_dense_prefix_layers=2,
      moe_cycle=4,
      num_attention_heads=32,
      num_key_value_heads=4,
      head_dim=128,
      num_experts=16,
      num_experts_per_tok=6,
      sliding_window=513,
      scan_layers=True,
  )
  sharding_cfg = flint_config.FlintShardingConfig(
      fsdp_size=fsdp_size,
      cp_size=cp_size,
      ep_size=ep_size,
  )
  kernel_cfg = flint_config.FlintKernelConfig(
      use_splash=use_splash,
      splash_block_q=splash_sliding_block_size,
      splash_block_kv=splash_sliding_block_size,
      splash_block_kv_compute=min(512, splash_sliding_block_size),
      splash_block_q_dkv=splash_sliding_block_size,
      splash_block_kv_dkv=splash_sliding_block_size,
      splash_block_kv_dkv_compute=min(512, splash_sliding_block_size),
      splash_dq_reduction_steps=splash_dq_reduction_steps,
      splash_use_base2_exp=use_base2_exp,
      splash_fuse_reciprocal=True,
      num_vocab_chunks=5,
      sc_collective_mode=sc_collective_mode,
      fuse_layer_projections=fuse_layer_projections,
      splash_global_block_q=splash_global_block_size,
      splash_global_block_kv=splash_global_block_size,
      splash_global_block_kv_compute=splash_global_block_size,
      splash_global_block_q_dkv=splash_global_block_size,
      splash_global_block_kv_dkv=splash_global_block_size,
      splash_global_block_kv_dkv_compute=splash_global_block_size,
      splash_qk_diag_skip=splash_qk_diag_skip,
      splash_sv_diag_skip=splash_sv_diag_skip,
      transpose_dense_rhs_ag=transpose_dense_rhs_ag,
      precompute_rope_tables=precompute_rope_tables,
  )
  train_cfg = flint_config.FlintTrainConfig(
      global_batch_size=global_batch_size,
      seq_len=seq_len,
      grad_accum_steps=1,
      compute_dtype=compute_dtype,
      param_dtype=param_dtype,
  )
  return flint_config.FlintConfig(
      model=model_cfg,
      sharding=sharding_cfg,
      kernels=kernel_cfg,
      train=train_cfg,
  )


def _shard_array(arr: np.ndarray, mesh: Mesh, spec: PartitionSpec) -> jax.Array:
  sharding = NamedSharding(mesh, spec)
  return jax.make_array_from_callback(arr.shape, sharding, lambda idx: arr[idx])


def _shard_attn_weights(
    np_weights: flint_types.FlintAttnWeights[np.ndarray],
    mesh: Mesh,
    attn_pspecs: flint_types.FlintAttnWeights[PartitionSpec],
) -> flint_types.FlintAttnWeights[jax.Array]:
  return jax.tree.map(
      lambda arr, spec: _shard_array(arr, mesh, spec),
      np_weights,
      attn_pspecs,
  )


def _gather_to_np(arr: jax.Array) -> np.ndarray:
  return np.asarray(jax.device_get(arr))


def _make_attn_fwd_and_bwd(
    mesh: Mesh,
    cfg: flint_config.FlintConfig,
    *,
    is_global: bool,
):
  attn_pspecs = flint_sharding.param_pspecs(cfg).decoder.prefix_dense[0].self_attn
  act_spec = flint_sharding.activation_pspec()
  tok_spec = flint_sharding.batch_pspec()

  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=(attn_pspecs, act_spec, tok_spec, tok_spec, act_spec),
      out_specs=(P(), act_spec, attn_pspecs, act_spec),
      check_vma=False,
  )
  def _step(
      w_shard: flint_types.FlintAttnWeights[jax.Array],
      x_loc: jax.Array,
      pos_loc: jax.Array,
      seg_loc: jax.Array,
      upstream_loc: jax.Array,
  ):
    def _local_loss(
        w_inner: flint_types.FlintAttnWeights[jax.Array],
        x_inner: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
      out_loc = flint_attention.flint_attention_forward(
          x_inner,
          w_inner,
          pos_loc,
          seg_loc,
          cfg,
          is_global=is_global,
          rope_cos_sin=(
              ops.compute_split_half_rope_cos_sin(
                  pos_loc,
                  cfg.model.head_dim,
                  rope_theta=cfg.model.rope_theta,
                  dtype=cfg.train.compute_jnp_dtype,
              )
              if (not is_global and cfg.kernels.precompute_rope_tables)
              else None
          ),
      )
      local_sum = jnp.sum(out_loc.astype(jnp.float32) * upstream_loc.astype(jnp.float32))
      local_count = jnp.asarray(out_loc.size, dtype=jnp.float32)
      global_count = jax.lax.stop_gradient(jax.lax.psum(local_count, ("fsdp", "cp", "ep")))
      return local_sum / global_count, out_loc

    (local_loss, out_loc), (grads_w, grad_x_loc) = jax.value_and_grad(_local_loss, argnums=(0, 1), has_aux=True)(
        w_shard, x_loc
    )
    global_loss = jax.lax.psum(local_loss, ("fsdp", "cp", "ep"))
    grads_w = cast(
        flint_types.FlintAttnWeights[jax.Array],
        ops.psum_replicated_grads(
            cast(Any, grads_w),
            cast(Any, attn_pspecs),
            cfg.sharding,
        ),
    )
    return global_loss, out_loc, grads_w, grad_x_loc

  return _step


def _make_attn_fwd_only(
    mesh: Mesh,
    cfg: flint_config.FlintConfig,
    *,
    is_global: bool,
):
  attn_pspecs = flint_sharding.param_pspecs(cfg).decoder.prefix_dense[0].self_attn
  act_spec = flint_sharding.activation_pspec()
  tok_spec = flint_sharding.batch_pspec()

  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=(attn_pspecs, act_spec, tok_spec, tok_spec),
      out_specs=act_spec,
      check_vma=False,
  )
  def _fwd(
      w_shard: flint_types.FlintAttnWeights[jax.Array],
      x_loc: jax.Array,
      pos_loc: jax.Array,
      seg_loc: jax.Array,
  ) -> jax.Array:
    return flint_attention.flint_attention_forward(
        x_loc,
        w_shard,
        pos_loc,
        seg_loc,
        cfg,
        is_global=is_global,
    )

  return _fwd


class FlintAttentionTpuTest(absltest.TestCase):
  """Verifies Flint hybrid GQA attention, fast MaskInfo, Splash HLO, and CP."""

  def test_fast_mask_info_matches_tokamax_process_mask(self) -> None:
    seq_len = 2048
    for mask_kind in ("causal", "local"):
      for cp_size in (1, 2, 4):
        for is_dkv in (False, True):
          for return_dynamic_grid in (True, False):
            for bq in (128, 256):
              for bkv in (128, 256, 512):
                with self.subTest(
                    mask_kind=mask_kind,
                    cp_size=cp_size,
                    is_dkv=is_dkv,
                    dyn=return_dynamic_grid,
                    bq=bq,
                    bkv=bkv,
                ):
                  if mask_kind == "causal":
                    mask = tokamax_splash_mask.CausalMask(shape=(seq_len, seq_len), offset=0)
                  else:
                    mask = tokamax_splash_mask.LocalMask(
                        shape=(seq_len, seq_len),
                        window_size=(512, 0),
                        offset=0,
                    )
                  mask.q_sequence = ops.lb_q_sequence_np(seq_len, cp_size)

                  fast_info, fast_fn = flint_attention.make_fast_tokamax_mask_info(
                      mask,
                      (bq, bkv),
                      is_dkv=is_dkv,
                      return_dynamic_grid=return_dynamic_grid,
                      q_seq_shards=cp_size,
                  )
                  ref_info, ref_fn = tokamax_splash_mask_info._process_mask(
                      mask,
                      (bq, bkv),
                      is_dkv=is_dkv,
                      return_dynamic_grid=return_dynamic_grid,
                      downcast_smem_data=True,
                      q_seq_shards=cp_size,
                  )

                  self.assertEqual(fast_fn is None, ref_fn is None)
                  if ref_fn is not None:
                    self.assertIs(fast_fn, ref_fn)

                  for field in tokamax_splash_mask_info.MaskInfo._fields:
                    fast_val = getattr(fast_info, field)
                    ref_val = getattr(ref_info, field)
                    self.assertEqual(
                        fast_val is None,
                        ref_val is None,
                        msg=f"Mismatch on None-ness for {field}",
                    )
                    if ref_val is not None:
                      self.assertEqual(
                          fast_val.dtype,
                          ref_val.dtype,
                          msg=f"dtype mismatch on {field}",
                      )
                      self.assertEqual(
                          fast_val.shape,
                          ref_val.shape,
                          msg=f"shape mismatch on {field}",
                      )
                      np.testing.assert_array_equal(
                          fast_val,
                          ref_val,
                          err_msg=f"Bitwise mismatch on {field}",
                      )

    bad_causal = tokamax_splash_mask.CausalMask(shape=(512, 512), offset=1)
    with self.assertRaises(ValueError):
      flint_attention.make_fast_tokamax_mask_info(
          bad_causal,
          (128, 128),
          is_dkv=False,
          return_dynamic_grid=True,
          q_seq_shards=1,
      )

    bad_local = tokamax_splash_mask.LocalMask(shape=(512, 512), window_size=(512, 1), offset=0)
    with self.assertRaises(ValueError):
      flint_attention.make_fast_tokamax_mask_info(
          bad_local,
          (128, 128),
          is_dkv=False,
          return_dynamic_grid=True,
          q_seq_shards=1,
      )

    bad_qseq = tokamax_splash_mask.CausalMask(shape=(512, 512), offset=0)
    scrambled = np.arange(512, dtype=np.int32)
    scrambled[0], scrambled[1] = scrambled[1], scrambled[0]
    bad_qseq.q_sequence = scrambled
    with self.assertRaises(ValueError):
      flint_attention.make_fast_tokamax_mask_info(
          bad_qseq,
          (128, 128),
          is_dkv=False,
          return_dynamic_grid=True,
          q_seq_shards=1,
      )

  def test_splash_pallas_present_in_lowered_hlo(self) -> None:
    self.assertEqual(jax.default_backend(), "tpu")
    self.assertLen(jax.devices(), 8)

    for is_global in (True, False):
      for compute_dtype in ("bfloat16", "float32"):
        cfg = _make_test_config(
            fsdp_size=4,
            cp_size=2,
            ep_size=1,
            use_splash=True,
            compute_dtype=compute_dtype,
        )
        mesh = flint_sharding.create_mesh(cfg.sharding)
        attn_pspecs = flint_sharding.param_pspecs(cfg).decoder.prefix_dense[0].self_attn
        attn_sds = flint_sharding.abstract_param_shapes(cfg).decoder.prefix_dense[0].self_attn
        sharded_w = jax.tree.map(
            lambda sds, spec, m=mesh: jax.ShapeDtypeStruct(sds.shape, sds.dtype, sharding=NamedSharding(m, spec)),
            attn_sds,
            attn_pspecs,
        )
        act_sharding = NamedSharding(mesh, flint_sharding.activation_pspec())
        tok_sharding = NamedSharding(mesh, flint_sharding.batch_pspec())
        b, s, d = (
            cfg.train.global_batch_size,
            cfg.train.seq_len,
            cfg.model.hidden_size,
        )
        x_sds = jax.ShapeDtypeStruct((b, s, d), cfg.train.compute_jnp_dtype, sharding=act_sharding)
        pos_sds = jax.ShapeDtypeStruct((b, s), jnp.int32, sharding=tok_sharding)
        seg_sds = jax.ShapeDtypeStruct((b, s), jnp.int32, sharding=tok_sharding)
        up_sds = jax.ShapeDtypeStruct((b, s, d), jnp.float32, sharding=act_sharding)

        step_fn = _make_attn_fwd_and_bwd(mesh, cfg, is_global=is_global)
        lowered = step_fn.lower(sharded_w, x_sds, pos_sds, seg_sds, up_sds)
        hlo_text = lowered.as_text("hlo", debug_info=True)
        stablehlo_text = lowered.as_text("stablehlo")
        self.assertIn(
            "splash_mha",
            hlo_text,
            msg=f"Missing splash_mha in HLO for {is_global=}, {compute_dtype=}",
        )
        self.assertIn(
            "splash_mha_fwd",
            stablehlo_text,
            msg=(f"Missing splash_mha_fwd in StableHLO for {is_global=}," f" {compute_dtype=}"),
        )
        self.assertIn(
            "splash_mha_dkv",
            stablehlo_text,
            msg=(f"Missing splash_mha_dkv in StableHLO for {is_global=}," f" {compute_dtype=}"),
        )

        cfg_ref = _make_test_config(
            fsdp_size=4,
            cp_size=2,
            ep_size=1,
            use_splash=False,
            compute_dtype=compute_dtype,
        )
        ref_step_fn = _make_attn_fwd_and_bwd(mesh, cfg_ref, is_global=is_global)
        ref_lowered = ref_step_fn.lower(sharded_w, x_sds, pos_sds, seg_sds, up_sds)
        self.assertNotIn("splash_mha", ref_lowered.as_text("hlo", debug_info=True))
        self.assertNotIn("splash_mha", ref_lowered.as_text("stablehlo"))

  def test_fp32_and_bf16_splash_vs_dot_ref_and_pytorch_fwd_bwd(self) -> None:
    self.assertEqual(jax.default_backend(), "tpu")
    self.assertLen(jax.devices(), 8)

    gbs = 8
    seq_len = 2048
    hidden_size = 512
    base_cfg = _make_test_config(
        fsdp_size=8,
        cp_size=1,
        ep_size=1,
        use_splash=False,
        compute_dtype="float32",
        param_dtype="float32",
        seq_len=seq_len,
        global_batch_size=gbs,
        hidden_size=hidden_size,
    )
    m = base_cfg.model

    rng = np.random.default_rng(20261107)
    x_np = (rng.standard_normal((gbs, seq_len, hidden_size)) * 0.5 + 0.25).astype(np.float32)
    raw_up = rng.standard_normal((gbs, seq_len, hidden_size)).astype(np.float32)

    # Multi-segment document IDs: [1 on 0..1023, 2 on 1024..2047] on even batch
    # rows, and 3 segments [1 on 0..639, 2 on 640..1407, 3 on 1408..2047] on odd
    # batch rows to exercise per-example `jax.vmap` over `SegmentIds`.
    seg_np = np.ones((gbs, seq_len), dtype=np.int32)
    pos_np = np.zeros((gbs, seq_len), dtype=np.int32)
    for b_idx in range(gbs):
      if b_idx % 2 == 0:
        seg_np[b_idx, :1024] = 1
        seg_np[b_idx, 1024:] = 2
        pos_np[b_idx, :1024] = np.arange(1024, dtype=np.int32)
        pos_np[b_idx, 1024:] = np.arange(1024, dtype=np.int32)
      else:
        seg_np[b_idx, :640] = 1
        seg_np[b_idx, 640:1408] = 2
        seg_np[b_idx, 1408:] = 3
        pos_np[b_idx, :640] = np.arange(640, dtype=np.int32)
        pos_np[b_idx, 640:1408] = np.arange(768, dtype=np.int32)
        pos_np[b_idx, 1408:] = np.arange(640, dtype=np.int32)

    q_dim = m.num_attention_heads * m.head_dim
    kv_dim = m.num_key_value_heads * m.head_dim
    sinks_np = np.linspace(-1.5, 1.5, m.num_attention_heads, dtype=np.float32)
    q_norm_np = np.linspace(2.0, 8.0, m.head_dim, dtype=np.float32)
    k_norm_np = np.linspace(0.5, 1.2, m.head_dim, dtype=np.float32)
    np_weights_f32 = flint_types.FlintAttnWeights(
        q_proj=(rng.standard_normal((hidden_size, q_dim)) / np.sqrt(float(hidden_size))).astype(np.float32),
        k_proj=(rng.standard_normal((hidden_size, kv_dim)) / np.sqrt(float(hidden_size))).astype(np.float32),
        v_proj=(rng.standard_normal((hidden_size, kv_dim)) / np.sqrt(float(hidden_size)) * 0.5).astype(np.float32),
        o_proj=(rng.standard_normal((q_dim, hidden_size)) / np.sqrt(float(q_dim)) * 0.5).astype(np.float32),
        q_norm=q_norm_np,
        k_norm=k_norm_np,
        sinks=sinks_np,
    )
    gqa = m.num_attention_heads // m.num_key_value_heads
    v_head_dir = np.repeat(
        (np.ones((1, hidden_size), dtype=np.float32) @ np_weights_f32.v_proj).reshape(m.num_key_value_heads, m.head_dim),
        gqa,
        axis=0,
    ).reshape(1, 1, q_dim)
    coherent_out_dir = v_head_dir @ np_weights_f32.o_proj
    coherent_out_dir = coherent_out_dir / np.linalg.norm(coherent_out_dir) * np.sqrt(float(hidden_size))
    upstream_np = (raw_up * 8.0 + coherent_out_dir * 32.0).astype(np.float32)

    meshes = (
        (8, 1, 1),
        (4, 2, 1),
        (2, 4, 1),
        (2, 2, 2),
    )

    for is_global in (True, False):
      pt_attn = hf_reference.TorchFlintAttention(m, is_global=is_global).to(dtype=torch.float32)
      with torch.no_grad():
        pt_attn.q_proj.weight.copy_(torch.from_numpy(np_weights_f32.q_proj.T))
        pt_attn.k_proj.weight.copy_(torch.from_numpy(np_weights_f32.k_proj.T))
        pt_attn.v_proj.weight.copy_(torch.from_numpy(np_weights_f32.v_proj.T))
        pt_attn.o_proj.weight.copy_(torch.from_numpy(np_weights_f32.o_proj.T))
        pt_attn.q_norm.weight.copy_(torch.from_numpy(np_weights_f32.q_norm))
        pt_attn.k_norm.weight.copy_(torch.from_numpy(np_weights_f32.k_norm))
        pt_attn.softmax_offset.copy_(torch.from_numpy(np_weights_f32.sinks))

      x_pt = torch.from_numpy(x_np).requires_grad_(True)
      pos_pt = torch.from_numpy(pos_np.astype(np.int64))
      seg_pt = torch.from_numpy(seg_np.astype(np.int64))
      up_pt = torch.from_numpy(upstream_np)
      out_pt_tensor = pt_attn(x_pt, positions=pos_pt, segment_ids=seg_pt)
      loss_pt_tensor = (out_pt_tensor * up_pt).mean()
      loss_pt_tensor.backward()

      out_pt = out_pt_tensor.detach().numpy()
      loss_pt = float(loss_pt_tensor.detach().item())
      assert x_pt.grad is not None
      dx_pt = x_pt.grad.detach().numpy()
      assert pt_attn.q_proj.weight.grad is not None
      assert pt_attn.k_proj.weight.grad is not None
      assert pt_attn.v_proj.weight.grad is not None
      assert pt_attn.o_proj.weight.grad is not None
      assert pt_attn.q_norm.weight.grad is not None
      assert pt_attn.k_norm.weight.grad is not None
      assert pt_attn.softmax_offset.grad is not None
      pt_grads = {
          "q_proj": pt_attn.q_proj.weight.grad.detach().numpy().T,
          "k_proj": pt_attn.k_proj.weight.grad.detach().numpy().T,
          "v_proj": pt_attn.v_proj.weight.grad.detach().numpy().T,
          "o_proj": pt_attn.o_proj.weight.grad.detach().numpy().T,
          "q_norm": pt_attn.q_norm.weight.grad.detach().numpy(),
          "k_norm": pt_attn.k_norm.weight.grad.detach().numpy(),
          "sinks": pt_attn.softmax_offset.grad.detach().numpy(),
      }
      self.assertGreater(float(np.min(np.abs(pt_grads["sinks"]))), 1e-6)

      for fsdp_sz, cp_sz, ep_sz in meshes:
        mesh_cfg_ref = _make_test_config(
            fsdp_size=fsdp_sz,
            cp_size=cp_sz,
            ep_size=ep_sz,
            use_splash=False,
            compute_dtype="float32",
            param_dtype="float32",
            seq_len=seq_len,
            global_batch_size=gbs,
            hidden_size=hidden_size,
        )
        mesh = flint_sharding.create_mesh(mesh_cfg_ref.sharding)
        attn_pspecs = flint_sharding.param_pspecs(mesh_cfg_ref).decoder.prefix_dense[0].self_attn
        act_spec = flint_sharding.activation_pspec()
        tok_spec = flint_sharding.batch_pspec()

        w_sh = _shard_attn_weights(np_weights_f32, mesh, attn_pspecs)
        x_f32_sh = _shard_array(x_np, mesh, act_spec)
        pos_sh = _shard_array(pos_np, mesh, tok_spec)
        seg_sh = _shard_array(seg_np, mesh, tok_spec)
        up_sh = _shard_array(upstream_np, mesh, act_spec)

        # 1) FP32 dot-product reference (`use_splash=False`) vs PyTorch `< 1e-5`
        ref_step = _make_attn_fwd_and_bwd(mesh, mesh_cfg_ref, is_global=is_global)
        loss_ref, out_ref_sh, grads_ref_sh, dx_ref_sh = ref_step(w_sh, x_f32_sh, pos_sh, seg_sh, up_sh)
        out_ref = _gather_to_np(out_ref_sh)
        dx_ref = _gather_to_np(dx_ref_sh)
        grads_ref = jax.tree.map(_gather_to_np, grads_ref_sh)

        with self.subTest(
            mode="fp32_dot_ref",
            is_global=is_global,
            mesh=(fsdp_sz, cp_sz, ep_sz),
        ):
          self.assertLess(abs(float(_gather_to_np(loss_ref)) - loss_pt), 1e-6)
          self.assertLess(float(np.max(np.abs(out_ref - out_pt))), 1e-5)
          self.assertLess(float(np.max(np.abs(dx_ref - dx_pt))), 1e-5)
          self.assertEqual(grads_ref_sh.sinks.dtype, jnp.float32)
          for leaf_name, pt_g in pt_grads.items():
            jax_g = getattr(grads_ref, leaf_name)
            max_diff = float(np.max(np.abs(jax_g - pt_g)))
            self.assertLess(
                max_diff,
                1e-5,
                msg=(
                    f"FP32 dot-ref grad {leaf_name} diff={max_diff} >= 1e-5 on"
                    f" mesh={(fsdp_sz, cp_sz, ep_sz)}, {is_global=}"
                ),
            )

        # 2) FP32 SplashAttention (`use_splash=True`) vs dot-ref & PyTorch
        mesh_cfg_splash_f32 = _make_test_config(
            fsdp_size=fsdp_sz,
            cp_size=cp_sz,
            ep_size=ep_sz,
            use_splash=True,
            compute_dtype="float32",
            param_dtype="float32",
            seq_len=seq_len,
            global_batch_size=gbs,
            hidden_size=hidden_size,
        )
        splash_f32_step = _make_attn_fwd_and_bwd(mesh, mesh_cfg_splash_f32, is_global=is_global)
        loss_sp_f32, out_sp_f32_sh, grads_sp_f32_sh, dx_sp_f32_sh = splash_f32_step(w_sh, x_f32_sh, pos_sh, seg_sh, up_sh)
        out_sp_f32 = _gather_to_np(out_sp_f32_sh)
        dx_sp_f32 = _gather_to_np(dx_sp_f32_sh)
        grads_sp_f32 = jax.tree.map(_gather_to_np, grads_sp_f32_sh)

        with self.subTest(
            mode="fp32_splash",
            is_global=is_global,
            mesh=(fsdp_sz, cp_sz, ep_sz),
        ):
          self.assertEqual(grads_sp_f32_sh.sinks.dtype, jnp.float32)
          self.assertTrue(np.all(np.isfinite(grads_sp_f32.sinks)))
          self.assertGreater(float(np.linalg.norm(grads_sp_f32.sinks)), 1e-4)
          self.assertLess(abs(float(_gather_to_np(loss_sp_f32)) - loss_pt), 5e-4)
          self.assertGreater(_cosine_similarity(out_sp_f32, out_pt), 0.9999)
          self.assertGreater(_cosine_similarity(out_sp_f32, out_ref), 0.9999)
          self.assertGreater(_cosine_similarity(dx_sp_f32, dx_pt), 0.9999)
          for leaf_name, pt_g in pt_grads.items():
            jax_g = getattr(grads_sp_f32, leaf_name)
            ref_g = getattr(grads_ref, leaf_name)
            self.assertGreater(
                _cosine_similarity(jax_g, pt_g),
                0.9999,
                msg=f"FP32 splash {leaf_name} vs PT cosine too low",
            )
            self.assertGreater(
                _cosine_similarity(jax_g, ref_g),
                0.9999,
                msg=f"FP32 splash {leaf_name} vs dot-ref cosine too low",
            )
          self.assertLess(
              float(np.max(np.abs(grads_sp_f32.sinks - pt_grads["sinks"]))),
              5e-4,
          )

        # 3) BF16 SplashAttention (`compute_dtype='bfloat16'`, FP32 master)
        # Exercise both `use_base2_exp=False` and `use_base2_exp=True` (on CP=2)
        base2_modes = (False, True) if (fsdp_sz, cp_sz, ep_sz) == (4, 2, 1) else (False,)
        x_bf16_sh = jax.jit(
            lambda a: a.astype(jnp.bfloat16),
            out_shardings=NamedSharding(mesh, act_spec),
        )(x_f32_sh)
        for use_base2 in base2_modes:
          mesh_cfg_splash_bf16 = _make_test_config(
              fsdp_size=fsdp_sz,
              cp_size=cp_sz,
              ep_size=ep_sz,
              use_splash=True,
              compute_dtype="bfloat16",
              param_dtype="float32",
              seq_len=seq_len,
              global_batch_size=gbs,
              hidden_size=hidden_size,
              use_base2_exp=use_base2,
          )
          splash_bf16_step = _make_attn_fwd_and_bwd(mesh, mesh_cfg_splash_bf16, is_global=is_global)
          loss_sp_bf16, out_sp_bf16_sh, grads_sp_bf16_sh, dx_sp_bf16_sh = splash_bf16_step(
              w_sh, x_bf16_sh, pos_sh, seg_sh, up_sh
          )
          self.assertEqual(out_sp_bf16_sh.dtype, jnp.bfloat16)
          self.assertEqual(grads_sp_bf16_sh.sinks.dtype, jnp.float32)
          out_sp_bf16 = _gather_to_np(out_sp_bf16_sh.astype(jnp.float32))
          dx_sp_bf16 = _gather_to_np(dx_sp_bf16_sh.astype(jnp.float32))
          grads_sp_bf16 = jax.tree.map(_gather_to_np, grads_sp_bf16_sh)

          with self.subTest(
              mode="bf16_splash",
              use_base2=use_base2,
              is_global=is_global,
              mesh=(fsdp_sz, cp_sz, ep_sz),
          ):
            self.assertTrue(np.all(np.isfinite(grads_sp_bf16.sinks)))
            self.assertGreater(float(np.linalg.norm(grads_sp_bf16.sinks)), 1e-4)
            self.assertLess(abs(float(_gather_to_np(loss_sp_bf16)) - loss_pt), 5e-3)
            self.assertGreater(_cosine_similarity(out_sp_bf16, out_pt), 0.999)
            self.assertGreater(_cosine_similarity(out_sp_bf16, out_ref), 0.999)
            self.assertGreater(_cosine_similarity(dx_sp_bf16, dx_pt), 0.999)
            for leaf_name, pt_g in pt_grads.items():
              jax_g = getattr(grads_sp_bf16, leaf_name)
              self.assertGreater(
                  _cosine_similarity(jax_g, pt_g),
                  0.999,
                  msg=f"BF16 splash {leaf_name} vs PT cosine too low",
              )
            self.assertLess(
                float(np.max(np.abs(grads_sp_bf16.sinks - pt_grads["sinks"]))),
                2e-3,
            )

    # 4) Exact Sliding-Window (`W=513`) & Segment Boundary Causality Oracle
    # On `(fsdp=4, cp=2, ep=1)`, perturbing token `0` in segment `1` (`0..1023`
    # on even batch rows) MUST change `pos=512` (`512 - 0 = 512 < 513`) and MUST
    # produce bit-identical outputs on `pos=513..1023` AND `pos=1024..2047`.
    oracle_mesh = flint_sharding.create_mesh(flint_config.FlintShardingConfig(fsdp_size=4, cp_size=2, ep_size=1))
    act_spec = flint_sharding.activation_pspec()
    tok_spec = flint_sharding.batch_pspec()
    seg_two_doc = np.ones((gbs, seq_len), dtype=np.int32)
    seg_two_doc[:, 1024:] = 2
    pos_two_doc = np.tile(
        np.concatenate([np.arange(1024, dtype=np.int32), np.arange(1024, dtype=np.int32)])[None, :],
        (gbs, 1),
    )
    x_perturbed_np = x_np.copy()
    x_perturbed_np[:, 0, :] += 5.0

    for use_splash in (False, True):
      for compute_dtype in ("float32", "bfloat16"):
        cfg_local = _make_test_config(
            fsdp_size=4,
            cp_size=2,
            ep_size=1,
            use_splash=use_splash,
            compute_dtype=compute_dtype,
            param_dtype="float32",
            seq_len=seq_len,
            global_batch_size=gbs,
            hidden_size=hidden_size,
        )
        attn_pspecs = flint_sharding.param_pspecs(cfg_local).decoder.prefix_dense[0].self_attn
        w_sh = _shard_attn_weights(np_weights_f32, oracle_mesh, attn_pspecs)
        dt = np.float32 if compute_dtype == "float32" else jnp.bfloat16
        x0_sh = _shard_array(x_np.astype(dt), oracle_mesh, act_spec)
        x1_sh = _shard_array(x_perturbed_np.astype(dt), oracle_mesh, act_spec)
        pos_sh = _shard_array(pos_two_doc, oracle_mesh, tok_spec)
        seg_sh = _shard_array(seg_two_doc, oracle_mesh, tok_spec)

        fwd_fn = _make_attn_fwd_only(oracle_mesh, cfg_local, is_global=False)
        y0 = _gather_to_np(fwd_fn(w_sh, x0_sh, pos_sh, seg_sh).astype(jnp.float32))
        y1 = _gather_to_np(fwd_fn(w_sh, x1_sh, pos_sh, seg_sh).astype(jnp.float32))

        with self.subTest(
            oracle="sliding_window_513",
            use_splash=use_splash,
            compute_dtype=compute_dtype,
        ):
          # Inside window at boundary pos=512 (512 - 0 < 513): MUST change.
          self.assertGreater(
              float(np.max(np.abs(y1[:, 512, :] - y0[:, 512, :]))),
              1e-4,
          )
          # Outside window in same segment (513..1023): MUST be bit-identical.
          np.testing.assert_array_equal(y1[:, 513:1024, :], y0[:, 513:1024, :])
          # Across segment boundary (`1024..2047`): MUST be bit-identical!
          np.testing.assert_array_equal(y1[:, 1024:, :], y0[:, 1024:, :])

  def test_k5_fused_qkv_and_k6_global_splash_tiles_diag_skip_fwd_bwd(
      self,
  ) -> None:
    gbs = 8
    seq_len = 1024
    hidden_size = 512
    ref_cfg = _make_test_config(
        fsdp_size=8,
        cp_size=1,
        ep_size=1,
        use_splash=True,
        seq_len=seq_len,
        global_batch_size=gbs,
        hidden_size=hidden_size,
    )
    m = ref_cfg.model
    rng = np.random.default_rng(20261108)
    x_np = (rng.standard_normal((gbs, seq_len, hidden_size)) * 0.5 + 0.2).astype(np.float32)
    raw_up = rng.standard_normal((gbs, seq_len, hidden_size)).astype(np.float32)

    seg_np = np.ones((gbs, seq_len), dtype=np.int32)
    pos_np = np.zeros((gbs, seq_len), dtype=np.int32)
    for b_idx in range(gbs):
      split = 640 if b_idx % 2 == 0 else 512
      seg_np[b_idx, :split] = 1
      seg_np[b_idx, split:] = 2
      pos_np[b_idx, :split] = np.arange(split, dtype=np.int32)
      pos_np[b_idx, split:] = np.arange(seq_len - split, dtype=np.int32)

    q_dim = m.num_attention_heads * m.head_dim
    kv_dim = m.num_key_value_heads * m.head_dim
    np_weights_f32 = flint_types.FlintAttnWeights(
        q_proj=(rng.standard_normal((hidden_size, q_dim)) / np.sqrt(float(hidden_size))).astype(np.float32),
        k_proj=(rng.standard_normal((hidden_size, kv_dim)) / np.sqrt(float(hidden_size))).astype(np.float32),
        v_proj=(rng.standard_normal((hidden_size, kv_dim)) / np.sqrt(float(hidden_size)) * 0.5).astype(np.float32),
        o_proj=(rng.standard_normal((q_dim, hidden_size)) / np.sqrt(float(q_dim)) * 0.5).astype(np.float32),
        q_norm=np.linspace(1.5, 4.5, m.head_dim, dtype=np.float32),
        k_norm=np.linspace(0.7, 1.3, m.head_dim, dtype=np.float32),
        sinks=np.linspace(-1.2, 1.2, m.num_attention_heads, dtype=np.float32),
    )
    gqa = m.num_attention_heads // m.num_key_value_heads
    v_head_dir = np.repeat(
        (np.ones((1, hidden_size), dtype=np.float32) @ np_weights_f32.v_proj).reshape(m.num_key_value_heads, m.head_dim),
        gqa,
        axis=0,
    ).reshape(1, 1, q_dim)
    coherent_dir = v_head_dir @ np_weights_f32.o_proj
    coherent_dir = coherent_dir / np.linalg.norm(coherent_dir) * np.sqrt(float(hidden_size))
    upstream_np = (raw_up * 8.0 + coherent_dir * 32.0).astype(np.float32)

    leaf_names = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "sinks",
    )

    for fsdp_sz, cp_sz, ep_sz in ((8, 1, 1), (4, 2, 1)):
      mesh = flint_sharding.create_mesh(flint_config.FlintShardingConfig(fsdp_size=fsdp_sz, cp_size=cp_sz, ep_size=ep_sz))
      attn_pspecs = flint_sharding.param_pspecs(ref_cfg).decoder.prefix_dense[0].self_attn
      act_spec = flint_sharding.activation_pspec()
      tok_spec = flint_sharding.batch_pspec()
      w_sh = _shard_attn_weights(np_weights_f32, mesh, attn_pspecs)
      pos_sh = _shard_array(pos_np, mesh, tok_spec)
      seg_sh = _shard_array(seg_np, mesh, tok_spec)
      up_sh = _shard_array(upstream_np, mesh, act_spec)

      for compute_dtype in ("float32", "bfloat16"):
        dt = np.float32 if compute_dtype == "float32" else jnp.bfloat16
        x_sh = _shard_array(x_np.astype(dt), mesh, act_spec)
        min_cos = 0.9999 if compute_dtype == "float32" else 0.9995

        for is_global in (True, False):
          base_cfg = _make_test_config(
              fsdp_size=fsdp_sz,
              cp_size=cp_sz,
              ep_size=ep_sz,
              use_splash=True,
              compute_dtype=compute_dtype,
              param_dtype="float32",
              seq_len=seq_len,
              global_batch_size=gbs,
              hidden_size=hidden_size,
          )
          base_step = _make_attn_fwd_and_bwd(mesh, base_cfg, is_global=is_global)
          loss_ref, out_ref_sh, grads_ref_sh, dx_ref_sh = base_step(w_sh, x_sh, pos_sh, seg_sh, up_sh)
          out_ref = _gather_to_np(out_ref_sh.astype(jnp.float32))
          dx_ref = _gather_to_np(dx_ref_sh.astype(jnp.float32))
          grads_ref = jax.tree.map(_gather_to_np, grads_ref_sh)

          variants = [
              (
                  "k5_fused_qkv_dual_sc",
                  _make_test_config(
                      fsdp_size=fsdp_sz,
                      cp_size=cp_sz,
                      ep_size=ep_sz,
                      use_splash=True,
                      compute_dtype=compute_dtype,
                      param_dtype="float32",
                      seq_len=seq_len,
                      global_batch_size=gbs,
                      hidden_size=hidden_size,
                      fuse_layer_projections=True,
                      sc_collective_mode="dual_sc",
                  ),
              ),
          ]
          if is_global:
            variants.extend(
                [
                    (
                        "k6_qk_diag_skip_256",
                        _make_test_config(
                            fsdp_size=fsdp_sz,
                            cp_size=cp_sz,
                            ep_size=ep_sz,
                            use_splash=True,
                            compute_dtype=compute_dtype,
                            param_dtype="float32",
                            seq_len=seq_len,
                            global_batch_size=gbs,
                            hidden_size=hidden_size,
                            splash_global_block_size=256,
                            splash_qk_diag_skip=True,
                        ),
                    ),
                    (
                        "k6_sv_diag_skip_256",
                        _make_test_config(
                            fsdp_size=fsdp_sz,
                            cp_size=cp_sz,
                            ep_size=ep_sz,
                            use_splash=True,
                            compute_dtype=compute_dtype,
                            param_dtype="float32",
                            seq_len=seq_len,
                            global_batch_size=gbs,
                            hidden_size=hidden_size,
                            splash_global_block_size=256,
                            splash_sv_diag_skip=True,
                        ),
                    ),
                ]
            )

          for var_name, cand_cfg in variants:
            cand_step = _make_attn_fwd_and_bwd(mesh, cand_cfg, is_global=is_global)
            loss_got, out_got_sh, grads_got_sh, dx_got_sh = cand_step(w_sh, x_sh, pos_sh, seg_sh, up_sh)
            out_got = _gather_to_np(out_got_sh.astype(jnp.float32))
            dx_got = _gather_to_np(dx_got_sh.astype(jnp.float32))
            grads_got = jax.tree.map(_gather_to_np, grads_got_sh)
            with self.subTest(
                var=var_name,
                mesh=(fsdp_sz, cp_sz, ep_sz),
                dtype=compute_dtype,
                is_global=is_global,
            ):
              self.assertEqual(grads_got_sh.sinks.dtype, jnp.float32)
              self.assertTrue(np.isfinite(float(_gather_to_np(loss_got))))
              self.assertLess(
                  abs(float(_gather_to_np(loss_got)) - float(_gather_to_np(loss_ref))),
                  5e-3,
              )
              self.assertTrue(np.all(np.isfinite(out_got)))
              self.assertTrue(np.all(np.isfinite(dx_got)))
              self.assertGreaterEqual(_cosine_similarity(out_got, out_ref), min_cos)
              self.assertGreaterEqual(_cosine_similarity(dx_got, dx_ref), min_cos)
              for leaf_name in leaf_names:
                g_got = getattr(grads_got, leaf_name)
                g_ref = getattr(grads_ref, leaf_name)
                self.assertTrue(np.all(np.isfinite(g_got)))
                self.assertGreaterEqual(
                    _cosine_similarity(g_got, g_ref),
                    min_cos,
                    msg=f"{var_name} grad {leaf_name} cosine below {min_cos}",
                )

  def test_r3_dq_reduction1_sliding1024_dense_rhs_and_precomputed_rope(
      self,
  ) -> None:
    gbs = 8
    seq_len = 2048
    hidden_size = 512
    ref_cfg = _make_test_config(
        fsdp_size=8,
        cp_size=1,
        ep_size=1,
        use_splash=True,
        seq_len=seq_len,
        global_batch_size=gbs,
        hidden_size=hidden_size,
    )
    m = ref_cfg.model
    rng = np.random.default_rng(20261109)
    x_np = (rng.standard_normal((gbs, seq_len, hidden_size)) * 0.5 + 0.2).astype(np.float32)
    raw_up = rng.standard_normal((gbs, seq_len, hidden_size)).astype(np.float32)

    seg_np = np.ones((gbs, seq_len), dtype=np.int32)
    pos_np = np.zeros((gbs, seq_len), dtype=np.int32)
    for b_idx in range(gbs):
      split = 1280 if b_idx % 2 == 0 else 1024
      seg_np[b_idx, :split] = 1
      seg_np[b_idx, split:] = 2
      pos_np[b_idx, :split] = np.arange(split, dtype=np.int32)
      pos_np[b_idx, split:] = np.arange(seq_len - split, dtype=np.int32)

    q_dim = m.num_attention_heads * m.head_dim
    kv_dim = m.num_key_value_heads * m.head_dim
    np_weights_f32 = flint_types.FlintAttnWeights(
        q_proj=(rng.standard_normal((hidden_size, q_dim)) / np.sqrt(float(hidden_size))).astype(np.float32),
        k_proj=(rng.standard_normal((hidden_size, kv_dim)) / np.sqrt(float(hidden_size))).astype(np.float32),
        v_proj=(rng.standard_normal((hidden_size, kv_dim)) / np.sqrt(float(hidden_size)) * 0.5).astype(np.float32),
        o_proj=(rng.standard_normal((q_dim, hidden_size)) / np.sqrt(float(q_dim)) * 0.5).astype(np.float32),
        q_norm=np.linspace(1.5, 4.5, m.head_dim, dtype=np.float32),
        k_norm=np.linspace(0.7, 1.3, m.head_dim, dtype=np.float32),
        sinks=np.linspace(-1.2, 1.2, m.num_attention_heads, dtype=np.float32),
    )
    gqa = m.num_attention_heads // m.num_key_value_heads
    v_head_dir = np.repeat(
        (np.ones((1, hidden_size), dtype=np.float32) @ np_weights_f32.v_proj).reshape(m.num_key_value_heads, m.head_dim),
        gqa,
        axis=0,
    ).reshape(1, 1, q_dim)
    coherent_dir = v_head_dir @ np_weights_f32.o_proj
    coherent_dir = coherent_dir / np.linalg.norm(coherent_dir) * np.sqrt(float(hidden_size))
    upstream_np = (raw_up * 8.0 + coherent_dir * 32.0).astype(np.float32)
    leaf_names = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "sinks",
    )

    for fsdp_sz, cp_sz, ep_sz in ((8, 1, 1), (4, 2, 1)):
      mesh = flint_sharding.create_mesh(flint_config.FlintShardingConfig(fsdp_size=fsdp_sz, cp_size=cp_sz, ep_size=ep_sz))
      attn_pspecs = flint_sharding.param_pspecs(ref_cfg).decoder.prefix_dense[0].self_attn
      act_spec = flint_sharding.activation_pspec()
      tok_spec = flint_sharding.batch_pspec()
      w_sh = _shard_attn_weights(np_weights_f32, mesh, attn_pspecs)
      pos_sh = _shard_array(pos_np, mesh, tok_spec)
      seg_sh = _shard_array(seg_np, mesh, tok_spec)
      up_sh = _shard_array(upstream_np, mesh, act_spec)

      for compute_dtype in ("float32", "bfloat16"):
        dt = np.float32 if compute_dtype == "float32" else jnp.bfloat16
        x_sh = _shard_array(x_np.astype(dt), mesh, act_spec)
        min_cos = 0.9999 if compute_dtype == "float32" else 0.9995

        for is_global in (True, False):
          base_cfg = _make_test_config(
              fsdp_size=fsdp_sz,
              cp_size=cp_sz,
              ep_size=ep_sz,
              use_splash=True,
              compute_dtype=compute_dtype,
              param_dtype="float32",
              seq_len=seq_len,
              global_batch_size=gbs,
              hidden_size=hidden_size,
          )
          base_step = _make_attn_fwd_and_bwd(mesh, base_cfg, is_global=is_global)
          loss_ref, out_ref_sh, grads_ref_sh, dx_ref_sh = base_step(w_sh, x_sh, pos_sh, seg_sh, up_sh)
          out_ref = _gather_to_np(out_ref_sh.astype(jnp.float32))
          dx_ref = _gather_to_np(dx_ref_sh.astype(jnp.float32))
          grads_ref = jax.tree.map(_gather_to_np, grads_ref_sh)

          variants: list[tuple[str, flint_config.FlintConfig, bool]] = [
              (
                  "r3_transpose_dense_rhs_dual_sc",
                  _make_test_config(
                      fsdp_size=fsdp_sz,
                      cp_size=cp_sz,
                      ep_size=ep_sz,
                      use_splash=True,
                      compute_dtype=compute_dtype,
                      param_dtype="float32",
                      seq_len=seq_len,
                      global_batch_size=gbs,
                      hidden_size=hidden_size,
                      fuse_layer_projections=True,
                      sc_collective_mode="dual_sc",
                      transpose_dense_rhs_ag=True,
                  ),
                  False,
              ),
          ]
          if not is_global:
            variants.append(
                (
                    "r3_precompute_rope_tables_exact",
                    _make_test_config(
                        fsdp_size=fsdp_sz,
                        cp_size=cp_sz,
                        ep_size=ep_sz,
                        use_splash=True,
                        compute_dtype=compute_dtype,
                        param_dtype="float32",
                        seq_len=seq_len,
                        global_batch_size=gbs,
                        hidden_size=hidden_size,
                        precompute_rope_tables=True,
                    ),
                    True,
                )
            )
          if cp_sz == 1:
            if is_global:
              variants.append(
                  (
                      "r3_global_dq_red1_diag_skip_base2",
                      _make_test_config(
                          fsdp_size=fsdp_sz,
                          cp_size=cp_sz,
                          ep_size=ep_sz,
                          use_splash=True,
                          compute_dtype=compute_dtype,
                          param_dtype="float32",
                          seq_len=seq_len,
                          global_batch_size=gbs,
                          hidden_size=hidden_size,
                          splash_global_block_size=256,
                          splash_qk_diag_skip=True,
                          splash_sv_diag_skip=True,
                          use_base2_exp=True,
                          splash_dq_reduction_steps=1,
                          fuse_layer_projections=True,
                          sc_collective_mode="dual_sc",
                          transpose_dense_rhs_ag=True,
                      ),
                      False,
                  )
              )
            else:
              variants.append(
                  (
                      "r3_sliding_1024_dq_red1_rope",
                      _make_test_config(
                          fsdp_size=fsdp_sz,
                          cp_size=cp_sz,
                          ep_size=ep_sz,
                          use_splash=True,
                          compute_dtype=compute_dtype,
                          param_dtype="float32",
                          seq_len=seq_len,
                          global_batch_size=gbs,
                          hidden_size=hidden_size,
                          splash_sliding_block_size=1024,
                          use_base2_exp=True,
                          splash_dq_reduction_steps=1,
                          fuse_layer_projections=True,
                          sc_collective_mode="dual_sc",
                          transpose_dense_rhs_ag=True,
                          precompute_rope_tables=True,
                      ),
                      False,
                  )
              )

          for var_name, cand_cfg, require_exact in variants:
            cand_step = _make_attn_fwd_and_bwd(mesh, cand_cfg, is_global=is_global)
            loss_got, out_got_sh, grads_got_sh, dx_got_sh = cand_step(w_sh, x_sh, pos_sh, seg_sh, up_sh)
            out_got = _gather_to_np(out_got_sh.astype(jnp.float32))
            dx_got = _gather_to_np(dx_got_sh.astype(jnp.float32))
            grads_got = jax.tree.map(_gather_to_np, grads_got_sh)
            with self.subTest(
                var=var_name,
                mesh=(fsdp_sz, cp_sz, ep_sz),
                dtype=compute_dtype,
                is_global=is_global,
            ):
              self.assertEqual(grads_got_sh.sinks.dtype, jnp.float32)
              if require_exact:
                np.testing.assert_array_equal(out_got, out_ref)
                np.testing.assert_array_equal(dx_got, dx_ref)
                for leaf_name in leaf_names:
                  np.testing.assert_array_equal(
                      getattr(grads_got, leaf_name),
                      getattr(grads_ref, leaf_name),
                  )
              else:
                self.assertTrue(np.isfinite(float(_gather_to_np(loss_got))))
                self.assertLess(
                    abs(float(_gather_to_np(loss_got)) - float(_gather_to_np(loss_ref))),
                    5e-3,
                )
                self.assertTrue(np.all(np.isfinite(out_got)))
                self.assertTrue(np.all(np.isfinite(dx_got)))
                self.assertGreaterEqual(_cosine_similarity(out_got, out_ref), min_cos)
                self.assertGreaterEqual(_cosine_similarity(dx_got, dx_ref), min_cos)
                for leaf_name in leaf_names:
                  g_got = getattr(grads_got, leaf_name)
                  g_ref = getattr(grads_ref, leaf_name)
                  self.assertTrue(np.all(np.isfinite(g_got)))
                  self.assertGreaterEqual(
                      _cosine_similarity(g_got, g_ref),
                      min_cos,
                      msg=f"{var_name} grad {leaf_name} cosine below {min_cos}",
                  )

  def test_stacked_q_heads_asymmetric_tiles_and_local_halo_cp_parity(
      self,
  ) -> None:
    self.assertEqual(jax.default_backend(), "tpu")
    self.assertLen(jax.devices(), 8)

    rng = np.random.default_rng(2026)
    gbs, seq_len, hidden_size = 8, 2048, 512
    base_cfg = _make_test_config(
        fsdp_size=8,
        cp_size=1,
        ep_size=1,
        use_splash=False,
        compute_dtype="float32",
        param_dtype="float32",
        seq_len=seq_len,
        global_batch_size=gbs,
        hidden_size=hidden_size,
    )
    torch_attn = hf_reference.TorchFlintAttention(base_cfg.model, is_global=True)
    hf_reference.randomize_torch_flint(torch_attn, seed=909)
    with torch.no_grad():
      torch_attn.softmax_offset.copy_(
          torch.tensor(
              rng.uniform(-1.2, 1.2, size=(base_cfg.model.num_attention_heads,)),
              dtype=torch.float32,
          )
      )
    w_np = flint_types.FlintAttnWeights(
        q_proj=torch_attn.q_proj.weight.detach().cpu().numpy().T.astype(np.float32),
        k_proj=torch_attn.k_proj.weight.detach().cpu().numpy().T.astype(np.float32),
        v_proj=torch_attn.v_proj.weight.detach().cpu().numpy().T.astype(np.float32),
        o_proj=torch_attn.o_proj.weight.detach().cpu().numpy().T.astype(np.float32),
        q_norm=torch_attn.q_norm.weight.detach().cpu().numpy().astype(np.float32),
        k_norm=torch_attn.k_norm.weight.detach().cpu().numpy().astype(np.float32),
        sinks=torch_attn.softmax_offset.detach().cpu().numpy().astype(np.float32),
    )
    x_np = rng.standard_normal((gbs, seq_len, hidden_size)).astype(np.float32)
    up_np = rng.standard_normal((gbs, seq_len, hidden_size)).astype(np.float32)
    pos_np = np.broadcast_to(np.arange(seq_len, dtype=np.int32), (gbs, seq_len)).copy()
    # Multi-document packed segments crossing CP=2 (1024) and CP=4 (512) splits.
    seg_packed_np = np.zeros((gbs, seq_len), dtype=np.int32)
    seg_packed_np[:, 300:900] = 1
    seg_packed_np[:, 900:1400] = 2
    seg_packed_np[:, 1400:] = 3
    seg_cont_np = np.ones((gbs, seq_len), dtype=np.int32)

    leaf_names = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "sinks",
    )

    for fsdp_sz, cp_sz in ((4, 2), (2, 4)):
      for is_global in (True, False):
        for ignore_seg, seg_np in (
            (False, seg_packed_np),
            (True, seg_cont_np),
        ):
          for compute_dtype, min_cos in (
              ("float32", 0.999),
              ("bfloat16", 0.995),
          ):
            ref_cfg = _make_test_config(
                fsdp_size=fsdp_sz,
                cp_size=cp_sz,
                ep_size=1,
                use_splash=False,
                compute_dtype=compute_dtype,
                param_dtype="float32",
                seq_len=seq_len,
                global_batch_size=gbs,
                hidden_size=hidden_size,
            )
            local_q = seq_len // (2 * cp_sz)
            opt_kernel_cfg = flint_config.FlintKernelConfig(
                use_splash=True,
                splash_block_q=512,
                splash_block_kv=512,
                splash_block_kv_compute=256,
                splash_block_q_dkv=512,
                splash_block_kv_dkv=512,
                splash_block_kv_dkv_compute=512,
                splash_dq_reduction_steps=3,
                splash_local_dq_reduction_steps=1,
                splash_num_stacked_q_heads=8,
                splash_num_stacked_q_heads_dkv=8,
                splash_use_scheduler=True,
                splash_use_scheduler_dkv=False,
                splash_dkv_reuse_scaled_q=True,
                splash_use_local_halo_cp=True,
                splash_ignore_segment_ids=ignore_seg,
                splash_global_block_q=min(512, local_q),
                splash_global_block_kv=1024,
                splash_global_block_kv_compute=512,
                splash_global_block_q_dkv=min(256, local_q),
                splash_global_block_kv_dkv=2048,
                splash_global_block_kv_dkv_compute=512,
                splash_global_num_stacked_q_heads=8,
                splash_global_num_stacked_q_heads_dkv=4,
                splash_global_use_scheduler=True,
                splash_global_use_scheduler_dkv=False,
                splash_global_dkv_reuse_scaled_q=True,
                splash_use_base2_exp=True,
                splash_fuse_reciprocal=True,
                num_vocab_chunks=5,
                fuse_layer_projections=True,
                transpose_dense_rhs_ag=True,
                precompute_rope_tables=True,
            )
            cand_cfg = flint_config.FlintConfig(
                model=ref_cfg.model,
                sharding=ref_cfg.sharding,
                kernels=opt_kernel_cfg,
                train=ref_cfg.train,
            )
            mesh = flint_sharding.create_mesh(ref_cfg.sharding)
            attn_pspecs = flint_sharding.param_pspecs(ref_cfg).decoder.prefix_dense[0].self_attn
            act_spec = flint_sharding.activation_pspec()
            tok_spec = flint_sharding.batch_pspec()
            w_sh = _shard_attn_weights(w_np, mesh, attn_pspecs)
            x_sh = _shard_array(x_np, mesh, act_spec)
            pos_sh = _shard_array(pos_np, mesh, tok_spec)
            seg_sh = _shard_array(seg_np, mesh, tok_spec)
            up_sh = _shard_array(up_np, mesh, act_spec)

            ref_step = _make_attn_fwd_and_bwd(mesh, ref_cfg, is_global=is_global)
            cand_step = _make_attn_fwd_and_bwd(mesh, cand_cfg, is_global=is_global)
            if not is_global and cp_sz == 4 and ignore_seg and compute_dtype == "bfloat16":
              halo_hlo = cand_step.lower(w_sh, x_sh, pos_sh, seg_sh, up_sh).as_text("hlo")
              self.assertIn("collective-permute", halo_hlo)

            loss_ref, out_ref_sh, grads_ref_sh, dx_ref_sh = ref_step(w_sh, x_sh, pos_sh, seg_sh, up_sh)
            loss_got, out_got_sh, grads_got_sh, dx_got_sh = cand_step(w_sh, x_sh, pos_sh, seg_sh, up_sh)
            out_ref = _gather_to_np(out_ref_sh.astype(jnp.float32))
            dx_ref = _gather_to_np(dx_ref_sh.astype(jnp.float32))
            grads_ref = jax.tree.map(_gather_to_np, grads_ref_sh)
            out_got = _gather_to_np(out_got_sh.astype(jnp.float32))
            dx_got = _gather_to_np(dx_got_sh.astype(jnp.float32))
            grads_got = jax.tree.map(_gather_to_np, grads_got_sh)

            with self.subTest(
                cp_sz=cp_sz,
                is_global=is_global,
                ignore_seg=ignore_seg,
                dtype=compute_dtype,
            ):
              self.assertTrue(np.isfinite(float(_gather_to_np(loss_got))))
              self.assertLess(
                  abs(float(_gather_to_np(loss_got)) - float(_gather_to_np(loss_ref))),
                  5e-3,
              )
              self.assertGreaterEqual(_cosine_similarity(out_got, out_ref), min_cos)
              self.assertGreaterEqual(_cosine_similarity(dx_got, dx_ref), min_cos)
              for leaf_name in leaf_names:
                g_got = getattr(grads_got, leaf_name)
                g_ref = getattr(grads_ref, leaf_name)
                self.assertTrue(np.all(np.isfinite(g_got)))
                self.assertGreaterEqual(
                    _cosine_similarity(g_got, g_ref),
                    min_cos,
                    msg=(f"cp={cp_sz} global={is_global} ign={ignore_seg}" f" dtype={compute_dtype} grad {leaf_name}"),
                )

  def test_v4_prologue_lb_reorder_and_head_stitch_parity(self):
    rng = np.random.default_rng(2028)

    # 1) Direct Pallas VMEM QK prologue fwd+bwd parity (RoPE and NoPE).
    x_loc_np = rng.standard_normal((1, 1024, 8, 128)).astype(np.float32)
    dy_loc_np = rng.standard_normal((1, 8, 1024, 128)).astype(np.float32)
    sc_np = (1.0 + 0.1 * rng.standard_normal((128,))).astype(np.float32)
    ang = rng.uniform(-1.5, 1.5, size=(1, 1024, 64)).astype(np.float32)
    ang_full = np.concatenate([ang, ang], axis=-1)
    cos_np = np.cos(ang_full).astype(np.float32)
    sin_np = np.sin(ang_full).astype(np.float32)

    x_j = jnp.asarray(x_loc_np, dtype=jnp.bfloat16)
    dy_j = jnp.asarray(dy_loc_np, dtype=jnp.bfloat16)
    sc_j = jnp.asarray(sc_np, dtype=jnp.float32)
    cos_j = jnp.asarray(cos_np, dtype=jnp.bfloat16)
    sin_j = jnp.asarray(sin_np, dtype=jnp.bfloat16)
    q_scalar = float(128**-0.5)

    for use_rope in (False, True):
      c_arg = cos_j if use_rope else None
      s_arg = sin_j if use_rope else None

      @jax.jit
      def _pro_vg(x_in, s_in, dy_in, c_in=c_arg, sn_in=s_arg):
        y, vjp_f = jax.vjp(
            lambda xi, si: flint_attention_prologue.fused_qk_prologue_local(
                xi,
                si,
                epsilon=1e-6,
                pre_attn_scalar=q_scalar,
                cos_2d=c_in,
                sin_2d=sn_in,
            ),
            x_in,
            s_in,
        )
        dx, ds = vjp_f(dy_in)
        return y, dx, ds

      pro_hlo = _pro_vg.lower(x_j, sc_j, dy_j).as_text("hlo", debug_info=True)
      self.assertIn("pallas_qk_prologue", pro_hlo)
      y_got, dx_got, ds_got = _pro_vg(x_j, sc_j, dy_j)
      y_ref, vjp_ref = jax.vjp(
          lambda xi, si, c_in=c_arg, sn_in=s_arg: (
              flint_attention_prologue._ref_qk_prologue_local(  # pylint: disable=protected-access
                  xi,
                  si,
                  epsilon=1e-6,
                  pre_attn_scalar=q_scalar,
                  cos_2d=c_in,
                  sin_2d=sn_in,
              )
          ),
          x_j,
          sc_j,
      )
      dx_ref, ds_ref = vjp_ref(dy_j)
      self.assertGreaterEqual(
          _cosine_similarity(np.asarray(y_got, np.float32), np.asarray(y_ref, np.float32)),
          0.9999,
      )
      self.assertGreaterEqual(
          _cosine_similarity(np.asarray(dx_got, np.float32), np.asarray(dx_ref, np.float32)),
          0.9995,
      )
      self.assertGreaterEqual(
          _cosine_similarity(np.asarray(ds_got, np.float32), np.asarray(ds_ref, np.float32)),
          0.9995,
      )

    # 2) Direct Pallas LB KV reorder bit-exact fwd+bwd parity.
    kv_j = jnp.asarray(
        rng.standard_normal((1, 4, 4096, 128)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    dkv_j = jnp.asarray(
        rng.standard_normal((1, 4, 4096, 128)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    for cp_sz in (2, 4):

      @jax.jit
      def _reord_vg(x_in, dy_in, c=cp_sz):
        y, vjp_f = jax.vjp(
            lambda xi: flint_attention_prologue.reorder_lb_kv_to_contiguous(xi, c, tile_s=512),
            x_in,
        )
        return y, vjp_f(dy_in)[0]

      reord_hlo = _reord_vg.lower(kv_j, dkv_j).as_text("hlo", debug_info=True)
      self.assertIn("lb_kv_vmem_copy", reord_hlo)
      y_r, dx_r = _reord_vg(kv_j, dkv_j)
      y_r_ref = flint_attention_prologue._ref_permute_lb_chunks(kv_j, cp_sz, True)  # pylint: disable=protected-access
      dx_r_ref = flint_attention_prologue._ref_permute_lb_chunks(dkv_j, cp_sz, False)  # pylint: disable=protected-access
      np.testing.assert_array_equal(np.asarray(y_r), np.asarray(y_r_ref))
      np.testing.assert_array_equal(np.asarray(dx_r), np.asarray(dx_r_ref))

    # 3) End-to-end 2-stripe LB CP + head-stitch + early KV + prologue parity.
    hidden_size = 512
    seq_len = 4096
    cp_sz = 4
    fsdp_sz = 2
    gbs = 2
    ref_base = _make_test_config(
        fsdp_size=fsdp_sz,
        cp_size=cp_sz,
        ep_size=1,
        use_splash=False,
        compute_dtype="bfloat16",
        param_dtype="float32",
        seq_len=seq_len,
        global_batch_size=gbs,
        hidden_size=hidden_size,
    )
    m_cfg = ref_base.model
    q_dim = m_cfg.num_attention_heads * m_cfg.head_dim
    kv_dim = m_cfg.num_key_value_heads * m_cfg.head_dim
    w_np = flint_types.FlintAttnWeights(
        q_proj=(rng.standard_normal((hidden_size, q_dim)) / np.sqrt(float(hidden_size))).astype(np.float32),
        k_proj=(rng.standard_normal((hidden_size, kv_dim)) / np.sqrt(float(hidden_size))).astype(np.float32),
        v_proj=(rng.standard_normal((hidden_size, kv_dim)) / np.sqrt(float(hidden_size)) * 0.5).astype(np.float32),
        o_proj=(rng.standard_normal((q_dim, hidden_size)) / np.sqrt(float(q_dim)) * 0.5).astype(np.float32),
        q_norm=np.linspace(1.5, 4.5, m_cfg.head_dim, dtype=np.float32),
        k_norm=np.linspace(0.7, 1.3, m_cfg.head_dim, dtype=np.float32),
        sinks=np.linspace(-1.2, 1.2, m_cfg.num_attention_heads, dtype=np.float32),
    )
    x_np = rng.standard_normal((gbs, seq_len, hidden_size)).astype(np.float32)
    up_np = rng.standard_normal((gbs, seq_len, hidden_size)).astype(np.float32)
    pos_np = np.zeros((gbs, seq_len), dtype=np.int32)
    seg_np = np.ones((gbs, seq_len), dtype=np.int32)
    for b in range(gbs):
      c1, c2 = 1100 + b * 64, 2700 + b * 64
      pos_np[b, :c1] = np.arange(c1, dtype=np.int32)
      pos_np[b, c1:c2] = np.arange(c2 - c1, dtype=np.int32)
      pos_np[b, c2:] = np.arange(seq_len - c2, dtype=np.int32)
      seg_np[b, c1:c2] = 2
      seg_np[b, c2:] = 3

    mesh = flint_sharding.create_mesh(ref_base.sharding)
    attn_pspecs = flint_sharding.param_pspecs(ref_base).decoder.prefix_dense[0].self_attn
    act_spec = flint_sharding.activation_pspec()
    tok_spec = flint_sharding.batch_pspec()
    w_sh = _shard_attn_weights(w_np, mesh, attn_pspecs)
    x_sh = _shard_array(x_np, mesh, act_spec)
    pos_sh = _shard_array(pos_np, mesh, tok_spec)
    seg_sh = _shard_array(seg_np, mesh, tok_spec)
    up_sh = _shard_array(up_np, mesh, act_spec)

    def _make_lb_attn_step(cfg_lb: flint_config.FlintConfig, is_glob: bool):
      def _local_fn(w_l, x_nat, pos_nat, seg_nat, up_nat):
        x_lb = ops.cp_natural_to_lb(x_nat, cp_sz, axis=1)
        pos_lb = ops.cp_natural_to_lb(pos_nat, cp_sz, axis=1)
        seg_lb = ops.cp_natural_to_lb(seg_nat, cp_sz, axis=1)
        up_lb = ops.cp_natural_to_lb(up_nat, cp_sz, axis=1)
        if is_glob:
          rope_t = None
        else:
          rope_t = ops.compute_split_half_rope_cos_sin(
              pos_lb,
              cfg_lb.model.head_dim,
              rope_theta=cfg_lb.model.rope_theta,
              dtype=jnp.bfloat16,
          )

        def _loss(wi, xi):
          yi = flint_attention.flint_attention_forward(
              xi,
              wi,
              pos_lb,
              seg_lb,
              cfg_lb,
              is_global=is_glob,
              rope_cos_sin=rope_t,
          )
          ls = jnp.sum(yi.astype(jnp.float32) * up_lb.astype(jnp.float32)) / float(gbs * seq_len * hidden_size)
          return ls, yi

        (l_loc, y_lb), (gw_l, dx_lb) = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(w_l, x_lb)
        y_nat = ops.cp_lb_to_natural(y_lb, cp_sz, axis=1)
        dx_nat = ops.cp_lb_to_natural(dx_lb, cp_sz, axis=1)
        gw_s = jax.tree.map(
            lambda g, sp: (jax.lax.psum(g, ("fsdp", "cp", "ep")) if sp == P() else g),
            gw_l,
            attn_pspecs,
        )
        return jax.lax.psum(l_loc, ("fsdp", "cp", "ep")), y_nat, gw_s, dx_nat

      return jax.jit(
          jax.shard_map(
              _local_fn,
              mesh=mesh,
              in_specs=(attn_pspecs, act_spec, tok_spec, tok_spec, act_spec),
              out_specs=(P(), act_spec, attn_pspecs, act_spec),
              check_vma=False,
          )
      )

    for is_global, ignore_seg in (
        (False, True),
        (True, True),
        (True, False),
    ):
      if ignore_seg:
        pos_in = _shard_array(
            np.broadcast_to(np.arange(seq_len, dtype=np.int32), (gbs, seq_len)).copy(),
            mesh,
            tok_spec,
        )
        seg_in = _shard_array(np.ones((gbs, seq_len), dtype=np.int32), mesh, tok_spec)
      else:
        pos_in = pos_sh
        seg_in = seg_sh
      v4_k = flint_config.FlintKernelConfig(
          use_splash=True,
          splash_block_q=512,
          splash_block_kv=512,
          splash_block_kv_compute=256,
          splash_block_q_dkv=512,
          splash_block_kv_dkv=512,
          splash_block_kv_dkv_compute=512,
          splash_dq_reduction_steps=3,
          splash_local_dq_reduction_steps=1,
          splash_num_stacked_q_heads=8,
          splash_num_stacked_q_heads_dkv=8,
          splash_use_scheduler=True,
          splash_use_scheduler_dkv=False,
          splash_dkv_reuse_scaled_q=True,
          splash_use_local_halo_cp=True,
          splash_ignore_segment_ids=ignore_seg,
          splash_global_block_q=512,
          splash_global_block_kv=1024,
          splash_global_block_kv_compute=512,
          splash_global_block_q_dkv=256,
          splash_global_block_kv_dkv=2048,
          splash_global_block_kv_dkv_compute=512,
          splash_global_num_stacked_q_heads=8,
          splash_global_num_stacked_q_heads_dkv=4,
          splash_global_use_scheduler=True,
          splash_global_use_scheduler_dkv=False,
          splash_global_dkv_reuse_scaled_q=True,
          splash_fwd_strip_schedule="auto",
          splash_compact_lse_residual=True,
          splash_global_dkv_head_stitch=True,
          splash_use_base2_exp=True,
          splash_fuse_reciprocal=True,
          num_vocab_chunks=1,
          fuse_layer_projections=True,
          transpose_dense_rhs_ag=True,
          precompute_rope_tables=True,
          use_lb_cp_layout=True,
          early_global_kv_proj=True,
          use_pallas_lb_kv_reorder=True,
          use_fused_qk_prologue=True,
      )
      v4_cfg = flint_config.FlintConfig(
          model=ref_base.model,
          sharding=ref_base.sharding,
          kernels=v4_k,
          train=ref_base.train,
      )
      ref_step = _make_attn_fwd_and_bwd(mesh, ref_base, is_global=is_global)
      cand_step = _make_lb_attn_step(v4_cfg, is_global)
      hlo_txt = cand_step.lower(w_sh, x_sh, pos_in, seg_in, up_sh).as_text("hlo", debug_info=True)
      self.assertIn("pallas_qk_prologue", hlo_txt)
      if is_global:
        self.assertIn("lb_kv_vmem_copy", hlo_txt)
        if ignore_seg:
          self.assertIn("splash_mha_dkv_no_residuals_head_stitch", hlo_txt)
        else:
          self.assertNotIn("splash_mha_dkv_no_residuals_head_stitch", hlo_txt)

      loss_ref, out_ref_sh, grads_ref_sh, dx_ref_sh = ref_step(w_sh, x_sh, pos_in, seg_in, up_sh)
      loss_got, out_got_sh, grads_got_sh, dx_got_sh = cand_step(w_sh, x_sh, pos_in, seg_in, up_sh)
      out_ref = _gather_to_np(out_ref_sh.astype(jnp.float32))
      out_got = _gather_to_np(out_got_sh.astype(jnp.float32))
      dx_ref = _gather_to_np(dx_ref_sh.astype(jnp.float32))
      dx_got = _gather_to_np(dx_got_sh.astype(jnp.float32))
      grads_ref = jax.tree.map(_gather_to_np, grads_ref_sh)
      grads_got = jax.tree.map(_gather_to_np, grads_got_sh)
      with self.subTest(is_global=is_global, ignore_seg=ignore_seg):
        self.assertLess(
            abs(float(_gather_to_np(loss_got)) - float(_gather_to_np(loss_ref))),
            5e-3,
        )
        self.assertGreaterEqual(_cosine_similarity(out_got, out_ref), 0.999)
        self.assertGreaterEqual(_cosine_similarity(dx_got, dx_ref), 0.999)
        for g_got, g_ref in zip(jax.tree.leaves(grads_got), jax.tree.leaves(grads_ref), strict=True):
          self.assertGreaterEqual(_cosine_similarity(g_got, g_ref), 0.999)


if __name__ == "__main__":
  absltest.main()
