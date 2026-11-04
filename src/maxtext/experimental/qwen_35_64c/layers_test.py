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

"""Numerical parity & optimizer step unit tests for pure-JAX Qwen3.5 on 8-device CPU."""

from __future__ import annotations

import dataclasses
import os
from absl.testing import absltest  # pylint: disable=g-import-not-at-top

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # pylint: disable=g-import-not-at-top

jax.config.update("jax_num_cpu_devices", 8)
import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
from jax.sharding import NamedSharding, PartitionSpec as P  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top

from maxtext.experimental import qwen_35_64c as qwen35_jax  # pylint: disable=g-import-not-at-top


def _equalize_expert_amax(w: jax.Array) -> jax.Array:
  sb_max = jnp.max(jnp.abs(w), axis=(1, 2, 3), keepdims=True)
  exp_max = jnp.max(jnp.abs(w), axis=(2, 3), keepdims=True)
  return (w * (sb_max / jnp.maximum(exp_max, 1e-12))).astype(w.dtype)


def _make_synthetic_batch(cfg: qwen35_jax.Qwen35Config, mesh):
  global_b = cfg.global_batch_size
  key = jax.random.PRNGKey(123)
  k1, k2 = jax.random.split(key)
  tokens = jax.random.randint(k1, (global_b, cfg.seq_len), 0, cfg.vocab_size, dtype=jnp.int32)
  targets = jax.random.randint(k2, (global_b, cfg.seq_len), 0, cfg.vocab_size, dtype=jnp.int32)
  # Packed 2-segment sequence per row to exercise segment masking in GDN & Attention!
  half_s = cfg.seq_len // 2
  pos_row = jnp.concatenate([jnp.arange(half_s), jnp.arange(cfg.seq_len - half_s)])
  positions = jnp.broadcast_to(pos_row[None, :], (global_b, cfg.seq_len)).astype(jnp.int32)
  seg_row = jnp.concatenate(
      [
          jnp.ones((half_s,), dtype=jnp.int32),
          jnp.full((cfg.seq_len - half_s - 2,), 2, dtype=jnp.int32),
          jnp.zeros((2,), dtype=jnp.int32),
      ]
  )
  segment_ids = jnp.broadcast_to(seg_row[None, :], (global_b, cfg.seq_len)).astype(jnp.int32)

  data_sharding = NamedSharding(mesh, qwen35_jax.data_pspec(cfg))
  return tuple(jax.device_put(arr, data_sharding) for arr in (tokens, targets, positions, segment_ids))


class Qwen38PureJaxTest(absltest.TestCase):

  def test_unsplit_vs_batchsplit_fp32_parity(self):
    cfg_unsplit = qwen35_jax.tiny_config(
        schedule_mode="unsplit",
        mu_dtype="float32",
        bwd_tie_mode="all4",
        global_batch_size=16,
    )
    cfg_bs = qwen35_jax.tiny_config(
        schedule_mode="batchsplit",
        mu_dtype="float32",
        bwd_tie_mode="all4",
        global_batch_size=16,
    )
    mesh = qwen35_jax.create_mesh(cfg_unsplit, aot_topology=None)
    params = qwen35_jax.init_random_params(cfg_unsplit, mesh, jax.random.PRNGKey(42), param_dtype=jnp.float32)
    batch = _make_synthetic_batch(cfg_unsplit, mesh)

    loss_u, grads_u = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_unsplit, mesh))(params, *batch)
    loss_b, grads_b = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_bs, mesh))(params, *batch)

    # Routed MoE quantizes EP tokens to FP8 with one scale per microbatch, so the two
    # schedules see different FP8 rounding. The forward loss agrees to that noise, but
    # the e4m3/e5m2-quantized backward amplifies it through 8 layers into O(1) gradient
    # differences on this random tiny model, so only the loss and finiteness are checked.
    loss_rel = float(jnp.abs(loss_u - loss_b) / jnp.abs(loss_u))
    self.assertLess(
        loss_rel,
        5e-3,
        f"Loss mismatch: unsplit={loss_u} vs batchsplit={loss_b}",
    )
    for path, g in jax.tree.leaves_with_path((grads_u, grads_b)):
      self.assertTrue(
          bool(jnp.all(jnp.isfinite(g))),
          f"Non-finite grad on {jax.tree_util.keystr(path)}",
      )

  def test_cp_load_balanced_layout_permutes(self):
    from maxtext.experimental.qwen_35_64c import layers as layers_lib  # pylint: disable=g-import-not-at-top

    for ov in (dict(decouple_ep_d2d=False), dict(decouple_ep_d2d=True)):
      with self.subTest(**ov):
        cfg = qwen35_jax.tiny_config(**ov)
        mesh = qwen35_jax.create_mesh(cfg, aot_topology=None)
        pspec = qwen35_jax.data_pspec(cfg)
        b, s = cfg.global_batch_size, cfg.seq_len
        idx = jnp.broadcast_to(jnp.arange(s, dtype=jnp.int32)[None, :], (b, s))
        idx = jax.device_put(idx, NamedSharding(mesh, pspec))

        def _local(x):
          lb = layers_lib.cp_natural_to_lb(x, cfg)
          expected_lb = layers_lib.cp_lb_global_index(x.shape[1], cfg)[None, :]
          back = layers_lib.cp_lb_to_natural(lb, cfg)
          full = layers_lib.cp_all_gather_lb_to_natural(lb, cfg)
          err = (
              jnp.sum(jnp.abs(lb - expected_lb))
              + jnp.sum(jnp.abs(back - x))
              + jnp.sum(jnp.abs(full - jnp.arange(s, dtype=jnp.int32)[None, :]))
          )
          return jax.lax.psum(err, cfg.all_sharded_axes)

        err = jax.jit(
            jax.shard_map(
                _local,
                mesh=mesh,
                in_specs=(pspec,),
                out_specs=P(),
                check_vma=False,
            )
        )(idx)
        self.assertEqual(int(err), 0)

  def test_context_parallel_matches_single_rank_sequence(self):
    """CP=4 (sequence split over 'expert') matches CP=1 (full sequence per device)."""
    from maxtext.experimental.qwen_35_64c import layers as layers_lib  # pylint: disable=g-import-not-at-top
    from maxtext.experimental.qwen_35_64c import moe_kernels as mk_lib  # pylint: disable=g-import-not-at-top
    from maxtext.experimental.qwen_35_64c import ops as ops_lib  # pylint: disable=g-import-not-at-top

    common = dict(schedule_mode="batchsplit", mu_dtype="float32", global_batch_size=16)
    cfg_cp = qwen35_jax.tiny_config(**common)
    cfg_ref = qwen35_jax.tiny_config(fsdp_size=8, ep_size=1, **common)
    self.assertEqual(cfg_cp.cp_size, 4)
    self.assertEqual(cfg_ref.cp_size, 1)
    mesh_cp = qwen35_jax.create_mesh(cfg_cp, aot_topology=None)
    mesh_ref = qwen35_jax.create_mesh(cfg_ref, aot_topology=None)
    raw_params = qwen35_jax.init_random_params(cfg_cp, mesh_cp, jax.random.PRNGKey(5), param_dtype=jnp.float32)
    batch_cp = _make_synthetic_batch(cfg_cp, mesh_cp)
    batch_ref = _make_synthetic_batch(cfg_ref, mesh_ref)

    # 1. Exact 8-layer CP=4 vs CP=1 parity across all CP sequence mixers (6 CP GDN
    #    scan/conv1d layers + 2 CP dual-chunk load-balanced causal attention/RoPE
    #    layers + 8 shared MLPs + embed + chunked LM head, with routed wi=wo=0 so
    #    discontinuous e4m3fn rounding does not butterfly-amplify 1e-6 reassociations).
    blocks_dense_only = dataclasses.replace(
        raw_params.blocks,
        **{
            sub_name: dataclasses.replace(
                getattr(raw_params.blocks, sub_name),
                moe=dataclasses.replace(
                    getattr(raw_params.blocks, sub_name).moe,
                    wi=jnp.zeros_like(getattr(raw_params.blocks, sub_name).moe.wi),
                    wo=jnp.zeros_like(getattr(raw_params.blocks, sub_name).moe.wo),
                ),
            )
            for sub_name in ("gdn_0", "gdn_1", "gdn_2", "attn_3")
        },
    )
    params_dense_np = jax.tree.map(np.asarray, dataclasses.replace(raw_params, blocks=blocks_dense_only))
    p_dense_cp = jax.tree.map(
        lambda arr, ps: jax.device_put(arr, NamedSharding(mesh_cp, ps)),
        params_dense_np,
        qwen35_jax.get_param_pspecs(cfg_cp),
    )
    p_dense_ref = jax.tree.map(
        lambda arr, ps: jax.device_put(arr, NamedSharding(mesh_ref, ps)),
        params_dense_np,
        qwen35_jax.get_param_pspecs(cfg_ref),
    )
    loss_d_cp, grads_d_cp = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_cp, mesh_cp))(p_dense_cp, *batch_cp)
    loss_d_ref, grads_d_ref = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_ref, mesh_ref))(p_dense_ref, *batch_ref)
    self.assertLess(
        abs(float(loss_d_cp) - float(loss_d_ref)),
        1e-4,
        f"{loss_d_cp} vs {loss_d_ref}",
    )
    for (path, gc), (_, gr) in zip(
        jax.tree.leaves_with_path(grads_d_cp),
        jax.tree.leaves_with_path(grads_d_ref),
    ):
      gc_np = np.asarray(gc, dtype=np.float64)
      gr_np = np.asarray(gr, dtype=np.float64)
      rel_l2 = float(np.linalg.norm(gc_np - gr_np) / (np.linalg.norm(gr_np) + 1e-12))
      self.assertLess(
          rel_l2,
          5e-3,
          f"CP mixer grad mismatch on {jax.tree_util.keystr(path)}: {rel_l2}",
      )

    # 2. Exact isolated CP=4 vs CP=1 parity of a no-drop CF=0.5 FP8 routed MoE sublayer
    #    on identical input activations (equalizing per-expert wi/wo absmax and tiling 2
    #    distinct sequences across FSDP shards so FP8 scales match bit-for-bit).
    blocks_eq = dataclasses.replace(
        raw_params.blocks,
        **{
            sub_name: dataclasses.replace(
                getattr(raw_params.blocks, sub_name),
                moe=dataclasses.replace(
                    getattr(raw_params.blocks, sub_name).moe,
                    wi=_equalize_expert_amax(getattr(raw_params.blocks, sub_name).moe.wi),
                    wo=_equalize_expert_amax(getattr(raw_params.blocks, sub_name).moe.wo),
                ),
            )
            for sub_name in ("gdn_0", "gdn_1", "gdn_2", "attn_3")
        },
    )
    params_eq_np = jax.tree.map(np.asarray, dataclasses.replace(raw_params, blocks=blocks_eq))
    kx, kc = jax.random.split(jax.random.PRNGKey(99))
    x_2 = np.asarray(jax.random.normal(kx, (2, cfg_cp.seq_len, cfg_cp.emb_dim), dtype=jnp.float32))
    ct_2 = np.asarray(jax.random.normal(kc, (2, cfg_cp.seq_len, cfg_cp.emb_dim), dtype=jnp.float32))
    x_tiled = np.tile(x_2, (cfg_cp.global_batch_size // 2, 1, 1))
    ct_tiled = np.tile(ct_2, (cfg_cp.global_batch_size // 2, 1, 1))

    def _run_cp_moe(cfg, mesh):
      p_sharded = jax.tree.map(
          lambda arr, ps: jax.device_put(arr, NamedSharding(mesh, ps)),
          params_eq_np,
          qwen35_jax.get_param_pspecs(cfg),
      )
      act_pspec = P("fsdp", cfg.cp_axis_names, None)
      x_s = jax.device_put(x_tiled, NamedSharding(mesh, act_pspec))
      ct_s = jax.device_put(ct_tiled, NamedSharding(mesh, act_pspec))

      def _body(p_l, x_l, ct_l):
        sb_amax = qwen35_jax.model._compute_hoisted_superblock_absmax(p_l.blocks, cfg)
        sub0 = jax.tree.map(lambda z: z[0], p_l.blocks.gdn_0)
        rg_f = ops_lib.gather_or_replicate_dense_weight(
            sub0.moe.router_gate,
            cfg,
            param_name="router_gate",
            sharded_axis=0,
        )
        wi_f = mk_lib.gather_routed_weight_sc1(sub0.moe.wi, sb_amax.gdn_0.wi[0], cfg, is_wi=True)
        wo_f = mk_lib.gather_routed_weight_sc1(sub0.moe.wo, sb_amax.gdn_0.wo[0], cfg, is_wi=False)
        disp = layers_lib.moe_local_route_and_dispatch(x_l, rg_f, cfg)
        part = layers_lib.moe_expert_gmm_and_unsort(disp, wi_f, wo_f, cfg, sublayer_idx=0)
        y_l = layers_lib.moe_combine_and_add_base(part, x_l, cfg)
        return jnp.sum(y_l * ct_l), y_l

      def _sharded(p_l, x_l, ct_l):
        (l_v, y_l), (gp, gx) = jax.value_and_grad(_body, argnums=(0, 1), has_aux=True)(p_l, x_l, ct_l)
        return (
            jax.lax.psum(l_v, axis_name=cfg.all_sharded_axes),
            y_l,
            gp.blocks.gdn_0.moe,
            gx,
        )

      return jax.jit(
          jax.shard_map(
              _sharded,
              mesh=mesh,
              in_specs=(qwen35_jax.get_param_pspecs(cfg), act_pspec, act_pspec),
              out_specs=(
                  P(),
                  act_pspec,
                  qwen35_jax.get_param_pspecs(cfg).blocks.gdn_0.moe,
                  act_pspec,
              ),
              check_vma=False,
          )
      )(p_sharded, x_s, ct_s)

    moe_loss_cp, y_cp, gmoe_cp, gx_cp = _run_cp_moe(cfg_cp, mesh_cp)
    moe_loss_ref, y_ref, gmoe_ref, gx_ref = _run_cp_moe(cfg_ref, mesh_ref)
    self.assertLess(
        abs(float(moe_loss_cp) - float(moe_loss_ref)),
        1e-4 * max(1.0, abs(float(moe_loss_ref))),
    )
    for name, a_cp, a_ref in (
        ("y", y_cp, y_ref),
        ("dx", gx_cp, gx_ref),
        ("dwi", gmoe_cp.wi, gmoe_ref.wi),
        ("dwo", gmoe_cp.wo, gmoe_ref.wo),
        ("drouter_gate", gmoe_cp.router_gate, gmoe_ref.router_gate),
    ):
      c_np = np.asarray(a_cp, dtype=np.float64)
      r_np = np.asarray(a_ref, dtype=np.float64)
      rel = float(np.linalg.norm(c_np - r_np) / (np.linalg.norm(r_np) + 1e-12))
      self.assertLess(rel, 5e-3, f"CP MoE {name} mismatch: {rel}")

    # 3. Full 8-layer end-to-end CP=4 vs CP=1 with FP8 no-drop CF=0.5 routed MoE active
    #    agrees within FP8 dynamic-scale rounding noise and produces finite gradients.
    params_ref_raw = jax.tree.map(
        lambda arr, ps: jax.device_put(np.asarray(arr), NamedSharding(mesh_ref, ps)),
        raw_params,
        qwen35_jax.get_param_pspecs(cfg_ref),
    )
    loss_cp, grads_cp = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_cp, mesh_cp))(raw_params, *batch_cp)
    loss_ref, grads_ref = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_ref, mesh_ref))(params_ref_raw, *batch_ref)
    self.assertLess(
        float(jnp.abs(loss_cp - loss_ref) / jnp.abs(loss_ref)),
        1e-2,
        f"{loss_cp} vs {loss_ref}",
    )
    for path, g in jax.tree.leaves_with_path((grads_cp, grads_ref)):
      self.assertTrue(
          bool(jnp.all(jnp.isfinite(g))),
          f"Non-finite CP grad on {jax.tree_util.keystr(path)}",
      )

  def test_schedule_and_sharding_variants_parity(self):
    # global_batch_size=16 and equalized per-expert wi/wo absmax (as in the CP test)
    # keep the per-microbatch FP8 scales identical across schedules, so the FP8
    # backward does not amplify schedule-induced f32 reorderings into O(1e-2) grad
    # differences.
    base_cfg = qwen35_jax.tiny_config(
        schedule_mode="batchsplit",
        mu_dtype="float32",
        global_batch_size=16,
    )
    mesh = qwen35_jax.create_mesh(base_cfg, aot_topology=None)
    raw_params = qwen35_jax.init_random_params(base_cfg, mesh, jax.random.PRNGKey(7), param_dtype=jnp.float32)
    params_base = dataclasses.replace(
        raw_params,
        blocks=dataclasses.replace(
            raw_params.blocks,
            **{
                sub_name: dataclasses.replace(
                    getattr(raw_params.blocks, sub_name),
                    moe=dataclasses.replace(
                        getattr(raw_params.blocks, sub_name).moe,
                        wi=_equalize_expert_amax(getattr(raw_params.blocks, sub_name).moe.wi),
                        wo=_equalize_expert_amax(getattr(raw_params.blocks, sub_name).moe.wo),
                    ),
                )
                for sub_name in ("gdn_0", "gdn_1", "gdn_2", "attn_3")
            },
        ),
    )
    batch = _make_synthetic_batch(base_cfg, mesh)
    ref_loss, ref_grads = jax.jit(qwen35_jax.make_value_and_grad_fn(base_cfg, mesh))(params_base, *batch)

    variants = [
        dict(
            schedule_mode="batchsplit",
            fsdp_pipeline_mode="sublayer_prefetch",
            bwd_tie_mode="all4",
        ),
        dict(
            schedule_mode="batchsplit",
            tie_f4_before_wo=True,
            bwd_tie_mode="moe_only",
        ),
        dict(schedule_mode="batchsplit", bwd_tie_mode="symmetric_4phase"),
        dict(
            schedule_mode="batchsplit",
            bwd_tie_mode="symmetric_4phase",
            embed_mode="ep_all_to_all_fsdp_gather",
            moe_ag_start_tie="f2_base_mb0",
            ep_combine_fwd_unpinned=True,
        ),
        dict(
            schedule_mode="batchsplit",
            routed_w_tiled_layout=False,
            embed_mode="sc_gather",
        ),
        dict(schedule_mode="batchsplit", dense_sharding_mode="fsdp_only"),
        dict(
            schedule_mode="batchsplit",
            route_before_shared_expert=True,
            fsdp_pipeline_mode="full_prefetch_defer_rs",
            ep_combine_fwd_unpinned=False,
        ),
        dict(
            schedule_mode="batchsplit",
            route_before_shared_expert=False,
            fsdp_pipeline_mode="full_prefetch_defer_rs",
            moe_rw_prefetch_in_f3=True,
            bwd_sc_balance_mode="wi_dense_rs_sc0",
            ep_combine_fwd_unpinned=False,
        ),
        dict(
            schedule_mode="batchsplit",
            strict_f2_before_f3=True,
            defer_mb1_combine_after_f1=True,
            precompute_rope_tables=True,
            fsdp_pipeline_mode="full_prefetch_defer_rs",
            bwd_sc_balance_mode="dwo_sc0",
            mesh_mode="xcore_ep8_yz_fsdp64",
            ep_combine_fwd_unpinned=False,
        ),
        dict(
            schedule_mode="batchsplit",
            strict_f2_before_f3=False,
            defer_mb1_combine_after_f1=False,
            precompute_rope_tables=False,
            ep_combine_fwd_unpinned=False,
        ),
    ]
    for ov in variants:
      with self.subTest(**ov):
        cfg_v = qwen35_jax.tiny_config(mu_dtype="float32", global_batch_size=16, **ov)
        mesh_v = qwen35_jax.create_mesh(cfg_v, aot_topology=None)
        pspecs_v = qwen35_jax.get_param_pspecs(cfg_v)
        params_v = jax.tree.map(
            lambda arr, ps: jax.device_put(np.asarray(arr), NamedSharding(mesh_v, ps)),
            params_base,
            pspecs_v,
        )
        batch_v = tuple(
            jax.device_put(
                np.asarray(arr),
                NamedSharding(mesh_v, qwen35_jax.data_pspec(cfg_v)),
            )
            for arr in batch
        )
        loss_v, grads_v = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_v, mesh_v))(params_v, *batch_v)
        loss_diff = float(jnp.max(jnp.abs(ref_loss - loss_v)))
        rels = {}
        for (path, gr), (_, gv) in zip(
            jax.tree.leaves_with_path(ref_grads),
            jax.tree.leaves_with_path(grads_v),
        ):
          gr_np = np.asarray(gr, np.float64)
          gv_np = np.asarray(gv, np.float64)
          rels[jax.tree_util.keystr(path)] = float(np.linalg.norm(gr_np - gv_np) / max(np.linalg.norm(gr_np), 1e-12))
        self.assertLess(loss_diff, 1e-5)
        # Schedule-only variants: FP8 backward quantization amplifies f32 reorderings.
        for path, rel in rels.items():
          self.assertLess(rel, 1e-2, f"{ov} mismatch on {path}: {rel}")

  def test_ragged_ops_silu_mul_and_quantize_parity(self):
    from maxtext.experimental.qwen_35_64c import ragged_ops  # pylint: disable=g-import-not-at-top

    key = jax.random.PRNGKey(314)
    k1, k2, k3 = jax.random.split(key, 3)
    gate_up = jax.random.normal(k1, (128, 256), dtype=jnp.bfloat16)
    w_rows = jax.nn.softmax(jax.random.normal(k2, (128, 1), dtype=jnp.float32), axis=0) * 16.0
    d_act = jax.random.normal(k3, (128, 128), dtype=jnp.bfloat16)
    cnt = jnp.int32(90)

    act = ragged_ops.ragged_silu_mul_fwd(gate_up, 0, w_rows, cnt, use_pallas=False)
    valid = (jnp.arange(128) < cnt)[:, None]
    g_f32 = gate_up[:, :128].astype(jnp.float32)
    u_f32 = gate_up[:, 128:].astype(jnp.float32)
    ref_act = jnp.where(valid, jax.nn.silu(g_f32) * u_f32 * w_rows, 0.0).astype(jnp.bfloat16)
    self.assertEqual(
        float(jnp.max(jnp.abs(act.astype(jnp.float32) - ref_act.astype(jnp.float32)))),
        0.0,
    )
    self.assertEqual(float(jnp.max(jnp.abs(act[90:].astype(jnp.float32)))), 0.0)

    act_fp8, s_act = ragged_ops.ragged_absmax_quantize(act, jnp.float8_e4m3fn, cnt, use_pallas=False)
    act_fp8_fused = ragged_ops.ragged_silu_mul_fwd(
        gate_up,
        0,
        w_rows,
        cnt,
        out_dtype=jnp.float8_e4m3fn,
        out_scale=s_act,
        use_pallas=False,
    )
    self.assertEqual(
        float(jnp.max(jnp.abs(act_fp8.astype(jnp.float32) - act_fp8_fused.astype(jnp.float32)))),
        0.0,
    )
    self.assertEqual(float(jnp.max(jnp.abs(act_fp8[90:].astype(jnp.float32)))), 0.0)

    dg_gu, dw_r, amax_dg = ragged_ops.ragged_silu_mul_bwd(
        d_act, gate_up, 0, w_rows, cnt, with_absmax=True, use_pallas=False
    )

    def _ref_fn(gu_f32, wr_f32):
      mask = (jnp.arange(128) < cnt)[:, None]
      y = jnp.where(mask, jax.nn.silu(gu_f32[:, :128]) * gu_f32[:, 128:] * wr_f32, 0.0)
      return jnp.sum(y * d_act.astype(jnp.float32))

    ref_dg, ref_dw = jax.grad(_ref_fn, argnums=(0, 1))(gate_up.astype(jnp.float32), w_rows)
    self.assertLess(
        float(jnp.max(jnp.abs(dg_gu.astype(jnp.float32) - ref_dg.astype(jnp.bfloat16).astype(jnp.float32)))),
        1e-2,
    )
    self.assertLess(float(jnp.max(jnp.abs(dw_r - ref_dw))), 5e-3)
    s_dg = ragged_ops.absmax_to_scale(amax_dg, jnp.float8_e4m3fn)
    dg_q = ragged_ops.ragged_quantize(dg_gu, s_dg, jnp.float8_e4m3fn, cnt, use_pallas=False)
    self.assertEqual(float(jnp.max(jnp.abs(dg_q[90:].astype(jnp.float32)))), 0.0)

  def test_fp8_moe_and_optax_train_step_decreases_loss(self):
    cfg = qwen35_jax.tiny_config(
        emb_dim=256,
        moe_mlp_dim=128,
        num_query_heads=8,
        gdn_num_v_heads=8,
        schedule_mode="batchsplit",
        moe_fp8_bwd_dispatch=True,
        moe_bwd_dy_fp8_e5m2=True,
        sa_qk_seq_minor=True,
        moe_bwd_tie_wgrad_before_gate=True,
        f2_light_router_ties=True,
        save_moe_routing=True,
        offload_moe_gate_up_sublayers=(1, 2, 3),
        strict_f2_before_f3=True,
        defer_mb1_combine_after_f1=True,
        precompute_rope_tables=True,
        route_before_shared_expert=False,
        bwd_sc_balance_mode="wi_rs_sc0",
        fsdp_pipeline_mode="full_prefetch_defer_rs",
        learning_rate=3e-3,
    )
    mesh = qwen35_jax.create_mesh(cfg, aot_topology=None)
    params = qwen35_jax.init_random_params(cfg, mesh, jax.random.PRNGKey(99), param_dtype=jnp.bfloat16)
    tx = qwen35_jax.create_optimizer(cfg)
    param_shardings = jax.tree.map(lambda x: x.sharding, params)
    opt_state = jax.jit(tx.init, in_shardings=(param_shardings,))(params)
    batch = _make_synthetic_batch(cfg, mesh)

    step_fn = qwen35_jax.make_train_step(cfg, mesh, donate_state=True)
    losses = []
    for _ in range(4):
      params, opt_state, loss = step_fn(params, opt_state, *batch)
      losses.append(float(loss))

    self.assertTrue(all(np.isfinite(losses)), f"Non-finite losses: {losses}")
    self.assertLess(losses[-1], losses[0], f"Loss did not decrease over 4 steps: {losses}")

  def test_forked_3d_gmmv2_and_65key_sort_parity(self):
    common = dict(
        emb_dim=256,
        moe_mlp_dim=128,
        num_query_heads=8,
        gdn_num_v_heads=8,
        mu_dtype="float32",
        moe_fp8_weights=True,
        moe_fp8_gmm_compute=True,
        moe_fp8_token_dispatch=True,
        moe_fp8_bwd_dispatch=True,
        moe_use_ragged_silu_3d_fp8=True,
        moe_use_forked_3d_gmmv2=True,
        moe_bwd_dy_fp8_e5m2=True,
        moe_use_tc_ragged=True,
        moe_3d_ep_tokens=True,
        strict_f2_before_f3=True,
        defer_mb1_combine_after_f1=True,
        precompute_rope_tables=True,
        fsdp_pipeline_mode="full_prefetch_defer_rs",
        ep_combine_fwd_unpinned=False,
    )
    cfg_65k = qwen35_jax.tiny_config(schedule_mode="batchsplit", moe_local_65key_sort=True, **common)
    cfg_512k = qwen35_jax.tiny_config(schedule_mode="batchsplit", moe_local_65key_sort=False, **common)
    mesh = qwen35_jax.create_mesh(cfg_65k, aot_topology=None)
    params = qwen35_jax.init_random_params(cfg_65k, mesh, jax.random.PRNGKey(17), param_dtype=jnp.bfloat16)
    batch = _make_synthetic_batch(cfg_65k, mesh)

    l_65, g_65 = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_65k, mesh))(params, *batch)
    l_512, g_512 = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_512k, mesh))(params, *batch)
    self.assertLess(float(jnp.max(jnp.abs(l_65 - l_512))), 1e-6)
    for (path, ga), (_, gb) in zip(jax.tree.leaves_with_path(g_65), jax.tree.leaves_with_path(g_512)):
      diff = float(jnp.max(jnp.abs(np.asarray(ga, np.float32) - np.asarray(gb, np.float32))))
      self.assertEqual(
          diff,
          0.0,
          f"65-key vs 512-key sort mismatch on {jax.tree_util.keystr(path)}:" f" {diff}",
      )

  def test_moe_capacity_factor_chunks_match_single_chunk_and_dense_reference(
      self,
  ):
    from maxtext.experimental.qwen_35_64c import layers as layers_lib  # pylint: disable=g-import-not-at-top
    from maxtext.experimental.qwen_35_64c import moe_kernels as mk_lib  # pylint: disable=g-import-not-at-top
    from maxtext.experimental.qwen_35_64c import ops as ops_lib  # pylint: disable=g-import-not-at-top

    common = dict(
        emb_dim=256,
        moe_mlp_dim=128,
        num_query_heads=8,
        gdn_num_v_heads=8,
        schedule_mode="unsplit",
        num_superblocks=1,
        enable_remat=False,
    )
    cfg_ref = qwen35_jax.tiny_config(capacity_factor=4.0, **common)
    mesh = qwen35_jax.create_mesh(cfg_ref, aot_topology=None)
    params = qwen35_jax.init_random_params(cfg_ref, mesh, jax.random.PRNGKey(5), param_dtype=jnp.float32)
    axes = cfg_ref.all_sharded_axes
    rows = cfg_ref.num_devices * cfg_ref.local_batch_size
    seq, emb, topk = (
        cfg_ref.seq_len,
        cfg_ref.emb_dim,
        cfg_ref.num_experts_per_tok,
    )
    num_tokens = rows * seq
    kx, kg, kw, kc = jax.random.split(jax.random.PRNGKey(2027), 4)
    x_np = np.asarray(jax.random.normal(kx, (rows, seq, emb), dtype=jnp.float32))
    ct_np = np.asarray(jax.random.normal(kc, (rows, seq, emb), dtype=jnp.float32))
    # Deliberately imbalanced routing: EP rank 0's experts (0..3) are strongly preferred.
    bias = np.zeros((cfg_ref.num_experts,), np.float32)
    bias[: cfg_ref.local_num_experts] = 2.5
    gumbel = np.asarray(jax.random.gumbel(kg, (num_tokens, cfg_ref.num_experts)))
    _, ids_j = jax.lax.top_k(jnp.asarray(gumbel + bias), topk)
    ids_np = np.asarray(ids_j, np.int32)
    w_np = np.asarray(jax.nn.softmax(jax.random.normal(kw, (num_tokens, topk)), axis=-1))
    hot_frac = float(np.mean(ids_np < cfg_ref.local_num_experts))
    self.assertGreater(hot_frac, 0.6, f"Routing not imbalanced enough: {hot_frac}")

    tok_sharding = NamedSharding(mesh, P(axes, None, None))
    route_sharding = NamedSharding(mesh, P(axes, None))
    x_g = jax.device_put(x_np, tok_sharding)
    ct_g = jax.device_put(ct_np, tok_sharding)
    ids_g = jax.device_put(ids_np, route_sharding)
    w_g = jax.device_put(w_np, route_sharding)

    def _routed_moe_vg(cfg):
      def _body(p_l, x_l, w_l, ids_l, ct_l):
        sb_amax = qwen35_jax.model._compute_hoisted_superblock_absmax(p_l.blocks, cfg)
        sub0 = jax.tree.map(lambda z: z[0], p_l.blocks.gdn_0)
        wi_f = mk_lib.gather_routed_weight_sc1(sub0.moe.wi, sb_amax.gdn_0.wi[0], cfg, is_wi=True)
        wo_f = mk_lib.gather_routed_weight_sc1(sub0.moe.wo, sb_amax.gdn_0.wo[0], cfg, is_wi=False)
        disp = layers_lib.moe_dispatch_from_routed(x_l, w_l, ids_l, cfg, order_tokens_on_router=False)
        part = layers_lib.moe_expert_gmm_and_unsort(disp, wi_f, wo_f, cfg, wrap_partial_fp8_bwd=True, sublayer_idx=0)
        y = ops_lib.ep_token_combine(part, cfg).reshape(x_l.shape)
        return jnp.sum(y * ct_l), y

      def _sharded(p_l, x_l, w_l, ids_l, ct_l):
        (l_v, y), grads = jax.value_and_grad(_body, argnums=(0, 1, 2), has_aux=True)(p_l, x_l, w_l, ids_l, ct_l)
        return jax.lax.psum(l_v, axis_name=axes), y, grads

      pspecs = qwen35_jax.get_param_pspecs(cfg)
      tok_spec, route_spec = P(axes, None, None), P(axes, None)
      l_v, y, (gp, gx, gw) = jax.jit(
          jax.shard_map(
              _sharded,
              mesh=mesh,
              in_specs=(pspecs, tok_spec, route_spec, route_spec, tok_spec),
              out_specs=(P(), tok_spec, (pspecs, tok_spec, route_spec)),
              check_vma=False,
          )
      )(params, x_g, w_g, ids_g, ct_g)
      moe_g = gp.blocks.gdn_0.moe
      return {
          "loss": np.asarray(l_v),
          "y": np.asarray(y),
          "dx": np.asarray(gx),
          "dw": np.asarray(gw),
          "dwi": np.asarray(moe_g.wi[0]),
          "dwo": np.asarray(moe_g.wo[0]),
      }

    def _rel_l2(a, b):
      return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-12))

    # Dense pure-JAX f32 reference: every routed slot contributes (no drops).
    wi_dense = np.asarray(params.blocks.gdn_0.moe.wi[0])
    wo_dense = np.asarray(params.blocks.gdn_0.moe.wo[0])
    mlp = cfg_ref.moe_mlp_dim

    def _dense_loss(wi_d, wo_d, x_d, w_d):
      xf = x_d.reshape(num_tokens, emb)
      gate_up = jnp.einsum("nd,edf->nef", xf, wi_d)
      act = jax.nn.silu(gate_up[..., :mlp]) * gate_up[..., mlp:]
      out_e = jnp.einsum("nef,efm->nem", act, wo_d)
      sel = jnp.take_along_axis(out_e, jnp.asarray(ids_np)[:, :, None], axis=1)
      y = jnp.sum(sel * w_d[..., None], axis=1).reshape(x_d.shape)
      return jnp.sum(y * ct_np), y

    (l_dense, y_dense), (g_wi, g_wo, g_x, g_w) = jax.jit(
        jax.value_and_grad(_dense_loss, argnums=(0, 1, 2, 3), has_aux=True)
    )(wi_dense, wo_dense, x_np, w_np)
    dense = {
        "y": np.asarray(y_dense),
        "dx": np.asarray(g_x),
        "dw": np.asarray(g_w),
        "dwi": np.asarray(g_wi),
        "dwo": np.asarray(g_wo),
    }
    del l_dense

    single = _routed_moe_vg(cfg_ref)
    # Single chunk covers all T_ep * top_k slots.
    t_ep = num_tokens // cfg_ref.fsdp_size
    self.assertEqual(cfg_ref.moe_chunk_capacity(t_ep // cfg_ref.ep_size), t_ep * topk)
    for key, ref in dense.items():
      rel = _rel_l2(single[key], ref)
      self.assertLess(rel, 0.12, f"single-chunk vs dense reference mismatch on {key}: {rel}")

    # 0.25 -> 32-slot chunks (~10 chunks on the hot rank), 0.75 -> 96 (unaligned last
    # chunk), 1.0 -> 128; plus the 512-key global sort path.
    for cf, extra in (
        (0.25, {}),
        (0.75, {}),
        (1.0, {}),
        (0.75, dict(moe_local_65key_sort=False)),
    ):
      with self.subTest(capacity_factor=cf, **extra):
        cfg_cf = qwen35_jax.tiny_config(capacity_factor=cf, **common, **extra)
        cap = cfg_cf.moe_chunk_capacity(t_ep // cfg_cf.ep_size)
        self.assertLess(3 * cap, int(hot_frac * t_ep * topk) if cf < 1.0 else 10**9)
        chunked = _routed_moe_vg(cfg_cf)
        self.assertLess(
            abs(float(chunked["loss"]) - float(single["loss"])),
            1e-4 * max(1.0, abs(float(single["loss"]))),
        )
        # The FP8 act / e5m2 d_gate roundings make y, dx, dwi, dwo chunk-invariant up to f32
        # summation order. dw reads the bf16 bank directly, and CPU ragged_dot's f32
        # accumulation order depends on the chunk's row count, so a few bank elements land
        # on the other side of a bf16 rounding boundary.
        for key, tol in (
            ("y", 1e-5),
            ("dx", 1e-5),
            ("dw", 5e-3),
            ("dwi", 1e-5),
            ("dwo", 1e-5),
        ):
          rel = _rel_l2(chunked[key], single[key])
          self.assertLess(
              rel,
              tol,
              f"cf={cf} {extra} vs single chunk mismatch on {key}: {rel}",
          )
        # Overflow windows gather their routing weights and scatter-add `d(topk_weights)`
        # inside the chunk loops: every routed slot (incl. overflow) must match the dense
        # reference, and slots never routed to this rank's experts get no spurious grads.
        for key, ref in dense.items():
          rel = _rel_l2(chunked[key], ref)
          self.assertLess(
              rel,
              0.12,
              f"cf={cf} {extra} vs dense reference mismatch on {key}: {rel}",
          )
        self.assertTrue(np.all(np.isfinite(chunked["dw"])), f"cf={cf} {extra} non-finite dw")

  def test_wave2_checkpoint_bank_routing_and_schedule_ties_exact_parity(self):
    from maxtext.experimental.qwen_35_64c import layers as layers_lib  # pylint: disable=g-import-not-at-top
    from maxtext.experimental.qwen_35_64c import moe_kernels as mk_lib  # pylint: disable=g-import-not-at-top
    from maxtext.experimental.qwen_35_64c import ops as ops_lib  # pylint: disable=g-import-not-at-top

    common = dict(
        emb_dim=256,
        moe_mlp_dim=128,
        num_query_heads=8,
        gdn_num_v_heads=8,
        mu_dtype="float32",
        moe_fp8_weights=True,
        moe_fp8_gmm_compute=True,
        moe_fp8_token_dispatch=True,
        moe_fp8_bwd_dispatch=True,
        moe_use_ragged_silu_3d_fp8=True,
        moe_use_forked_3d_gmmv2=True,
        moe_bwd_dy_fp8_e5m2=True,
        moe_local_65key_sort=True,
        moe_use_tc_ragged=True,
        moe_3d_ep_tokens=True,
        strict_f2_before_f3=True,
        defer_mb1_combine_after_f1=True,
        precompute_rope_tables=True,
        fsdp_pipeline_mode="full_prefetch_defer_rs",
        ep_combine_fwd_unpinned=False,
        sa_qk_seq_minor=True,
    )

    def _assert_exact_tree(tag: str, l_a, g_a, l_b, g_b, tol: float = 0.0):
      self.assertLessEqual(float(jnp.max(jnp.abs(l_a - l_b))), tol, f"{tag} loss mismatch")
      for (path, ga), (_, gb) in zip(jax.tree.leaves_with_path(g_a), jax.tree.leaves_with_path(g_b)):
        a32 = np.asarray(ga, np.float32)
        b32 = np.asarray(gb, np.float32)
        diff = float(np.max(jnp.abs(a32 - b32)))
        self.assertLessEqual(
            diff,
            tol,
            f"{tag} grad mismatch on {jax.tree_util.keystr(path)}: {diff}",
        )

    def _assert_rel_l2_tree(tag: str, l_a, g_a, l_b, g_b, loss_tol: float, rel_tol: float):
      self.assertLessEqual(float(jnp.max(jnp.abs(l_a - l_b))), loss_tol, f"{tag} loss mismatch")
      for (path, ga), (_, gb) in zip(jax.tree.leaves_with_path(g_a), jax.tree.leaves_with_path(g_b)):
        a32 = np.asarray(ga, np.float32)
        b32 = np.asarray(gb, np.float32)
        rel_l2 = float(np.linalg.norm((a32 - b32).ravel())) / max(float(np.linalg.norm(a32.ravel())), 1e-6)
        self.assertLessEqual(
            rel_l2,
            rel_tol,
            f"{tag} rel_l2 mismatch on {jax.tree_util.keystr(path)}: {rel_l2}",
        )

    # 1. Single-sublayer exact (0.0) parity of `routed_moe_forked_3d_fp8` across
    #    no-checkpoint, full remat, HBM `T(8, 128)` bank save, `S(5)` bank offload, and FP8 disp save!
    cfg_sub_base = qwen35_jax.tiny_config(schedule_mode="unsplit", num_superblocks=1, enable_remat=False, **common)
    mesh = qwen35_jax.create_mesh(cfg_sub_base, aot_topology=None)
    params_sub = qwen35_jax.init_random_params(cfg_sub_base, mesh, jax.random.PRNGKey(23), param_dtype=jnp.float32)
    x_sharding = NamedSharding(mesh, P(("fsdp", "expert"), None, None))
    x_global = jax.device_put(
        jax.random.normal(
            jax.random.PRNGKey(77),
            (cfg_sub_base.num_devices * 2, 32, 256),
            dtype=jnp.float32,
        ),
        x_sharding,
    )

    def _moe_sub_vg(use_ckpt: bool, **ckpt_ov):
      c_sub = qwen35_jax.tiny_config(
          schedule_mode="unsplit",
          num_superblocks=1,
          enable_remat=use_ckpt,
          **common,
          **ckpt_ov,
      )
      pol = qwen35_jax.build_remat_policy(c_sub)

      def _body(p_l, x_in):
        sb_amax = qwen35_jax.model._compute_hoisted_superblock_absmax(p_l.blocks, c_sub)
        amax_wi = sb_amax.gdn_0.wi[0]
        amax_wo = sb_amax.gdn_0.wo[0]

        def _inner(x_a, sub_p):
          rg = ops_lib.gather_dense_weight(sub_p.moe.router_gate, c_sub, sharded_axis=0, is_fsdp_expert=True)
          wi_f = mk_lib.gather_routed_weight_sc1(sub_p.moe.wi, amax_wi, c_sub, is_wi=True)
          wo_f = mk_lib.gather_routed_weight_sc1(sub_p.moe.wo, amax_wo, c_sub, is_wi=False)
          disp = layers_lib.moe_local_route_and_dispatch(x_a, rg, c_sub)
          part = layers_lib.moe_expert_gmm_and_unsort(disp, wi_f, wo_f, c_sub, wrap_partial_fp8_bwd=True, sublayer_idx=0)
          comb = ops_lib.ep_token_combine(part, c_sub).reshape(x_a.shape)
          return x_a + comb

        fn = jax.checkpoint(_inner, policy=pol, prevent_cse=False) if use_ckpt else _inner
        sub0 = jax.tree.map(lambda z: z[0], p_l.blocks.gdn_0)
        y = fn(x_in, sub0)
        return jnp.sum(y * jnp.sin(jnp.arange(y.size, dtype=jnp.float32).reshape(y.shape)))

      def _sharded_vg(p_l, x_l):
        l_v, (gp, gx) = jax.value_and_grad(_body, argnums=(0, 1))(p_l, x_l)
        return jax.lax.psum(l_v, axis_name=("fsdp", "expert")), (gp, gx)

      return jax.jit(
          jax.shard_map(
              _sharded_vg,
              mesh=mesh,
              in_specs=(
                  qwen35_jax.get_param_pspecs(c_sub),
                  P(("fsdp", "expert"), None, None),
              ),
              out_specs=(
                  P(),
                  (
                      qwen35_jax.get_param_pspecs(c_sub),
                      P(("fsdp", "expert"), None, None),
                  ),
              ),
              check_vma=False,
          )
      )(params_sub, x_global)

    ml_nockpt, mg_nockpt = _moe_sub_vg(False)
    ml_remat, mg_remat = _moe_sub_vg(True, save_moe_routing=True)
    ml_savegu, mg_savegu = _moe_sub_vg(True, save_moe_gate_up_sublayers=(0,))
    ml_offloadgu, mg_offloadgu = _moe_sub_vg(True, offload_moe_gate_up_sublayers=(0,))
    ml_saveall, mg_saveall = _moe_sub_vg(True, save_moe_gate_up_sublayers=(0,), save_moe_disp_ep_sublayers=(0,))
    _assert_exact_tree("moe_nockpt_vs_remat", ml_nockpt, mg_nockpt, ml_remat, mg_remat, tol=0.0)
    _assert_exact_tree(
        "moe_nockpt_vs_savegu",
        ml_nockpt,
        mg_nockpt,
        ml_savegu,
        mg_savegu,
        tol=0.0,
    )
    _assert_exact_tree(
        "moe_nockpt_vs_offloadgu",
        ml_nockpt,
        mg_nockpt,
        ml_offloadgu,
        mg_offloadgu,
        tol=0.0,
    )
    _assert_exact_tree(
        "moe_nockpt_vs_saveall",
        ml_nockpt,
        mg_nockpt,
        ml_saveall,
        mg_saveall,
        tol=0.0,
    )

    # 2. Multi-superblock (`8`-layer) bit-exact (`0.0` tol) schedule-tie & bank-save/offload invariants
    #    in both FP32 and BF16!
    cfg_base_fp8 = qwen35_jax.tiny_config(schedule_mode="batchsplit", enable_remat=False, **common)
    batch = _make_synthetic_batch(cfg_base_fp8, mesh)
    for dtype_tag, p_dtype in (("fp32", jnp.float32), ("bf16", jnp.bfloat16)):
      p_multi = qwen35_jax.init_random_params(cfg_base_fp8, mesh, jax.random.PRNGKey(29), param_dtype=p_dtype)
      l_nr, g_nr = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_base_fp8, mesh))(p_multi, *batch)
      for tie_label, tie_ov in (
          (
              "bwd_tie_wgrad",
              dict(enable_remat=False, moe_bwd_tie_wgrad_before_gate=True),
          ),
          (
              "f2_light_ties",
              dict(enable_remat=False, f2_light_router_ties=True),
          ),
      ):
        cfg_t = qwen35_jax.tiny_config(schedule_mode="batchsplit", **common, **tie_ov)
        l_t, g_t = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_t, mesh))(p_multi, *batch)
        _assert_exact_tree(f"{dtype_tag}_{tie_label}", l_nr, g_nr, l_t, g_t, tol=0.0)

      # Verify `save_moe_gate_up_sublayers` alone auto-saves `"moe_routing"` and matches explicit
      # `save_moe_routing=True` bit-for-bit (`0.0` tol), matches `+ save_moe_disp_ep_sublayers`
      # (`r4_w2_20L_save_gu_disp_all_hbm`), and `S(5)` `offload_moe_gate_up_sublayers=(1,2,3)`
      # matches HBM bit-for-bit (`0.0` tol)!
      cfg_gu_alone = qwen35_jax.tiny_config(
          schedule_mode="batchsplit",
          enable_remat=True,
          hbm_save_offload_targets=True,
          save_moe_gate_up_sublayers=(0, 1, 2, 3),
          **common,
      )
      cfg_gu_explicit = qwen35_jax.tiny_config(
          schedule_mode="batchsplit",
          enable_remat=True,
          hbm_save_offload_targets=True,
          moe_bwd_tie_wgrad_before_gate=True,
          f2_light_router_ties=True,
          save_moe_routing=True,
          save_moe_gate_up_sublayers=(0, 1, 2, 3),
          **common,
      )
      cfg_gu_and_disp = qwen35_jax.tiny_config(
          schedule_mode="batchsplit",
          enable_remat=True,
          hbm_save_offload_targets=True,
          moe_bwd_tie_wgrad_before_gate=True,
          f2_light_router_ties=True,
          save_moe_routing=True,
          save_moe_gate_up_sublayers=(0, 1, 2, 3),
          save_moe_disp_ep_sublayers=(0, 1, 2, 3),
          **common,
      )
      l_ga, g_ga = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_gu_alone, mesh))(p_multi, *batch)
      l_ge, g_ge = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_gu_explicit, mesh))(p_multi, *batch)
      l_gd, g_gd = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_gu_and_disp, mesh))(p_multi, *batch)
      _assert_exact_tree(
          f"{dtype_tag}_gu_auto_routing_vs_explicit",
          l_ga,
          g_ga,
          l_ge,
          g_ge,
          tol=0.0,
      )
      _assert_rel_l2_tree(
          f"{dtype_tag}_gu_vs_gu_and_disp",
          l_ge,
          g_ge,
          l_gd,
          g_gd,
          loss_tol=1e-4 if dtype_tag == "fp32" else 5e-3,
          rel_tol=2e-3 if dtype_tag == "fp32" else 2e-2,
      )

      r4_champ_common = dict(
          enable_remat=True,
          moe_bwd_tie_wgrad_before_gate=True,
          f2_light_router_ties=True,
          save_moe_routing=True,
          offload_decoder_input=True,
          save_attn_context_hbm=True,
          save_query_proj_hbm=True,
          save_out_proj_hbm=True,
      )
      cfg_gu123_hbm = qwen35_jax.tiny_config(
          schedule_mode="batchsplit",
          save_moe_gate_up_sublayers=(1, 2, 3),
          **common,
          **r4_champ_common,
      )
      cfg_gu123_s5 = qwen35_jax.tiny_config(
          schedule_mode="batchsplit",
          offload_moe_gate_up_sublayers=(1, 2, 3),
          **common,
          **r4_champ_common,
      )
      l_hbm, g_hbm = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_gu123_hbm, mesh))(p_multi, *batch)
      l_s5, g_s5 = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_gu123_s5, mesh))(p_multi, *batch)
      _assert_exact_tree(
          f"{dtype_tag}_gu123_hbm_vs_s5_offload",
          l_hbm,
          g_hbm,
          l_s5,
          g_s5,
          tol=0.0,
      )

    # 3. Full-superblock `enable_remat=False` vs `enable_remat=True` with
    #    schedule ties (`< 5e-5` rel L2):
    common_cont = dict(
        common,
        num_superblocks=1,
        moe_bwd_tie_wgrad_before_gate=True,
        f2_light_router_ties=True,
    )
    cfg_cont_nr = qwen35_jax.tiny_config(schedule_mode="batchsplit", enable_remat=False, **common_cont)
    cfg_cont_remat = qwen35_jax.tiny_config(
        schedule_mode="batchsplit",
        enable_remat=True,
        save_moe_routing=True,
        hbm_save_offload_targets=True,
        **common_cont,
    )
    l_cnr, g_cnr = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_cont_nr, mesh))(params_sub, *batch)
    l_cr, g_cr = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_cont_remat, mesh))(params_sub, *batch)
    # FP8 e5m2 backward quantization amplifies remat-recompute f32 reorderings.
    _assert_rel_l2_tree(
        "continuous_remat",
        l_cnr,
        g_cnr,
        l_cr,
        g_cr,
        loss_tol=1e-5,
        rel_tol=5e-2,
    )

    # 4. Guard against unsupported FP8 `S(5)` pinned-host offload on TPU:
    cfg_bad_s5 = qwen35_jax.tiny_config(
        enable_remat=True,
        offload_moe_disp_ep_sublayers=(1,),
        hbm_save_offload_targets=False,
    )
    with self.assertRaises(ValueError):
      qwen35_jax.build_remat_policy(cfg_bad_s5)

  def test_max_async_overlap_transform_queues_and_quant_bundles(self):
    from maxtext.experimental.qwen_35_64c import max_async_overlap_transform as maot  # pylint: disable=g-import-not-at-top

    # 1. Verify JSON consumers[] + core_ids[] disambiguation and S(5) async-start host queues.
    sc0_line = (
        "%call-start.523 = ((), (), (s32[8192]{0:T(1024)S(9)})) async-start(),"
        ' async_execution_thread="sparsecore",'
        ' backend_config={"flag_configs":[],"sparse_core_config":{"consumers":[{"core_ids":[],"core_type":"CORE_TYPE_TC"}],"core_ids":["0"]}}'
    )
    sc1_line = sc0_line.replace('"core_ids":["0"]', '"core_ids":["1"]')
    sc01_line = (
        "%all-gather.841.call-start = ((), ()) async-start(),"
        ' async_execution_thread="sparsecore",'
        ' backend_config={"megachip_parallelism_config":{"cores":["0","1"]}}'
    )
    h2d_line = (
        "%dynamic-slice-start ="
        " ((bf16[23,1,4096,8192]{3,2,1,0:T(8,128)(2,1)S(5)}, s32[]{:T(128)}),"
        " bf16[1,1,4096,8192]{3,2,1,0:T(8,128)(2,1)}, u32[]) async-start(%p0,"
        " %i0)"
    )
    d2h_line = (
        "%dynamic-update-slice-start ="
        " ((bf16[23,1,4096,8192]{3,2,1,0:T(8,128)(2,1)S(5)},"
        " bf16[1,1,4096,8192]{3,2,1,0:T(8,128)(2,1)}, s32[]{:T(128)}),"
        " bf16[23,1,4096,8192]{3,2,1,0:T(8,128)(2,1)S(5)}, u32[])"
        " async-start(%p0, %x0, %i0)"
    )
    self.assertEqual(maot.async_queue("kAsyncStart", sc0_line), "sparsecore:0")
    self.assertEqual(maot.async_queue("kAsyncStart", sc1_line), "sparsecore:1")
    self.assertEqual(maot.async_queue("kAsyncStart", sc01_line), "sparsecore:0,1")
    self.assertEqual(maot.async_queue("kAsyncStart", h2d_line), "host:h2d")
    self.assertEqual(maot.async_queue("kAsyncStart", d2h_line), "host:d2h")

    # 2. Verify start_data_formatting bubbles dynamic quant + scalar pmax left without
    #    leaking into bundle_for_done, serialized host queues preserve order, and queue_filter works.
    order = [
        "p0",
        "rag_gather",
        "compute_mid_1",
        "sc1_ag_start",
        "h2d_0_start",
        "h2d_0_done",
        "h2d_1_start",
        "amax_fusion",
        "pmax_scalar",
        "quant_div",
        "ep_ag_start",
        "ep_ag_done",
        "post_bitcast",
        "post_abs_quant_only",
        "compute_mid_2",
        "sc1_ag_done",
        "h2d_1_done",
        "gmm_consumer",
        "root_tuple",
    ]
    opcodes = {
        "p0": "kParameter",
        "rag_gather": "kCustomCall",
        "compute_mid_1": "kCustomCall",
        "sc1_ag_start": "kAsyncStart",
        "h2d_0_start": "kAsyncStart",
        "h2d_0_done": "kAsyncDone",
        "h2d_1_start": "kAsyncStart",
        "amax_fusion": "kFusion",
        "pmax_scalar": "kAllReduce",
        "quant_div": "kDivide",
        "ep_ag_start": "kAsyncStart",
        "ep_ag_done": "kAsyncDone",
        "post_bitcast": "kBitcast",
        "post_abs_quant_only": "kAbs",
        "compute_mid_2": "kCustomCall",
        "sc1_ag_done": "kAsyncDone",
        "h2d_1_done": "kAsyncDone",
        "gmm_consumer": "kCustomCall",
        "root_tuple": "kTuple",
    }
    operands = {
        "p0": [],
        "rag_gather": ["p0"],
        "compute_mid_1": ["p0"],
        "sc1_ag_start": ["p0"],
        "h2d_0_start": ["p0"],
        "h2d_0_done": ["h2d_0_start"],
        "h2d_1_start": ["p0"],
        "amax_fusion": ["rag_gather"],
        "pmax_scalar": ["amax_fusion"],
        "quant_div": ["rag_gather", "pmax_scalar"],
        "ep_ag_start": ["quant_div"],
        "ep_ag_done": ["ep_ag_start"],
        "post_bitcast": ["ep_ag_done"],
        "post_abs_quant_only": ["post_bitcast"],
        "compute_mid_2": ["compute_mid_1", "h2d_0_done"],
        "sc1_ag_done": ["sc1_ag_start"],
        "h2d_1_done": ["h2d_1_start"],
        "gmm_consumer": [
            "post_abs_quant_only",
            "compute_mid_2",
            "sc1_ag_done",
            "h2d_1_done",
        ],
        "root_tuple": ["gmm_consumer"],
    }
    graph = maot.make_schedule_graph(
        order=order,
        opcodes=opcodes,
        operands=operands,
        data_formatting=["post_bitcast"],
        start_data_formatting=[
            "amax_fusion",
            "pmax_scalar",
            "quant_div",
            "post_abs_quant_only",
        ],
        queues={
            "ep_ag_start": "sparsecore:0",
            "sc1_ag_start": "sparsecore:1",
            "h2d_0_start": "host:h2d",
            "h2d_1_start": "host:h2d",
        },
        serialized_queues=["host:h2d", "host:d2h"],
    )
    self.assertEqual(
        maot.bundle_for_start(graph, "ep_ag_start"),
        frozenset({"ep_ag_start", "quant_div", "pmax_scalar", "amax_fusion"}),
    )
    self.assertEqual(
        maot.bundle_for_done(graph, "ep_ag_done"),
        frozenset({"ep_ag_done", "post_bitcast"}),
    )

    res_all = maot.reschedule(graph, queue_filter="all")
    maot.validate_order(graph, res_all.order)
    pos_all = {n: i for i, n in enumerate(res_all.order)}
    self.assertLess(pos_all["ep_ag_start"], pos_all["compute_mid_1"])
    self.assertLess(pos_all["sc1_ag_start"], pos_all["rag_gather"])
    self.assertLess(pos_all["h2d_0_done"], pos_all["h2d_1_start"])

    res_sc0 = maot.reschedule(graph, queue_filter="sc0_only")
    maot.validate_order(graph, res_sc0.order)
    pos_sc0 = {n: i for i, n in enumerate(res_sc0.order)}
    self.assertLess(pos_sc0["ep_ag_start"], pos_sc0["compute_mid_1"])
    self.assertGreater(pos_sc0["sc1_ag_start"], pos_sc0["compute_mid_1"])

    # 3. Verify live POST_SCHEDULER hook registration and stats recording on CPU backend.
    maot.reset_transform_stats()
    try:
      plats = maot.register_transform(
          platforms="cpu",
          bundle_quant_scales=True,
          bundle_scalar_all_reduce=True,
          order_mode="bwd_done_first",
      )
      self.assertIn("cpu", plats)
      out = jax.jit(lambda x: jnp.sin(x) + 1.0)(jnp.ones((16,), dtype=jnp.float32))
      self.assertEqual(out.shape, (16,))
      stats = maot.get_last_transform_stats()
      self.assertGreaterEqual(stats["modules_transformed"], 1)
    finally:
      maot.unregister_transform(platforms="cpu")

  def test_r5_amax_no_combine_and_f2_early_dispatch_exact_parity(self):
    common = dict(
        emb_dim=256,
        moe_mlp_dim=128,
        num_query_heads=8,
        gdn_num_v_heads=8,
        mu_dtype="float32",
        schedule_mode="batchsplit",
        enable_remat=True,
        hbm_save_offload_targets=True,
        moe_fp8_weights=True,
        moe_fp8_gmm_compute=True,
        moe_fp8_token_dispatch=True,
        moe_fp8_bwd_dispatch=True,
        moe_use_ragged_silu_3d_fp8=True,
        moe_use_forked_3d_gmmv2=True,
        moe_bwd_dy_fp8_e5m2=True,
        moe_local_65key_sort=True,
        moe_use_tc_ragged=True,
        moe_3d_ep_tokens=True,
        strict_f2_before_f3=True,
        defer_mb1_combine_after_f1=True,
        precompute_rope_tables=True,
        fsdp_pipeline_mode="full_prefetch_defer_rs",
        ep_combine_fwd_unpinned=False,
        sa_qk_seq_minor=True,
        moe_bwd_tie_wgrad_before_gate=True,
        f2_light_router_ties=True,
        save_moe_routing=True,
        save_moe_gate_up_sublayers=(1, 2, 3),
    )
    cfg_ref = qwen35_jax.tiny_config(fp8_amax_no_combine=False, f2_early_mb0_dispatch=False, **common)
    cfg_r5 = qwen35_jax.tiny_config(fp8_amax_no_combine=True, f2_early_mb0_dispatch=True, **common)
    mesh = qwen35_jax.create_mesh(cfg_ref, aot_topology=None)
    params = qwen35_jax.init_random_params(cfg_ref, mesh, jax.random.PRNGKey(53), param_dtype=jnp.bfloat16)
    batch = _make_synthetic_batch(cfg_ref, mesh)
    l_ref, g_ref = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_ref, mesh))(params, *batch)
    l_r5, g_r5 = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_r5, mesh))(params, *batch)
    self.assertEqual(float(jnp.max(jnp.abs(l_ref - l_r5))), 0.0)
    for (path, ga), (_, gb) in zip(jax.tree.leaves_with_path(g_ref), jax.tree.leaves_with_path(g_r5)):
      diff = float(np.max(np.abs(np.asarray(ga, np.float32) - np.asarray(gb, np.float32))))
      self.assertEqual(
          diff,
          0.0,
          f"Parity mismatch on {jax.tree_util.keystr(path)}: {diff}",
      )

  def test_r6_copy_free_ties_2d_qkv_bwd_overlap_and_sb_prefetch_parity(self):
    from maxtext.experimental.qwen_35_64c import max_async_overlap_transform as maot  # pylint: disable=g-import-not-at-top

    # 1. Verify `order_mode="done_first"` eliminates FIFO capping of early-ready/late-deadline
    # async pairs (`early_start -> early_done`) behind mid-ready/mid-deadline pairs on `sparsecore:1`.
    fifo_order = [
        "p0",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "barrier_mid",
        "c6",
        "c7",
        "early_start",
        "mid_start",
        "mid_done",
        "mid_consumer",
        "c8",
        "early_done",
        "late_consumer",
        "root",
    ]
    fifo_opcodes = {
        "p0": "kParameter",
        "c1": "kCustomCall",
        "c2": "kCustomCall",
        "c3": "kCustomCall",
        "c4": "kCustomCall",
        "c5": "kCustomCall",
        "barrier_mid": "kOptimizationBarrier",
        "c6": "kCustomCall",
        "c7": "kCustomCall",
        "early_start": "kAsyncStart",
        "mid_start": "kAsyncStart",
        "mid_done": "kAsyncDone",
        "mid_consumer": "kCustomCall",
        "c8": "kCustomCall",
        "early_done": "kAsyncDone",
        "late_consumer": "kCustomCall",
        "root": "kTuple",
    }
    fifo_operands = {
        "p0": [],
        "c1": ["p0"],
        "c2": ["c1"],
        "c3": ["c2"],
        "c4": ["c3"],
        "c5": ["c4"],
        "barrier_mid": ["c5"],
        "c6": ["barrier_mid"],
        "c7": ["c6"],
        "early_start": ["p0"],
        "mid_start": ["barrier_mid"],
        "mid_done": ["mid_start"],
        "mid_consumer": ["c7", "mid_done"],
        "c8": ["mid_consumer"],
        "early_done": ["early_start"],
        "late_consumer": ["c8", "early_done"],
        "root": ["late_consumer"],
    }
    fifo_graph = maot.make_schedule_graph(
        order=fifo_order,
        opcodes=fifo_opcodes,
        operands=fifo_operands,
        queues={
            "early_start": "sparsecore:1",
            "mid_start": "sparsecore:1",
        },
    )
    res_sum = maot.reschedule(fifo_graph, queue_filter="sc_only", order_mode="sum")
    maot.validate_order(fifo_graph, res_sum.order)
    pos_sum = {n: i for i, n in enumerate(res_sum.order)}
    # Under `sum`, `early` (1+15=16) sorts before `mid` (7+12=19), capping `early_done` before `mid_done`!
    self.assertLess(pos_sum["early_start"], pos_sum["mid_start"])
    self.assertLess(pos_sum["early_done"], pos_sum["mid_done"])

    res_df = maot.reschedule(fifo_graph, queue_filter="sc_only", order_mode="done_first")
    maot.validate_order(fifo_graph, res_df.order)
    pos_df = {n: i for i, n in enumerate(res_df.order)}
    # Under `done_first`, `mid` (done=12) precedes `early` (done=15), keeping `early_done` uncapped!
    self.assertLess(pos_df["mid_start"], pos_df["early_start"])
    self.assertLess(pos_df["mid_done"], pos_df["mid_consumer"])
    self.assertGreater(pos_df["early_done"], pos_df["c8"])
    self.assertLess(pos_df["early_done"], pos_df["late_consumer"])

    # 2. Verify numerical parity across 2D QKV matmul, copy-free scalar ties,
    # GDN norm layouts, `bwd_tie_b1_before_prev_b4`, and `bwd_prefetch_prev_attn3_moe_weights`.
    common = dict(
        emb_dim=256,
        moe_mlp_dim=128,
        num_query_heads=8,
        gdn_num_v_heads=8,
        mu_dtype="float32",
        schedule_mode="batchsplit",
        enable_remat=True,
        hbm_save_offload_targets=True,
        moe_fp8_weights=True,
        moe_fp8_gmm_compute=True,
        moe_fp8_token_dispatch=True,
        moe_fp8_bwd_dispatch=True,
        moe_use_ragged_silu_3d_fp8=True,
        moe_use_forked_3d_gmmv2=True,
        moe_bwd_dy_fp8_e5m2=True,
        moe_local_65key_sort=True,
        moe_use_tc_ragged=True,
        moe_3d_ep_tokens=True,
        strict_f2_before_f3=True,
        defer_mb1_combine_after_f1=True,
        precompute_rope_tables=True,
        fsdp_pipeline_mode="full_prefetch_defer_rs",
        ep_combine_fwd_unpinned=False,
        sa_qk_seq_minor=True,
        moe_bwd_tie_wgrad_before_gate=True,
        f2_light_router_ties=True,
        fp8_amax_no_combine=True,
        f2_early_mb0_dispatch=True,
        save_moe_routing=True,
        save_moe_gate_up_sublayers=(1, 2, 3),
    )
    cfg_r5 = qwen35_jax.tiny_config(**common)
    mesh = qwen35_jax.create_mesh(cfg_r5, aot_topology=None)
    params = qwen35_jax.init_random_params(cfg_r5, mesh, jax.random.PRNGKey(61), param_dtype=jnp.bfloat16)
    batch = _make_synthetic_batch(cfg_r5, mesh)
    l_r5, g_r5 = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_r5, mesh))(params, *batch)

    # 2a. Exact `0.0` parity for pure schedule/tie/prefetch changes (`bwd_early_mb1_dy_pmax`
    # + `bwd_tie_b1_before_prev_b4` + `bwd_prefetch_prev_attn3_moe_weights` + `moe_gmm_no_alias_out`).
    cfg_sched = qwen35_jax.tiny_config(
        moe_gmm_no_alias_out=True,
        bwd_early_mb1_dy_pmax=True,
        bwd_tie_b1_before_prev_b4=True,
        bwd_prefetch_prev_attn3_moe_weights=True,
        **common,
    )
    l_sched, g_sched = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_sched, mesh))(params, *batch)
    self.assertEqual(float(jnp.max(jnp.abs(l_r5 - l_sched))), 0.0)
    for (path, ga), (_, gb) in zip(jax.tree.leaves_with_path(g_r5), jax.tree.leaves_with_path(g_sched)):
      diff = float(np.max(np.abs(np.asarray(ga, np.float32) - np.asarray(gb, np.float32))))
      self.assertEqual(
          diff,
          0.0,
          "Schedule/prefetch parity mismatch on" f" {jax.tree_util.keystr(path)}: {diff}",
      )

    # 2b. Verify 2-step `make_train_step(donate_state=True)` AdamW execution with `bwd_prefetch_prev_attn3_moe_weights=True`.
    params_step = jax.tree.map(jnp.copy, params)
    tx = qwen35_jax.create_optimizer(cfg_sched)
    param_shardings = jax.tree.map(lambda x: x.sharding, params_step)
    opt_state = jax.jit(tx.init, in_shardings=(param_shardings,))(params_step)
    step_fn = qwen35_jax.make_train_step(cfg_sched, mesh, donate_state=True)
    params_step, opt_state, loss1 = step_fn(params_step, opt_state, *batch)
    params_step, opt_state, loss2 = step_fn(params_step, opt_state, *batch)
    self.assertTrue(bool(jnp.isfinite(loss1)))
    self.assertTrue(bool(jnp.isfinite(loss2)))

    # 2c. Exact `0.0` bitwise parity in bf16 for layout-only changes (`attn_gate_bf16_align` +
    # `gdn_norm_layout_mode`), and tight float32 equivalence for `attn_qkv_2d_matmul=True`
    # (avoiding bf16 Top-K router tie-flip discontinuities on random tiny models).
    for gdn_mode in ("bf16_z_token_minor", "natural"):
      cfg_lay = qwen35_jax.tiny_config(
          moe_gmm_no_alias_out=True,
          attn_gate_bf16_align=True,
          gdn_norm_layout_mode=gdn_mode,
          bwd_tie_b1_before_prev_b4=True,
          bwd_prefetch_prev_attn3_moe_weights=True,
          **common,
      )
      l_lay, g_lay = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_lay, mesh))(params, *batch)
      self.assertEqual(float(jnp.max(jnp.abs(l_r5 - l_lay))), 0.0)
      for (path, ga), (_, gb) in zip(jax.tree.leaves_with_path(g_r5), jax.tree.leaves_with_path(g_lay)):
        diff = float(np.max(np.abs(np.asarray(ga, np.float32) - np.asarray(gb, np.float32))))
        self.assertEqual(
            diff,
            0.0,
            f"{gdn_mode} layout parity mismatch on" f" {jax.tree_util.keystr(path)}: {diff}",
        )

    params_f32 = qwen35_jax.init_random_params(cfg_r5, mesh, jax.random.PRNGKey(61), param_dtype=jnp.float32)
    cfg_r5_f32 = dataclasses.replace(
        cfg_r5,
        moe_fp8_bwd_dispatch=False,
        moe_bwd_dy_fp8_e5m2=False,
        save_moe_gate_up_sublayers=(),
    )
    cfg_qkv2d_f32 = dataclasses.replace(
        cfg_r5_f32,
        attn_qkv_2d_matmul=True,
        attn_gate_bf16_align=True,
        gdn_norm_layout_mode="bf16_z_token_minor",
        moe_gmm_no_alias_out=True,
        bwd_early_mb1_dy_pmax=True,
        bwd_tie_b1_before_prev_b4=True,
    )
    l_f32_ref, g_f32_ref = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_r5_f32, mesh))(params_f32, *batch)
    l_f32_2d, g_f32_2d = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_qkv2d_f32, mesh))(params_f32, *batch)
    self.assertLess(float(jnp.max(jnp.abs(l_f32_ref - l_f32_2d))), 1e-4)
    for (path, ga), (_, gb) in zip(
        jax.tree.leaves_with_path(g_f32_ref),
        jax.tree.leaves_with_path(g_f32_2d),
    ):
      a32 = np.asarray(ga, np.float32)
      b32 = np.asarray(gb, np.float32)
      rel = float(np.max(np.abs(a32 - b32)) / max(float(np.max(np.abs(a32))), 1e-4))
      self.assertLess(
          rel,
          1e-3,
          "attn_qkv_2d_matmul f32 gradient drift on" f" {jax.tree_util.keystr(path)}: {rel}",
      )

  def test_ep16_radix_packed_stable_sort_and_sharding_parity(self):
    import warnings  # pylint: disable=g-import-not-at-top
    from maxtext.experimental.qwen_35_64c.ext_kernels.ragged import ragged_gather_tc as tc  # pylint: disable=g-import-not-at-top

    # 1. Verify arithmetic-radix uint32 packed_stable_sort at EP=16 shapes
    #    (n=43008, max_key=65536 -> (65537)*43008 = 2,818,615,296 <= 2**32)
    #    stays on the single-array uint32 path (no fallback warning) and is bit-exact!
    rng = np.random.default_rng(2026)
    key_np = rng.integers(0, 65537, size=(43008,), dtype=np.int32)
    key_np[0] = 65536
    key_np[-1] = 65536
    key_np[100:200] = 0
    key_jax = jnp.asarray(key_np, dtype=jnp.int32)
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      sorted_keys_fast, perm_fast = jax.jit(lambda k: tc.packed_stable_sort(k, 65536))(key_jax)
    fallback_warns = [w for w in caught if "packed_stable_sort" in str(w.message)]
    self.assertEqual(len(fallback_warns), 0, f"Unexpected fallback warning: {fallback_warns}")
    perm_ref = jnp.argsort(key_jax, stable=True)
    np.testing.assert_array_equal(np.asarray(perm_fast), np.asarray(perm_ref))
    np.testing.assert_array_equal(np.asarray(sorted_keys_fast), np.asarray(key_jax[perm_ref]))

    # 2. Verify fsdp_size=2, ep_size=4 (8 devices, x1e_yx_core) train_step & tile config parity.
    cfg_ep = qwen35_jax.tiny_config(
        fsdp_size=2,
        ep_size=4,
        mesh_mode="x1e_yx_core",
        schedule_mode="batchsplit",
        moe_gmm_no_alias_out=True,
        bwd_early_mb1_dy_pmax=True,
        gate_fwd_tiles=(512, 1024, 4096, 256),
        lbwd_wgrad_tiles=(512, 256, 2048),
        gbwd_fused_tiles=(512, 256, 1024),
    )
    mesh = qwen35_jax.create_mesh(cfg_ep, aot_topology=None)
    self.assertEqual(mesh.shape["fsdp"], 2)
    self.assertEqual(mesh.shape["expert"], 4)
    params = qwen35_jax.init_random_params(cfg_ep, mesh, jax.random.PRNGKey(77), param_dtype=jnp.bfloat16)
    batch = _make_synthetic_batch(cfg_ep, mesh)
    tx = qwen35_jax.create_optimizer(cfg_ep)
    param_shardings = jax.tree.map(lambda x: x.sharding, params)
    opt_state = jax.jit(tx.init, in_shardings=(param_shardings,))(params)
    step_fn = qwen35_jax.make_train_step(cfg_ep, mesh, donate_state=True)
    params, opt_state, loss = step_fn(params, opt_state, *batch)
    self.assertTrue(bool(jnp.isfinite(loss)))

    # 3. Verify 3-axis decoupled EP D2D mesh ('fsdp', 'ep_d2d', 'expert') = (2, 2, 2)
    #    exact fp32 forward + backward numerical parity against 2-axis ('fsdp', 'expert') = (2, 4)!
    cfg_2ax = qwen35_jax.tiny_config(
        fsdp_size=2,
        ep_size=4,
        decouple_ep_d2d=False,
        mesh_mode="x1e_yx_core",
        schedule_mode="batchsplit",
        mu_dtype="float32",
        moe_use_tc_ragged=True,
        moe_3d_ep_tokens=True,
    )
    cfg_3ax = qwen35_jax.tiny_config(
        fsdp_size=2,
        ep_size=4,
        decouple_ep_d2d=True,
        mesh_mode="x1e_yx_core",
        schedule_mode="batchsplit",
        mu_dtype="float32",
        moe_use_tc_ragged=True,
        moe_3d_ep_tokens=True,
    )
    mesh_2ax = qwen35_jax.create_mesh(cfg_2ax, aot_topology=None)
    mesh_3ax = qwen35_jax.create_mesh(cfg_3ax, aot_topology=None)
    self.assertEqual(dict(mesh_3ax.shape), {"fsdp": 2, "ep_d2d": 2, "expert": 2})

    params_2ax = qwen35_jax.init_random_params(cfg_2ax, mesh_2ax, jax.random.PRNGKey(99), param_dtype=jnp.float32)
    pspecs_3ax = qwen35_jax.get_param_pspecs(cfg_3ax)
    params_3ax = jax.tree.map(
        lambda x, ps: jax.device_put(np.asarray(x), NamedSharding(mesh_3ax, ps)),
        params_2ax,
        pspecs_3ax,
    )
    batch_2ax = _make_synthetic_batch(cfg_2ax, mesh_2ax)
    batch_3ax = _make_synthetic_batch(cfg_3ax, mesh_3ax)

    loss_2ax, grads_2ax = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_2ax, mesh_2ax))(params_2ax, *batch_2ax)
    loss_3ax, grads_3ax = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_3ax, mesh_3ax))(params_3ax, *batch_3ax)
    np.testing.assert_allclose(np.asarray(loss_3ax), np.asarray(loss_2ax), rtol=1e-5, atol=1e-5)
    for (path, g2), (_, g3) in zip(
        jax.tree.leaves_with_path(grads_2ax),
        jax.tree.leaves_with_path(grads_3ax),
    ):
      np.testing.assert_allclose(
          np.asarray(g3),
          np.asarray(g2),
          rtol=2e-5,
          atol=2e-5,
          err_msg=("decouple_ep_d2d gradient mismatch at" f" {jax.tree_util.keystr(path)}"),
      )

  def test_gdn_cp8_prefix_scans_and_saved_cp_state_parity(self):
    layers_lib = qwen35_jax.layers
    model_lib = qwen35_jax.model

    default_cfg = qwen35_jax.full_config()
    self.assertTrue(default_cfg.save_gdn_cp_state)
    self.assertTrue(default_cfg.gdn_cp_m_scan)
    self.assertTrue(default_cfg.gdn_cp_dh_scan)
    self.assertTrue(default_cfg.gdn_cp_matmul_high)

    base_kw = dict(
        fsdp_size=1,
        ep_size=8,
        global_batch_size=1,
        seq_len=1024,  # 128 tokens = 2 chunks of 64 per CP rank across 8 ranks
        emb_dim=256,
        gdn_num_k_heads=2,
        gdn_num_v_heads=4,
        gdn_head_k_dim=128,
        gdn_head_v_dim=128,
        gdn_conv_kernel_size=4,
        gdn_chunk_size=64,
        enable_remat=True,
    )
    cfg_pure = qwen35_jax.tiny_config(use_pallas_kernels=False, **base_kw)
    mesh = qwen35_jax.create_mesh(cfg_pure, aot_topology=None)
    self.assertEqual(cfg_pure.cp_size, 8)

    kx, kct, kw1, kw2, kw3, kw4 = jax.random.split(jax.random.PRNGKey(2028), 6)
    x_np = np.asarray(jax.random.normal(kx, (1, 1024, 256), dtype=jnp.bfloat16) * 0.25)
    ct_np = np.asarray(jax.random.normal(kct, (1, 1024, 256), dtype=jnp.bfloat16) * 0.25)
    seg_np = np.concatenate(
        [
            np.ones((300,), np.int32),
            np.full((400,), 2, np.int32),
            np.full((290,), 3, np.int32),
            np.zeros((34,), np.int32),
        ]
    )[None, :]

    dummy_moe = layers_lib.GatheredMoeWeights(
        router_gate=jnp.zeros((1,), dtype=jnp.float32),
        wi=jnp.zeros((1,), dtype=jnp.float32),
        wo=jnp.zeros((1,), dtype=jnp.float32),
        shared_wi_0=jnp.zeros((1,), dtype=jnp.float32),
        shared_wi_1=jnp.zeros((1,), dtype=jnp.float32),
        shared_wo=jnp.zeros((1,), dtype=jnp.float32),
        shared_gate=jnp.zeros((1,), dtype=jnp.float32),
    )
    w_gdn = layers_lib.GatheredGdnWeights(
        pre_attn_norm=jnp.zeros((256,), dtype=jnp.float32),
        post_attn_norm=jnp.zeros((256,), dtype=jnp.float32),
        in_proj_qkvz=(jax.random.normal(kw1, (256, cfg_pure.gdn_qkvz_dim), dtype=jnp.bfloat16) * 0.05),
        in_proj_ba=(jax.random.normal(kw2, (256, 2 * cfg_pure.gdn_num_v_heads), dtype=jnp.bfloat16) * 0.05),
        conv1d=(jax.random.normal(kw3, (4, 1, cfg_pure.gdn_qkv_dim), dtype=jnp.bfloat16) * 0.1),
        A_log=jnp.linspace(-2.0, -0.5, cfg_pure.gdn_num_v_heads, dtype=jnp.float32),
        dt_bias=jnp.zeros((cfg_pure.gdn_num_v_heads,), dtype=jnp.float32),
        gdn_norm=jnp.ones((cfg_pure.gdn_head_v_dim,), dtype=jnp.float32),
        out_proj=(jax.random.normal(kw4, (cfg_pure.gdn_value_dim, 256), dtype=jnp.bfloat16) * 0.05),
        moe=dummy_moe,
    )

    act_spec = P(None, cfg_pure.cp_axis_names, None)
    seg_spec = P(None, cfg_pure.cp_axis_names)
    w_spec = jax.tree.map(lambda _: P(), w_gdn)
    x_s = jax.device_put(x_np, NamedSharding(mesh, act_spec))
    ct_s = jax.device_put(ct_np, NamedSharding(mesh, act_spec))
    seg_s = jax.device_put(seg_np, NamedSharding(mesh, seg_spec))
    w_s = jax.tree.map(lambda a: jax.device_put(a, NamedSharding(mesh, P())), w_gdn)

    def _make_mixer_vg(cfg_v: qwen35_jax.Qwen35Config, use_ckpt: bool = False):
      policy = model_lib.build_remat_policy(cfg_v)

      def _sharded(w_l, x_l, seg_l, ct_l):
        def _loss_fn(w_in, x_in):
          if use_ckpt:
            run_fn = jax.checkpoint(
                lambda ww, xx: layers_lib.gdn_mixer(xx, ww, seg_l, cfg_v),
                policy=policy,
            )
            y = run_fn(w_in, x_in)
          else:
            y = layers_lib.gdn_mixer(x_in, w_in, seg_l, cfg_v)
          local_loss = jnp.sum(y.astype(jnp.float32) * ct_l.astype(jnp.float32))
          return local_loss, y

        (l_loc, y_loc), (gw_loc, gx_loc) = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)(w_l, x_l)
        gw_sum = jax.tree.map(
            lambda g: jax.lax.psum(g.astype(jnp.float32), cfg_v.cp_axis_names),
            gw_loc,
        )
        l_sum = jax.lax.psum(l_loc, cfg_v.cp_axis_names)
        return l_sum, y_loc, gw_sum, gx_loc

      return jax.jit(
          jax.shard_map(
              _sharded,
              mesh=mesh,
              in_specs=(w_spec, act_spec, seg_spec, act_spec),
              out_specs=(P(), act_spec, w_spec, act_spec),
              check_vma=False,
          )
      )

    cfg_pbase = qwen35_jax.tiny_config(use_pallas_kernels=True, **base_kw)
    l_pbase, y_pbase, gw_pbase, gx_pbase = _make_mixer_vg(cfg_pbase, use_ckpt=True)(w_s, x_s, seg_s, ct_s)

    cfg_opt = qwen35_jax.tiny_config(
        use_pallas_kernels=True,
        save_gdn_cp_state=True,
        gdn_cp_m_scan=True,
        gdn_cp_dh_scan=True,
        gdn_cp_matmul_high=True,
        **base_kw,
    )
    l_opt, y_opt, gw_opt, gx_opt = _make_mixer_vg(cfg_opt, use_ckpt=True)(w_s, x_s, seg_s, ct_s)
    self.assertLess(
        abs(float(l_opt) - float(l_pbase)) / (abs(float(l_pbase)) + 1e-6),
        2e-3,
    )
    y_rel = float(
        np.linalg.norm(np.asarray(y_opt, np.float32) - np.asarray(y_pbase, np.float32))
        / (np.linalg.norm(np.asarray(y_pbase, np.float32)) + 1e-6)
    )
    self.assertLess(y_rel, 2e-3)
    gx_rel = float(
        np.linalg.norm(np.asarray(gx_opt, np.float32) - np.asarray(gx_pbase, np.float32))
        / (np.linalg.norm(np.asarray(gx_pbase, np.float32)) + 1e-6)
    )
    self.assertLess(gx_rel, 5e-3)
    for (path, ga), (_, gb) in zip(
        jax.tree.leaves_with_path(gw_pbase),
        jax.tree.leaves_with_path(gw_opt),
    ):
      a32 = np.asarray(ga, np.float32)
      b32 = np.asarray(gb, np.float32)
      rel = float(np.linalg.norm(a32 - b32) / max(float(np.linalg.norm(a32)), 1e-6))
      self.assertLess(rel, 5e-3, f"grad mismatch on {jax.tree_util.keystr(path)}")

    cfg_no_save = dataclasses.replace(cfg_opt, save_gdn_cp_state=False)
    hlo_no_save = _make_mixer_vg(cfg_no_save, use_ckpt=True).lower(w_s, x_s, seg_s, ct_s).as_text("hlo")
    hlo_save = _make_mixer_vg(cfg_opt, use_ckpt=True).lower(w_s, x_s, seg_s, ct_s).as_text("hlo")
    self.assertLessEqual(
        hlo_save.count("collective-permute("),
        hlo_no_save.count("collective-permute(") - 6,
    )

  def test_tokamax_splash_and_gqa_fold_parity(self):
    layers_lib = qwen35_jax.layers
    tokamax_splash = layers_lib.tokamax_splash
    tokamax_splash_mask = layers_lib.tokamax_splash_mask

    default_cfg = qwen35_jax.full_config()
    self.assertTrue(default_cfg.use_tokamax_splash)
    self.assertTrue(default_cfg.sa_fold_gqa_into_seq)
    self.assertFalse(default_cfg.sa_qk_seq_minor)
    self.assertTrue(default_cfg.experimental_sa_quant_q_fp8)
    self.assertTrue(default_cfg.experimental_sa_quant_k_fp8)
    self.assertTrue(default_cfg.sa_q_prescaled_log2e)
    self.assertTrue(default_cfg.sa_skip_uniform_segment_mask)
    self.assertTrue(default_cfg.sa_cast_s_to_v_dtype)
    self.assertEqual(default_cfg.sa_dq_reduction_steps, 3)
    self.assertEqual(
        (
            default_cfg.sa_block_q,
            default_cfg.sa_block_kv,
            default_cfg.sa_block_kv_compute,
            default_cfg.sa_block_q_dkv,
            default_cfg.sa_block_kv_dkv,
            default_cfg.sa_block_kv_dkv_compute,
        ),
        (2048, 2048, 1024, 1024, 4096, 512),
    )
    with self.assertRaisesRegex(ValueError, "sa_dq_reduction_steps"):
      qwen35_jax.tiny_config(sa_dq_reduction_steps=2)
    with self.assertRaisesRegex(ValueError, "divisible by num_kv_heads"):
      qwen35_jax.tiny_config(num_query_heads=5, num_kv_heads=2, sa_fold_gqa_into_seq=True)

    seq_len, cp_size, gqa_ratio = 512, 4, 4
    causal_folded = tokamax_splash_mask.CausalMask(shape=(seq_len * gqa_ratio, seq_len))
    causal_folded.q_sequence = np.repeat(layers_lib._lb_q_sequence(seq_len, cp_size), gqa_ratio)
    for dq_steps in (None, 3):
      sa_cfg = tokamax_splash.SplashConfig(
          block_q=256,
          block_kv=128,
          block_kv_compute=128,
          block_q_dkv=256,
          block_kv_dkv=128,
          block_kv_dkv_compute=128,
          use_fused_bwd_kernel=True,
          dq_reduction_steps=dq_steps,
      )
      ref_k = tokamax_splash.make_splash_mha(causal_folded, config=sa_cfg, q_seq_shards=cp_size)
      fast_k = layers_lib._make_fast_causal_tokamax_mha(causal_folded, config=sa_cfg, q_seq_shards=cp_size)
      self.assertEqual(fast_k.kwargs["fwd_mask_sparsity"], ref_k.kwargs["fwd_mask_sparsity"])
      self.assertEqual(fast_k.kwargs["dkv_mask_sparsity"], ref_k.kwargs["dkv_mask_sparsity"])
      for info_r, info_f in (
          (ref_k.fwd_mask_info, fast_k.fwd_mask_info),
          (ref_k.dkv_mask_info, fast_k.dkv_mask_info),
      ):
        for field in info_r._fields:
          r_v = getattr(info_r, field)
          f_v = getattr(info_f, field)
          if r_v is None:
            self.assertIsNone(f_v)
          else:
            np.testing.assert_array_equal(np.asarray(f_v), np.asarray(r_v))

    common_gqa = dict(
        emb_dim=128,
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=32,
        global_batch_size=2,
        seq_len=128,
        fsdp_size=2,
        ep_size=4,
        decouple_ep_d2d=True,
        use_pallas_kernels=False,
    )
    cfg_ref = qwen35_jax.tiny_config(sa_fold_gqa_into_seq=False, **common_gqa)
    cfg_fold = qwen35_jax.tiny_config(sa_fold_gqa_into_seq=True, **common_gqa)
    mesh_g = qwen35_jax.create_mesh(cfg_ref, aot_topology=None)
    params_g = qwen35_jax.init_random_params(cfg_ref, mesh_g, jax.random.PRNGKey(314), param_dtype=jnp.float32)
    act_pspec = P("fsdp", cfg_ref.cp_axis_names, None)
    data_pspec = qwen35_jax.data_pspec(cfg_ref)
    seg_np = np.zeros((2, 128), dtype=np.int32)
    pos_np = np.zeros((2, 128), dtype=np.int32)
    for b_i, spans in enumerate([((0, 35, 1), (35, 78, 2), (78, 118, 3)), ((0, 57, 1), (57, 122, 2))]):
      for s0, s1, sid in spans:
        seg_np[b_i, s0:s1] = sid
        pos_np[b_i, s0:s1] = np.arange(s1 - s0, dtype=np.int32)
    positions = jax.device_put(pos_np, NamedSharding(mesh_g, data_pspec))
    segment_ids = jax.device_put(seg_np, NamedSharding(mesh_g, data_pspec))

    kx, kct = jax.random.split(jax.random.PRNGKey(99))
    x_in = jax.device_put(
        jax.random.normal(kx, (2, 128, 128), dtype=jnp.float32),
        NamedSharding(mesh_g, act_pspec),
    )
    valid_mask = (seg_np > 0)[..., None].astype(np.float32)
    ct_in = jax.device_put(
        np.asarray(jax.random.normal(kct, (2, 128, 128), dtype=jnp.float32)) * valid_mask,
        NamedSharding(mesh_g, act_pspec),
    )

    def _eval_cfg(cfg_c):
      def _step(p_l, x_l, pos_l, seg_l, ct_l):
        sub_a = jax.tree.map(lambda z: z[0], p_l.blocks.attn_3)

        def _loss_fn(sub_p, x_v):
          w = layers_lib.gather_attn_sublayer_weights(sub_p, None, None, cfg_c)
          pos_lb = layers_lib.cp_natural_to_lb(pos_l, cfg_c)
          y = layers_lib.full_attn_mixer(x_v, w, pos_lb, seg_l, cfg_c)
          y_valid = y * (seg_l > 0)[..., None].astype(y.dtype)
          return jnp.sum(y_valid * ct_l), y_valid

        (l_v, y_l), (gp, gx) = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)(sub_a, x_l)
        return (
            jax.lax.psum(l_v, axis_name=cfg_c.all_sharded_axes),
            y_l,
            gp.q_proj,
            gp.k_proj,
            gp.v_proj,
            gx,
        )

      p_specs = qwen35_jax.get_param_pspecs(cfg_c)
      sub_specs = p_specs.blocks.attn_3
      return jax.jit(
          jax.shard_map(
              _step,
              mesh=mesh_g,
              in_specs=(p_specs, act_pspec, data_pspec, data_pspec, act_pspec),
              out_specs=(
                  P(),
                  act_pspec,
                  P(*sub_specs.q_proj[1:]),
                  P(*sub_specs.k_proj[1:]),
                  P(*sub_specs.v_proj[1:]),
                  act_pspec,
              ),
              check_vma=False,
          )
      )(params_g, x_in, positions, segment_ids, ct_in)

    for name, a_ref, a_cand in zip(
        ("loss", "y", "dq_proj", "dk_proj", "dv_proj", "dx"),
        _eval_cfg(cfg_ref),
        _eval_cfg(cfg_fold),
        strict=True,
    ):
      r_np = np.asarray(a_ref, dtype=np.float64)
      c_np = np.asarray(a_cand, dtype=np.float64)
      rel_vs_ref = float(np.linalg.norm(c_np - r_np) / (np.linalg.norm(r_np) + 1e-12))
      self.assertLess(rel_vs_ref, 1e-6, f"GQA fold vs ref {name}={rel_vs_ref}")

  def test_tokamax_splash_fp8_qk_and_prescaled_log2e(self):
    layers_lib = qwen35_jax.layers

    with self.assertRaisesRegex(ValueError, "use_tokamax_splash=True"):
      qwen35_jax.tiny_config(use_tokamax_splash=False, experimental_sa_quant_q_fp8=True)
    with self.assertRaisesRegex(ValueError, "use_tokamax_splash=True"):
      qwen35_jax.tiny_config(use_tokamax_splash=False, experimental_sa_quant_k_fp8=True)
    with self.assertRaisesRegex(ValueError, "use_tokamax_splash=True"):
      qwen35_jax.tiny_config(use_tokamax_splash=False, sa_skip_uniform_segment_mask=True)
    with self.assertRaisesRegex(ValueError, "use_tokamax_splash=True"):
      qwen35_jax.tiny_config(use_tokamax_splash=False, sa_cast_s_to_v_dtype=True)
    with self.assertRaisesRegex(ValueError, "sa_use_base2_exp=True"):
      qwen35_jax.tiny_config(
          use_tokamax_splash=True,
          sa_use_base2_exp=False,
          sa_q_prescaled_log2e=True,
      )

    common = dict(
        emb_dim=128,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=128,
        global_batch_size=2,
        seq_len=512,
        fsdp_size=2,
        ep_size=4,
        decouple_ep_d2d=True,
        use_pallas_kernels=False,
        use_tokamax_splash=True,
        sa_interpret=True,
        sa_use_base2_exp=True,
        sa_block_q=128,
        sa_block_kv=128,
        sa_block_kv_compute=128,
        sa_block_q_dkv=128,
        sa_block_kv_dkv=128,
        sa_block_kv_dkv_compute=128,
    )
    cfg_bf16 = qwen35_jax.tiny_config(
        sa_fold_gqa_into_seq=True,
        experimental_sa_quant_q_fp8=False,
        experimental_sa_quant_k_fp8=False,
        sa_q_prescaled_log2e=False,
        **common,
    )
    mesh = qwen35_jax.create_mesh(cfg_bf16, aot_topology=None)
    params = qwen35_jax.init_random_params(cfg_bf16, mesh, jax.random.PRNGKey(2718), param_dtype=jnp.bfloat16)
    act_pspec = P("fsdp", cfg_bf16.cp_axis_names, None)
    data_pspec = qwen35_jax.data_pspec(cfg_bf16)
    seg_np = np.zeros((2, 512), dtype=np.int32)
    pos_np = np.zeros((2, 512), dtype=np.int32)
    for b_i, spans in enumerate(
        [
            ((0, 140, 1), (140, 310, 2), (310, 490, 3)),
            ((0, 220, 1), (220, 500, 2)),
        ]
    ):
      for s0, s1, sid in spans:
        seg_np[b_i, s0:s1] = sid
        pos_np[b_i, s0:s1] = np.arange(s1 - s0, dtype=np.int32)
    positions = jax.device_put(pos_np, NamedSharding(mesh, data_pspec))
    segment_ids = jax.device_put(seg_np, NamedSharding(mesh, data_pspec))

    kx, kct = jax.random.split(jax.random.PRNGKey(1618))
    x_in = jax.device_put(
        jax.random.normal(kx, (2, 512, 128), dtype=jnp.bfloat16),
        NamedSharding(mesh, act_pspec),
    )
    valid_mask = (seg_np > 0)[..., None].astype(np.float32)
    ct_in = jax.device_put(
        (np.asarray(jax.random.normal(kct, (2, 512, 128), dtype=jnp.float32)) * valid_mask).astype(jnp.bfloat16),
        NamedSharding(mesh, act_pspec),
    )

    def _build_fn(cfg_c):
      def _step(p_l, x_l, pos_l, seg_l, ct_l):
        sub_a = jax.tree.map(lambda z: z[0], p_l.blocks.attn_3)

        def _loss_fn(sub_p, x_v):
          w = layers_lib.gather_attn_sublayer_weights(sub_p, None, None, cfg_c)
          pos_lb = layers_lib.cp_natural_to_lb(pos_l, cfg_c)
          y = layers_lib.full_attn_mixer(x_v, w, pos_lb, seg_l, cfg_c)
          y_valid = y.astype(jnp.float32) * (seg_l > 0)[..., None].astype(jnp.float32)
          return jnp.sum(y_valid * ct_l.astype(jnp.float32)), y

        (l_v, y_l), (gp, gx) = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)(sub_a, x_l)
        return (
            jax.lax.psum(l_v, axis_name=cfg_c.all_sharded_axes),
            y_l,
            gp.q_proj,
            gp.k_proj,
            gp.v_proj,
            gx,
        )

      p_specs = qwen35_jax.get_param_pspecs(cfg_c)
      sub_specs = p_specs.blocks.attn_3
      return jax.jit(
          jax.shard_map(
              _step,
              mesh=mesh,
              in_specs=(p_specs, act_pspec, data_pspec, data_pspec, act_pspec),
              out_specs=(
                  P(),
                  act_pspec,
                  P(*sub_specs.q_proj[1:]),
                  P(*sub_specs.k_proj[1:]),
                  P(*sub_specs.v_proj[1:]),
                  act_pspec,
              ),
              check_vma=False,
          )
      )

    ref_outs = _build_fn(cfg_bf16)(params, x_in, positions, segment_ids, ct_in)

    # 1. Prescaled LOG2E in bf16 matches un-prescaled bf16 within bf16 rounding.
    cfg_bf16_prescaled = dataclasses.replace(cfg_bf16, sa_q_prescaled_log2e=True)
    prescaled_outs = _build_fn(cfg_bf16_prescaled)(params, x_in, positions, segment_ids, ct_in)
    for name, a_ref, a_cand in zip(
        ("loss", "y", "dq_proj", "dk_proj", "dv_proj", "dx"),
        ref_outs,
        prescaled_outs,
        strict=True,
    ):
      r_np = np.asarray(a_ref, dtype=np.float64)
      c_np = np.asarray(a_cand, dtype=np.float64)
      rel = float(np.linalg.norm(c_np - r_np) / (np.linalg.norm(r_np) + 1e-12))
      self.assertLess(rel, 1e-2, f"bf16 prescaled log2e {name}={rel}")

    # 1.5. skip_uniform_segment_mask=True is bit-exact on packed + 0-padded.
    cfg_skip_seg = dataclasses.replace(cfg_bf16, sa_skip_uniform_segment_mask=True)
    skip_seg_outs = _build_fn(cfg_skip_seg)(params, x_in, positions, segment_ids, ct_in)
    for name, a_ref, a_cand in zip(
        ("loss", "y", "dq_proj", "dk_proj", "dv_proj", "dx"),
        ref_outs,
        skip_seg_outs,
        strict=True,
    ):
      np.testing.assert_array_equal(
          np.asarray(a_cand),
          np.asarray(a_ref),
          err_msg=f"skip_uniform_segment_mask mismatch on {name}",
      )

    # 2. All Splash optimizations together (FP8 Q/K + prescaled LOG2E +
    # skip_uniform_segment_mask + cast_s_to_v_dtype + dq_reduction_steps=3 +
    # bf16 ds.T) match bf16 Splash reference (folded, KV-grouped, & unfolded).
    for fold_gqa, kv_grouped in ((True, True), (True, False), (False, False)):
      with self.subTest(sa_fold_gqa_into_seq=fold_gqa, attn_kv_grouped_layout=kv_grouped):
        cfg_fp8 = dataclasses.replace(
            cfg_bf16,
            sa_fold_gqa_into_seq=fold_gqa,
            attn_kv_grouped_layout=kv_grouped,
            experimental_sa_quant_q_fp8=True,
            experimental_sa_quant_k_fp8=True,
            sa_q_prescaled_log2e=True,
            sa_skip_uniform_segment_mask=True,
            sa_cast_s_to_v_dtype=True,
            sa_dq_reduction_steps=3,
        )
        fn_fp8 = _build_fn(cfg_fp8)
        shlo = fn_fp8.lower(params, x_in, positions, segment_ids, ct_in).as_text("stablehlo")
        self.assertIn("f8E4M3FN", shlo)
        fp8_outs = fn_fp8(params, x_in, positions, segment_ids, ct_in)
        for name, a_ref, a_cand in zip(
            ("loss", "y", "dq_proj", "dk_proj", "dv_proj", "dx"),
            ref_outs,
            fp8_outs,
            strict=True,
        ):
          r_np = np.asarray(a_ref, dtype=np.float64)
          c_np = np.asarray(a_cand, dtype=np.float64)
          self.assertTrue(bool(np.all(np.isfinite(c_np))), f"Non-finite {name}")
          self.assertGreater(float(np.linalg.norm(c_np)), 0.0, f"Zero {name}")
          rel = float(np.linalg.norm(c_np - r_np) / (np.linalg.norm(r_np) + 1e-12))
          self.assertLess(
              rel,
              8e-2,
              f"fp8 splash (fold={fold_gqa}, grouped={kv_grouped})" f" {name}={rel}",
          )

  def test_moe_router_dense_bwd_scatter_set_and_v7x_tile_defaults(self):
    default_cfg = qwen35_jax.full_config()
    self.assertTrue(default_cfg.moe_router_dense_bwd)
    self.assertTrue(default_cfg.moe_dw_scatter_set)
    self.assertEqual(default_cfg.gate_fwd_tiles, (256, 4096, 2048, 256))
    self.assertEqual(default_cfg.tc_gather_block_size, 3072)
    self.assertEqual(default_cfg.tc_reduce_block_size, 1792)

    cfg_exact = qwen35_jax.tiny_config(
        num_experts=32,
        num_experts_per_tok=4,
        moe_router_dense_bwd=False,
    )
    cfg_dense = qwen35_jax.tiny_config(
        num_experts=32,
        num_experts_per_tok=4,
        moe_router_dense_bwd=True,
    )
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(606), 3)
    normed_tok = jax.random.normal(k1, (2, 16, cfg_exact.emb_dim), dtype=jnp.bfloat16)
    router_gate = jax.random.normal(k2, (cfg_exact.emb_dim, cfg_exact.num_experts), dtype=jnp.bfloat16)
    upstream_cot = jax.random.normal(k3, (32, cfg_exact.num_experts_per_tok), dtype=jnp.float32)

    def _loss_fn(c):
      def _inner(nt, rg):
        w, ids = qwen35_jax.layers.moe_local_route_only(nt, rg, c)
        return jnp.sum(w * upstream_cot), ids

      return _inner

    (l_ref, ids_ref), (dnt_ref, drg_ref) = jax.jit(jax.value_and_grad(_loss_fn(cfg_exact), argnums=(0, 1), has_aux=True))(
        normed_tok, router_gate
    )
    (l_den, ids_den), (dnt_den, drg_den) = jax.jit(jax.value_and_grad(_loss_fn(cfg_dense), argnums=(0, 1), has_aux=True))(
        normed_tok, router_gate
    )

    self.assertEqual(float(jnp.max(jnp.abs(l_den - l_ref))), 0.0)
    np.testing.assert_array_equal(np.asarray(ids_den), np.asarray(ids_ref))
    np.testing.assert_array_equal(np.asarray(dnt_den), np.asarray(dnt_ref))
    np.testing.assert_array_equal(np.asarray(drg_den), np.asarray(drg_ref))

    hlo_dense = (
        jax.jit(jax.value_and_grad(_loss_fn(cfg_dense), argnums=(0, 1), has_aux=True))
        .lower(normed_tok, router_gate)
        .as_text("stablehlo")
    )
    self.assertNotIn("stablehlo.scatter", hlo_dense)

    base_kw = dict(
        fsdp_size=2,
        ep_size=4,
        decouple_ep_d2d=True,
        mesh_mode="x1e_yx_core",
        schedule_mode="batchsplit",
        enable_remat=True,
        moe_bwd_tie_wgrad_before_gate=True,
        f2_light_router_ties=True,
        save_moe_routing=True,
        save_attn_context_hbm=True,
        save_query_proj_hbm=True,
        save_out_proj_hbm=True,
        save_moe_gate_up_sublayers=(0, 1, 2, 3),
        hbm_save_offload_targets=True,
        mu_dtype="float32",
    )
    base_cfg = qwen35_jax.tiny_config(
        moe_router_dense_bwd=False,
        moe_dw_scatter_set=False,
        **base_kw,
    )
    opt_cfg = qwen35_jax.tiny_config(
        moe_router_dense_bwd=True,
        moe_dw_scatter_set=True,
        **base_kw,
    )
    mesh = qwen35_jax.create_mesh(base_cfg, aot_topology=None)
    params = qwen35_jax.init_random_params(base_cfg, mesh, jax.random.PRNGKey(607), param_dtype=jnp.bfloat16)
    batch = _make_synthetic_batch(base_cfg, mesh)

    loss_r, grads_r = jax.jit(qwen35_jax.make_value_and_grad_fn(base_cfg, mesh))(params, *batch)
    loss_o, grads_o = jax.jit(qwen35_jax.make_value_and_grad_fn(opt_cfg, mesh))(params, *batch)
    self.assertEqual(float(jnp.max(jnp.abs(loss_o - loss_r))), 0.0)
    for (path, gr), (_, go) in zip(
        jax.tree.leaves_with_path(grads_r),
        jax.tree.leaves_with_path(grads_o),
    ):
      np.testing.assert_array_equal(
          np.asarray(go),
          np.asarray(gr),
          err_msg=f"MoE opt mismatch at {jax.tree_util.keystr(path)}",
      )

    shlo_base = jax.jit(qwen35_jax.make_value_and_grad_fn(base_cfg, mesh)).lower(params, *batch).as_text("stablehlo")
    shlo_opt = jax.jit(qwen35_jax.make_value_and_grad_fn(opt_cfg, mesh)).lower(params, *batch).as_text("stablehlo")
    self.assertEqual(
        shlo_opt.count('call_target_name = "TopK"') * 2,
        shlo_base.count('call_target_name = "TopK"'),
    )

  def test_hbm_activation_caching_and_decoder_input_save_parity(self):
    default_cfg = qwen35_jax.full_config()
    self.assertTrue(default_cfg.save_decoder_input_hbm)
    self.assertFalse(default_cfg.offload_decoder_input)
    self.assertEqual(default_cfg.save_moe_gate_up_sublayers, (1, 3))
    self.assertEqual(default_cfg.offload_moe_gate_up_sublayers, (0, 2))

    base_kw = dict(
        fsdp_size=2,
        ep_size=4,
        decouple_ep_d2d=True,
        mesh_mode="x1e_yx_core",
        schedule_mode="batchsplit",
        enable_remat=True,
        moe_bwd_tie_wgrad_before_gate=True,
        f2_light_router_ties=True,
        save_moe_routing=True,
        save_attn_context_hbm=True,
        save_query_proj_hbm=True,
        save_out_proj_hbm=True,
        mu_dtype="float32",
    )
    ref_cfg = qwen35_jax.tiny_config(
        save_decoder_input_hbm=False,
        offload_decoder_input=True,
        save_moe_gate_up_sublayers=(),
        offload_moe_gate_up_sublayers=(0, 1, 2, 3),
        **base_kw,
    )
    hbm_cfg = qwen35_jax.tiny_config(
        save_decoder_input_hbm=True,
        offload_decoder_input=False,
        save_moe_gate_up_sublayers=(1, 3),
        offload_moe_gate_up_sublayers=(0, 2),
        **base_kw,
    )
    mesh = qwen35_jax.create_mesh(ref_cfg, aot_topology=None)
    params = qwen35_jax.init_random_params(ref_cfg, mesh, jax.random.PRNGKey(2027), param_dtype=jnp.float32)
    batch = _make_synthetic_batch(ref_cfg, mesh)

    ref_loss, ref_grads = jax.jit(qwen35_jax.make_value_and_grad_fn(ref_cfg, mesh))(params, *batch)
    hbm_loss, hbm_grads = jax.jit(qwen35_jax.make_value_and_grad_fn(hbm_cfg, mesh))(params, *batch)
    np.testing.assert_allclose(np.asarray(hbm_loss), np.asarray(ref_loss), rtol=1e-6, atol=1e-6)
    for (path, gr), (_, gh) in zip(
        jax.tree.leaves_with_path(ref_grads),
        jax.tree.leaves_with_path(hbm_grads),
    ):
      np.testing.assert_allclose(
          np.asarray(gh),
          np.asarray(gr),
          rtol=1e-5,
          atol=1e-5,
          err_msg=f"HBM cache mismatch at {jax.tree_util.keystr(path)}",
      )

    cfg_remat_di = qwen35_jax.tiny_config(
        save_decoder_input_hbm=False,
        offload_decoder_input=False,
        save_moe_gate_up_sublayers=(0, 1, 2, 3),
        offload_moe_gate_up_sublayers=(),
        **base_kw,
    )
    cfg_save_di = qwen35_jax.tiny_config(
        save_decoder_input_hbm=True,
        offload_decoder_input=False,
        save_moe_gate_up_sublayers=(0, 1, 2, 3),
        offload_moe_gate_up_sublayers=(),
        **base_kw,
    )
    shlo_remat_di = (
        jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_remat_di, mesh)).lower(params, *batch).as_text("stablehlo")
    )
    shlo_save_di = (
        jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_save_di, mesh)).lower(params, *batch).as_text("stablehlo")
    )
    self.assertNotIn("pinned_host", shlo_save_di)
    self.assertGreater(len(shlo_remat_di), len(shlo_save_di))

  def test_decoupled_ep_d2d_intra_chip_across_ep_8_16_32(self):
    import itertools  # pylint: disable=g-import-not-at-top
    import types  # pylint: disable=g-import-not-at-top
    from maxtext.experimental.qwen_35_64c import mesh as mesh_lib  # pylint: disable=g-import-not-at-top

    fake_devices = [
        types.SimpleNamespace(coords=(x, y, z), core_on_chip=c)
        for x, y, z, c in itertools.product(range(4), range(4), range(4), range(2))
    ]
    for fsdp_size, ep_size, seq_len, gbs in [
        (16, 8, 65536, 32),
        (8, 16, 131072, 16),
        (4, 32, 262144, 8),
    ]:
      cfg = qwen35_jax.full_config(
          fsdp_size=fsdp_size,
          ep_size=ep_size,
          seq_len=seq_len,
          global_batch_size=gbs,
          decouple_ep_d2d=True,
      )
      arr, axes = mesh_lib.arrange_mesh_devices(cfg, fake_devices)
      self.assertEqual(axes, ("fsdp", "ep_d2d", "expert"))
      self.assertEqual(arr.shape, (fsdp_size, 2, ep_size // 2))
      for f_idx in range(fsdp_size):
        for e_idx in range(ep_size // 2):
          d0 = arr[f_idx, 0, e_idx]
          d1 = arr[f_idx, 1, e_idx]
          self.assertEqual(d0.coords, d1.coords)
          self.assertEqual({d0.core_on_chip, d1.core_on_chip}, {0, 1})

  def test_grad_accum_exact_fp32_parity(self):
    cfg_ga1 = qwen35_jax.tiny_config(
        global_batch_size=4,
        fsdp_size=2,
        ep_size=4,
        grad_accum_steps=1,
        schedule_mode="batchsplit",
        mu_dtype="float32",
    )
    cfg_ga2 = qwen35_jax.tiny_config(
        global_batch_size=8,
        fsdp_size=2,
        ep_size=4,
        grad_accum_steps=2,
        grad_accum_dtype="float32",
        schedule_mode="batchsplit",
        mu_dtype="float32",
    )
    mesh = qwen35_jax.create_mesh(cfg_ga1, aot_topology=None)
    params = qwen35_jax.init_random_params(cfg_ga1, mesh, jax.random.PRNGKey(123), param_dtype=jnp.float32)
    batch_full = _make_synthetic_batch(cfg_ga2, mesh)
    sharding = NamedSharding(mesh, qwen35_jax.data_pspec(cfg_ga1))
    # make_value_and_grad_fn reshapes each local FSDP shard (4 rows -> (2, 2))
    # along the micro-step axis, so micro-step s on FSDP shard f takes rows
    # [2*f + s] of each 4-row local shard.
    idx_step0 = jnp.array([0, 1, 4, 5], dtype=jnp.int32)
    idx_step1 = jnp.array([2, 3, 6, 7], dtype=jnp.int32)
    batch_0 = tuple(jax.device_put(jnp.asarray(arr)[idx_step0], sharding) for arr in batch_full)
    batch_1 = tuple(jax.device_put(jnp.asarray(arr)[idx_step1], sharding) for arr in batch_full)
    vg_ga1 = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_ga1, mesh))
    loss_0, grads_0 = vg_ga1(params, *batch_0)
    loss_1, grads_1 = vg_ga1(params, *batch_1)
    loss_ref = 0.5 * (loss_0 + loss_1)
    grads_ref = jax.tree.map(lambda g0, g1: 0.5 * (g0 + g1), grads_0, grads_1)

    loss_2, grads_2 = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_ga2, mesh))(params, *batch_full)
    np.testing.assert_allclose(np.asarray(loss_2), np.asarray(loss_ref), rtol=1e-5, atol=1e-5)
    for (path, g1), (_, g2) in zip(jax.tree.leaves_with_path(grads_ref), jax.tree.leaves_with_path(grads_2)):
      np.testing.assert_allclose(
          np.asarray(g2),
          np.asarray(g1),
          rtol=2e-5,
          atol=2e-5,
          err_msg=f"grad_accum parity mismatch at {jax.tree_util.keystr(path)}",
      )

  def test_full_config_auto_sc_min_pspecs_across_ep_modes(self):
    cfg_default = qwen35_jax.full_config()
    self.assertEqual(cfg_default.dense_sharding_mode, "auto_sc_min")
    self.assertTrue(cfg_default.replicate_sub_sc_dense)
    self.assertTrue(cfg_default.route_before_shared_expert)
    self.assertTrue(cfg_default.pipeline_lm_head_chunks)

    for decouple_ep_d2d in (True, False):
      cfg = qwen35_jax.full_config(decouple_ep_d2d=decouple_ep_d2d)
      ps = qwen35_jax.get_param_pspecs(cfg)
      s2_ax = cfg.dense_stage2_axis_names
      all_ax = cfg.all_sharded_axes

      for sub_ps in (ps.blocks.gdn_0, ps.blocks.attn_3):
        self.assertEqual(sub_ps.moe.shared_gate, P(None, None, None))
        self.assertEqual(sub_ps.moe.router_gate, P(None, s2_ax, None))
        self.assertEqual(sub_ps.moe.shared_wi_0, P(None, s2_ax, None))
        self.assertEqual(sub_ps.moe.shared_wi_1, P(None, s2_ax, None))
        self.assertEqual(sub_ps.moe.shared_wo, P(None, None, s2_ax))

      self.assertEqual(ps.blocks.gdn_0.in_proj_ba, P(None, None, None))
      self.assertEqual(ps.blocks.gdn_0.in_proj_qkvz, P(None, all_ax, None))
      self.assertEqual(ps.blocks.gdn_0.out_proj, P(None, all_ax, None))

      self.assertEqual(ps.blocks.attn_3.k_proj, P(None, s2_ax, None, None))
      self.assertEqual(ps.blocks.attn_3.v_proj, P(None, s2_ax, None, None))
      self.assertEqual(ps.blocks.attn_3.q_proj, P(None, all_ax, None, None))
      self.assertEqual(ps.blocks.attn_3.out_proj, P(None, all_ax, None))

  def test_auto_sc_min_sharding_and_pipelined_lm_head_fp32_parity(self):
    cfg_ref = qwen35_jax.tiny_config(
        schedule_mode="batchsplit",
        fsdp_pipeline_mode="full_prefetch_defer_rs",
        dense_sharding_mode="fsdp_expert",
        replicate_sub_sc_dense=False,
        route_before_shared_expert=False,
        pipeline_lm_head_chunks=False,
        sc_min_collective_bytes=12_000,
    )
    cfg_sharding_pipe = qwen35_jax.tiny_config(
        schedule_mode="batchsplit",
        fsdp_pipeline_mode="full_prefetch_defer_rs",
        dense_sharding_mode="auto_sc_min",
        replicate_sub_sc_dense=True,
        route_before_shared_expert=False,
        pipeline_lm_head_chunks=True,
        sc_min_collective_bytes=12_000,
    )
    cfg_opt = qwen35_jax.tiny_config(
        schedule_mode="batchsplit",
        fsdp_pipeline_mode="full_prefetch_defer_rs",
        dense_sharding_mode="auto_sc_min",
        replicate_sub_sc_dense=True,
        route_before_shared_expert=True,
        pipeline_lm_head_chunks=True,
        sc_min_collective_bytes=12_000,
    )
    # Verify 12 KB threshold exercises all 3 sharding tiers simultaneously.
    self.assertEqual(cfg_opt.dense_param_sharding_kind("shared_gate"), "replicated")
    self.assertEqual(cfg_opt.dense_param_sharding_kind("in_proj_ba"), "replicated")
    self.assertEqual(cfg_opt.dense_param_sharding_kind("shared_wi"), "fsdp_only")
    self.assertEqual(cfg_opt.dense_param_sharding_kind("kv_proj"), "fsdp_only")
    self.assertEqual(cfg_opt.dense_param_sharding_kind("in_proj_qkvz"), "fsdp_expert")
    self.assertEqual(cfg_opt.dense_param_sharding_kind("q_proj"), "fsdp_expert")

    mesh = qwen35_jax.create_mesh(cfg_ref, aot_topology=None)
    params_ref = qwen35_jax.init_random_params(cfg_ref, mesh, jax.random.PRNGKey(314), param_dtype=jnp.float32)
    pspecs_opt = qwen35_jax.get_param_pspecs(cfg_opt)
    params_opt = jax.tree.map(
        lambda arr, ps: jax.device_put(np.asarray(arr), NamedSharding(mesh, ps)),
        params_ref,
        pspecs_opt,
    )
    batch = _make_synthetic_batch(cfg_ref, mesh)

    loss_ref, grads_ref = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_ref, mesh))(params_ref, *batch)
    loss_sp, grads_sp = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_sharding_pipe, mesh))(params_opt, *batch)
    loss_opt, grads_opt = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_opt, mesh))(params_opt, *batch)

    np.testing.assert_allclose(np.asarray(loss_sp), np.asarray(loss_ref), rtol=1e-5, atol=1e-5)
    for (path, g_ref), (_, g_sp) in zip(
        jax.tree.leaves_with_path(grads_ref),
        jax.tree.leaves_with_path(grads_sp),
    ):
      np.testing.assert_allclose(
          np.asarray(g_sp),
          np.asarray(g_ref),
          rtol=5e-5,
          atol=5e-5,
          err_msg=(f"sharding+pipe parity mismatch at {jax.tree_util.keystr(path)}"),
      )

    np.testing.assert_allclose(np.asarray(loss_opt), np.asarray(loss_ref), rtol=1e-5, atol=1e-5)
    for (path, g_ref), (_, g_opt) in zip(
        jax.tree.leaves_with_path(grads_ref),
        jax.tree.leaves_with_path(grads_opt),
    ):
      np.testing.assert_allclose(
          np.asarray(g_opt),
          np.asarray(g_ref),
          rtol=5e-4,
          atol=5e-4,
          err_msg=(f"auto_sc_min parity mismatch at {jax.tree_util.keystr(path)}"),
      )

  def test_ep_routing_weights_bwd_sc_bit_exact(self):
    full_cfg = qwen35_jax.full_config()
    self.assertEqual(full_cfg.ep_routing_weights_bwd_sc, "sc1")

    common = dict(
        num_superblocks=1,
        schedule_mode="batchsplit",
        emb_dim=256,
        global_batch_size=8,
        seq_len=256,
        num_experts=16,
        num_experts_per_tok=8,
        decouple_ep_d2d=True,
        fsdp_size=2,
        ep_size=4,
        ep_combine_fwd_unpinned=False,
        route_before_shared_expert=True,
        pipeline_lm_head_chunks=True,
        mu_dtype="float32",
    )
    # Per microbatch: 128 local tokens * 8 topk = 1024 (exercises % 1024 == 0).
    cfg_base = qwen35_jax.tiny_config(**common)
    mesh = qwen35_jax.create_mesh(cfg_base, aot_topology=None)
    params = qwen35_jax.init_random_params(cfg_base, mesh, jax.random.PRNGKey(777), param_dtype=jnp.float32)
    batch = _make_synthetic_batch(cfg_base, mesh)
    loss_base, grads_base = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_base, mesh))(params, *batch)

    cfg_sc1 = qwen35_jax.tiny_config(
        **common,
        ep_routing_weights_bwd_sc="sc1",
    )
    loss_sc1, grads_sc1 = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_sc1, mesh))(params, *batch)
    np.testing.assert_array_equal(np.asarray(loss_sc1), np.asarray(loss_base))
    for (path, gb), (_, gv) in zip(
        jax.tree.leaves_with_path(grads_base),
        jax.tree.leaves_with_path(grads_sc1),
    ):
      np.testing.assert_allclose(
          np.asarray(gv),
          np.asarray(gb),
          rtol=1e-6,
          atol=1e-6,
          err_msg=("ep_routing_weights_bwd_sc=sc1 grad mismatch at" f" {jax.tree_util.keystr(path)}"),
      )

  def test_qwen35_init_scales(self):
    """Verifies official Qwen3.5 4D projection, norm, and GDN init scales."""
    cfg = qwen35_jax.tiny_config(
        schedule_mode="batchsplit",
        mu_dtype="float32",
        global_batch_size=16,
    )
    mesh = qwen35_jax.create_mesh(cfg, aot_topology=None)
    params = qwen35_jax.init_random_params(cfg, mesh, jax.random.PRNGKey(77), param_dtype=jnp.float32)

    # Zero-centered RMSNorm weights must start at 0 (identity scale 1 + 0 = 1).
    for arr in (
        params.decoder_norm,
        params.blocks.gdn_0.pre_attn_norm,
        params.blocks.attn_3.pre_attn_norm,
        params.blocks.attn_3.q_norm,
        params.blocks.attn_3.k_norm,
    ):
      np.testing.assert_array_equal(np.asarray(arr), np.zeros_like(arr))
    np.testing.assert_array_equal(
        np.asarray(params.blocks.gdn_0.gdn_norm),
        np.ones_like(params.blocks.gdn_0.gdn_norm),
    )
    np.testing.assert_array_equal(
        np.asarray(params.blocks.gdn_0.dt_bias),
        np.ones_like(params.blocks.gdn_0.dt_bias),
    )
    a_exp = np.exp(np.asarray(params.blocks.gdn_0.A_log))
    self.assertTrue(bool(np.all((a_exp >= 0.01) & (a_exp <= 16.0))))

    # 4D q/k/v projections must scale by emb_dim**-0.5, NOT num_heads**-0.5.
    expected_std = cfg.emb_dim**-0.5
    for name in ("q_proj", "k_proj", "v_proj"):
      w = np.asarray(getattr(params.blocks.attn_3, name))
      self.assertAlmostEqual(float(np.std(w)), expected_std, delta=0.03)
    self.assertAlmostEqual(
        float(np.std(np.asarray(params.token_embedder))),
        expected_std,
        delta=0.02,
    )

  def test_qwen35_switch_aux_loss(self):
    """Verifies Qwen3.5 Switch load-balancing loss value and router gradient."""
    cfg = qwen35_jax.tiny_config(
        schedule_mode="batchsplit",
        mu_dtype="float32",
        global_batch_size=16,
        router_aux_loss_coef=0.001,
    )
    cfg_no_aux = dataclasses.replace(cfg, router_aux_loss_coef=0.0)
    mesh = qwen35_jax.create_mesh(cfg, aot_topology=None)
    params = qwen35_jax.init_random_params(cfg, mesh, jax.random.PRNGKey(77), param_dtype=jnp.float32)

    # Switch aux loss: when router_gate == 0, router probs are uniform 1/E;
    # sum_e f_e == top_k for every layer regardless of tie-break counts, so
    # exact aux_loss == coef * top_k == 0.001 * top_k.
    def _zero_router_and_routed(sub):
      return dataclasses.replace(
          sub,
          moe=dataclasses.replace(
              sub.moe,
              router_gate=jnp.zeros_like(sub.moe.router_gate),
              wi=jnp.zeros_like(sub.moe.wi),
              wo=jnp.zeros_like(sub.moe.wo),
          ),
      )

    blocks_z = dataclasses.replace(
        params.blocks,
        gdn_0=_zero_router_and_routed(params.blocks.gdn_0),
        gdn_1=_zero_router_and_routed(params.blocks.gdn_1),
        gdn_2=_zero_router_and_routed(params.blocks.gdn_2),
        attn_3=_zero_router_and_routed(params.blocks.attn_3),
    )
    p_zero = dataclasses.replace(params, blocks=blocks_z)
    batch = _make_synthetic_batch(cfg, mesh)
    loss_aux, grads_aux = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg, mesh))(p_zero, *batch)
    loss_ce, grads_ce = jax.jit(qwen35_jax.make_value_and_grad_fn(cfg_no_aux, mesh))(p_zero, *batch)
    expected_aux = cfg.router_aux_loss_coef * cfg.num_experts_per_tok
    self.assertAlmostEqual(float(loss_aux - loss_ce), expected_aux, places=5)
    # With wi=wo=0, CE grad wrt router_gate is identically 0, whereas Switch aux
    # loss produces a non-zero load-balancing gradient pushing away from expert 0..K-1.
    self.assertEqual(float(jnp.max(jnp.abs(grads_ce.blocks.gdn_0.moe.router_gate))), 0.0)
    self.assertGreater(float(jnp.max(jnp.abs(grads_aux.blocks.gdn_0.moe.router_gate))), 1e-6)

  def test_128k_sharding_invariance_grad_norm_and_c4_pack(self):
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
    from maxtext.experimental.qwen_35_64c import data as data_lib  # pylint: disable=g-import-not-at-top
    from array_record.python import array_record_module  # pylint: disable=g-import-not-at-top

    base_kw = dict(
        fsdp_size=2,
        ep_size=4,
        decouple_ep_d2d=True,
        schedule_mode="batchsplit",
        fsdp_pipeline_mode="full_prefetch_defer_rs",
        enable_remat=True,
        save_moe_gate_up_sublayers=(0, 1, 2, 3),
        offload_moe_gate_up_sublayers=(),
        route_before_shared_expert=True,
        pipeline_lm_head_chunks=True,
        mu_dtype="float32",
        learning_rate=1e-3,
        router_aux_loss_coef=0.001,
    )
    ref_cfg = qwen35_jax.tiny_config(**base_kw)
    mesh = qwen35_jax.create_mesh(ref_cfg, aot_topology=None)

    # 1. Verify multi-chunk ArrayRecordC4 packing (rec_len=16 -> seq_len=32, 2 chunks/row).
    tmp_dir = self.create_tempdir().full_path
    ar_path = os.path.join(tmp_dir, "c4_s16-00000-of-00001.array_record")
    writer = array_record_module.ArrayRecordWriter(ar_path, "group_size:1")
    raw_recs = []
    for i in range(8):
      arr = np.arange(1 + i * 16, 1 + (i + 1) * 16, dtype=np.int64) % 250 + 1
      raw_recs.append(arr.astype(np.int32))
      ex = tf.train.Example(
          features=tf.train.Features(
              feature={data_lib.TOKEN_FEATURE: tf.train.Feature(int64_list=tf.train.Int64List(value=arr.tolist()))}
          )
      )
      writer.write(ex.SerializeToString())
    writer.close()

    c4 = data_lib.ArrayRecordC4(tmp_dir, seq_len=32)
    self.assertEqual(c4.num_rows, 4)
    data_sharding = NamedSharding(mesh, qwen35_jax.data_pspec(ref_cfg))
    tok, tgt, pos, seg = c4.make_batch(0, ref_cfg.global_batch_size, data_sharding)
    tok_np, tgt_np, pos_np, seg_np = (
        np.asarray(tok),
        np.asarray(tgt),
        np.asarray(pos),
        np.asarray(seg),
    )
    full_row0 = np.concatenate([raw_recs[0], raw_recs[1]])
    np.testing.assert_array_equal(tok_np[0], full_row0)
    np.testing.assert_array_equal(tgt_np[0, :31], full_row0[1:])
    self.assertEqual(int(tgt_np[0, 31]), 0)
    np.testing.assert_array_equal(pos_np[0], np.arange(32, dtype=np.int32))
    expected_seg = np.array([1] * 31 + [0], dtype=np.int32)
    np.testing.assert_array_equal(seg_np[0], expected_seg)

    # 2. Verify sharding-invariant init + Switch aux loss + strict FP32 grad_norm parity across
    #    pure sharding/collective variants (`fsdp_expert` vs `auto_sc_min` vs `fsdp_only` +
    #    `replicate_sub_sc_dense` + `ep_routing_weights_bwd_sc="sc1"`), plus remat bank-save parity.
    key = jax.random.PRNGKey(2028)
    ref_params = qwen35_jax.init_random_params(ref_cfg, mesh, key, param_dtype=jnp.float32)
    ref_opt = qwen35_jax.init_opt_state(ref_cfg, mesh, ref_params)
    ref_step = qwen35_jax.make_train_step(ref_cfg, mesh, donate_state=False, return_grad_norm=True)
    ref_p1, _, ref_loss, ref_gn = ref_step(ref_params, ref_opt, tok, tgt, pos, seg)

    sharding_variants = [
        dict(
            dense_sharding_mode="fsdp_only",
            replicate_sub_sc_dense=True,
        ),
        dict(
            dense_sharding_mode="auto_sc_min",
            replicate_sub_sc_dense=True,
            ep_routing_weights_bwd_sc="sc1",
            sc_min_collective_bytes=12_000,
        ),
        dict(
            dense_sharding_mode="fsdp_only",
            replicate_sub_sc_dense=True,
            ep_routing_weights_bwd_sc="sc1",
            sc_min_collective_bytes=12_000,
        ),
    ]
    for ov in sharding_variants:
      with self.subTest(**ov):
        v_cfg = qwen35_jax.tiny_config(**base_kw, **ov)
        v_params = qwen35_jax.init_random_params(v_cfg, mesh, key, param_dtype=jnp.float32)
        for (p_path, rp), (_, vp) in zip(
            jax.tree.leaves_with_path(ref_params),
            jax.tree.leaves_with_path(v_params),
        ):
          np.testing.assert_array_equal(
              np.asarray(vp),
              np.asarray(rp),
              err_msg=(f"init mismatch for {ov} at {jax.tree_util.keystr(p_path)}"),
          )
        v_opt = qwen35_jax.init_opt_state(v_cfg, mesh, v_params)
        v_step = qwen35_jax.make_train_step(v_cfg, mesh, donate_state=False, return_grad_norm=True)
        v_p1, _, v_loss, v_gn = v_step(v_params, v_opt, tok, tgt, pos, seg)
        np.testing.assert_allclose(float(v_loss), float(ref_loss), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(float(v_gn), float(ref_gn), rtol=1e-4, atol=1e-4)
        for (p_path, rp1), (_, vp1) in zip(
            jax.tree.leaves_with_path(ref_p1),
            jax.tree.leaves_with_path(v_p1),
        ):
          np.testing.assert_allclose(
              np.asarray(vp1),
              np.asarray(rp1),
              rtol=2e-4,
              atol=5e-5,
              err_msg=(f"step1 param mismatch for {ov} at" f" {jax.tree_util.keystr(p_path)}"),
          )

    # Remat MoE bank recomputation vs saved `moe_gate_up_(0..3)` matches forward loss strictly
    # and matches grad_norm within FP8 e5m2 rowwise power-of-2 quantization noise (<2%).
    remat_cfg = qwen35_jax.tiny_config(**{**base_kw, "save_moe_gate_up_sublayers": ()})
    remat_step = qwen35_jax.make_train_step(remat_cfg, mesh, donate_state=False, return_grad_norm=True)
    _, _, remat_loss, remat_gn = remat_step(ref_params, ref_opt, tok, tgt, pos, seg)
    np.testing.assert_allclose(float(remat_loss), float(ref_loss), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(remat_gn), float(ref_gn), rtol=2e-2, atol=2e-2)

  def test_pallas_rmsnorm_and_scatter_dw_cpu_interpret_parity(self):
    """Verifies Pallas VMEM RMSNorm (fwd+bwd) and 4-bank LUT scatter_dw on CPU."""
    from maxtext.experimental.qwen_35_64c import ops as ops_lib  # pylint: disable=g-import-not-at-top
    from maxtext.experimental.qwen_35_64c.ext_kernels import rmsnorm_tc  # pylint: disable=g-import-not-at-top

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(101), 4)
    for shape in ((2, 16, 256), (1, 2, 16, 4, 128)):
      d = shape[-1]
      x = jax.random.normal(k1, shape, dtype=jnp.float32)
      z = jax.random.normal(k2, shape, dtype=jnp.float32)
      s = 0.1 * jax.random.normal(k3, (d,), dtype=jnp.float32)
      ct = jax.random.normal(k4, shape, dtype=jnp.float32)

      def _loss_zc_ref(x_i, s_i, ct_i=ct):
        return jnp.sum(ops_lib.rms_norm_zero_centered(x_i, s_i, 1e-6, use_pallas=False) * ct_i)

      def _loss_zc_pl(x_i, s_i, ct_i=ct):
        return jnp.sum(rmsnorm_tc.rms_norm_zero_centered_pallas(x_i, s_i, 1e-6, interpret=True) * ct_i)

      v_ref, (gx_ref, gs_ref) = jax.jit(jax.value_and_grad(_loss_zc_ref, argnums=(0, 1)))(x, s)
      v_pl, (gx_pl, gs_pl) = jax.jit(jax.value_and_grad(_loss_zc_pl, argnums=(0, 1)))(x, s)
      np.testing.assert_allclose(v_pl, v_ref, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(gx_pl, gx_ref, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(gs_pl, gs_ref, rtol=1e-5, atol=1e-5)

      def _loss_gated_ref(x_i, z_i, s_i, ct_i=ct):
        return jnp.sum(ops_lib.rms_norm_gated(x_i, z_i, s_i, 1e-6, use_pallas=False) * ct_i)

      def _loss_gated_pl(x_i, z_i, s_i, ct_i=ct):
        return jnp.sum(rmsnorm_tc.rms_norm_gated_pallas(x_i, z_i, s_i, 1e-6, interpret=True) * ct_i)

      vg_ref, (ggx_ref, ggz_ref, ggs_ref) = jax.jit(jax.value_and_grad(_loss_gated_ref, argnums=(0, 1, 2)))(x, z, s)
      vg_pl, (ggx_pl, ggz_pl, ggs_pl) = jax.jit(jax.value_and_grad(_loss_gated_pl, argnums=(0, 1, 2)))(x, z, s)
      np.testing.assert_allclose(vg_pl, vg_ref, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(ggx_pl, ggx_ref, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(ggz_pl, ggz_ref, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(ggs_pl, ggs_ref, rtol=1e-5, atol=1e-5)

    num_slots = 4096
    capacity = 2048
    perm = jax.random.permutation(jax.random.PRNGKey(202), num_slots).astype(jnp.int32)
    rows_all = jax.random.normal(jax.random.PRNGKey(203), (num_slots,), dtype=jnp.float32)
    for start_v, cnt_v in ((0, 3500), (2048, 3500), (2048, 1500)):
      slot_win = jax.lax.dynamic_slice_in_dim(jnp.pad(perm, (0, capacity)), start_v, capacity, axis=0)
      rows_win = jax.lax.dynamic_slice_in_dim(jnp.pad(rows_all, (0, capacity)), start_v, capacity, axis=0)
      expected = rmsnorm_tc._scatter_dw_fallback(slot_win, rows_win, start_v, cnt_v, num_slots=num_slots)
      actual = jax.jit(
          lambda sw, rw, st, c: rmsnorm_tc.scatter_dw_vmem_tc(sw, rw, st, c, num_slots=num_slots, interpret=True)
      )(slot_win, rows_win, jnp.int32(start_v), jnp.int32(cnt_v))
      np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  def test_xla_layout_offload_and_routing_optimizations_parity(self):
    """Verifies E2E loss & grad parity across all 5 XLA/layout/offload flags."""
    full_cfg = qwen35_jax.full_config()
    self.assertEqual(full_cfg.gdn_norm_layout_mode, "natural")
    self.assertTrue(full_cfg.use_pallas_rmsnorm)
    self.assertFalse(full_cfg.use_pallas_gated_rmsnorm)
    self.assertTrue(full_cfg.gdn_split_qkvz_proj)
    self.assertTrue(full_cfg.attn_kv_grouped_layout)
    self.assertFalse(full_cfg.use_pallas_dw_scatter)
    self.assertTrue(full_cfg.moe_chunked_searchsorted)
    self.assertTrue(full_cfg.pack_scan_routing_metadata)
    self.assertEqual(full_cfg.peel_tail_hbm_superblocks, 1)

    base_cfg = qwen35_jax.tiny_config(
        schedule_mode="batchsplit",
        mu_dtype="float32",
        global_batch_size=16,
        sa_fold_gqa_into_seq=True,
        moe_local_65key_sort=True,
        save_moe_routing=True,
        save_moe_gate_up_sublayers=(1, 3),
        offload_moe_gate_up_sublayers=(),
        router_aux_loss_coef=0.001,
    )
    mesh = qwen35_jax.create_mesh(base_cfg, aot_topology=None)
    raw_params = qwen35_jax.init_random_params(base_cfg, mesh, jax.random.PRNGKey(314), param_dtype=jnp.float32)
    params_eq = dataclasses.replace(
        raw_params,
        blocks=dataclasses.replace(
            raw_params.blocks,
            **{
                sub_name: dataclasses.replace(
                    getattr(raw_params.blocks, sub_name),
                    moe=dataclasses.replace(
                        getattr(raw_params.blocks, sub_name).moe,
                        wi=_equalize_expert_amax(getattr(raw_params.blocks, sub_name).moe.wi),
                        wo=_equalize_expert_amax(getattr(raw_params.blocks, sub_name).moe.wo),
                    ),
                )
                for sub_name in ("gdn_0", "gdn_1", "gdn_2", "attn_3")
            },
        ),
    )
    batch = _make_synthetic_batch(base_cfg, mesh)

    # 1. Dense/mixer grad parity across all 6 layout, projection, router, scan,
    #    and peeled-superblock flags (with wi=wo=0 so discontinuous e4m3fn
    #    rounding does not butterfly-amplify bf16 contraction reassociations).
    all_opt_cfg = dataclasses.replace(
        base_cfg,
        gdn_norm_layout_mode="natural",
        gdn_split_qkvz_proj=True,
        attn_kv_grouped_layout=True,
        moe_chunked_searchsorted=True,
        pack_scan_routing_metadata=True,
        peel_tail_hbm_superblocks=1,
    )
    params_dense = dataclasses.replace(
        raw_params,
        blocks=dataclasses.replace(
            raw_params.blocks,
            **{
                sub_name: dataclasses.replace(
                    getattr(raw_params.blocks, sub_name),
                    moe=dataclasses.replace(
                        getattr(raw_params.blocks, sub_name).moe,
                        wi=jnp.zeros_like(getattr(raw_params.blocks, sub_name).moe.wi),
                        wo=jnp.zeros_like(getattr(raw_params.blocks, sub_name).moe.wo),
                    ),
                )
                for sub_name in ("gdn_0", "gdn_1", "gdn_2", "attn_3")
            },
        ),
    )
    dense_ref_cfg = dataclasses.replace(base_cfg, router_aux_loss_coef=0.0)
    dense_opt_cfg = dataclasses.replace(all_opt_cfg, router_aux_loss_coef=0.0)
    loss_d_ref, grads_d_ref = jax.jit(qwen35_jax.make_value_and_grad_fn(dense_ref_cfg, mesh))(params_dense, *batch)
    loss_d_opt, grads_d_opt = jax.jit(qwen35_jax.make_value_and_grad_fn(dense_opt_cfg, mesh))(params_dense, *batch)
    self.assertLess(
        abs(float(loss_d_opt) - float(loss_d_ref)),
        1e-3,
        f"dense {loss_d_opt} vs {loss_d_ref}",
    )
    for (path, go), (_, gr) in zip(
        jax.tree.leaves_with_path(grads_d_opt),
        jax.tree.leaves_with_path(grads_d_ref),
    ):
      go_np = np.asarray(go, dtype=np.float64)
      gr_np = np.asarray(gr, dtype=np.float64)
      rel_l2 = float(np.linalg.norm(go_np - gr_np) / (np.linalg.norm(gr_np) + 1e-12))
      self.assertLess(
          rel_l2,
          5e-3,
          f"dense grad mismatch on {jax.tree_util.keystr(path)}: {rel_l2}",
      )

    # 2. Full FP8 routed-MoE E2E loss parity and gradient finiteness (matching
    #    test_unsplit_vs_batchsplit_fp32_parity).
    loss_ref, grads_ref = jax.jit(qwen35_jax.make_value_and_grad_fn(base_cfg, mesh))(params_eq, *batch)
    loss_all, grads_all = jax.jit(qwen35_jax.make_value_and_grad_fn(all_opt_cfg, mesh))(params_eq, *batch)
    self.assertLess(
        float(jnp.abs(loss_all - loss_ref) / jnp.abs(loss_ref)),
        5e-3,
        f"full E2E {loss_all} vs {loss_ref}",
    )
    for path, g in jax.tree.leaves_with_path((grads_ref, grads_all)):
      self.assertTrue(
          bool(jnp.all(jnp.isfinite(g))),
          f"Non-finite grad on {jax.tree_util.keystr(path)}",
      )


if __name__ == "__main__":
  absltest.main()
