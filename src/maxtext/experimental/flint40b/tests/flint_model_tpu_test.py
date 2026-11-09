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

"""TPU v7x end-to-end 9-layer Flint model + AdamW trainer tests."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np
import torch

from maxtext.experimental.flint40b import data
from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_loss
from maxtext.experimental.flint40b import flint_model
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import flint_train
from maxtext.experimental.flint40b import flint_types
from maxtext.experimental.flint40b.tests import hf_reference


def _cosine_similarity(a: np.ndarray | jax.Array, b: np.ndarray | jax.Array) -> float:
  a_f64 = np.asarray(a, dtype=np.float64).reshape(-1)
  b_f64 = np.asarray(b, dtype=np.float64).reshape(-1)
  denom = np.linalg.norm(a_f64) * np.linalg.norm(b_f64)
  if denom == 0.0:
    return 1.0 if np.allclose(a_f64, b_f64) else 0.0
  return float(np.dot(a_f64, b_f64) / denom)


def _round_to_bf16_f32(arr: np.ndarray) -> np.ndarray:
  """Rounds `float32` array to nearest-even `bfloat16` stored in `float32`."""
  u32 = np.asarray(arr, dtype=np.float32).view(np.uint32)
  lsb = (u32 >> np.uint32(16)) & np.uint32(1)
  rounded = (u32 + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
  return rounded.view(np.float32).copy()


def _condition_9layer_torch_model(
    torch_model: hf_reference.TorchFlintForCausalLM,
    seed: int = 1042,
) -> None:
  """Conditions norms, residual projections, and 16-cluster MoE router gates."""
  cfg_m = torch_model.config
  num_layers = cfg_m.num_hidden_layers
  hidden_size = cfg_m.hidden_size
  vocab_size = cfg_m.vocab_size
  num_experts = cfg_m.num_experts
  top_k = cfg_m.num_experts_per_tok

  gen = torch.Generator(device="cpu")
  gen.manual_seed(seed)
  rng = np.random.default_rng(seed)
  resid_scale = 1.0 / math.sqrt(2.0 * float(num_layers))

  with torch.no_grad():
    for name, param in torch_model.named_parameters():
      if "norm" in name and name.endswith("weight"):
        vals = torch.empty(param.shape, dtype=torch.float32).uniform_(0.8, 1.2, generator=gen)
        param.copy_(vals.to(device=param.device, dtype=param.dtype))
      elif name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
        param.mul_(resid_scale)

    q_mat, _ = np.linalg.qr(rng.standard_normal((hidden_size, num_experts)))
    clusters = (q_mat.T * math.sqrt(float(hidden_size))).astype(np.float32)

    vocab_clusters = np.arange(vocab_size, dtype=np.int64) % num_experts
    embed_jitter = rng.standard_normal((vocab_size, hidden_size)).astype(np.float32)
    embed_np = _round_to_bf16_f32(2.5 * clusters[vocab_clusters] + 0.20 * embed_jitter)
    torch_model.model.embed_tokens.weight.copy_(torch.from_numpy(embed_np))

    for layer_idx in range(cfg_m.num_dense_prefix_layers, num_layers):
      target_logits = np.zeros((num_experts, num_experts), dtype=np.float32)
      for c_idx in range(num_experts):
        for slot in range(top_k):
          target_logits[c_idx, (c_idx + layer_idx + slot) % num_experts] = 2.6 - 0.35 * slot
        for slot in range(top_k, num_experts):
          target_logits[c_idx, (c_idx + layer_idx + slot) % num_experts] = -1.0 - 0.22 * (slot - top_k)
      gate_pt = _round_to_bf16_f32(((target_logits.T @ clusters) / float(hidden_size)).astype(np.float32))
      layer_mlp = torch_model.model.layers[layer_idx].mlp
      assert isinstance(layer_mlp, hf_reference.TorchFlintSparseMoeBlock)
      layer_mlp.gate.weight.copy_(torch.from_numpy(gate_pt))


def _make_9layer_tpu_config(
    *,
    fsdp_size: int,
    cp_size: int,
    ep_size: int,
    use_splash: bool,
    scan_layers: bool,
    compute_dtype: str = "bfloat16",
    param_dtype: str = "float32",
    learning_rate: float = 1e-3,
    warmup_steps: int = 0,
) -> flint_config.FlintConfig:
  return flint_config.FlintConfig(
      model=flint_config.FlintModelConfig(
          vocab_size=2176,
          hidden_size=512,
          intermediate_size=1024,
          moe_intermediate_size=128,
          shared_expert_intermediate_size=256,
          num_hidden_layers=9,
          num_dense_prefix_layers=2,
          moe_cycle=4,
          num_attention_heads=8,
          num_key_value_heads=2,
          head_dim=128,
          num_experts=16,
          num_experts_per_tok=6,
          sliding_window=513,
          scan_layers=scan_layers,
      ),
      sharding=flint_config.FlintShardingConfig(
          fsdp_size=fsdp_size,
          cp_size=cp_size,
          ep_size=ep_size,
      ),
      kernels=flint_config.FlintKernelConfig(
          use_splash=use_splash,
          num_vocab_chunks=17,
          vocab_token_chunk_size=512,
      ),
      train=flint_config.FlintTrainConfig(
          global_batch_size=8,
          seq_len=1024,
          compute_dtype=compute_dtype,
          param_dtype=param_dtype,
          remat_policy="layer",
          learning_rate=learning_rate,
          warmup_steps=warmup_steps,
          total_steps=20,
          lr_schedule="warmup_cosine",
      ),
  )


class FlintModelTpuTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.default_backend(), "tpu")
    self.assertLen(jax.devices(), 8)

  def test_splash_vs_ref_and_scan_vs_unscan_across_meshes(self):
    base_cfg = _make_9layer_tpu_config(
        fsdp_size=8,
        cp_size=1,
        ep_size=1,
        use_splash=False,
        scan_layers=True,
    )
    torch_model = hf_reference.TorchFlintForCausalLM(base_cfg.model)
    hf_reference.randomize_torch_flint(torch_model, seed=42)
    _condition_9layer_torch_model(torch_model, seed=1042)
    state_dict: dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in torch_model.state_dict().items()}

    # Verify all 16 experts receive active tokens on every MoE layer (2..8).
    check_cfg = _make_9layer_tpu_config(
        fsdp_size=8,
        cp_size=1,
        ep_size=1,
        use_splash=True,
        scan_layers=True,
    )
    check_mesh = flint_sharding.create_mesh(check_cfg.sharding)
    check_params = hf_reference.torch_state_dict_to_flint_params(state_dict, check_cfg, check_mesh)
    check_batch = data.make_synthetic_batch(
        step=0,
        global_batch_size=check_cfg.train.global_batch_size,
        seq_len=check_cfg.train.seq_len,
        vocab_size=check_cfg.model.vocab_size,
        sharding=NamedSharding(check_mesh, flint_sharding.batch_pspec()),
        seed=17,
        pack_num_segments=2,
    )
    _, _, check_aux = flint_model.forward_with_intermediates(
        check_params,
        check_batch,
        check_cfg,
        check_mesh,
        return_router_aux=True,
    )
    for layer_idx in range(
        check_cfg.model.num_dense_prefix_layers,
        check_cfg.model.num_hidden_layers,
    ):
      aux_i = check_aux[layer_idx]
      self.assertIsNotNone(aux_i)
      assert aux_i is not None
      counts_i = np.asarray(aux_i.expert_counts, dtype=np.float32)
      self.assertEqual(counts_i.shape, (check_cfg.model.num_experts,))
      self.assertTrue(
          np.all(counts_i > 0),
          msg=f"Inactive routed experts at layer {layer_idx}: {counts_i}",
      )

    def _run_case(
        mesh_dims: tuple[int, int, int],
        *,
        use_splash: bool,
        scan_layers: bool,
    ) -> tuple[np.ndarray, float, float]:
      cfg = _make_9layer_tpu_config(
          fsdp_size=mesh_dims[0],
          cp_size=mesh_dims[1],
          ep_size=mesh_dims[2],
          use_splash=use_splash,
          scan_layers=scan_layers,
      )
      mesh = flint_sharding.create_mesh(cfg.sharding)
      params = hf_reference.torch_state_dict_to_flint_params(state_dict, cfg, mesh)
      b_sharding = NamedSharding(mesh, flint_sharding.batch_pspec())
      batch = data.make_synthetic_batch(
          step=0,
          global_batch_size=cfg.train.global_batch_size,
          seq_len=cfg.train.seq_len,
          vocab_size=cfg.model.vocab_size,
          sharding=b_sharding,
          seed=17,
          pack_num_segments=2,
      )
      logits = np.asarray(flint_model.forward_logits(params, batch, cfg, mesh))
      vg_fn = flint_model.make_value_and_grad_fn(cfg, mesh)
      (_, ce_loss, _), grads = vg_fn(params, batch)
      gnorm = float(flint_train.compute_global_grad_norm_f32(grads))
      return logits, float(ce_loss), gnorm

    ref_logits, ref_ce, ref_gnorm = _run_case((8, 1, 1), use_splash=False, scan_layers=True)
    self.assertTrue(np.isfinite(ref_ce))
    self.assertTrue(np.isfinite(ref_gnorm))
    self.assertGreater(ref_gnorm, 0.0)

    for mesh_dims in ((8, 1, 1), (4, 2, 1), (2, 2, 2)):
      scan_outputs: dict[bool, tuple[np.ndarray, float, float]] = {}
      for scan_layers in (True, False):
        got_logits, got_ce, got_gnorm = _run_case(mesh_dims, use_splash=True, scan_layers=scan_layers)
        scan_outputs[scan_layers] = (got_logits, got_ce, got_gnorm)
        with self.subTest(mesh_dims=mesh_dims, scan_layers=scan_layers):
          self.assertGreater(_cosine_similarity(got_logits, ref_logits), 0.999)
          self.assertLess(abs(got_ce - ref_ce), 5e-3)
          self.assertTrue(np.isfinite(got_gnorm))
          self.assertGreater(got_gnorm, 0.0)
          rel_gnorm = abs(got_gnorm - ref_gnorm) / ref_gnorm
          self.assertLess(rel_gnorm, 1e-2)

      with self.subTest(mesh_dims=mesh_dims, mode="scan_vs_unscan_splash"):
        self.assertGreater(
            _cosine_similarity(scan_outputs[True][0], scan_outputs[False][0]),
            0.99999,
        )
        self.assertLess(abs(scan_outputs[True][1] - scan_outputs[False][1]), 1e-4)

      ref_m_logits, ref_m_ce, ref_m_gnorm = _run_case(mesh_dims, use_splash=False, scan_layers=True)
      with self.subTest(mesh_dims=mesh_dims, mode="ref_mesh_invariance"):
        self.assertGreater(_cosine_similarity(ref_m_logits, ref_logits), 0.9999)
        self.assertLess(abs(ref_m_ce - ref_ce), 1e-3)
        self.assertLess(abs(ref_m_gnorm - ref_gnorm) / ref_gnorm, 1e-3)

  def test_four_step_adamw_training_loss_decreases(self):
    cfg = _make_9layer_tpu_config(
        fsdp_size=2,
        cp_size=2,
        ep_size=2,
        use_splash=True,
        scan_layers=True,
        compute_dtype="bfloat16",
        param_dtype="float32",
        learning_rate=1e-3,
        warmup_steps=0,
    )
    mesh = flint_sharding.create_mesh(cfg.sharding)
    state = flint_train.init_train_state(jax.random.key(0), mesh, cfg)
    b_sharding = NamedSharding(mesh, flint_sharding.batch_pspec())
    batch = data.make_synthetic_batch(
        step=0,
        global_batch_size=cfg.train.global_batch_size,
        seq_len=cfg.train.seq_len,
        vocab_size=cfg.model.vocab_size,
        sharding=b_sharding,
        seed=99,
        pack_num_segments=2,
    )
    train_step = flint_train.make_train_step(cfg, mesh, donate_state=True)

    ce_losses: list[float] = []
    for _ in range(4):
      state, metrics = train_step(state, batch)
      ce_val = float(metrics.ce_loss)
      gnorm_val = float(metrics.grad_norm)
      lr_val = float(metrics.learning_rate)
      self.assertTrue(np.isfinite(ce_val))
      self.assertTrue(np.isfinite(gnorm_val))
      self.assertGreater(gnorm_val, 0.0)
      self.assertGreater(lr_val, 0.0)
      ce_losses.append(ce_val)

    self.assertEqual(int(state.step), 4)
    self.assertLess(ce_losses[3], ce_losses[0])

  def test_full_40b_lm_head_ce3_dual_sc_zipf_packed(self):
    mesh = flint_sharding.create_mesh(flint_config.FlintShardingConfig(fsdp_size=8, cp_size=1, ep_size=1))
    hidden_size = 2048
    vocab_size = 150144
    gbs = 8
    seq_len = 1024

    rng = np.random.default_rng(777)
    x_np = _round_to_bf16_f32(rng.standard_normal((gbs, seq_len, hidden_size)).astype(np.float32) * 0.5)
    w_np = rng.standard_normal((hidden_size, vocab_size)).astype(np.float32) * 0.02
    emb_np = rng.standard_normal((vocab_size, hidden_size)).astype(np.float32) * 0.5
    is_zipf = rng.random((gbs, seq_len)) < 0.55
    zipf_ids = rng.integers(0, 8, size=(gbs, seq_len), dtype=np.int32)
    uni_ids = rng.integers(0, vocab_size, size=(gbs, seq_len), dtype=np.int32)
    tgt_np = np.where(is_zipf, zipf_ids, uni_ids).astype(np.int32)
    tok_np = np.roll(tgt_np, shift=1, axis=1)
    msk_np = np.ones((gbs, seq_len), dtype=np.int32)
    msk_np[:, -37:] = 0
    msk_np[:, 300] = 0
    msk_np[:, 700] = 0
    tgt_np[msk_np == 0] = -1

    act_spec = PartitionSpec(("fsdp", "ep"), "cp", None)
    b_spec = PartitionSpec(("fsdp", "ep"), "cp")
    w_spec = PartitionSpec(("fsdp", "cp", "ep"), None)
    emb_spec = PartitionSpec(None, ("fsdp", "cp", "ep"))
    x_sh = jax.device_put(jnp.asarray(x_np), NamedSharding(mesh, act_spec))
    w_sh = jax.device_put(jnp.asarray(w_np), NamedSharding(mesh, w_spec))
    emb_sh = jax.device_put(jnp.asarray(emb_np), NamedSharding(mesh, emb_spec))
    tok_sh = jax.device_put(jnp.asarray(tok_np), NamedSharding(mesh, b_spec))
    tgt_sh = jax.device_put(jnp.asarray(tgt_np), NamedSharding(mesh, b_spec))
    msk_sh = jax.device_put(jnp.asarray(msk_np), NamedSharding(mesh, b_spec))
    dy_sh = jax.device_put(jnp.asarray(x_np, dtype=jnp.bfloat16), NamedSharding(mesh, act_spec))

    def _eval_ce(
        *,
        use_dense: bool,
        num_chunks: int,
        tok_chunk: int,
        sc_mode: str,
    ) -> tuple[float, np.ndarray, np.ndarray, float]:
      cfg = flint_config.flint_40b_config(
          fsdp_size=8,
          cp_size=1,
          ep_size=1,
          global_batch_size=gbs,
          seq_len=seq_len,
          num_vocab_chunks=num_chunks,
          vocab_token_chunk_size=tok_chunk,
          sc_collective_mode=sc_mode,
          sc_min_collective_bytes=4096,
      )
      total_valid = float(np.sum(msk_np > 0))

      def _local(x_l, w_l, t_l, m_l):
        def _loss_fn(xi, wi):
          fn = flint_loss.dense_ce_loss_sum if use_dense else flint_loss.chunked_lm_head_ce_sum
          return fn(xi, wi, t_l, m_l, cfg) / total_valid

        l_loc, (dx_l, dw_l) = jax.value_and_grad(_loss_fn, argnums=(0, 1))(x_l, w_l)
        g_l = jax.lax.psum(l_loc, ("fsdp", "cp", "ep"))
        dw_sq = jax.lax.psum(
            jnp.sum(jnp.square(dw_l.astype(jnp.float32))),
            ("fsdp", "cp", "ep"),
        )
        return g_l, dx_l, dw_l, jnp.sqrt(dw_sq)

      fn = jax.jit(
          jax.shard_map(
              _local,
              mesh=mesh,
              in_specs=(act_spec, w_spec, b_spec, b_spec),
              out_specs=(PartitionSpec(), act_spec, w_spec, PartitionSpec()),
              check_vma=False,
          )
      )
      loss_v, dx_v, dw_v, dwn_v = fn(x_sh, w_sh, tgt_sh, msk_sh)
      return (
          float(loss_v),
          np.asarray(dx_v, dtype=np.float32),
          np.asarray(dw_v, dtype=np.float32),
          float(dwn_v),
      )

    ref_loss, ref_dx, ref_dw, ref_dwn = _eval_ce(use_dense=True, num_chunks=17, tok_chunk=1024, sc_mode="off")
    report: list[str] = []
    for label, nc, tc, scm in (
        ("ce17_off", 17, 1024, "off"),
        ("ce3_off", 3, 1024, "off"),
        ("ce17_dual_sc", 17, 1024, "dual_sc"),
        ("ce3_dual_sc", 3, 1024, "dual_sc"),
        ("ce3_tiled512_dual_sc", 3, 512, "dual_sc"),
    ):
      lv, dxv, dwv, dwnv = _eval_ce(use_dense=False, num_chunks=nc, tok_chunk=tc, sc_mode=scm)
      dx_cos = _cosine_similarity(dxv, ref_dx)
      dw_cos = _cosine_similarity(dwv, ref_dw)
      dw_max = float(np.max(np.abs(dwv - ref_dw)))
      rel_dwn = abs(dwnv - ref_dwn) / max(ref_dwn, 1e-12)
      if abs(lv - ref_loss) > 1e-3 or dx_cos < 0.9999 or dw_cos < 0.9999 or rel_dwn > 1e-3:
        report.append(
            f"{label}: loss={lv:.6f} (ref={ref_loss:.6f}), dwn={dwnv:.6f}"
            f" (ref={ref_dwn:.6f}, rel={rel_dwn:.3e}), dx_cos={dx_cos:.6f},"
            f" dw_cos={dw_cos:.6f}, dw_max={dw_max:.3e}"
        )

    def _eval_emb(sc_mode: str) -> tuple[np.ndarray, np.ndarray, float]:
      cfg = flint_config.flint_40b_config(
          fsdp_size=8,
          cp_size=1,
          ep_size=1,
          global_batch_size=gbs,
          seq_len=seq_len,
          sc_collective_mode=sc_mode,
          sc_min_collective_bytes=4096,
      )

      def _local_e(t_l, e_l, dy_l):
        y_l, vjp_fn = jax.vjp(lambda ew: flint_model.embed_tokens_local(t_l, ew, cfg), e_l)
        (de_l,) = vjp_fn(dy_l)
        de_sq = jax.lax.psum(
            jnp.sum(jnp.square(de_l.astype(jnp.float32))),
            ("fsdp", "cp", "ep"),
        )
        return y_l, de_l, jnp.sqrt(de_sq)

      fn = jax.jit(
          jax.shard_map(
              _local_e,
              mesh=mesh,
              in_specs=(b_spec, emb_spec, act_spec),
              out_specs=(act_spec, emb_spec, PartitionSpec()),
              check_vma=False,
          )
      )
      yv, dev, den = fn(tok_sh, emb_sh, dy_sh)
      return (
          np.asarray(yv, dtype=np.float32),
          np.asarray(dev, dtype=np.float32),
          float(den),
      )

    emb_y_off, emb_de_off, emb_den_off = _eval_emb("off")
    emb_y_sc, emb_de_sc, emb_den_sc = _eval_emb("dual_sc")
    emb_y_cos = _cosine_similarity(emb_y_sc, emb_y_off)
    emb_de_cos = _cosine_similarity(emb_de_sc, emb_de_off)
    if emb_y_cos < 0.99999 or emb_de_cos < 0.9999:
      report.append(
          f"embed_dual_sc: y_cos={emb_y_cos:.6f}, de_cos={emb_de_cos:.6f},"
          f" den={emb_den_sc:.6f} (ref={emb_den_off:.6f})"
      )

    self.assertEmpty(report, msg="40B lm_head/embed failures:\n" + "\n".join(report))

  def _make_stacked_r1_fixture(
      self,
  ) -> tuple[
      Any,
      flint_config.FlintConfig,
      jax.sharding.Mesh,
      flint_types.FlintBatch,
      dict[str, dict[str, Any]],
  ]:
    gbs = 8
    seq_len = 2048
    vocab_size = 3072

    def _cfg(**kw: Any) -> flint_config.FlintConfig:
      base_kw: dict[str, Any] = dict(
          vocab_size=vocab_size,
          hidden_size=512,
          intermediate_size=1536,
          moe_intermediate_size=128,
          shared_expert_intermediate_size=256,
          num_hidden_layers=9,
          num_dense_prefix_layers=2,
          moe_cycle=4,
          num_attention_heads=8,
          num_key_value_heads=2,
          head_dim=128,
          num_experts=16,
          num_experts_per_tok=6,
          sliding_window=513,
          bos_token_id=3000,
          pad_token_id=3001,
          scan_layers=True,
          fsdp_size=8,
          cp_size=1,
          ep_size=1,
          num_vocab_chunks=16,
          vocab_token_chunk_size=1024,
          sc_min_collective_bytes=4096,
          global_batch_size=gbs,
          seq_len=seq_len,
          compute_dtype="bfloat16",
          param_dtype="float32",
          remat_policy="layer",
          learning_rate=2e-3,
          warmup_steps=1,
          total_steps=10,
          lr_schedule="warmup_cosine",
      )
      base_kw.update(kw)
      use_splash = bool(base_kw.pop("use_splash", True))
      return flint_config.flint_40b_config(use_splash=use_splash, **base_kw)

    base_cfg = _cfg()
    mesh = flint_sharding.create_mesh(base_cfg.sharding)
    b_sharding = NamedSharding(mesh, flint_sharding.batch_pspec())

    # Build a Zipfian 4-segment packed batch with boundary resets + tail pad.
    rng = np.random.default_rng(2026)
    bounds = [0, 310, 790, 1410, 1980]
    tok_np = np.full((gbs, seq_len), 3001, dtype=np.int32)
    tgt_np = np.full((gbs, seq_len), -1, dtype=np.int32)
    pos_np = np.zeros((gbs, seq_len), dtype=np.int32)
    seg_np = np.zeros((gbs, seq_len), dtype=np.int32)
    msk_np = np.zeros((gbs, seq_len), dtype=np.int32)
    for b in range(gbs):
      shift = (b * 17) % 60
      b_bounds = [
          0,
          bounds[1] + shift,
          bounds[2] + shift,
          bounds[3] + shift,
          bounds[4],
      ]
      for s_idx in range(4):
        st, en = b_bounds[s_idx], b_bounds[s_idx + 1]
        slen = en - st
        raw_docs = np.where(
            rng.random(slen + 1) < 0.55,
            rng.integers(0, 8, size=slen + 1, dtype=np.int32),
            rng.integers(0, vocab_size - 80, size=slen + 1, dtype=np.int32),
        )
        raw_docs[0] = 3000
        tok_np[b, st:en] = raw_docs[:-1]
        tgt_np[b, st : en - 1] = raw_docs[1:-1]
        tgt_np[b, en - 1] = -1
        pos_np[b, st:en] = np.arange(slen, dtype=np.int32)
        seg_np[b, st:en] = s_idx + 1
        msk_np[b, st : en - 1] = 1

    batch = flint_types.FlintBatch(
        tokens=jax.device_put(jnp.asarray(tok_np), b_sharding),
        targets=jax.device_put(jnp.asarray(tgt_np), b_sharding),
        positions=jax.device_put(jnp.asarray(pos_np), b_sharding),
        segment_ids=jax.device_put(jnp.asarray(seg_np), b_sharding),
        loss_mask=jax.device_put(jnp.asarray(msk_np), b_sharding),
    )

    k1_kw: dict[str, Any] = {"moe_impl": "pallas_gmm_v2"}
    k2_kw: dict[str, Any] = {
        "sc_collective_mode": "dual_sc",
        "routed_w_tiled": True,
    }
    k3_kw: dict[str, Any] = {"remat_policy": "selective"}
    k4_kw: dict[str, Any] = {"moe_bijective_permute": True}
    k5_fuse_kw: dict[str, Any] = {"fuse_layer_projections": True}
    k5_ce3_kw: dict[str, Any] = {
        "num_vocab_chunks": 3,
        "vocab_token_chunk_size": 2048,
    }
    k5_kw: dict[str, Any] = {**k5_fuse_kw, **k5_ce3_kw}
    k6_kw: dict[str, Any] = {
        "splash_global_block_q": 1024,
        "splash_global_block_kv": 1024,
        "splash_global_block_kv_compute": 512,
        "splash_global_block_q_dkv": 1024,
        "splash_global_block_kv_dkv": 1024,
        "splash_global_block_kv_dkv_compute": 1024,
        "splash_qk_diag_skip": True,
    }
    knobs = {
        "k1": k1_kw,
        "k2": k2_kw,
        "k3": k3_kw,
        "k4": k4_kw,
        "k5_fuse": k5_fuse_kw,
        "k5_ce3": k5_ce3_kw,
        "k5": k5_kw,
        "k6": k6_kw,
        "all6": {**k1_kw, **k2_kw, **k3_kw, **k4_kw, **k5_kw, **k6_kw},
    }
    return _cfg, base_cfg, mesh, batch, knobs

  def test_stacked_r1_knobs_zipf_packed_grad_parity(self):
    _cfg, base_cfg, mesh, batch, kn = self._make_stacked_r1_fixture()
    base_params = flint_sharding.init_random_params(jax.random.key(42), mesh, base_cfg)

    def _run_vg(c: flint_config.FlintConfig):
      @jax.jit
      def _fn(p, b):
        return flint_model.shard_mapped_value_and_grad(p, b, c, mesh, fp32_grads=True)

      (loss_v, ce_v, _), grads = _fn(base_params, batch)
      gnorm = float(flint_train.compute_global_grad_norm_f32(grads))
      flat_leaves = []
      for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        name = ".".join(str(k) for k in path)
        flat_leaves.append((name, np.asarray(leaf, dtype=np.float32)))
      return float(loss_v), float(ce_v), gnorm, flat_leaves

    ref_loss, _, ref_gn, ref_leaves = _run_vg(base_cfg)
    cases = [
        ("k1", kn["k1"]),
        ("k2", kn["k2"]),
        ("k5_fuse", kn["k5_fuse"]),
        ("k5_ce3", kn["k5_ce3"]),
        ("k6", kn["k6"]),
        ("k1_k2", {**kn["k1"], **kn["k2"]}),
        ("k2_k5_fuse", {**kn["k2"], **kn["k5_fuse"]}),
        ("k2_k5_ce3", {**kn["k2"], **kn["k5_ce3"]}),
        ("k1_k5", {**kn["k1"], **kn["k5"]}),
        ("k2_k6", {**kn["k2"], **kn["k6"]}),
        ("k5_k6", {**kn["k5"], **kn["k6"]}),
        (
            "k12456",
            {**kn["k1"], **kn["k2"], **kn["k4"], **kn["k5"], **kn["k6"]},
        ),
        ("all6", kn["all6"]),
    ]

    failures: list[str] = []
    for cname, ckw in cases:
      lv, _, gn, leaves = _run_vg(_cfg(**ckw))
      bad_leaves: list[str] = []
      for (name, arr), (_, rarr) in zip(leaves, ref_leaves, strict=True):
        rn = float(np.linalg.norm(rarr.astype(np.float64)))
        cn = float(np.linalg.norm(arr.astype(np.float64)))
        if rn == 0.0 and cn == 0.0:
          continue
        cos = _cosine_similarity(arr, rarr)
        rel = abs(cn - rn) / max(rn, 1e-12)
        if not np.all(np.isfinite(arr)) or cos < 0.998 or rel > 1e-2:
          bad_leaves.append(f"{name}(rn={rn:.5f},cn={cn:.5f},cos={cos:.5f},rel={rel:.2e})")
      rel_gn = abs(gn - ref_gn) / max(ref_gn, 1e-12)
      if abs(lv - ref_loss) > 2e-3 or rel_gn > 5e-3 or bad_leaves:
        failures.append(
            f"[{cname}] loss={lv:.6f}(ref={ref_loss:.6f})"
            f" gn={gn:.6f}(ref={ref_gn:.6f},rel={rel_gn:.2e})"
            f" bad({len(bad_leaves)})={bad_leaves[:8]}"
        )

    self.assertEmpty(
        failures,
        msg="Stacked R1 grad parity failed:\n" + "\n".join(failures),
    )

  def test_stacked_r1_knobs_zipf_packed_10step_convergence(self):
    _cfg, base_cfg, mesh, batch, kn = self._make_stacked_r1_fixture()

    def _run_10s(c: flint_config.FlintConfig) -> list[tuple[float, float]]:
      st = flint_train.init_train_state(jax.random.key(42), mesh, c)
      step_fn = flint_train.make_train_step(c, mesh, donate_state=True)
      traj = []
      for _ in range(10):
        st, m = step_fn(st, batch)
        traj.append((float(m.loss), float(m.grad_norm)))
      return traj

    traj_base = _run_10s(base_cfg)
    traj_all6 = _run_10s(_cfg(**kn["all6"]))
    self.assertLess(traj_all6[-1][0], traj_all6[0][0])
    loss_diff_9 = abs(traj_all6[-1][0] - traj_base[-1][0])
    self.assertLessEqual(
        loss_diff_9,
        0.02,
        msg=(f"[10step_all6_vs_base] base={traj_base} vs all6={traj_all6}" f" (diff9={loss_diff_9:.5f})"),
    )

  def _make_dloc16_eshard2_fixture(
      self,
  ) -> tuple[
      Any,
      flint_config.FlintConfig,
      jax.sharding.Mesh,
      flint_types.FlintBatch,
      dict[str, dict[str, Any]],
  ]:
    gbs = 8
    seq_len = 1024
    vocab_size = 3456  # v_chunk = 3456 // 3 = 1152 = 9 * 128 (odd multiple)

    def _cfg(**kw: Any) -> flint_config.FlintConfig:
      base_kw: dict[str, Any] = dict(
          vocab_size=vocab_size,
          hidden_size=128,  # d_loc = 128 // 8 = 16 (exact 40B/128 twin)
          intermediate_size=384,
          moe_intermediate_size=128,
          shared_expert_intermediate_size=256,
          num_hidden_layers=9,
          num_dense_prefix_layers=2,
          moe_cycle=4,
          num_attention_heads=6,
          num_key_value_heads=2,
          head_dim=128,
          num_experts=16,  # e_shard = 16 // 8 = 2 (exact 256/128 twin)
          num_experts_per_tok=6,
          sliding_window=513,
          bos_token_id=3400,
          pad_token_id=3401,
          scan_layers=True,
          fsdp_size=8,
          cp_size=1,
          ep_size=1,
          num_vocab_chunks=16,
          vocab_token_chunk_size=1024,
          sc_min_collective_bytes=4096,
          global_batch_size=gbs,
          seq_len=seq_len,
          compute_dtype="bfloat16",
          param_dtype="float32",
          remat_policy="layer",
          learning_rate=2e-3,
          warmup_steps=1,
          total_steps=8,
          lr_schedule="warmup_cosine",
      )
      base_kw.update(kw)
      use_splash = bool(base_kw.pop("use_splash", True))
      return flint_config.flint_40b_config(use_splash=use_splash, **base_kw)

    base_cfg = _cfg()
    mesh = flint_sharding.create_mesh(base_cfg.sharding)
    b_sharding = NamedSharding(mesh, flint_sharding.batch_pspec())

    rng = np.random.default_rng(4242)
    bounds = [0, 180, 460, 740, 990]
    tok_np = np.full((gbs, seq_len), 3401, dtype=np.int32)
    tgt_np = np.full((gbs, seq_len), -1, dtype=np.int32)
    pos_np = np.zeros((gbs, seq_len), dtype=np.int32)
    seg_np = np.zeros((gbs, seq_len), dtype=np.int32)
    msk_np = np.zeros((gbs, seq_len), dtype=np.int32)
    for b in range(gbs):
      shift = (b * 11) % 30
      b_bounds = [
          0,
          bounds[1] + shift,
          bounds[2] + shift,
          bounds[3] + shift,
          bounds[4],
      ]
      for s_idx in range(4):
        st, en = b_bounds[s_idx], b_bounds[s_idx + 1]
        slen = en - st
        raw_docs = np.where(
            rng.random(slen + 1) < 0.55,
            rng.integers(0, 8, size=slen + 1, dtype=np.int32),
            rng.integers(0, vocab_size - 80, size=slen + 1, dtype=np.int32),
        )
        raw_docs[0] = 3400
        tok_np[b, st:en] = raw_docs[:-1]
        tgt_np[b, st : en - 1] = raw_docs[1:-1]
        tgt_np[b, en - 1] = -1
        pos_np[b, st:en] = np.arange(slen, dtype=np.int32)
        seg_np[b, st:en] = s_idx + 1
        msk_np[b, st : en - 1] = 1

    batch = flint_types.FlintBatch(
        tokens=jax.device_put(jnp.asarray(tok_np), b_sharding),
        targets=jax.device_put(jnp.asarray(tgt_np), b_sharding),
        positions=jax.device_put(jnp.asarray(pos_np), b_sharding),
        segment_ids=jax.device_put(jnp.asarray(seg_np), b_sharding),
        loss_mask=jax.device_put(jnp.asarray(msk_np), b_sharding),
    )

    r2_base_kw: dict[str, Any] = dict(
        moe_impl="pallas_gmm_v2",
        sc_collective_mode="dual_sc",
        routed_w_tiled=True,
        remat_policy="selective",
        moe_bijective_permute=True,
        fuse_layer_projections=True,
        num_vocab_chunks=3,
        vocab_token_chunk_size=1024,
        splash_global_block_q=512,
        splash_global_block_kv=512,
        splash_global_block_kv_compute=256,
        splash_global_block_q_dkv=512,
        splash_global_block_kv_dkv=512,
        splash_global_block_kv_dkv_compute=512,
        splash_qk_diag_skip=True,
    )
    knobs: dict[str, dict[str, Any]] = {
        "r2_base": r2_base_kw,
        "r2_k1b_rep_router": {**r2_base_kw, "replicate_router_gate": True},
        "r2_k2_expert_shard": {
            **r2_base_kw,
            "shard_routed_on_expert_axis": True,
            "routed_w_tiled": False,
        },
        "r2_k3_slot_major": {**r2_base_kw, "moe_slot_major_pack": True},
        "r2_k4_extra_res": {
            **r2_base_kw,
            "save_extra_selective_residuals": True,
        },
        "r2_k5_splash_b2_sv": {
            **r2_base_kw,
            "splash_global_block_kv_compute": 512,
            "splash_sv_diag_skip": True,
            "splash_use_base2_exp": True,
        },
        "r2_k6_prefetch_gu": {**r2_base_kw, "prefetch_routed_weights": True},
        "r2_k7_fp8_ag": {**r2_base_kw, "fp8_routed_all_gather": True},
        "r2_all_exact": {
            **r2_base_kw,
            "replicate_router_gate": True,
            "shard_routed_on_expert_axis": True,
            "routed_w_tiled": False,
            "moe_slot_major_pack": True,
            "save_extra_selective_residuals": True,
            "splash_global_block_kv_compute": 512,
            "splash_sv_diag_skip": True,
            "splash_use_base2_exp": True,
            "prefetch_routed_weights": True,
        },
    }
    return _cfg, base_cfg, mesh, batch, knobs

  def test_dloc16_eshard2_hardware_twin_r2_grad_parity(self):
    _cfg, base_cfg, mesh, batch, kn = self._make_dloc16_eshard2_fixture()
    base_params = flint_sharding.init_random_params(jax.random.key(42), mesh, base_cfg)
    host_params = jax.tree.map(np.asarray, base_params)

    # Sharding-invariant PRNG init across new PartitionSpecs (R2-K1b + R2-K2).
    all_init_params = flint_sharding.init_random_params(jax.random.key(42), mesh, _cfg(**kn["r2_all_exact"]))
    for p_got, p_ref in zip(
        jax.tree.leaves(all_init_params),
        jax.tree.leaves(host_params),
        strict=True,
    ):
      np.testing.assert_array_equal(np.asarray(p_got), p_ref)

    def _run_vg(c: flint_config.FlintConfig):
      p_c = jax.tree.map(
          jax.device_put,
          host_params,
          flint_sharding.named_param_shardings(mesh, c),
      )

      @jax.jit
      def _fn(p, b):
        return flint_model.shard_mapped_value_and_grad(p, b, c, mesh, fp32_grads=True)

      (loss_v, ce_v, _), grads = _fn(p_c, batch)
      gnorm = float(flint_train.compute_global_grad_norm_f32(grads))
      flat_leaves = []
      for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        name = ".".join(str(k) for k in path)
        flat_leaves.append((name, np.asarray(leaf, dtype=np.float32)))
      return float(loss_v), float(ce_v), gnorm, flat_leaves

    def _check_against(
        cname: str,
        got: tuple[float, float, float, list[tuple[str, np.ndarray]]],
        ref: tuple[float, float, float, list[tuple[str, np.ndarray]]],
        *,
        ref_tag: str,
        min_cos: float,
        max_rel: float,
        norm_atol: float,
        max_dloss: float,
        max_rel_gn: float,
        out_failures: list[str],
    ) -> None:
      lv, _, gn, leaves = got
      rlv, _, rgn, rleaves = ref
      bad_leaves: list[str] = []
      for (name, arr), (_, rarr) in zip(leaves, rleaves, strict=True):
        rn = float(np.linalg.norm(rarr.astype(np.float64)))
        cn = float(np.linalg.norm(arr.astype(np.float64)))
        if rn == 0.0 and cn == 0.0:
          continue
        cos = _cosine_similarity(arr, rarr)
        dn = abs(cn - rn)
        if not np.all(np.isfinite(arr)) or cos < min_cos or dn > max_rel * rn + norm_atol:
          bad_leaves.append(f"{name}(rn={rn:.5f},cn={cn:.5f},cos={cos:.5f},dn={dn:.2e})")
      rel_gn = abs(gn - rgn) / max(rgn, 1e-12)
      if abs(lv - rlv) > max_dloss or rel_gn > max_rel_gn or bad_leaves:
        out_failures.append(
            f"[{cname} vs {ref_tag}] loss={lv:.6f}(ref={rlv:.6f})"
            f" gn={gn:.6f}(ref={rgn:.6f},rel={rel_gn:.2e})"
            f" bad({len(bad_leaves)})={bad_leaves[:8]}"
        )

    base_res = _run_vg(base_cfg)
    r2_base_res = _run_vg(_cfg(**kn["r2_base"]))
    r2_k5_res = _run_vg(_cfg(**kn["r2_k5_splash_b2_sv"]))
    r2_all_exact_res = _run_vg(_cfg(**kn["r2_all_exact"]))
    r2_k7_res = _run_vg(_cfg(**kn["r2_k7_fp8_ag"]))

    failures: list[str] = []
    # 1. End-to-end R1+R2 Splash/GMM parity vs ragged_dot + 256-tile base_cfg
    # (accounting for ~2e-4 BF16 floor on 6-elem sinks & d=128 Top-6 ties).
    for cname, res in (
        ("r2_base", r2_base_res),
        ("r2_k5_splash_b2_sv", r2_k5_res),
        ("r2_all_exact", r2_all_exact_res),
    ):
      _check_against(
          cname,
          res,
          base_res,
          ref_tag="base",
          min_cos=0.995,
          max_rel=1e-2,
          norm_atol=3e-4,
          max_dloss=2e-3,
          max_rel_gn=2e-3,
          out_failures=failures,
      )

    # 2. Strict lossless R2 single-knob parity directly against r2_base
    # (Dual-SC 2D expert sharding, replicated router, slot-major, extra
    # selective residuals, and gate_up prefetch must not change math).
    for cname in (
        "r2_k1b_rep_router",
        "r2_k2_expert_shard",
        "r2_k3_slot_major",
        "r2_k4_extra_res",
        "r2_k6_prefetch_gu",
    ):
      _check_against(
          cname,
          _run_vg(_cfg(**kn[cname])),
          r2_base_res,
          ref_tag="r2_base",
          min_cos=0.9999,
          max_rel=1e-3,
          norm_atol=1e-5,
          max_dloss=1e-4,
          max_rel_gn=2e-4,
          out_failures=failures,
      )

    # 3. Stacking K1b+K2+K3+K4+K6 atop K5 must match K5 nearly bit-for-bit.
    _check_against(
        "r2_all_exact",
        r2_all_exact_res,
        r2_k5_res,
        ref_tag="r2_k5",
        min_cos=0.9999,
        max_rel=1e-3,
        norm_atol=1e-5,
        max_dloss=1e-4,
        max_rel_gn=2e-4,
        out_failures=failures,
    )

    # 4. Lossy FP8 routed all-gather check vs r2_base & base_cfg.
    _check_against(
        "r2_k7_fp8_ag",
        r2_k7_res,
        r2_base_res,
        ref_tag="r2_base",
        min_cos=0.985,
        max_rel=5e-2,
        norm_atol=3e-4,
        max_dloss=2e-2,
        max_rel_gn=5e-2,
        out_failures=failures,
    )

    self.assertEmpty(
        failures,
        msg="d_loc=16 e_shard=2 R2 grad parity failed:\n" + "\n".join(failures),
    )

  def test_dloc16_eshard2_hardware_twin_r2_8step_convergence(self):
    _cfg, base_cfg, mesh, batch, kn = self._make_dloc16_eshard2_fixture()

    def _run_8s(c: flint_config.FlintConfig) -> list[tuple[float, float]]:
      st = flint_train.init_train_state(jax.random.key(42), mesh, c)
      step_fn = flint_train.make_train_step(c, mesh, donate_state=True)
      traj = []
      for _ in range(8):
        st, m = step_fn(st, batch)
        traj.append((float(m.loss), float(m.grad_norm)))
      return traj

    traj_base = _run_8s(base_cfg)
    failures: list[str] = []
    for tname, tcfg, tol in (
        ("8step_r2_base", _cfg(**kn["r2_base"]), 0.02),
        ("8step_r2_all_exact", _cfg(**kn["r2_all_exact"]), 0.02),
        (
            "8step_r2_all_fp8",
            _cfg(**kn["r2_all_exact"], fp8_routed_all_gather=True),
            0.05,
        ),
    ):
      traj_c = _run_8s(tcfg)
      diff_last = abs(traj_c[-1][0] - traj_base[-1][0])
      if not (traj_c[-1][0] < traj_c[0][0]) or diff_last > tol:
        failures.append(f"[{tname}] base={traj_base} vs cand={traj_c}" f" (diff_last={diff_last:.5f})")

    self.assertEmpty(
        failures,
        msg="d_loc=16 e_shard=2 R2 8-step failed:\n" + "\n".join(failures),
    )

  def test_v4_full_stack_cp2_grad_parity(self):
    gbs = 8
    seq_len = 2048
    vocab_size = 2048
    common_kw: dict[str, Any] = dict(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=384,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=256,
        num_hidden_layers=9,
        num_dense_prefix_layers=2,
        moe_cycle=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=128,
        num_experts_per_tok=6,
        sliding_window=513,
        bos_token_id=2000,
        pad_token_id=2001,
        scan_layers=True,
        fsdp_size=4,
        cp_size=2,
        ep_size=1,
        global_batch_size=gbs,
        seq_len=seq_len,
        compute_dtype="bfloat16",
        param_dtype="float32",
        sc_min_collective_bytes=4096,
    )
    ref_cfg = flint_config.flint_40b_config(
        remat_policy="layer",
        num_vocab_chunks=2,
        vocab_token_chunk_size=1024,
        **common_kw,
    )
    v3_cfg = flint_config.flint_40b_config(
        remat_policy="global_selective",
        moe_impl="pallas_gmm_v2",
        sc_collective_mode="dual_sc",
        routed_w_tiled=True,
        moe_bijective_permute=True,
        moe_zero_scatter_router=True,
        replicate_router_gate=True,
        fuse_layer_projections=True,
        precompute_rope_tables=True,
        prefetch_routed_weights=True,
        transpose_routed_down_ag=True,
        transpose_dense_rhs_ag=True,
        num_vocab_chunks=2,
        vocab_token_chunk_size=1024,
        splash_block_q=512,
        splash_block_kv=512,
        splash_block_kv_compute=256,
        splash_block_q_dkv=512,
        splash_block_kv_dkv=512,
        splash_block_kv_dkv_compute=512,
        splash_local_dq_reduction_steps=1,
        splash_num_stacked_q_heads=4,
        splash_num_stacked_q_heads_dkv=4,
        splash_use_local_halo_cp=True,
        splash_ignore_segment_ids=True,
        splash_global_block_q=512,
        splash_global_block_kv=1024,
        splash_global_block_kv_compute=512,
        splash_global_block_q_dkv=256,
        splash_global_block_kv_dkv=1024,
        splash_global_block_kv_dkv_compute=512,
        splash_global_num_stacked_q_heads=4,
        splash_global_num_stacked_q_heads_dkv=4,
        splash_use_base2_exp=True,
        **common_kw,
    )
    v4_cfg = flint_config.flint_40b_config(
        remat_policy="global_selective",
        moe_impl="pallas_gmm_v2",
        sc_collective_mode="dual_sc",
        routed_w_tiled=True,
        moe_bijective_permute=True,
        moe_zero_scatter_router=True,
        use_sc_moe_gather_reduce=True,
        use_sc_counting_sort=True,
        use_pallas_router_topk=True,
        compact_router_maps=True,
        skip_router_aux_stats_when_zero_coef=True,
        replicate_router_gate=True,
        fuse_layer_projections=True,
        precompute_rope_tables=True,
        prefetch_routed_weights=True,
        transpose_routed_down_ag=True,
        transpose_dense_rhs_ag=True,
        num_vocab_chunks=1,
        vocab_token_chunk_size=1024,
        use_fused_vocab_vjp_scan=True,
        splash_block_q=512,
        splash_block_kv=512,
        splash_block_kv_compute=256,
        splash_block_q_dkv=512,
        splash_block_kv_dkv=512,
        splash_block_kv_dkv_compute=512,
        splash_local_dq_reduction_steps=1,
        splash_num_stacked_q_heads=4,
        splash_num_stacked_q_heads_dkv=4,
        splash_use_local_halo_cp=True,
        splash_ignore_segment_ids=True,
        splash_global_block_q=512,
        splash_global_block_kv=1024,
        splash_global_block_kv_compute=512,
        splash_global_block_q_dkv=256,
        splash_global_block_kv_dkv=1024,
        splash_global_block_kv_dkv_compute=512,
        splash_global_num_stacked_q_heads=4,
        splash_global_num_stacked_q_heads_dkv=4,
        splash_fwd_strip_schedule="auto",
        splash_compact_lse_residual=True,
        splash_global_dkv_head_stitch=True,
        splash_use_base2_exp=True,
        use_lb_cp_layout=True,
        early_global_kv_proj=True,
        use_pallas_lb_kv_reorder=True,
        use_fused_qk_prologue=True,
        **common_kw,
    )
    mesh = flint_sharding.create_mesh(ref_cfg.sharding)
    b_sharding = NamedSharding(mesh, flint_sharding.batch_pspec())
    rng = np.random.default_rng(2029)
    tok_np = rng.integers(0, vocab_size - 64, (gbs, seq_len), dtype=np.int32)
    tgt_np = rng.integers(0, vocab_size - 64, (gbs, seq_len), dtype=np.int32)
    pos_np = np.broadcast_to(np.arange(seq_len, dtype=np.int32), (gbs, seq_len)).copy()
    seg_np = np.ones((gbs, seq_len), dtype=np.int32)
    msk_np = np.ones((gbs, seq_len), dtype=np.int32)
    msk_np[:, -16:] = 0
    tgt_np[:, -16:] = -1
    batch = flint_types.FlintBatch(
        tokens=jax.device_put(jnp.asarray(tok_np), b_sharding),
        targets=jax.device_put(jnp.asarray(tgt_np), b_sharding),
        positions=jax.device_put(jnp.asarray(pos_np), b_sharding),
        segment_ids=jax.device_put(jnp.asarray(seg_np), b_sharding),
        loss_mask=jax.device_put(jnp.asarray(msk_np), b_sharding),
    )

    ref_params = flint_sharding.init_random_params(jax.random.key(77), mesh, ref_cfg)
    host_params = jax.tree.map(np.asarray, ref_params)
    v3_params = jax.tree.map(
        jax.device_put,
        host_params,
        flint_sharding.named_param_shardings(mesh, v3_cfg),
    )
    v4_params = jax.tree.map(
        jax.device_put,
        host_params,
        flint_sharding.named_param_shardings(mesh, v4_cfg),
    )

    @jax.jit
    def _vg_ref(p, b):
      return flint_model.shard_mapped_value_and_grad(p, b, ref_cfg, mesh, fp32_grads=True)

    @jax.jit
    def _vg_v3(p, b):
      return flint_model.shard_mapped_value_and_grad(p, b, v3_cfg, mesh, fp32_grads=True)

    @jax.jit
    def _vg_v4(p, b):
      return flint_model.shard_mapped_value_and_grad(p, b, v4_cfg, mesh, fp32_grads=True)

    hlo_v4 = _vg_v4.lower(v4_params, batch).as_text("hlo", debug_info=True)
    for marker in (
        "pallas_qk_prologue",
        "lb_kv_vmem_copy",
        "splash_mha_dkv_no_residuals_head_stitch",
        "router_topk",
        "sc_counting_sort",
        "dense_gather_reduce",
    ):
      self.assertIn(marker, hlo_v4)

    v4_exact_cfg = dataclasses.replace(
        v3_cfg,
        kernels=dataclasses.replace(
            v3_cfg.kernels,
            use_pallas_router_topk=True,
            use_sc_counting_sort=True,
            compact_router_maps=True,
            skip_router_aux_stats_when_zero_coef=True,
            num_vocab_chunks=1,
            vocab_token_chunk_size=1024,
            use_fused_vocab_vjp_scan=True,
            splash_fwd_strip_schedule="auto",
            splash_compact_lse_residual=True,
            splash_global_dkv_head_stitch=True,
        ),
    )
    v4_exact_params = jax.tree.map(
        jax.device_put,
        host_params,
        flint_sharding.named_param_shardings(mesh, v4_exact_cfg),
    )

    @jax.jit
    def _vg_v4_exact(p, b):
      return flint_model.shard_mapped_value_and_grad(p, b, v4_exact_cfg, mesh, fp32_grads=True)

    (loss_ref, ce_ref, _), grads_ref = _vg_ref(ref_params, batch)
    (loss_v3, ce_v3, _), grads_v3 = _vg_v3(v3_params, batch)
    (loss_v4_ex, ce_v4_ex, aux_v4_ex), grads_v4_ex = _vg_v4_exact(v4_exact_params, batch)
    (loss_v4, ce_v4, aux_v4), grads_v4 = _vg_v4(v4_params, batch)
    self.assertEqual(float(aux_v4_ex), 0.0)
    self.assertEqual(float(aux_v4), 0.0)
    self.assertLess(abs(float(loss_v4_ex) - float(loss_v3)), 1e-5)
    self.assertLess(abs(float(ce_v4_ex) - float(ce_v3)), 1e-5)
    self.assertLess(abs(float(loss_v4) - float(loss_v3)), 1e-3)
    self.assertLess(abs(float(ce_v4) - float(ce_v3)), 1e-3)
    self.assertLess(abs(float(loss_v4) - float(loss_ref)), 5e-3)
    self.assertLess(abs(float(ce_v4) - float(ce_ref)), 5e-3)

    def _grad_global_norm(g_tree) -> float:
      sq = 0.0
      for leaf in jax.tree.leaves(g_tree):
        arr = np.asarray(leaf, dtype=np.float64).reshape(-1)
        sq += float(np.dot(arr, arr))
      return float(np.sqrt(sq))

    gn_ref = _grad_global_norm(grads_ref)
    gn_v3 = _grad_global_norm(grads_v3)
    gn_v4 = _grad_global_norm(grads_v4)
    self.assertLess(abs(gn_v4 - gn_v3) / max(gn_v3, 1e-12), 2e-3)
    self.assertLess(abs(gn_v4 - gn_ref) / max(gn_ref, 1e-12), 5e-3)

    def _collect_bad(
        g_cand,
        g_base,
        *,
        cont_min_cos: float,
        routed_min_cos: float,
        router_min_cos: float,
    ) -> list[str]:
      bad = []
      for (path, g_got), (_, g_b) in zip(
          jax.tree_util.tree_leaves_with_path(g_cand),
          jax.tree_util.tree_leaves_with_path(g_base),
          strict=True,
      ):
        name = ".".join(str(k) for k in path)
        a = np.asarray(g_got, dtype=np.float64).reshape(-1)
        b = np.asarray(g_b, dtype=np.float64).reshape(-1)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        denom = na * nb
        if denom > 1e-8:
          cos = float(np.dot(a, b) / denom)
          if "router" in name and "gate" in name:
            req_cos = router_min_cos
          elif "routed_experts" in name:
            req_cos = routed_min_cos
          else:
            req_cos = cont_min_cos
          if cos < req_cos:
            bad.append(
                f"{name}(shape={tuple(g_got.shape)},na={na:.5f},nb={nb:.5f},"
                f"cos={cos:.6f}<{req_cos},dn={abs(na - nb):.2e})"
            )
      return bad

    # 1. Exact/lossless v4 stack (Splash strip+compact-LSE+head-stitch, fused
    # single-scan CE, Pallas router top-k, Dual-SC counting sort, compact router
    # maps, zero-coef aux skip) must match v3 nearly bit-for-bit across all 128
    # experts (`cos >= 0.9998` on every leaf).
    bad_exact_vs_v3 = _collect_bad(
        grads_v4_ex,
        grads_v3,
        cont_min_cos=0.9998,
        routed_min_cos=0.9998,
        router_min_cos=0.9998,
    )
    self.assertEmpty(bad_exact_vs_v3, msg=f"v4_exact vs v3 failed: {bad_exact_vs_v3}")

    # 2. Full v4 stack adding 2-stripe LB CP + early KV + Pallas VMEM prologue
    # (FP32 RoPE) + Dual-SC gather-reduce (pairwise RNE tree) preserves `<1e-3`
    # loss parity, `<2e-3` global grad norm parity, `>=0.98` continuous-leaf
    # cosine, and expected discrete Top-6 128-expert routing cosine (`>=0.90`).
    bad_vs_v3 = _collect_bad(
        grads_v4,
        grads_v3,
        cont_min_cos=0.98,
        routed_min_cos=0.94,
        router_min_cos=0.90,
    )
    bad_vs_ref = _collect_bad(
        grads_v4,
        grads_ref,
        cont_min_cos=0.98,
        routed_min_cos=0.94,
        router_min_cos=0.90,
    )
    self.assertEmpty(
        bad_vs_v3 + bad_vs_ref,
        msg=f"v4 vs v3: {bad_vs_v3}\nv4 vs ref: {bad_vs_ref}",
    )


if __name__ == "__main__":
  absltest.main()
