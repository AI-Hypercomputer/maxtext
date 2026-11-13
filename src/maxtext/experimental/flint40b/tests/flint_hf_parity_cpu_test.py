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

"""End-to-end 8-CPU-device HF parity, custom-VJP CE, and AdamW tests."""

from __future__ import annotations

import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np
import torch

from maxtext.experimental.flint40b import flint_checkpoint
from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_loss
from maxtext.experimental.flint40b import flint_model
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import flint_sparse_layer
from maxtext.experimental.flint40b import flint_train
from maxtext.experimental.flint40b import flint_types
from maxtext.experimental.flint40b.tests import hf_reference

P = PartitionSpec


def _max_abs_diff(a: np.ndarray | jax.Array, b: np.ndarray | jax.Array) -> float:
  a_f64 = np.asarray(a, dtype=np.float64)
  b_f64 = np.asarray(b, dtype=np.float64)
  return float(np.max(np.abs(a_f64 - b_f64)))


def _recenter_rmsnorm_weights(
    module: hf_reference.TorchFlintForCausalLM,
    seed: int = 0,
    low: float = 0.75,
    high: float = 1.25,
) -> None:
  """Conditions `*norm*.weight` and `1/sqrt(2*L)` residual projections."""
  gen = torch.Generator(device="cpu")
  gen.manual_seed(seed)
  num_layers = module.config.num_hidden_layers
  resid_scale = 1.0 / math.sqrt(2.0 * float(num_layers))
  with torch.no_grad():
    for name, param in module.named_parameters():
      if "norm" in name and name.endswith("weight"):
        vals = torch.empty(param.shape, dtype=torch.float32).uniform_(low, high, generator=gen)
        param.copy_(vals.to(device=param.device, dtype=param.dtype))
      elif name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
        param.mul_(resid_scale)


def _put_sharded_batch(
    tokens: np.ndarray,
    targets: np.ndarray,
    positions: np.ndarray,
    segment_ids: np.ndarray,
    loss_mask: np.ndarray,
    mesh: Mesh,
) -> flint_types.FlintBatch:
  sharding = NamedSharding(mesh, flint_sharding.batch_pspec())

  def _put(arr: np.ndarray) -> jax.Array:
    return jax.make_array_from_callback(arr.shape, sharding, lambda idx: arr[idx].copy())

  return flint_types.FlintBatch(
      tokens=_put(tokens),
      targets=_put(targets),
      positions=_put(positions),
      segment_ids=_put(segment_ids),
      loss_mask=_put(loss_mask),
  )


def _make_packed_test_arrays(
    batch_size: int = 8,
    seq_len: int = 32,
    vocab_size: int = 256,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Builds 2-segment packed sequences with non-trivial `loss_mask` zeros."""
  rng = np.random.default_rng(seed)
  tokens = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
  half = seq_len // 2
  targets = np.zeros_like(tokens)
  targets[:, : half - 1] = tokens[:, 1:half]
  targets[:, half : seq_len - 1] = tokens[:, half + 1 : seq_len]
  targets[:, half - 1] = -100  # Deliberately out-of-range on masked boundary.
  targets[:, seq_len - 1] = vocab_size + 55

  pos_1d = np.concatenate([np.arange(half, dtype=np.int32), np.arange(half, dtype=np.int32)])
  seg_1d = np.concatenate(
      [
          np.ones(half, dtype=np.int32),
          np.full(half, 2, dtype=np.int32),
      ]
  )
  positions = np.broadcast_to(pos_1d[None, :], (batch_size, seq_len)).copy()
  segment_ids = np.broadcast_to(seg_1d[None, :], (batch_size, seq_len)).copy()

  loss_mask = np.ones((batch_size, seq_len), dtype=np.int32)
  loss_mask[:, half - 1] = 0
  loss_mask[:, seq_len - 1] = 0
  loss_mask[:, 3] = 0
  loss_mask[0, half + 2] = 0
  return tokens, targets, positions, segment_ids, loss_mask


def _build_mesh_for_dims(mesh_dims: tuple[int, int, int]) -> Mesh:
  sharding_cfg = flint_config.FlintShardingConfig(
      fsdp_size=mesh_dims[0],
      cp_size=mesh_dims[1],
      ep_size=mesh_dims[2],
  )
  devs = jax.devices()[: sharding_cfg.num_devices]
  return flint_sharding.create_mesh(sharding_cfg, devices=devs)


class FlintHfParityCpuTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.assertGreaterEqual(len(jax.devices()), 8)

  def test_chunked_ce_custom_vjp_matches_dense_ce_fwd_and_bwd(self):
    rng = np.random.default_rng(19)
    bsz, seq_len, hidden_size, vocab_size = 8, 32, 128, 256
    x_np = rng.standard_normal((bsz, seq_len, hidden_size)).astype(np.float32)
    w_np = (rng.standard_normal((hidden_size, vocab_size)) / np.sqrt(hidden_size)).astype(np.float32)
    _, targets_np, _, _, mask_np = _make_packed_test_arrays(
        batch_size=bsz, seq_len=seq_len, vocab_size=vocab_size, seed=21
    )
    total_valid = float(np.sum(mask_np > 0))
    self.assertGreater(total_valid, 0.0)

    act_spec = flint_sharding.activation_pspec()
    b_spec = flint_sharding.batch_pspec()
    w_spec = P(flint_config.ALL_AXES, None)

    for mesh_dims in ((1, 1, 1), (4, 2, 1), (2, 2, 2)):
      for tok_chunk in (8, 7):  # `7` exercises non-divisible token tile pad.
        cfg = flint_config.mini_flint_config(
            fsdp_size=mesh_dims[0],
            cp_size=mesh_dims[1],
            ep_size=mesh_dims[2],
            global_batch_size=bsz,
            seq_len=seq_len,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_vocab_chunks=4,
            vocab_token_chunk_size=tok_chunk,
            compute_dtype="float32",
            param_dtype="float32",
        )
        mesh = _build_mesh_for_dims(mesh_dims)

        x_sh = jax.device_put(x_np, NamedSharding(mesh, act_spec))
        w_sh = jax.device_put(w_np, NamedSharding(mesh, w_spec))
        t_sh = jax.device_put(targets_np, NamedSharding(mesh, b_spec))
        m_sh = jax.device_put(mask_np, NamedSharding(mesh, b_spec))

        def _make_vg(loss_fn, cfg_local=cfg, mesh_local=mesh):
          def _local(x_l, w_l, t_l, m_l):
            def _mean_ce(x_in, w_in):
              return loss_fn(x_in, w_in, t_l, m_l, cfg_local) / total_valid

            val, (dx_l, dw_l) = jax.value_and_grad(_mean_ce, argnums=(0, 1))(x_l, w_l)
            if cfg_local.sharding.num_devices > 1:
              val = jax.lax.psum(val, flint_config.ALL_AXES)
            return val, dx_l, dw_l

          return jax.jit(
              jax.shard_map(
                  _local,
                  mesh=mesh_local,
                  in_specs=(act_spec, w_spec, b_spec, b_spec),
                  out_specs=(P(), act_spec, w_spec),
                  check_vma=False,
              )
          )

        chunk_val, chunk_dx, chunk_dw = _make_vg(flint_loss.chunked_lm_head_ce_sum)(x_sh, w_sh, t_sh, m_sh)
        dense_val, dense_dx, dense_dw = _make_vg(flint_loss.dense_ce_loss_sum)(x_sh, w_sh, t_sh, m_sh)

        with self.subTest(mesh_dims=mesh_dims, tok_chunk=tok_chunk):
          self.assertLess(_max_abs_diff(chunk_val, dense_val), 1e-5)
          rel_loss = abs(float(chunk_val) - float(dense_val)) / max(abs(float(dense_val)), 1.0)
          self.assertLess(rel_loss, 1e-6)
          self.assertLess(_max_abs_diff(chunk_dx, dense_dx), 1e-5)
          self.assertLess(_max_abs_diff(chunk_dw, dense_dw), 1e-5)

  @parameterized.product(
      scan_layers=(True, False),
      mesh_dims=((1, 1, 1), (4, 2, 1), (4, 1, 2), (2, 2, 2)),
  )
  def test_full_forward_and_backward_parity_across_meshes_and_scan_modes(
      self,
      scan_layers: bool,
      mesh_dims: tuple[int, int, int],
  ):
    bsz, seq_len = 8, 32
    ref_cfg = flint_config.mini_flint_config(
        global_batch_size=bsz,
        seq_len=seq_len,
        num_vocab_chunks=4,
        vocab_token_chunk_size=8,
        compute_dtype="float32",
        param_dtype="float32",
        use_splash=False,
    )
    torch_model = hf_reference.TorchFlintForCausalLM(ref_cfg.model)
    hf_reference.randomize_torch_flint(torch_model, seed=123)
    _recenter_rmsnorm_weights(torch_model, seed=1123)
    torch_state = {k: v.detach().clone() for k, v in torch_model.state_dict().items()}

    tokens_np, targets_np, pos_np, seg_np, mask_np = _make_packed_test_arrays(
        batch_size=bsz,
        seq_len=seq_len,
        vocab_size=ref_cfg.model.vocab_size,
        seed=31,
    )

    torch_model.zero_grad(set_to_none=True)
    torch_logits, torch_hiddens, torch_routers = torch_model(
        torch.from_numpy(tokens_np).to(torch.int64),
        positions=torch.from_numpy(pos_np).to(torch.int64),
        segment_ids=torch.from_numpy(seg_np).to(torch.int64),
    )
    torch_loss = hf_reference.compute_torch_ce_loss(
        torch_logits,
        torch.from_numpy(targets_np).to(torch.int64),
        torch.from_numpy(mask_np).to(torch.int64),
    )
    torch_loss.backward()

    torch_logits_np = torch_logits.detach().cpu().numpy()
    torch_hiddens_np = [h.detach().cpu().numpy() for h in torch_hiddens]
    torch_loss_val = float(torch_loss.detach().cpu().item())
    torch_grad_dict: dict[str, torch.Tensor] = {}
    for name, param in torch_model.named_parameters():
      self.assertIsNotNone(param.grad, msg=f"Missing PyTorch grad for {name}")
      assert param.grad is not None
      torch_grad_dict[name] = param.grad.detach().clone()

    cfg = flint_config.mini_flint_config(
        fsdp_size=mesh_dims[0],
        cp_size=mesh_dims[1],
        ep_size=mesh_dims[2],
        global_batch_size=bsz,
        seq_len=seq_len,
        num_vocab_chunks=4,
        vocab_token_chunk_size=8,
        compute_dtype="float32",
        param_dtype="float32",
        use_splash=False,
        scan_layers=scan_layers,
    )
    mesh = _build_mesh_for_dims(mesh_dims)
    params = hf_reference.torch_state_dict_to_flint_params(torch_state, cfg, mesh)
    batch = _put_sharded_batch(tokens_np, targets_np, pos_np, seg_np, mask_np, mesh)

    jax_logits, jax_hiddens, jax_aux = flint_model.forward_with_intermediates(params, batch, cfg, mesh)
    vg_fn = flint_model.make_value_and_grad_fn(cfg, mesh)
    (jax_loss, jax_ce, jax_aux_loss), jax_grads = vg_fn(params, batch)

    expected_grads = hf_reference.torch_state_dict_to_flint_params(torch_grad_dict, cfg, mesh)

    self.assertLen(jax_hiddens, cfg.model.num_hidden_layers)
    self.assertLen(jax_aux, cfg.model.num_hidden_layers)
    for layer_idx in range(cfg.model.num_hidden_layers):
      diff_h = _max_abs_diff(jax_hiddens[layer_idx], torch_hiddens_np[layer_idx])
      self.assertLess(
          diff_h,
          1e-5,
          msg=f"Hidden mismatch at layer {layer_idx}: {diff_h}",
      )
      if layer_idx < cfg.model.num_dense_prefix_layers:
        self.assertIsNone(jax_aux[layer_idx])
      else:
        aux_i = jax_aux[layer_idx]
        ref_r = torch_routers[layer_idx]
        self.assertIsNotNone(aux_i)
        self.assertIsNotNone(ref_r)
        assert aux_i is not None and ref_r is not None
        np.testing.assert_array_equal(
            np.asarray(aux_i.topk_ids),
            ref_r.topk_ids.detach().cpu().numpy(),
            err_msg=f"Router topk_ids mismatch at layer {layer_idx}",
        )
        self.assertLess(
            _max_abs_diff(
                aux_i.topk_weights,
                ref_r.topk_weights.detach().cpu().numpy(),
            ),
            1e-5,
        )

    self.assertLess(_max_abs_diff(jax_logits, torch_logits_np), 1e-5)
    logits_only = flint_model.forward_logits(params, batch, cfg, mesh)
    self.assertLess(_max_abs_diff(logits_only, torch_logits_np), 1e-5)

    self.assertLess(abs(float(jax_ce) - torch_loss_val), 1e-6)
    self.assertLess(abs(float(jax_loss) - torch_loss_val), 1e-6)
    self.assertGreater(float(jax_aux_loss), 0.0)

    jax_leaves = jax.tree_util.tree_leaves_with_path(jax_grads)
    exp_leaves = jax.tree.leaves(expected_grads)
    self.assertLen(jax_leaves, len(exp_leaves))
    for (path, got_g), exp_g in zip(jax_leaves, exp_leaves, strict=True):
      path_str = jax.tree_util.keystr(path)
      self.assertGreater(
          float(np.linalg.norm(np.asarray(got_g))),
          0.0,
          msg=f"Zero gradient norm at {path_str}",
      )
      diff_g = _max_abs_diff(got_g, exp_g)
      self.assertLess(
          diff_g,
          1e-5,
          msg=(f"Grad mismatch on {path_str} ({scan_layers=}," f" {mesh_dims=}): {diff_g}"),
      )

  def test_router_aux_loss_mesh_invariance(self):
    bsz, seq_len = 8, 32
    base_cfg = flint_config.mini_flint_config(
        global_batch_size=bsz,
        seq_len=seq_len,
        router_aux_loss_coef=0.05,
        compute_dtype="float32",
        param_dtype="float32",
    )
    torch_model = hf_reference.TorchFlintForCausalLM(base_cfg.model)
    hf_reference.randomize_torch_flint(torch_model, seed=211)
    state_dict = torch_model.state_dict()
    tokens_np, targets_np, pos_np, seg_np, mask_np = _make_packed_test_arrays(batch_size=bsz, seq_len=seq_len, seed=53)

    def _eval_on(mesh_dims: tuple[int, int, int], scan_layers: bool):
      cfg = flint_config.mini_flint_config(
          fsdp_size=mesh_dims[0],
          cp_size=mesh_dims[1],
          ep_size=mesh_dims[2],
          global_batch_size=bsz,
          seq_len=seq_len,
          router_aux_loss_coef=0.05,
          scan_layers=scan_layers,
          compute_dtype="float32",
          param_dtype="float32",
      )
      mesh = _build_mesh_for_dims(mesh_dims)
      params = hf_reference.torch_state_dict_to_flint_params(state_dict, cfg, mesh)
      batch = _put_sharded_batch(tokens_np, targets_np, pos_np, seg_np, mask_np, mesh)
      return flint_model.make_value_and_grad_fn(cfg, mesh)(params, batch)

    (loss_ref, ce_ref, aux_ref), grads_ref = _eval_on((1, 1, 1), True)
    (loss_222, ce_222, aux_222), grads_222 = _eval_on((2, 2, 2), True)
    (loss_unscan, ce_unscan, aux_unscan), _ = _eval_on((2, 2, 2), False)

    self.assertGreater(float(aux_ref), 0.0)
    self.assertLess(abs(float(loss_ref) - (float(ce_ref) + 0.05 * float(aux_ref))), 1e-6)
    self.assertLess(abs(float(loss_222) - float(loss_ref)), 1e-5)
    self.assertLess(abs(float(ce_222) - float(ce_ref)), 1e-5)
    self.assertLess(abs(float(aux_222) - float(aux_ref)), 1e-5)
    self.assertLess(abs(float(loss_unscan) - float(loss_ref)), 1e-5)
    self.assertLess(abs(float(ce_unscan) - float(ce_ref)), 1e-5)
    self.assertLess(abs(float(aux_unscan) - float(aux_ref)), 1e-5)

    for (path, g_got), g_ref in zip(
        jax.tree_util.tree_leaves_with_path(grads_222),
        jax.tree.leaves(grads_ref),
        strict=True,
    ):
      self.assertLess(
          _max_abs_diff(g_got, g_ref),
          1e-5,
          msg=f"Aux-loss grad mismatch on {jax.tree_util.keystr(path)}",
      )

  def test_grad_accum_unequal_microbatch_loss_mask_exact_parity(self):
    bsz, seq_len = 8, 32
    base_cfg = flint_config.mini_flint_config(
        fsdp_size=2,
        cp_size=2,
        ep_size=1,
        global_batch_size=bsz,
        seq_len=seq_len,
    )
    torch_model = hf_reference.TorchFlintForCausalLM(base_cfg.model)
    hf_reference.randomize_torch_flint(torch_model, seed=307)
    state_dict = torch_model.state_dict()

    tokens_np, targets_np, pos_np, seg_np, _ = _make_packed_test_arrays(batch_size=bsz, seq_len=seq_len, seed=71)
    # On `(fsdp=2, cp=2, ep=1)`, `dp_ways=2` -> rank 0 owns rows 0..3 and rank 1
    # owns rows 4..7. With `grad_accum_steps=2`, microbatch 0 owns rows
    # `[0, 1, 4, 5]` (half masked) and microbatch 1 owns `[2, 3, 6, 7]` (nearly
    # all valid).
    unequal_mask = np.ones((bsz, seq_len), dtype=np.int32)
    unequal_mask[:, 15] = 0
    unequal_mask[:, 31] = 0
    unequal_mask[[0, 1, 4, 5], :16] = 0  # Microbatch 0 has half valid tokens!

    mesh = _build_mesh_for_dims((2, 2, 1))
    batch = _put_sharded_batch(tokens_np, targets_np, pos_np, seg_np, unequal_mask, mesh)

    for aux_coef in (0.0, 0.05):
      cfg_ga1 = flint_config.mini_flint_config(
          fsdp_size=2,
          cp_size=2,
          ep_size=1,
          global_batch_size=bsz,
          seq_len=seq_len,
          grad_accum_steps=1,
          router_aux_loss_coef=aux_coef,
          compute_dtype="float32",
          param_dtype="float32",
      )
      cfg_ga2 = flint_config.mini_flint_config(
          fsdp_size=2,
          cp_size=2,
          ep_size=1,
          global_batch_size=bsz,
          seq_len=seq_len,
          grad_accum_steps=2,
          router_aux_loss_coef=aux_coef,
          compute_dtype="float32",
          param_dtype="float32",
      )
      params = hf_reference.torch_state_dict_to_flint_params(state_dict, cfg_ga1, mesh)
      (loss1, ce1, _), grads1 = flint_model.make_value_and_grad_fn(cfg_ga1, mesh)(params, batch)
      (loss2, ce2, _), grads2 = flint_model.make_value_and_grad_fn(cfg_ga2, mesh)(params, batch)

      with self.subTest(aux_coef=aux_coef):
        self.assertLess(abs(float(ce2) - float(ce1)), 1e-5)
        if aux_coef == 0.0:
          self.assertLess(abs(float(loss2) - float(loss1)), 1e-5)
          for (path, g2), g1 in zip(
              jax.tree_util.tree_leaves_with_path(grads2),
              jax.tree.leaves(grads1),
              strict=True,
          ):
            self.assertLess(
                _max_abs_diff(g2, g1),
                1e-5,
                msg=f"Grad accum mismatch on {jax.tree_util.keystr(path)}",
            )

  def test_safetensors_checkpoint_roundtrip_forward_parity(self):
    bsz, seq_len = 8, 32
    cfg_src = flint_config.mini_flint_config(
        fsdp_size=4,
        cp_size=2,
        ep_size=1,
        global_batch_size=bsz,
        seq_len=seq_len,
        scan_layers=False,
        compute_dtype="float32",
        param_dtype="float32",
    )
    torch_model = hf_reference.TorchFlintForCausalLM(cfg_src.model)
    hf_reference.randomize_torch_flint(torch_model, seed=409)
    _recenter_rmsnorm_weights(torch_model, seed=1409)

    tokens_np, targets_np, pos_np, seg_np, mask_np = _make_packed_test_arrays(batch_size=bsz, seq_len=seq_len, seed=89)
    with torch.no_grad():
      torch_logits, _, _ = torch_model(
          torch.from_numpy(tokens_np).to(torch.int64),
          positions=torch.from_numpy(pos_np).to(torch.int64),
          segment_ids=torch.from_numpy(seg_np).to(torch.int64),
      )
    torch_logits_np = torch_logits.detach().cpu().numpy()

    mesh_src = _build_mesh_for_dims((4, 2, 1))
    params_src = hf_reference.torch_state_dict_to_flint_params(torch_model.state_dict(), cfg_src, mesh_src)
    ckpt_dir = self.create_tempdir().full_path
    shard_paths = flint_checkpoint.save_hf_checkpoint(
        params_src, cfg_src, ckpt_dir, hf_dtype="F32", max_shard_bytes=64 * 1024
    )
    self.assertGreater(len(shard_paths), 1)

    cfg_dst = flint_config.mini_flint_config(
        fsdp_size=2,
        cp_size=2,
        ep_size=2,
        global_batch_size=bsz,
        seq_len=seq_len,
        scan_layers=True,
        compute_dtype="float32",
        param_dtype="float32",
    )
    mesh_dst = _build_mesh_for_dims((2, 2, 2))
    params_loaded = flint_checkpoint.load_hf_checkpoint(ckpt_dir, cfg_dst, mesh_dst)
    batch_dst = _put_sharded_batch(tokens_np, targets_np, pos_np, seg_np, mask_np, mesh_dst)
    jax_logits = flint_model.forward_logits(params_loaded, batch_dst, cfg_dst, mesh_dst)
    self.assertLess(_max_abs_diff(jax_logits, torch_logits_np), 1e-5)

  def test_production_sliding_window_513_boundary_at_seq_528(self):
    bsz, seq_len = 4, 528
    cfg = flint_config.mini_flint_config(
        num_hidden_layers=5,
        num_dense_prefix_layers=2,
        sliding_window=513,
        fsdp_size=4,
        cp_size=2,
        ep_size=1,
        global_batch_size=bsz,
        seq_len=seq_len,
        scan_layers=True,
        compute_dtype="float32",
        param_dtype="float32",
    )
    torch_model = hf_reference.TorchFlintForCausalLM(cfg.model)
    hf_reference.randomize_torch_flint(torch_model, seed=521)
    _recenter_rmsnorm_weights(torch_model, seed=1521)

    rng = np.random.default_rng(523)
    tokens_np = rng.integers(0, cfg.model.vocab_size, size=(bsz, seq_len), dtype=np.int32)
    targets_np = np.roll(tokens_np, -1, axis=1)
    pos_np = np.broadcast_to(np.arange(seq_len, dtype=np.int32)[None, :], (bsz, seq_len)).copy()
    seg_np = np.ones((bsz, seq_len), dtype=np.int32)
    mask_np = np.ones((bsz, seq_len), dtype=np.int32)
    mask_np[:, -1] = 0

    with torch.no_grad():
      torch_logits, torch_hiddens, _ = torch_model(
          torch.from_numpy(tokens_np).to(torch.int64),
          positions=torch.from_numpy(pos_np).to(torch.int64),
          segment_ids=torch.from_numpy(seg_np).to(torch.int64),
      )

    mesh = _build_mesh_for_dims((4, 2, 1))
    params = hf_reference.torch_state_dict_to_flint_params(torch_model.state_dict(), cfg, mesh)
    batch = _put_sharded_batch(tokens_np, targets_np, pos_np, seg_np, mask_np, mesh)
    jax_logits, jax_hiddens, _ = flint_model.forward_with_intermediates(params, batch, cfg, mesh)
    for idx in range(5):
      self.assertLess(_max_abs_diff(jax_hiddens[idx], torch_hiddens[idx].numpy()), 1e-5)
    self.assertLess(_max_abs_diff(jax_logits, torch_logits.numpy()), 1e-5)

    # Direct single-layer perturbation oracle at token 0 (`W = 513`, `S = 528`).
    act_spec = flint_sharding.activation_pspec()
    b_spec = flint_sharding.batch_pspec()
    sparse_spec = flint_sharding.param_pspecs(cfg).decoder.tail_moe[0]
    sparse_w = params.decoder.tail_moe[0]

    x0_np = rng.standard_normal((bsz, seq_len, cfg.model.hidden_size)).astype(np.float32)
    x1_np = x0_np.copy()
    x1_np[:, 0, :] += 5.0

    x0_sh = jax.device_put(x0_np, NamedSharding(mesh, act_spec))
    x1_sh = jax.device_put(x1_np, NamedSharding(mesh, act_spec))

    def _run_layer(is_glob: bool):
      def _local(x_l, w_l, pos_l, seg_l):
        y_l, _, _ = flint_sparse_layer.flint_sparse_layer_forward(x_l, w_l, pos_l, seg_l, None, cfg, is_global=is_glob)
        return y_l

      return jax.jit(
          jax.shard_map(
              _local,
              mesh=mesh,
              in_specs=(act_spec, sparse_spec, b_spec, b_spec),
              out_specs=act_spec,
              check_vma=False,
          )
      )

    slide_fn = _run_layer(is_glob=False)
    glob_fn = _run_layer(is_glob=True)

    y_slide_0 = np.asarray(slide_fn(x0_sh, sparse_w, batch.positions, batch.segment_ids))
    y_slide_1 = np.asarray(slide_fn(x1_sh, sparse_w, batch.positions, batch.segment_ids))
    y_glob_0 = np.asarray(glob_fn(x0_sh, sparse_w, batch.positions, batch.segment_ids))
    y_glob_1 = np.asarray(glob_fn(x1_sh, sparse_w, batch.positions, batch.segment_ids))

    # Inside `W=513` at `pos=512` (`512 - 0 < 513`): sliding layer MUST change.
    self.assertGreater(_max_abs_diff(y_slide_1[:, 512], y_slide_0[:, 512]), 1e-4)
    # Strictly outside `W=513` (`pos >= 513`): sliding layer MUST NOT change!
    np.testing.assert_array_equal(y_slide_1[:, 513:], y_slide_0[:, 513:])
    # Global layer at `pos >= 513` MUST change!
    self.assertGreater(_max_abs_diff(y_glob_1[:, 513:], y_glob_0[:, 513:]), 1e-4)

  @parameterized.product(
      param_dtype=("float32", "bfloat16"),
      wd_exclude=(False, True),
  )
  def test_optax_adamw_two_step_smoke_and_schedules(self, param_dtype: str, wd_exclude: bool):
    cfg = flint_config.mini_flint_config(
        fsdp_size=2,
        cp_size=2,
        ep_size=2,
        global_batch_size=8,
        seq_len=32,
        compute_dtype=param_dtype,
        param_dtype=param_dtype,
        learning_rate=3e-3,
        warmup_steps=0,
        total_steps=20,
        lr_schedule="warmup_cosine",
        wd_exclude_norms_and_embed=wd_exclude,
    )
    mesh = _build_mesh_for_dims((2, 2, 2))
    state = flint_train.init_train_state(jax.random.key(0), mesh, cfg)

    # Verify `sinks` and Adam moments `mu`, `nu` are strictly `float32`.
    assert state.params.decoder.scanned_blocks is not None
    self.assertEqual(
        state.params.decoder.scanned_blocks.sliding_0.self_attn.sinks.dtype,
        jnp.float32,
    )
    for opt_leaf in jax.tree.leaves(state.opt_state):
      if opt_leaf.ndim > 0:
        self.assertEqual(opt_leaf.dtype, jnp.float32)

    if wd_exclude:
      wd_mask = flint_train.build_weight_decay_mask(state.params)
      self.assertFalse(wd_mask.embed)
      self.assertFalse(wd_mask.final_norm)
      self.assertTrue(wd_mask.lm_head)
      self.assertFalse(wd_mask.decoder.prefix_dense[0].input_layernorm)
      self.assertFalse(wd_mask.decoder.prefix_dense[0].self_attn.q_norm)
      self.assertFalse(wd_mask.decoder.prefix_dense[0].self_attn.sinks)
      self.assertTrue(wd_mask.decoder.prefix_dense[0].self_attn.q_proj)

    tokens_np, targets_np, pos_np, seg_np, mask_np = _make_packed_test_arrays(batch_size=8, seq_len=32, seed=97)
    batch = _put_sharded_batch(tokens_np, targets_np, pos_np, seg_np, mask_np, mesh)
    step_fn = flint_train.make_train_step(cfg, mesh, donate_state=True)

    state, m0 = step_fn(state, batch)
    state, m1 = step_fn(state, batch)

    loss0 = float(m0.loss)
    loss1 = float(m1.loss)
    gnorm0 = float(m0.grad_norm)
    gnorm1 = float(m1.grad_norm)
    lr0 = float(m0.learning_rate)
    lr1 = float(m1.learning_rate)

    self.assertTrue(np.isfinite(loss0) and np.isfinite(loss1))
    self.assertTrue(np.isfinite(gnorm0) and np.isfinite(gnorm1))
    self.assertGreater(gnorm0, 0.0)
    self.assertGreater(gnorm1, 0.0)
    self.assertGreater(lr0, 0.0)
    self.assertGreater(lr1, 0.0)
    self.assertEqual(int(state.step), 2)
    self.assertLess(loss1, loss0)

    # Verify warmup_cosine and constant schedule edge cases.
    sched_wc = flint_train.build_lr_schedule(
        flint_config.FlintTrainConfig(
            learning_rate=1e-3,
            warmup_steps=4,
            total_steps=10,
            min_lr_ratio=0.1,
            lr_schedule="warmup_cosine",
        )
    )
    self.assertAlmostEqual(float(jnp.asarray(sched_wc(0), dtype=jnp.float32)), 0.0, places=7)
    self.assertAlmostEqual(float(jnp.asarray(sched_wc(4), dtype=jnp.float32)), 1e-3, places=7)
    self.assertAlmostEqual(float(jnp.asarray(sched_wc(10), dtype=jnp.float32)), 1e-4, places=7)

    sched_const = flint_train.build_lr_schedule(
        flint_config.FlintTrainConfig(
            learning_rate=2e-4,
            warmup_steps=2,
            total_steps=10,
            lr_schedule="constant",
        )
    )
    self.assertAlmostEqual(float(jnp.asarray(sched_const(1), dtype=jnp.float32)), 1e-4, places=7)
    self.assertAlmostEqual(float(jnp.asarray(sched_const(5), dtype=jnp.float32)), 2e-4, places=7)


if __name__ == "__main__":
  absltest.main()
