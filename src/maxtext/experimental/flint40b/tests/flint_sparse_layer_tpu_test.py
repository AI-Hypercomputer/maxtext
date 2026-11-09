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

"""TPU v7x tests for `flint_sparse_layer` (Splash + ragged_dot)."""

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

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_dense_layer
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import flint_sparse_layer
from maxtext.experimental.flint40b import flint_types
from maxtext.experimental.flint40b import ops
from maxtext.experimental.flint40b.tests import hf_reference

P = PartitionSpec
_ALL_AXES = flint_config.ALL_AXES


def _cosine_similarity(a: np.ndarray | jax.Array, b: np.ndarray | jax.Array) -> float:
  a_f64 = np.asarray(a, dtype=np.float64).reshape(-1)
  b_f64 = np.asarray(b, dtype=np.float64).reshape(-1)
  denom = np.linalg.norm(a_f64) * np.linalg.norm(b_f64)
  if denom == 0.0:
    return 1.0 if np.allclose(a_f64, b_f64) else 0.0
  return float(np.dot(a_f64, b_f64) / denom)


def _max_abs_diff(a: np.ndarray | jax.Array, b: np.ndarray | jax.Array) -> float:
  a_f64 = np.asarray(a, dtype=np.float64)
  b_f64 = np.asarray(b, dtype=np.float64)
  return float(np.max(np.abs(a_f64 - b_f64)))


def _round_to_bf16_f32(arr: np.ndarray) -> np.ndarray:
  """Rounds `float32` array to nearest-even `bfloat16` stored in `float32`."""
  u32 = np.asarray(arr, dtype=np.float32).view(np.uint32)
  lsb = (u32 >> np.uint32(16)) & np.uint32(1)
  rounded = (u32 + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
  return rounded.view(np.float32).copy()


def _recenter_rmsnorm_weights(
    module: torch.nn.Module,
    seed: int = 0,
    low: float = 0.75,
    high: float = 1.25,
) -> None:
  """Initializes `*norm*.weight` to deterministic `Uniform(low, high)`."""
  gen = torch.Generator(device="cpu")
  gen.manual_seed(seed)
  with torch.no_grad():
    for name, param in module.named_parameters():
      if "norm" in name and name.endswith("weight"):
        vals = torch.empty(param.shape, dtype=torch.float32).uniform_(low, high, generator=gen)
        param.copy_(vals.to(device=param.device, dtype=param.dtype))


def _build_coherent_up_np(
    torch_layer: hf_reference.TorchFlintDecoderLayer,
    x_np: np.ndarray,
    pos_np: np.ndarray,
    seg_np: np.ndarray,
    rng: np.random.Generator,
    *,
    bf16_exact: bool,
) -> np.ndarray:
  """Builds upstream cotangent with coherent attention component for `dsinks`."""
  with torch.no_grad():
    normed = torch_layer.input_layernorm(torch.from_numpy(x_np))
    attn_out = torch_layer.self_attn(
        normed,
        positions=torch.from_numpy(pos_np).to(torch.int64),
        segment_ids=torch.from_numpy(seg_np).to(torch.int64),
    )
    attn_np = attn_out.detach().cpu().to(torch.float32).numpy()
  noise_np = rng.standard_normal(x_np.shape).astype(np.float32) * 0.25
  up_np = (noise_np + 0.5 * attn_np).astype(np.float32)
  return _round_to_bf16_f32(up_np) if bf16_exact else up_np


def _condition_sparse_router_and_inputs(
    torch_layer: hf_reference.TorchFlintDecoderLayer,
    bsz: int,
    seq_len: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    pos_np: np.ndarray,
    seg_np: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
  """Builds 16-cluster BF16-exact `x_np` and margin-separated `mlp.gate`."""
  q_mat, _ = np.linalg.qr(rng.standard_normal((hidden_size, num_experts + 1)))
  clusters = (q_mat[:, :num_experts].T * math.sqrt(float(hidden_size))).astype(np.float32)
  c_global = (q_mat[:, num_experts] * math.sqrt(float(hidden_size))).astype(np.float32)

  total_tok = bsz * seq_len
  tok_arange = np.arange(total_tok, dtype=np.int64)
  cluster_ids = (tok_arange * 7 + (tok_arange // num_experts)) % num_experts
  jitter = rng.standard_normal((bsz, seq_len, hidden_size)).astype(np.float32)
  x_raw = (
      2.0 * clusters[cluster_ids].reshape(bsz, seq_len, hidden_size) + 0.6 * c_global[None, None, :] + 0.20 * jitter
  ).astype(np.float32)
  x_np = _round_to_bf16_f32(x_raw)

  target_logits = np.zeros((num_experts, num_experts), dtype=np.float32)
  for c_idx in range(num_experts):
    for slot in range(top_k):
      target_logits[c_idx, (c_idx + slot) % num_experts] = 2.4 - 0.32 * slot
    for slot in range(top_k, num_experts):
      target_logits[c_idx, (c_idx + slot) % num_experts] = -0.80 - 0.20 * (slot - top_k)

  gate_pt = _round_to_bf16_f32(((target_logits.T @ clusters) / float(hidden_size)).astype(np.float32))
  assert isinstance(torch_layer.mlp, hf_reference.TorchFlintSparseMoeBlock)
  with torch.no_grad():
    torch_layer.mlp.gate.weight.copy_(torch.from_numpy(gate_pt))
    x_pt = torch.from_numpy(x_np)
    h_pt = x_pt + torch_layer.self_attn(
        torch_layer.input_layernorm(x_pt),
        positions=torch.from_numpy(pos_np).to(torch.int64),
        segment_ids=torch.from_numpy(seg_np).to(torch.int64),
    )
    u_np = torch_layer.post_attention_layernorm(h_pt).cpu().numpy()
  r_logits = u_np.reshape(total_tok, hidden_size) @ gate_pt.T
  sorted_logits = np.sort(r_logits, axis=-1)
  min_gap = float(np.min(sorted_logits[:, -top_k] - sorted_logits[:, -top_k - 1]))
  return x_np, min_gap


def _make_sparse_test_config(
    *,
    fsdp_size: int,
    cp_size: int,
    ep_size: int,
    use_splash: bool,
    compute_dtype: str = "float32",
    param_dtype: str = "float32",
    seq_len: int = 1024,
    global_batch_size: int = 8,
    remat_policy: str = "layer",
    moe_impl: str = "xla_ragged_dot",
    sc_collective_mode: str = "off",
    routed_w_tiled: bool = False,
    moe_bijective_permute: bool = False,
    fuse_layer_projections: bool = False,
    splash_global_block_size: int | None = None,
    splash_qk_diag_skip: bool = False,
    splash_sv_diag_skip: bool = False,
    **extra_kernel_overrides,
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
          scan_layers=False,
      ),
      sharding=flint_config.FlintShardingConfig(
          fsdp_size=fsdp_size,
          cp_size=cp_size,
          ep_size=ep_size,
      ),
      kernels=flint_config.FlintKernelConfig(
          use_splash=use_splash,
          num_vocab_chunks=17,
          moe_impl=moe_impl,
          sc_collective_mode=sc_collective_mode,
          routed_w_tiled=routed_w_tiled,
          moe_bijective_permute=moe_bijective_permute,
          fuse_layer_projections=fuse_layer_projections,
          splash_global_block_q=splash_global_block_size,
          splash_global_block_kv=splash_global_block_size,
          splash_global_block_kv_compute=splash_global_block_size,
          splash_global_block_q_dkv=splash_global_block_size,
          splash_global_block_kv_dkv=splash_global_block_size,
          splash_global_block_kv_dkv_compute=splash_global_block_size,
          splash_qk_diag_skip=splash_qk_diag_skip,
          splash_sv_diag_skip=splash_sv_diag_skip,
          **extra_kernel_overrides,
      ),
      train=flint_config.FlintTrainConfig(
          global_batch_size=global_batch_size,
          seq_len=seq_len,
          compute_dtype=compute_dtype,
          param_dtype=param_dtype,
          remat_policy=remat_policy,
      ),
  )


def _sparse_pspecs() -> flint_types.FlintSparseLayerWeights[PartitionSpec]:
  cfg = _make_sparse_test_config(fsdp_size=8, cp_size=1, ep_size=1, use_splash=False)
  return flint_sharding.param_pspecs(cfg).decoder.unscanned_moe[0]


def _tensor_map_to_sparse_np(
    tmap: dict[str, torch.Tensor],
    num_experts: int,
    param_dt: jnp.dtype,
) -> flint_types.FlintSparseLayerWeights[np.ndarray]:
  def _arr(key: str, dt: jnp.dtype = param_dt, transpose: bool = False):
    val = tmap[key].detach().cpu().to(torch.float32).numpy()
    if transpose:
      val = val.T
    return val.astype(dt)

  gu_list = []
  down_list = []
  for exp_idx in range(num_experts):
    epfx = f"mlp.experts.{exp_idx}"
    gate_w = _arr(f"{epfx}.gate_proj.weight", transpose=True)
    up_w = _arr(f"{epfx}.up_proj.weight", transpose=True)
    gu_list.append(np.concatenate([gate_w, up_w], axis=-1))
    down_list.append(_arr(f"{epfx}.down_proj.weight", transpose=True))

  return flint_types.FlintSparseLayerWeights(
      input_layernorm=_arr("input_layernorm.weight"),
      self_attn=flint_types.FlintAttnWeights(
          q_proj=_arr("self_attn.q_proj.weight", transpose=True),
          k_proj=_arr("self_attn.k_proj.weight", transpose=True),
          v_proj=_arr("self_attn.v_proj.weight", transpose=True),
          o_proj=_arr("self_attn.o_proj.weight", transpose=True),
          q_norm=_arr("self_attn.q_norm.weight"),
          k_norm=_arr("self_attn.k_norm.weight"),
          sinks=_arr("self_attn.softmax_offset", dt=jnp.dtype(jnp.float32)),
      ),
      post_attention_layernorm=_arr("post_attention_layernorm.weight"),
      moe=flint_types.FlintMoEWeights(
          router=flint_types.FlintRouterWeights(
              gate=_arr("mlp.gate.weight", transpose=True),
          ),
          shared_expert=flint_types.FlintMLPWeights(
              gate_proj=_arr("mlp.shared_expert.gate_proj.weight", transpose=True),
              up_proj=_arr("mlp.shared_expert.up_proj.weight", transpose=True),
              down_proj=_arr("mlp.shared_expert.down_proj.weight", transpose=True),
          ),
          routed_experts=flint_types.FlintRoutedExpertsWeights(
              gate_up_proj=np.stack(gu_list, axis=0),
              down_proj=np.stack(down_list, axis=0),
          ),
      ),
  )


def _extract_torch_layer_weights(
    torch_layer: hf_reference.TorchFlintDecoderLayer,
    cfg: flint_config.FlintConfig,
    mesh: Mesh,
) -> flint_types.FlintSparseLayerWeights[jax.Array]:
  sd = {k: v.detach().cpu() for k, v in torch_layer.state_dict().items()}
  np_layer = _tensor_map_to_sparse_np(sd, cfg.model.num_experts, cfg.train.param_jnp_dtype)
  specs = flint_sharding.param_pspecs(cfg).decoder.unscanned_moe[0]
  return jax.tree.map(
      lambda arr, sp: jax.make_array_from_callback(arr.shape, NamedSharding(mesh, sp), lambda idx: arr[idx].copy()),
      np_layer,
      specs,
  )


def _extract_torch_layer_grads_np(
    torch_layer: hf_reference.TorchFlintDecoderLayer,
    num_experts: int,
) -> flint_types.FlintSparseLayerWeights[np.ndarray]:
  grad_sd: dict[str, torch.Tensor] = {}
  for name, param in torch_layer.named_parameters():
    assert param.grad is not None, f"Missing grad for {name}"
    grad_sd[name] = param.grad.detach().cpu()
  return _tensor_map_to_sparse_np(grad_sd, num_experts, jnp.dtype(jnp.float32))


def _psum_sparse_replicated_grads(
    grads: flint_types.FlintSparseLayerWeights[jax.Array],
    specs: flint_types.FlintSparseLayerWeights[PartitionSpec],
    num_devices: int,
) -> flint_types.FlintSparseLayerWeights[jax.Array]:
  if num_devices == 1:
    return grads
  empty_spec = P()
  return jax.tree.map(
      lambda g, sp: (jax.lax.psum(g.astype(jnp.float32), _ALL_AXES).astype(g.dtype) if sp == empty_spec else g),
      grads,
      specs,
  )


class FlintSparseLayerTpuTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.default_backend(), "tpu")
    self.assertLen(jax.devices(), 8)

  def test_sparse_layer_hlo_contains_splash_mha_and_ragged_dot(self):
    w_specs = _sparse_pspecs()
    act_spec = flint_sharding.activation_pspec()
    b_spec = flint_sharding.batch_pspec()

    for is_global in (True, False):
      cfg = _make_sparse_test_config(
          fsdp_size=2,
          cp_size=2,
          ep_size=2,
          use_splash=True,
          compute_dtype="bfloat16",
          param_dtype="float32",
          seq_len=1024,
          global_batch_size=8,
      )
      mesh = flint_sharding.create_mesh(cfg.sharding)

      w_sds = flint_sharding.abstract_param_shapes(cfg).decoder.unscanned_moe[0]
      sharded_w = jax.tree.map(
          lambda s, sp, m=mesh: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=NamedSharding(m, sp)),
          w_sds,
          w_specs,
      )
      b, s, d = (
          cfg.train.global_batch_size,
          cfg.train.seq_len,
          cfg.model.hidden_size,
      )
      act_sh = NamedSharding(mesh, act_spec)
      tok_sh = NamedSharding(mesh, b_spec)
      x_sds = jax.ShapeDtypeStruct((b, s, d), jnp.bfloat16, sharding=act_sh)
      pos_sds = jax.ShapeDtypeStruct((b, s), jnp.int32, sharding=tok_sh)
      seg_sds = jax.ShapeDtypeStruct((b, s), jnp.int32, sharding=tok_sh)
      msk_sds = jax.ShapeDtypeStruct((b, s), jnp.int32, sharding=tok_sh)
      up_sds = jax.ShapeDtypeStruct((b, s, d), jnp.bfloat16, sharding=act_sh)

      def _step(w_l, x_l, pos_l, seg_l, msk_l, up_l, cfg_local=cfg, glob=is_global):
        def _loss(w_in, x_in):
          y_l, _, _ = flint_sparse_layer.flint_sparse_layer_forward(
              x_in, w_in, pos_l, seg_l, msk_l, cfg_local, is_global=glob
          )
          loc = jnp.sum(y_l.astype(jnp.float32) * up_l.astype(jnp.float32))
          return loc, y_l

        (loc, y_l), (dw_l, dx_l) = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(w_l, x_l)
        dw_l = _psum_sparse_replicated_grads(dw_l, w_specs, cfg_local.sharding.num_devices)
        return jax.lax.psum(loc, _ALL_AXES), y_l, dw_l, dx_l

      jitted = jax.jit(
          jax.shard_map(
              _step,
              mesh=mesh,
              in_specs=(w_specs, act_spec, b_spec, b_spec, b_spec, act_spec),
              out_specs=(P(), act_spec, w_specs, act_spec),
              check_vma=False,
          )
      )
      lowered = jitted.lower(sharded_w, x_sds, pos_sds, seg_sds, msk_sds, up_sds)
      hlo_debug = lowered.as_text("hlo", debug_info=True)
      hlo_plain = lowered.as_text("hlo")

      with self.subTest(is_global=is_global):
        self.assertIn("splash_mha", hlo_debug)
        self.assertIn("ragged-dot", hlo_plain)

  @parameterized.parameters(False, True)
  def test_sparse_layer_fwd_bwd_parity_across_meshes_and_dtypes(self, is_global: bool):
    bsz, seq_len, hidden_size = 8, 1024, 512
    base_cfg = _make_sparse_test_config(
        fsdp_size=8,
        cp_size=1,
        ep_size=1,
        use_splash=False,
        seq_len=seq_len,
        global_batch_size=bsz,
    )
    # Layer 5 is global (`(5 - 2 + 1) % 4 == 0`); layer 2 is sliding (`W=513`).
    layer_idx = 5 if is_global else 2
    self.assertEqual(flint_config.is_global_layer(layer_idx, base_cfg.model), is_global)

    torch_layer = hf_reference.TorchFlintDecoderLayer(base_cfg.model, layer_idx)
    hf_reference.randomize_torch_flint(torch_layer, seed=101 + layer_idx)
    _recenter_rmsnorm_weights(torch_layer, seed=1101 + layer_idx)

    rng = np.random.default_rng(202 + layer_idx)
    # Segment 1 has length 768 (`> W=513`); segment 2 has length 256.
    pos_1d = np.concatenate([np.arange(768, dtype=np.int32), np.arange(256, dtype=np.int32)])
    seg_1d = np.concatenate([np.ones(768, dtype=np.int32), np.full(256, 2, dtype=np.int32)])
    pos_np = np.broadcast_to(pos_1d[None, :], (bsz, seq_len)).copy()
    seg_np = np.broadcast_to(seg_1d[None, :], (bsz, seq_len)).copy()
    mask_np = np.ones((bsz, seq_len), dtype=np.int32)
    mask_np[:, 767] = 0
    mask_np[:, 1023] = 0

    w_specs = _sparse_pspecs()
    act_spec = flint_sharding.activation_pspec()
    b_spec = flint_sharding.batch_pspec()

    def _eval_jax(
        x_arr: np.ndarray,
        up_arr: np.ndarray,
        mesh_dims: tuple[int, int, int],
        *,
        use_splash: bool,
        compute_dtype: str,
        param_dtype: str,
    ):
      cfg = _make_sparse_test_config(
          fsdp_size=mesh_dims[0],
          cp_size=mesh_dims[1],
          ep_size=mesh_dims[2],
          use_splash=use_splash,
          compute_dtype=compute_dtype,
          param_dtype=param_dtype,
          seq_len=seq_len,
          global_batch_size=bsz,
      )
      mesh = flint_sharding.create_mesh(cfg.sharding)
      w_sh = _extract_torch_layer_weights(torch_layer, cfg, mesh)
      comp_dt = cfg.train.compute_jnp_dtype
      act_sh = NamedSharding(mesh, act_spec)
      tok_sh = NamedSharding(mesh, b_spec)

      x_sh = jax.device_put(jnp.asarray(x_arr, dtype=comp_dt), act_sh)
      up_sh = jax.device_put(jnp.asarray(up_arr, dtype=comp_dt), act_sh)
      pos_sh = jax.device_put(pos_np, tok_sh)
      seg_sh = jax.device_put(seg_np, tok_sh)
      msk_sh = jax.device_put(mask_np, tok_sh)

      def _local(w_l, x_l, pos_l, seg_l, msk_l, up_l):
        def _loss(w_in, x_in):
          y_l, counts_l, _ = flint_sparse_layer.flint_sparse_layer_forward(
              x_in, w_in, pos_l, seg_l, msk_l, cfg, is_global=is_global
          )
          loc = jnp.sum(y_l.astype(jnp.float32) * up_l.astype(jnp.float32)) / float(bsz * seq_len)
          return loc, (y_l, counts_l)

        (loc, (y_l, counts_l)), (dw_l, dx_l) = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(w_l, x_l)
        dw_l = _psum_sparse_replicated_grads(dw_l, w_specs, cfg.sharding.num_devices)
        glob_counts = jax.lax.psum(counts_l, _ALL_AXES)
        return jax.lax.psum(loc, _ALL_AXES), y_l, dw_l, dx_l, glob_counts

      fn = jax.jit(
          jax.shard_map(
              _local,
              mesh=mesh,
              in_specs=(w_specs, act_spec, b_spec, b_spec, b_spec, act_spec),
              out_specs=(P(), act_spec, w_specs, act_spec, P()),
              check_vma=False,
          )
      )
      return fn(w_sh, x_sh, pos_sh, seg_sh, msk_sh, up_sh)

    # 1) Exact FP32 `use_splash=False` vs PyTorch on unstructured random `x_np`
    x_unstruct_np = (rng.standard_normal((bsz, seq_len, hidden_size)) * 0.5 + 0.2).astype(np.float32)
    up_unstruct_np = _build_coherent_up_np(torch_layer, x_unstruct_np, pos_np, seg_np, rng, bf16_exact=False)
    x_unstruct_pt = torch.from_numpy(x_unstruct_np).requires_grad_(True)
    torch_layer.zero_grad(set_to_none=True)
    pt_u_out, _ = torch_layer(
        x_unstruct_pt,
        positions=torch.from_numpy(pos_np).to(torch.int64),
        segment_ids=torch.from_numpy(seg_np).to(torch.int64),
    )
    pt_u_loss = (pt_u_out * torch.from_numpy(up_unstruct_np)).sum() / float(bsz * seq_len)
    pt_u_loss.backward()
    assert x_unstruct_pt.grad is not None
    pt_u_out_np = pt_u_out.detach().cpu().numpy()
    pt_u_dx_np = x_unstruct_pt.grad.detach().cpu().numpy()
    pt_u_dw_np = _extract_torch_layer_grads_np(torch_layer, base_cfg.model.num_experts)

    _, ref_u_y, ref_u_dw, ref_u_dx, ref_u_counts = _eval_jax(
        x_unstruct_np,
        up_unstruct_np,
        (2, 2, 2),
        use_splash=False,
        compute_dtype="float32",
        param_dtype="float32",
    )
    self.assertTrue(np.all(np.asarray(ref_u_counts) > 0))
    self.assertLess(_max_abs_diff(ref_u_y, pt_u_out_np), 1e-4)
    self.assertLess(_max_abs_diff(ref_u_dx, pt_u_dx_np), 1e-4)
    self.assertGreater(_cosine_similarity(ref_u_y, pt_u_out_np), 0.99999)
    self.assertGreater(_cosine_similarity(ref_u_dx, pt_u_dx_np), 0.99999)
    for (path, g_ref_u), g_pt_u in zip(
        jax.tree_util.tree_leaves_with_path(ref_u_dw),
        jax.tree.leaves(pt_u_dw_np),
        strict=True,
    ):
      p_str = jax.tree_util.keystr(path)
      self.assertLess(
          _max_abs_diff(g_ref_u, g_pt_u),
          1e-4,
          msg=f"FP32 ref max_abs_diff >= 1e-4 on {p_str}",
      )
      self.assertGreater(
          _cosine_similarity(g_ref_u, g_pt_u),
          0.99999,
          msg=f"FP32 ref cosine <= 0.99999 on {p_str}",
      )

    # 2) `use_splash=True` multi-mesh & dtype parity on all 16 active experts.
    x_np, min_router_gap = _condition_sparse_router_and_inputs(
        torch_layer,
        bsz,
        seq_len,
        hidden_size,
        base_cfg.model.num_experts,
        base_cfg.model.num_experts_per_tok,
        pos_np,
        seg_np,
        rng,
    )
    self.assertGreaterEqual(min_router_gap, 0.15)
    up_np = _build_coherent_up_np(torch_layer, x_np, pos_np, seg_np, rng, bf16_exact=True)

    x_pt = torch.from_numpy(x_np).requires_grad_(True)
    up_pt = torch.from_numpy(up_np)
    torch_layer.zero_grad(set_to_none=True)
    pt_out, _ = torch_layer(
        x_pt,
        positions=torch.from_numpy(pos_np).to(torch.int64),
        segment_ids=torch.from_numpy(seg_np).to(torch.int64),
    )
    pt_loss = (pt_out * up_pt).sum() / float(bsz * seq_len)
    pt_loss.backward()

    assert x_pt.grad is not None
    pt_out_np = pt_out.detach().cpu().numpy()
    pt_dx_np = x_pt.grad.detach().cpu().numpy()
    pt_dw_np = _extract_torch_layer_grads_np(torch_layer, base_cfg.model.num_experts)

    _, ref_y, ref_dw, ref_dx, ref_counts = _eval_jax(
        x_np,
        up_np,
        (8, 1, 1),
        use_splash=False,
        compute_dtype="float32",
        param_dtype="float32",
    )
    self.assertTrue(np.all(np.asarray(ref_counts) > 0))

    for mesh_dims in ((8, 1, 1), (4, 2, 1), (4, 1, 2), (2, 2, 2)):
      for compute_dtype, param_dtype, min_cos in (
          ("float32", "float32", 0.9999),
          ("bfloat16", "bfloat16", 0.999),
      ):
        _, got_y, got_dw, got_dx, got_counts = _eval_jax(
            x_np,
            up_np,
            mesh_dims,
            use_splash=True,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        with self.subTest(
            is_global=is_global,
            mesh_dims=mesh_dims,
            compute_dtype=compute_dtype,
        ):
          self.assertEqual(got_dw.self_attn.sinks.dtype, jnp.float32)
          counts_arr = np.asarray(got_counts, dtype=np.float32)
          self.assertEqual(counts_arr.shape, (base_cfg.model.num_experts,))
          self.assertTrue(np.all(counts_arr > 0))

          gu_grad_np = np.asarray(got_dw.moe.routed_experts.gate_up_proj, dtype=np.float32)
          dn_grad_np = np.asarray(got_dw.moe.routed_experts.down_proj, dtype=np.float32)
          for exp_idx in range(base_cfg.model.num_experts):
            self.assertGreater(float(np.linalg.norm(gu_grad_np[exp_idx])), 0.0)
            self.assertGreater(float(np.linalg.norm(dn_grad_np[exp_idx])), 0.0)

          self.assertGreater(_cosine_similarity(got_y, pt_out_np), min_cos)
          self.assertGreater(_cosine_similarity(got_dx, pt_dx_np), min_cos)
          self.assertGreater(_cosine_similarity(got_y, ref_y), min_cos)
          self.assertGreater(_cosine_similarity(got_dx, ref_dx), min_cos)

          for (path, g_got), g_pt, g_ref in zip(
              jax.tree_util.tree_leaves_with_path(got_dw),
              jax.tree.leaves(pt_dw_np),
              jax.tree.leaves(ref_dw),
              strict=True,
          ):
            p_str = jax.tree_util.keystr(path)
            self.assertGreater(
                _cosine_similarity(g_got, g_pt),
                min_cos,
                msg=f"Torch grad cosine below {min_cos} on {p_str}",
            )
            self.assertGreater(
                _cosine_similarity(g_got, g_ref),
                min_cos,
                msg=f"Ref grad cosine below {min_cos} on {p_str}",
            )

  def test_k3_selective_remat_matches_layer_remat_sparse_and_dense(self):
    bsz, seq_len, hidden_size = 8, 1024, 512
    rng = np.random.default_rng(303)
    pos_1d = np.concatenate([np.arange(768, dtype=np.int32), np.arange(256, dtype=np.int32)])
    seg_1d = np.concatenate([np.ones(768, dtype=np.int32), np.full(256, 2, dtype=np.int32)])
    pos_np = np.broadcast_to(pos_1d[None, :], (bsz, seq_len)).copy()
    seg_np = np.broadcast_to(seg_1d[None, :], (bsz, seq_len)).copy()
    mask_np = np.ones((bsz, seq_len), dtype=np.int32)
    mask_np[:, 767] = 0

    act_spec = flint_sharding.activation_pspec()
    b_spec = flint_sharding.batch_pspec()

    # 1) Dense prefix layer (`flint_dense_layer_forward`) on CP=1 and CP=2.
    x_dense_np = _round_to_bf16_f32((rng.standard_normal((bsz, seq_len, hidden_size)) * 0.5).astype(np.float32))
    up_dense_np = _round_to_bf16_f32(rng.standard_normal((bsz, seq_len, hidden_size)).astype(np.float32))
    ref_init_cfg = _make_sparse_test_config(
        fsdp_size=8,
        cp_size=1,
        ep_size=1,
        use_splash=True,
        compute_dtype="bfloat16",
        param_dtype="float32",
    )
    init_mesh = flint_sharding.create_mesh(ref_init_cfg.sharding)
    init_params = flint_sharding.init_random_params(jax.random.PRNGKey(77), init_mesh, ref_init_cfg)
    dense_host_w = jax.tree.map(
        lambda a: np.asarray(jax.device_get(a)),
        init_params.decoder.prefix_dense[0],
    )
    dense_specs = flint_sharding.param_pspecs(ref_init_cfg).decoder.prefix_dense[0]

    for mesh_dims in ((8, 1, 1), (4, 2, 1)):
      mesh = flint_sharding.create_mesh(
          flint_config.FlintShardingConfig(
              fsdp_size=mesh_dims[0],
              cp_size=mesh_dims[1],
              ep_size=mesh_dims[2],
          )
      )
      w_dense_sh = jax.tree.map(
          lambda arr, sp, m=mesh: jax.make_array_from_callback(
              arr.shape, NamedSharding(m, sp), lambda idx: arr[idx].copy()
          ),
          dense_host_w,
          dense_specs,
      )
      act_sh = NamedSharding(mesh, act_spec)
      tok_sh = NamedSharding(mesh, b_spec)
      x_sh = jax.device_put(jnp.asarray(x_dense_np, dtype=jnp.bfloat16), act_sh)
      up_sh = jax.device_put(jnp.asarray(up_dense_np, dtype=jnp.bfloat16), act_sh)
      pos_sh = jax.device_put(pos_np, tok_sh)
      seg_sh = jax.device_put(seg_np, tok_sh)

      def _run_dense(policy: str, cur_mesh=mesh, m_dims=mesh_dims):
        cfg = _make_sparse_test_config(
            fsdp_size=m_dims[0],
            cp_size=m_dims[1],
            ep_size=m_dims[2],
            use_splash=True,
            compute_dtype="bfloat16",
            param_dtype="float32",
            remat_policy=policy,
        )

        def _local_dense(w_l, x_l, p_l, s_l, u_l):
          def _loss(w_in, x_in):
            y_l = flint_dense_layer.flint_dense_layer_forward(x_in, w_in, p_l, s_l, cfg)
            loc = jnp.sum(y_l.astype(jnp.float32) * u_l.astype(jnp.float32)) / float(bsz * seq_len)
            return loc, y_l

          (loc, y_l), (dw_l, dx_l) = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(w_l, x_l)
          empty_spec = P()
          dw_l = jax.tree.map(
              lambda g, sp: (jax.lax.psum(g.astype(jnp.float32), _ALL_AXES).astype(g.dtype) if sp == empty_spec else g),
              dw_l,
              dense_specs,
          )
          return jax.lax.psum(loc, _ALL_AXES), y_l, dw_l, dx_l

        fn = jax.jit(
            jax.shard_map(
                _local_dense,
                mesh=cur_mesh,
                in_specs=(dense_specs, act_spec, b_spec, b_spec, act_spec),
                out_specs=(P(), act_spec, dense_specs, act_spec),
                check_vma=False,
            )
        )
        return fn(w_dense_sh, x_sh, pos_sh, seg_sh, up_sh)

      _, y_d_layer, dw_d_layer, dx_d_layer = _run_dense("layer")
      _, y_d_sel, dw_d_sel, dx_d_sel = _run_dense("selective")
      self.assertGreaterEqual(_cosine_similarity(y_d_sel, y_d_layer), 0.9999)
      self.assertGreaterEqual(_cosine_similarity(dx_d_sel, dx_d_layer), 0.9999)
      for g_s, g_l in zip(jax.tree.leaves(dw_d_sel), jax.tree.leaves(dw_d_layer), strict=True):
        self.assertGreaterEqual(_cosine_similarity(g_s, g_l), 0.9999)

    # 2) Sparse layer (`inside_scan=False` and `True`, `is_global=False`/`True`)
    base_cfg = _make_sparse_test_config(fsdp_size=8, cp_size=1, ep_size=1, use_splash=False)
    w_specs = _sparse_pspecs()
    for is_global in (False, True):
      layer_idx = 5 if is_global else 2
      torch_layer = hf_reference.TorchFlintDecoderLayer(base_cfg.model, layer_idx)
      hf_reference.randomize_torch_flint(torch_layer, seed=401 + layer_idx)
      _recenter_rmsnorm_weights(torch_layer, seed=1401 + layer_idx)
      x_np, _ = _condition_sparse_router_and_inputs(
          torch_layer,
          bsz,
          seq_len,
          hidden_size,
          base_cfg.model.num_experts,
          base_cfg.model.num_experts_per_tok,
          pos_np,
          seg_np,
          rng,
      )
      up_np = _build_coherent_up_np(torch_layer, x_np, pos_np, seg_np, rng, bf16_exact=True)
      for mesh_dims in ((8, 1, 1), (4, 2, 1)):
        for inside_scan in (False, True):

          def _run_sparse_policy(
              policy: str,
              m_dims=mesh_dims,
              glob=is_global,
              in_scan=inside_scan,
              tl=torch_layer,
              x_arr=x_np,
              u_arr=up_np,
          ):
            cfg = _make_sparse_test_config(
                fsdp_size=m_dims[0],
                cp_size=m_dims[1],
                ep_size=m_dims[2],
                use_splash=True,
                compute_dtype="bfloat16",
                param_dtype="float32",
                remat_policy=policy,
            )
            mesh = flint_sharding.create_mesh(cfg.sharding)
            w_sh = _extract_torch_layer_weights(tl, cfg, mesh)
            act_sh = NamedSharding(mesh, act_spec)
            tok_sh = NamedSharding(mesh, b_spec)
            x_s = jax.device_put(jnp.asarray(x_arr, dtype=jnp.bfloat16), act_sh)
            u_s = jax.device_put(jnp.asarray(u_arr, dtype=jnp.bfloat16), act_sh)
            p_s = jax.device_put(pos_np, tok_sh)
            sg_s = jax.device_put(seg_np, tok_sh)
            mk_s = jax.device_put(mask_np, tok_sh)

            def _local(w_l, x_l, p_l, sg_l, mk_l, u_l):
              def _loss(w_in, x_in):
                y_l, _, _ = flint_sparse_layer.flint_sparse_layer_forward(
                    x_in,
                    w_in,
                    p_l,
                    sg_l,
                    mk_l,
                    cfg,
                    is_global=glob,
                    inside_scan=in_scan,
                )
                loc = jnp.sum(y_l.astype(jnp.float32) * u_l.astype(jnp.float32)) / float(bsz * seq_len)
                return loc, y_l

              (loc, y_l), (dw_l, dx_l) = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(w_l, x_l)
              dw_l = _psum_sparse_replicated_grads(dw_l, w_specs, cfg.sharding.num_devices)
              return jax.lax.psum(loc, _ALL_AXES), y_l, dw_l, dx_l

            fn = jax.jit(
                jax.shard_map(
                    _local,
                    mesh=mesh,
                    in_specs=(
                        w_specs,
                        act_spec,
                        b_spec,
                        b_spec,
                        b_spec,
                        act_spec,
                    ),
                    out_specs=(P(), act_spec, w_specs, act_spec),
                    check_vma=False,
                )
            )
            return fn(w_sh, x_s, p_s, sg_s, mk_s, u_s)

          _, y_l, dw_l, dx_l = _run_sparse_policy("layer")
          _, y_s, dw_s, dx_s = _run_sparse_policy("selective")
          with self.subTest(
              is_global=is_global,
              mesh=mesh_dims,
              inside_scan=inside_scan,
          ):
            self.assertEqual(dw_s.self_attn.sinks.dtype, jnp.float32)
            self.assertGreaterEqual(_cosine_similarity(y_s, y_l), 0.9999)
            self.assertGreaterEqual(_cosine_similarity(dx_s, dx_l), 0.9999)
            for (path, g_s), g_l in zip(
                jax.tree_util.tree_leaves_with_path(dw_s),
                jax.tree.leaves(dw_l),
                strict=True,
            ):
              self.assertGreaterEqual(
                  _cosine_similarity(g_s, g_l),
                  0.9999,
                  msg=f"Selective remat grad mismatch on {path}",
              )

  @parameterized.parameters(False, True)
  def test_combined_k1_through_k6_stacked_sparse_layer_fwd_bwd(self, is_global: bool):
    bsz, seq_len, hidden_size = 8, 1024, 512
    base_cfg = _make_sparse_test_config(fsdp_size=8, cp_size=1, ep_size=1, use_splash=False)
    layer_idx = 5 if is_global else 2
    torch_layer = hf_reference.TorchFlintDecoderLayer(base_cfg.model, layer_idx)
    hf_reference.randomize_torch_flint(torch_layer, seed=505 + layer_idx)
    _recenter_rmsnorm_weights(torch_layer, seed=1505 + layer_idx)

    rng = np.random.default_rng(606 + layer_idx)
    pos_1d = np.concatenate([np.arange(768, dtype=np.int32), np.arange(256, dtype=np.int32)])
    seg_1d = np.concatenate([np.ones(768, dtype=np.int32), np.full(256, 2, dtype=np.int32)])
    pos_np = np.broadcast_to(pos_1d[None, :], (bsz, seq_len)).copy()
    seg_np = np.broadcast_to(seg_1d[None, :], (bsz, seq_len)).copy()
    mask_np = np.ones((bsz, seq_len), dtype=np.int32)
    mask_np[:, 767] = 0
    mask_np[:, 1023] = 0

    x_np, min_gap = _condition_sparse_router_and_inputs(
        torch_layer,
        bsz,
        seq_len,
        hidden_size,
        base_cfg.model.num_experts,
        base_cfg.model.num_experts_per_tok,
        pos_np,
        seg_np,
        rng,
    )
    self.assertGreaterEqual(min_gap, 0.15)
    up_np = _build_coherent_up_np(torch_layer, x_np, pos_np, seg_np, rng, bf16_exact=True)

    w_specs = _sparse_pspecs()
    act_spec = flint_sharding.activation_pspec()
    b_spec = flint_sharding.batch_pspec()

    def _run_cfg(cfg: flint_config.FlintConfig):
      mesh = flint_sharding.create_mesh(cfg.sharding)
      w_sh = _extract_torch_layer_weights(torch_layer, cfg, mesh)
      comp_dt = cfg.train.compute_jnp_dtype
      act_sh = NamedSharding(mesh, act_spec)
      tok_sh = NamedSharding(mesh, b_spec)
      x_sh = jax.device_put(jnp.asarray(x_np, dtype=comp_dt), act_sh)
      up_sh = jax.device_put(jnp.asarray(up_np, dtype=comp_dt), act_sh)
      pos_sh = jax.device_put(pos_np, tok_sh)
      seg_sh = jax.device_put(seg_np, tok_sh)
      msk_sh = jax.device_put(mask_np, tok_sh)

      def _local(w_l, x_l, p_l, sg_l, mk_l, u_l):
        def _loss(w_in, x_in):
          y_l, counts_l, _ = flint_sparse_layer.flint_sparse_layer_forward(
              x_in,
              w_in,
              p_l,
              sg_l,
              mk_l,
              cfg,
              is_global=is_global,
              inside_scan=True,
          )
          loc = jnp.mean(y_l.astype(jnp.float32) * u_l.astype(jnp.float32)) / (float(cfg.sharding.num_devices))
          return loc, (y_l, counts_l)

        (loc, (y_l, counts_l)), (dw_l, dx_l) = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(w_l, x_l)
        dw_l = _psum_sparse_replicated_grads(dw_l, w_specs, cfg.sharding.num_devices)
        glob_counts = jax.lax.psum(counts_l, _ALL_AXES)
        return jax.lax.psum(loc, _ALL_AXES), y_l, dw_l, dx_l, glob_counts

      fn = jax.jit(
          jax.shard_map(
              _local,
              mesh=mesh,
              in_specs=(w_specs, act_spec, b_spec, b_spec, b_spec, act_spec),
              out_specs=(P(), act_spec, w_specs, act_spec, P()),
              check_vma=False,
          )
      )
      return fn(w_sh, x_sh, pos_sh, seg_sh, msk_sh, up_sh)

    for mesh_dims in ((4, 2, 1), (4, 1, 2), (2, 2, 2)):
      for compute_dtype, min_cos in (
          ("float32", 0.9999),
          ("bfloat16", 0.999),
      ):
        ref_cfg = _make_sparse_test_config(
            fsdp_size=mesh_dims[0],
            cp_size=mesh_dims[1],
            ep_size=mesh_dims[2],
            use_splash=True,
            compute_dtype=compute_dtype,
            param_dtype="float32",
        )
        stacked_cfg = _make_sparse_test_config(
            fsdp_size=mesh_dims[0],
            cp_size=mesh_dims[1],
            ep_size=mesh_dims[2],
            use_splash=True,
            compute_dtype=compute_dtype,
            param_dtype="float32",
            remat_policy="selective",
            moe_impl="pallas_gmm_v2",
            sc_collective_mode="dual_sc",
            routed_w_tiled=True,
            moe_bijective_permute=True,
            fuse_layer_projections=True,
            splash_global_block_size=256,
            splash_qk_diag_skip=True,
        )
        loss_ref, y_ref, dw_ref, dx_ref, counts_ref = _run_cfg(ref_cfg)
        loss_got, y_got, dw_got, dx_got, counts_got = _run_cfg(stacked_cfg)
        with self.subTest(
            is_global=is_global,
            mesh=mesh_dims,
            compute_dtype=compute_dtype,
        ):
          self.assertTrue(np.isfinite(float(np.asarray(loss_got))))
          self.assertLess(
              abs(float(np.asarray(loss_got)) - float(np.asarray(loss_ref))),
              5e-3,
          )
          self.assertEqual(dw_got.self_attn.sinks.dtype, jnp.float32)
          np.testing.assert_array_equal(np.asarray(counts_got), np.asarray(counts_ref))
          self.assertTrue(np.all(np.isfinite(np.asarray(y_got, np.float32))))
          self.assertTrue(np.all(np.isfinite(np.asarray(dx_got, np.float32))))
          self.assertGreaterEqual(_cosine_similarity(y_got, y_ref), min_cos)
          self.assertGreaterEqual(_cosine_similarity(dx_got, dx_ref), min_cos)
          for (path, g_got), g_ref in zip(
              jax.tree_util.tree_leaves_with_path(dw_got),
              jax.tree.leaves(dw_ref),
              strict=True,
          ):
            p_str = jax.tree_util.keystr(path)
            g_got_np = np.asarray(g_got, dtype=np.float32)
            self.assertTrue(np.all(np.isfinite(g_got_np)))
            self.assertGreater(float(np.linalg.norm(g_got_np)), 0.0)
            self.assertGreaterEqual(
                _cosine_similarity(g_got_np, g_ref),
                min_cos,
                msg=f"Stacked K1..K6 cosine below {min_cos} on {p_str}",
            )

  @parameterized.parameters(False, True)
  def test_r3_stacked_all_knobs_sparse_layer_fwd_bwd(self, is_global: bool):
    bsz, seq_len, hidden_size = 8, 1024, 512
    base_cfg = _make_sparse_test_config(fsdp_size=8, cp_size=1, ep_size=1, use_splash=False)
    layer_idx = 5 if is_global else 2
    torch_layer = hf_reference.TorchFlintDecoderLayer(base_cfg.model, layer_idx)
    hf_reference.randomize_torch_flint(torch_layer, seed=808 + layer_idx)
    _recenter_rmsnorm_weights(torch_layer, seed=1808 + layer_idx)

    rng = np.random.default_rng(909 + layer_idx)
    pos_1d = np.concatenate([np.arange(768, dtype=np.int32), np.arange(256, dtype=np.int32)])
    seg_1d = np.concatenate([np.ones(768, dtype=np.int32), np.full(256, 2, dtype=np.int32)])
    pos_np = np.broadcast_to(pos_1d[None, :], (bsz, seq_len)).copy()
    seg_np = np.broadcast_to(seg_1d[None, :], (bsz, seq_len)).copy()
    mask_np = np.ones((bsz, seq_len), dtype=np.int32)
    mask_np[:, 767] = 0
    mask_np[:, 1023] = 0

    x_np, min_gap = _condition_sparse_router_and_inputs(
        torch_layer,
        bsz,
        seq_len,
        hidden_size,
        base_cfg.model.num_experts,
        base_cfg.model.num_experts_per_tok,
        pos_np,
        seg_np,
        rng,
    )
    self.assertGreaterEqual(min_gap, 0.15)
    up_np = _build_coherent_up_np(torch_layer, x_np, pos_np, seg_np, rng, bf16_exact=True)

    act_spec = flint_sharding.activation_pspec()
    b_spec = flint_sharding.batch_pspec()

    def _run_cfg(cfg: flint_config.FlintConfig):
      mesh = flint_sharding.create_mesh(cfg.sharding)
      w_specs = flint_sharding.param_pspecs(cfg).decoder.unscanned_moe[0]
      w_sh = _extract_torch_layer_weights(torch_layer, cfg, mesh)
      comp_dt = cfg.train.compute_jnp_dtype
      act_sh = NamedSharding(mesh, act_spec)
      tok_sh = NamedSharding(mesh, b_spec)
      x_sh = jax.device_put(jnp.asarray(x_np, dtype=comp_dt), act_sh)
      up_sh = jax.device_put(jnp.asarray(up_np, dtype=comp_dt), act_sh)
      pos_sh = jax.device_put(pos_np, tok_sh)
      seg_sh = jax.device_put(seg_np, tok_sh)
      msk_sh = jax.device_put(mask_np, tok_sh)

      def _local(w_l, x_l, p_l, sg_l, mk_l, u_l):
        rope_cs = (
            ops.compute_split_half_rope_cos_sin(
                p_l,
                cfg.model.head_dim,
                rope_theta=cfg.model.rope_theta,
                dtype=comp_dt,
            )
            if (not is_global and cfg.kernels.precompute_rope_tables)
            else None
        )

        def _loss(w_in, x_in):
          y_l, counts_l, _ = flint_sparse_layer.flint_sparse_layer_forward(
              x_in,
              w_in,
              p_l,
              sg_l,
              mk_l,
              cfg,
              is_global=is_global,
              inside_scan=True,
              rope_cos_sin=rope_cs,
          )
          loc = jnp.mean(y_l.astype(jnp.float32) * u_l.astype(jnp.float32)) / (float(cfg.sharding.num_devices))
          return loc, (y_l, counts_l)

        (loc, (y_l, counts_l)), (dw_l, dx_l) = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(w_l, x_l)
        dw_l = _psum_sparse_replicated_grads(dw_l, w_specs, cfg.sharding.num_devices)
        glob_counts = jax.lax.psum(counts_l, _ALL_AXES)
        return jax.lax.psum(loc, _ALL_AXES), y_l, dw_l, dx_l, glob_counts

      fn = jax.jit(
          jax.shard_map(
              _local,
              mesh=mesh,
              in_specs=(w_specs, act_spec, b_spec, b_spec, b_spec, act_spec),
              out_specs=(P(), act_spec, w_specs, act_spec, P()),
              check_vma=False,
          )
      )
      return fn(w_sh, x_sh, pos_sh, seg_sh, msk_sh, up_sh)

    for mesh_dims in ((8, 1, 1), (4, 2, 1)):
      cp_sz = mesh_dims[1]
      dq_steps = 1 if cp_sz == 1 else 3
      sliding_tile = 1024 if cp_sz == 1 else 512
      for compute_dtype, min_cos in (
          ("float32", 0.9999),
          ("bfloat16", 0.999),
      ):
        ref_cfg = _make_sparse_test_config(
            fsdp_size=mesh_dims[0],
            cp_size=cp_sz,
            ep_size=mesh_dims[2],
            use_splash=True,
            compute_dtype=compute_dtype,
            param_dtype="float32",
        )
        r3_cfg = _make_sparse_test_config(
            fsdp_size=mesh_dims[0],
            cp_size=cp_sz,
            ep_size=mesh_dims[2],
            use_splash=True,
            compute_dtype=compute_dtype,
            param_dtype="float32",
            remat_policy="selective",
            moe_impl="pallas_gmm_v2",
            sc_collective_mode="dual_sc",
            routed_w_tiled=True,
            moe_bijective_permute=True,
            fuse_layer_projections=True,
            splash_global_block_size=256,
            splash_qk_diag_skip=True,
            splash_sv_diag_skip=True,
            splash_use_base2_exp=True,
            splash_dq_reduction_steps=dq_steps,
            splash_block_q=sliding_tile,
            splash_block_kv=sliding_tile,
            splash_block_q_dkv=sliding_tile,
            splash_block_kv_dkv=sliding_tile,
            replicate_router_gate=True,
            moe_slot_major_pack=True,
            save_extra_selective_residuals=True,
            prefetch_routed_weights=True,
            transpose_routed_down_ag=True,
            transpose_dense_rhs_ag=True,
            moe_zero_scatter_router=True,
            prefetch_routed_down=True,
            precompute_rope_tables=True,
        )
        loss_ref, y_ref, dw_ref, dx_ref, counts_ref = _run_cfg(ref_cfg)
        loss_got, y_got, dw_got, dx_got, counts_got = _run_cfg(r3_cfg)
        with self.subTest(
            is_global=is_global,
            mesh=mesh_dims,
            compute_dtype=compute_dtype,
        ):
          self.assertTrue(np.isfinite(float(np.asarray(loss_got))))
          self.assertLess(
              abs(float(np.asarray(loss_got)) - float(np.asarray(loss_ref))),
              5e-3,
          )
          self.assertEqual(dw_got.self_attn.sinks.dtype, jnp.float32)
          np.testing.assert_array_equal(np.asarray(counts_got), np.asarray(counts_ref))
          self.assertTrue(np.all(np.isfinite(np.asarray(y_got, np.float32))))
          self.assertTrue(np.all(np.isfinite(np.asarray(dx_got, np.float32))))
          self.assertGreaterEqual(_cosine_similarity(y_got, y_ref), min_cos)
          self.assertGreaterEqual(_cosine_similarity(dx_got, dx_ref), min_cos)
          for (path, g_got), g_ref in zip(
              jax.tree_util.tree_leaves_with_path(dw_got),
              jax.tree.leaves(dw_ref),
              strict=True,
          ):
            p_str = jax.tree_util.keystr(path)
            g_got_np = np.asarray(g_got, dtype=np.float32)
            self.assertTrue(np.all(np.isfinite(g_got_np)))
            self.assertGreater(float(np.linalg.norm(g_got_np)), 0.0)
            self.assertGreaterEqual(
                _cosine_similarity(g_got_np, g_ref),
                min_cos,
                msg=f"Stacked R3 all-knobs cosine below {min_cos} on {p_str}",
            )


if __name__ == "__main__":
  absltest.main()
