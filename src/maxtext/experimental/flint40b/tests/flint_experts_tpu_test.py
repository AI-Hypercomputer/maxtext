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

"""TPU v7x tests for Flint SwiGLU MLP, FP32 router, and MoE experts."""

from __future__ import annotations

import re
from typing import Any

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
from maxtext.experimental.flint40b import flint_experts
from maxtext.experimental.flint40b import flint_gmm_v2
from maxtext.experimental.flint40b import flint_mlp
from maxtext.experimental.flint40b import flint_moe_kernels
from maxtext.experimental.flint40b import flint_router
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import flint_types
from maxtext.experimental.flint40b import ops
from maxtext.experimental.flint40b.tests import hf_reference

P = PartitionSpec
_ALL_AXES: tuple[str, str, str] = ("fsdp", "cp", "ep")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
  a_f64 = np.asarray(a, dtype=np.float64).reshape(-1)
  b_f64 = np.asarray(b, dtype=np.float64).reshape(-1)
  denom = np.linalg.norm(a_f64) * np.linalg.norm(b_f64)
  if denom == 0.0:
    return 1.0 if np.allclose(a_f64, b_f64) else 0.0
  return float(np.dot(a_f64, b_f64) / denom)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, np.float64))))


def _mlp_pspecs() -> flint_types.FlintMLPWeights[PartitionSpec]:
  return flint_types.FlintMLPWeights(
      gate_proj=P(_ALL_AXES, None),
      up_proj=P(_ALL_AXES, None),
      down_proj=P(None, _ALL_AXES),
  )


def _moe_pspecs() -> flint_types.FlintMoEWeights[PartitionSpec]:
  return flint_types.FlintMoEWeights(
      router=flint_types.FlintRouterWeights(gate=P(_ALL_AXES, None)),
      shared_expert=_mlp_pspecs(),
      routed_experts=flint_types.FlintRoutedExpertsWeights(
          gate_up_proj=P("ep", ("fsdp", "cp"), None),
          down_proj=P("ep", None, ("fsdp", "cp")),
      ),
  )


def _torch_mlp_to_np(
    torch_mlp: hf_reference.TorchFlintMLP,
    dtype: jnp.dtype = jnp.dtype(jnp.float32),
) -> flint_types.FlintMLPWeights[np.ndarray]:
  return flint_types.FlintMLPWeights(
      gate_proj=torch_mlp.gate_proj.weight.detach().cpu().numpy().T.astype(dtype),
      up_proj=torch_mlp.up_proj.weight.detach().cpu().numpy().T.astype(dtype),
      down_proj=torch_mlp.down_proj.weight.detach().cpu().numpy().T.astype(dtype),
  )


def _torch_mlp_grads_np(
    torch_mlp: hf_reference.TorchFlintMLP,
) -> flint_types.FlintMLPWeights[np.ndarray]:
  assert torch_mlp.gate_proj.weight.grad is not None
  assert torch_mlp.up_proj.weight.grad is not None
  assert torch_mlp.down_proj.weight.grad is not None
  return flint_types.FlintMLPWeights(
      gate_proj=torch_mlp.gate_proj.weight.grad.detach().cpu().numpy().T,
      up_proj=torch_mlp.up_proj.weight.grad.detach().cpu().numpy().T,
      down_proj=torch_mlp.down_proj.weight.grad.detach().cpu().numpy().T,
  )


def _torch_moe_to_np(
    torch_moe: hf_reference.TorchFlintSparseMoeBlock,
    dtype: jnp.dtype = jnp.dtype(jnp.float32),
) -> flint_types.FlintMoEWeights[np.ndarray]:
  gu_list = []
  down_list = []
  for exp in torch_moe.experts:
    assert isinstance(exp, hf_reference.TorchFlintMLP)
    gate_w = exp.gate_proj.weight.detach().cpu().numpy().T.astype(dtype)
    up_w = exp.up_proj.weight.detach().cpu().numpy().T.astype(dtype)
    gu_list.append(np.concatenate([gate_w, up_w], axis=-1))
    down_list.append(exp.down_proj.weight.detach().cpu().numpy().T.astype(dtype))
  return flint_types.FlintMoEWeights(
      router=flint_types.FlintRouterWeights(gate=torch_moe.gate.weight.detach().cpu().numpy().T.astype(dtype)),
      shared_expert=_torch_mlp_to_np(torch_moe.shared_expert, dtype=dtype),
      routed_experts=flint_types.FlintRoutedExpertsWeights(
          gate_up_proj=np.stack(gu_list, axis=0),
          down_proj=np.stack(down_list, axis=0),
      ),
  )


def _torch_moe_grads_np(
    torch_moe: hf_reference.TorchFlintSparseMoeBlock,
) -> flint_types.FlintMoEWeights[np.ndarray]:
  assert torch_moe.gate.weight.grad is not None
  gu_grads = []
  down_grads = []
  for exp in torch_moe.experts:
    assert isinstance(exp, hf_reference.TorchFlintMLP)
    assert exp.gate_proj.weight.grad is not None
    assert exp.up_proj.weight.grad is not None
    assert exp.down_proj.weight.grad is not None
    gate_g = exp.gate_proj.weight.grad.detach().cpu().numpy().T
    up_g = exp.up_proj.weight.grad.detach().cpu().numpy().T
    gu_grads.append(np.concatenate([gate_g, up_g], axis=-1))
    down_grads.append(exp.down_proj.weight.grad.detach().cpu().numpy().T)
  return flint_types.FlintMoEWeights(
      router=flint_types.FlintRouterWeights(gate=torch_moe.gate.weight.grad.detach().cpu().numpy().T),
      shared_expert=_torch_mlp_grads_np(torch_moe.shared_expert),
      routed_experts=flint_types.FlintRoutedExpertsWeights(
          gate_up_proj=np.stack(gu_grads, axis=0),
          down_proj=np.stack(down_grads, axis=0),
      ),
  )


def _put_tree_on_mesh(tree_np, pspecs, mesh: Mesh):
  return jax.tree.map(
      lambda arr, sp: jax.device_put(arr, NamedSharding(mesh, sp)),
      tree_np,
      pspecs,
  )


class FlintExpertsTpuTest(parameterized.TestCase):

  def test_native_ragged_dot_in_lowered_hlo(self):
    self.assertLen(jax.devices(), 8)
    torch_cfg = flint_config.mini_flint_config().model
    torch_moe = hf_reference.TorchFlintSparseMoeBlock(torch_cfg)
    hf_reference.randomize_torch_flint(torch_moe, seed=7)
    moe_np = _torch_moe_to_np(torch_moe)

    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((8, 16, 128)).astype(np.float32)
    mask_np = np.ones((8, 16), dtype=np.int32)

    for fsdp_size, ep_size in ((8, 1), (1, 8)):
      cfg = flint_config.mini_flint_config(
          fsdp_size=fsdp_size,
          cp_size=1,
          ep_size=ep_size,
          global_batch_size=8,
          seq_len=16,
          compute_dtype="bfloat16",
          param_dtype="float32",
      )
      mesh = flint_sharding.create_mesh(cfg.sharding)
      w_specs = _moe_pspecs()
      act_spec = flint_sharding.activation_pspec()
      b_spec = flint_sharding.batch_pspec()

      moe_sharded = _put_tree_on_mesh(moe_np, w_specs, mesh)
      x_sharded = jax.device_put(jnp.asarray(x_np, dtype=jnp.bfloat16), NamedSharding(mesh, act_spec))
      mask_sharded = jax.device_put(mask_np, NamedSharding(mesh, b_spec))

      def _fwd(x_loc, w_loc, m_loc, cfg_local=cfg):
        out_loc, aux = flint_experts.flint_moe_forward(x_loc, w_loc, m_loc, cfg_local)
        return out_loc, aux.topk_ids

      jit_fn = jax.jit(
          jax.shard_map(
              _fwd,
              mesh=mesh,
              in_specs=(act_spec, w_specs, b_spec),
              out_specs=(act_spec, act_spec),
              check_vma=False,
          )
      )
      hlo_text = jit_fn.lower(x_sharded, moe_sharded, mask_sharded).as_text("hlo")
      self.assertIn("ragged-dot", hlo_text)
      self.assertGreaterEqual(hlo_text.count("ragged-dot"), 2)

  def test_mlp_fp32_and_bf16_parity_vs_pytorch(self):
    torch_mlp = hf_reference.TorchFlintMLP(hidden_size=128, intermediate_size=384)
    hf_reference.randomize_torch_flint(torch_mlp, seed=19)
    mlp_np = _torch_mlp_to_np(torch_mlp)

    rng = np.random.default_rng(19)
    x_np = rng.standard_normal((8, 16, 128)).astype(np.float32)
    upstream_np = rng.standard_normal((8, 16, 128)).astype(np.float32)

    x_pt = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    upstream_pt = torch.tensor(upstream_np, dtype=torch.float32)
    pt_out = torch_mlp(x_pt)
    pt_loss = (pt_out * upstream_pt).sum() / float(8 * 16)
    pt_loss.backward()
    assert x_pt.grad is not None
    pt_out_np = pt_out.detach().cpu().numpy()
    pt_dx_np = x_pt.grad.detach().cpu().numpy()
    pt_dw_np = _torch_mlp_grads_np(torch_mlp)

    for mesh_dims in ((1, 1, 1), (4, 2, 1), (2, 2, 2)):
      for compute_dtype in ("float32", "bfloat16"):
        cfg = flint_config.mini_flint_config(
            fsdp_size=mesh_dims[0],
            cp_size=mesh_dims[1],
            ep_size=mesh_dims[2],
            global_batch_size=8,
            seq_len=16,
            compute_dtype=compute_dtype,
            param_dtype="float32",
        )
        devs = jax.devices()[: cfg.sharding.num_devices]
        mesh = flint_sharding.create_mesh(cfg.sharding, devices=devs)
        w_specs = _mlp_pspecs()
        act_spec = flint_sharding.activation_pspec()

        w_jax = _put_tree_on_mesh(mlp_np, w_specs, mesh)
        comp_dt = cfg.train.compute_jnp_dtype
        x_jax = jax.device_put(jnp.asarray(x_np, dtype=comp_dt), NamedSharding(mesh, act_spec))
        up_jax = jax.device_put(
            jnp.asarray(upstream_np, dtype=comp_dt),
            NamedSharding(mesh, act_spec),
        )

        def _sharded_vg(w_loc, x_loc, up_loc, cfg_local=cfg):
          def _loss_and_out(w_in, x_in):
            y_loc = flint_mlp.flint_mlp_forward(x_in, w_in, cfg_local)
            loc_loss = jnp.sum(y_loc.astype(jnp.float32) * up_loc.astype(jnp.float32)) / float(8 * 16)
            return loc_loss, y_loc

          (loc_loss, y_loc), (dw_loc, dx_loc) = jax.value_and_grad(_loss_and_out, argnums=(0, 1), has_aux=True)(
              w_loc, x_loc
          )
          glob_loss = jax.lax.psum(loc_loss, _ALL_AXES)
          return y_loc, glob_loss, dw_loc, dx_loc

        vg_fn = jax.jit(
            jax.shard_map(
                _sharded_vg,
                mesh=mesh,
                in_specs=(w_specs, act_spec, act_spec),
                out_specs=(act_spec, P(), w_specs, act_spec),
                check_vma=False,
            )
        )
        got_out, got_loss, got_dw, got_dx = vg_fn(w_jax, x_jax, up_jax)
        got_out_np = np.asarray(got_out, dtype=np.float32)
        got_dx_np = np.asarray(got_dx, dtype=np.float32)
        got_dw_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float32), got_dw)

        self.assertTrue(np.all(np.isfinite(got_out_np)))
        self.assertTrue(np.all(np.isfinite(got_dx_np)))
        for leaf in jax.tree.leaves(got_dw_np):
          self.assertTrue(np.all(np.isfinite(leaf)))

        if compute_dtype == "float32":
          self.assertAlmostEqual(float(np.asarray(got_loss)), float(pt_loss.item()), delta=1e-5)
          self.assertLess(_max_abs_diff(got_out_np, pt_out_np), 1e-5)
          self.assertLess(_max_abs_diff(got_dx_np, pt_dx_np), 1e-5)
          self.assertLess(_max_abs_diff(got_dw_np.gate_proj, pt_dw_np.gate_proj), 1e-5)
          self.assertLess(_max_abs_diff(got_dw_np.up_proj, pt_dw_np.up_proj), 1e-5)
          self.assertLess(_max_abs_diff(got_dw_np.down_proj, pt_dw_np.down_proj), 1e-5)
        else:
          self.assertEqual(got_out.dtype, jnp.bfloat16)
          self.assertEqual(got_dw.gate_proj.dtype, jnp.float32)
          self.assertGreater(_cosine_similarity(got_out_np, pt_out_np), 0.999)
          self.assertGreater(_cosine_similarity(got_dx_np, pt_dx_np), 0.999)
          self.assertGreater(_cosine_similarity(got_dw_np.gate_proj, pt_dw_np.gate_proj), 0.999)
          self.assertGreater(_cosine_similarity(got_dw_np.up_proj, pt_dw_np.up_proj), 0.999)
          self.assertGreater(_cosine_similarity(got_dw_np.down_proj, pt_dw_np.down_proj), 0.999)

  def test_moe_fp32_and_bf16_fwd_bwd_parity_across_ep_meshes(self):
    base_cfg = flint_config.mini_flint_config(
        num_experts=16,
        num_experts_per_tok=6,
        hidden_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=64,
        global_batch_size=8,
        seq_len=16,
    )
    torch_moe = hf_reference.TorchFlintSparseMoeBlock(base_cfg.model)
    hf_reference.randomize_torch_flint(torch_moe, seed=29)
    moe_np = _torch_moe_to_np(torch_moe)

    rng = np.random.default_rng(29)
    x_np = rng.standard_normal((8, 16, 128)).astype(np.float32)
    upstream_np = rng.standard_normal((8, 16, 128)).astype(np.float32)

    x_pt = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    upstream_pt = torch.tensor(upstream_np, dtype=torch.float32)
    pt_out, pt_router = torch_moe(x_pt)
    pt_loss = (pt_out * upstream_pt).sum() / float(8 * 16)
    pt_loss.backward()
    assert x_pt.grad is not None

    pt_out_np = pt_out.detach().cpu().numpy()
    pt_ids_np = pt_router.topk_ids.detach().cpu().numpy().astype(np.int32)
    pt_weights_np = pt_router.topk_weights.detach().cpu().numpy()
    pt_dx_np = x_pt.grad.detach().cpu().numpy()
    pt_dw_np = _torch_moe_grads_np(torch_moe)

    meshes = ((8, 1, 1), (4, 1, 2), (2, 1, 4), (1, 1, 8), (2, 2, 2))
    for mesh_dims in meshes:
      for compute_dtype in ("float32", "bfloat16"):
        cfg = flint_config.mini_flint_config(
            num_experts=16,
            num_experts_per_tok=6,
            hidden_size=128,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=64,
            fsdp_size=mesh_dims[0],
            cp_size=mesh_dims[1],
            ep_size=mesh_dims[2],
            global_batch_size=8,
            seq_len=16,
            compute_dtype=compute_dtype,
            param_dtype="float32",
        )
        mesh = flint_sharding.create_mesh(cfg.sharding)
        w_specs = _moe_pspecs()
        act_spec = flint_sharding.activation_pspec()

        w_jax = _put_tree_on_mesh(moe_np, w_specs, mesh)
        comp_dt = cfg.train.compute_jnp_dtype
        x_jax = jax.device_put(jnp.asarray(x_np, dtype=comp_dt), NamedSharding(mesh, act_spec))
        up_jax = jax.device_put(
            jnp.asarray(upstream_np, dtype=comp_dt),
            NamedSharding(mesh, act_spec),
        )

        def _sharded_moe_vg(w_loc, x_loc, up_loc, cfg_local=cfg):
          def _loss_and_aux(w_in, x_in):
            y_loc, aux = flint_experts.flint_moe_forward(x_in, w_in, None, cfg_local)
            loc_loss = jnp.sum(y_loc.astype(jnp.float32) * up_loc.astype(jnp.float32)) / float(8 * 16)
            return loc_loss, (y_loc, aux.topk_ids, aux.topk_weights)

          (loc_loss, (y_loc, ids_loc, tw_loc)), (dw_loc, dx_loc) = jax.value_and_grad(
              _loss_and_aux, argnums=(0, 1), has_aux=True
          )(w_loc, x_loc)
          glob_loss = jax.lax.psum(loc_loss, _ALL_AXES)
          return y_loc, ids_loc, tw_loc, glob_loss, dw_loc, dx_loc

        vg_fn = jax.jit(
            jax.shard_map(
                _sharded_moe_vg,
                mesh=mesh,
                in_specs=(w_specs, act_spec, act_spec),
                out_specs=(
                    act_spec,
                    act_spec,
                    act_spec,
                    P(),
                    w_specs,
                    act_spec,
                ),
                check_vma=False,
            )
        )
        got_out, got_ids, got_tw, got_loss, got_dw, got_dx = vg_fn(w_jax, x_jax, up_jax)
        got_out_np = np.asarray(got_out, dtype=np.float32)
        got_ids_np = np.asarray(got_ids, dtype=np.int32)
        got_tw_np = np.asarray(got_tw, dtype=np.float32)
        got_dx_np = np.asarray(got_dx, dtype=np.float32)
        got_dw_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float32), got_dw)

        self.assertTrue(np.all(np.isfinite(got_out_np)))
        self.assertTrue(np.all(np.isfinite(got_tw_np)))
        self.assertTrue(np.all(np.isfinite(got_dx_np)))
        for leaf in jax.tree.leaves(got_dw_np):
          self.assertTrue(np.all(np.isfinite(leaf)))
          self.assertGreater(float(np.linalg.norm(leaf)), 0.0)

        if compute_dtype == "float32":
          np.testing.assert_array_equal(
              got_ids_np,
              pt_ids_np,
              err_msg=f"topk_ids mismatch on mesh {mesh_dims}",
          )
          self.assertAlmostEqual(float(np.asarray(got_loss)), float(pt_loss.item()), delta=1e-5)
          self.assertLess(_max_abs_diff(got_tw_np, pt_weights_np), 1e-5)
          self.assertLess(_max_abs_diff(got_out_np, pt_out_np), 1e-5)
          self.assertLess(_max_abs_diff(got_dx_np, pt_dx_np), 1e-5)
          self.assertLess(_max_abs_diff(got_dw_np.router.gate, pt_dw_np.router.gate), 1e-5)
          self.assertLess(
              _max_abs_diff(
                  got_dw_np.shared_expert.gate_proj,
                  pt_dw_np.shared_expert.gate_proj,
              ),
              1e-5,
          )
          self.assertLess(
              _max_abs_diff(
                  got_dw_np.shared_expert.up_proj,
                  pt_dw_np.shared_expert.up_proj,
              ),
              1e-5,
          )
          self.assertLess(
              _max_abs_diff(
                  got_dw_np.shared_expert.down_proj,
                  pt_dw_np.shared_expert.down_proj,
              ),
              1e-5,
          )
          self.assertLess(
              _max_abs_diff(
                  got_dw_np.routed_experts.gate_up_proj,
                  pt_dw_np.routed_experts.gate_up_proj,
              ),
              1e-5,
          )
          self.assertLess(
              _max_abs_diff(
                  got_dw_np.routed_experts.down_proj,
                  pt_dw_np.routed_experts.down_proj,
              ),
              1e-5,
          )
        else:
          self.assertEqual(got_out.dtype, jnp.bfloat16)
          self.assertEqual(got_dw.router.gate.dtype, jnp.float32)
          self.assertEqual(got_dw.routed_experts.gate_up_proj.dtype, jnp.float32)
          self.assertGreater(_cosine_similarity(got_out_np, pt_out_np), 0.999)
          self.assertGreater(_cosine_similarity(got_dx_np, pt_dx_np), 0.999)
          self.assertGreater(
              _cosine_similarity(got_dw_np.router.gate, pt_dw_np.router.gate),
              0.999,
          )
          self.assertGreater(
              _cosine_similarity(
                  got_dw_np.shared_expert.gate_proj,
                  pt_dw_np.shared_expert.gate_proj,
              ),
              0.999,
          )
          self.assertGreater(
              _cosine_similarity(
                  got_dw_np.shared_expert.up_proj,
                  pt_dw_np.shared_expert.up_proj,
              ),
              0.999,
          )
          self.assertGreater(
              _cosine_similarity(
                  got_dw_np.shared_expert.down_proj,
                  pt_dw_np.shared_expert.down_proj,
              ),
              0.999,
          )
          self.assertGreater(
              _cosine_similarity(
                  got_dw_np.routed_experts.gate_up_proj,
                  pt_dw_np.routed_experts.gate_up_proj,
              ),
              0.999,
          )
          self.assertGreater(
              _cosine_similarity(
                  got_dw_np.routed_experts.down_proj,
                  pt_dw_np.routed_experts.down_proj,
              ),
              0.999,
          )

  def test_moe_aux_loss_mask_and_mesh_invariance(self):
    rng = np.random.default_rng(43)
    gate_np = (rng.standard_normal((128, 16)) * 0.1).astype(np.float32)
    x_np = rng.standard_normal((8, 16, 128)).astype(np.float32)
    mask_np = (rng.uniform(size=(8, 16)) > 0.45).astype(np.int32)
    mask_np[0, 0] = 1
    mask_np[0, 1] = 0

    def _run_aux(mesh_dims: tuple[int, int, int], x_arr: np.ndarray):
      cfg = flint_config.mini_flint_config(
          fsdp_size=mesh_dims[0],
          cp_size=mesh_dims[1],
          ep_size=mesh_dims[2],
          global_batch_size=8,
          seq_len=16,
          compute_dtype="float32",
          param_dtype="float32",
      )
      devs = jax.devices()[: cfg.sharding.num_devices]
      mesh = flint_sharding.create_mesh(cfg.sharding, devices=devs)
      r_spec = flint_types.FlintRouterWeights(gate=P(_ALL_AXES, None))
      act_spec = flint_sharding.activation_pspec()
      b_spec = flint_sharding.batch_pspec()

      r_jax = _put_tree_on_mesh(flint_types.FlintRouterWeights(gate=gate_np), r_spec, mesh)
      x_jax = jax.device_put(x_arr, NamedSharding(mesh, act_spec))
      m_jax = jax.device_put(mask_np, NamedSharding(mesh, b_spec))

      def _sharded_aux(r_loc, x_loc, m_loc):
        global_valid = jax.lax.psum(jnp.sum((m_loc > 0).astype(jnp.float32)), _ALL_AXES)

        def _local_aux(r_in):
          aux = flint_router.flint_router_forward(x_loc, r_in, m_loc, cfg)
          return flint_router.compute_local_moe_aux_loss(
              aux.expert_counts,
              aux.prob_sum,
              global_valid,
              num_experts=cfg.model.num_experts,
              top_k=cfg.model.num_experts_per_tok,
          )

        loc_val, dr_loc = jax.value_and_grad(_local_aux)(r_loc)
        return jax.lax.psum(loc_val, _ALL_AXES), dr_loc

      fn = jax.jit(
          jax.shard_map(
              _sharded_aux,
              mesh=mesh,
              in_specs=(r_spec, act_spec, b_spec),
              out_specs=(P(), r_spec),
              check_vma=False,
          )
      )
      loss_val, dr = fn(r_jax, x_jax, m_jax)
      return float(np.asarray(loss_val)), np.asarray(dr.gate, dtype=np.float32)

    loss_111, grad_111 = _run_aux((1, 1, 1), x_np)
    loss_811, grad_811 = _run_aux((8, 1, 1), x_np)
    loss_222, grad_222 = _run_aux((2, 2, 2), x_np)

    self.assertAlmostEqual(loss_811, loss_111, delta=1e-6)
    self.assertAlmostEqual(loss_222, loss_111, delta=1e-6)
    self.assertLess(_max_abs_diff(grad_811, grad_111), 1e-6)
    self.assertLess(_max_abs_diff(grad_222, grad_111), 1e-6)
    self.assertGreater(float(np.linalg.norm(grad_811)), 0.0)

    # Perturb masked-out (loss_mask == 0) tokens heavily; value & grad must
    # remain strictly invariant.
    x_perturbed = x_np.copy()
    x_perturbed[mask_np == 0] = 250.0 * rng.standard_normal(x_perturbed[mask_np == 0].shape).astype(np.float32)
    loss_pert, grad_pert = _run_aux((2, 2, 2), x_perturbed)
    self.assertAlmostEqual(loss_pert, loss_222, delta=1e-6)
    self.assertLess(_max_abs_diff(grad_pert, grad_222), 1e-6)

  def test_capacity_factor_drops_excess_tokens_cleanly(self):
    rng = np.random.default_rng(71)
    base_cfg = flint_config.mini_flint_config(
        num_experts=16,
        num_experts_per_tok=6,
        hidden_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=64,
        global_batch_size=8,
        seq_len=16,
    )
    torch_moe = hf_reference.TorchFlintSparseMoeBlock(base_cfg.model)
    hf_reference.randomize_torch_flint(torch_moe, seed=71)
    moe_np = _torch_moe_to_np(torch_moe)

    # Skew every token strongly onto experts 0..5 so all N_ep tokens hit the
    # same 6 experts, and give expert 0..5 strictly positive SwiGLU weights
    # while zeroing shared_expert so output magnitude directly measures active
    # routed expert tokens.
    skewed_gate = np.full((128, 16), -1.0, dtype=np.float32)
    skewed_gate[:, 0] = 1.0
    skewed_gate[:, 1:6] = 0.2
    pos_gu = np.abs(moe_np.routed_experts.gate_up_proj) + 0.05
    pos_dn = np.abs(moe_np.routed_experts.down_proj) + 0.05
    zero_shared = jax.tree.map(np.zeros_like, moe_np.shared_expert)
    skewed_moe_np = flint_types.FlintMoEWeights(
        router=flint_types.FlintRouterWeights(gate=skewed_gate),
        shared_expert=zero_shared,
        routed_experts=flint_types.FlintRoutedExpertsWeights(
            gate_up_proj=pos_gu,
            down_proj=pos_dn,
        ),
    )
    x_pos_np = np.ones((8, 16, 128), dtype=np.float32) + (
        0.01 * np.abs(rng.standard_normal((8, 16, 128)).astype(np.float32))
    )
    upstream_np = rng.standard_normal((8, 16, 128)).astype(np.float32)

    for mesh_dims in ((8, 1, 1), (2, 2, 2)):

      def _eval_cap(cap_factor: float | None, w_tree_np, m_dims=mesh_dims):
        cfg = flint_config.mini_flint_config(
            num_experts=16,
            num_experts_per_tok=6,
            hidden_size=128,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=64,
            fsdp_size=m_dims[0],
            cp_size=m_dims[1],
            ep_size=m_dims[2],
            global_batch_size=8,
            seq_len=16,
            capacity_factor=cap_factor,
            compute_dtype="bfloat16",
            param_dtype="float32",
        )
        mesh = flint_sharding.create_mesh(cfg.sharding)
        w_specs = _moe_pspecs()
        act_spec = flint_sharding.activation_pspec()
        w_jax = _put_tree_on_mesh(w_tree_np, w_specs, mesh)
        x_jax = jax.device_put(
            jnp.asarray(x_pos_np, dtype=jnp.bfloat16),
            NamedSharding(mesh, act_spec),
        )
        up_jax = jax.device_put(
            jnp.asarray(upstream_np, dtype=jnp.bfloat16),
            NamedSharding(mesh, act_spec),
        )

        def _sharded(w_loc, x_loc, up_loc):
          def _loss(w_in, x_in):
            y_loc, _ = flint_experts.flint_moe_forward(x_in, w_in, None, cfg)
            loc_loss = jnp.sum(y_loc.astype(jnp.float32) * up_loc.astype(jnp.float32))
            return loc_loss, y_loc

          (loc_loss, y_loc), (dw_loc, dx_loc) = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(w_loc, x_loc)
          return y_loc, jax.lax.psum(loc_loss, _ALL_AXES), dw_loc, dx_loc

        fn = jax.jit(
            jax.shard_map(
                _sharded,
                mesh=mesh,
                in_specs=(w_specs, act_spec, act_spec),
                out_specs=(act_spec, P(), w_specs, act_spec),
                check_vma=False,
            )
        )
        y, loss, dw, dx = fn(w_jax, x_jax, up_jax)
        return (
            np.asarray(y, dtype=np.float32),
            float(np.asarray(loss)),
            jax.tree.map(lambda a: np.asarray(a, dtype=np.float32), dw),
            np.asarray(dx, dtype=np.float32),
        )

      y_dropless, _, _, _ = _eval_cap(None, skewed_moe_np)
      y_capped, loss_cap, dw_cap, dx_cap = _eval_cap(0.25, skewed_moe_np)

      self.assertTrue(np.isfinite(loss_cap))
      self.assertTrue(np.all(np.isfinite(y_capped)))
      self.assertTrue(np.all(np.isfinite(dx_cap)))
      for leaf in jax.tree.leaves(dw_cap):
        self.assertTrue(np.all(np.isfinite(leaf)))

      dropless_token_norms = np.linalg.norm(y_dropless.reshape(-1, 128), axis=1)
      capped_token_norms = np.linalg.norm(y_capped.reshape(-1, 128), axis=1)
      self.assertTrue(np.all(dropless_token_norms > 0.0))
      active_capped = int(np.sum(capped_token_norms > 0.0))
      self.assertGreater(active_capped, 0)
      self.assertLess(active_capped, int(dropless_token_norms.size))
      self.assertLess(float(np.sum(capped_token_norms)), float(np.sum(dropless_token_norms)))

      # Also verify full non-zero shared+routed MoE fwd+bwd finiteness at 0.25.
      y_full, loss_full, dw_full, dx_full = _eval_cap(
          0.25,
          flint_types.FlintMoEWeights(
              router=flint_types.FlintRouterWeights(gate=skewed_gate),
              shared_expert=moe_np.shared_expert,
              routed_experts=moe_np.routed_experts,
          ),
      )
      self.assertTrue(np.isfinite(loss_full))
      self.assertTrue(np.all(np.isfinite(y_full)))
      self.assertTrue(np.all(np.isfinite(dx_full)))
      for leaf in jax.tree.leaves(dw_full):
        self.assertTrue(np.all(np.isfinite(leaf)))

  def _make_aligned_moe_config(
      self,
      mesh_dims: tuple[int, int, int],
      *,
      compute_dtype: str = "float32",
      capacity_factor: float | None = None,
      moe_impl: str = "xla_ragged_dot",
      gmm_tile_m: int | None = None,
      tgmm_tile_m: int | None = None,
      sc_collective_mode: str = "off",
      routed_w_tiled: bool = False,
      moe_bijective_permute: bool = False,
      fuse_layer_projections: bool = False,
      **extra_overrides,
  ) -> flint_config.FlintConfig:
    return flint_config.mini_flint_config(
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=16,
        num_experts_per_tok=6,
        fsdp_size=mesh_dims[0],
        cp_size=mesh_dims[1],
        ep_size=mesh_dims[2],
        global_batch_size=8,
        seq_len=256,
        compute_dtype=compute_dtype,
        param_dtype="float32",
        capacity_factor=capacity_factor,
        moe_impl=moe_impl,
        gmm_tile_m=gmm_tile_m,
        tgmm_tile_m=tgmm_tile_m,
        sc_collective_mode=sc_collective_mode,
        routed_w_tiled=routed_w_tiled,
        moe_bijective_permute=moe_bijective_permute,
        fuse_layer_projections=fuse_layer_projections,
        **extra_overrides,
    )

  def _eval_moe_vg(
      self,
      cfg: flint_config.FlintConfig,
      moe_np: flint_types.FlintMoEWeights[np.ndarray],
      x_np: np.ndarray,
      upstream_np: np.ndarray,
      loss_mask_np: np.ndarray | None = None,
  ):
    mesh = flint_sharding.create_mesh(cfg.sharding)
    dec_specs = flint_sharding.param_pspecs(cfg).decoder
    w_specs = dec_specs.tail_moe[0].moe if cfg.model.scan_layers else dec_specs.unscanned_moe[0].moe
    act_spec = flint_sharding.activation_pspec()
    tok_spec = flint_sharding.batch_pspec()
    comp_dt = cfg.train.compute_jnp_dtype
    w_jax = _put_tree_on_mesh(moe_np, w_specs, mesh)
    x_jax = jax.device_put(jnp.asarray(x_np, dtype=comp_dt), NamedSharding(mesh, act_spec))
    up_jax = jax.device_put(jnp.asarray(upstream_np, dtype=comp_dt), NamedSharding(mesh, act_spec))
    use_mask = loss_mask_np is not None
    msk_arr = loss_mask_np if loss_mask_np is not None else np.ones(x_np.shape[:2], dtype=np.int32)
    msk_jax = jax.device_put(jnp.asarray(msk_arr, dtype=jnp.int32), NamedSharding(mesh, tok_spec))

    def _sharded_vg(w_loc, x_loc, up_loc, msk_loc):
      def _loss_and_aux(w_in, x_in):
        y_loc, aux = flint_experts.flint_moe_forward(x_in, w_in, msk_loc if use_mask else None, cfg)
        loc_loss = jnp.sum(y_loc.astype(jnp.float32) * up_loc.astype(jnp.float32)) / float(x_np.shape[0] * x_np.shape[1])
        return loc_loss, (y_loc, aux.expert_counts)

      (loc_loss, (y_loc, counts_loc)), (dw_loc, dx_loc) = jax.value_and_grad(_loss_and_aux, argnums=(0, 1), has_aux=True)(
          w_loc, x_loc
      )
      empty_spec = P()
      dw_loc = jax.tree.map(
          lambda g, sp: (jax.lax.psum(g, _ALL_AXES) if (cfg.sharding.num_devices > 1 and sp == empty_spec) else g),
          dw_loc,
          w_specs,
      )
      glob_loss = jax.lax.psum(loc_loss, _ALL_AXES)
      glob_counts = jax.lax.psum(counts_loc, _ALL_AXES)
      return y_loc, glob_counts, glob_loss, dw_loc, dx_loc

    vg_fn = jax.jit(
        jax.shard_map(
            _sharded_vg,
            mesh=mesh,
            in_specs=(w_specs, act_spec, act_spec, tok_spec),
            out_specs=(act_spec, P(), P(), w_specs, act_spec),
            check_vma=False,
        )
    )
    y_got, counts_got, loss_got, dw_got, dx_got = vg_fn(w_jax, x_jax, up_jax, msk_jax)
    return (
        vg_fn,
        (w_jax, x_jax, up_jax, msk_jax),
        np.asarray(y_got, dtype=np.float32),
        np.asarray(counts_got, dtype=np.float32),
        float(np.asarray(loss_got)),
        jax.tree.map(lambda a: np.asarray(a, dtype=np.float32), dw_got),
        np.asarray(dx_got, dtype=np.float32),
    )

  def test_k1_pallas_gmm_v2_hlo_and_fwd_bwd_parity(self):
    ref_cfg = self._make_aligned_moe_config((8, 1, 1))
    torch_moe = hf_reference.TorchFlintSparseMoeBlock(ref_cfg.model)
    hf_reference.randomize_torch_flint(torch_moe, seed=131)
    moe_np = _torch_moe_to_np(torch_moe)

    rng = np.random.default_rng(131)
    x_np = rng.standard_normal((8, 256, 256)).astype(np.float32)
    upstream_np = rng.standard_normal((8, 256, 256)).astype(np.float32)

    # 1) Verify lowered HLO contains gmm_v2 & tgmm_v2 and no ragged-dot.
    cfg_hlo = self._make_aligned_moe_config((4, 1, 2), compute_dtype="bfloat16", moe_impl="pallas_gmm_v2")
    vg_fn, args, _, _, _, _, _ = self._eval_moe_vg(cfg_hlo, moe_np, x_np, upstream_np)
    hlo_dbg = vg_fn.lower(*args).as_text("hlo", debug_info=True)
    self.assertIn("gmm_v2", hlo_dbg)
    self.assertIn("tgmm_v2", hlo_dbg)
    self.assertNotIn("ragged-dot", vg_fn.lower(*args).as_text("hlo"))

    # 2) Dropless, capacity-capped, and tile-overridden FP32/BF16 parity.
    cases = (
        ((8, 1, 1), "float32", None, None, None),
        ((4, 1, 2), "float32", None, 128, 128),
        ((8, 1, 1), "bfloat16", None, None, None),
        ((4, 1, 2), "bfloat16", None, None, None),
        ((4, 1, 2), "bfloat16", 0.5, 128, 128),
    )
    for mesh_dims, compute_dtype, cap_factor, gmm_tm, tgmm_tm in cases:
      base_cfg = self._make_aligned_moe_config(
          mesh_dims,
          compute_dtype=compute_dtype,
          capacity_factor=cap_factor,
          moe_impl="xla_ragged_dot",
      )
      cand_cfg = self._make_aligned_moe_config(
          mesh_dims,
          compute_dtype=compute_dtype,
          capacity_factor=cap_factor,
          moe_impl="pallas_gmm_v2",
          gmm_tile_m=gmm_tm,
          tgmm_tile_m=tgmm_tm,
      )
      _, _, y_ref, _, loss_ref, dw_ref, dx_ref = self._eval_moe_vg(base_cfg, moe_np, x_np, upstream_np)
      _, _, y_got, _, loss_got, dw_got, dx_got = self._eval_moe_vg(cand_cfg, moe_np, x_np, upstream_np)
      with self.subTest(
          mesh=mesh_dims,
          dtype=compute_dtype,
          cap=cap_factor,
          gmm_tm=gmm_tm,
      ):
        self.assertTrue(np.isfinite(loss_got))
        self.assertTrue(np.all(np.isfinite(y_got)))
        self.assertTrue(np.all(np.isfinite(dx_got)))
        for leaf in jax.tree.leaves(dw_got):
          self.assertTrue(np.all(np.isfinite(leaf)))
        min_cos = 0.9999 if compute_dtype == "float32" else 0.9995
        self.assertGreaterEqual(_cosine_similarity(y_got, y_ref), min_cos)
        self.assertGreaterEqual(_cosine_similarity(dx_got, dx_ref), min_cos)
        for g_got, g_ref in zip(jax.tree.leaves(dw_got), jax.tree.leaves(dw_ref), strict=True):
          self.assertGreaterEqual(_cosine_similarity(g_got, g_ref), min_cos)
        if compute_dtype == "float32":
          self.assertLess(abs(loss_got - loss_ref), 1e-4)
          self.assertLess(_max_abs_diff(y_got, y_ref), 5e-4)
          self.assertLess(_max_abs_diff(dx_got, dx_ref), 5e-4)

    # 3) Skewed routing: experts 6..15 receive 0 tokens (including empty EP1!).
    # Use strictly positive-sum inputs (`|x| + 0.5`) with small positive router
    # columns `0..5` and negative columns `6..15` so linear projection without
    # bias puts every token's top-6 strictly into `{0..5}` (`counts[6:] == 0`),
    # leaving experts `6, 7` empty on EP0 and the entire EP1 shard `8..15`
    # (`num_gm == 0`) empty on `(4, 1, 2)`.
    x_pos_skew = (np.abs(x_np) + 0.5).astype(np.float32)
    skewed_gate = np.full((256, 16), -0.5, dtype=np.float32)
    for idx in range(6):
      skewed_gate[:, idx] = 0.01 * (2.0 - 0.2 * idx)
    skewed_moe_np = flint_types.FlintMoEWeights(
        router=flint_types.FlintRouterWeights(gate=skewed_gate),
        shared_expert=moe_np.shared_expert,
        routed_experts=moe_np.routed_experts,
    )
    for mesh_dims in ((8, 1, 1), (4, 1, 2)):
      for cap_factor in (None, 0.5):
        base_skew_cfg = self._make_aligned_moe_config(
            mesh_dims,
            compute_dtype="bfloat16",
            capacity_factor=cap_factor,
            moe_impl="xla_ragged_dot",
        )
        cand_skew_cfg = self._make_aligned_moe_config(
            mesh_dims,
            compute_dtype="bfloat16",
            capacity_factor=cap_factor,
            moe_impl="pallas_gmm_v2",
        )
        _, _, y_ref, counts_ref, _, dw_ref, dx_ref = self._eval_moe_vg(
            base_skew_cfg, skewed_moe_np, x_pos_skew, upstream_np
        )
        _, _, y_got, counts_got, _, dw_got, dx_got = self._eval_moe_vg(
            cand_skew_cfg, skewed_moe_np, x_pos_skew, upstream_np
        )
        np.testing.assert_array_equal(counts_got, counts_ref)
        np.testing.assert_array_equal(counts_got[6:], np.zeros(10))
        np.testing.assert_array_equal(
            dw_got.routed_experts.gate_up_proj[6:],
            np.zeros_like(dw_got.routed_experts.gate_up_proj[6:]),
        )
        np.testing.assert_array_equal(
            dw_got.routed_experts.down_proj[6:],
            np.zeros_like(dw_got.routed_experts.down_proj[6:]),
        )
        self.assertGreaterEqual(_cosine_similarity(y_got, y_ref), 0.9995)
        self.assertGreaterEqual(_cosine_similarity(dx_got, dx_ref), 0.9995)
        self.assertGreaterEqual(
            _cosine_similarity(
                dw_got.routed_experts.gate_up_proj[:6],
                dw_ref.routed_experts.gate_up_proj[:6],
            ),
            0.9995,
        )
        self.assertGreaterEqual(
            _cosine_similarity(
                dw_got.routed_experts.down_proj[:6],
                dw_ref.routed_experts.down_proj[:6],
            ),
            0.9995,
        )

  def test_k2_k4_k5_dual_sc_bijective_permute_and_fused_mlp_parity(self):
    ref_cfg = self._make_aligned_moe_config((8, 1, 1))
    torch_moe = hf_reference.TorchFlintSparseMoeBlock(ref_cfg.model)
    hf_reference.randomize_torch_flint(torch_moe, seed=173)
    moe_np = _torch_moe_to_np(torch_moe)

    rng = np.random.default_rng(173)
    x_np = rng.standard_normal((8, 256, 256)).astype(np.float32)
    upstream_np = rng.standard_normal((8, 256, 256)).astype(np.float32)

    for mesh_dims in ((8, 1, 1), (4, 1, 2)):
      for compute_dtype in ("float32", "bfloat16"):
        base_cfg = self._make_aligned_moe_config(mesh_dims, compute_dtype=compute_dtype)
        _, _, y_ref, _, _, dw_ref, dx_ref = self._eval_moe_vg(base_cfg, moe_np, x_np, upstream_np)

        # K2: sc_collective_mode='dual_sc', routed_w_tiled=True
        k2_cfg = self._make_aligned_moe_config(
            mesh_dims,
            compute_dtype=compute_dtype,
            sc_collective_mode="dual_sc",
            routed_w_tiled=True,
        )
        _, _, y_k2, _, _, dw_k2, dx_k2 = self._eval_moe_vg(k2_cfg, moe_np, x_np, upstream_np)
        # K4 + K5 together & standalone K4: moe_bijective_permute + fused MLP
        k4_cfg = self._make_aligned_moe_config(
            mesh_dims,
            compute_dtype=compute_dtype,
            moe_bijective_permute=True,
            fuse_layer_projections=True,
        )
        _, _, y_k4, _, _, dw_k4, dx_k4 = self._eval_moe_vg(k4_cfg, moe_np, x_np, upstream_np)

        for label, y_got, dw_got, dx_got in (
            ("k2_dual_sc_tiled", y_k2, dw_k2, dx_k2),
            ("k4_k5_bijective_fused", y_k4, dw_k4, dx_k4),
        ):
          with self.subTest(knob=label, mesh=mesh_dims, compute_dtype=compute_dtype):
            if compute_dtype == "float32":
              self.assertLess(_max_abs_diff(y_got, y_ref), 1e-5)
              self.assertLess(_max_abs_diff(dx_got, dx_ref), 1e-5)
              for g_got, g_ref in zip(jax.tree.leaves(dw_got), jax.tree.leaves(dw_ref), strict=True):
                self.assertLess(_max_abs_diff(g_got, g_ref), 1e-5)
            else:
              self.assertGreaterEqual(_cosine_similarity(y_got, y_ref), 0.9999)
              self.assertGreaterEqual(_cosine_similarity(dx_got, dx_ref), 0.9999)
              for g_got, g_ref in zip(jax.tree.leaves(dw_got), jax.tree.leaves(dw_ref), strict=True):
                self.assertGreaterEqual(_cosine_similarity(g_got, g_ref), 0.9999)

    # K4 under capacity_factor=1.0 vs baseline capacity_factor=1.0
    base_cap = self._make_aligned_moe_config((4, 1, 2), compute_dtype="float32", capacity_factor=1.0)
    k4_cap = self._make_aligned_moe_config(
        (4, 1, 2),
        compute_dtype="float32",
        capacity_factor=1.0,
        moe_bijective_permute=True,
    )
    _, _, y_c0, _, _, dw_c0, dx_c0 = self._eval_moe_vg(base_cap, moe_np, x_np, upstream_np)
    _, _, y_c1, _, _, dw_c1, dx_c1 = self._eval_moe_vg(k4_cap, moe_np, x_np, upstream_np)
    self.assertLess(_max_abs_diff(y_c1, y_c0), 1e-5)
    self.assertLess(_max_abs_diff(dx_c1, dx_c0), 1e-5)
    for g_got, g_ref in zip(jax.tree.leaves(dw_c1), jax.tree.leaves(dw_c0), strict=True):
      self.assertLess(_max_abs_diff(g_got, g_ref), 1e-5)

  def test_r3_transposed_down_ag_dense_rhs_and_zero_scatter_router_parity(self):
    ref_cfg = self._make_aligned_moe_config((8, 1, 1))
    torch_moe = hf_reference.TorchFlintSparseMoeBlock(ref_cfg.model)
    hf_reference.randomize_torch_flint(torch_moe, seed=313)
    moe_np = _torch_moe_to_np(torch_moe)

    rng = np.random.default_rng(313)
    x_np = rng.standard_normal((8, 256, 256)).astype(np.float32)
    upstream_np = rng.standard_normal((8, 256, 256)).astype(np.float32)
    mask_np = np.ones((8, 256), dtype=np.int32)
    mask_np[:, 127] = 0
    mask_np[:, 255] = 0

    # 1) Verify moe_zero_scatter_router + moe_bijective_permute emits ZERO
    # HLO scatter ops across fwd+bwd while matching FP32 baseline bit-for-bit
    # on expert_counts and within 1e-5 max abs diff on all activations/grads.
    for mesh_dims in ((8, 1, 1), (4, 1, 2)):
      base_f32 = self._make_aligned_moe_config(mesh_dims, compute_dtype="float32", moe_bijective_permute=True)
      zero_sc_f32 = self._make_aligned_moe_config(
          mesh_dims,
          compute_dtype="float32",
          moe_bijective_permute=True,
          moe_zero_scatter_router=True,
          moe_slot_major_pack=True,
          replicate_router_gate=True,
      )
      _, _, y_ref, c_ref, loss_ref, dw_ref, dx_ref = self._eval_moe_vg(
          base_f32, moe_np, x_np, upstream_np, loss_mask_np=mask_np
      )
      vg_fn, args, y_zs, c_zs, loss_zs, dw_zs, dx_zs = self._eval_moe_vg(
          zero_sc_f32, moe_np, x_np, upstream_np, loss_mask_np=mask_np
      )
      hlo_zs = vg_fn.lower(*args).compile().as_text()
      self.assertIsNone(re.search(r"(?<![A-Za-z0-9_-])scatter\(", hlo_zs))
      np.testing.assert_array_equal(c_zs, c_ref)
      self.assertLess(abs(loss_zs - loss_ref), 1e-5)
      self.assertLess(_max_abs_diff(y_zs, y_ref), 1e-5)
      self.assertLess(_max_abs_diff(dx_zs, dx_ref), 1e-5)
      for g_got, g_ref in zip(jax.tree.leaves(dw_zs), jax.tree.leaves(dw_ref), strict=True):
        self.assertLess(_max_abs_diff(g_got, g_ref), 1e-5)

    # 2) R3-K1a (transpose_routed_down_ag), R3-K1b (transpose_dense_rhs_ag),
    # R3-K2, and R3-K7 (gmm_tile_m=256) across xla_ragged_dot & pallas_gmm_v2.
    cases = (
        ((8, 1, 1), "float32", "xla_ragged_dot", None, None),
        ((8, 1, 1), "float32", "pallas_gmm_v2", None, None),
        ((4, 1, 2), "float32", "pallas_gmm_v2", 256, None),
        ((8, 1, 1), "bfloat16", "pallas_gmm_v2", 256, None),
        ((4, 1, 2), "bfloat16", "pallas_gmm_v2", None, 0.5),
    )
    for mesh_dims, comp_dt, moe_impl, gmm_tm, cap_f in cases:
      base_cfg = self._make_aligned_moe_config(
          mesh_dims,
          compute_dtype=comp_dt,
          capacity_factor=cap_f,
          moe_impl="xla_ragged_dot",
          moe_slot_major_pack=(cap_f is not None),
      )
      cand_cfg = self._make_aligned_moe_config(
          mesh_dims,
          compute_dtype=comp_dt,
          capacity_factor=cap_f,
          moe_impl=moe_impl,
          gmm_tile_m=gmm_tm,
          sc_collective_mode="dual_sc",
          routed_w_tiled=True,
          moe_bijective_permute=True,
          moe_slot_major_pack=True,
          fuse_layer_projections=True,
          replicate_router_gate=True,
          transpose_routed_down_ag=True,
          transpose_dense_rhs_ag=True,
          moe_zero_scatter_router=True,
      )
      _, _, y_r, c_r, loss_r, dw_r, dx_r = self._eval_moe_vg(base_cfg, moe_np, x_np, upstream_np, loss_mask_np=mask_np)
      _, _, y_g, c_g, loss_g, dw_g, dx_g = self._eval_moe_vg(cand_cfg, moe_np, x_np, upstream_np, loss_mask_np=mask_np)
      with self.subTest(mesh=mesh_dims, dtype=comp_dt, impl=moe_impl, tm=gmm_tm, cap=cap_f):
        self.assertTrue(np.isfinite(loss_g))
        if comp_dt == "float32":
          np.testing.assert_array_equal(c_g, c_r)
          self.assertLess(abs(loss_g - loss_r), 1e-4)
          self.assertLess(_max_abs_diff(y_g, y_r), 5e-4)
          self.assertLess(_max_abs_diff(dx_g, dx_r), 5e-4)
        min_cos = 0.9999 if comp_dt == "float32" else 0.9995
        self.assertGreaterEqual(_cosine_similarity(y_g, y_r), min_cos)
        self.assertGreaterEqual(_cosine_similarity(dx_g, dx_r), min_cos)
        for g_got, g_ref in zip(jax.tree.leaves(dw_g), jax.tree.leaves(dw_r), strict=True):
          self.assertTrue(np.all(np.isfinite(g_got)))
          self.assertGreaterEqual(_cosine_similarity(g_got, g_ref), min_cos)

  def test_fp8_pallas_gmm_v2_and_moe_fwd_bwd_parity(self):
    rng = np.random.default_rng(2026)
    lhs_np = rng.standard_normal((512, 256)).astype(np.float32)
    rhs_kn_np = (rng.standard_normal((4, 256, 256)) * 0.5).astype(np.float32)
    gs_np = np.array([160, 0, 96, 256], dtype=np.int32)
    offsets = [0, 160, 160, 256, 512]

    # Exact float32 grouped matmul reference: `lhs [M, K] @ rhs [G, K, N]`.
    ref_gmm = np.zeros((512, 256), dtype=np.float32)
    for g in range(4):
      s, e = offsets[g], offsets[g + 1]
      if e > s:
        ref_gmm[s:e] = lhs_np[s:e] @ rhs_kn_np[g]

    lhs_bf16 = jnp.asarray(lhs_np, dtype=jnp.bfloat16)
    gs_jax = jnp.asarray(gs_np)
    for transpose_rhs, rhs_scale_on_k in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
      rhs_arr = np.swapaxes(rhs_kn_np, 1, 2) if transpose_rhs else rhs_kn_np
      # If `rhs_scale_on_k=True`, quantize along the `K` axis? Wait:
      # `rhs_scale_on_k=False` means scale has length `N` (reduced over `K`),
      # `rhs_scale_on_k=True` means scale has length `K` (reduced over `N`).
      if transpose_rhs:
        # `rhs_arr` is `[G, N, K]`.
        red_ax = 1 if rhs_scale_on_k else 2
      else:
        # `rhs_arr` is `[G, K, N]`.
        red_ax = 2 if rhs_scale_on_k else 1
      q_rhs, raw_sc = ops.quantize_fp8_axis(jnp.asarray(rhs_arr), reduce_axis=red_ax)
      rhs_sc = jnp.swapaxes(raw_sc, 1, 2) if raw_sc.shape[1] != 1 else raw_sc

      def _run_fp8_gmm(
          lhs_input: jax.Array,
          quant_rhs: jax.Array = q_rhs,
          scale: jax.Array = rhs_sc,
          group_sizes: jax.Array = gs_jax,
          is_transposed: bool = transpose_rhs,
          scale_on_k: bool = rhs_scale_on_k,
      ) -> jax.Array:
        return flint_gmm_v2.gmm_v2(
            lhs_input,
            quant_rhs,
            group_sizes,
            preferred_element_type=jnp.bfloat16,
            transpose_rhs=is_transposed,
            zero_initialize=True,
            rhs_scale=scale,
            rhs_scale_on_k=scale_on_k,
            lhs_qtype=jnp.float8_e4m3fn,
        )

      got_gmm = jax.jit(_run_fp8_gmm)(lhs_bf16)
      got_gmm_np = np.asarray(got_gmm, dtype=np.float32)
      self.assertTrue(np.all(np.isfinite(got_gmm_np)))
      self.assertGreaterEqual(_cosine_similarity(got_gmm_np, ref_gmm), 0.998)

    # Standalone `tgmm_v2` native-`bfloat16` wgrad parity vs `f32` reference.
    do_np = rng.standard_normal((512, 256)).astype(np.float32)
    ref_tgmm = np.zeros((4, 256, 256), dtype=np.float32)
    for g in range(4):
      s, e = offsets[g], offsets[g + 1]
      if e > s:
        ref_tgmm[g] = lhs_np[s:e].T @ do_np[s:e]
    do_bf16 = jnp.asarray(do_np, dtype=jnp.bfloat16)
    got_tgmm = jax.jit(
        lambda lhs_in, rhs_in, group_sizes: flint_gmm_v2.tgmm_v2(
            lhs_in,
            rhs_in,
            group_sizes,
            num_actual_groups=4,
            preferred_element_type=jnp.bfloat16,
        )
    )(lhs_bf16, do_bf16, gs_jax)
    got_tgmm_np = np.asarray(got_tgmm, dtype=np.float32)
    self.assertTrue(np.all(np.isfinite(got_tgmm_np)))
    np.testing.assert_array_equal(got_tgmm_np[1], np.zeros((256, 256), dtype=np.float32))
    for g in (0, 2, 3):
      self.assertGreaterEqual(_cosine_similarity(got_tgmm_np[g], ref_tgmm[g]), 0.998)

    # Full `flint_moe_forward` fwd+bwd parity with `use_fp8=True` on 8 devices.
    ref_cfg = self._make_aligned_moe_config((8, 1, 1))
    torch_moe = hf_reference.TorchFlintSparseMoeBlock(ref_cfg.model)
    hf_reference.randomize_torch_flint(torch_moe, seed=409)
    moe_np = _torch_moe_to_np(torch_moe)
    x_np = rng.standard_normal((8, 256, 256)).astype(np.float32)
    upstream_np = rng.standard_normal((8, 256, 256)).astype(np.float32)

    for mesh_dims in ((8, 1, 1), (4, 1, 2)):
      for moe_impl in ("pallas_gmm_v2", "xla_ragged_dot"):
        bf16_cfg = self._make_aligned_moe_config(
            mesh_dims,
            compute_dtype="bfloat16",
            moe_impl=moe_impl,
            sc_collective_mode="dual_sc",
            routed_w_tiled=True,
            moe_bijective_permute=True,
            moe_slot_major_pack=True,
            fuse_layer_projections=True,
            replicate_router_gate=True,
            transpose_routed_down_ag=True,
            transpose_dense_rhs_ag=True,
            moe_zero_scatter_router=True,
            use_fp8=False,
        )
        fp8_cfg = self._make_aligned_moe_config(
            mesh_dims,
            compute_dtype="bfloat16",
            moe_impl=moe_impl,
            sc_collective_mode="dual_sc",
            routed_w_tiled=True,
            moe_bijective_permute=True,
            moe_slot_major_pack=True,
            fuse_layer_projections=True,
            replicate_router_gate=True,
            transpose_routed_down_ag=True,
            transpose_dense_rhs_ag=True,
            moe_zero_scatter_router=True,
            use_fp8=True,
        )
        _, _, y_ref, c_ref, loss_ref, dw_ref, dx_ref = self._eval_moe_vg(bf16_cfg, moe_np, x_np, upstream_np)
        vg_fp8, args_fp8, y_fp8, c_fp8, loss_fp8, dw_fp8, dx_fp8 = self._eval_moe_vg(fp8_cfg, moe_np, x_np, upstream_np)
        if moe_impl == "pallas_gmm_v2" and mesh_dims == (8, 1, 1):
          hlo_fp8 = vg_fp8.lower(*args_fp8).as_text("hlo")
          self.assertIn("f8e4m3fn[", hlo_fp8)
          self.assertIn("all-gather", hlo_fp8)

        with self.subTest(mesh=mesh_dims, moe_impl=moe_impl):
          np.testing.assert_array_equal(c_fp8, c_ref)
          self.assertTrue(np.isfinite(loss_fp8))
          self.assertLess(abs(loss_fp8 - loss_ref), 0.05)
          self.assertGreaterEqual(_cosine_similarity(y_fp8, y_ref), 0.99)
          self.assertGreaterEqual(_cosine_similarity(dx_fp8, dx_ref), 0.99)
          for g_got, g_ref in zip(jax.tree.leaves(dw_fp8), jax.tree.leaves(dw_ref), strict=True):
            self.assertTrue(np.all(np.isfinite(g_got)))
            self.assertGreaterEqual(_cosine_similarity(g_got, g_ref), 0.99)

  def test_tgmm_v2_40b_gate_up_and_down_tiling(self):
    vmem_48m = 48 * 1024 * 1024
    tiles_gu = flint_gmm_v2.calculate_tgmm_tiling(
        flint_gmm_v2.Dimensions(
            size_m=16384,
            size_k=2048,
            size_n=1024,
            size_group=256,
            size_lhs_group=256,
            size_lhs_sublane=8,
        ),
        lhs_dtype=jnp.dtype(jnp.bfloat16),
        rhs_dtype=jnp.dtype(jnp.bfloat16),
        out_dtype=jnp.dtype(jnp.float32),
        acc_dtype=jnp.dtype(jnp.float32),
        vmem_limit_bytes=vmem_48m,
    )
    self.assertEqual(
        (tiles_gu.tile_m, tiles_gu.tile_k, tiles_gu.tile_n),
        (512, 2048, 512),
    )
    tiles_dn = flint_gmm_v2.calculate_tgmm_tiling(
        flint_gmm_v2.Dimensions(
            size_m=16384,
            size_k=512,
            size_n=2048,
            size_group=256,
            size_lhs_group=256,
            size_lhs_sublane=8,
        ),
        lhs_dtype=jnp.dtype(jnp.bfloat16),
        rhs_dtype=jnp.dtype(jnp.bfloat16),
        out_dtype=jnp.dtype(jnp.float32),
        acc_dtype=jnp.dtype(jnp.float32),
        vmem_limit_bytes=vmem_48m,
    )
    self.assertEqual(
        (tiles_dn.tile_m, tiles_dn.tile_k, tiles_dn.tile_n),
        (512, 512, 2048),
    )

  def test_v4_moe_sc_kernels_and_router_topk_parity(self):
    rng = np.random.default_rng(2027)

    # 1) Dual-SC 32-subcore counting sort bit-exact parity.
    for num_experts in (128, 256):
      flat_np = rng.integers(0, num_experts, size=(8192,), dtype=np.int32)
      flat_j = jnp.asarray(flat_np)

      @jax.jit
      def _run_cs(arr, n_exp=num_experts):
        res = flint_moe_kernels.counting_sort_permute_sc(arr, n_exp)
        assert res is not None
        return res

      cs_hlo = _run_cs.lower(flat_j).as_text("hlo", debug_info=True)
      self.assertIn("sc_counting_sort", cs_hlo)
      perm_got, inv_got, gs_got = _run_cs(flat_j)
      perm_ref = jnp.argsort(flat_j, stable=True)
      inv_ref = jnp.argsort(perm_ref)
      gs_ref = jnp.bincount(flat_j, length=num_experts).astype(jnp.int32)
      np.testing.assert_array_equal(np.asarray(perm_got), np.asarray(perm_ref))
      np.testing.assert_array_equal(np.asarray(inv_got), np.asarray(inv_ref))
      np.testing.assert_array_equal(np.asarray(gs_got), np.asarray(gs_ref))

    # 2) Pallas VMEM router top-k fwd + bwd parity vs lax.top_k.
    scores_np = rng.standard_normal((512, 256)).astype(np.float32)
    g_vals_np = rng.standard_normal((512, 6)).astype(np.float32)
    scores_j = jnp.asarray(scores_np)
    g_vals_j = jnp.asarray(g_vals_np)

    def _pallas_topk_fn(s):
      res = flint_moe_kernels.router_topk_fwd_tpu(s, 6)
      assert res is not None
      vals, idx = res
      return flint_moe_kernels.attach_topk_vjp(s, vals, idx, 256), idx

    @jax.jit
    def _topk_vg(s, g):
      vals, vjp_fn, idx = jax.vjp(_pallas_topk_fn, s, has_aux=True)
      return vals, idx, vjp_fn(g)[0]

    vals_got, _, ds_got = _topk_vg(scores_j, g_vals_j)
    vals_ref, vjp_ref = jax.vjp(lambda x: jax.lax.top_k(x, 6)[0], scores_j)
    ds_ref = vjp_ref(g_vals_j)[0]
    np.testing.assert_allclose(np.asarray(vals_got), np.asarray(vals_ref), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(ds_got), np.asarray(ds_ref), rtol=1e-6, atol=1e-6)

    # 3) Dual-SC gather-reduce (`route_tokens_direct` & `weighted_combine`).
    n_tok = 2048
    tok_np = rng.standard_normal((n_tok, 256)).astype(np.float32)
    exp_np = rng.integers(0, 128, size=(n_tok * 6,), dtype=np.int32)
    w_np = rng.uniform(0.05, 0.35, size=(n_tok * 6,)).astype(np.float32)
    dout_np = rng.standard_normal((n_tok, 256)).astype(np.float32)
    tok_j = jnp.asarray(tok_np, dtype=jnp.bfloat16)
    perm_j = jnp.argsort(jnp.asarray(exp_np), stable=True)
    inv_j = jnp.argsort(perm_j)
    w_j = jnp.asarray(w_np)
    dout_j = jnp.asarray(dout_np, dtype=jnp.bfloat16)

    def _make_pipe(use_sc: bool):
      @jax.jit
      def _fn(t, w, dy):
        def _fwd(ti, wi):
          routed = flint_moe_kernels.route_tokens_direct(ti, perm_j, inv_j, 6, use_sc)
          return flint_moe_kernels.weighted_combine_direct(routed * jnp.bfloat16(1.25), perm_j, inv_j, wi, 6, use_sc)

        y, vjp_f = jax.vjp(_fwd, t, w)
        dt, dw = vjp_f(dy)
        return y, dt, dw

      return _fn

    sc_pipe = _make_pipe(True)
    ref_pipe = _make_pipe(False)
    gr_hlo = sc_pipe.lower(tok_j, w_j, dout_j).as_text("hlo", debug_info=True)
    self.assertIn("dense_gather_reduce", gr_hlo)
    y_sc, dt_sc, dw_sc = sc_pipe(tok_j, w_j, dout_j)
    y_rf, dt_rf, dw_rf = ref_pipe(tok_j, w_j, dout_j)
    self.assertGreaterEqual(_cosine_similarity(np.asarray(y_sc), np.asarray(y_rf)), 0.9999)
    self.assertGreaterEqual(_cosine_similarity(np.asarray(dt_sc), np.asarray(dt_rf)), 0.9999)
    self.assertGreaterEqual(_cosine_similarity(np.asarray(dw_sc), np.asarray(dw_rf)), 0.9999)

    # 4) End-to-end 128-expert `flint_moe_forward` with all v4 MoE kernels on.
    common_128: dict[str, Any] = dict(
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=128,
        num_experts_per_tok=6,
        fsdp_size=8,
        cp_size=1,
        ep_size=1,
        global_batch_size=8,
        seq_len=2048,
        compute_dtype="bfloat16",
        param_dtype="float32",
        moe_impl="pallas_gmm_v2",
        sc_collective_mode="dual_sc",
        routed_w_tiled=True,
        moe_bijective_permute=True,
        fuse_layer_projections=True,
        replicate_router_gate=True,
        prefetch_routed_weights=True,
        transpose_routed_down_ag=True,
        transpose_dense_rhs_ag=True,
        moe_zero_scatter_router=True,
    )
    cfg_v3 = flint_config.mini_flint_config(
        use_sc_moe_gather_reduce=False,
        use_sc_counting_sort=False,
        use_pallas_router_topk=False,
        compact_router_maps=False,
        **common_128,
    )
    cfg_v4 = flint_config.mini_flint_config(
        use_sc_moe_gather_reduce=True,
        use_sc_counting_sort=True,
        use_pallas_router_topk=True,
        compact_router_maps=True,
        **common_128,
    )
    torch_moe_128 = hf_reference.TorchFlintSparseMoeBlock(cfg_v3.model)
    hf_reference.randomize_torch_flint(torch_moe_128, seed=509)
    moe_128_np = _torch_moe_to_np(torch_moe_128)
    x_128_np = rng.standard_normal((8, 2048, 256)).astype(np.float32)
    up_128_np = rng.standard_normal((8, 2048, 256)).astype(np.float32)

    _, _, y_v3, c_v3, loss_v3, dw_v3, dx_v3 = self._eval_moe_vg(cfg_v3, moe_128_np, x_128_np, up_128_np)
    vg_v4, args_v4, y_v4, c_v4, loss_v4, dw_v4, dx_v4 = self._eval_moe_vg(cfg_v4, moe_128_np, x_128_np, up_128_np)
    hlo_v4 = vg_v4.lower(*args_v4).as_text("hlo", debug_info=True)
    self.assertIn("router_topk", hlo_v4)
    self.assertIn("sc_counting_sort", hlo_v4)
    self.assertIn("dense_gather_reduce", hlo_v4)
    np.testing.assert_array_equal(c_v4, c_v3)
    self.assertLess(abs(loss_v4 - loss_v3), 1e-3)
    self.assertGreaterEqual(_cosine_similarity(y_v4, y_v3), 0.999)
    self.assertGreaterEqual(_cosine_similarity(dx_v4, dx_v3), 0.999)
    for g_got, g_ref in zip(jax.tree.leaves(dw_v4), jax.tree.leaves(dw_v3), strict=True):
      self.assertGreaterEqual(_cosine_similarity(g_got, g_ref), 0.999)


if __name__ == "__main__":
  absltest.main()
