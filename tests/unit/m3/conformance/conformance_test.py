# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for MaxText m3 conformance spec registry and Checks 1-8."""

from collections.abc import Mapping
import dataclasses
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.core import sharding as m3_sharding
from tests.unit.m3.conformance import checks
from tests.unit.m3.conformance import spec
import numpy as np


class _NonParamVar(nnx.Variable):
  """Synthetic non-`Param` buffer variable for testing Checks 1, 2, and 3."""


class _ToyModule(nnx.Module):
  """Synthetic m3-style module exercising Checks 1-4 failure modes."""

  def __init__(self, rngs: nnx.Rngs, scan: bool, remat: str, m: str):
    self.scan, self.remat, self.mode = scan, remat, m
    buf_len = 16 if m == "non_param_mismatch" and scan else 8
    self.bias_buf = _NonParamVar(jnp.zeros((buf_len,), jnp.float32))
    self.cache_var = _NonParamVar("idle" if m == "str_cache" else None)
    used_rngs: Any = rngs
    if m == "rngs_fallback" and used_rngs is None:
      used_rngs = nnx.Rngs(0)
    k0, k1, k2, k3 = jax.random.split(used_rngs.params(), 4)
    if m == "no_params":
      return
    k_name = "scanned_kernel_0" if m == "path_mismatch" and scan else "kernel_0"
    setattr(self, k_name, nnx.Param(jax.random.normal(k0, (8, 8))))
    self.kernel_1 = nnx.Param(jax.random.normal(k1, (8, 8)))
    frz_val = jax.random.normal(k2, (8, 4))
    if m == "int32":
      frz_val = jnp.ones((8, 4), jnp.int32)
    self.frozen_proj = nnx.Param(frz_val)
    self.gate = nnx.Param(jax.random.normal(k3, (8, 4)))
    if m in ("rng_good_aux", "rng_detached_aux"):
      self.rngs = used_rngs
    if m in ("dead_param", "nan_param"):
      fill = jnp.nan if m == "nan_param" else 1.0
      self.unused = nnx.Param(jnp.full((4,), fill, jnp.float32))
    elif m == "raw_array_weight":
      self.raw_w = jnp.ones((8,), dtype=jnp.float32)

  def __call__(self, x: jax.Array) -> spec.ForwardOutput:
    m = self.mode
    use_alt = m == "path_mismatch" and self.scan
    w0 = (self.scanned_kernel_0 if use_alt else self.kernel_0)[...]
    zero_w0 = w0 - jax.lax.stop_gradient(w0)
    is_var = self.scan or self.remat == "full"
    if m == "nan_grad" or (m == "grad_drift" and is_var):
      w0 = w0 + (jnp.sqrt(zero_w0) if m == "nan_grad" else zero_w0)

    def _step(h_in: jax.Array, weight: jax.Array) -> tuple[jax.Array, None]:
      return jnp.tanh(h_in @ weight), None

    step_fn = jax.checkpoint(_step) if self.remat == "full" else _step
    w1 = self.kernel_1[...]
    if self.scan:
      h = jax.lax.scan(step_fn, x, jnp.stack([w0, w1]))[0]
    else:
      h = step_fn(step_fn(x, w0)[0], w1)[0]
    if m == "value_drift" and is_var:
      h = h * 1.1
    f_scores = h @ self.frozen_proj[...].astype(h.dtype)
    frz_zero = 0.0 * jax.lax.stop_gradient(f_scores)
    f_term = f_scores if m == "unfrozen" else frz_zero
    h = h + jnp.mean(f_term, axis=-1, keepdims=True)
    if m in ("leaky_non_param", "raw_array_weight"):
      leak = self.bias_buf[...] + 1.0 if m == "leaky_non_param" else self.raw_w
      h = h + leak
    if m == "sow_intermediate":
      self.sowed = nnx.Intermediate(h)
    elif m in ("grow_non_param", "nan_non_param", "buf_val"):
      fill = jnp.nan if m == "nan_non_param" else 1.0
      b_val = jnp.zeros((9,)) if m == "grow_non_param" else jnp.full((8,), fill)
      self.bias_buf = _NonParamVar(b_val)
    elif m in ("cache_arr", "cache_val", "str_cache"):
      c_val = jnp.zeros((4,)) if m == "cache_arr" else "mutated"
      self.cache_var = _NonParamVar(c_val)
    scores = jax.nn.softmax(h @ self.gate[...], axis=-1)
    scale = 2.0 if (m == "aux_drift" and is_var) else 1.0
    aux_lb = jnp.mean(scores**2) * scale
    if m in ("rng_good_aux", "rng_detached_aux"):
      noise = jax.random.normal(self.rngs(), scores.shape)
      h = h + jnp.mean(scores * noise, axis=-1, keepdims=True)
    if m == "rng_detached_aux":
      aux_lb = jax.lax.stop_gradient(aux_lb)
    aux_bad = {"vec_aux": jnp.ones((2,)), "zero_aux": jnp.zeros(())}
    aux_val = aux_bad[m] if m in aux_bad else aux_lb
    if m == "int_aux":
      aux_val = jnp.ones((), jnp.int32)
    aux_out: Any = {} if m == "missing_aux" else {"lb": aux_val}
    if m == "non_map_aux":
      aux_out = None
    elif m == "extra_aux":
      aux_out["undeclared"] = aux_val
    p_map: dict[str, Any] = {"rank2": h[:, 0, :], "none": None, "tuple": (h, h)}
    return spec.ForwardOutput(p_map[m] if m in p_map else h, aux_out)


def _toy_spec(
    mode: str = "",
    *,
    kind: Any = spec.OutputKind.LOGITS,
    mut: Any = (),
) -> spec.ConformanceSpec:
  """Constructs a `ConformanceSpec` wrapping `_ToyModule` for unit tests."""
  calls = [0]

  def _forward(mod: Any, batch: Mapping[str, jax.Array]) -> spec.ForwardOutput:
    calls[0] += 1
    if mode == "bwd_cache" and calls[0] >= 2:
      mod.cache_var.set_value(jnp.ones((2,), jnp.float32))
    return mod(batch["x"])

  def _loss(
      out: spec.ForwardOutput,
      unused_b: Any,
      weights: Mapping[str, float],
  ) -> jax.Array:
    aux_w = 0.0 if mode in ("drop_aux", "aux_drift") else weights["lb"]
    return jnp.mean(out.primary**2) + aux_w * out.aux_losses["lb"]

  def _build(_: Any, rngs: nnx.Rngs, scan: bool, remat: str) -> nnx.Module:
    return _ToyModule(rngs, scan, remat, mode)

  return spec.ConformanceSpec(
      model_name="synthetic_toy",
      output_kind=kind,
      build_model=_build,
      build_batch=lambda k, _: {"x": jax.random.normal(k, (2, 4, 8))},
      forward_fn=_forward,
      loss_fn=_loss,
      frozen_param_patterns=(r"^frozen_proj$",),
      mutable_non_param_patterns=mut,
      aux_losses={"lb": (r"^gate$",)},
  )


class _ShardedExpertSubmodule(nnx.Module):
  """Synthetic MoE expert submodule for multi-device Checks 5-8."""

  def __init__(self, mesh: jax.sharding.Mesh, key: jax.Array, mode: str):
    self.mesh = jax.sharding.Mesh(np.flip(mesh.devices), mesh.axis_names) if mode == "private_sub_mesh" else mesh
    k_gu, k_dn = jax.random.split(key)
    if mode in ("4d_experts", "4d_expert_all_gather"):
      gu_shape: tuple[int, ...] = (4, 16, 2, 10)
      gu_spec = jax.sharding.PartitionSpec(None, "fsdp", None, "tensor")
    elif mode == "unsharded_expert":
      gu_shape, gu_spec = (4, 16, 20), jax.sharding.PartitionSpec()
    elif mode == "no_tensor_axis":
      gu_shape, gu_spec = (4, 16, 20), jax.sharding.PartitionSpec(None, "fsdp", None)
    else:
      gu_shape, gu_spec = (4, 16, 20), jax.sharding.PartitionSpec(None, "fsdp", "tensor")
    dn_spec = (
        jax.sharding.PartitionSpec(None, None, "fsdp")
        if mode == "no_tensor_axis"
        else jax.sharding.PartitionSpec(None, "tensor", "fsdp")
    )
    self.gate_up = nnx.Param(jax.random.normal(k_gu, gu_shape) * 0.1, out_sharding=gu_spec)
    self.down = nnx.Param(jax.random.normal(k_dn, (4, 20, 16)) * 0.1, out_sharding=dn_spec)


class _ShardedToyModule(nnx.Module):
  """Synthetic 4-device sharded scanned MoE module (`prefix + 2 scanned blocks + suffix MoE`)."""

  def __init__(
      self,
      mesh: jax.sharding.Mesh,
      rngs: nnx.Rngs,
      scan: bool,
      remat: str,
      mode: str,
  ):
    if rngs is None:
      raise ValueError("rngs must be provided")
    self.scan, self.remat, self.mode = scan, remat, mode
    if mode == "private_root_mesh":
      self.mesh = jax.sharding.Mesh(np.flip(mesh.devices), mesh.axis_names)
    elif mode != "no_root_mesh":
      self.mesh = mesh
    self.state_buf = _NonParamVar(
        jnp.zeros((8,), jnp.float32),
        out_sharding=jax.sharding.PartitionSpec(None),
    )
    self.cache_none = _NonParamVar(None)
    k0, k1, k2, k3, k4 = jax.random.split(rngs.params(), 5)
    p_init = jax.random.normal(k0, (16, 32)) * 0.1
    if mode == "unsharded_non_named":
      self.prefix_proj = nnx.Param(jax.device_put(p_init, jax.devices()[0]), eager_sharding=False)
    elif mode == "param_wrong_mesh":
      alt_mesh = jax.sharding.Mesh(np.array(mesh.devices).reshape(2, 2), ("data", "stage"))
      alt_shd = jax.sharding.NamedSharding(alt_mesh, jax.sharding.PartitionSpec("data", "stage"))
      self.prefix_proj = nnx.Param(jax.device_put(p_init, alt_shd))
    elif mode == "bad_meta_axis":
      ok_shd = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp", "tensor"))
      self.prefix_proj = nnx.Param(
          jax.device_put(p_init, ok_shd),
          out_sharding=jax.sharding.PartitionSpec("stage", "tensor"),
          eager_sharding=False,
      )
    elif mode == "unconstrained_spec":
      ok_shd = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp", "tensor"))
      unc = jax.sharding.PartitionSpec(jax.sharding.PartitionSpec.UNCONSTRAINED, None)
      self.prefix_proj = nnx.Param(jax.device_put(p_init, ok_shd), out_sharding=unc, eager_sharding=False)
    else:
      p_map = {
          "unsharded_2d": jax.sharding.PartitionSpec(),
          "size1_axis_only": jax.sharding.PartitionSpec("data", None),
          "no_tensor_axis": jax.sharding.PartitionSpec("fsdp", None),
      }
      p_spec = p_map[mode] if mode in p_map else jax.sharding.PartitionSpec("fsdp", "tensor")
      self.prefix_proj = nnx.Param(p_init, out_sharding=p_spec)
    b_axes = (None, "fsdp", None) if mode == "no_tensor_axis" else (None, "fsdp", "tensor")
    self.blocks_w = nnx.Param(
        jax.random.normal(k1, (2, 16, 16)) * 0.1,
        out_sharding=jax.sharding.PartitionSpec(*b_axes),
    )
    self.blocks_scale = nnx.Param(
        jnp.ones((2, 16), jnp.float32),
        out_sharding=jax.sharding.PartitionSpec(None, None),
    )
    self.router = nnx.Param(
        jax.random.normal(k2, (16, 4)) * 0.2,
        out_sharding=jax.sharding.PartitionSpec("fsdp", None),
    )
    self.experts = _ShardedExpertSubmodule(mesh, k3, mode)
    h_axes = ("fsdp", None) if mode == "no_tensor_axis" else ("fsdp", "tensor")
    self.head = nnx.Param(
        jax.random.normal(k4, (16, 32)) * 0.1,
        out_sharding=jax.sharding.PartitionSpec(*h_axes),
    )

  def __call__(self, x: jax.Array) -> spec.ForwardOutput:
    ft_spec = jax.sharding.PartitionSpec("fsdp", "tensor")
    act_spec = jax.sharding.PartitionSpec("fsdp", None, None)
    p_w = self.prefix_proj[...]
    if self.mode == "constrain_direct":
      p_w = jax.lax.with_sharding_constraint(p_w, ft_spec)
    elif self.mode == "reshard_direct":
      p_w = jax.sharding.reshard(p_w, jax.sharding.PartitionSpec(None, None))
    elif self.mode == "constrain_in_jit":

      @jax.jit
      def _bad_jit(w_in: jax.Array) -> jax.Array:
        return jax.lax.with_sharding_constraint(w_in.astype(jnp.float32), ft_spec)

      p_w = _bad_jit(p_w)
    elif self.mode == "constrain_in_remat":

      @jax.checkpoint
      def _bad_remat(w_in: jax.Array) -> jax.Array:
        return jax.lax.with_sharding_constraint(w_in, ft_spec)

      p_w = _bad_remat(p_w)
    elif self.mode == "constrain_in_cond":

      def _cond_true(w_in: jax.Array) -> jax.Array:
        return jax.lax.with_sharding_constraint(w_in, ft_spec)

      def _cond_false(w_in: jax.Array) -> jax.Array:
        return w_in

      p_w = jax.lax.cond(jnp.sum(x) > 0.0, _cond_true, _cond_false, p_w)
    elif self.mode == "constrain_in_while":

      def _while_cond(
          state: tuple[jax.Array, jax.Array, jax.Array],
      ) -> jax.Array:
        return state[0] < jnp.int32(2)

      def _while_body(
          state: tuple[jax.Array, jax.Array, jax.Array],
      ) -> tuple[jax.Array, jax.Array, jax.Array]:
        step_idx, w_carry, w_acc = state
        w_constrained = jax.lax.with_sharding_constraint(w_carry, ft_spec)
        return step_idx + jnp.int32(1), p_w, w_acc + w_constrained

      init_while = (jnp.int32(0), jnp.zeros_like(p_w), jnp.zeros_like(p_w))
      _, _, p_w = jax.lax.while_loop(_while_cond, _while_body, init_while)
    elif self.mode == "constrain_in_while_cond":

      def _wcond_check(state: tuple[jax.Array, jax.Array]) -> jax.Array:
        w_c = jax.lax.with_sharding_constraint(state[1], ft_spec)
        return (state[0] < jnp.int32(2)) & (jnp.sum(w_c) > -1e9)

      def _wbody_step(
          state: tuple[jax.Array, jax.Array],
      ) -> tuple[jax.Array, jax.Array]:
        return state[0] + jnp.int32(1), p_w

      _, p_w = jax.lax.while_loop(_wcond_check, _wbody_step, (jnp.int32(0), jnp.zeros_like(p_w)))
    h = jax.lax.with_sharding_constraint(jnp.tanh(x @ p_w[:, :16] + x @ p_w[:, 16:]), act_spec)

    def _block_step(h_in: jax.Array, w_b: jax.Array, s_b: jax.Array) -> jax.Array:
      if self.mode == "constrain_in_scan":
        w_b = jax.lax.with_sharding_constraint(w_b, ft_spec)
      return h_in + jnp.tanh((h_in * s_b) @ w_b)

    run_blk = jax.checkpoint(_block_step) if self.remat == "full" else _block_step
    if self.mode == "constrain_in_scan_carry":

      def _carry_scan_body(
          carry: tuple[jax.Array, jax.Array], w_slice: jax.Array
      ) -> tuple[tuple[jax.Array, jax.Array], None]:
        h_c, w_prev = carry
        w_prev_c = jax.lax.with_sharding_constraint(w_prev, ft_spec)
        h_next = h_c + jnp.tanh(h_c @ w_prev_c + h_c @ w_slice)
        return (h_next, w_slice), None

      init_scan = (h, jnp.zeros((16, 16), dtype=h.dtype))
      (h, _), _ = jax.lax.scan(_carry_scan_body, init_scan, self.blocks_w[...])
    elif self.scan:

      def _scan_body(carry: jax.Array, xs: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, None]:
        return run_blk(carry, xs[0], xs[1]), None

      h, _ = jax.lax.scan(_scan_body, h, (self.blocks_w[...], self.blocks_scale[...]))
    else:
      for b_idx in range(2):
        h = run_blk(h, self.blocks_w[...][b_idx], self.blocks_scale[...][b_idx])
    probs = jax.nn.softmax(h @ self.router[...], axis=-1)
    aux_lb = jnp.mean(probs**2)
    gu = self.experts.gate_up[...]
    gu_3d = gu.reshape(4, 16, 20) if gu.ndim == 4 else gu
    dn = self.experts.down[...]
    if self.mode in ("expert_all_gather", "4d_expert_all_gather"):

      def _replicated_dispatch(h_s: jax.Array, p_s: jax.Array, gu_rep: jax.Array, dn_rep: jax.Array) -> jax.Array:
        act_s = jax.nn.silu(jnp.einsum("bsh,ehm->bsem", h_s, gu_rep))
        return jnp.einsum("bsem,bse,emh->bsh", act_s, p_s, dn_rep)

      rep_spec = jax.sharding.PartitionSpec()
      moe_out = jax.shard_map(
          _replicated_dispatch,
          mesh=self.mesh,
          in_specs=(act_spec, act_spec, rep_spec, rep_spec),
          out_specs=act_spec,
          check_vma=False,
      )(h, probs, gu_3d, dn)
    else:
      mid = jax.nn.silu(jnp.einsum("bsh,ehm->bsem", h, gu_3d))
      mid = jax.lax.with_sharding_constraint(mid, jax.sharding.PartitionSpec("fsdp", None, None, "tensor"))
      moe_out = jax.lax.with_sharding_constraint(jnp.einsum("bsem,bse,emh->bsh", mid, probs, dn), act_spec)
    logits = (h + moe_out) @ self.head[...]
    return spec.ForwardOutput(primary=logits, aux_losses={"moe_lb": aux_lb})


def _sharded_toy_spec(mode: str = "", **overrides: Any) -> spec.ConformanceSpec:
  """Constructs a 4-device sharded scanned MoE `ConformanceSpec` for Checks 1-8."""

  def _build_model(m_arg: jax.sharding.Mesh, rngs: nnx.Rngs, scan_flag: bool, remat_pol: str) -> nnx.Module:
    return _ShardedToyModule(m_arg, rngs, scan_flag, remat_pol, mode)

  def _build_batch(b_key: jax.Array, m_arg: jax.sharding.Mesh) -> Mapping[str, jax.Array]:
    raw_x = jax.random.normal(b_key, (2, 8, 16), dtype=jnp.float32)
    x_shd = jax.sharding.NamedSharding(m_arg, jax.sharding.PartitionSpec("fsdp", None, None))
    return {"x": jax.device_put(raw_x, x_shd)}

  def _forward(mod: Any, batch: Mapping[str, jax.Array]) -> spec.ForwardOutput:
    return mod(batch["x"])

  def _loss(
      out: spec.ForwardOutput,
      unused_b: Mapping[str, jax.Array],
      weights: Mapping[str, float],
  ) -> jax.Array:
    return jnp.mean(out.primary**2) + weights["moe_lb"] * out.aux_losses["moe_lb"]

  base = spec.ConformanceSpec(
      model_name="synthetic_sharded_moe",
      output_kind=spec.OutputKind.LOGITS,
      build_model=_build_model,
      build_batch=_build_batch,
      forward_fn=_forward,
      loss_fn=_loss,
      aux_losses={"moe_lb": (r"^router$",)},
      stacked_param_path_patterns=(r"^blocks_",),
      num_scanned_blocks=2,
      min_sharded_param_elements=256,
      expert_weight_param_patterns=(r"^experts\.(gate_up|down)$",),
  )
  return dataclasses.replace(base, **overrides) if overrides else base


class ConformanceTest(absltest.TestCase):
  """Tests `spec` registry, `SkipDeclaration` guards, and Checks 1-8."""

  def setUp(self):
    super().setUp()
    self.enter_context(nnx.use_eager_sharding(True))
    self.mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))

  def test_spec_registry_idempotence_and_validation(self):
    """Verifies `spec` registry idempotence and invariant rejections."""
    s = _toy_spec()
    with mock.patch.object(spec, "_REGISTERED_SPECS", [s]) as reg:
      spec.register_conformance_spec(s)
      self.assertEqual(spec.get_registered_conformance_specs(), (s,))
      spec.register_conformance_spec(_toy_spec)
      with self.assertRaisesRegex(ValueError, "Duplicate ConformanceSpec"):
        spec.get_registered_conformance_specs()
      for bad_spec, err_pat in (
          (dataclasses.replace(s, model_name=" "), "is blank"),
          (_toy_spec(kind="bad_kind"), "invalid output_kind"),
          (_toy_spec(mut=["bias_*"]), "invalid mutable_non_param_patterns"),
          (_toy_spec(mut=("",)), "invalid mutable_non_param_patterns"),
          (_toy_spec(mut=("a b",)), "invalid mutable_non_param_patterns"),
          (
              dataclasses.replace(s, stacked_param_path_patterns=["^blocks_"]),
              "invalid stacked_param_path_patterns",
          ),
          (
              dataclasses.replace(s, stacked_param_path_patterns=("",)),
              "invalid stacked_param_path_patterns",
          ),
          (
              dataclasses.replace(s, stacked_param_path_patterns=(" ^blocks_",)),
              "invalid stacked_param_path_patterns",
          ),
          (
              dataclasses.replace(s, stacked_param_path_patterns=("(",)),
              "invalid stacked_param_path_patterns regex",
          ),
          (
              dataclasses.replace(s, expert_weight_param_patterns=(" ",)),
              "invalid expert_weight_param_patterns",
          ),
          (
              dataclasses.replace(s, expert_weight_param_patterns=("(",)),
              "invalid expert_weight_param_patterns regex",
          ),
          (
              dataclasses.replace(s, num_scanned_blocks=0),
              "num_scanned_blocks must be int >= 1",
          ),
          (
              dataclasses.replace(s, num_scanned_blocks=True),
              "num_scanned_blocks must be int >= 1",
          ),
          (
              dataclasses.replace(s, min_sharded_param_elements=0),
              "min_sharded_param_elements must be int >= 1",
          ),
      ):
        reg[:] = [bad_spec]
        with self.assertRaisesRegex(ValueError, err_pat):
          spec.get_registered_conformance_specs()

  def test_skip_declaration_validation(self):
    """Verifies `run_check_or_skip` skips and rejects empty skip reasons."""
    cid = spec.CheckId.AUX_LOSS_GRAD_COUPLING
    fn = checks.check_aux_loss_grad_coupling
    self.assertTrue(checks.run_check_or_skip(_toy_spec(), cid, fn, self.mesh))
    bad = _toy_spec("drop_aux")
    with self.assertRaisesRegex(AssertionError, "bad/zero grad delta on gate"):
      checks.run_check_or_skip(bad, cid, fn, self.mesh)
    ok_decl = spec.SkipDeclaration(spec.SkipStatus.NOT_APPLICABLE, "No router.")
    skipped = dataclasses.replace(bad, skips={cid: ok_decl})
    self.assertFalse(checks.run_check_or_skip(skipped, cid, fn, self.mesh))
    bad_decl = spec.SkipDeclaration(spec.SkipStatus.NOT_APPLICABLE, "   ")
    bad_skip = dataclasses.replace(bad, skips={cid: bad_decl})
    with self.assertRaisesRegex(AssertionError, "invalid or empty skip"):
      checks.run_check_or_skip(bad_skip, cid, fn, self.mesh)

  def test_output_kind_validation_and_embeddings(self):
    """Verifies Checks 1-4 across `OutputKind` values and invalid outputs."""
    d_kind, e_kind = spec.OutputKind.DIFFUSION, spec.OutputKind.EMBEDDINGS
    s_diff, s_emb = _toy_spec(kind=d_kind), _toy_spec("rank2", kind=e_kind)
    for active in (_toy_spec(), s_diff, s_emb):
      checks.check_scan_remat_equivalence(active, self.mesh)
      checks.check_param_and_grad_health(active, self.mesh)
      checks.check_non_param_state_stability(active, self.mesh)
      checks.check_aux_loss_grad_coupling(active, self.mesh)
    for invalid_spec, err_pat in (
        (_toy_spec(kind="bad_kind"), "unsupported output_kind 'bad_kind'"),
        (_toy_spec("rank2", kind=d_kind), r"OutputKind\.DIFFUSION \(2, 8\)"),
        (_toy_spec("none"), r"must be jax\.Array, got NoneType"),
        (_toy_spec("tuple"), r"must be jax\.Array, got tuple"),
    ):
      with self.assertRaisesRegex(AssertionError, err_pat):
        checks.check_param_and_grad_health(invalid_spec, self.mesh)

  def test_check_1_negative_detection(self):
    """Verifies Check 1 catches state/path mismatch and output/grad drift."""
    checks.check_scan_remat_equivalence(_toy_spec(), self.mesh)
    for mode, pattern in (
        ("path_mismatch", "state paths/shapes/dtypes differ"),
        ("non_param_mismatch", r"state paths/shapes/dtypes differ.*bias_buf"),
        ("value_drift", r"Elementwise mismatch.*\.primary"),
        ("aux_drift", r"Elementwise mismatch.*\.aux\.lb"),
        ("grad_drift", r"\.grad\.kernel_0"),
    ):
      with self.assertRaisesRegex(AssertionError, pattern):
        checks.check_scan_remat_equivalence(_toy_spec(mode), self.mesh)
    dup = ((False, "none"), (False, "none"))
    dup_spec = dataclasses.replace(_toy_spec(), scan_remat_variants=dup)
    with self.assertRaisesRegex(ValueError, "scan_remat_variants"):
      checks.check_scan_remat_equivalence(dup_spec, self.mesh)

  def test_check_2_negative_detection(self):
    """Verifies Check 2 catches bad Params, `rngs=None`, and grad bugs."""
    checks.check_param_and_grad_health(_toy_spec(), self.mesh)
    for mode, pattern in (
        ("rngs_fallback", "accepted rngs=None"),
        ("no_params", r"model has no nnx\.Param"),
        ("int32", r"non-inexact nnx\.Param frozen_proj dtype=int32"),
        ("dead_param", "dead:unused"),
        ("nan_param", "nan_param:unused"),
        ("nan_grad", "non_finite:kernel_0"),
        ("unfrozen", "unfrozen:frozen_proj"),
        ("leaky_non_param", "leaky_non_param:bias_buf"),
        ("raw_array_weight", "leaky_non_param:raw_w"),
        ("rank2", r"OutputKind\.LOGITS \(2, 8\)"),
    ):
      with self.assertRaisesRegex(AssertionError, pattern):
        checks.check_param_and_grad_health(_toy_spec(mode), self.mesh)
    bad_frz = dataclasses.replace(_toy_spec(), frozen_param_patterns=("^no$",))
    with self.assertRaisesRegex(AssertionError, "unmatched frozen pattern"):
      checks.check_param_and_grad_health(bad_frz, self.mesh)

  def test_check_3_negative_detection(self):
    """Verifies Check 3 allows declared mutable state and catches mutation."""
    checks.check_non_param_state_stability(_toy_spec(), self.mesh)
    checks.check_non_param_state_stability(_toy_spec("rng_good_aux"), self.mesh)
    s_mut = _toy_spec("buf_val", mut=("bias_*",))
    checks.check_non_param_state_stability(s_mut, self.mesh)
    for bad_spec, pattern in (
        (_toy_spec("grow_non_param"), "non-Param state mutated at bias_buf"),
        (_toy_spec("grow_non_param", mut=("bias_*",)), "mutated at bias_buf"),
        (_toy_spec("buf_val"), "non-Param state mutated at bias_buf"),
        (_toy_spec("sow_intermediate"), r"sowed nnx\.Intermediate"),
        (_toy_spec("nan_non_param"), r"non-finite non-Param state"),
        (_toy_spec(mut=("unmatched*",)), "unmatched mutable pattern"),
    ):
      with self.assertRaisesRegex(AssertionError, pattern):
        checks.check_non_param_state_stability(bad_spec, self.mesh)
    for mode in ("cache_arr", "cache_val", "str_cache", "bwd_cache"):
      with self.assertRaisesRegex(AssertionError, "mutated at cache_var"):
        checks.check_non_param_state_stability(_toy_spec(mode), self.mesh)

  def test_check_4_negative_detection(self):
    """Verifies Check 4 catches invalid, uncoupled, or undeclared aux loss."""
    checks.check_aux_loss_grad_coupling(_toy_spec(), self.mesh)
    checks.check_aux_loss_grad_coupling(_toy_spec("rng_good_aux"), self.mesh)
    for mode, pattern in (
        ("drop_aux", "bad/zero grad delta on gate"),
        ("rng_detached_aux", "bad/zero grad delta on gate"),
        ("missing_aux", "missing aux_loss 'lb'"),
        ("extra_aux", r"Undeclared aux_losses keys \['undeclared'\]"),
        ("non_map_aux", r"out\.aux_losses must be Mapping"),
        ("vec_aux", r"invalid/zero aux_loss 'lb'.*\[1\., 1\.\]"),
        ("zero_aux", r"invalid/zero aux_loss 'lb'"),
        ("int_aux", r"invalid/zero aux_loss 'lb'.*int32"),
    ):
      with self.assertRaisesRegex(AssertionError, pattern):
        checks.check_aux_loss_grad_coupling(_toy_spec(mode), self.mesh)
    bad_cfgs = (({}, "is empty"), ({"lb": ()}, "empty pattern"))
    for cfg, pat in (*bad_cfgs, ({"lb": ("^gate$", "^no$")}, "matched no")):
      bad_aux = dataclasses.replace(_toy_spec(), aux_losses=cfg)
      with self.assertRaisesRegex(AssertionError, pat):
        checks.check_aux_loss_grad_coupling(bad_aux, self.mesh)

  def test_all_8_checks_and_dispatcher(self):
    """Verifies `run_conformance_check` passes all 8 `CheckId` invariants (`m3/README.md` (1)-(7)) on 4 CPUs."""
    driver_mesh = checks.create_conformance_driver_mesh()
    self.assertEqual(driver_mesh.axis_names, m3_sharding.MESH_AXIS_NAMES)
    self.assertEqual(
        {**driver_mesh.shape},
        {"data": 1, "fsdp": 2, "tensor": 2, "expert": 1},
    )
    with mock.patch.object(jax, "devices", return_value=jax.devices()[:2]):
      with self.assertRaisesRegex(AssertionError, "require >= 4 JAX devices"):
        checks.create_conformance_driver_mesh()
    good_spec = _sharded_toy_spec()
    for check_id in spec.CheckId:
      self.assertTrue(checks.run_conformance_check(good_spec, check_id))
    with nnx.use_eager_sharding(False):
      for check_id in spec.CheckId:
        self.assertTrue(checks.run_conformance_check(good_spec, check_id))
    skip_id = spec.CheckId.NO_EXPERT_WEIGHT_ALL_GATHER
    ok_skip = spec.SkipDeclaration(spec.SkipStatus.NOT_APPLICABLE, "Dense model.")
    skipped = _sharded_toy_spec("unsharded_expert", skips={skip_id: ok_skip})
    self.assertFalse(checks.run_conformance_check(skipped, skip_id))
    bad_skip = spec.SkipDeclaration(spec.SkipStatus.NOT_APPLICABLE, " ")
    with self.assertRaisesRegex(AssertionError, "invalid or empty skip"):
      checks.run_conformance_check(_sharded_toy_spec(skips={skip_id: bad_skip}), skip_id)
    with self.assertRaisesRegex(ValueError, "unsupported check_id"):
      checks.run_conformance_check(good_spec, cast(Any, "not_a_check"))

  def test_check_5_negative_detection(self):
    """Verifies Check 5 (`m3/README.md` Invariants (5) & (6)) catches private Mesh, unsharded leaves, and bad axes."""
    checks.check_driver_mesh_and_axis_names(_sharded_toy_spec())
    for mode, pattern in (
        ("no_root_mesh", r"model\.mesh must be identical"),
        ("private_root_mesh", r"model\.mesh must be identical"),
        ("private_sub_mesh", r"experts.*driver_mesh"),
        ("unsharded_non_named", r"prefix_proj sharding must be NamedSharding"),
        ("param_wrong_mesh", r"prefix_proj sharding\.mesh is not driver_mesh"),
        (
            "bad_meta_axis",
            r"prefix_proj\.out_sharding: non-m3 mesh axes \['stage'\]",
        ),
        (
            "unconstrained_spec",
            r"prefix_proj\.out_sharding: invalid PartitionSpec entry",
        ),
    ):
      with self.assertRaisesRegex(AssertionError, pattern):
        checks.check_driver_mesh_and_axis_names(_sharded_toy_spec(mode))

  def test_check_6_negative_detection(self):
    """Verifies Check 6 (`m3/README.md` Invariant (7)) catches param constraints across HOPs and <4 devices."""
    good_spec = _sharded_toy_spec()
    checks.check_no_post_init_param_constraints(good_spec)
    for mode, pattern in (
        ("constrain_direct", r"prefix_proj.*sharding_constraint"),
        ("reshard_direct", r"prefix_proj.*reshard"),
        ("constrain_in_jit", r"prefix_proj.*sharding_constraint"),
        ("constrain_in_remat", r"prefix_proj.*sharding_constraint"),
        ("constrain_in_cond", r"prefix_proj.*sharding_constraint"),
        ("constrain_in_while", r"prefix_proj.*sharding_constraint"),
        ("constrain_in_while_cond", r"prefix_proj.*sharding_constraint"),
        ("constrain_in_scan", r"blocks_w.*sharding_constraint"),
        ("constrain_in_scan_carry", r"blocks_w.*sharding_constraint"),
    ):
      bad_mode_spec = _sharded_toy_spec(mode)
      with self.assertRaisesRegex(AssertionError, pattern):
        checks.check_no_post_init_param_constraints(bad_mode_spec)
    with mock.patch.object(jax, "devices", return_value=jax.devices()[:2]):
      with self.assertRaisesRegex(AssertionError, "require >= 4 JAX devices"):
        checks.check_no_post_init_param_constraints(good_spec)
    real_make_jaxpr = jax.make_jaxpr

    def _corrupt_scan_arity(fn: Any) -> Any:
      def _traced(*args: Any) -> Any:
        closed = real_make_jaxpr(fn)(*args)
        for eqn in closed.jaxpr.eqns:
          if eqn.primitive.name == "scan":
            eqn.invars = list(eqn.invars[:-1])
        return closed

      return _traced

    with mock.patch.object(jax, "make_jaxpr", side_effect=_corrupt_scan_arity):
      with self.assertRaisesRegex(AssertionError, r"scan: invars .* != ft_in"):
        checks.check_no_post_init_param_constraints(good_spec)

  def test_check_7_negative_detection(self):
    """Verifies Check 7 (`m3/README.md` Invariant (6) & comparison-doc weight sharding) catches unsharded 2-D+."""
    checks.check_weight_sharding_coverage(_sharded_toy_spec())
    for bad_spec, pattern in (
        (
            _sharded_toy_spec("unsharded_2d"),
            r"prefix_proj.*unsharded PartitionSpec",
        ),
        (
            _sharded_toy_spec("size1_axis_only"),
            r"prefix_proj.*size-1 mesh axes",
        ),
        (_sharded_toy_spec("no_tensor_axis"), r"both 'fsdp' and 'tensor'"),
        (
            _sharded_toy_spec(stacked_param_path_patterns=(r"^missing_",)),
            "unmatched stacked pattern",
        ),
        (
            _sharded_toy_spec(num_scanned_blocks=1),
            "requires num_scanned_blocks > 1",
        ),
        (
            _sharded_toy_spec(num_scanned_blocks=3),
            "leading axis != num_scanned_blocks=3",
        ),
    ):
      with self.assertRaisesRegex(AssertionError, pattern):
        checks.check_weight_sharding_coverage(bad_spec)

  def test_check_8_negative_detection(self):
    """Verifies Check 8 (comparison-doc expert `all-gather` OOM finding) catches full expert `all-gather` in HLO."""
    checks.check_no_expert_weight_all_gather(_sharded_toy_spec())
    checks.check_no_expert_weight_all_gather(_sharded_toy_spec("4d_experts"))
    for bad_spec, pattern in (
        (
            _sharded_toy_spec(expert_weight_param_patterns=()),
            "expert_weight_param_patterns is empty",
        ),
        (
            _sharded_toy_spec(expert_weight_param_patterns=(r"^experts\.none$",)),
            "unmatched expert_weight_param_patterns",
        ),
        (
            _sharded_toy_spec("unsharded_expert"),
            r"experts\.gate_up.*is unsharded",
        ),
        (
            _sharded_toy_spec("expert_all_gather"),
            r"full expert weight all-gather of experts\.",
        ),
        (
            _sharded_toy_spec("4d_expert_all_gather"),
            r"full expert weight all-gather of experts\.",
        ),
    ):
      with self.assertRaisesRegex(AssertionError, pattern):
        checks.check_no_expert_weight_all_gather(bad_spec)


if __name__ == "__main__":
  absltest.main()
