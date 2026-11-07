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

"""Unit tests for Qwen3-MoE gate kernel sharding and Ring of Experts parity."""

import contextlib
from typing import Any
from unittest import mock

from absl.testing import absltest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from maxtext.common.common_types import Config, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import nnx_scan
from maxtext.models import qwen3
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from tests.utils.test_helpers import get_test_config_path
import numpy as np


@contextlib.contextmanager
def _fp64_mode():
  """Context manager promoting float32 casts and dtypes in JAX to float64."""
  orig_x64 = jax.config.jax_enable_x64
  orig_dtype = jnp.dtype
  mblx_backend = qwen3.moe.mblx.ops.backend
  orig_gmm, orig_tgmm = mblx_backend.gmm, mblx_backend.tgmm
  orig_meta = mblx_backend.make_group_metadata
  jax.config.update("jax_enable_x64", True)

  def _dtype_fp64(dt: Any) -> Any:
    if dt in ("float32", np.float32, orig_dtype("float32")):
      return orig_dtype(jnp.float64)
    return orig_dtype(dt)

  def _meta_fp64(**kw: Any) -> Any:
    meta, n = orig_meta(**kw)
    return tuple(x.astype(jnp.int32) for x in meta), n.astype(jnp.int32)

  def _wrap_mblx(fn: Any) -> Any:
    def _wrapped(lhs: Any, rhs: Any, gs: Any, *a: Any, **kw: Any) -> Any:
      a = tuple(
          jnp.asarray(x, dtype=jnp.int32) if i == 2 and x is not None else x
          for i, x in enumerate(a)
      )
      return fn(lhs, rhs, gs.astype(jnp.int32), *a, **kw)

    return _wrapped

  try:
    with contextlib.ExitStack() as stack:
      stack.enter_context(mock.patch.object(jnp, "float32", jnp.float64))
      stack.enter_context(
          mock.patch.object(jnp, "dtype", side_effect=_dtype_fp64)
      )
      stack.enter_context(
          mock.patch.object(
              mblx_backend, "make_group_metadata", side_effect=_meta_fp64
          )
      )
      stack.enter_context(
          mock.patch.object(
              mblx_backend, "gmm", side_effect=_wrap_mblx(orig_gmm)
          )
      )
      stack.enter_context(
          mock.patch.object(
              mblx_backend, "tgmm", side_effect=_wrap_mblx(orig_tgmm)
          )
      )
      yield
  finally:
    jax.config.update("jax_enable_x64", orig_x64)


def _assert_leaf_parity(
    actual: jax.Array | np.ndarray,
    ref: jax.Array | np.ndarray,
    name: str = "",
    *,
    is_fp64: bool = False,
) -> None:
  """Asserts numerical parity using elementwise and floored Frobenius bounds."""
  dt = np.float64 if is_fp64 else np.float32
  a_f = np.asarray(actual, dtype=dt)
  b_f = np.asarray(ref, dtype=dt)
  diff = float(np.linalg.norm(a_f - b_f))
  ref_frob = float(np.linalg.norm(b_f))
  if ref_frob <= 1e-7:
    raise AssertionError(
        f"{name}: degenerate reference norm {ref_frob:.3e} <= 1e-7"
    )
  ref_inf = float(np.max(np.abs(b_f))) if b_f.size > 0 else 0.0

  if is_fp64:
    frob_rel = diff / max(1.0, ref_frob)
    if frob_rel > 1e-10:
      raise AssertionError(
          f"FP64 Frobenius mismatch in {name}: frob_rel={frob_rel:.3e} > 1e-10 "
          f"(diff={diff:.3e}, ref_norm={ref_frob:.3e})"
      )
    return

  atol = 1e-5 * max(1.0, ref_inf)
  np.testing.assert_allclose(a_f, b_f, atol=atol, rtol=1e-5, err_msg=name)
  floored_frob_rel = diff / max(1.0, ref_frob)
  if floored_frob_rel > 1e-5:
    raise AssertionError(
        f"{name} floored frob_rel={floored_frob_rel:.3e} exceeds tolerance 1e-5"
        f" (diff={diff:.3e}, ref_norm={ref_frob:.3e})"
    )


class Qwen3MoeTestModel(nnx.Module):
  """4-layer Qwen3 MoE model with prefix, 2 scanned blocks, and suffix."""

  def __init__(self, config: Config, mesh: Mesh, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.prefix = qwen3.Qwen3MoeDecoderLayer(
        config, mesh, MODEL_MODE_TRAIN, None, rngs
    )
    scanned = nnx_scan.create_scanned_layers(
        lambda r: qwen3.Qwen3MoeDecoderLayer(
            config, mesh, MODEL_MODE_TRAIN, None, r
        ),
        length=2,
        param_scan_axis=1,
        metadata_axis_name="layers",
        rngs=rngs,
    )
    if scanned is None:
      raise ValueError("create_scanned_layers returned None for length 2")
    self.scanned: nnx.Module = scanned
    self.suffix = qwen3.Qwen3MoeDecoderLayer(
        config, mesh, MODEL_MODE_TRAIN, None, rngs
    )

  def __call__(
      self, x: jax.Array, positions: jax.Array | None = None
  ) -> jax.Array:
    h, _ = self.prefix(
        x,
        decoder_segment_ids=None,
        decoder_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    def scan_apply(
        layer: qwen3.Qwen3MoeDecoderLayer, carry: jax.Array
    ) -> jax.Array:
      out, _ = layer(
          carry,
          decoder_segment_ids=None,
          decoder_positions=positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return out

    h = nnx_scan.apply_scanned_layers(
        self.scanned,
        h,
        length=2,
        param_scan_axis=1,
        apply_fn=scan_apply,
    )

    out, _ = self.suffix(
        h,
        decoder_segment_ids=None,
        decoder_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    return out


def _get_test_config(**kwargs: Any) -> Config:
  """Initializes a compact Qwen3-MoE configuration for unit testing."""
  argv = [None, get_test_config_path("base.yml")]
  default_kwargs = {
      "model_name": "qwen3-30b-a3b",
      "base_emb_dim": 64,
      "head_dim": 16,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 2,
      "base_mlp_dim": 128,
      "base_moe_mlp_dim": 128,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "max_target_length": 8,
      "per_device_batch_size": 1.0,
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "attention": "dot_product",
      "sparse_matmul": True,
      "megablox": True,
      "use_custom_sort_vjp": False,
      "enable_dropout": False,
      "use_ring_of_experts": False,
      "ici_expert_parallelism": 4,
      **{
          f"{p}_tile_{s}_{d}": 128 if d == "mlp_dim" else 64
          for p in ("wi", "wo")
          for s in ("fwd", "dlhs", "drhs")
          for d in ("batch_seq", "embed_dim", "mlp_dim")
      },
      "scan_layers": False,
      "load_balance_loss_weight": 0.0,
      "override_model_config": True,
  }
  default_kwargs.update(kwargs)
  return pyconfig.initialize(argv, **default_kwargs)


class Qwen3MoeShardingTest(absltest.TestCase):
  """Tests Qwen3-MoE gate kernel sharding and Ring of Experts gradient parity."""

  def _out_sharding(self, var: Any) -> Any:
    if hasattr(var, "out_sharding") and var.out_sharding is not None:
      return var.out_sharding
    return var.get_metadata()["out_sharding"]

  def test_gate_kernel_logical_axes(self):
    """Verifies moe_block.gate.kernel has logical axes ('embed_router', None)."""
    cfg = _get_test_config()
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = Qwen3MoeTestModel(cfg, mesh, nnx.Rngs(0))

    prefix_gate = model.prefix.moe_block.gate
    self.assertEqual(prefix_gate.kernel_axes, ("embed_router", None))
    self.assertEqual(
        self._out_sharding(prefix_gate.kernel), ("embed_router", None)
    )

    suffix_gate = model.suffix.moe_block.gate
    self.assertEqual(suffix_gate.kernel_axes, ("embed_router", None))
    self.assertEqual(
        self._out_sharding(suffix_gate.kernel), ("embed_router", None)
    )

    scanned_moe = getattr(model.scanned, "moe_block")
    scanned_gate = getattr(scanned_moe, "gate")
    scanned_sharding = self._out_sharding(scanned_gate.kernel)
    self.assertEqual(tuple(scanned_sharding), ("embed_router", "layers", None))

  def test_gate_kernel_sharding_fsdp(self):
    """Verifies model gate.kernel axes shard over fsdp under default and rf rules."""
    cfg_default = _get_test_config()
    mesh = Mesh(
        maxtext_utils.create_device_mesh(cfg_default), cfg_default.mesh_axes
    )
    model = Qwen3MoeTestModel(cfg_default, mesh, nnx.Rngs(0))
    prefix_axes = tuple(self._out_sharding(model.prefix.moe_block.gate.kernel))
    suffix_axes = tuple(self._out_sharding(model.suffix.moe_block.gate.kernel))
    scanned_gate = getattr(getattr(model.scanned, "moe_block"), "gate")
    scanned_axes = tuple(self._out_sharding(scanned_gate.kernel))

    expected_default_2d = PartitionSpec(
        ("fsdp", "fsdp_transpose", "context", "context_usp_ulysses"), None
    )
    self.assertEqual(
        sharding.logical_to_spec(
            prefix_axes, rules=cfg_default.logical_axis_rules
        ),
        expected_default_2d,
    )
    self.assertEqual(
        sharding.logical_to_spec(
            suffix_axes, rules=cfg_default.logical_axis_rules
        ),
        expected_default_2d,
    )
    self.assertEqual(
        sharding.logical_to_spec(
            scanned_axes, rules=cfg_default.logical_axis_rules
        ),
        PartitionSpec(
            ("fsdp", "fsdp_transpose", "context", "context_usp_ulysses"),
            "stage",
            None,
        ),
    )

    cfg_rf = _get_test_config(custom_mesh_and_rule="fsdp-as-dp-for-attn-rf")
    self.assertEqual(
        sharding.logical_to_spec(prefix_axes, rules=cfg_rf.logical_axis_rules),
        PartitionSpec("fsdp", None),
    )
    self.assertEqual(
        sharding.logical_to_spec(suffix_axes, rules=cfg_rf.logical_axis_rules),
        PartitionSpec("fsdp", None),
    )
    self.assertEqual(
        sharding.logical_to_spec(scanned_axes, rules=cfg_rf.logical_axis_rules),
        PartitionSpec("fsdp", None, None),
    )

  def _run_roe_parity(self, *, is_fp64: bool) -> None:
    expected_dtype = jnp.float64 if is_fp64 else jnp.float32
    cfg_roe = _get_test_config(use_ring_of_experts=True)
    cfg_ref = _get_test_config(use_ring_of_experts=False, sparse_matmul=False)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_roe), cfg_roe.mesh_axes)
    self.assertEqual(mesh.shape["expert"], 4)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(
        cfg_roe.logical_axis_rules
    ):
      model_roe = Qwen3MoeTestModel(cfg_roe, mesh, nnx.Rngs(42))
      params_init = nnx.state(model_roe, nnx.Param)
      model_ref = Qwen3MoeTestModel(cfg_ref, mesh, nnx.Rngs(999))
      nnx.update(model_ref, params_init)

      x = jax.random.normal(
          jax.random.PRNGKey(101), (4, 8, 64), dtype=expected_dtype
      )
      positions = jnp.broadcast_to(
          jnp.arange(8, dtype=jnp.int32)[None, :], (4, 8)
      )

      graphdef_roe, params_roe, rest_roe = nnx.split(model_roe, nnx.Param, ...)
      graphdef_ref, params_ref, rest_ref = nnx.split(model_ref, nnx.Param, ...)

      def loss_roe(p: Any, r: Any) -> tuple[jax.Array, jax.Array]:
        m = nnx.merge(graphdef_roe, p, r, copy=True)
        out = m(x, positions)
        return jnp.sum(out**2), out

      def loss_ref(p: Any, r: Any) -> tuple[jax.Array, jax.Array]:
        m = nnx.merge(graphdef_ref, p, r, copy=True)
        out = m(x, positions)
        return jnp.sum(out**2), out

      grad_roe = jax.jit(jax.value_and_grad(loss_roe, has_aux=True))
      grad_ref = jax.jit(jax.value_and_grad(loss_ref, has_aux=True))
      roe_hlo = grad_roe.lower(params_roe, rest_roe).as_text("stablehlo")
      ref_hlo = grad_ref.lower(params_ref, rest_ref).as_text("stablehlo")
      self.assertIn("reduce_scatter", roe_hlo)
      self.assertIn("all_gather", roe_hlo)
      self.assertNotIn("ragged_all_to_all", roe_hlo)
      self.assertNotIn("reduce_scatter", ref_hlo)

      (_, out_roe), grads_roe = grad_roe(params_roe, rest_roe)
      (_, out_ref), grads_ref = grad_ref(params_ref, rest_ref)

    self.assertEqual(out_roe.dtype, expected_dtype)
    self.assertEqual(out_ref.dtype, expected_dtype)
    _assert_leaf_parity(
        out_roe, out_ref, name="forward_output", is_fp64=is_fp64
    )

    flat_roe = nnx.to_flat_state(grads_roe)
    flat_ref = nnx.to_flat_state(grads_ref)
    self.assertNotEmpty(flat_roe)
    self.assertEqual(len(flat_roe), len(flat_ref))

    for (path_roe, val_roe), (path_ref, val_ref) in zip(flat_roe, flat_ref):
      self.assertEqual(path_roe, path_ref)
      v_roe = val_roe.value if hasattr(val_roe, "value") else val_roe
      v_ref = val_ref.value if hasattr(val_ref, "value") else val_ref
      self.assertEqual(v_roe.dtype, expected_dtype)
      self.assertEqual(v_ref.dtype, expected_dtype)
      _assert_leaf_parity(
          v_roe, v_ref, name=f"grad_{path_roe}", is_fp64=is_fp64
      )

  def test_ring_of_experts_parity_fp32(self):
    """Verifies RoE forward + backward execution and all-parameter leaf-gradient parity in fp32."""
    self._run_roe_parity(is_fp64=False)

  def test_ring_of_experts_parity_fp64(self):
    """Verifies RoE forward + backward execution and all-parameter leaf-gradient parity in fp64."""
    with _fp64_mode():
      self._run_roe_parity(is_fp64=True)


if __name__ == "__main__":
  absltest.main()
