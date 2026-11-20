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

"""Unit and TPU hardware tests for gmm_v2 subtile-M, transposed dW_o^T, and wi forward RHS buffer count."""

import functools
import os
import re
import typing
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest
import qwix
import qwix.pallas as qpl

from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.kernels.megablox import ops as megablox_ops
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from tests.utils.test_helpers import get_post_train_test_config_path
from tests.utils.test_helpers import get_test_config_path


def _tpu_target(device_kind: str = "TPU7x"):
  """Context targeting an abstract TPU device for host-side AOT lowering."""
  device = jax.sharding.AbstractDevice(device_kind=device_kind, num_cores=1, platform="tpu")
  return jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), (), abstract_device=device))


def _kernel_names_and_mlir(fn, *args, device_kind: str = "TPU7x") -> tuple[list[str], str]:
  with _tpu_target(device_kind):
    mlir = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
  return re.findall(r'kernel_name = "([^"]+)"', mlir), mlir


def _init_moe_from_eval_shape(build_fn, mesh: Mesh, *, materialize_params: bool = True, seed: int = 42):
  """Constructs an NNX MoE module via nnx.eval_shape and materializes params/state without XLA init compilations."""
  abs_mod = nnx.eval_shape(build_fn)
  gdef, abs_params, abs_rest = nnx.split(abs_mod, nnx.Param, ...)
  param_shardings = nnx.get_named_sharding(abs_params, mesh)
  rest_shardings = nnx.get_named_sharding(abs_rest, mesh)
  rest = jax.tree.map(
      lambda l, s: jax.device_put(
          jnp.zeros(l.shape, l.dtype) if jnp.issubdtype(l.dtype, jnp.number) else jax.random.key(0), s
      ),
      abs_rest,
      rest_shardings,
  )
  if not materialize_params:
    return gdef, None, rest, param_shardings
  rng = np.random.default_rng(seed)
  leaves, treedef = jax.tree.flatten(abs_params)
  s_leaves = jax.tree.leaves(param_shardings)
  p_leaves = [
      jax.device_put((rng.standard_normal(l.shape, np.float32) * 0.02).astype(l.dtype), s)
      for l, s in zip(leaves, s_leaves)
  ]
  return gdef, treedef.unflatten(p_leaves), rest, param_shardings


def _same_bits(a, b) -> bool:
  a_np, b_np = np.asarray(a), np.asarray(b)
  if a_np.shape != b_np.shape or a_np.dtype != b_np.dtype:
    return False
  unsigned = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a_np.dtype.itemsize]
  return bool(np.array_equal(a_np.view(unsigned), b_np.view(unsigned)))


def _poison_tpu_memory():
  """Fills TPU VMEM/SMEM scratchpad with NaN bit patterns to catch out-of-bounds reads."""
  tpu_info = pltpu.get_tpu_info()
  vmem_size = (4 * 1024 * 1024) // 4
  smem_size = (tpu_info.smem_capacity_bytes // 4) - 8192

  def poison_kernel(in_ref, out_ref, v_scratch, s_scratch):
    del in_ref, out_ref
    v_scratch[...] = jnp.full_like(v_scratch, jnp.nan)
    for i in range(s_scratch.shape[0]):
      s_scratch[i] = 0x7FC00000

  pl.pallas_call(
      poison_kernel,
      out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
      grid=(1,),
      scratch_shapes=[
          pltpu.VMEM((vmem_size // 128, 128), jnp.float32),
          pltpu.SMEM((smem_size,), jnp.int32),
      ],
      compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
  )(jnp.zeros((1,), dtype=jnp.float32))


def _quantize_blockwise(x: jax.Array, qdtype: jnp.dtype, block_size: int = 128) -> tuple[jax.Array, jax.Array]:
  """Quantizes [G, K, N] along K (axis 1) into blocks, returning (q, [G, num_blocks, 1, N] f32 scale)."""
  if jnp.issubdtype(qdtype, jnp.integer):
    max_val = float(jnp.iinfo(qdtype).max)
    min_val = float(jnp.iinfo(qdtype).min)
  else:
    max_val = float(jnp.finfo(qdtype).max)
    min_val = float(jnp.finfo(qdtype).min)
  g, k, n = x.shape
  num_blocks = k // block_size
  x_blk = x.astype(jnp.float32).reshape(g, num_blocks, block_size, n)
  absmax = jnp.max(jnp.abs(x_blk), axis=2, keepdims=True)
  scale = jnp.where(absmax == 0, 1.0, absmax / max_val).astype(jnp.float32)
  q = jnp.clip(x_blk / scale, min_val, max_val).astype(qdtype).reshape(g, k, n)
  return q, scale


def _reference_ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    group_offset: int = 0,
    transpose_rhs: bool = False,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    partial_sum: jax.Array | None = None,
    fuse_act: str | None = None,
    out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> jax.Array:
  """Float32 reference grouped matmul with zeroing of unvisited rows when partial_sum is None."""
  w = rhs.astype(jnp.float32)
  if transpose_rhs:
    w = jnp.swapaxes(w, 1, 2)
  if rhs_scale is not None:
    g, num_blocks, _, n = rhs_scale.shape
    block_size = w.shape[1] // num_blocks
    w = (w.reshape(g, num_blocks, block_size, n) * rhs_scale.astype(jnp.float32)).reshape(g, -1, n)
  local_gs = group_sizes[group_offset : group_offset + w.shape[0]]
  x_f32 = lhs.astype(jnp.float32)
  m = x_f32.shape[0]
  n_out = w.shape[2] // 2 if fuse_act is not None else w.shape[2]
  if partial_sum is not None:
    out = partial_sum.astype(jnp.float32).copy()
  else:
    out = jnp.zeros((m, n_out), dtype=jnp.float32)
  # Prefix of lhs before group_offset:
  row_cursor = int(np.sum(np.asarray(group_sizes)[:group_offset]))
  for g_idx in range(w.shape[0]):
    cnt = int(np.asarray(local_gs)[g_idx])
    if cnt > 0:
      seg = x_f32[row_cursor : row_cursor + cnt] @ w[g_idx]
      if rhs_bias is not None:
        seg = seg + rhs_bias[g_idx].astype(jnp.float32)
      if fuse_act == "silu":
        half = seg.shape[1] // 2
        seg = jax.nn.silu(seg[:, :half]) * seg[:, half:]
      if partial_sum is not None:
        seg = seg + partial_sum[row_cursor : row_cursor + cnt].astype(jnp.float32)
      out = out.at[row_cursor : row_cursor + cnt].set(seg)
    row_cursor += cnt
  return out.astype(out_dtype)


def _gmm_v2_standin(
    lhs,
    rhs,
    group_sizes,
    rhs_scale=None,
    partial_sum=None,
    group_offset=None,
    *,
    transpose_rhs=False,
    preferred_element_type=None,
    lhs_scale=None,
    maybe_quantize_lhs=True,
    **kwargs,
):
  """CPU JAX stand-in for gmm_v2 (including FP8 rhs_scale/lhs_scale) used to test ops.gmm and RoutedMoE."""
  del kwargs
  out_dtype = preferred_element_type or lhs.dtype
  w = rhs.swapaxes(1, 2) if transpose_rhs else rhs
  off = 0 if group_offset is None else jnp.reshape(group_offset, ())
  gs = jax.lax.dynamic_slice_in_dim(group_sizes, off, w.shape[0])
  if rhs_scale is None and lhs_scale is None and lhs.dtype == w.dtype:
    out = jax.lax.ragged_dot(lhs, w, gs, preferred_element_type=jnp.float32).astype(out_dtype)
    if partial_sum is not None:
      out = out + partial_sum[: out.shape[0]]
    return out
  w_f32 = w.astype(jnp.float32)
  if rhs_scale is not None:
    g, num_blocks, _, n = rhs_scale.shape
    block_size = w_f32.shape[1] // num_blocks
    w_f32 = (w_f32.reshape(g, num_blocks, block_size, n) * rhs_scale.astype(jnp.float32)).reshape(g, -1, n)
  x_f32 = lhs.astype(jnp.float32)
  if lhs_scale is not None and maybe_quantize_lhs:
    s = lhs_scale.astype(jnp.float32)
    x_f32 = jnp.clip(x_f32 / s, -448.0, 448.0).astype(jnp.float8_e4m3fn).astype(jnp.float32) * s
  out_f32 = jax.lax.ragged_dot(x_f32, w_f32, gs, preferred_element_type=jnp.float32)
  if lhs_scale is not None and not maybe_quantize_lhs:
    out_f32 = out_f32 * lhs_scale.astype(jnp.float32)
  if partial_sum is not None:
    out_f32 = out_f32 + partial_sum[: out_f32.shape[0]].astype(jnp.float32)
  return out_f32.astype(out_dtype)


def _tgmm_v2_standin(
    lhs,
    rhs,
    group_sizes,
    num_actual_groups,
    rhs_scale=None,
    partial_sum=None,
    group_offset=None,
    *,
    transpose_out=False,
    preferred_element_type=None,
    **kwargs,
):
  """CPU JAX stand-in for tgmm_v2 (including FP8 rhs_scale and transpose_out) used to test ops.gmm and RoutedMoE."""
  del kwargs
  out_dtype = preferred_element_type or lhs.dtype
  off = 0 if group_offset is None else jnp.reshape(group_offset, ())
  gs = jax.lax.dynamic_slice_in_dim(group_sizes, off, num_actual_groups)
  dim_nums = jax.lax.RaggedDotDimensionNumbers(
      dot_dimension_numbers=(([0], [0]), ([], [])),
      lhs_ragged_dimensions=[0],
      rhs_group_dimensions=[],
  )
  if rhs_scale is None and lhs.dtype == rhs.dtype and lhs.dtype in (jnp.bfloat16, jnp.float32):
    out = jax.lax.ragged_dot_general(lhs, rhs, gs, dim_nums, preferred_element_type=jnp.float32).astype(out_dtype)
    if transpose_out:
      out = jnp.swapaxes(out, 1, 2)
    if partial_sum is not None:
      out = out + partial_sum
    return out
  out_f32 = jax.lax.ragged_dot_general(
      lhs.astype(jnp.float32), rhs.astype(jnp.float32), gs, dim_nums, preferred_element_type=jnp.float32
  )
  if rhs_scale is not None:
    out_f32 = out_f32 * rhs_scale.astype(jnp.float32)
  if transpose_out:
    out_f32 = jnp.swapaxes(out_f32, 1, 2)
  if partial_sum is not None:
    out_f32 = out_f32 + partial_sum.astype(jnp.float32)
  return out_f32.astype(out_dtype)


class ConfigValidationTest(parameterized.TestCase):
  """Tests Tier-2 MoEKernels fields and validators on MaxTextConfig and RLConfig."""

  def _init_cfg(self, **overrides):
    """Initializes a MaxTextConfig with Tokamax gmm_v2 enabled."""
    base = {
        "skip_jax_distributed_system": True,
        "model_name": "deepseek3-test",
        "sparse_matmul": True,
        "megablox": True,
        "use_tokamax_gmm": True,
        "use_gmm_v2": True,
    }
    base.update(overrides)
    return pyconfig.initialize([os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()], **base)

  def test_defaults_on_maxtext_and_rl_configs(self):
    cfg = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
    )
    self.assertFalse(cfg.enable_subtile_m)
    self.assertFalse(cfg.moe_wo_rhs_transpose)
    self.assertIsNone(cfg.wi_tile_fwd_rhs_buffer_count)

    # Verify MoEKernels is in RLConfig MRO and carries identical defaults/validators.
    self.assertIn(config_types.MoEKernels, config_types.RLConfig.__mro__)
    rl_cfg = pyconfig.initialize(
        ["maxtext.trainers.post_train.rl.train_rl", get_post_train_test_config_path("rl")],
        config_class=config_types.RLConfig,
        skip_jax_distributed_system=True,
    )
    self.assertFalse(rl_cfg.enable_subtile_m)
    self.assertFalse(rl_cfg.moe_wo_rhs_transpose)
    self.assertIsNone(rl_cfg.wi_tile_fwd_rhs_buffer_count)

  @parameterized.parameters(None, 2, 3, 4)
  def test_valid_options_accepted(self, rhs_buffers):
    cfg = self._init_cfg(
        enable_subtile_m=True,
        moe_wo_rhs_transpose=True,
        wi_tile_fwd_rhs_buffer_count=rhs_buffers,
    )
    self.assertTrue(cfg.enable_subtile_m)
    self.assertTrue(cfg.moe_wo_rhs_transpose)
    self.assertEqual(cfg.wi_tile_fwd_rhs_buffer_count, rhs_buffers)

  @parameterized.named_parameters(
      ("rhs_buf_1", {"wi_tile_fwd_rhs_buffer_count": 1}, "wi_tile_fwd_rhs_buffer_count >= 2"),
      ("rhs_buf_0", {"wi_tile_fwd_rhs_buffer_count": 0}, "wi_tile_fwd_rhs_buffer_count >= 2"),
      ("rhs_buf_neg", {"wi_tile_fwd_rhs_buffer_count": -2}, "wi_tile_fwd_rhs_buffer_count >= 2"),
      ("rhs_buf_bool_true", {"wi_tile_fwd_rhs_buffer_count": True}, "must be None or an int >= 2"),
      ("rhs_buf_bool_false", {"wi_tile_fwd_rhs_buffer_count": False}, "must be None or an int >= 2"),
      ("rhs_buf_float", {"wi_tile_fwd_rhs_buffer_count": 2.0}, "must be None or an int >= 2"),
      ("rhs_buf_no_gmm_v2", {"wi_tile_fwd_rhs_buffer_count": 2, "use_gmm_v2": False}, "use_gmm_v2=True"),
      ("subtile_no_gmm_v2", {"enable_subtile_m": True, "use_gmm_v2": False}, "use_gmm_v2=True"),
      ("subtile_no_tokamax", {"enable_subtile_m": True, "use_tokamax_gmm": False}, "use_tokamax_gmm=True"),
      ("subtile_dense_moe", {"enable_subtile_m": True, "sparse_matmul": False}, "sparse_matmul=True"),
      ("wo_transpose_no_gmm_v2", {"moe_wo_rhs_transpose": True, "use_gmm_v2": False}, "use_gmm_v2=True"),
      (
          "wo_transpose_flat_fsdp",
          {"moe_wo_rhs_transpose": True, "moe_flat_fsdp_weights": True},
          "moe_flat_fsdp_weights=False",
      ),
      (
          "wo_transpose_2d_fsdp",
          {"moe_wo_rhs_transpose": True, "use_2d_fsdp_sharding": True},
          "use_2d_fsdp_sharding=False",
      ),
      (
          "wo_transpose_two_stage_ag",
          {"moe_wo_rhs_transpose": True, "moe_fsdp_use_two_stage_all_gather": True},
          "moe_fsdp_use_two_stage_all_gather=False",
      ),
      (
          "wo_transpose_batch_split",
          {"moe_wo_rhs_transpose": True, "use_batch_split_schedule": True},
          "use_batch_split_schedule=False",
      ),
      (
          "wo_transpose_prefetch_fsdp",
          {"moe_wo_rhs_transpose": True, "prefetch_fsdp_weights": True},
          "prefetch_fsdp_weights=False",
      ),
  )
  def test_invalid_options_rejected(self, overrides, expected_msg):
    with self.assertRaisesRegex(Exception, re.escape(expected_msg)):
      self._init_cfg(**overrides)

  @parameterized.named_parameters(
      ("embed_fixed", {"shard_embed_moe_on_fsdp": True, "weight_quantization_calibration_method": "fixed,-224,224"}),
      (
          "embed_absmax_per_tensor",
          {
              "shard_embed_moe_on_fsdp": True,
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_tensor",
          },
      ),
      (
          "embed_absmax_per_channel",
          {
              "shard_embed_moe_on_fsdp": True,
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_channel",
          },
      ),
      (
          "embed_absmax_per_expert_channel",
          {
              "shard_embed_moe_on_fsdp": True,
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_expert_channel",
              "use_ring_of_experts": False,
          },
      ),
      (
          "exp_absmax_per_expert_channel",
          {
              "shard_exp_on_fsdp": True,
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_expert_channel",
              "use_ring_of_experts": False,
          },
      ),
  )
  def test_moe_wo_rhs_transpose_compatible_with_static_and_dynamic_qwag(self, qwag_overrides):
    cfg = self._init_cfg(
        quantization="fp8_full",
        use_qwix_quantization=True,
        enable_subtile_m=True,
        moe_wo_rhs_transpose=True,
        wi_tile_fwd_rhs_buffer_count=2,
        **qwag_overrides,
    )
    self.assertTrue(cfg.moe_wo_rhs_transpose)
    self.assertTrue(cfg.enable_subtile_m)
    self.assertEqual(cfg.wi_tile_fwd_rhs_buffer_count, 2)


class KernelMetadataAndLoweringTest(parameterized.TestCase):
  """Host-side config, BlockSpec, metadata, and TPU7x AOT lowering checks."""

  def _make_cfgs(self, *, enable_subtile_m: bool = False, rhs_buffer_count=None):
    """Builds GmmConfigs and returns (cfgs, weight_buffer_count) under TPU7x target."""
    lhs = typing.cast(jax.Array, jax.ShapeDtypeStruct((256, 256), jnp.bfloat16))
    rhs = typing.cast(jax.Array, jax.ShapeDtypeStruct((4, 256, 512), jnp.bfloat16))
    gs = jnp.array([64, 0, 128, 64], dtype=jnp.int32)
    off = jnp.array([0], dtype=jnp.int32)
    extra = {}
    if enable_subtile_m:
      extra["enable_subtile_m"] = True
    if rhs_buffer_count is not None:
      extra["rhs_buffer_count"] = rhs_buffer_count
    with _tpu_target():
      cfgs = gmm_v2.make_gmm_configs(
          lhs,
          rhs,
          None,
          None,
          None,
          gs,
          off,
          tile_info=gmm_v2.TileSizes(128, 256, 256),
          vmem_limit_bytes=None,
          out_dtype=jnp.dtype(jnp.bfloat16),
          acc_dtype=None,
          maybe_quantize_lhs=True,
          zero_initialize=True,
          **extra,
      )
      (_, weights_spec, _), _ = gmm_v2.generate_block_specs(
          typing.cast(typing.Any, None), cfgs, num_gm=typing.cast(jax.Array, 4)
      )
    return cfgs, weights_spec.weight.pipeline_mode.buffer_count

  def test_gmm_v2_metadata_scope_and_rhs_buffer_spec(self):
    base_cfgs, base_buf = self._make_cfgs()
    self.assertFalse(base_cfgs.enable_subtile_m)
    self.assertIsNone(base_cfgs.rhs_buffer_count)
    self.assertEqual(base_buf, 3)
    base_meta = gmm_v2.get_metadata(base_cfgs)
    self.assertNotIn("enable_subtile_m", base_meta)
    self.assertNotIn("rhs_buffer_count", base_meta)
    base_scope = gmm_v2.get_scope_name(base_cfgs)
    self.assertNotIn("subtile_m", base_scope)
    self.assertNotIn("rhsbuf", base_scope)

    opt_cfgs, opt_buf = self._make_cfgs(enable_subtile_m=True, rhs_buffer_count=2)
    self.assertTrue(opt_cfgs.enable_subtile_m)
    self.assertEqual(opt_cfgs.rhs_buffer_count, 2)
    self.assertEqual(opt_buf, 2)
    opt_meta = gmm_v2.get_metadata(opt_cfgs)
    self.assertEqual(opt_meta.pop("enable_subtile_m"), 1)
    self.assertEqual(opt_meta.pop("rhs_buffer_count"), 2)
    self.assertEqual(opt_meta, base_meta)
    self.assertEqual(gmm_v2.get_scope_name(opt_cfgs), base_scope + "-subtile_m-rhsbuf_2")

  @parameterized.parameters(1, 0, -1, True, False, 2.0)
  def test_gmm_v2_rejects_invalid_rhs_buffer_count(self, invalid_depth):
    with self.assertRaisesRegex(ValueError, "rhs_buffer_count must be None or an int >= 2"):
      self._make_cfgs(rhs_buffer_count=invalid_depth)

  def test_ops_gmm_zero_hlo_drift_at_defaults_and_lowering_flags(self):
    lhs = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    rhs = jax.ShapeDtypeStruct((4, 256, 512), jnp.bfloat16)
    rhs_t = jax.ShapeDtypeStruct((4, 512, 256), jnp.bfloat16)
    gs = jax.ShapeDtypeStruct((4,), jnp.int32)
    tiling = (128, 256, 256, 128, 256, 256, 128, 256, 256)

    def run_ops_gmm(l, r, g, **extra_kwargs):
      def f(x, w):
        return megablox_ops.gmm(
            x,
            w,
            g,
            preferred_element_type=jnp.dtype(jnp.bfloat16),
            tiling=tiling,
            use_tokamax_backend=True,
            use_gmm_v2=True,
            use_dlhs_transpose_rhs=True,
            **extra_kwargs,
        )

      out, vjp_fn = jax.vjp(f, l, r)
      dlhs, drhs = vjp_fn(jnp.ones_like(out))
      return out, dlhs, drhs

    names_implicit, mlir_implicit = _kernel_names_and_mlir(run_ops_gmm, lhs, rhs, gs)
    names_explicit_default, mlir_explicit_default = _kernel_names_and_mlir(
        functools.partial(run_ops_gmm, enable_subtile_m=False, rhs_transpose=False, fwd_rhs_buffer_count=None),
        lhs,
        rhs,
        gs,
    )
    self.assertEqual(names_implicit, names_explicit_default)
    self.assertEqual(mlir_implicit, mlir_explicit_default)

    names_sub_buf, _ = _kernel_names_and_mlir(
        functools.partial(run_ops_gmm, enable_subtile_m=True, fwd_rhs_buffer_count=2),
        lhs,
        rhs,
        gs,
    )
    self.assertLen(names_sub_buf, 3)
    # Forward kernel gets both -subtile_m and -rhsbuf_2; DLHS gets -subtile_m ONLY; TGMM gets neither.
    self.assertTrue(names_sub_buf[0].endswith("-subtile_m-rhsbuf_2"), names_sub_buf)
    self.assertTrue(names_sub_buf[1].endswith("-subtile_m") and "rhsbuf" not in names_sub_buf[1], names_sub_buf)
    self.assertTrue(names_sub_buf[2].startswith("tgmm_v2-") and "rhsbuf" not in names_sub_buf[2], names_sub_buf)

    names_rhs_t, mlir_rhs_t = _kernel_names_and_mlir(
        functools.partial(run_ops_gmm, enable_subtile_m=True, rhs_transpose=True),
        lhs,
        rhs_t,
        gs,
    )
    self.assertLen(names_rhs_t, 3)
    self.assertIn("-transpose_rhs-subtile_m", names_rhs_t[0])
    # DLHS reads rhs_t [G, N, K] directly without transpose_rhs or swapaxes!
    self.assertNotIn("transpose_rhs", names_rhs_t[1])
    self.assertIn("-subtile_m", names_rhs_t[1])
    self.assertTrue(names_rhs_t[2].startswith("tgmm_v2-") and names_rhs_t[2].endswith("-transpose_out"), names_rhs_t)
    self.assertNotIn("stablehlo.transpose", mlir_rhs_t)

  @parameterized.parameters(
      ({"enable_subtile_m": True}, "enable_subtile_m=True requires"),
      ({"rhs_transpose": True}, "rhs_transpose=True requires"),
      ({"fwd_rhs_buffer_count": 2}, "fwd_rhs_buffer_count requires"),
  )
  def test_ops_gmm_rejects_v2_only_options_without_gmm_v2(self, bad_kwarg, msg):
    lhs = jnp.zeros((16, 16), jnp.bfloat16)
    rhs = jnp.zeros((2, 16, 16), jnp.bfloat16)
    gs = jnp.array([8, 8], jnp.int32)
    with self.assertRaisesRegex(ValueError, msg):
      megablox_ops.gmm(lhs, rhs, gs, use_tokamax_backend=True, use_gmm_v2=False, **bad_kwarg)

  def test_unaligned_m_and_oversized_tile_m_lowering_shapes(self):
    """Unaligned M (m % sublane != 0) and tile_m > m lower cleanly on TPU7x target and preserve shapes."""
    lhs = jax.ShapeDtypeStruct((19, 256), jnp.bfloat16)
    rhs_t = jax.ShapeDtypeStruct((4, 384, 256), jnp.bfloat16)
    dout = jax.ShapeDtypeStruct((19, 384), jnp.bfloat16)
    gs = jax.ShapeDtypeStruct((4,), jnp.int32)
    tiling = (64, 128, 128, 64, 128, 128, 64, 128, 128)

    def fwd_bwd(x, w_t, dy, g):
      def loss(x_in, w_in):
        y = megablox_ops.gmm(
            x_in,
            w_in,
            g,
            preferred_element_type=jnp.dtype(jnp.bfloat16),
            tiling=tiling,
            use_tokamax_backend=True,
            use_gmm_v2=True,
            use_dlhs_transpose_rhs=True,
            enable_subtile_m=True,
            rhs_transpose=True,
            fwd_rhs_buffer_count=2,
        )
        return jnp.sum(y.astype(jnp.float32) * dy.astype(jnp.float32)), y

      (_, y), (dlhs, drhs_t) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(x, w_t)
      return y, dlhs, drhs_t

    with _tpu_target():
      out_shapes = jax.eval_shape(fwd_bwd, lhs, rhs_t, dout, gs)
    self.assertEqual(out_shapes[0].shape, (19, 384))
    self.assertEqual(out_shapes[1].shape, (19, 256))
    self.assertEqual(out_shapes[2].shape, (4, 384, 256))
    names, _ = _kernel_names_and_mlir(fwd_bwd, lhs, rhs_t, dout, gs)
    self.assertLen(names, 3)
    self.assertIn("-m_32-", names[0])
    self.assertIn("-tm_32-", names[0])
    self.assertTrue(names[2].endswith("-transpose_out"), names[2])

  @parameterized.parameters(
      (jnp.float8_e4m3fn, jnp.bfloat16, 128, False),
      (jnp.float8_e4m3fn, jnp.bfloat16, 128, True),
      (jnp.bfloat16, jnp.float32, 128, False),
      (jnp.bfloat16, jnp.float32, 19, True),
  )
  def test_tgmm_v2_rejects_mixed_sublane_dtypes_cleanly(self, lhs_dtype, rhs_dtype, m_rows, transpose_out):
    """Mixed-sublane tgmm_v2 operands must raise a clean AssertionError rather than crashing Mosaic."""
    lhs = jax.ShapeDtypeStruct((m_rows, 256), lhs_dtype)
    rhs = jax.ShapeDtypeStruct((m_rows, 256), rhs_dtype)
    gs = jax.ShapeDtypeStruct((4,), jnp.int32)
    with _tpu_target():
      with self.assertRaisesRegex(AssertionError, "size_lhs_sublane should be the same as size_rhs_sublane"):
        jax.eval_shape(
            functools.partial(
                tgmm_v2.tgmm_v2,
                num_actual_groups=4,
                tile_info=gmm_v2.TileSizes(128, 128, 128),
                preferred_element_type=jnp.dtype(jnp.bfloat16),
                transpose_out=transpose_out,
            ),
            lhs,
            rhs,
            gs,
        )


class RoutedMoELayerParityAndZeroDriftTest(parameterized.TestCase):
  """End-to-end RoutedMoE parity, kernel-argument routing, and zero HLO drift checks."""

  def _moe_cfg(self, num_devices: int, **overrides):
    """Initializes a compact DeepSeek3 MoE config for multi-device testing."""
    ep = 2 if num_devices >= 4 else 1
    fsdp = num_devices // ep
    base = {
        "skip_jax_distributed_system": True,
        "model_name": "deepseek3-test",
        "override_model_config": True,
        "base_emb_dim": 256,
        "base_moe_mlp_dim": 128,
        "base_mlp_dim": 256,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "per_device_batch_size": 1.0,
        "max_target_length": 64,
        "ici_expert_parallelism": ep,
        "ici_fsdp_parallelism": fsdp,
        "sparse_matmul": True,
        "megablox": True,
        "use_tokamax_gmm": True,
        "use_gmm_v2": True,
        "use_ring_of_experts": ep > 1,
        "use_ragged_sort": False,
        "wi_tile_fwd_batch_seq": 128,
        "wi_tile_fwd_embed_dim": 256,
        "wi_tile_fwd_mlp_dim": 128,
        "wi_tile_dlhs_batch_seq": 128,
        "wi_tile_dlhs_mlp_dim": 128,
        "wi_tile_dlhs_embed_dim": 256,
        "wi_tile_drhs_batch_seq": 128,
        "wi_tile_drhs_embed_dim": 256,
        "wi_tile_drhs_mlp_dim": 128,
        "wo_tile_fwd_batch_seq": 128,
        "wo_tile_fwd_mlp_dim": 128,
        "wo_tile_fwd_embed_dim": 256,
        "wo_tile_dlhs_batch_seq": 128,
        "wo_tile_dlhs_embed_dim": 256,
        "wo_tile_dlhs_mlp_dim": 128,
        "wo_tile_drhs_batch_seq": 128,
        "wo_tile_drhs_mlp_dim": 128,
        "wo_tile_drhs_embed_dim": 256,
        "dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "hardware": "tpu",
    }
    base.update(overrides)
    return pyconfig.initialize([os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()], **base)

  def _build_moe(self, cfg, mesh, *, materialize_params: bool = True):
    """Instantiates RoutedMoE via nnx.eval_shape and shards its parameters onto mesh."""
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      return _init_moe_from_eval_shape(
          lambda: moe.RoutedMoE(
              config=cfg,
              num_experts=cfg.num_experts,
              num_experts_per_tok=cfg.num_experts_per_tok,
              mesh=mesh,
              kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
              kernel_axes=("embed", None),
              intermediate_dim=cfg.base_moe_mlp_dim,
              dtype=cfg.dtype,
              weight_dtype=cfg.weight_dtype,
              quant=None,
              rngs=nnx.Rngs(params=0, dropout=1),
          ),
          mesh,
          materialize_params=materialize_params,
      )

  @parameterized.named_parameters(
      ("split_wi_buf2", {"wi_tile_fwd_rhs_buffer_count": 2, "prefuse_moe_weights": False}),
      ("prefused_wi_buf3", {"wi_tile_fwd_rhs_buffer_count": 3, "prefuse_moe_weights": True}),
      (
          "chunked_accum_wgrad_buf2",
          {
              "wi_tile_fwd_rhs_buffer_count": 2,
              "num_moe_token_chunks": 2,
              "moe_accumulate_chunk_wgrad": True,
          },
      ),
  )
  def test_routed_moe_fwd_bwd_bit_exact_and_call_routing(self, extra_overrides):
    num_dev = jax.device_count()
    cfg_ref = self._moe_cfg(
        num_dev,
        prefuse_moe_weights=extra_overrides.get("prefuse_moe_weights", False),
        num_moe_token_chunks=extra_overrides.get("num_moe_token_chunks", 1),
        moe_accumulate_chunk_wgrad=extra_overrides.get("moe_accumulate_chunk_wgrad", False),
    )
    cfg_opt = self._moe_cfg(
        num_dev,
        enable_subtile_m=True,
        moe_wo_rhs_transpose=True,
        **extra_overrides,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)
    gdef_ref, params, rest_ref, param_shardings = self._build_moe(cfg_ref, mesh)
    gdef_opt, _, rest_opt, _ = self._build_moe(cfg_opt, mesh, materialize_params=False)

    batch = max(1, num_dev)
    x = jax.random.normal(jax.random.key(11), (batch, 64, 256), dtype=jnp.bfloat16)
    cotangent = jax.random.normal(jax.random.key(12), (batch, 64, 256), dtype=jnp.bfloat16)

    def loss_fn(gdef, rest, p, inp):
      mod = nnx.merge(gdef, p, rest)
      out = mod(inp)[0]
      return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32)), out

    recorded_gmm = []
    recorded_tgmm = []

    def rec_gmm(lhs, rhs, *args, **kwargs):
      recorded_gmm.append(
          (
              rhs.shape,
              kwargs.get("enable_subtile_m", False),
              kwargs.get("transpose_rhs", False),
              kwargs.get("rhs_buffer_count", None),
          )
      )
      return _gmm_v2_standin(lhs, rhs, *args, **kwargs)

    def rec_tgmm(lhs, rhs, *args, **kwargs):
      recorded_tgmm.append(((lhs.shape[-1], rhs.shape[-1]), kwargs.get("transpose_out", False)))
      return _tgmm_v2_standin(lhs, rhs, *args, **kwargs)

    aux_shardings = (
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("data", "fsdp", "expert"), None, None)),
    )

    with (
        mock.patch.object(gmm_v2, "gmm_v2", side_effect=_gmm_v2_standin),
        mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=_tgmm_v2_standin),
        jax.set_mesh(mesh),
        nn_partitioning.axis_rules(cfg_ref.logical_axis_rules),
    ):
      (loss_ref, out_ref), grads_ref = jax.jit(
          jax.value_and_grad(lambda p: loss_fn(gdef_ref, rest_ref, p, x), has_aux=True),
          out_shardings=(aux_shardings, param_shardings),
      )(params)

    with (
        mock.patch.object(gmm_v2, "gmm_v2", side_effect=rec_gmm),
        mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=rec_tgmm),
        jax.set_mesh(mesh),
        nn_partitioning.axis_rules(cfg_opt.logical_axis_rules),
    ):
      (loss_opt, out_opt), grads_opt = jax.jit(
          jax.value_and_grad(lambda p: loss_fn(gdef_opt, rest_opt, p, x), has_aux=True),
          out_shardings=(aux_shardings, param_shardings),
      )(params)

    self.assertTrue(_same_bits(loss_opt, loss_ref))
    self.assertTrue(_same_bits(out_opt, out_ref))
    opt_leaves = jax.tree_util.tree_leaves_with_path(grads_opt)
    ref_leaves = jax.tree_util.tree_leaves_with_path(grads_ref)
    self.assertLen(opt_leaves, len(ref_leaves))
    for (path, g_opt_leaf), (_, g_ref_leaf) in zip(opt_leaves, ref_leaves):
      self.assertEqual(g_opt_leaf.sharding, g_ref_leaf.sharding, msg=f"sharding mismatch on {path}")
      self.assertTrue(_same_bits(g_opt_leaf, g_ref_leaf), msg=f"grad mismatch on {path}")

    # Every gmm_v2 call had enable_subtile_m=True.
    self.assertTrue(all(sub for _, sub, _, _ in recorded_gmm), recorded_gmm)
    # rhs_buffer_count was passed ONLY to wi forward calls (never wo, never DLHS).
    expected_buf = extra_overrides["wi_tile_fwd_rhs_buffer_count"]
    with_buf = [entry for entry in recorded_gmm if entry[3] is not None]
    self.assertNotEmpty(with_buf)
    self.assertTrue(all(entry[3] == expected_buf and entry[0][1] == 256 and not entry[2] for entry in with_buf))
    # Only wo's TGMM call had transpose_out=True.
    transposed_tgmms = [t for t in recorded_tgmm if t[1]]
    self.assertNotEmpty(transposed_tgmms)
    self.assertTrue(all(dims == (128, 256) for dims, _ in transposed_tgmms), recorded_tgmm)
    self.assertTrue(any(not is_trans for _, is_trans in recorded_tgmm), recorded_tgmm)

  def test_routed_moe_cpu_ragged_dot_fallback_parity(self):
    if jax.default_backend() != "cpu":
      self.skipTest("CPU ragged_dot fallback test only runs on CPU backend.")
    num_dev = jax.device_count()
    cfg_ref = self._moe_cfg(num_dev, hardware="cpu", megablox=False, use_tokamax_gmm=False, use_gmm_v2=False)
    cfg_opt = self._moe_cfg(num_dev, hardware="cpu", moe_wo_rhs_transpose=True)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)
    gdef_ref, params, rest_ref, _ = self._build_moe(cfg_ref, mesh)
    gdef_opt, _, rest_opt, _ = self._build_moe(cfg_opt, mesh, materialize_params=False)
    batch = max(1, num_dev)
    x = jax.random.normal(jax.random.key(21), (batch, 64, 256), dtype=jnp.bfloat16)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      loss_ref, grads_ref = jax.jit(jax.value_and_grad(lambda p: jnp.sum(nnx.merge(gdef_ref, p, rest_ref)(x)[0])))(params)
      loss_opt, grads_opt = jax.jit(jax.value_and_grad(lambda p: jnp.sum(nnx.merge(gdef_opt, p, rest_opt)(x)[0])))(params)
    np.testing.assert_allclose(float(loss_opt), float(loss_ref), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(
        np.asarray(grads_opt["wo"][...], np.float32),
        np.asarray(grads_ref["wo"][...], np.float32),
        rtol=3e-2,
        atol=3e-2,
    )

  def test_routed_moe_hlo_zero_drift_and_transposed_dwo_hlo(self):
    num_dev = jax.device_count()
    cfg_implicit = self._moe_cfg(num_dev)
    cfg_explicit = self._moe_cfg(
        num_dev,
        enable_subtile_m=False,
        moe_wo_rhs_transpose=False,
        wi_tile_fwd_rhs_buffer_count=None,
    )
    cfg_dwo = self._moe_cfg(
        num_dev,
        enable_subtile_m=True,
        moe_wo_rhs_transpose=True,
        wi_tile_fwd_rhs_buffer_count=2,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_implicit), cfg_implicit.mesh_axes)
    gdef_imp, params, rest_imp, param_shardings = self._build_moe(cfg_implicit, mesh)
    gdef_exp, _, rest_exp, _ = self._build_moe(cfg_explicit, mesh, materialize_params=False)
    gdef_dwo, _, rest_dwo, _ = self._build_moe(cfg_dwo, mesh, materialize_params=False)

    batch = max(1, num_dev)
    x = jax.random.normal(jax.random.key(31), (batch, 64, 256), dtype=jnp.bfloat16)
    loss_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def compile_moe_hlo(cfg, gdef, rest):
      with (
          mock.patch.object(gmm_v2, "gmm_v2", side_effect=_gmm_v2_standin),
          mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=_tgmm_v2_standin),
          jax.set_mesh(mesh),
          nn_partitioning.axis_rules(cfg.logical_axis_rules),
      ):

        def step(p, inp):
          mod = nnx.merge(gdef, p, rest)
          out = mod(inp)[0]
          return jnp.sum(out.astype(jnp.float32))

        lowered = jax.jit(jax.value_and_grad(step), out_shardings=(loss_sharding, param_shardings)).lower(params, x)
        return lowered.as_text()

    mlir_imp = compile_moe_hlo(cfg_implicit, gdef_imp, rest_imp)
    mlir_exp = compile_moe_hlo(cfg_explicit, gdef_exp, rest_exp)
    mlir_dwo = compile_moe_hlo(cfg_dwo, gdef_dwo, rest_dwo)

    def _strip_mod(s: str) -> str:
      return re.sub(r"module @jit_\w+", "module @jit_step", s)

    # Strip module name header to verify 100% zero StableHLO drift at default config.
    self.assertEqual(_strip_mod(mlir_imp), _strip_mod(mlir_exp))

    if num_dev >= 4:
      # In Shardy StableHLO, _gather_transposed_wo constrains [4, 256, 128] to [{"expert"}, {"fsdp"}, {}]
      # in both forward (before all-gather) and backward (reduce-scatter along axis 1 before shard transpose),
      # whereas baseline mlir_exp has zero transposed [{"expert"}, {"fsdp"}, {}] constraints on [4, 256, 128].
      pat = r'sdy\.sharding_constraint[^\n]*\[\{"expert"\}, \{"fsdp"\}, \{\}\][^\n]*tensor<4x256x128xbf16>'
      self.assertLen(re.findall(pat, mlir_dwo), 2, mlir_dwo)
      self.assertEmpty(re.findall(pat, mlir_exp), mlir_exp)


class DynamicQwagRhsTransposeTest(parameterized.TestCase):
  """Tests dynamic (absmax) and static QWAG integration with rhs_transpose=True and moe_wo_rhs_transpose=True."""

  @parameterized.named_parameters(
      ("embed_per_tensor", (4, 32, 64), P("expert", None, "fsdp"), P("expert", "fsdp", None), 1, "per_tensor"),
      ("embed_per_channel", (4, 32, 64), P("expert", None, "fsdp"), P("expert", "fsdp", None), 1, "per_channel"),
      (
          "embed_per_expert_channel",
          (4, 32, 64),
          P("expert", None, "fsdp"),
          P("expert", "fsdp", None),
          1,
          "per_expert_channel",
      ),
      ("exp_per_channel", (4, 32, 64), P("fsdp", None, None), P("fsdp", None, None), 0, "per_channel"),
      ("exp_per_expert_channel", (4, 32, 64), P("fsdp", None, None), P("fsdp", None, None), 0, "per_expert_channel"),
  )
  def test_qwag_amax_axes_and_quantize_gather_rhs_transpose(self, wo_shape, wo_spec, wo_t_spec, gather_dim, granularity):
    if jax.device_count() < 4:
      self.skipTest("Needs >= 4 devices.")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("expert", "fsdp"))
    rng = np.random.default_rng(42)
    wo_np = (rng.standard_normal(wo_shape, np.float32) * 0.05).astype(jnp.bfloat16)
    wo = jax.device_put(wo_np, NamedSharding(mesh, wo_spec))
    wo_t = jax.device_put(jnp.swapaxes(wo_np, 1, 2), NamedSharding(mesh, wo_t_spec))

    bf16_dtype = jnp.dtype(jnp.bfloat16)
    (amax_orig,) = jax.jit(lambda w: quantizations.qwag_weight_amax([w], bf16_dtype, granularity))(wo)
    (amax_direct_t,) = jax.jit(
        lambda w_t: quantizations.qwag_weight_amax([w_t], bf16_dtype, granularity, rhs_transpose=True)
    )(wo_t)
    amax_swapped_t = jnp.swapaxes(amax_orig, 1, 2)
    self.assertTrue(_same_bits(amax_direct_t, amax_swapped_t))

    amax_t_spec = moe._qwag_amax_pspec(amax_swapped_t, wo_t_spec)  # pylint: disable=protected-access
    kept_axes_t = quantizations.qwag_amax_kept_axes(granularity, amax_swapped_t.ndim, rhs_transpose=True)
    rule = qwix.QtRule(
        module_path=".*",
        weight_qtype=jnp.float8_e4m3fn,
        act_qtype=None,
        bwd_qtype=jnp.float8_e5m2,
        weight_calibration_method="absmax",
        op_names=("gmm",),
    )

    def body(shard_t, amax_t):
      lhs = jnp.zeros((8, shard_t.shape[2]), shard_t.dtype)
      _, rhs_q = megablox_ops._fwd_quantize_activation_and_weight(  # pylint: disable=protected-access
          lhs,
          shard_t,
          rule,
          use_gmm_v2=True,
          use_manual_quantization=False,
          rhs_amax=amax_t,
          rhs_transpose=True,
      )
      assert isinstance(rhs_q, qwix.QArray)
      rhs_q = megablox_ops._fwd_gather_weight(  # pylint: disable=protected-access
          rhs_q,
          [("fsdp", gather_dim)],
          scale_kept_axes=kept_axes_t,
      )
      return rhs_q.qvalue, rhs_q.scale

    out_spec = P(None if wo_t_spec[0] == "fsdp" else "expert", None, None)
    scale_out_spec = P("expert", None, None) if (0 in kept_axes_t and out_spec[0] == "expert") else P()
    qval_t, scale_t = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(wo_t_spec, amax_t_spec),
            out_specs=(out_spec, scale_out_spec),
            check_vma=False,
        )
    )(wo_t, amax_swapped_t)

    cw_axes = {"per_tensor": (), "per_channel": (2,), "per_expert_channel": (0, 2)}[granularity]
    ref_q = jax.jit(
        functools.partial(qpl.quantize, qtype=jnp.float8_e4m3fn, channelwise_axes=cw_axes, calibration_method="absmax")
    )(wo_np)
    self.assertTrue(_same_bits(qval_t, jnp.swapaxes(ref_q.qvalue, 1, 2)))
    self.assertTrue(_same_bits(scale_t, jnp.swapaxes(ref_q.scale, 1, 2)))

  def test_ops_gmm_rejects_mismatched_rhs_amax_and_kept_axes_under_rhs_transpose(self):
    rule = qwix.QtRule(
        module_path=".*",
        weight_qtype=jnp.float8_e4m3fn,
        act_qtype=jnp.float8_e4m3fn,
        bwd_qtype=jnp.float8_e5m2,
        weight_calibration_method="absmax",
        act_calibration_method="absmax",
        bwd_calibration_method="absmax",
        op_names=("gmm",),
    )
    # Square N == K == 128 where naive broadcasting would silently scale K instead of N!
    lhs = jnp.ones((64, 128), jnp.bfloat16)
    rhs_sq = jnp.ones((4, 128, 128), jnp.bfloat16)
    gs = jnp.array([16, 16, 16, 16], jnp.int32)

    with (
        mock.patch.object(gmm_v2, "gmm_v2", side_effect=_gmm_v2_standin),
        self.assertRaisesRegex(ValueError, "rhs_amax must have size 1 along contracting axis 2"),
    ):
      megablox_ops.gmm(
          lhs,
          rhs_sq,
          gs,
          use_qwix_quantization=True,
          qwix_rule=rule,
          use_tokamax_backend=True,
          use_gmm_v2=True,
          rhs_transpose=True,
          rhs_amax=jnp.ones((1, 1, 128), jnp.bfloat16),
          rhs_scale_kept_axes=(1,),
      )
    with (
        mock.patch.object(gmm_v2, "gmm_v2", side_effect=_gmm_v2_standin),
        self.assertRaisesRegex(ValueError, "rhs_amax must have size 1 along contracting axis 1"),
    ):
      megablox_ops.gmm(
          lhs,
          rhs_sq,
          gs,
          use_qwix_quantization=True,
          qwix_rule=rule,
          use_tokamax_backend=True,
          use_gmm_v2=True,
          rhs_transpose=False,
          rhs_amax=jnp.ones((1, 128, 1), jnp.bfloat16),
          rhs_scale_kept_axes=(2,),
      )
    with self.assertRaisesRegex(ValueError, "rhs_scale_kept_axes cannot keep axis 2"):
      megablox_ops.gmm(
          lhs,
          rhs_sq,
          gs,
          use_qwix_quantization=True,
          qwix_rule=rule,
          use_tokamax_backend=True,
          use_gmm_v2=True,
          rhs_transpose=True,
          rhs_amax=jnp.ones((1, 128, 1), jnp.bfloat16),
          rhs_scale_kept_axes=(2,),
      )
    with self.assertRaisesRegex(ValueError, "rhs_scale_kept_axes cannot keep contracting axis 1"):
      megablox_ops.gmm(
          lhs,
          rhs_sq,
          gs,
          use_qwix_quantization=True,
          qwix_rule=rule,
          use_tokamax_backend=True,
          use_gmm_v2=True,
          rhs_transpose=False,
          rhs_amax=jnp.ones((1, 1, 128), jnp.bfloat16),
          rhs_scale_kept_axes=(1,),
      )
    with self.assertRaisesRegex(ValueError, "rhs_transpose=True QWAG weight scales need a 3D"):
      quantizations.qwag_amax_reduce_axes("per_channel", 2, rhs_transpose=True)

  def _qwag_moe_cfg(self, num_devices: int, **overrides):
    """Initializes a compact DeepSeek3 FP8 QWAG MoE config for multi-device testing."""
    ep = overrides.pop("ici_expert_parallelism", 2 if num_devices >= 4 else 1)
    fsdp = overrides.pop("ici_fsdp_parallelism", max(1, num_devices // ep))
    base = {
        "skip_jax_distributed_system": True,
        "model_name": "deepseek3-test",
        "override_model_config": True,
        "base_emb_dim": 256,
        "base_moe_mlp_dim": 128,
        "base_mlp_dim": 256,
        "num_experts": max(4, ep * 2, fsdp if overrides.get("shard_exp_on_fsdp", False) else 4),
        "num_experts_per_tok": 2,
        "per_device_batch_size": 1.0,
        "max_target_length": 32,
        "ici_expert_parallelism": ep,
        "ici_fsdp_parallelism": fsdp,
        "sparse_matmul": True,
        "megablox": True,
        "use_tokamax_gmm": True,
        "use_gmm_v2": True,
        "use_ring_of_experts": ep > 1,
        "use_ragged_sort": False,
        "quantization": "fp8_full",
        "use_qwix_quantization": True,
        "act_quantization_calibration_method": "fixed,-224,224",
        "bwd_quantization_calibration_method": "absmax",
        "wi_tile_fwd_batch_seq": 128,
        "wi_tile_fwd_embed_dim": 256,
        "wi_tile_fwd_mlp_dim": 128,
        "wi_tile_dlhs_batch_seq": 128,
        "wi_tile_dlhs_mlp_dim": 128,
        "wi_tile_dlhs_embed_dim": 256,
        "wi_tile_drhs_batch_seq": 128,
        "wi_tile_drhs_embed_dim": 256,
        "wi_tile_drhs_mlp_dim": 128,
        "wo_tile_fwd_batch_seq": 128,
        "wo_tile_fwd_mlp_dim": 128,
        "wo_tile_fwd_embed_dim": 256,
        "wo_tile_dlhs_batch_seq": 128,
        "wo_tile_dlhs_embed_dim": 256,
        "wo_tile_dlhs_mlp_dim": 128,
        "wo_tile_drhs_batch_seq": 128,
        "wo_tile_drhs_mlp_dim": 128,
        "wo_tile_drhs_embed_dim": 256,
        "dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "hardware": "tpu",
    }
    base.update(overrides)
    return pyconfig.initialize([os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()], **base)

  def _build_quantized_moe(self, cfg, mesh, batch: int, *, materialize_params: bool = True):
    """Instantiates and Qwix-quantizes RoutedMoE on mesh via nnx.eval_shape."""
    rules = [
        qwix.QtRule(
            module_path=".*",
            weight_qtype=jnp.float8_e4m3fn,
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e5m2,
            weight_calibration_method=cfg.weight_quantization_calibration_method,
            act_calibration_method=cfg.act_quantization_calibration_method,
            bwd_calibration_method=cfg.bwd_quantization_calibration_method,
            op_names=("gmm", "ragged_dot"),
        )
    ]

    def _build():
      x0 = jax.lax.with_sharding_constraint(
          jnp.zeros((batch, cfg.max_target_length, cfg.base_emb_dim), jnp.bfloat16),
          P(("data", "fsdp", "expert"), None, None),
      )
      mod = qwix.quantize_model(
          moe.RoutedMoE(
              config=cfg,
              num_experts=cfg.num_experts,
              num_experts_per_tok=cfg.num_experts_per_tok,
              mesh=mesh,
              kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
              kernel_axes=("embed", None),
              intermediate_dim=cfg.base_moe_mlp_dim,
              dtype=cfg.dtype,
              weight_dtype=cfg.weight_dtype,
              quant=None,
              rngs=nnx.Rngs(params=0, dropout=1),
          ),
          qwix.QtProvider(rules),
          x0,
      )
      nnx.pop(mod, nnx.Intermediate)
      return mod

    return _init_moe_from_eval_shape(_build, mesh, materialize_params=materialize_params)

  @parameterized.named_parameters(
      (
          "embed_fixed",
          {"shard_embed_moe_on_fsdp": True, "weight_quantization_calibration_method": "fixed,-224,224"},
          False,
      ),
      (
          "embed_absmax_per_tensor",
          {
              "shard_embed_moe_on_fsdp": True,
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_tensor",
          },
          False,
      ),
      (
          "embed_absmax_per_channel_scanned_amax",
          {
              "shard_embed_moe_on_fsdp": True,
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_channel",
          },
          True,
      ),
      (
          "embed_absmax_per_channel_prefused",
          {
              "shard_embed_moe_on_fsdp": True,
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_channel",
              "prefuse_moe_weights": True,
          },
          True,
      ),
      (
          "embed_absmax_per_channel_chunk_accum",
          {
              "shard_embed_moe_on_fsdp": True,
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_channel",
              "num_moe_token_chunks": 2,
              "moe_accumulate_chunk_wgrad": True,
          },
          False,
      ),
      (
          "embed_absmax_per_expert_channel",
          {
              "shard_embed_moe_on_fsdp": True,
              "ici_expert_parallelism": 1,
              "use_ring_of_experts": False,
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_expert_channel",
          },
          True,
      ),
      (
          "exp_absmax_per_expert_channel",
          {
              "shard_exp_on_fsdp": True,
              "ici_expert_parallelism": 1,
              "use_ring_of_experts": False,
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_expert_channel",
          },
          False,
      ),
  )
  def test_routed_moe_qwag_rhs_transpose_bit_exact_and_major_axis_collectives(self, qwag_overrides, use_scanned_amax):
    num_dev = jax.device_count()
    if num_dev < 4:
      self.skipTest("Needs >= 4 devices.")
    cfg_ref = self._qwag_moe_cfg(num_dev, moe_wo_rhs_transpose=False, **qwag_overrides)
    cfg_opt = self._qwag_moe_cfg(
        num_dev,
        moe_wo_rhs_transpose=True,
        enable_subtile_m=True,
        wi_tile_fwd_rhs_buffer_count=2,
        **qwag_overrides,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)
    batch = max(1, num_dev)
    x = jax.random.normal(jax.random.key(91), (batch, 32, 256), dtype=jnp.bfloat16)
    dy = jax.random.normal(jax.random.key(92), (batch, 32, 256), dtype=jnp.bfloat16)

    with (
        mock.patch.object(gmm_v2, "gmm_v2", side_effect=_gmm_v2_standin),
        mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=_tgmm_v2_standin),
        jax.set_mesh(mesh),
        nn_partitioning.axis_rules(cfg_ref.logical_axis_rules),
    ):
      gdef_ref, params, rest_ref, pshard = self._build_quantized_moe(cfg_ref, mesh, batch)
      gdef_opt, _, rest_opt, _ = self._build_quantized_moe(cfg_opt, mesh, batch, materialize_params=False)

      def make_step(cfg, gdef, rest):
        def loss(p):
          amax_arg = None
          if use_scanned_amax:
            stacked = jax.tree.map(lambda v: v[None, ...], p)
            amaxes = moe.scanned_qwag_weight_amax(stacked, cfg.dtype, cfg.moe_weight_quantization_granularity)
            amax_arg = tuple(a[0] for a in amaxes)
          out = nnx.merge(gdef, p, rest)(x, moe_qwag_weight_amax=amax_arg)[0]
          return jnp.sum(out.astype(jnp.float32) * dy.astype(jnp.float32)), out

        return jax.jit(jax.value_and_grad(loss, has_aux=True), out_shardings=((NamedSharding(mesh, P()), None), pshard))

      c_ref = make_step(cfg_ref, gdef_ref, rest_ref).lower(params).compile()
      c_opt = make_step(cfg_opt, gdef_opt, rest_opt).lower(params).compile()
      (loss_ref, out_ref), grads_ref = c_ref(params)
      (loss_opt, out_opt), grads_opt = c_opt(params)

      self.assertTrue(_same_bits(loss_opt, loss_ref))
      self.assertTrue(_same_bits(out_opt, out_ref))
      pkeys = ("gate", "wi", "wo") if cfg_ref.prefuse_moe_weights else ("gate", "wi_0", "wi_1", "wo")
      for k in pkeys:
        g_opt_leaf = jax.tree_util.tree_leaves(grads_opt[k])[0]
        g_ref_leaf = jax.tree_util.tree_leaves(grads_ref[k])[0]
        self.assertEqual(g_opt_leaf.sharding, g_ref_leaf.sharding, msg=f"sharding mismatch on {k}")
        self.assertTrue(_same_bits(g_opt_leaf, g_ref_leaf), msg=f"grad mismatch on {k}")

      if cfg_ref.shard_embed_moe_on_fsdp and cfg_ref.ici_fsdp_parallelism > 1:
        hlo_ref = c_ref.as_text()
        hlo_opt = c_opt.as_text()
        minor_ag_pat = r"all-gather(?:-start)?\([^\n]*dimensions=\{2\}"
        minor_rs_pat = r"reduce-scatter(?:-start)?\([^\n]*dimensions=\{2\}"
        self.assertGreater(len(re.findall(minor_ag_pat, hlo_ref)), 0)
        self.assertEqual(len(re.findall(minor_ag_pat, hlo_opt)), 0)
        self.assertGreater(len(re.findall(minor_rs_pat, hlo_ref)), 0)
        self.assertEqual(len(re.findall(minor_rs_pat, hlo_opt)), 0)

  def test_dynamic_qwag_zero_hlo_drift_at_default_flags(self):
    num_dev = jax.device_count()
    if num_dev < 4:
      self.skipTest("Needs >= 4 devices.")
    cfg_imp = self._qwag_moe_cfg(
        num_dev,
        shard_embed_moe_on_fsdp=True,
        weight_quantization_calibration_method="absmax",
        moe_weight_quantization_granularity="per_channel",
    )
    cfg_exp = self._qwag_moe_cfg(
        num_dev,
        shard_embed_moe_on_fsdp=True,
        weight_quantization_calibration_method="absmax",
        moe_weight_quantization_granularity="per_channel",
        enable_subtile_m=False,
        moe_wo_rhs_transpose=False,
        wi_tile_fwd_rhs_buffer_count=None,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_imp), cfg_imp.mesh_axes)
    batch = max(1, num_dev)
    x = jax.random.normal(jax.random.key(95), (batch, 32, 256), dtype=jnp.bfloat16)

    def lower_mlir(cfg):
      with (
          mock.patch.object(gmm_v2, "gmm_v2", side_effect=_gmm_v2_standin),
          mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=_tgmm_v2_standin),
          jax.set_mesh(mesh),
          nn_partitioning.axis_rules(cfg.logical_axis_rules),
      ):
        gdef, params, rest, pshard = self._build_quantized_moe(cfg, mesh, batch)
        step = jax.jit(
            jax.value_and_grad(lambda p: jnp.sum(nnx.merge(gdef, p, rest)(x)[0].astype(jnp.float32))),
            out_shardings=(NamedSharding(mesh, P()), pshard),
        )
        return re.sub(r"module @jit_\w+", "module @jit_step", step.lower(params).as_text())

    self.assertEqual(lower_mlir(cfg_imp), lower_mlir(cfg_exp))


@pytest.mark.tpu_only
class SubtileMDynamicTpuTest(parameterized.TestCase):
  """Real TPU execution tests for gmm_v2 enable_subtile_m=True across skewed/edge group distributions."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Requires TPU backend.")

  @parameterized.named_parameters(
      ("skewed_all_counts_tm128_fp32", [0, 1, 7, 8, 9, 15, 16, 31, 33, 64, 129], 320, 128, jnp.float32),
      ("skewed_all_counts_tm128_bf16", [0, 1, 7, 8, 9, 15, 16, 31, 33, 64, 129], 320, 128, jnp.bfloat16),
      ("skewed_all_counts_tm512_fp32", [0, 1, 7, 8, 9, 15, 16, 31, 33, 64, 129], 520, 512, jnp.float32),
      ("skewed_all_counts_tm512_bf16", [0, 1, 7, 8, 9, 15, 16, 31, 33, 64, 129], 520, 512, jnp.bfloat16),
      ("all_zero_groups", [0, 0, 0, 0], 128, 128, jnp.float32),
      ("last_expert_unaligned_m", [0, 0, 0, 15], 19, 64, jnp.float32),
      ("tail_groups_unaligned_m", [64, 0, 7, 1], 77, 32, jnp.float32),
      ("tiny_tm16_sub8", [0, 7, 8, 9, 16], 45, 16, jnp.float32),
      ("buckets_64_256", [8, 16, 32, 64, 128, 200], 460, 256, jnp.float32),
  )
  def test_subtile_m_skewed_groups_and_accumulators(self, group_sizes_list, m_rows, tile_m, acc_dtype):
    _poison_tpu_memory()
    num_groups = len(group_sizes_list)
    k0, k1 = jax.random.split(jax.random.key(101))
    lhs = jax.random.normal(k0, (m_rows, 256), jnp.float32).astype(jnp.bfloat16)
    rhs = (jax.random.normal(k1, (num_groups, 256, 256), jnp.float32) * 0.1).astype(jnp.bfloat16)
    gs = jnp.array(group_sizes_list, dtype=jnp.int32)

    out_sub = gmm_v2.gmm_v2(
        lhs,
        rhs,
        gs,
        tile_info=gmm_v2.TileSizes(tile_m, 128, 128),
        preferred_element_type=jnp.dtype(jnp.bfloat16),
        acc_dtype=acc_dtype,
        enable_subtile_m=True,
    )
    out_nosub = gmm_v2.gmm_v2(
        lhs,
        rhs,
        gs,
        tile_info=gmm_v2.TileSizes(tile_m, 128, 128),
        preferred_element_type=jnp.dtype(jnp.bfloat16),
        acc_dtype=acc_dtype,
        enable_subtile_m=False,
    )
    ref = _reference_ragged_dot(lhs, rhs, gs, out_dtype=jnp.bfloat16)
    self.assertFalse(bool(jnp.any(jnp.isnan(out_sub))))
    atol = 3e-2 if acc_dtype == jnp.bfloat16 else 1e-2
    rtol = 3e-2 if acc_dtype == jnp.bfloat16 else 1e-2
    np.testing.assert_allclose(np.asarray(out_sub, np.float32), np.asarray(ref, np.float32), rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.asarray(out_sub, np.float32), np.asarray(out_nosub, np.float32), rtol=rtol, atol=atol)

  def test_subtile_m_group_offset_partial_k_bias_partial_sum_and_silu(self):
    _poison_tpu_memory()
    k0, k1, k2, k3 = jax.random.split(jax.random.key(202), 4)
    # 6 global groups, local shard handles 4 experts starting at group_offset=1; size_k=384 with tile_k=256 (valid_k=128).
    gs = jnp.array([19, 0, 7, 15, 65, 11], dtype=jnp.int32)
    m_rows = 140
    lhs = jax.random.normal(k0, (m_rows, 384), jnp.float32).astype(jnp.bfloat16)
    rhs = (jax.random.normal(k1, (4, 384, 256), jnp.float32) * 0.1).astype(jnp.bfloat16)
    bias = (jax.random.normal(k2, (4, 1, 256), jnp.float32) * 0.05).astype(jnp.bfloat16)
    ps = (jax.random.normal(k3, (m_rows, 256), jnp.float32) * 0.1).astype(jnp.bfloat16)
    offset = jnp.array([1], dtype=jnp.int32)

    out_sub = gmm_v2.gmm_v2(
        lhs,
        rhs,
        gs,
        rhs_bias=bias,
        partial_sum=ps,
        group_offset=offset,
        tile_info=gmm_v2.TileSizes(128, 256, 128),
        preferred_element_type=jnp.dtype(jnp.bfloat16),
        zero_initialize=False,
        enable_subtile_m=True,
    )
    out_nosub = gmm_v2.gmm_v2(
        lhs,
        rhs,
        gs,
        rhs_bias=bias,
        partial_sum=ps,
        group_offset=offset,
        tile_info=gmm_v2.TileSizes(128, 256, 128),
        preferred_element_type=jnp.dtype(jnp.bfloat16),
        zero_initialize=False,
        enable_subtile_m=False,
    )
    self.assertFalse(bool(jnp.any(jnp.isnan(out_sub))))
    np.testing.assert_allclose(np.asarray(out_sub, np.float32), np.asarray(out_nosub, np.float32), rtol=1e-2, atol=1e-2)

    # Fused SiLU with unaligned M (m_rows=140):
    rhs_fused = (jax.random.normal(k1, (4, 256, 512), jnp.float32) * 0.1).astype(jnp.bfloat16)
    lhs_256 = lhs[:, :256]
    act_sub = gmm_v2.gmm_v2(
        lhs_256,
        rhs_fused,
        gs,
        group_offset=offset,
        tile_info=gmm_v2.TileSizes(128, 256, 128),
        preferred_element_type=jnp.dtype(jnp.bfloat16),
        fuse_act="silu",
        enable_subtile_m=True,
    )
    act_ref = gmm_v2.gmm_v2(
        lhs_256,
        rhs_fused,
        gs,
        group_offset=offset,
        tile_info=gmm_v2.TileSizes(128, 256, 128),
        preferred_element_type=jnp.dtype(jnp.bfloat16),
        fuse_act="silu",
        enable_subtile_m=False,
    )
    self.assertEqual(act_sub.shape, (m_rows, 256))
    np.testing.assert_allclose(np.asarray(act_sub, np.float32), np.asarray(act_ref, np.float32), rtol=1e-2, atol=1e-2)

  @parameterized.parameters(jnp.float8_e4m3fn, jnp.int8)
  def test_subtile_m_quantized_fp8_and_int8(self, qdtype):
    _poison_tpu_memory()
    k0, k1 = jax.random.split(jax.random.key(303))
    gs = jnp.array([0, 7, 15, 33, 64], dtype=jnp.int32)
    lhs = jax.random.normal(k0, (128, 256), jnp.float32).astype(jnp.bfloat16)
    rhs_f = jax.random.normal(k1, (5, 256, 256), jnp.float32) * 0.1
    rhs_q, rhs_scale = _quantize_blockwise(rhs_f, qdtype, block_size=128)

    out_sub = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        gs,
        rhs_scale=rhs_scale,
        tile_info=gmm_v2.TileSizes(128, 128, 128),
        preferred_element_type=jnp.dtype(jnp.bfloat16),
        enable_subtile_m=True,
    )
    out_nosub = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        gs,
        rhs_scale=rhs_scale,
        tile_info=gmm_v2.TileSizes(128, 128, 128),
        preferred_element_type=jnp.dtype(jnp.bfloat16),
        enable_subtile_m=False,
    )
    self.assertFalse(bool(jnp.any(jnp.isnan(out_sub))))
    np.testing.assert_allclose(np.asarray(out_sub, np.float32), np.asarray(out_nosub, np.float32), rtol=2e-2, atol=2e-2)


@pytest.mark.tpu_only
class TransposedDwoAndRhsBufferTpuTest(parameterized.TestCase):
  """Real TPU execution tests for tgmm_v2(transpose_out=True), ops.gmm(rhs_transpose=True), and RoutedMoE."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Requires TPU backend.")

  def test_tgmm_v2_transpose_out_exact_parity_with_scale_and_partial_sum(self):
    _poison_tpu_memory()
    k0, k1, k2, k3 = jax.random.split(jax.random.key(404), 4)
    gs = jnp.array([0, 1, 7, 15, 33, 129], dtype=jnp.int32)
    m_rows = 200
    lhs = jax.random.normal(k0, (m_rows, 256), jnp.float32).astype(jnp.bfloat16)
    rhs = jax.random.normal(k1, (m_rows, 384), jnp.float32).astype(jnp.bfloat16)
    scale = jnp.abs(jax.random.normal(k2, (1, 1, 384), jnp.float32)) * 0.1 + 0.05
    ps = (jax.random.normal(k3, (6, 256, 384), jnp.float32) * 0.05).astype(jnp.bfloat16)

    for use_scale in (False, True):
      for use_ps in (False, True):
        s_arg = scale if use_scale else None
        ps_arg = ps if use_ps else None
        ps_t_arg = jnp.swapaxes(ps, 1, 2) if use_ps else None
        ref = tgmm_v2.tgmm_v2(
            lhs,
            rhs,
            gs,
            6,
            rhs_scale=s_arg,
            partial_sum=ps_arg,
            tile_info=gmm_v2.TileSizes(128, 128, 256),
            preferred_element_type=jnp.dtype(jnp.bfloat16),
            transpose_out=False,
        )
        got_t = tgmm_v2.tgmm_v2(
            lhs,
            rhs,
            gs,
            6,
            rhs_scale=s_arg,
            partial_sum=ps_t_arg,
            tile_info=gmm_v2.TileSizes(128, 128, 256),
            preferred_element_type=jnp.dtype(jnp.bfloat16),
            transpose_out=True,
        )
        self.assertEqual(got_t.shape, (6, 384, 256))
        np.testing.assert_allclose(
            np.asarray(got_t, np.float32),
            np.asarray(jnp.swapaxes(ref, 1, 2), np.float32),
            rtol=1e-2,
            atol=1e-2,
        )

  @parameterized.parameters(False, True)
  def test_ops_gmm_fwd_bwd_subtile_rhs_transpose_and_buffers_on_tpu(self, quantized):
    _poison_tpu_memory()
    k0, k1, k2 = jax.random.split(jax.random.key(505), 3)
    gs = jnp.array([0, 1, 7, 8, 15, 31, 33, 129], dtype=jnp.int32)
    lhs = jax.random.normal(k0, (256, 256), jnp.float32).astype(jnp.bfloat16)
    rhs = (jax.random.normal(k1, (8, 256, 384), jnp.float32) * 0.05).astype(jnp.bfloat16)
    rhs_t = jnp.swapaxes(rhs, 1, 2)
    dout = jax.random.normal(k2, (256, 384), jnp.float32).astype(jnp.bfloat16)
    tiling = (128, 256, 128, 128, 128, 256, 128, 256, 128)
    qrule = (
        qwix.QtRule(
            weight_qtype=jnp.float8_e4m3fn,
            weight_calibration_method="absmax",
            act_qtype=jnp.float8_e4m3fn,
            act_calibration_method="absmax",
            bwd_qtype=jnp.float8_e5m2,
            bwd_calibration_method="absmax",
        )
        if quantized
        else None
    )

    def loss_base(x, w):
      y = megablox_ops.gmm(
          x,
          w,
          gs,
          preferred_element_type=jnp.dtype(jnp.bfloat16),
          tiling=tiling,
          use_tokamax_backend=True,
          use_gmm_v2=True,
          use_qwix_quantization=quantized,
          qwix_rule=qrule,
          use_dlhs_transpose_rhs=True,
      )
      return jnp.sum(y.astype(jnp.float32) * dout.astype(jnp.float32)), y

    def loss_opt(x, w_t):
      y = megablox_ops.gmm(
          x,
          w_t,
          gs,
          preferred_element_type=jnp.dtype(jnp.bfloat16),
          tiling=tiling,
          use_tokamax_backend=True,
          use_gmm_v2=True,
          use_qwix_quantization=quantized,
          qwix_rule=qrule,
          use_dlhs_transpose_rhs=True,
          enable_subtile_m=True,
          rhs_transpose=True,
          fwd_rhs_buffer_count=2,
      )
      return jnp.sum(y.astype(jnp.float32) * dout.astype(jnp.float32)), y

    (_, y_base), (dlhs_base, drhs_base) = jax.jit(jax.value_and_grad(loss_base, argnums=(0, 1), has_aux=True))(lhs, rhs)
    (_, y_opt), (dlhs_opt, drhs_opt_t) = jax.jit(jax.value_and_grad(loss_opt, argnums=(0, 1), has_aux=True))(lhs, rhs_t)

    tol = 5e-2 if quantized else 1e-2
    np.testing.assert_allclose(np.asarray(y_opt, np.float32), np.asarray(y_base, np.float32), rtol=tol, atol=tol)
    np.testing.assert_allclose(np.asarray(dlhs_opt, np.float32), np.asarray(dlhs_base, np.float32), rtol=tol, atol=tol)
    np.testing.assert_allclose(
        np.asarray(jnp.swapaxes(drhs_opt_t, 1, 2), np.float32),
        np.asarray(drhs_base, np.float32),
        rtol=tol,
        atol=tol,
    )

  @parameterized.parameters(2, 3)
  def test_rhs_buffer_count_bit_exact_on_tpu(self, buf_count):
    k0, k1 = jax.random.split(jax.random.key(606))
    gs = jnp.array([31, 0, 129, 64], dtype=jnp.int32)
    lhs = jax.random.normal(k0, (256, 256), jnp.float32).astype(jnp.bfloat16)
    rhs = (jax.random.normal(k1, (4, 256, 512), jnp.float32) * 0.05).astype(jnp.bfloat16)
    ref = gmm_v2.gmm_v2(
        lhs,
        rhs,
        gs,
        tile_info=gmm_v2.TileSizes(128, 256, 256),
        preferred_element_type=jnp.dtype(jnp.bfloat16),
    )
    got = gmm_v2.gmm_v2(
        lhs,
        rhs,
        gs,
        tile_info=gmm_v2.TileSizes(128, 256, 256),
        preferred_element_type=jnp.dtype(jnp.bfloat16),
        rhs_buffer_count=buf_count,
    )
    self.assertTrue(_same_bits(got, ref))

  def test_tgmm_v2_rejects_mixed_sublane_on_tpu(self):
    gs = jnp.array([32, 32, 32, 32], dtype=jnp.int32)
    lhs_fp8 = jnp.zeros((128, 256), dtype=jnp.float8_e4m3fn)
    rhs_bf16 = jnp.zeros((128, 256), dtype=jnp.bfloat16)
    rhs_f32 = jnp.zeros((128, 256), dtype=jnp.float32)
    for l_arr, r_arr, t_out in ((lhs_fp8, rhs_bf16, False), (lhs_fp8, rhs_bf16, True), (rhs_bf16, rhs_f32, False)):
      with self.assertRaisesRegex(AssertionError, "size_lhs_sublane should be the same as size_rhs_sublane"):
        tgmm_v2.tgmm_v2(
            l_arr,
            r_arr,
            gs,
            4,
            tile_info=gmm_v2.TileSizes(128, 128, 128),
            preferred_element_type=jnp.dtype(jnp.bfloat16),
            transpose_out=t_out,
        )

  def test_routed_moe_multi_device_tpu_parity_and_compiled_hlo(self):
    num_dev = jax.device_count()
    ep = 2 if num_dev >= 4 else 1
    fsdp = num_dev // ep

    def make_tpu_cfg(**extra):
      kw = {
          "skip_jax_distributed_system": True,
          "model_name": "deepseek3-test",
          "override_model_config": True,
          "base_emb_dim": 2048,
          "base_moe_mlp_dim": 1024,
          "base_mlp_dim": 2048,
          "num_experts": 16,
          "num_experts_per_tok": 2,
          "per_device_batch_size": 1.0,
          "max_target_length": 128,
          "ici_expert_parallelism": ep,
          "ici_fsdp_parallelism": fsdp,
          "sparse_matmul": True,
          "megablox": True,
          "use_tokamax_gmm": True,
          "use_gmm_v2": True,
          "use_ring_of_experts": ep > 1,
          "use_ragged_sort": ep > 1,
          "wi_tile_fwd_batch_seq": 256,
          "wi_tile_fwd_embed_dim": 2048,
          "wi_tile_fwd_mlp_dim": 1024,
          "wi_tile_dlhs_batch_seq": 256,
          "wi_tile_dlhs_mlp_dim": 1024,
          "wi_tile_dlhs_embed_dim": 2048,
          "wi_tile_drhs_batch_seq": 256,
          "wi_tile_drhs_embed_dim": 2048,
          "wi_tile_drhs_mlp_dim": 1024,
          "wo_tile_fwd_batch_seq": 256,
          "wo_tile_fwd_mlp_dim": 1024,
          "wo_tile_fwd_embed_dim": 2048,
          "wo_tile_dlhs_batch_seq": 256,
          "wo_tile_dlhs_embed_dim": 2048,
          "wo_tile_dlhs_mlp_dim": 1024,
          "wo_tile_drhs_batch_seq": 256,
          "wo_tile_drhs_mlp_dim": 1024,
          "wo_tile_drhs_embed_dim": 2048,
          "dtype": "bfloat16",
          "weight_dtype": "bfloat16",
          "hardware": "tpu",
      }
      kw.update(extra)
      return pyconfig.initialize([os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()], **kw)

    cfg_ref = make_tpu_cfg()
    cfg_opt2 = make_tpu_cfg(enable_subtile_m=True, moe_wo_rhs_transpose=True, wi_tile_fwd_rhs_buffer_count=2)
    cfg_opt3 = make_tpu_cfg(enable_subtile_m=True, moe_wo_rhs_transpose=True, wi_tile_fwd_rhs_buffer_count=3)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)

    def _build_bf16(c, materialize: bool):
      return _init_moe_from_eval_shape(
          lambda: moe.RoutedMoE(
              config=c,
              num_experts=c.num_experts,
              num_experts_per_tok=c.num_experts_per_tok,
              mesh=mesh,
              kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
              kernel_axes=("embed", None),
              intermediate_dim=c.base_moe_mlp_dim,
              dtype=c.dtype,
              weight_dtype=c.weight_dtype,
              quant=None,
              rngs=nnx.Rngs(params=0, dropout=1),
          ),
          mesh,
          materialize_params=materialize,
      )

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      gdef_ref, params, rest_ref, param_sharding = _build_bf16(cfg_ref, True)
      gdef_opt2, _, rest_opt2, _ = _build_bf16(cfg_opt2, False)
      gdef_opt3, _, rest_opt3, _ = _build_bf16(cfg_opt3, False)

    rng = np.random.default_rng(707)
    x_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("data", "fsdp", "expert"), None, None))
    x = jax.device_put(rng.standard_normal((num_dev, 128, 2048), np.float32).astype(jnp.bfloat16), x_sharding)
    dy = jax.device_put(rng.standard_normal((num_dev, 128, 2048), np.float32).astype(jnp.bfloat16), x_sharding)
    aux_sharding = (jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), x_sharding)

    def make_step(gdef, rest):
      def step(p, inp, grad_out):
        def loss(weights):
          mod = nnx.merge(gdef, weights, rest)
          out = mod(inp)[0]
          return jnp.sum(out.astype(jnp.float32) * grad_out.astype(jnp.float32)), out

        return jax.value_and_grad(loss, has_aux=True)(p)

      return jax.jit(step, out_shardings=(aux_sharding, param_sharding))

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      c_ref = make_step(gdef_ref, rest_ref).lower(params, x, dy).compile()
      c_opt2 = make_step(gdef_opt2, rest_opt2).lower(params, x, dy).compile()
      c_opt3 = make_step(gdef_opt3, rest_opt3).lower(params, x, dy).compile()
      (_, out_ref), grads_ref = c_ref(params, x, dy)
      (_, out_opt2), grads_opt2 = c_opt2(params, x, dy)
      (_, out_opt3), grads_opt3 = c_opt3(params, x, dy)

      if fsdp > 1:
        hlo_ref = c_ref.as_text()
        hlo_opt2 = c_opt2.as_text()
        # Match actual full-shape HLO transpose / physical-relayout copy opcodes
        # (`= bf16[E_local, ...]{layout} (transpose|copy)(` — XLA canonicalizes 3D layout
        # swaps to physical-layout `copy(` + `bitcast(`), never metadata strings.
        e_local = cfg_ref.num_experts // ep
        full_wo_relayout_pat = rf"=\s*bf16\[{e_local},(?:1024,2048|2048,1024)\]\S*\s+(?:transpose|copy)\("
        ref_count = len(re.findall(full_wo_relayout_pat, hlo_ref))
        opt2_count = len(re.findall(full_wo_relayout_pat, hlo_opt2))
        self.assertGreater(ref_count, 0)
        self.assertLess(opt2_count, ref_count)
        self.assertEqual(opt2_count, 0)
        minor_rs_pat = r"reduce-scatter(?:-start)?\([^\n]*dimensions=\{2\}"
        self.assertGreater(len(re.findall(minor_rs_pat, hlo_ref)), 0)
        self.assertEqual(len(re.findall(minor_rs_pat, hlo_opt2)), 0)
        self.assertNotIn("-transpose_out", hlo_ref)
        self.assertIn("-transpose_out", hlo_opt2)

    np.testing.assert_allclose(np.asarray(out_opt2, np.float32), np.asarray(out_ref, np.float32), rtol=2e-2, atol=2e-2)
    self.assertTrue(_same_bits(out_opt3, out_opt2))
    for k in ("gate", "wi_0", "wi_1", "wo"):
      g_opt2_leaf = jax.tree_util.tree_leaves(grads_opt2[k])[0]
      g_opt3_leaf = jax.tree_util.tree_leaves(grads_opt3[k])[0]
      g_ref_leaf = jax.tree_util.tree_leaves(grads_ref[k])[0]
      np.testing.assert_allclose(
          np.asarray(g_opt2_leaf, np.float32),
          np.asarray(g_ref_leaf, np.float32),
          rtol=2e-2,
          atol=2e-2,
      )
      self.assertTrue(_same_bits(g_opt3_leaf, g_opt2_leaf))

  @parameterized.named_parameters(
      (
          "embed_fixed_roe",
          {"shard_embed_moe_on_fsdp": True, "weight_quantization_calibration_method": "fixed,-224,224"},
          False,
      ),
      (
          "embed_absmax_per_tensor_roe",
          {
              "shard_embed_moe_on_fsdp": True,
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_tensor",
          },
          False,
      ),
      (
          "embed_absmax_per_channel_prefused",
          {
              "shard_embed_moe_on_fsdp": True,
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_channel",
              "prefuse_moe_weights": True,
          },
          True,
      ),
      (
          "embed_absmax_per_channel_scanned_chunk",
          {
              "shard_embed_moe_on_fsdp": True,
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_channel",
              "num_moe_token_chunks": 2,
              "moe_accumulate_chunk_wgrad": True,
          },
          True,
      ),
      (
          "embed_absmax_per_expert_channel_qra2a",
          {
              "shard_embed_moe_on_fsdp": True,
              "use_ring_of_experts": False,
              "use_ragged_sort": True,
              "moe_quantize_dispatch_fwd_method": "rowwise",
              "moe_quantize_combine_fwd_method": "rowwise",
              "moe_quantize_combine_bwd_method": "rowwise",
              "moe_quantize_dispatch_bwd_method": "rowwise",
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_expert_channel",
          },
          True,
      ),
      (
          "exp_absmax_per_expert_channel",
          {
              "shard_exp_on_fsdp": True,
              "ici_expert_parallelism": 1,
              "use_ring_of_experts": False,
              "use_ragged_sort": False,
              "weight_quantization_calibration_method": "absmax",
              "moe_weight_quantization_granularity": "per_expert_channel",
          },
          False,
      ),
  )
  def test_routed_moe_qwag_dynamic_and_static_tpu_parity_and_hlo(self, qwag_overrides, use_scanned_amax):
    num_dev = jax.device_count()
    if num_dev < 4:
      self.skipTest("Requires >= 4 TPU devices.")
    ep = qwag_overrides.get("ici_expert_parallelism", 2)
    fsdp = num_dev // ep

    def make_qwag_tpu_cfg(**extra):
      kw = {
          "skip_jax_distributed_system": True,
          "model_name": "deepseek3-test",
          "override_model_config": True,
          "base_emb_dim": 2048,
          "base_moe_mlp_dim": 1024,
          "base_mlp_dim": 2048,
          "num_experts": 16,
          "num_experts_per_tok": 2,
          "per_device_batch_size": 1.0,
          "max_target_length": 128,
          "ici_expert_parallelism": ep,
          "ici_fsdp_parallelism": fsdp,
          "sparse_matmul": True,
          "megablox": True,
          "use_tokamax_gmm": True,
          "use_gmm_v2": True,
          "use_ring_of_experts": ep > 1,
          "use_ragged_sort": ep > 1,
          "quantization": "fp8_full",
          "use_qwix_quantization": True,
          "act_quantization_calibration_method": "fixed,-224,224",
          "bwd_quantization_calibration_method": "absmax",
          "wi_tile_fwd_batch_seq": 256,
          "wi_tile_fwd_embed_dim": 2048,
          "wi_tile_fwd_mlp_dim": 1024,
          "wi_tile_dlhs_batch_seq": 256,
          "wi_tile_dlhs_mlp_dim": 1024,
          "wi_tile_dlhs_embed_dim": 2048,
          "wi_tile_drhs_batch_seq": 256,
          "wi_tile_drhs_embed_dim": 2048,
          "wi_tile_drhs_mlp_dim": 1024,
          "wo_tile_fwd_batch_seq": 256,
          "wo_tile_fwd_mlp_dim": 1024,
          "wo_tile_fwd_embed_dim": 2048,
          "wo_tile_dlhs_batch_seq": 256,
          "wo_tile_dlhs_embed_dim": 2048,
          "wo_tile_dlhs_mlp_dim": 1024,
          "wo_tile_drhs_batch_seq": 256,
          "wo_tile_drhs_mlp_dim": 1024,
          "wo_tile_drhs_embed_dim": 2048,
          "dtype": "bfloat16",
          "weight_dtype": "bfloat16",
          "hardware": "tpu",
      }
      kw.update(qwag_overrides)
      kw.update(extra)
      return pyconfig.initialize([os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()], **kw)

    cfg_ref = make_qwag_tpu_cfg(moe_wo_rhs_transpose=False)
    cfg_opt = make_qwag_tpu_cfg(
        enable_subtile_m=True,
        moe_wo_rhs_transpose=True,
        wi_tile_fwd_rhs_buffer_count=2,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)
    rules = [
        qwix.QtRule(
            module_path=".*",
            weight_qtype=jnp.float8_e4m3fn,
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e5m2,
            weight_calibration_method=cfg_ref.weight_quantization_calibration_method,
            act_calibration_method=cfg_ref.act_quantization_calibration_method,
            bwd_calibration_method=cfg_ref.bwd_quantization_calibration_method,
            op_names=("gmm", "ragged_dot"),
        )
    ]
    x_sharding = NamedSharding(mesh, P(("data", "fsdp", "expert"), None, None))

    def _build_qwag(c, materialize: bool):
      def _fn():
        x0 = jax.lax.with_sharding_constraint(
            jnp.zeros((num_dev, 128, 2048), dtype=jnp.bfloat16),
            P(("data", "fsdp", "expert"), None, None),
        )
        m = qwix.quantize_model(
            moe.RoutedMoE(
                config=c,
                num_experts=c.num_experts,
                num_experts_per_tok=c.num_experts_per_tok,
                mesh=mesh,
                kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
                kernel_axes=("embed", None),
                intermediate_dim=c.base_moe_mlp_dim,
                dtype=c.dtype,
                weight_dtype=c.weight_dtype,
                quant=None,
                rngs=nnx.Rngs(params=0, dropout=1),
            ),
            qwix.QtProvider(rules),
            x0,
        )
        nnx.pop(m, nnx.Intermediate)
        return m

      return _init_moe_from_eval_shape(_fn, mesh, materialize_params=materialize)

    rng = np.random.default_rng(808)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      gdef_ref, params, rest_ref, param_sharding = _build_qwag(cfg_ref, True)
      gdef_opt, _, rest_opt, _ = _build_qwag(cfg_opt, False)

      x = jax.device_put(rng.standard_normal((num_dev, 128, 2048), np.float32).astype(jnp.bfloat16), x_sharding)
      dy = jax.device_put(rng.standard_normal((num_dev, 128, 2048), np.float32).astype(jnp.bfloat16), x_sharding)
      aux_sharding = (NamedSharding(mesh, P()), x_sharding)

      def make_step(cfg, gdef, rest):
        def step(p, inp, grad_out):
          def loss(weights, x_in):
            amax_arg = None
            if use_scanned_amax:
              stacked = jax.tree.map(lambda v: v[None, ...], weights)
              amaxes = moe.scanned_qwag_weight_amax(stacked, cfg.dtype, cfg.moe_weight_quantization_granularity)
              amax_arg = tuple(a[0] for a in amaxes)
            mod = nnx.merge(gdef, weights, rest)
            out = mod(x_in, moe_qwag_weight_amax=amax_arg)[0]
            return jnp.sum(out.astype(jnp.float32) * grad_out.astype(jnp.float32)), out

          return jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(p, inp)

        return jax.jit(step, out_shardings=(aux_sharding, (param_sharding, x_sharding)))

      c_ref = make_step(cfg_ref, gdef_ref, rest_ref).lower(params, x, dy).compile()
      c_opt = make_step(cfg_opt, gdef_opt, rest_opt).lower(params, x, dy).compile()
      (loss_ref, out_ref), (grads_ref, dx_ref) = c_ref(params, x, dy)
      (loss_opt, out_opt), (grads_opt, dx_opt) = c_opt(params, x, dy)
      loss_diff = abs(float(loss_opt) - float(loss_ref))
      out_diff = float(np.max(np.abs(np.asarray(out_opt, np.float32) - np.asarray(out_ref, np.float32))))
      dx_diff = float(np.max(np.abs(np.asarray(dx_opt, np.float32) - np.asarray(dx_ref, np.float32))))
      wo_diff = float(
          np.max(
              np.abs(
                  np.asarray(jax.tree_util.tree_leaves(grads_opt["wo"])[0], np.float32)
                  - np.asarray(jax.tree_util.tree_leaves(grads_ref["wo"])[0], np.float32)
              )
          )
      )
      print(
          f"[QWAG TPU parity] overrides={qwag_overrides} scanned_amax={use_scanned_amax} "
          f"loss_abs_diff={loss_diff:.6e} out_max_abs={out_diff:.6e} "
          f"dx_max_abs={dx_diff:.6e} wo_grad_max_abs={wo_diff:.6e}",
          flush=True,
      )

      if cfg_ref.shard_embed_moe_on_fsdp and fsdp > 1:
        hlo_ref = c_ref.as_text()
        hlo_opt = c_opt.as_text()
        minor_ag_pat = r"all-gather(?:-start)?\([^\n]*dimensions=\{2\}"
        minor_rs_pat = r"reduce-scatter(?:-start)?\([^\n]*dimensions=\{2\}"
        self.assertGreater(len(re.findall(minor_ag_pat, hlo_ref)), 0)
        self.assertEqual(len(re.findall(minor_ag_pat, hlo_opt)), 0)
        self.assertGreater(len(re.findall(minor_rs_pat, hlo_ref)), 0)
        self.assertEqual(len(re.findall(minor_rs_pat, hlo_opt)), 0)
        self.assertNotIn("-transpose_out", hlo_ref)
        self.assertIn("-transpose_out", hlo_opt)
        self.assertIn("-transpose_rhs-subtile_m", hlo_opt)

    tol = 5e-2 if cfg_ref.moe_quantize_dispatch_fwd_method else 3e-2
    np.testing.assert_allclose(np.asarray(out_opt, np.float32), np.asarray(out_ref, np.float32), rtol=tol, atol=tol)
    np.testing.assert_allclose(np.asarray(dx_opt, np.float32), np.asarray(dx_ref, np.float32), rtol=tol, atol=tol)
    pkeys = ("gate", "wi", "wo") if cfg_ref.prefuse_moe_weights else ("gate", "wi_0", "wi_1", "wo")
    for k in pkeys:
      g_opt_leaf = jax.tree_util.tree_leaves(grads_opt[k])[0]
      g_ref_leaf = jax.tree_util.tree_leaves(grads_ref[k])[0]
      self.assertEqual(g_opt_leaf.sharding, g_ref_leaf.sharding, msg=f"sharding mismatch on {k}")
      np.testing.assert_allclose(
          np.asarray(g_opt_leaf, np.float32),
          np.asarray(g_ref_leaf, np.float32),
          rtol=tol,
          atol=tol,
      )


if __name__ == "__main__":
  absltest.main()
