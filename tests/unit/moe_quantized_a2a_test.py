# Copyright 2023–2026 Google LLC
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

"""Tests the FP8-quantized EP ragged all-to-all (`moe._quantized_ragged_all_to_all`) and its per-leg config knobs.

Legs of the ragged all-to-all expert-parallel path (use_ring_of_experts=False):
  1 dispatch forward  moe_quantize_dispatch_fwd_method (FP8 straight into gmm_v2: moe_quantize_dispatch_fwd_passthrough)
  2 combine forward   moe_quantize_combine_fwd_method
  3 combine backward  moe_quantize_combine_bwd_method
  4 dispatch backward moe_quantize_dispatch_bwd_method
"""

import contextlib
import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import qwix
import qwix.pallas as qpl

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers.initializers import nd_dense_init, variable_to_logically_partitioned
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_NUM_SHARDS = 4
_E4M3, _E5M2 = jnp.float8_e4m3fn, jnp.float8_e5m2
_F8 = (jnp.dtype(_E4M3), jnp.dtype(_E5M2))
_LEG1 = "moe_quantize_dispatch_fwd_method"
_FWD_KNOBS = (_LEG1, "moe_quantize_combine_fwd_method")
_BWD_KNOBS = ("moe_quantize_dispatch_bwd_method", "moe_quantize_combine_bwd_method")
_A2A_KNOBS = (*_FWD_KNOBS, _BWD_KNOBS[0])  # moe_quantize_combine_bwd_method also exists on the ring path.


def _qtype(method, forward):
  """FP8 wire dtype of a method: its '@<format>', else E4M3 forward and E5M2 backward."""
  fmt = method.partition("@")[2]
  return {"e4m3": _E4M3, "e5m2": _E5M2}[fmt] if fmt else (_E4M3 if forward else _E5M2)


def _mantissa_bits(qtype):
  return 3 if jnp.dtype(qtype) == jnp.dtype(_E4M3) else 2


# Tiny float32 DeepSeek-V3 with plain gmms.
_PLAIN = {
    "model_name": "deepseek3-tiny",
    "dtype": "float32",
    "weight_dtype": "float32",
    "matmul_precision": "highest",
    "megablox": True,
    "per_device_batch_size": 2,
    "max_target_length": 16,
    "ici_expert_parallelism": 2,
}
# bf16 Mixtral with qwix fp8_full tokamax gmm_v2 (in-kernel lhs quantization per (row, 512)), EP=4.
_GMM_V2 = {
    "model_name": "mixtral-8x7b",
    "base_emb_dim": 512,
    "base_mlp_dim": 512,
    "base_moe_mlp_dim": 512,
    "weight_dtype": "float32",
    "dtype": "bfloat16",
    "per_device_batch_size": 1,
    "max_target_length": 64,
    "quantize_router_proj": False,
    "ici_expert_parallelism": _NUM_SHARDS,
    "megablox": False,
    "use_tokamax_gmm": True,
    "use_gmm_v2": True,
    "quantization": "fp8_full",
    "use_qwix_quantization": True,
    "weight_quantization_calibration_method": "absmax",
    "act_quantization_calibration_method": "absmax",
    "bwd_quantization_calibration_method": "absmax",
    "check_vma": False,  # Rejected with the tokamax gmm; QuantizedA2aTpuTest covers the collectives with check_vma.
    **dict.fromkeys(("wi_tile_fwd_batch_seq", "wi_tile_dlhs_batch_seq", "wi_tile_drhs_batch_seq"), 128),
    **dict.fromkeys(("wo_tile_fwd_batch_seq", "wo_tile_dlhs_batch_seq", "wo_tile_drhs_batch_seq"), 128),
    **dict.fromkeys(("wi_tile_dlhs_embed_dim", "wo_tile_fwd_embed_dim", "wo_tile_dlhs_mlp_dim"), 256),
}


def _config(run_name, gmm_v2=False, **overrides):
  """`_PLAIN` or `_GMM_V2` on the ragged all-to-all path (ring of experts off)."""
  kwargs = {
      "run_name": run_name,
      "enable_checkpointing": False,
      "override_model_config": True,
      "float32_gate_logits": True,
      "sparse_matmul": True,
      "use_ring_of_experts": False,
      **(_GMM_V2 if gmm_v2 else _PLAIN),
      **overrides,
  }
  if kwargs.get("use_ragged_sort"):
    # The SparseCore ragged sort kernels need emb dim >= 1024 (float32) or 2048 (bf16), a multiple of 1024.
    kwargs["base_emb_dim"] = 2048 if gmm_v2 else 1024
  return pyconfig.initialize([None, get_test_config_path()], **kwargs)


class QuantizedA2aConfigTest(parameterized.TestCase):
  """Validation of the EP ragged all-to-all quantization knobs."""

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform == "tpu":
      self.skipTest("Config validation runs in the CPU target.")

  @parameterized.product(knob=_A2A_KNOBS, method=("rowwise", "rowwise@e4m3", "rowwise@e5m2"))
  def test_new_knob_accepts_rowwise(self, knob, method):
    cfg = _config(f"qa2a_ok_{knob}_{method}", **{knob: method})
    self.assertEqual(getattr(cfg, knob), method)

  @parameterized.parameters("rowwise", "rowwise@e4m3", "rowwise@e5m2")
  def test_combine_bwd_accepts_rowwise_without_ring(self, method):
    cfg = _config(f"qa2a_cbwd_ok_{method}", moe_quantize_combine_bwd_method=method)
    self.assertEqual(cfg.moe_quantize_combine_bwd_method, method)

  @parameterized.product(knob=_FWD_KNOBS + _BWD_KNOBS, method=("block512", "block512@e4m3", "block512@e5m2"))
  def test_block512_accepted(self, knob, method):
    cfg = _config(f"qa2a_blk_{knob}", base_emb_dim=512, **{knob: method})
    self.assertEqual(getattr(cfg, knob), method)

  @parameterized.product(knob=_FWD_KNOBS, method=("fixed,1.0", "fixed,448@e4m3", "fixed,2.5e-3@e5m2"))
  def test_fixed_accepted_on_forward_legs(self, knob, method):
    cfg = _config(f"qa2a_fixed_{knob}", **{knob: method})
    self.assertEqual(getattr(cfg, knob), method)

  @parameterized.product(knob=_BWD_KNOBS, method=("fixed,1.0", "fixed,57344@e5m2", "fixed,0.00125@e5m2"))
  def test_fixed_accepted_on_backward_legs(self, knob, method):
    cfg = _config(f"qa2a_fixed_bwd_{knob}", **{knob: method})
    self.assertEqual(getattr(cfg, knob), method)

  @parameterized.product(knob=_FWD_KNOBS + _BWD_KNOBS, method=("tensor", "tensor@e4m3", "block256", "block1024@e5m2"))
  def test_unimplemented_methods_rejected(self, knob, method):
    with self.assertRaisesRegex(ValueError, re.escape(f"{knob}={method!r} is not implemented on the EP ragged")):
      _config(f"qa2a_unimpl_{knob}", base_emb_dim=1024, **{knob: method})

  @parameterized.product(
      knob=_FWD_KNOBS + _BWD_KNOBS,
      method=(
          "foo",
          "Rowwise",
          "rowwise@e3m4",
          "rowwise@",
          "rowwise,1.0",
          "block512@",
          "block512,1.0",
          "fixed",
          "fixed,",
          "fixed,0",
          "fixed,-1",
          "fixed,abc",
          "fixed,inf",
          "fixed,nan",
          "fixed,1.0@e3m4",
      ),
  )
  def test_junk_methods_rejected(self, knob, method):
    with self.assertRaisesRegex(ValueError, re.escape(f"Unsupported {knob}={method!r} on the EP ragged all-to-all")):
      _config(f"qa2a_junk_{knob}", base_emb_dim=512, **{knob: method})

  @parameterized.parameters(*(_FWD_KNOBS + _BWD_KNOBS))
  def test_block512_needs_divisible_feature_dim(self, knob):
    """deepseek3-tiny's emb_dim is 64."""
    with self.assertRaisesRegex(
        ValueError, re.escape("needs the ragged all-to-all feature dim divisible by 512, got 64")
    ):
      _config(f"qa2a_blk_dim_{knob}", **{knob: "block512"})

  def test_block512_combine_needs_divisible_tensor_parallel_shard(self):
    """Combine sends the tensor-parallel shard of each row, dispatch the whole row."""
    common = {"base_emb_dim": 512, "ici_tensor_parallelism": 2}
    cfg = _config("qa2a_blk_tp_dispatch", moe_quantize_dispatch_fwd_method="block512", **common)
    self.assertEqual(cfg.moe_quantize_dispatch_fwd_method, "block512")
    with self.assertRaisesRegex(ValueError, re.escape("got 256 (expert input dim 512 / tensor parallelism 2)")):
      _config("qa2a_blk_tp_combine", moe_quantize_combine_fwd_method="block512", **common)

  @parameterized.parameters(*_A2A_KNOBS)
  def test_new_knob_rejected_with_ring_of_experts(self, knob):
    with self.assertRaisesRegex(ValueError, re.escape(f"{knob}='rowwise' quantizes the EP ragged all-to-all")):
      _config(f"qa2a_ring_{knob}", use_ring_of_experts=True, **{knob: "rowwise"})

  def test_new_knob_requires_sparse_matmul(self):
    with self.assertRaisesRegex(ValueError, re.escape("requires sparse_matmul=True")):
      _config("qa2a_dense", sparse_matmul=False, moe_quantize_dispatch_fwd_method="rowwise")

  @parameterized.parameters("rowwise", "fixed,1.0", "fixed,57344")
  def test_ring_combine_bwd_unchanged(self, method):
    cfg = _config(f"qa2a_ring_cbwd_{method}", use_ring_of_experts=True, moe_quantize_combine_bwd_method=method)
    self.assertEqual(cfg.moe_quantize_combine_bwd_method, method)

  @parameterized.parameters("rowwise@e4m3", "rowwise@e5m2", "block512", "foo")
  def test_ring_combine_bwd_rejects_ragged_all_to_all_methods(self, method):
    with self.assertRaisesRegex(ValueError, re.escape(f"'{method}' with use_ring_of_experts=True")):
      _config("qa2a_ring_cbwd_bad", use_ring_of_experts=True, moe_quantize_combine_bwd_method=method)


def _spread_rows(key, shape, dtype=jnp.float32):
  """Rows, and runs of 512 features, with magnitudes spread over orders of magnitude, so a wrong scale is caught."""
  rows, features = shape
  k1, k2, k3 = jax.random.split(key, 3)
  x = jax.random.normal(k1, shape, jnp.float32) * jnp.exp(2.0 * jax.random.normal(k2, (rows, 1)))
  if features % 512 == 0:
    x = x * jnp.repeat(jnp.exp(jax.random.normal(k3, (rows, features // 512))), 512, axis=1)
  return x.astype(dtype)


def _bits(q):
  return np.asarray(jax.lax.bitcast_convert_type(q, jnp.uint8))


def _gmm_v2_lhs_quantize(x, qtype, block=512):
  """tokamax gmm_v2's in-kernel lhs quantization (gmm_v2.py quantized matmul path), in `x.dtype` arithmetic."""
  rows, features = x.shape
  blocks = x.reshape(rows, features // block, block)
  scale = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True) / float(jnp.finfo(qtype).max)
  scale_inv = jnp.where(scale == 0, 0, 1 / scale)
  return (blocks * scale_inv).astype(qtype).reshape(rows, features)


def _ragged_all_to_all_dtypes(jaxpr):
  """Sorted result dtype names of every ragged_all_to_all in a jaxpr (recursing into sub-jaxprs)."""
  out = []
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "ragged_all_to_all":
      out.append(jnp.dtype(eqn.outvars[0].aval.dtype).name)
    for param in eqn.params.values():
      for sub in param if isinstance(param, (tuple, list)) else (param,):
        sub = getattr(sub, "jaxpr", sub)  # ClosedJaxpr -> Jaxpr.
        if hasattr(sub, "eqns"):
          out.extend(_ragged_all_to_all_dtypes(sub))
  return sorted(out)


class QuantizedA2aWireTest(parameterized.TestCase):
  """Parsing, quantization and scale packing of the ragged all-to-all wire (no collective; runs on any backend)."""

  def test_parse_a2a_quant_method(self):
    parse = moe._parse_a2a_quant_method  # pylint: disable=protected-access
    spec = moe.A2aQuantSpec
    self.assertIsNone(parse("", _E4M3))
    self.assertEqual(parse("rowwise", _E4M3), spec("rowwise", _E4M3))
    self.assertEqual(parse("rowwise", _E5M2), spec("rowwise", _E5M2))
    self.assertEqual(parse("rowwise@e5m2", _E4M3), spec("rowwise", _E5M2))
    self.assertEqual(parse("block512", _E5M2), spec("block512", _E5M2))
    self.assertEqual(parse("block512@e4m3", _E5M2), spec("block512", _E4M3))
    self.assertEqual(parse("fixed,448", _E4M3), spec("fixed", _E4M3, 448.0))
    self.assertEqual(parse("fixed,2.5e-3@e5m2", _E4M3), spec("fixed", _E5M2, 2.5e-3))

  @parameterized.parameters(
      "foo", "rowwise@", "rowwise@e3m4", "block256", "block512,1", "tensor", "fixed", "fixed,0", "fixed,-2", "fixed,x"
  )
  def test_parse_rejects(self, method):
    with self.assertRaisesRegex(ValueError, "Unsupported ragged all-to-all quantization method"):
      moe._parse_a2a_quant_method(method, _E4M3)  # pylint: disable=protected-access

  def test_fixed_backward_traces_one_fp8_ragged_all_to_all(self):
    """'fixed' backward wire: one FP8 ragged all-to-all of the cotangent and no scale collective.

    Traced only (XLA:CPU has no ragged-all-to-all); the forward wire stays unquantized.
    """
    x = jnp.zeros((8, 512), jnp.float32)
    routing = (jnp.zeros((_NUM_SHARDS,), jnp.int32),) * 4
    for bwd_method in ("fixed,1.0", "fixed,0.00125@e5m2", "fixed,448@e4m3"):

      def fwd_and_grad(x, ct, bwd_method=bwd_method):
        y, vjp = jax.vjp(
            lambda x_: moe._quantized_ragged_all_to_all(  # pylint: disable=protected-access
                x_, jnp.zeros_like(x_), *routing, axis_name="expert", bwd_method=bwd_method
            ),
            x,
        )
        return y, vjp(ct)[0]

      jaxpr = jax.make_jaxpr(fwd_and_grad, axis_env=[("expert", _NUM_SHARDS)])(x, x).jaxpr
      self.assertEqual(
          _ragged_all_to_all_dtypes(jaxpr),
          sorted([jnp.dtype(jnp.float32).name, jnp.dtype(_qtype(bwd_method, False)).name]),
          bwd_method,
      )

  @parameterized.product(dtype=(jnp.bfloat16, jnp.float32), qtype=(_E4M3, _E5M2))
  def test_block512_requant_lossless(self, dtype, qtype):
    """block512 wire -> dequantize -> gmm_v2's in-kernel requantize gives back exactly the wire's FP8 values."""
    x = _spread_rows(jax.random.PRNGKey(0), (64, 2048), dtype)
    x = x.at[0, :512].set(0).at[1].set(0)  # An all-zero block and an all-zero row.
    spec = moe.A2aQuantSpec("block512", qtype)
    qvalue, scale = moe._a2a_quantize_rows(x, spec)  # pylint: disable=protected-access
    self.assertEqual(scale.shape, (64, 4))
    dequantized = moe._a2a_dequantize_rows(qvalue, scale, dtype)  # pylint: disable=protected-access
    requantized = _gmm_v2_lhs_quantize(dequantized, qtype)
    np.testing.assert_array_equal(_bits(requantized), _bits(qvalue))
    # For reference: the kernel's own quantization of x rounds a few values differently (bf16 reciprocal multiply vs
    # the wire's f32 divide), so the forward with leg-1 block512 is close to, not bitwise equal to, knobs off.
    same = float(np.mean(_bits(_gmm_v2_lhs_quantize(x, qtype)) == _bits(qvalue)))
    print(f"[{jnp.dtype(dtype).name}/{jnp.dtype(qtype).name}] kernel(x) == wire values: {same:.4%}", flush=True)

  @parameterized.product(dtype=(jnp.bfloat16, jnp.float32), qtype=(_E4M3, _E5M2))
  def test_block512_dequantize_error_bound(self, dtype, qtype):
    """Every value comes back within half an FP8 ulp of its own block's scale; blocks differ 4x in magnitude."""
    x = _spread_rows(jax.random.PRNGKey(5), (16, 2048), jnp.float32)
    x = (x * jnp.repeat(4.0 ** jnp.arange(4), 512)).astype(dtype)
    qvalue, scale = moe._a2a_quantize_rows(x, moe.A2aQuantSpec("block512", qtype))  # pylint: disable=protected-access
    got = np.asarray(moe._a2a_dequantize_rows(qvalue, scale, dtype), np.float32)  # pylint: disable=protected-access
    want = np.asarray(x, np.float32)
    block_scale = np.repeat(np.asarray(scale), 512, axis=1)
    out_rounding = 2.0**-8 if dtype == jnp.bfloat16 else 0.0
    tol = (2.0 ** -(_mantissa_bits(qtype) + 1) + out_rounding) * np.abs(want) + 0.5 * float(
        jnp.finfo(qtype).smallest_subnormal
    ) * block_scale
    self.assertTrue(np.all(np.abs(got - want) <= tol * 1.001), f"max excess {np.max(np.abs(got - want) - tol)}")

  @parameterized.parameters("rowwise", "block512")
  def test_pack_roundtrip_bit_exact(self, scaling):
    x = _spread_rows(jax.random.PRNGKey(1), (32, 1024), jnp.bfloat16)
    qvalue, scale = moe._a2a_quantize_rows(x, moe.A2aQuantSpec(scaling, _E4M3))  # pylint: disable=protected-access
    packed = moe._a2a_pack_rows(qvalue, scale)  # pylint: disable=protected-access
    self.assertEqual(packed.dtype, jnp.uint8)
    self.assertEqual(packed.shape, (32, 1024 + 4 * scale.shape[1]))
    got_qvalue, got_scale = moe._a2a_unpack_rows(packed, 1024, _E4M3)  # pylint: disable=protected-access
    np.testing.assert_array_equal(_bits(got_qvalue), _bits(qvalue))
    np.testing.assert_array_equal(np.asarray(got_scale), np.asarray(scale))

  @parameterized.product(qtype=(_E4M3, _E5M2), bound=(8.0, 0.00125))
  def test_fixed_scale_and_saturation(self, qtype, bound):
    x = _spread_rows(jax.random.PRNGKey(2), (32, 512), jnp.float32)
    self.assertGreater(float(jnp.max(jnp.abs(x))), bound)
    qvalue, scale = moe._a2a_quantize_rows(x, moe.A2aQuantSpec("fixed", qtype, bound))  # pylint: disable=protected-access
    qmax = float(jnp.finfo(qtype).max)
    np.testing.assert_allclose(np.asarray(scale), [[bound / qmax]], rtol=1e-7)
    got = np.asarray(moe._a2a_dequantize_rows(qvalue, scale, jnp.float32))  # pylint: disable=protected-access
    want = np.clip(np.asarray(x), -bound, bound)
    tol = 2.0 ** -(_mantissa_bits(qtype) + 1) * np.abs(want) + 0.5 * float(jnp.finfo(qtype).smallest_subnormal) * (
        bound / qmax
    )
    self.assertTrue(np.all(np.abs(got - want) <= tol * 1.001 + 1e-7 * bound))
    self.assertLessEqual(float(np.max(np.abs(got))), bound * (1 + 1e-6))

  def test_quantize_rejects_indivisible_features(self):
    with self.assertRaisesRegex(ValueError, "needs features \\(768\\) divisible by 512"):
      moe._a2a_quantize_rows(  # pylint: disable=protected-access
          jnp.ones((4, 768), jnp.float32), moe.A2aQuantSpec("block512", _E4M3)
      )


class LocalPermuteQArrayTest(parameterized.TestCase):
  """`local_permute` moves a QArray's FP8 rows bit-exactly with their per-row scale; tail rows get scale 0.

  On TPU, use_ragged_sort=True runs the FP8 rows through the SparseCore ragged gather (as uint32 words holding 4 rows).
  """

  # 4 EP shards, 2 local experts each. Rows [s, e] = tokens batch shard s sends to global expert e. Shard 1 receives 5.
  GLOBAL_GROUP_SIZES = np.array(
      [
          [2, 1, 1, 1, 1, 1, 0, 0],
          [2, 2, 1, 1, 0, 0, 1, 1],
          [1, 2, 0, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 2, 1, 1, 1],
      ],
      dtype=np.int32,
  )

  @parameterized.product(
      use_ragged_sort=(True, False),
      # (ragged_buffer_factor, buffer rows): worst-case and capped, both under-full; 6 rows is not a multiple of the 4
      # FP8 rows per uint32 word.
      buffer=((-1.0, 8), (1.5, 6)),
      hidden=(128, 7168),
  )
  def test_scale_follows_rows(self, use_ragged_sort, buffer, hidden):
    ragged_buffer_factor, buffer_size = buffer
    num_valid = 5
    x = jnp.zeros((buffer_size, hidden), jnp.float32)
    x = x.at[:num_valid].set(_spread_rows(jax.random.PRNGKey(0), (num_valid, hidden)))
    # Tail rows have value 0 and a nonzero scale (qwix gives all-zero rows scale 1; a 'fixed' receiver puts its static
    # scale on every row), which local_permute must zero.
    x_q = qpl.quantize(x, qtype=_E4M3, channelwise_axes=(0,), calibration_method="absmax")
    self.assertTrue(np.all(np.asarray(x_q.scale) > 0))
    global_group_sizes = jnp.asarray(self.GLOBAL_GROUP_SIZES)
    static = {"local_expert_size": 2, "shard_index": 1, "ragged_buffer_factor": ragged_buffer_factor}
    permute = jax.jit(moe.RoutedMoE.local_permute, static_argnames=tuple(static) + ("use_ragged_sort",))
    got, idx, group_sizes, expert_ids = permute(x_q, global_group_sizes, use_ragged_sort=use_ragged_sort, **static)
    want, idx_ref, group_sizes_ref, expert_ids_ref = permute(
        x_q.qvalue.astype(jnp.float32) * x_q.scale, global_group_sizes, use_ragged_sort=False, **static
    )
    self.assertIsInstance(got, qpl.QArray)
    self.assertEqual(got.qvalue.dtype, _E4M3)
    self.assertEqual(got.scale.shape, (buffer_size, 1))
    np.testing.assert_array_equal(idx, idx_ref)
    np.testing.assert_array_equal(group_sizes, group_sizes_ref)
    np.testing.assert_array_equal(expert_ids, expert_ids_ref)
    # Bit-exact FP8 row moves: the sorted bytes equal an XLA gather of the received bytes; tail rows are 0.
    want_bytes = _bits(x_q.qvalue)[np.asarray(idx)]
    want_bytes[num_valid:] = 0
    np.testing.assert_array_equal(_bits(got.qvalue), want_bytes)
    np.testing.assert_array_equal(got.qvalue.astype(jnp.float32) * got.scale, want)
    np.testing.assert_array_equal(got.scale[num_valid:], 0.0)
    self.assertTrue(np.all(np.asarray(got.scale[:num_valid]) > 0))


def _routing(full, rows_per_shard, seed=0):
  """Per-shard ragged all-to-all parameters, each [shards, shards] int32.

  sizes[i, j] = rows shard i sends to shard j, taken contiguously from the front of shard i's operand. Receiver j
  stores the rows source-shard major from row 0. With `full` every receiver gets exactly `rows_per_shard` rows;
  otherwise every receiver buffer is under-full and some routes are empty.
  """
  n = _NUM_SHARDS
  if full:
    sizes = np.full((n, n), rows_per_shard // n, np.int32)
  else:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(0, rows_per_shard // n, size=(n, n)).astype(np.int32)
    sizes[0, 1] = 0
    sizes[2, 2] = 0
  input_offsets = np.cumsum(sizes, axis=1) - sizes
  output_offsets = np.cumsum(sizes, axis=0) - sizes  # [i, j]: offset of shard i's rows on receiver j
  recv_sizes = sizes.T
  return sizes, tuple(jnp.asarray(a, jnp.int32) for a in (input_offsets, sizes, output_offsets, recv_sizes))


@jax.custom_vjp
def _gmm_like_consumer(y):
  """Dequantizes a QArray lhs; its backward mirrors gmm_v2's for a QArray lhs (dlhs * scale, zero scale cotangent)."""
  return y.qvalue.astype(jnp.float32) * y.scale.astype(jnp.float32)


def _gmm_like_consumer_fwd(y):
  return _gmm_like_consumer(y), y.scale


def _gmm_like_consumer_bwd(scale, g):
  dlhs = g.astype(scale.dtype) * scale
  return (qpl.QArray(qvalue=dlhs, scale=jnp.zeros_like(scale), zero_point=None, qtype=_E4M3),)


_gmm_like_consumer.defvjp(_gmm_like_consumer_fwd, _gmm_like_consumer_bwd)


def _ra2a_fwd_and_grad(mesh, impl, fwd_method, bwd_method, check_vma=False, remat=False):
  """jit(x, ct, *routing) -> (y, dx) for one ragged all-to-all over the `expert` axis of `mesh`.

  impl: 'jax' (jax.lax.ragged_all_to_all), 'vjp' (the custom_vjp core), 'wrapper' (`_quantized_ragged_all_to_all`),
  'qarray' (`_ragged_all_to_all_qarray` of the rows quantized by `fwd_method`, into a gmm_v2-like consumer) or
  'sender_dequant' (the same quantized rows, dequantized before a plain ragged all-to-all).
  """

  def per_shard(x, input_offsets, send_sizes, output_offsets, recv_sizes):
    # A fresh (non-varying under check_vma) zero buffer, as at the RoutedMoE call sites.
    output = jnp.zeros(x.shape, x.dtype)
    routing = (input_offsets[0], send_sizes[0], output_offsets[0], recv_sizes[0])
    if impl == "jax":
      return jax.lax.ragged_all_to_all(x, output, *routing, axis_name="expert")
    if impl == "vjp":  # The custom_vjp core, also with both methods ''.
      return moe._quantized_ragged_all_to_all_vjp(  # pylint: disable=protected-access
          x, output, *routing, "expert", fwd_method, bwd_method, x.shape[0]
      )
    if impl == "wrapper":
      return moe._quantized_ragged_all_to_all(  # pylint: disable=protected-access
          x, output, *routing, axis_name="expert", fwd_method=fwd_method, bwd_method=bwd_method
      )
    if fwd_method == "rowwise":
      x_q = moe._ste_quantize_tokens(x, _E4M3, (0,), "absmax")  # pylint: disable=protected-access
    else:  # 'fixed,<bound>': the static scale bound / 448, in f32 math.
      x_q = qpl.quantize(x.astype(jnp.float32), _E4M3, channelwise_axes=(), calibration_method=fwd_method)
    if impl == "sender_dequant":
      x_deq = x_q.qvalue.astype(jnp.float32) * x_q.scale.astype(jnp.float32)
      return jax.lax.ragged_all_to_all(x_deq, jnp.zeros_like(x_deq), *routing, axis_name="expert")
    y = moe._ragged_all_to_all_qarray(  # pylint: disable=protected-access
        x_q, x.shape[0], *routing, axis_name="expert", bwd_method=bwd_method
    )
    assert y.qvalue.dtype == _E4M3 and y.scale.shape == (x.shape[0], 1), y
    return _gmm_like_consumer(y)

  if remat:
    per_shard = jax.checkpoint(per_shard)
  sharded = jax.shard_map(per_shard, mesh=mesh, in_specs=(P("expert"),) * 5, out_specs=P("expert"), check_vma=check_vma)

  @jax.jit
  def fn(x, ct, *routing):
    y, vjp = jax.vjp(lambda x_: sharded(x_, *routing), x)
    (dx,) = vjp(ct.astype(y.dtype))
    return y, dx

  return fn


def _ra2a_wire_forward(mesh, spec, pack_scales):
  """jit(x, *routing) -> y: forward of one `_ra2a_maybe_quantized` wire over the `expert` axis of `mesh`."""

  def per_shard(x, input_offsets, send_sizes, output_offsets, recv_sizes):
    routing = (input_offsets[0], send_sizes[0], output_offsets[0], recv_sizes[0])
    return moe._ra2a_maybe_quantized(  # pylint: disable=protected-access
        x, jnp.zeros(x.shape, x.dtype), *routing, "expert", spec, pack_scales
    )

  return jax.jit(jax.shard_map(per_shard, mesh=mesh, in_specs=(P("expert"),) * 5, out_specs=P("expert"), check_vma=False))


def _ragged_all_to_all_results(hlo):
  """Sorted (dtype, dims) of the result of every ragged all-to-all in a compiled HLO module.

  The compiled TPU HLO prints operands by name only and re-tiles each payload row (e.g. bf16[64,1024] as
  bf16[64,4,256], f32[64] as f32[64,1,128]), so this reads the result type, which equals the operand type.
  """
  out = []
  for line in hlo.splitlines():
    m = re.search(r"= (\w+)\[([\d,]*)\]\S* ragged-all-to-all(?:-start)?\(", line)
    if m:
      out.append((m.group(1), tuple(int(d) for d in m.group(2).split(",") if d)))
  return sorted(out)


class QuantizedA2aTpuTest(parameterized.TestCase):
  """`_quantized_ragged_all_to_all` and `_ragged_all_to_all_qarray` on 4 TPU chips (XLA:CPU has no ragged-all-to-all)."""

  ROWS = 64
  # Two 512-feature blocks per row. A multiple of 512 features: XLA:TPU tiles an f8 row in (4, 128) blocks.
  DIM = 1024

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform != "tpu":
      self.skipTest("The ragged all-to-all needs a TPU.")
    if jax.device_count() < _NUM_SHARDS:
      self.skipTest(f"Needs {_NUM_SHARDS} devices.")
    self.mesh = Mesh(np.array(jax.devices()[:_NUM_SHARDS]), ("expert",))

  def _inputs(self, dtype, full):
    sizes, routing = _routing(full, self.ROWS)
    shape = (_NUM_SHARDS * self.ROWS, self.DIM)
    x, ct = (_spread_rows(jax.random.PRNGKey(seed), shape, dtype) for seed in (0, 1))
    return sizes, routing, x, ct

  def _run(self, impl, fwd_method, bwd_method, x, ct, routing, **kwargs):
    y, dx = _ra2a_fwd_and_grad(self.mesh, impl, fwd_method, bwd_method, **kwargs)(x, ct, *routing)
    return np.asarray(y.astype(jnp.float32)), np.asarray(dx.astype(jnp.float32))

  def _assert_tail_zero(self, got, valid_rows, what):
    got = got.reshape(_NUM_SHARDS, self.ROWS, -1)
    for s in range(_NUM_SHARDS):
      np.testing.assert_array_equal(got[s, valid_rows[s] :], 0.0, err_msg=f"{what}: shard {s} tail rows")

  def _assert_close(self, got, want, valid_rows, rel_bound, what, block=None, lossy=True):
    """|got - want| <= rel_bound * the absmax of each row (or `block` features) on valid rows; tail rows exactly 0."""
    self._assert_tail_zero(got, valid_rows, what)
    got = got.reshape(_NUM_SHARDS, self.ROWS, -1, block or self.DIM)
    want = want.reshape(_NUM_SHARDS, self.ROWS, -1, block or self.DIM)
    for s in range(_NUM_SHARDS):
      n = valid_rows[s]
      err = np.max(np.abs(got[s, :n] - want[s, :n]), axis=-1)
      bad = np.argwhere(err > rel_bound * np.max(np.abs(want[s, :n]), axis=-1) * 1.001)
      self.assertEmpty(bad, f"{what}: shard {s} (row, block) {bad[:8].tolist()}")
      if lossy and n:  # A quantized wire must not be a silent passthrough of the BF16/F32 payload.
        self.assertGreater(float(np.max(err)), 0.0, f"{what}: shard {s} is not quantized")

  def _assert_quantized_close(self, got, want, valid_rows, method, forward, dtype, what):
    """Within half an FP8 ulp of the absmax of each row ('rowwise') or 512-feature block ('block512')."""
    bound = 2.0 ** -(_mantissa_bits(_qtype(method, forward)) + 1) + (2.0**-8 if dtype == jnp.bfloat16 else 0.0)
    self._assert_close(got, want, valid_rows, bound, what, block=512 if method.startswith("block512") else None)

  def _assert_fixed_close(self, got, want, valid_rows, method, what):
    """Values clipped to +-bound, then within half an FP8 ulp (or half the smallest subnormal step)."""
    self._assert_tail_zero(got, valid_rows, what)
    qtype = _qtype(method, True)
    bound = float(method.partition("@")[0].split(",")[1])
    step = float(jnp.finfo(qtype).smallest_subnormal) * bound / float(jnp.finfo(qtype).max)
    got, want = got.reshape(_NUM_SHARDS, self.ROWS, -1), want.reshape(_NUM_SHARDS, self.ROWS, -1)
    clipped = 0
    for s in range(_NUM_SHARDS):
      n = valid_rows[s]
      ref = np.clip(want[s, :n], -bound, bound)
      clipped += int(np.sum(np.abs(want[s, :n]) > bound))
      tol = 2.0 ** -(_mantissa_bits(qtype) + 1) * np.abs(ref) + 0.5 * step
      bad = np.argwhere(np.abs(got[s, :n] - ref) > tol * 1.001 + 1e-7 * bound)
      self.assertEmpty(bad, f"{what}: shard {s} (row, feature) {bad[:8].tolist()}")
    self.assertGreater(clipped, 0, "test data never exceeds the fixed bound")

  @parameterized.named_parameters(
      ("underfull_f32", False, jnp.float32),
      ("full_f32", True, jnp.float32),
      ("underfull_bf16", False, jnp.bfloat16),
  )
  def test_unquantized_matches_jax_bitwise(self, full, dtype):
    """Both methods '': the custom_vjp core and the public wrapper equal jax.lax.ragged_all_to_all bit for bit."""
    _, routing, x, ct = self._inputs(dtype, full)
    y_ref, dx_ref = self._run("jax", "", "", x, ct, routing)
    for impl in ("vjp", "wrapper"):
      y, dx = self._run(impl, "", "", x, ct, routing)
      np.testing.assert_array_equal(y, y_ref, err_msg=f"{impl} forward")
      np.testing.assert_array_equal(dx, dx_ref, err_msg=f"{impl} backward")
    self.assertGreater(float(np.max(np.abs(dx_ref))), 0.0)

  @parameterized.product(
      method=("rowwise@e4m3", "rowwise@e5m2", "block512", "block512@e4m3", "block512@e5m2"), full=(False, True)
  )
  def test_forward_quantized(self, method, full):
    """Forward wire quantized: FP8 error bound per row / block, zero tail rows, exact (unquantized) backward."""
    sizes, routing, x, ct = self._inputs(jnp.float32, full)
    y_ref, dx_ref = self._run("jax", "", "", x, ct, routing)
    y, dx = self._run("wrapper", method, "", x, ct, routing)
    self._assert_quantized_close(y, y_ref, sizes.sum(axis=0), method, True, jnp.float32, "forward")
    np.testing.assert_array_equal(dx, dx_ref, err_msg="unquantized backward")

  @parameterized.product(method=("fixed,8", "fixed,8@e4m3", "fixed,8@e5m2"), full=(False, True))
  def test_forward_fixed(self, method, full):
    """Forward wire with a static scale: saturates at +-bound, FP8 error bound, zero tail rows, exact backward."""
    sizes, routing, x, ct = self._inputs(jnp.float32, full)
    y_ref, dx_ref = self._run("jax", "", "", x, ct, routing)
    y, dx = self._run("wrapper", method, "", x, ct, routing)
    self._assert_fixed_close(y, y_ref, sizes.sum(axis=0), method, "forward")
    np.testing.assert_array_equal(dx, dx_ref, err_msg="unquantized backward")

  @parameterized.product(
      method=("rowwise@e4m3", "rowwise@e5m2", "block512", "block512@e4m3", "block512@e5m2"), full=(False, True)
  )
  def test_backward_quantized(self, method, full):
    """Backward wire quantized: exact forward, FP8 error bound on dx, zero dx on rows never sent."""
    sizes, routing, x, ct = self._inputs(jnp.float32, full)
    y_ref, dx_ref = self._run("jax", "", "", x, ct, routing)
    y, dx = self._run("wrapper", "", method, x, ct, routing)
    np.testing.assert_array_equal(y, y_ref, err_msg="unquantized forward")
    self._assert_quantized_close(dx, dx_ref, sizes.sum(axis=1), method, False, jnp.float32, "backward")

  @parameterized.product(method=("fixed,1@e5m2", "fixed,1", "fixed,0.00125@e5m2", "fixed,1@e4m3"), full=(False, True))
  def test_backward_fixed(self, method, full):
    """Static-scale backward wire: the plain cotangent clipped to +-bound, through FP8 and back (bit exact).

    The reference quantizes with qpl.quantize's static scale bound / qmax; saturated values come back as +-bound and
    the forward is exact.
    """
    sizes, routing, x, ct = self._inputs(jnp.float32, full)
    y_ref, dx_ref = self._run("jax", "", "", x, ct, routing)
    y, dx = self._run("wrapper", "", method, x, ct, routing)
    np.testing.assert_array_equal(y, y_ref, err_msg="unquantized forward")
    bound = float(method.partition("@")[0].split(",")[1])
    qtype = _qtype(method, False)

    @jax.jit
    def quantize_dequantize(g):
      g_q = qpl.quantize(g, qtype=qtype, channelwise_axes=(), calibration_method=f"fixed,{bound}")
      return g_q.qvalue.astype(jnp.float32) * g_q.scale

    np.testing.assert_array_equal(dx, np.asarray(quantize_dequantize(jnp.asarray(dx_ref))), err_msg="backward")
    self._assert_tail_zero(dx, sizes.sum(axis=1), "backward")
    over = np.abs(dx_ref) > bound
    self.assertGreater(int(np.sum(over)), 0, "test cotangent never exceeds the fixed bound")
    np.testing.assert_allclose(dx[over], np.sign(dx_ref[over]) * bound, rtol=1e-6, err_msg="saturation")
    self.assertLessEqual(float(np.max(np.abs(dx))), bound * (1 + 1e-6))
    self.assertGreater(float(np.max(np.abs(dx - np.clip(dx_ref, -bound, bound)))), 0.0, "backward is not quantized")

  @parameterized.parameters("rowwise", "block512")
  def test_bf16_both_quantized(self, method):
    sizes, routing, x, ct = self._inputs(jnp.bfloat16, False)
    y_ref, dx_ref = self._run("jax", "", "", x, ct, routing)
    y, dx = self._run("wrapper", method, method, x, ct, routing)
    self._assert_quantized_close(y, y_ref, sizes.sum(axis=0), method, True, jnp.bfloat16, "forward e4m3")
    self._assert_quantized_close(dx, dx_ref, sizes.sum(axis=1), method, False, jnp.bfloat16, "backward e5m2")

  @parameterized.parameters(
      ("", ""),
      ("rowwise", ""),
      ("", "rowwise"),
      ("rowwise@e5m2", "rowwise@e4m3"),
      ("rowwise", "rowwise"),
      ("block512", "block512"),
      ("fixed,4", "block512@e4m3"),
      ("fixed,4@e5m2", ""),
      ("rowwise", "fixed,1"),
  )
  def test_check_vma(self, fwd_method, bwd_method):
    """The custom_vjp forward and backward trace under shard_map(check_vma=True) and match check_vma=False."""
    _, routing, x, ct = self._inputs(jnp.bfloat16, False)
    y_ref, dx_ref = self._run("wrapper", fwd_method, bwd_method, x, ct, routing)
    y, dx = self._run("wrapper", fwd_method, bwd_method, x, ct, routing, check_vma=True)
    np.testing.assert_array_equal(y, y_ref)
    np.testing.assert_array_equal(dx, dx_ref)

  @parameterized.product(scaling=("rowwise", "block512"), full=(False, True))
  def test_packed_scales_match_separate_scales_bitwise(self, scaling, full):
    """Packing the scale bytes into the value rows changes only the wire layout, not a single received bit."""
    sizes, routing, x, _ = self._inputs(jnp.bfloat16, full)
    spec = moe.A2aQuantSpec(scaling, _E4M3)
    packed = np.asarray(_ra2a_wire_forward(self.mesh, spec, True)(x, *routing).astype(jnp.float32))
    separate = np.asarray(_ra2a_wire_forward(self.mesh, spec, False)(x, *routing).astype(jnp.float32))
    np.testing.assert_array_equal(packed, separate)
    self._assert_tail_zero(packed, sizes.sum(axis=0), "packed")
    self.assertGreater(float(np.max(np.abs(packed))), 0.0)

  def _ragged_all_to_all_payloads(self, fwd_method, bwd_method, remat=False):
    """Sorted (dtype, rows, element count) of every ragged all-to-all result in the compiled bf16 fwd + bwd."""
    _, routing, x, ct = self._inputs(jnp.bfloat16, False)
    fn = _ra2a_fwd_and_grad(self.mesh, "wrapper", fwd_method, bwd_method, remat=remat)
    results = _ragged_all_to_all_results(fn.lower(x, ct, *routing).compile().as_text())
    return sorted((dt, dims[0], int(np.prod(dims))) for dt, dims in results)

  def test_hlo_wire_dtypes(self):
    """Knobs on: FP8 (or packed uint8) payloads, f32 scales only where unpacked, no BF16; knobs off: BF16 only."""
    rows, n = self.ROWS, self.ROWS * self.DIM
    got = {
        methods: self._ragged_all_to_all_payloads(*methods)
        for methods in (
            ("", ""),
            ("rowwise", "rowwise"),
            ("", "rowwise@e4m3"),
            ("fixed,8", ""),
            ("block512", "block512"),
            ("", "fixed,8"),
        )
    }
    # XLA:TPU moves each f32 scale as a (1, 128) tile.
    scale = ("f32", rows, rows * 128)
    self.assertEqual(got[("", "")], [("bf16", rows, n)] * 2)
    self.assertEqual(got[("rowwise", "rowwise")], sorted([("f8e4m3fn", rows, n), ("f8e5m2", rows, n), scale, scale]))
    self.assertEqual(got[("", "rowwise@e4m3")], sorted([("bf16", rows, n), ("f8e4m3fn", rows, n), scale]))
    # 'fixed': the FP8 values only; the static scale never crosses the wire.
    self.assertEqual(got[("fixed,8", "")], sorted([("bf16", rows, n), ("f8e4m3fn", rows, n)]))
    self.assertEqual(got[("", "fixed,8")], sorted([("bf16", rows, n), ("f8e5m2", rows, n)]))
    # 'block512': one uint8 ragged all-to-all per leg, the 2 f32 scales per row packed after the values.
    self.assertEqual([(dt, r) for dt, r, _ in got[("block512", "block512")]], [("u8", rows)] * 2)
    for _, _, numel in got[("block512", "block512")]:
      self.assertGreaterEqual(numel, rows * (self.DIM + 8))

  def test_remat_adds_no_ragged_all_to_all(self):
    """The backward needs only the integer routing, so jax.checkpoint recomputes no ragged all-to-all."""
    for methods in (("", ""), ("rowwise", "rowwise"), ("fixed,8", "block512"), ("rowwise", "fixed,8")):
      self.assertEqual(
          self._ragged_all_to_all_payloads(*methods, remat=True),
          self._ragged_all_to_all_payloads(*methods),
          f"methods={methods}",
      )

  @parameterized.parameters(jnp.float32, jnp.bfloat16)
  def test_qarray_forward_matches_sender_dequant_and_backward_is_ste(self, dtype):
    """Leg-1 passthrough wire: the sender's quantized rows and scales; backward ('' wire): the plain transpose (STE)."""
    sizes, routing, x, ct = self._inputs(dtype, False)
    (y, dx), (y_deq, _), (y_ref, dx_ref) = (
        self._run(impl, "rowwise", "", x, ct, routing, check_vma=True) for impl in ("qarray", "sender_dequant", "jax")
    )
    recv, bf16 = sizes.sum(axis=0), dtype == jnp.bfloat16
    if not bf16:
      np.testing.assert_array_equal(y, y_deq, err_msg="received values/scales differ from the sender's")
    else:  # Separate compilations may round bf16 x / scale with different excess precision: allow one e4m3 ulp.
      self._assert_close(y, y_deq, recv, 2.0**-3, "fwd vs sender dequant", lossy=False)
    # Half an e4m3 ulp of the row absmax, plus the rounding of a bf16 scale.
    self._assert_close(y, y_ref, recv, 2.0**-4 + (2.0**-7 if bf16 else 0.0), "fwd")
    # dx = (ct * scale) / scale on the sender rows: equal to the plain transpose up to the scale round trip.
    self._assert_close(dx, dx_ref, sizes.sum(axis=1), 2.0**-7 if bf16 else 1e-6, "bwd", lossy=False)

  @parameterized.parameters("rowwise", "rowwise@e5m2", "rowwise@e4m3")
  def test_qarray_backward_quantized(self, bwd_method):
    sizes, routing, x, ct = self._inputs(jnp.bfloat16, False)
    _, dx = self._run("qarray", "rowwise", bwd_method, x, ct, routing, check_vma=True)
    _, dx_ref = self._run("jax", "", "", x, ct, routing)
    bound = 2.0 ** -(_mantissa_bits(_qtype(bwd_method, False)) + 1) + 2.0**-6
    self._assert_close(dx, dx_ref, sizes.sum(axis=1), bound, "bwd")

  def test_qarray_backward_fixed(self):
    """Leg-1 passthrough with a static-scale backward wire: the wire cotangent clipped to +-bound, one E5M2 rounding.

    The wire carries g = (consumer dlhs) * row scale and the sender divides by its row scale (STE), so dx * scale is
    clip(g, +-bound) through FP8, with g = dx * scale of the unquantized ('') backward wire.
    """
    sizes, routing, x, ct = self._inputs(jnp.float32, False)
    _, dx_ste = self._run("qarray", "rowwise", "", x, ct, routing, check_vma=True)
    ste_quantize = moe._ste_quantize_tokens  # pylint: disable=protected-access
    scale = np.asarray(ste_quantize(x, _E4M3, (0,), "absmax").scale, np.float32)
    g = dx_ste * scale
    bound = float(np.quantile(np.abs(g[g != 0]), 0.9))  # About 10% of the sent values saturate.
    _, dx = self._run("qarray", "rowwise", f"fixed,{bound!r}@e5m2", x, ct, routing, check_vma=True)
    got, want = dx * scale, np.clip(g, -bound, bound)
    step = float(jnp.finfo(_E5M2).smallest_subnormal) * bound / float(jnp.finfo(_E5M2).max)
    # Half an e5m2 ulp of the clipped value, or half the smallest subnormal step; rows never sent stay 0.
    np.testing.assert_array_less(np.abs(got - want), 2.0**-3 * np.abs(want) + 0.5 * step + 1e-6 * bound)
    self._assert_tail_zero(dx, sizes.sum(axis=1), "bwd")
    over = np.abs(g) > bound
    self.assertGreater(int(np.sum(over)), 0, "no wire cotangent exceeds the fixed bound")
    np.testing.assert_allclose(got[over], np.sign(g[over]) * bound, rtol=1e-5, err_msg="saturation")
    self.assertGreater(float(np.max(np.abs(got - want))), 0.0, "backward wire is not quantized")

  @parameterized.parameters(jnp.float32, jnp.bfloat16)
  def test_qarray_fixed_forward_matches_sender_dequant_and_backward_is_clipped_ste(self, dtype):
    """Static 'fixed,64' scale (no scale collective): the sender's clipped rows; backward: the clipped STE."""
    bound = 64.0
    sizes, routing, _, ct = self._inputs(dtype, False)
    x = (jax.random.normal(jax.random.PRNGKey(0), ct.shape) * (bound / 2)).astype(dtype)  # About 5% clip.
    (y, dx), (y_deq, _), (y_ref, dx_ref) = (
        self._run(impl, "fixed,64", "", x, ct, routing, check_vma=True) for impl in ("qarray", "sender_dequant", "jax")
    )
    x = np.asarray(x.astype(jnp.float32))
    # Both quantize in f32 math with the same static scale and multiply by it in f32.
    np.testing.assert_array_equal(y, y_deq, err_msg="received values differ from the sender's")
    self.assertGreater(float(np.max(np.abs(y_ref))), bound, "no value clips")
    self.assertLessEqual(float(np.max(np.abs(y))), bound)
    # Half an e4m3 ulp of the clipped value, or half the smallest subnormal step (2^-10 of the scale).
    want = np.clip(y_ref, -bound, bound)
    np.testing.assert_array_less(np.abs(y - want), 2.0**-4 * np.abs(want) + 2.0**-10 * bound / 448 + 1e-6)
    self.assertGreater(float(np.max(np.abs(y - want))), 0.0, "forward wire is not quantized")
    # dx = (ct * scale) / scale where |x| < bound, 0 where x clipped; skip values within 0.1% of the bound (ties).
    tie = np.abs(np.abs(x) - bound) <= 1e-3 * bound
    dx_want = np.where(tie, dx, np.where(np.abs(x) < bound, dx_ref, 0.0))
    self.assertGreater(int(np.sum((np.abs(x) > bound) & (dx_ref != 0))), 0, "no clipped element has a cotangent")
    self._assert_close(dx, dx_want, sizes.sum(axis=1), 1e-6, "bwd", lossy=False)


def _nan_empty(shape, dtype, **kwargs):
  """Stand-in for `jax.lax.empty` that makes uninitialized memory visible as NaN."""
  del kwargs
  return jnp.full(shape, jnp.nan, dtype=dtype)


def _force_dequant():
  """Leg 1 dequantizes after the collective and gmm_v2 quantizes again, whatever the passthrough knob says."""
  return mock.patch.object(moe.RoutedMoE, "_dispatch_fwd_passthrough_quantizer", return_value=None)


class _MoeRun:
  """One linen RoutedMoE (qwix-quantized if `cfg.quantization`) and its inputs, on shared parameters.

  Everything runs with NaN-filled `jax.lax.empty` and a spy on the leg-1 passthrough collective.
  """

  def __init__(self, cfg, variables=None, sigma=1.0):
    self.cfg = cfg
    self.mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    self.model = nnx_wrappers.to_linen(
        moe.RoutedMoE,
        name="MoeBlock",
        metadata_fn=variable_to_logically_partitioned,
        abstract_init=False,
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.moe_mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
    )
    if cfg.quantization:
      rule = qwix.QtRule(
          module_path=".*",
          weight_qtype=_E4M3,
          act_qtype=_E4M3,
          bwd_qtype=_E5M2,
          weight_calibration_method=cfg.weight_quantization_calibration_method,
          act_calibration_method=cfg.act_quantization_calibration_method,
          bwd_calibration_method=cfg.bwd_quantization_calibration_method,
          op_names=("gmm", "ragged_dot"),
      )
      self.model = qwix.quantize_model(self.model, qwix.QtProvider([rule]))
    shape = (cfg.per_device_batch_size * jax.device_count(), cfg.max_target_length, cfg.base_emb_dim)
    # Token magnitudes spread over ~e^(+-2 sigma): a row paired with another token's scale gives an O(1) error.
    x = jax.random.normal(jax.random.PRNGKey(3), shape)
    self.x = (x * jnp.exp(sigma * jax.random.normal(jax.random.PRNGKey(4), (*shape[:2], 1)))).astype(cfg.dtype)
    with self.context():
      rngs = {"params": jax.random.PRNGKey(5), "dropout": jax.random.PRNGKey(5)}
      self.variables = variables or self.model.init(rngs, self.x)

  @contextlib.contextmanager
  def context(self, *patches):
    """Yields the spy on `moe._ragged_all_to_all_qarray`."""
    with contextlib.ExitStack() as stack:
      spy = stack.enter_context(
          mock.patch.object(moe, "_ragged_all_to_all_qarray", wraps=moe._ragged_all_to_all_qarray)  # pylint: disable=protected-access
      )
      for cm in (jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.cfg.logical_axis_rules), *patches):
        stack.enter_context(cm)
      stack.enter_context(mock.patch("jax.lax.empty", side_effect=_nan_empty))
      yield spy

  def loss_fn(self, params, x):
    out, lb_loss, _ = self.model.apply({"params": params}, x)
    loss = jnp.mean(out.astype(jnp.float32) ** 2)
    return (loss if lb_loss is None else loss + lb_loss.astype(jnp.float32)), out

  def step(self, remat=False):
    # The input gradient is the only one leg 4 (dispatch backward) reaches.
    return jax.jit(jax.value_and_grad(jax.checkpoint(self.loss_fn) if remat else self.loss_fn, (0, 1), has_aux=True))

  def run(self, *patches):
    """(loss, out, {'params', 'inputs'} grads) as float32 numpy, and the traced leg-1 passthrough collectives."""
    with self.context(*patches) as spy:
      (loss, out), (grads, x_grad) = self.step()(self.variables["params"], self.x)
    to_np = lambda a: np.asarray(a, np.float32)  # pylint: disable=unnecessary-lambda-assignment
    return to_np(loss), to_np(out), jax.tree_util.tree_map(to_np, {"params": grads, "inputs": x_grad}), spy.call_count

  def forward_jaxpr(self, *patches):
    with self.context(*patches):
      return jax.make_jaxpr(self.loss_fn)(self.variables["params"], self.x).jaxpr

  def compiled_hlo(self, remat, *patches):
    with self.context(*patches):
      return self.step(remat).lower(self.variables["params"], self.x).compile().as_text()


def _rel_err(got, want):
  got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
  return float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30))


def _grad_errs(grads, grads_ref):
  return {
      jax.tree_util.keystr(path): _rel_err(g, g_ref)
      for (path, g), g_ref in zip(
          jax.tree_util.tree_leaves_with_path(grads), jax.tree_util.tree_leaves(grads_ref), strict=True
      )
  }


def _passthrough_expected(cfg):
  """Whether leg 1 hands its FP8 rows straight to gmm_v2 in these tests' configs."""
  scaling = cfg.moe_quantize_dispatch_fwd_method.partition("@")[0].split(",", maxsplit=1)[0]
  return bool(cfg.quantization) and cfg.moe_quantize_dispatch_fwd_passthrough and scaling in ("rowwise", "fixed")


def _f8_dequant_converts(jaxpr):
  """convert_element_type ops from an FP8 operand to a wider float outside Pallas kernels (recursing into sub-jaxprs)."""
  count = 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in ("pallas_call", "core_map"):
      continue
    if eqn.primitive.name == "convert_element_type":
      if jnp.dtype(eqn.invars[0].aval.dtype) in _F8 and jnp.dtype(eqn.outvars[0].aval.dtype) not in _F8:
        count += 1
    for param in eqn.params.values():
      for sub in param if isinstance(param, (tuple, list)) else (param,):
        sub = getattr(sub, "jaxpr", sub)  # ClosedJaxpr -> Jaxpr.
        if hasattr(sub, "eqns"):
          count += _f8_dequant_converts(sub)
  return count


def _case(name, knobs, gmm_v2=False, ragged_sort=True, dequant=False, worse=None, **overrides):
  """A RoutedMoE test case; `use_ragged_sort=True` (the production path) is primary, the XLA sort secondary."""
  return {
      "testcase_name": f"{'gmm_v2_' if gmm_v2 else ''}{name}_{'ragged' if ragged_sort else 'xla'}_sort",
      "knobs": knobs,
      "gmm_v2": gmm_v2,
      "overrides": {"use_ragged_sort": ragged_sort, **overrides},
      "dequant": dequant,
      "worse": worse,
  }


_ROWWISE_ALL = dict.fromkeys(_FWD_KNOBS + _BWD_KNOBS, "rowwise")
_BLOCK512_ALL = dict.fromkeys(_FWD_KNOBS + _BWD_KNOBS, "block512")
_FIXED_FWD = {_LEG1: "fixed,64", "moe_quantize_combine_fwd_method": "fixed,64@e5m2"}
_ROWWISE_BWD = dict.fromkeys(_BWD_KNOBS, "rowwise")
_BLOCK512_BWD = {"moe_quantize_combine_bwd_method": "block512", "moe_quantize_dispatch_bwd_method": "block512@e4m3"}
_DROPLESS = {"ragged_buffer_factor": 1.0, "moe_dropless_fallback": "layer"}
_LEG4_ROWWISE = {"moe_quantize_dispatch_bwd_method": "rowwise"}
_MOE_CASES = (
    # Plain gmms (deepseek3-tiny, float32); ragged_buffer_factor -1 unless set. The XLA sort cases also cover
    # check_vma, which the ragged sort does not support yet.
    _case("rowwise_leg1", {_LEG1: "rowwise"}),
    _case("rowwise_leg2", {"moe_quantize_combine_fwd_method": "rowwise@e4m3"}),
    _case("rowwise_leg3", {"moe_quantize_combine_bwd_method": "rowwise@e5m2"}),
    _case("rowwise_leg4", {"moe_quantize_dispatch_bwd_method": "rowwise"}),
    _case("rowwise_all_rbf_2p5", _ROWWISE_ALL, ragged_buffer_factor=2.5),
    _case("rowwise_all_dropless_fallback_layer", _ROWWISE_ALL, **_DROPLESS),
    _case("rowwise_all_check_vma", _ROWWISE_ALL, ragged_sort=False, check_vma=True),
    _case("fixed_leg1", {_LEG1: "fixed,64"}),
    _case("fixed_leg2", {"moe_quantize_combine_fwd_method": "fixed,64@e5m2"}),
    _case("fixed_fwd_rowwise_bwd_dropless_fallback_layer", {**_FIXED_FWD, **_ROWWISE_BWD}, **_DROPLESS),
    _case("fixed_fwd_rowwise_bwd_check_vma", {**_FIXED_FWD, **_ROWWISE_BWD}, ragged_sort=False, check_vma=True),
    _case("block512_leg1", {_LEG1: "block512"}),
    _case("block512_leg2", {"moe_quantize_combine_fwd_method": "block512@e5m2"}),
    _case("block512_leg3", {"moe_quantize_combine_bwd_method": "block512"}),
    _case("block512_leg4", {"moe_quantize_dispatch_bwd_method": "block512@e4m3"}),
    _case("block512_all_rbf_2p5", _BLOCK512_ALL, ragged_buffer_factor=2.5),
    _case("block512_all_dropless_fallback_layer", _BLOCK512_ALL, **_DROPLESS),
    # block512 needs the feature dim divisible by 512; deepseek3-tiny's is 64 (the ragged sort makes it 1024).
    _case("block512_all_check_vma", _BLOCK512_ALL, ragged_sort=False, check_vma=True, base_emb_dim=512),
    _case("fixed_fwd_block512_bwd", {**_FIXED_FWD, **_BLOCK512_BWD}),
    # qwix fp8_full tokamax gmm_v2 (Mixtral, bf16): leg 1 'rowwise' / 'fixed' passes the FP8 rows into gmm_v2.
    *(_case("rowwise_leg1", {_LEG1: "rowwise"}, True, sort, dequant=True) for sort in (True, False)),
    _case("rowwise_leg1_rbf_2p5", {_LEG1: "rowwise"}, True, ragged_buffer_factor=2.5),
    # The 0.5x buffer overflows, so the layer's worst-case force_dropless chunks run.
    _case("rowwise_leg1_dropless", {_LEG1: "rowwise"}, True, ragged_buffer_factor=0.5, moe_dropless_fallback="layer"),
    _case("rowwise_leg1_leg4", {_LEG1: "rowwise", **_LEG4_ROWWISE}, True),
    *(_case("fixed_leg1", {_LEG1: "fixed,64"}, True, sort, dequant=True) for sort in (True, False)),
    _case("fixed_leg1_rbf_2p5", {_LEG1: "fixed,64"}, True, ragged_buffer_factor=2.5),
    # 'fixed' passthrough does not depend on the act calibration: here the gmms quantize their inputs statically.
    _case("fixed_leg1_static_act", {_LEG1: "fixed,64"}, True, act_quantization_calibration_method="fixed,64"),
    _case("fixed_leg1_leg4", {_LEG1: "fixed,64", **_LEG4_ROWWISE}, True),
    # gmm_v2 re-quantizes the dequantized block512 rows to exactly the wire's FP8 values (test_block512_requant_lossless),
    # so leg-1 block512 is closer to knobs off than leg-1 rowwise, whose one scale per token replaces the kernel's
    # per-(token, 512) scales. At 512 features the two are the same quantizer.
    *(
        _case("block512_leg1", {_LEG1: "block512"}, True, sort, worse={_LEG1: "rowwise"}, base_emb_dim=2048)
        for sort in (True, False)
    ),
)


class QuantizedA2aRoutedMoETpuTest(parameterized.TestCase):
  """Full RoutedMoE (ring of experts off, EP=4) with quantized ragged all-to-all legs, vs all knobs off."""

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform != "tpu":
      self.skipTest("The ragged all-to-all path needs a TPU.")
    if jax.device_count() % _NUM_SHARDS:
      self.skipTest(f"Needs a multiple of {_NUM_SHARDS} devices for expert parallelism {_NUM_SHARDS}.")

  @parameterized.named_parameters(*_MOE_CASES)
  def test_close_to_unquantized(self, knobs, gmm_v2, overrides, dequant, worse):
    """Finite and close to knobs off, on shared parameters and inputs, with NaN-filled jax.lax.empty.

    dequant: also leg 1 without the passthrough, by moe_quantize_dispatch_fwd_passthrough=False and by a mock; both
      give the same bits, further from knobs off than the passthrough (two FP8 roundings instead of one).
    worse: knobs whose output is further from knobs off.
    """
    name = self._testMethodName
    overrides = {"ici_expert_parallelism": _NUM_SHARDS, **overrides}
    # gmm_v2 'fixed,64' needs tokens inside its bound; 'rowwise' needs spread magnitudes to catch scale/row mix-ups.
    sigma = (0.0 if knobs.get(_LEG1, "").startswith("fixed") else 2.0) if gmm_v2 else 1.0
    off = _MoeRun(_config(f"{name}_off", gmm_v2, **overrides), sigma=sigma)
    loss_ref, out_ref, grads_ref, calls_ref = off.run()
    self.assertEqual(calls_ref, 0)
    runs = {}
    for label, knobs_, patches in (
        ("on", knobs, ()),
        *((("knob_off", {**knobs, "moe_quantize_dispatch_fwd_passthrough": False}, ()),) if dequant else ()),
        *((("mock_off", knobs, (_force_dequant(),)),) if dequant else ()),
        *((("worse", worse, ()),) if worse else ()),
    ):
      cfg = _config(f"{name}_{label}", gmm_v2, **overrides, **knobs_)
      runs[label] = _MoeRun(cfg, off.variables, sigma).run(*patches)
      expected = _passthrough_expected(cfg) and not patches
      self.assertEqual(runs[label][3] > 0, expected, f"{label}: {runs[label][3]} leg-1 passthrough calls")

    errs = {}
    for label, (loss, out, grads, _) in runs.items():
      errs[label] = out_err = _rel_err(out, out_ref)
      grad_errs = _grad_errs(grads, grads_ref)
      print(f"[{name}] {label}: out rel err {out_err:.4e}; loss {loss_ref} -> {loss}; grad rel errs {grad_errs}")
      self.assertTrue(np.isfinite(loss), f"{label}: loss={loss}")
      self.assertTrue(np.all(np.isfinite(out)), f"{label}: non-finite MoE output")
      for path, g in jax.tree_util.tree_leaves_with_path(grads):
        self.assertTrue(np.all(np.isfinite(g)), f"{label}: non-finite gradient at {jax.tree_util.keystr(path)}")
      if any(knobs.get(k) for k in _FWD_KNOBS):
        self.assertGreater(out_err, 0.0, f"{label}: forward knob had no effect")
      else:
        np.testing.assert_array_equal(out, out_ref, err_msg="Backward-only knobs changed the forward.")
      self.assertLess(out_err, 0.08, label)
      self.assertLess(abs(float(loss) - float(loss_ref)), 0.08 * abs(float(loss_ref)) + 1e-6, label)
      self.assertGreater(max(grad_errs.values()), 0.0, f"{label}: knobs had no effect on the gradients")
      for path, err in grad_errs.items():
        self.assertLess(err, 0.22 if gmm_v2 else 0.15, f"{label}: grad rel err {err} at {path}")
    if dequant:
      for got, want in zip(runs["knob_off"][:3], runs["mock_off"][:3]):
        jax.tree_util.tree_map(np.testing.assert_array_equal, got, want)
      self.assertLess(errs["on"], errs["knob_off"])
      self.assertBetween(_rel_err(runs["on"][1], runs["knob_off"][1]), 1e-30, 0.08)
    if worse:
      self.assertLess(errs["on"], errs["worse"])

  @parameterized.named_parameters(
      ("rowwise_ragged_sort", "rowwise", True),
      ("rowwise_xla_sort", "rowwise", False),
      ("fixed_ragged_sort", "fixed,64", True),
  )
  def test_passthrough_hlo(self, method, use_ragged_sort):
    """Leg 1 carries FP8 (+ a 1-D bf16 scale for 'rowwise' only) through the local sort into gmm_v2.

    No FP8 dequantize, and as many ragged all-to-alls as the dequantize mode, with and without remat.
    """
    cfg = _config(self._testMethodName, True, use_ragged_sort=use_ragged_sort, **{_LEG1: method})
    run = _MoeRun(cfg)
    with mock.patch.object(moe, "a2a_ragged_sort", wraps=moe.a2a_ragged_sort) as sort_spy:
      self.assertEqual(_f8_dequant_converts(run.forward_jaxpr()), 0)
    self.assertGreater(_f8_dequant_converts(run.forward_jaxpr(_force_dequant())), 0)
    if use_ragged_sort:  # The SparseCore local sort moves the FP8 rows themselves (no upcast around it).
      self.assertEqual({jnp.dtype(c.args[0].dtype) for c in sort_spy.call_args_list}, {jnp.dtype(_E4M3)})
    for remat in (False, True):
      pt = _ragged_all_to_all_results(run.compiled_hlo(remat))
      dq = _ragged_all_to_all_results(run.compiled_hlo(remat, _force_dequant()))
      self.assertLen(pt, len(dq), f"remat={remat}: passthrough changed the number of ragged all-to-alls")
      rows = [dims[0] for dt, dims in pt if dt == "f8e4m3fn"]
      self.assertNotEmpty(rows, pt)
      if method.startswith("fixed"):
        # No scale collective in either mode: every ragged all-to-all carries full token rows.
        self.assertEqual(pt, dq)
        for dt, dims in pt:
          self.assertEqual(int(np.prod(dims[1:])), cfg.base_emb_dim, f"{dt}{list(dims)} is not a token payload")
      else:
        # Leg-1 scale: one per row in the activation dtype (bf16; padded tiles); the dequantize mode sends f32.
        self.assertTrue(any(dt == "bf16" and dims[0] == rows[0] and len(dims) != 2 for dt, dims in pt), pt)
        self.assertTrue(any(dt == "f32" and dims[0] == rows[0] for dt, dims in dq), dq)
        self.assertFalse(any(dt == "f32" for dt, _ in pt), pt)


if __name__ == "__main__":
  absltest.main()
