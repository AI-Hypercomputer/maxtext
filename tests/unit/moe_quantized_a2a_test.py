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

  @parameterized.product(knob=_FWD_KNOBS + _BWD_KNOBS, method=("block512", "tensor", "fixed,1.0", "block512@e4m3"))
  def test_phase3_methods_not_implemented(self, knob, method):
    with self.assertRaisesRegex(ValueError, re.escape(f"{knob}={method!r} is not implemented (Phase 3)")):
      _config(f"qa2a_p3_{knob}", **{knob: method})

  @parameterized.product(
      knob=_FWD_KNOBS + _BWD_KNOBS, method=("foo", "rowwise@e3m4", "rowwise@", "rowwise,1.0", "Rowwise")
  )
  def test_junk_methods_rejected(self, knob, method):
    with self.assertRaisesRegex(ValueError, re.escape(f"Unsupported {knob}={method!r} on the EP ragged all-to-all")):
      _config(f"qa2a_junk_{knob}", **{knob: method})

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

  @parameterized.parameters("rowwise@e4m3", "rowwise@e5m2", "foo")
  def test_ring_combine_bwd_rejects_format_suffix(self, method):
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


class QuantizedA2aWireTest(parameterized.TestCase):
  """Parsing of the ragged all-to-all quantization methods (runs on any backend)."""

  def test_parse_a2a_quant_method(self):
    parse = moe._parse_a2a_quant_method  # pylint: disable=protected-access
    self.assertIsNone(parse("", _E4M3))
    self.assertEqual(parse("rowwise", _E4M3), _E4M3)
    self.assertEqual(parse("rowwise", _E5M2), _E5M2)
    self.assertEqual(parse("rowwise@e5m2", _E4M3), _E5M2)
    self.assertEqual(parse("rowwise@e4m3", _E5M2), _E4M3)
    with self.assertRaisesRegex(ValueError, "got|Unsupported"):
      parse("block512", _E4M3)


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
    assert fwd_method == "rowwise", fwd_method
    x_q = moe._ste_quantize_tokens(x, _E4M3, (0,), "absmax")  # pylint: disable=protected-access
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
  # A multiple of 512 features: XLA:TPU tiles an f8 row in (4, 128) blocks.
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

  def _assert_close(self, got, want, valid_rows, rel_bound, what, lossy=True):
    """|got - want| <= rel_bound * the absmax of each row on valid rows; tail rows exactly 0."""
    self._assert_tail_zero(got, valid_rows, what)
    got, want = got.reshape(_NUM_SHARDS, self.ROWS, -1), want.reshape(_NUM_SHARDS, self.ROWS, -1)
    for s in range(_NUM_SHARDS):
      n = valid_rows[s]
      err = np.max(np.abs(got[s, :n] - want[s, :n]), axis=-1)
      bad = np.argwhere(err > rel_bound * np.max(np.abs(want[s, :n]), axis=-1) * 1.001)
      self.assertEmpty(bad, f"{what}: shard {s} rows {bad[:8].tolist()}")
      if lossy and n:  # A quantized wire must not be a silent passthrough of the BF16/F32 payload.
        self.assertGreater(float(np.max(err)), 0.0, f"{what}: shard {s} is not quantized")

  def _assert_quantized_close(self, got, want, valid_rows, method, forward, dtype, what):
    """Within half an FP8 ulp of the absmax of each row."""
    bound = 2.0 ** -(_mantissa_bits(_qtype(method, forward)) + 1) + (2.0**-8 if dtype == jnp.bfloat16 else 0.0)
    self._assert_close(got, want, valid_rows, bound, what)

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

  @parameterized.product(method=("rowwise@e4m3", "rowwise@e5m2"), full=(False, True))
  def test_forward_quantized(self, method, full):
    """Forward wire quantized: FP8 error bound per row, zero tail rows, exact (unquantized) backward."""
    sizes, routing, x, ct = self._inputs(jnp.float32, full)
    y_ref, dx_ref = self._run("jax", "", "", x, ct, routing)
    y, dx = self._run("wrapper", method, "", x, ct, routing)
    self._assert_quantized_close(y, y_ref, sizes.sum(axis=0), method, True, jnp.float32, "forward")
    np.testing.assert_array_equal(dx, dx_ref, err_msg="unquantized backward")

  @parameterized.product(method=("rowwise@e4m3", "rowwise@e5m2"), full=(False, True))
  def test_backward_quantized(self, method, full):
    """Backward wire quantized: exact forward, FP8 error bound on dx, zero dx on rows never sent."""
    sizes, routing, x, ct = self._inputs(jnp.float32, full)
    y_ref, dx_ref = self._run("jax", "", "", x, ct, routing)
    y, dx = self._run("wrapper", "", method, x, ct, routing)
    np.testing.assert_array_equal(y, y_ref, err_msg="unquantized forward")
    self._assert_quantized_close(dx, dx_ref, sizes.sum(axis=1), method, False, jnp.float32, "backward")

  @parameterized.parameters(("rowwise",))
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
  )
  def test_check_vma(self, fwd_method, bwd_method):
    """The custom_vjp forward and backward trace under shard_map(check_vma=True) and match check_vma=False."""
    _, routing, x, ct = self._inputs(jnp.bfloat16, False)
    y_ref, dx_ref = self._run("wrapper", fwd_method, bwd_method, x, ct, routing)
    y, dx = self._run("wrapper", fwd_method, bwd_method, x, ct, routing, check_vma=True)
    np.testing.assert_array_equal(y, y_ref)
    np.testing.assert_array_equal(dx, dx_ref)

  def _ragged_all_to_all_payloads(self, fwd_method, bwd_method, remat=False):
    """Sorted (dtype, rows, element count) of every ragged all-to-all result in the compiled bf16 fwd + bwd."""
    _, routing, x, ct = self._inputs(jnp.bfloat16, False)
    fn = _ra2a_fwd_and_grad(self.mesh, "wrapper", fwd_method, bwd_method, remat=remat)
    results = _ragged_all_to_all_results(fn.lower(x, ct, *routing).compile().as_text())
    return sorted((dt, dims[0], int(np.prod(dims))) for dt, dims in results)

  def test_hlo_wire_dtypes(self):
    """Knobs on: FP8 payloads and one f32 scale per row, no BF16; knobs off: BF16 only."""
    rows, n = self.ROWS, self.ROWS * self.DIM
    got = {
        methods: self._ragged_all_to_all_payloads(*methods)
        for methods in (("", ""), ("rowwise", "rowwise"), ("", "rowwise@e4m3"))
    }
    # XLA:TPU moves each f32 scale as a (1, 128) tile.
    scale = ("f32", rows, rows * 128)
    self.assertEqual(got[("", "")], [("bf16", rows, n)] * 2)
    self.assertEqual(got[("rowwise", "rowwise")], sorted([("f8e4m3fn", rows, n), ("f8e5m2", rows, n), scale, scale]))
    self.assertEqual(got[("", "rowwise@e4m3")], sorted([("bf16", rows, n), ("f8e4m3fn", rows, n), scale]))

  def test_remat_adds_no_ragged_all_to_all(self):
    """The backward needs only the integer routing, so jax.checkpoint recomputes no ragged all-to-all."""
    for methods in (("", ""), ("rowwise", "rowwise")):
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
  scaling = cfg.moe_quantize_dispatch_fwd_method.partition("@")[0]
  return bool(cfg.quantization) and cfg.moe_quantize_dispatch_fwd_passthrough and scaling == "rowwise"


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


def _case(name, knobs, gmm_v2=False, ragged_sort=True, dequant=False, **overrides):
  """A RoutedMoE test case; `use_ragged_sort=True` (the production path) is primary, the XLA sort secondary."""
  return {
      "testcase_name": f"{'gmm_v2_' if gmm_v2 else ''}{name}_{'ragged' if ragged_sort else 'xla'}_sort",
      "knobs": knobs,
      "gmm_v2": gmm_v2,
      "overrides": {"use_ragged_sort": ragged_sort, **overrides},
      "dequant": dequant,
  }


_ROWWISE_ALL = dict.fromkeys(_FWD_KNOBS + _BWD_KNOBS, "rowwise")
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
    # qwix fp8_full tokamax gmm_v2 (Mixtral, bf16): leg 1 'rowwise' passes the FP8 rows into gmm_v2.
    *(_case("rowwise_leg1", {_LEG1: "rowwise"}, True, sort, dequant=True) for sort in (True, False)),
    _case("rowwise_leg1_rbf_2p5", {_LEG1: "rowwise"}, True, ragged_buffer_factor=2.5),
    # The 0.5x buffer overflows, so the layer's worst-case force_dropless chunks run.
    _case("rowwise_leg1_dropless", {_LEG1: "rowwise"}, True, ragged_buffer_factor=0.5, moe_dropless_fallback="layer"),
    _case("rowwise_leg1_leg4", {_LEG1: "rowwise", **_LEG4_ROWWISE}, True),
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
  def test_close_to_unquantized(self, knobs, gmm_v2, overrides, dequant):
    """Finite and close to knobs off, on shared parameters and inputs, with NaN-filled jax.lax.empty.

    dequant: also leg 1 without the passthrough, by moe_quantize_dispatch_fwd_passthrough=False and by a mock; both
      give the same bits, further from knobs off than the passthrough (two FP8 roundings instead of one).
    """
    name = self._testMethodName
    overrides = {"ici_expert_parallelism": _NUM_SHARDS, **overrides}
    # gmm_v2 needs spread token magnitudes to catch scale/row mix-ups.
    sigma = 2.0 if gmm_v2 else 1.0
    off = _MoeRun(_config(f"{name}_off", gmm_v2, **overrides), sigma=sigma)
    loss_ref, out_ref, grads_ref, calls_ref = off.run()
    self.assertEqual(calls_ref, 0)
    runs = {}
    for label, knobs_, patches in (
        ("on", knobs, ()),
        *((("knob_off", {**knobs, "moe_quantize_dispatch_fwd_passthrough": False}, ()),) if dequant else ()),
        *((("mock_off", knobs, (_force_dequant(),)),) if dequant else ()),
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

  @parameterized.named_parameters(
      ("rowwise_ragged_sort", "rowwise", True),
      ("rowwise_xla_sort", "rowwise", False),
  )
  def test_passthrough_hlo(self, method, use_ragged_sort):
    """Leg 1 carries FP8 (+ a 1-D bf16 scale) through the local sort into gmm_v2.

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
      # Leg-1 scale: one per row in the activation dtype (bf16; padded tiles); the dequantize mode sends f32.
      self.assertTrue(any(dt == "bf16" and dims[0] == rows[0] and len(dims) != 2 for dt, dims in pt), pt)
      self.assertTrue(any(dt == "f32" and dims[0] == rows[0] for dt, dims in dq), dq)
      self.assertFalse(any(dt == "f32" for dt, _ in pt), pt)


if __name__ == "__main__":
  absltest.main()
