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
  1 dispatch forward  moe_quantize_dispatch_fwd_method
  2 combine forward   moe_quantize_combine_fwd_method
  3 combine backward  moe_quantize_combine_bwd_method
  4 dispatch backward moe_quantize_dispatch_bwd_method
"""

import contextlib
import functools
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
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from tests.utils import linen_wrappers
from tests.utils.test_helpers import get_test_config_path

_NUM_SHARDS = 4
_E4M3, _E5M2 = jnp.float8_e4m3fn, jnp.float8_e5m2
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

  @parameterized.product(knob=_FWD_KNOBS, method=("fixed,1.0", "fixed,448@e4m3", "fixed,2.5e-3@e5m2"))
  def test_fixed_accepted_on_forward_legs(self, knob, method):
    cfg = _config(f"qa2a_fixed_{knob}", **{knob: method})
    self.assertEqual(getattr(cfg, knob), method)

  @parameterized.product(knob=_BWD_KNOBS, method=("fixed,1.0", "fixed,57344@e5m2", "fixed,0.00125@e5m2"))
  def test_fixed_accepted_on_backward_legs(self, knob, method):
    cfg = _config(f"qa2a_fixed_bwd_{knob}", **{knob: method})
    self.assertEqual(getattr(cfg, knob), method)

  @parameterized.parameters(*zip(_FWD_KNOBS + _BWD_KNOBS, ("tensor", "tensor@e4m3", "block512", "block1024")))
  def test_unimplemented_methods_rejected(self, knob, method):
    with self.assertRaisesRegex(ValueError, re.escape(f"{knob}={method!r} is not implemented on the EP ragged")):
      _config(f"qa2a_unimpl_{knob}", base_emb_dim=1024, **{knob: method})

  # Every knob goes through the same parser (configs/types.py `parse_a2a_quant_method`, shared with
  # moe._parse_a2a_quant_method): the full junk list on leg 1, one junk value on each other leg.
  @parameterized.parameters(
      *(
          (_LEG1, method)
          for method in (
              "foo",
              "Rowwise",
              "rowwise@e3m4",
              "rowwise@",
              "rowwise,1.0",
              "fixed",
              "fixed,",
              "fixed,0",
              "fixed,-1",
              "fixed,abc",
              "fixed,inf",
              "fixed,nan",
              "fixed,1.0@e3m4",
          )
      ),
      ("moe_quantize_combine_fwd_method", "rowwise@e3m4"),
      ("moe_quantize_dispatch_bwd_method", "fixed,-1"),
      ("moe_quantize_combine_bwd_method", "fixed,nan"),
  )
  def test_junk_methods_rejected(self, knob, method):
    with self.assertRaisesRegex(ValueError, re.escape(f"Unsupported {knob}={method!r} on the EP ragged all-to-all")):
      _config(f"qa2a_junk_{knob}", base_emb_dim=512, **{knob: method})

  @parameterized.named_parameters(
      ("ragged_combine_bwd", {"moe_quantize_combine_bwd_method": "fixed,1.0"}, False, 2, True),
      ("ring_combine_bwd", {"moe_quantize_combine_bwd_method": "fixed,1.0"}, True, 2, True),
      ("ragged_dispatch_bwd", {"moe_quantize_dispatch_bwd_method": "fixed,1.0@e5m2"}, False, 2, True),
      ("no_gradient_accumulation", {"moe_quantize_dispatch_bwd_method": "fixed,1.0"}, False, 1, False),
      ("fixed_forward_only", {_LEG1: "fixed,1.0"}, False, 2, False),
      ("rowwise_backward", {"moe_quantize_combine_bwd_method": "rowwise"}, False, 2, False),
  )
  def test_fixed_backward_warns_with_manual_gradient_accumulation(self, knobs, ring, ga_steps, warns):
    """Manual gradient accumulation differentiates the unnormalized xent_sum: backward fixed bounds may saturate."""
    with mock.patch.object(max_logging, "warning") as warning:
      _config(f"qa2a_ga_{ga_steps}", use_ring_of_experts=ring, gradient_accumulation_steps=ga_steps, **knobs)
    ga_warnings = [call for call in warning.call_args_list if "xent_sum" in str(call.args[0])]
    self.assertEqual(bool(ga_warnings), warns, warning.call_args_list)

  @parameterized.parameters(*_A2A_KNOBS)
  def test_new_knob_rejected_with_ring_of_experts(self, knob):
    with self.assertRaisesRegex(ValueError, re.escape(f"{knob}='rowwise' quantizes the EP ragged all-to-all")):
      _config(f"qa2a_ring_{knob}", use_ring_of_experts=True, **{knob: "rowwise"})

  def test_new_knob_requires_sparse_matmul(self):
    with self.assertRaisesRegex(ValueError, re.escape("requires sparse_matmul=True")):
      _config("qa2a_dense", sparse_matmul=False, moe_quantize_dispatch_fwd_method="rowwise")

  @parameterized.parameters("rowwise", "rowwise@e4m3", "rowwise@e5m2", "fixed,1.0", "fixed,1.0@e4m3", "fixed,57344@e5m2")
  def test_ring_combine_bwd_accepted(self, method):
    cfg = _config(f"qa2a_ring_cbwd_{method}", use_ring_of_experts=True, moe_quantize_combine_bwd_method=method)
    self.assertEqual(cfg.moe_quantize_combine_bwd_method, method)

  @parameterized.parameters(
      "fixed", "fixed,-1", "fixed,0", "fixed,nan", "fixed,1.0@bad", "rowwise@bad", "block512", "foo"
  )
  def test_ring_combine_bwd_rejects_invalid_methods(self, method):
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


def _all_gather_dtypes(jaxpr):
  """Sorted operand dtype names of every all_gather in a jaxpr (recursing into sub-jaxprs)."""
  out = []
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "all_gather":
      out.append(jnp.dtype(eqn.invars[0].aval.dtype).name)
    for param in eqn.params.values():
      for sub in param if isinstance(param, (tuple, list)) else (param,):
        sub = getattr(sub, "jaxpr", sub)  # ClosedJaxpr -> Jaxpr.
        if hasattr(sub, "eqns"):
          out.extend(_all_gather_dtypes(sub))
  return sorted(out)


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
  """Parsing and quantization of the ragged all-to-all wire (no collective; runs on any backend)."""

  def test_parse_a2a_quant_method(self):
    parse = moe._parse_a2a_quant_method  # pylint: disable=protected-access
    spec = moe.A2aQuantSpec
    self.assertIsNone(parse("", _E4M3))
    self.assertEqual(parse("rowwise", _E4M3), spec("rowwise", _E4M3))
    self.assertEqual(parse("rowwise", _E5M2), spec("rowwise", _E5M2))
    self.assertEqual(parse("rowwise@e5m2", _E4M3), spec("rowwise", _E5M2))
    self.assertEqual(parse("fixed,448", _E4M3), spec("fixed", _E4M3, 448.0))
    self.assertEqual(parse("fixed,2.5e-3@e5m2", _E4M3), spec("fixed", _E5M2, 2.5e-3))

  @parameterized.parameters(
      "foo", "rowwise@", "rowwise@e3m4", "block512", "block256", "tensor", "fixed", "fixed,0", "fixed,-2", "fixed,x"
  )
  def test_parse_rejects(self, method):
    with self.assertRaisesRegex(ValueError, "Unsupported ragged all-to-all quantization method"):
      moe._parse_a2a_quant_method(method, _E4M3)  # pylint: disable=protected-access

  @parameterized.parameters(
      ("rowwise", _E5M2, True),
      ("rowwise@e4m3", _E4M3, True),
      ("rowwise@e5m2", _E5M2, True),
      ("fixed,1.0", _E5M2, False),
      ("fixed,1.0@e4m3", _E4M3, False),
  )
  def test_ring_all_gather_wire_dtype(self, method, qtype, scale_gathered):
    """The ring of experts combine-backward all-gather sends `method`'s FP8 format (default E5M2).

    Traced only: 'rowwise' also gathers its f32 per-token scales, 'fixed' gathers the FP8 values alone.
    """
    gather = functools.partial(
        moe._all_gather_quantized_payload,  # pylint: disable=protected-access
        axis_name="expert",
        axis=0,
        tiled=True,
        method=method,
    )
    jaxpr = jax.make_jaxpr(gather, axis_env=[("expert", _NUM_SHARDS)])(jnp.ones((8, 512), jnp.bfloat16)).jaxpr
    want = [jnp.dtype(qtype).name] + ([jnp.dtype(jnp.float32).name] if scale_gathered else [])
    self.assertEqual(_all_gather_dtypes(jaxpr), sorted(want), method)

  @parameterized.parameters("fixed", "fixed,-1", "rowwise@bad", "")
  def test_ring_all_gather_rejects_invalid_methods(self, method):
    with self.assertRaises(ValueError):
      moe._all_gather_quantized_payload(  # pylint: disable=protected-access
          jnp.ones((8, 512), jnp.bfloat16), "expert", axis=0, tiled=True, method=method
      )

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

  @parameterized.parameters("rowwise", "fixed,8")
  def test_zero_rows_quantize_to_zero(self, method):
    """All-zero rows (e.g. masked tokens) get a finite scale and dequantize to exactly 0."""
    x = _spread_rows(jax.random.PRNGKey(3), (32, 512), jnp.bfloat16).at[::3].set(0)
    parse, quantize, dequantize = (
        moe._parse_a2a_quant_method,  # pylint: disable=protected-access
        moe._a2a_quantize_rows,  # pylint: disable=protected-access
        moe._a2a_dequantize_rows,  # pylint: disable=protected-access
    )
    qvalue, scale = quantize(x, parse(method, _E4M3))
    self.assertTrue(np.all(np.isfinite(np.asarray(scale))), method)
    got = np.asarray(dequantize(qvalue, scale, jnp.bfloat16).astype(jnp.float32))
    self.assertTrue(np.all(np.isfinite(got)), method)
    np.testing.assert_array_equal(got[::3], 0.0, err_msg=method)
    self.assertGreater(float(np.min(np.max(np.abs(got[1::3]), axis=-1))), 0.0, method)

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


def _ra2a_fwd_and_grad(mesh, impl, fwd_method, bwd_method, check_vma=False, remat=False):
  """jit(x, ct, *routing) -> (y, dx) for one ragged all-to-all over the `expert` axis of `mesh`.

  impl: 'jax' (jax.lax.ragged_all_to_all), 'vjp' (the custom_vjp core) or 'wrapper' (`_quantized_ragged_all_to_all`).
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
    return moe._quantized_ragged_all_to_all(  # pylint: disable=protected-access
        x, output, *routing, axis_name="expert", fwd_method=fwd_method, bwd_method=bwd_method
    )

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
  """`_quantized_ragged_all_to_all` on 4 TPU chips (XLA:CPU has no ragged-all-to-all)."""

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

  @parameterized.product(method=("rowwise@e4m3", "rowwise@e5m2"), full=(False, True))
  def test_forward_quantized(self, method, full):
    """Forward wire quantized: FP8 error bound per row, zero tail rows, exact (unquantized) backward."""
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

  @parameterized.product(method=("rowwise@e4m3", "rowwise@e5m2"), full=(False, True))
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

  @parameterized.parameters(("rowwise",))
  def test_bf16_both_quantized(self, method):
    sizes, routing, x, ct = self._inputs(jnp.bfloat16, False)
    y_ref, dx_ref = self._run("jax", "", "", x, ct, routing)
    y, dx = self._run("wrapper", method, method, x, ct, routing)
    self._assert_quantized_close(y, y_ref, sizes.sum(axis=0), method, True, jnp.bfloat16, "forward e4m3")
    self._assert_quantized_close(dx, dx_ref, sizes.sum(axis=1), method, False, jnp.bfloat16, "backward e5m2")

  @parameterized.parameters("forward", "backward")
  def test_sent_zero_rows_round_trip_as_zero(self, leg):
    """Sent all-zero rows (e.g. masked tokens) arrive as exactly 0 with no NaN: qwix gives them scale 1."""
    sizes, routing, x, ct = self._inputs(jnp.bfloat16, False)
    zero_rows = (jnp.arange(x.shape[0]) % 3 == 0)[:, None]
    if leg == "forward":
      x, methods, valid_rows = jnp.where(zero_rows, 0, x), ("rowwise", ""), sizes.sum(axis=0)
    else:
      ct, methods, valid_rows = jnp.where(zero_rows, 0, ct), ("", "rowwise"), sizes.sum(axis=1)
    y_ref, dx_ref = self._run("jax", "", "", x, ct, routing)
    y, dx = self._run("wrapper", *methods, x, ct, routing)
    got, want = (y, y_ref) if leg == "forward" else (dx, dx_ref)
    self.assertTrue(np.all(np.isfinite(got)), f"{leg}: non-finite values")
    got3, want3 = got.reshape(_NUM_SHARDS, self.ROWS, -1), want.reshape(_NUM_SHARDS, self.ROWS, -1)
    zero_count = 0
    for s in range(_NUM_SHARDS):
      zero = np.all(want3[s, : valid_rows[s]] == 0, axis=-1)
      zero_count += int(zero.sum())
      np.testing.assert_array_equal(got3[s, : valid_rows[s]][zero], 0.0, err_msg=f"{leg}: shard {s} zero rows")
    self.assertGreater(zero_count, 0, f"{leg}: no all-zero row was sent")
    self._assert_quantized_close(got, want, valid_rows, "rowwise", leg == "forward", jnp.bfloat16, leg)

  @parameterized.parameters(
      ("", ""),
      ("rowwise", ""),
      ("", "rowwise"),
      ("rowwise@e5m2", "rowwise@e4m3"),
      ("rowwise", "rowwise"),
      ("fixed,4", "rowwise"),
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

  def _ragged_all_to_all_payloads(self, fwd_method, bwd_method, remat=False):
    """Sorted (dtype, rows, element count) of every ragged all-to-all result in the compiled bf16 fwd + bwd."""
    _, routing, x, ct = self._inputs(jnp.bfloat16, False)
    fn = _ra2a_fwd_and_grad(self.mesh, "wrapper", fwd_method, bwd_method, remat=remat)
    results = _ragged_all_to_all_results(fn.lower(x, ct, *routing).compile().as_text())
    return sorted((dt, dims[0], int(np.prod(dims))) for dt, dims in results)

  def test_hlo_wire_dtypes(self):
    """Knobs on: FP8 payloads and f32 scales (none for 'fixed'), no BF16; knobs off: BF16 only."""
    rows, n = self.ROWS, self.ROWS * self.DIM
    got = {
        methods: self._ragged_all_to_all_payloads(*methods)
        for methods in (("", ""), ("rowwise", "rowwise"), ("", "rowwise@e4m3"), ("fixed,8", ""), ("", "fixed,8"))
    }
    # XLA:TPU moves each f32 scale as a (1, 128) tile.
    scale = ("f32", rows, rows * 128)
    self.assertEqual(got[("", "")], [("bf16", rows, n)] * 2)
    self.assertEqual(got[("rowwise", "rowwise")], sorted([("f8e4m3fn", rows, n), ("f8e5m2", rows, n), scale, scale]))
    self.assertEqual(got[("", "rowwise@e4m3")], sorted([("bf16", rows, n), ("f8e4m3fn", rows, n), scale]))
    # 'fixed': the FP8 values only; the static scale never crosses the wire.
    self.assertEqual(got[("fixed,8", "")], sorted([("bf16", rows, n), ("f8e4m3fn", rows, n)]))
    self.assertEqual(got[("", "fixed,8")], sorted([("bf16", rows, n), ("f8e5m2", rows, n)]))

  def test_remat_adds_no_ragged_all_to_all(self):
    """The backward needs only the integer routing, so jax.checkpoint recomputes no ragged all-to-all."""
    for methods in (("", ""), ("rowwise", "rowwise"), ("fixed,8", "rowwise"), ("rowwise", "fixed,8")):
      self.assertEqual(
          self._ragged_all_to_all_payloads(*methods, remat=True),
          self._ragged_all_to_all_payloads(*methods),
          f"methods={methods}",
      )


def _nan_empty(shape, dtype, **kwargs):
  """Stand-in for `jax.lax.empty` that makes uninitialized memory visible as NaN."""
  del kwargs
  return jnp.full(shape, jnp.nan, dtype=dtype)


class _MoeRun:
  """One linen RoutedMoE (qwix-quantized if `cfg.quantization`) and its inputs, on shared parameters.

  Everything runs with NaN-filled `jax.lax.empty`.
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
      self.variables = variables or linen_wrappers.init_variables(self.model, rngs)

  @contextlib.contextmanager
  def context(self):
    with contextlib.ExitStack() as stack:
      for cm in (jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.cfg.logical_axis_rules)):
        stack.enter_context(cm)
      stack.enter_context(mock.patch("jax.lax.empty", side_effect=_nan_empty))
      yield

  def loss_fn(self, params, x):
    out, lb_loss, _ = self.model.apply({"params": params}, x)
    loss = jnp.mean(out.astype(jnp.float32) ** 2)
    return (loss if lb_loss is None else loss + lb_loss.astype(jnp.float32)), out

  def run(self):
    """(loss, out, {'params', 'inputs'} grads) as float32 numpy."""
    # The input gradient is the only one leg 4 (dispatch backward) reaches.
    step = jax.jit(jax.value_and_grad(self.loss_fn, (0, 1), has_aux=True))
    with self.context():
      (loss, out), (grads, x_grad) = step(self.variables["params"], self.x)
    to_np = lambda a: np.asarray(a, np.float32)  # pylint: disable=unnecessary-lambda-assignment
    return to_np(loss), to_np(out), jax.tree_util.tree_map(to_np, {"params": grads, "inputs": x_grad})


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


def _case(name, knobs, gmm_v2=False, ragged_sort=True, **overrides):
  """A RoutedMoE test case; `use_ragged_sort=True` (the production path) is primary, the XLA sort secondary."""
  return {
      "testcase_name": f"{'gmm_v2_' if gmm_v2 else ''}{name}_{'ragged' if ragged_sort else 'xla'}_sort",
      "knobs": knobs,
      "gmm_v2": gmm_v2,
      "overrides": {"use_ragged_sort": ragged_sort, **overrides},
  }


_ROWWISE_ALL = dict.fromkeys(_FWD_KNOBS + _BWD_KNOBS, "rowwise")
_ROWWISE_BWD = dict.fromkeys(_BWD_KNOBS, "rowwise")
_FIXED_FWD = {_LEG1: "fixed,64", "moe_quantize_combine_fwd_method": "fixed,64@e5m2"}
_DROPLESS = {"ragged_buffer_factor": 1.0, "moe_dropless_fallback": "layer"}
_MOE_CASES = (
    # Plain gmms (deepseek3-tiny, float32); ragged_buffer_factor -1 unless set. The XLA sort case also covers
    # check_vma, which the ragged sort does not support yet.
    _case("rowwise_all_dropless_fallback_layer", _ROWWISE_ALL, **_DROPLESS),
    _case("rowwise_all_check_vma", _ROWWISE_ALL, ragged_sort=False, check_vma=True),
    # Legs 3 and 4 only: the forward must stay bit-exact.
    _case("rowwise_bwd", _ROWWISE_BWD),
    _case("fixed_fwd_rowwise_bwd_dropless_fallback_layer", {**_FIXED_FWD, **_ROWWISE_BWD}, **_DROPLESS),
    # qwix fp8_full tokamax gmm_v2 (Mixtral, bf16): leg 1 is dequantized after the collective and quantized again.
    _case("rowwise_leg1", {_LEG1: "rowwise"}, True),
    _case("fixed_leg1", {_LEG1: "fixed,64"}, True),
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
  def test_close_to_unquantized(self, knobs, gmm_v2, overrides):
    """Finite and close to knobs off, on shared parameters and inputs, with NaN-filled jax.lax.empty."""
    name = self._testMethodName
    overrides = {"ici_expert_parallelism": _NUM_SHARDS, **overrides}
    # gmm_v2 'fixed,64' needs tokens inside its bound; 'rowwise' needs spread magnitudes to catch scale/row mix-ups.
    sigma = (0.0 if knobs.get(_LEG1, "").startswith("fixed") else 2.0) if gmm_v2 else 1.0
    off = _MoeRun(_config(f"{name}_off", gmm_v2, **overrides), sigma=sigma)
    loss_ref, out_ref, grads_ref = off.run()
    runs = {"on": _MoeRun(_config(f"{name}_on", gmm_v2, **overrides, **knobs), off.variables, sigma).run()}
    for label, (loss, out, grads) in runs.items():
      out_err = _rel_err(out, out_ref)
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


if __name__ == "__main__":
  absltest.main()
