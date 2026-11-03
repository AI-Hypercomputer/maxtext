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

"""gdn_fp8_projections: the GatedDeltaNet in_proj_qkvz and out_proj as FP8 matmuls (kernels/gdn/fp8_projections.py).

  - `fp8_projection` against a float32 fake-quantization reference of the recipe, for a narrow input (K <= N, the
    in_proj_qkvz wgrad folds the dgrad scale into x), the same with two column segments (gdn_in_proj_kernel_order's
    [qkv | z]) and a wide input (K > N, the out_proj wgrad folds the forward scale into dy): outputs, input gradient
    and kernel gradient to float32 summation order.
  - `fp8_projection` against the bfloat16 matmul: within the FP8 tolerance, and visibly quantized.
  - The split forward equals the column slices of the unsplit one.
  - `transpose_quant`: the Pallas kernel (interpret mode) gives the bits of the XLA form; dims that do not tile fall
    back to the XLA form.
  - The real Qwen3NextGatedDeltaNet in training (use_gdn_kernel=True; the kernels' CPU path), with and without
    gdn_in_proj_kernel_order: key on vs off gives the output and every gradient within the FP8 tolerance, and
    `fp8_projection` is called for in_proj_qkvz and out_proj only with the key on.
  - Config: the key defaults to False; the validator refuses it without the GDN kernel, without a Gated DeltaNet
    decoder and in explicit shard mode.
  - GSPMD on 4 CPU devices (child process): x token-sharded, the kernel sharded on its input dim, with the XLA and the
    shard_map'd Pallas (interpret) transpose-quantize: outputs and gradients match the single-device run to float32
    summation order. The same for the real layer with FSDP 4 (batch 4, gdn_in_proj_kernel_order), where the Pallas
    run also gives the bits of the XLA run.
"""

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_TRAIN, ShardMode
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.kernels.gdn import fp8_projections
from maxtext.models import qwen3
from tests.utils.test_helpers import get_test_config_path

_E4M3, _E5M2 = jnp.float8_e4m3fn, jnp.float8_e5m2
_ORIG_PROJECTION = fp8_projections.fp8_projection
_SHARDING_DEVICES = 4
_SHARDING_SENTINEL = "GDN_FP8_PROJECTIONS_SHARDING_CHECKS_PASSED"


def _fake_quant(x, axes, dtype):
  """Float32 quantize-dequantize factors (q, scale) of `x`, one absmax scale per slice reduced over `axes`."""
  xf = np.asarray(x, np.float32)
  scale = np.maximum(np.max(np.abs(xf), axis=axes, keepdims=True), 1e-30) * np.float32(1.0 / float(jnp.finfo(dtype).max))
  q = np.asarray(jnp.asarray(xf * (np.float32(1.0) / scale)).astype(dtype).astype(jnp.float32))
  return q, scale


def _reference(x, w, gs, splits):
  """Outputs, input gradient and kernel gradient of the FP8 recipe from float32 fake-quantized operands (float64)."""
  x, w = np.asarray(x, np.float32), np.asarray(w, np.float32)
  bounds = (0,) + tuple(splits) + (w.shape[1],)
  x8, sx = _fake_quant(x, (-1,), _E4M3)
  ys, dws, dx = [], [], 0.0
  for (lo, hi), g in zip(zip(bounds[:-1], bounds[1:]), gs):
    w8, sw = _fake_quant(w[:, lo:hi], (0,), _E4M3)
    ys.append(np.einsum("bsk,kn->bsn", (x8 * sx).astype(np.float64), (w8 * sw).astype(np.float64)))
    g = np.asarray(g, np.float32)
    g8, sg = _fake_quant(g * sw, (-1,), _E5M2)
    dx = dx + np.einsum("bsn,kn->bsk", (g8 * sg).astype(np.float64), w8.astype(np.float64))
    if w.shape[0] <= w.shape[1]:  # narrow input: (x * sg) per input channel against the reused dgrad operand
      a8, ca = _fake_quant(x * sg, (0, 1), _E4M3)
      dws.append(np.einsum("bsk,bsn->kn", (a8 * ca).astype(np.float64), g8.astype(np.float64)) / sw)
    else:  # wide input: (dy * sx) per output channel against the reused forward operand
      a8, ca = _fake_quant(g * sx, (0, 1), _E5M2)
      dws.append(np.einsum("bsn,bsk->kn", (a8 * ca).astype(np.float64), x8.astype(np.float64)))
  return ys, dx, np.concatenate(dws, axis=1)


def _operands(dtype, k, n, splits=(), batch=2, seq=128, seed=0):
  """x [batch, seq, k], kernel [k, n] and one output gradient per column segment, magnitudes over several decades."""
  rng = np.random.default_rng(seed)
  x = rng.standard_normal((batch, seq, k)) * np.exp(rng.uniform(-2, 2, (batch, seq, 1)))
  w = rng.standard_normal((k, n)) * 0.05 * np.exp(rng.uniform(-2, 2, (1, n)))
  bounds = (0,) + tuple(splits) + (n,)
  gs = [
      rng.standard_normal((batch, seq, hi - lo)) * 1e-3 * np.exp(rng.uniform(-2, 2, (batch, seq, 1)))
      for lo, hi in zip(bounds[:-1], bounds[1:])
  ]

  def as_dtype(a):
    return jnp.asarray(a.astype(np.float32)).astype(dtype)

  return as_dtype(x), as_dtype(w), [as_dtype(g) for g in gs]


def _fp8_value_and_grads(x, w, gs, splits=(), layout=fp8_projections.ProjectionLayout()):
  ys, vjp = jax.vjp(jax.jit(lambda xx, ww: fp8_projections.fp8_projection(xx, ww, splits, layout)), x, w)
  dx, dw = vjp(tuple(g.astype(y.dtype) for g, y in zip(gs, ys)))
  return ys, dx, dw


def _rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _cos(a, b):
  a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
  return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


_CASES = (
    ("narrow", 256, 384, ()),  # K <= N: in_proj_qkvz
    ("narrow_split", 256, 384, (256,)),  # gdn_in_proj_kernel_order: [qkv | z]
    ("wide", 384, 256, ()),  # K > N: out_proj
)


class Fp8ProjectionTest(parameterized.TestCase):

  @parameterized.named_parameters(*_CASES)
  def test_matches_fake_quant_reference(self, k, n, splits):
    x, w, gs = _operands(jnp.float32, k, n, splits)
    ys, dx, dw = _fp8_value_and_grads(x, w, gs, splits)
    ys_ref, dx_ref, dw_ref = _reference(x, w, gs, splits)
    self.assertLen(ys, len(splits) + 1)
    for i, (y, y_ref) in enumerate(zip(ys, ys_ref)):
      self.assertLess(_rel(y, y_ref), 1e-5, f"y{i}")
    self.assertLess(_rel(dx, dx_ref), 1e-5, "dx")
    self.assertLess(_rel(dw, dw_ref), 1e-5, "dw")

  def test_scales_on_the_wrong_axis_break_the_reference(self):
    """Positive control: per-channel instead of per-token input scales do not match the reference."""
    x, w, gs = _operands(jnp.float32, 256, 384)
    ys_ref, _, _ = _reference(x, w, gs, ())
    x8, sx = _fake_quant(np.asarray(x), (0, 1), _E4M3)
    w8, sw = _fake_quant(np.asarray(w), (0,), _E4M3)
    self.assertGreater(_rel(np.einsum("bsk,kn->bsn", x8 * sx, w8 * sw), ys_ref[0]), 1e-3)

  @parameterized.named_parameters(*_CASES)
  def test_close_to_bf16(self, k, n, splits):
    x, w, gs = _operands(jnp.bfloat16, k, n, splits)
    ys, dx, dw = _fp8_value_and_grads(x, w, gs, splits)
    g = jnp.concatenate(gs, axis=-1)
    y_bf16 = jnp.einsum("bsk,kn->bsn", x, w, preferred_element_type=jnp.float32)
    dx_bf16 = jnp.einsum("bsn,kn->bsk", g, w, preferred_element_type=jnp.float32)
    dw_bf16 = jnp.einsum("bsk,bsn->kn", x, g, preferred_element_type=jnp.float32)
    self.assertEqual((ys[0].dtype, dx.dtype, dw.dtype), (jnp.bfloat16,) * 3)
    # Measured: y 3.6%, dx 5.6%, dw 5.7% relative error, every cosine >= 0.998.
    for name, got, want, max_rel in (
        ("y", jnp.concatenate(ys, axis=-1), y_bf16, 0.06),
        ("dx", dx, dx_bf16, 0.1),
        ("dw", dw, dw_bf16, 0.1),
    ):
      rel = _rel(got, want)
      self.assertLess(rel, max_rel, name)
      self.assertGreater(_cos(got, want), 0.995, name)
      self.assertGreater(rel, 5e-3, f"{name}: FP8 rounding should be visible")

  def test_split_forward_matches_unsplit(self):
    x, w, gs = _operands(jnp.bfloat16, 256, 384, (256,))
    split_ys, _, _ = _fp8_value_and_grads(x, w, gs, (256,))
    (unsplit_y,), _, _ = _fp8_value_and_grads(x, w, [jnp.concatenate(gs, axis=-1)])
    np.testing.assert_array_equal(np.asarray(split_ys[0], np.float32), np.asarray(unsplit_y[..., :256], np.float32))
    np.testing.assert_array_equal(np.asarray(split_ys[1], np.float32), np.asarray(unsplit_y[..., 256:], np.float32))


class TransposeQuantTest(parameterized.TestCase):

  def test_pallas_interpret_matches_xla(self):
    rng = np.random.default_rng(1)
    a = jnp.asarray(rng.standard_normal((2, 256, 384)).astype(np.float32)).astype(jnp.bfloat16)
    rows = [jnp.asarray(np.exp(rng.uniform(-2, 2, (2, 256, 1))).astype(np.float32)) for _ in range(2)]
    cols = [jnp.asarray(np.exp(rng.uniform(-1, 1, (384,))).astype(np.float32)) for _ in range(2)]
    dtypes = [_E4M3, _E5M2]
    xla = fp8_projections.transpose_quant(a, rows, cols, dtypes, use_pallas=False)
    pallas = fp8_projections.transpose_quant(a, rows, cols, dtypes, use_pallas=True, interpret=True)
    for x_out, p_out, dtype in zip(xla, pallas, dtypes):
      self.assertEqual((x_out.shape, x_out.dtype), ((384, 2, 256), dtype))
      self.assertEqual((p_out.shape, p_out.dtype), ((384, 2, 256), dtype))
      np.testing.assert_array_equal(np.asarray(x_out, np.float32), np.asarray(p_out, np.float32))
    # The values: a * col * row in the FP8 dtype, channel dim first.
    want = np.moveaxis(np.asarray(a, np.float32) * np.asarray(cols[0]) * np.asarray(rows[0]), -1, 0)
    np.testing.assert_array_equal(np.asarray(xla[0], np.float32), np.asarray(jnp.asarray(want).astype(_E4M3), np.float32))

  @parameterized.parameters(
      (100, 1024, None), (512, 1024, 512), (16384, 1024, 1024), (12288, 2048, 2048), (384, 2048, 384)
  )
  def test_block_choice(self, dim, target, want):
    self.assertEqual(fp8_projections._tq_block(dim, target), want)  # pylint: disable=protected-access

  def test_unaligned_dims_fall_back_to_xla(self):
    a = jnp.ones((1, 100, 72), jnp.bfloat16)
    rows, cols = [jnp.ones((1, 100, 1), jnp.float32)], [jnp.full((72,), 2.0, jnp.float32)]
    with mock.patch.object(fp8_projections, "_tq_pallas_2d", side_effect=AssertionError("Pallas on unaligned dims")):
      (out,) = fp8_projections.transpose_quant(a, rows, cols, [_E4M3], use_pallas=True, interpret=True)
    self.assertEqual(out.shape, (72, 1, 100))
    np.testing.assert_array_equal(np.asarray(out, np.float32), 2.0)


_SEQ, _EMB = 128, 64
_HK, _DK, _HV, _DV = 2, 128, 4, 128


def _cfg(fp8, order, emb=_EMB, **overrides):
  return pyconfig.initialize(
      [
          None,
          get_test_config_path(),
          "run_name=gdn_fp8_projections_test",
          "dtype=bfloat16",
          "weight_dtype=float32",
          "decoder_block=qwen3_next",
          "attention=dot_product",
          f"base_emb_dim={emb}",
          "base_num_query_heads=2",
          "base_num_kv_heads=2",
          "head_dim=32",
          f"gdn_num_key_heads={_HK}",
          f"gdn_num_value_heads={_HV}",
          f"gdn_key_head_dim={_DK}",
          f"gdn_value_head_dim={_DV}",
          "gdn_conv_kernel_dim=4",
          "gdn_chunk_size=64",
          f"max_target_length={_SEQ}",
          "use_gdn_kernel=True",
          f"gdn_in_proj_kernel_order={order}",
          f"gdn_fp8_projections={fp8}",
          *[f"{key}={value}" for key, value in overrides.items()],
      ],
      skip_jax_distributed_system=True,
  )


def _layer_grads(fp8, order):
  """Output, input gradient and every parameter gradient of the real layer in training (float64 copies)."""
  cfg = _cfg(fp8, order)
  assert cfg.gdn_fp8_projections is fp8
  mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape([1] * len(cfg.mesh_axes)), cfg.mesh_axes)
  layer = qwen3.Qwen3NextGatedDeltaNet(
      config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, inputs_shape=(1, _SEQ, cfg.emb_dim), rngs=nnx.Rngs(0)
  )
  graphdef, params, rest = nnx.split(layer, nnx.Param, ...)
  rng = np.random.default_rng(7)
  hidden = jnp.asarray(rng.standard_normal((1, _SEQ, _EMB)).astype(np.float32)).astype(jnp.bfloat16)
  dy = jnp.asarray(rng.standard_normal((1, _SEQ, _EMB)).astype(np.float32))

  def loss(p, h):
    with mesh:
      out = nnx.merge(graphdef, p, rest)(h, model_mode=MODEL_MODE_TRAIN)[0]
    return jnp.sum(out.astype(jnp.float32) * dy), out

  (_, out), (g_params, g_hidden) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(params, hidden)
  named = [("output", out), ("d_input", g_hidden)]
  named += [(jax.tree_util.keystr(k), v) for k, v in jax.tree_util.tree_leaves_with_path(g_params)]
  return {n: np.asarray(jax.device_get(v), dtype=np.float64) for n, v in named}


class LayerTest(parameterized.TestCase):

  @parameterized.named_parameters(("kernel_order", True), ("plain", False))
  def test_close_to_bf16(self, order):
    ref = _layer_grads(False, order)
    got = _layer_grads(True, order)
    self.assertEqual(sorted(ref), sorted(got))
    # FP8 rounding in both projections: measured ~7% relative error on the output and ~4-10% on the gradients, every
    # cosine >= 0.994.
    self.assertLess(_rel(got["output"], ref["output"]), 0.15)
    for name in ref:
      self.assertGreater(_cos(got[name], ref[name]), 0.98, name)
      self.assertLess(_rel(got[name], ref[name]), 0.2, name)
    for name in ref:
      if "in_proj_qkvz" in name or "out_proj" in name:
        self.assertGreater(_rel(got[name], ref[name]), 1e-3, name)

  @parameterized.named_parameters(("kernel_order", True), ("plain", False))
  def test_key_selects_the_fp8_projection(self, order):
    for fp8, want_calls in ((True, 2), (False, 0)):
      jax.clear_caches()
      with mock.patch.object(fp8_projections, "fp8_projection", wraps=_ORIG_PROJECTION) as projection:
        _layer_grads(fp8, order)
      self.assertEqual(projection.call_count, want_calls, f"gdn_fp8_projections={fp8}")
      if want_calls:
        splits = sorted(str(c.args[2]) for c in projection.call_args_list)
        want_split = str((2 * _HK * _DK + _HV * _DV,)) if order else "()"
        self.assertEqual(splits, sorted(["()", want_split]))


def _validator_cfg(**overrides):
  """Minimal namespace satisfying validate_gdn_config."""
  fields = {
      "decoder_block": config_types.DecoderBlockType.QWEN3_5,
      "ici_context_parallelism": 1,
      "dcn_context_parallelism": 1,
      "ici_context_usp_ulysses_parallelism": 1,
      "dcn_context_usp_ulysses_parallelism": 1,
      "use_gdn_kernel": True,
      "ici_tensor_parallelism": 1,
      "dcn_tensor_parallelism": 1,
      "gdn_chunk_size": 64,
      "gdn_key_head_dim": 128,
      "gdn_value_head_dim": 128,
      "gdn_num_key_heads": 16,
      "gdn_num_value_heads": 64,
      "use_qk_norm_in_gdn": True,
      "max_target_length": 4096,
      "gdn_conv1d_input_in_qkv_dtype": False,
      "gdn_conv1d_dense_token_stack": False,
      "gdn_gated_norm_layout": "auto",
      "gdn_cp_mode": "seq",
      "gdn_cp_pass1_states_and_bf16_qkv": False,
      "dtype": "bfloat16",
      "packing": True,
      "gdn_in_proj_kernel_order": False,
      "gdn_fp8_projections": True,
      "shard_mode": ShardMode.AUTO,
  }
  fields.update(overrides)
  return SimpleNamespace(**fields)


class ConfigTest(parameterized.TestCase):

  def test_default_off(self):
    self.assertIs(config_types.Qwen3Next().gdn_fp8_projections, False)
    with open(os.path.join(os.path.dirname(config_types.__file__), "base.yml"), encoding="utf-8") as f:
      self.assertIn("\ngdn_fp8_projections: false\n", f.read())

  @parameterized.parameters(True, False)
  def test_accepted(self, order):
    config_types.MaxTextConfig.validate_gdn_config(_validator_cfg(gdn_in_proj_kernel_order=order), is_rl_config=False)

  @parameterized.named_parameters(
      ("without_the_kernel", {"use_gdn_kernel": False}, "requires use_gdn_kernel=True"),
      ("without_gdn", {"decoder_block": config_types.DecoderBlockType.LLAMA2}, "Gated DeltaNet"),
      ("explicit_sharding", {"shard_mode": ShardMode.EXPLICIT}, "shard_mode='auto'"),
  )
  def test_rejected(self, overrides, pattern):
    with self.assertRaisesRegex(ValueError, "gdn_fp8_projections=True") as ctx:
      config_types.MaxTextConfig.validate_gdn_config(_validator_cfg(**overrides), is_rl_config=False)
    self.assertRegex(str(ctx.exception), pattern)


class ShardingTest(absltest.TestCase):

  def test_gspmd_matches_single_device(self):
    """Runs `_sharding_checks` in a child process with 4 CPU devices (the device count is fixed at JAX start)."""
    env = os.environ.copy()
    env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_SHARDING_DEVICES}"
    env["JAX_PLATFORMS"] = "cpu"
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
    result = subprocess.run(
        [sys.executable, __file__, "--sharding_checks"], env=env, capture_output=True, text=True, check=False
    )
    report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    self.assertEqual(result.returncode, 0, report)
    self.assertIn(_SHARDING_SENTINEL, result.stdout, report)


def _sharding_checks():
  """x [8, 128, K] token-sharded over 4 devices, the kernel sharded on K: y, dx, dw equal the single-device run."""
  from jax.sharding import NamedSharding, PartitionSpec as P  # pylint: disable=import-outside-toplevel

  assert jax.device_count() >= _SHARDING_DEVICES, jax.devices()
  mesh = jax.sharding.Mesh(np.array(jax.devices()[:_SHARDING_DEVICES]), ("fsdp",))
  layout = fp8_projections.ProjectionLayout(
      gathered_sharding=NamedSharding(mesh, P(None, None)), mesh=mesh, token_spec=P("fsdp", None)
  )
  orig_tq = fp8_projections._tq_pallas_2d  # pylint: disable=protected-access

  def interpreted(*args):
    return orig_tq(*args[:-1], True)

  for (name, k, n, splits), pallas in [(case, p) for case in _CASES for p in (False, True)]:
    x, w, gs = _operands(jnp.float32, k, n, splits, batch=8)
    ref = _fp8_value_and_grads(x, w, gs, splits)
    xs = jax.device_put(x, NamedSharding(mesh, P("fsdp", None, None)))
    ws = jax.device_put(w, NamedSharding(mesh, P("fsdp", None)))
    gss = [jax.device_put(g, NamedSharding(mesh, P("fsdp", None, None))) for g in gs]
    patches = []
    if pallas:  # the shard_map'd Pallas transpose-quantize, interpreted on CPU
      patches = [
          mock.patch.object(fp8_projections, "target_platform", return_value="tpu"),
          mock.patch.object(fp8_projections, "_tq_pallas_2d", side_effect=interpreted),
      ]
    mocks = [patch.start() for patch in patches]
    try:
      with jax.set_mesh(mesh):
        got = _fp8_value_and_grads(xs, ws, gss, splits, layout)
    finally:
      for patch in patches:
        patch.stop()
    if pallas:
      assert mocks[1].call_count > 0, f"{name}: the Pallas transpose-quantize did not run"
    pairs = [(f"y{i}", y, r) for i, (y, r) in enumerate(zip(got[0], ref[0]))]
    pairs += [("dx", got[1], ref[1]), ("dw", got[2], ref[2])]
    for label, a, b in pairs:
      rel = _rel(a, b)
      print(f"{name} pallas={pallas} {label}: rel {rel:.2e}", flush=True)
      assert rel < 1e-5, (name, pallas, label, rel)
  _layer_sharding_checks(interpreted)
  print(_SHARDING_SENTINEL, flush=True)


def _layer_values(fp8, devices, hidden, dy):
  """The real layer (gdn_in_proj_kernel_order) under GSPMD with FSDP over `devices`: output and every gradient."""
  # pylint: disable=import-outside-toplevel
  from flax.linen import partitioning as nn_partitioning
  from jax.sharding import NamedSharding, PartitionSpec as P
  from maxtext.utils import maxtext_utils
  from maxtext.utils import sharding

  cfg = _cfg(fp8, True, emb=hidden.shape[-1], ici_fsdp_parallelism=_SHARDING_DEVICES, per_device_batch_size=1)
  if devices == 1:
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape([1] * len(cfg.mesh_axes)), cfg.mesh_axes)
  else:
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  rules = cfg.logical_axis_rules
  with nn_partitioning.axis_rules(rules):
    layer = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, inputs_shape=hidden.shape, rngs=nnx.Rngs(0)
    )
  graphdef, params, rest = nnx.split(layer, nnx.Param, ...)
  param_shardings = jax.tree.map(
      lambda spec: NamedSharding(mesh, P(*sharding.logical_to_mesh_axes(spec, mesh, rules))),
      nnx.to_pure_dict(nnx.get_partition_spec(params)),
      is_leaf=lambda s: isinstance(s, P),
  )
  x_sharding = sharding.create_sharding(mesh, ("activation_batch", "activation_length", "activation_embed"), rules)

  def loss(p, h):
    state = jax.tree.map(lambda v: v, params)
    nnx.replace_by_pure_dict(state, p)
    out = nnx.merge(graphdef, state, rest)(h, model_mode=MODEL_MODE_TRAIN)[0]
    return jnp.sum(out.astype(jnp.float32) * dy), out

  step = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True), in_shardings=(param_shardings, x_sharding))
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(rules):
    (_, out), (g_params, g_hidden) = step(
        jax.device_put(nnx.to_pure_dict(params), param_shardings), jax.device_put(hidden, x_sharding)
    )
  named = [("output", out), ("d_input", g_hidden)]
  named += [(jax.tree_util.keystr(k), v) for k, v in jax.tree_util.tree_leaves_with_path(g_params)]
  return {n: np.asarray(jax.device_get(v), np.float64) for n, v in named}


def _layer_sharding_checks(interpreted_tq):
  """The real layer with FSDP 4 (batch 4) vs 1 device, key on: the 1-device values to float32 summation order, with the
  XLA and the shard_map'd Pallas (interpret) transpose-quantize. The embed dim is 128 so the Pallas kernel tiles."""
  rng = np.random.default_rng(5)
  hidden = jnp.asarray(rng.standard_normal((_SHARDING_DEVICES, _SEQ, 128)), jnp.bfloat16)
  dy = jnp.asarray(rng.standard_normal((_SHARDING_DEVICES, _SEQ, 128)), jnp.float32)
  runs = {}
  for name, fp8, devices, pallas in (
      ("off_1", False, 1, False),
      ("off_4", False, _SHARDING_DEVICES, False),
      ("on_1", True, 1, False),
      ("on_4", True, _SHARDING_DEVICES, False),
      ("on_4_pallas", True, _SHARDING_DEVICES, True),
  ):
    patches = []
    if pallas:
      patches = [
          mock.patch.object(fp8_projections, "target_platform", return_value="tpu"),
          mock.patch.object(fp8_projections, "_tq_pallas_2d", side_effect=interpreted_tq),
      ]
    mocks = [patch.start() for patch in patches]
    try:
      jax.clear_caches()
      runs[name] = _layer_values(fp8, devices, hidden, dy)
    finally:
      for patch in patches:
        patch.stop()
    if pallas:
      assert mocks[1].call_count == 2, f"the Pallas transpose-quantize ran {mocks[1].call_count} times, not 2"
  for name in runs["on_4"]:
    on, off = _rel(runs["on_4"][name], runs["on_1"][name]), _rel(runs["off_4"][name], runs["off_1"][name])
    pallas = _rel(runs["on_4_pallas"][name], runs["on_4"][name])
    print(f"layer {name}: rel(4 devices, 1 device) on {on:.2e} off {off:.2e} | rel(Pallas, XLA) {pallas:.2e}", flush=True)
    assert pallas == 0.0, (name, pallas)
    if "in_proj_ba" in name:
      # Not an FP8 projection: its bf16 kernel gradient is reduced over the FSDP shards in bf16, key off too.
      assert on < max(3.0 * off, 1e-5), (name, on, off)
    else:
      assert on < 1e-4, (name, on)


if __name__ == "__main__":
  if "--sharding_checks" in sys.argv:
    _sharding_checks()
  else:
    absltest.main()
