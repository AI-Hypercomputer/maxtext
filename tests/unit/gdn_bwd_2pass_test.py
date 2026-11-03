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
"""gdn_bwd_2pass_bf16: the GDN backward kernel's precise matmuls as 2 MXU dots of bf16 operands.

The Pallas kernel runs in interpret mode on CPU, where XLA computes the `Precision.HIGH` float32 matmuls exactly, so
the comparisons bound the rounding of the 2-dot decompositions against exact float32 products, not TPU numerics.

  - `_gdn_matmul` splits: "lhs" / "rhs" keep the split operand to about 16 mantissa bits and round the other one to
    bf16 (exact when it is bf16 already), "both" is the 3-dot decomposition; float32 activations and the 1-pass
    matmuls ignore `split`; an unknown split raises.
  - `pallas_gdn_bwd_kernel(two_pass_bf16=True)` (relayout and "states" natural layout, t_inv read and rebuilt in
    VMEM): every gradient is within 3x (dA_log / dt_bias: 2.5x) of the HIGH kernel's error against the float32
    activation kernel, and close to the HIGH kernel. Negative control: splitting lhs where the kernel splits the keys
    / queries (every "rhs" split turned into "lhs") leaves dA_log / dt_bias with more than 5x HIGH's error. The
    kernel name gets `_2pass`.
  - float32 activations: the key is a no-op (same bits, same kernel name).
  - Wiring: gdn_decoupled_conv1d runs the `_2pass` backward kernel only with bwd_2pass_bf16 and gives the same forward
    bits either way; the backward rule's positional forms with 20 and 21 arguments trace the 22-argument program;
    the real Qwen3NextGatedDeltaNet runs the `_2pass` kernel only with gdn_bwd_2pass_bf16.
  - Config: the key defaults to False; the validator refuses it without the GDN kernel and without a Gated DeltaNet
    decoder.
  - 4 CPU devices, in a child process (see _sharding_checks): sequence-sharded CP through gdn_decoupled_conv1d (key
    on: forward bit-identical to key off, gradients close to key off and to the 1-device run), and the real layer
    with FSDP 4 under GSPMD (key on: the 1-device values to float32 summation order).
"""

import functools
import os
import subprocess
import sys
from types import SimpleNamespace

from flax import nnx
import jax
from jax.extend import core as jax_core
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.kernels.gdn.gdn_bwd import api as gdn_api
from maxtext.kernels.gdn.gdn_bwd import cp_gdn
from maxtext.kernels.gdn.gdn_bwd import pallas_mosaic_tpu_bwd as bwd_kernel
from maxtext.models import qwen3
from tests.utils.test_helpers import get_test_config_path

_D, _C, _KS = 128, 64, 4
_OUTPUTS = ("dy_conv", "d_b", "d_a", "d_a_log", "d_dt_bias", "dh0")
_SHARDING_DEVICES = 4
_SHARDING_SENTINEL = "GDN_BWD_2PASS_SHARDING_CHECKS_PASSED"


def _f64(x):
  return np.asarray(jnp.asarray(x, jnp.float32), np.float64)


def _rel(got, ref):
  got, ref = _f64(got), _f64(ref)
  return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))


def _cos(got, ref):
  got, ref = _f64(got).ravel(), _f64(ref).ravel()
  return float(got @ ref / max(np.linalg.norm(got) * np.linalg.norm(ref), 1e-30))


def _bf16_exact(x):
  return x.astype(jnp.bfloat16).astype(jnp.float32)


def _exact_matmul(lhs, rhs):
  return np.einsum("bij,bjk->bik", _f64(lhs), _f64(rhs))


def _pallas_names(closed_jaxpr):
  """Names of every pallas_call in a program, nested jaxprs included."""
  names = []

  def visit(jaxpr):
    for eqn in jaxpr.eqns:
      if eqn.primitive.name == "pallas_call":
        names.append(str(eqn.params.get("name", "")))
      for sub in jax_core.jaxprs_in_params(eqn.params):
        visit(sub)

  visit(closed_jaxpr.jaxpr)
  return names


# ----------------------------------------------------------------------------------------------------------------------
# _gdn_matmul
# ----------------------------------------------------------------------------------------------------------------------


def _operands():
  k = jax.random.split(jax.random.PRNGKey(0), 2)
  return jax.random.normal(k[0], (2, 64, 128), jnp.float32), jax.random.normal(k[1], (2, 128, 96), jnp.float32)


@pytest.mark.parametrize("split", ["lhs", "rhs"])
def test_split_keeps_the_split_operand_and_rounds_the_other(split):
  lhs, rhs = _operands()
  got = bwd_kernel._gdn_matmul(lhs, rhs, jnp.bfloat16, precise=True, split=split)  # pylint: disable=protected-access
  # The exact product with the other operand rounded to bf16, up to the split operand's lo half (~2^-16).
  rounded = _exact_matmul(lhs, _bf16_exact(rhs)) if split == "lhs" else _exact_matmul(_bf16_exact(lhs), rhs)
  assert _rel(got, rounded) < 2e-5
  assert _rel(got, _exact_matmul(lhs, rhs)) > 3e-4  # the other operand is visibly rounded
  # When the other operand is bf16 already, nothing is lost.
  lhs, rhs = (lhs, _bf16_exact(rhs)) if split == "lhs" else (_bf16_exact(lhs), rhs)
  got = bwd_kernel._gdn_matmul(lhs, rhs, jnp.bfloat16, precise=True, split=split)  # pylint: disable=protected-access
  assert _rel(got, _exact_matmul(lhs, rhs)) < 2e-5
  one_pass = bwd_kernel._gdn_matmul(lhs, rhs, jnp.bfloat16)  # pylint: disable=protected-access
  assert _rel(one_pass, _exact_matmul(lhs, rhs)) > 3e-4


def test_both_is_the_three_dot_decomposition():
  lhs, rhs = _operands()
  got = bwd_kernel._gdn_matmul(lhs, rhs, jnp.bfloat16, precise=True, split="both")  # pylint: disable=protected-access
  assert _rel(got, _exact_matmul(lhs, rhs)) < 2e-5


@pytest.mark.parametrize("split", bwd_kernel.MATMUL_SPLITS)
def test_split_is_ignored_off_the_precise_bf16_path(split):
  lhs, rhs = _operands()
  matmul = bwd_kernel._gdn_matmul  # pylint: disable=protected-access
  np.testing.assert_array_equal(
      matmul(lhs, rhs, jnp.float32, precise=True, split=split), matmul(lhs, rhs, jnp.float32, precise=True)
  )
  np.testing.assert_array_equal(matmul(lhs, rhs, jnp.bfloat16, split=split), matmul(lhs, rhs, jnp.bfloat16))


def test_unknown_split_raises():
  lhs, rhs = _operands()
  with pytest.raises(ValueError, match="split must be one of"):
    bwd_kernel._gdn_matmul(lhs, rhs, jnp.bfloat16, precise=True, split="hi")  # pylint: disable=protected-access


# ----------------------------------------------------------------------------------------------------------------------
# pallas_gdn_bwd_kernel
# ----------------------------------------------------------------------------------------------------------------------


def _kernel_inputs(s, nk, nv, dtype, seed):
  dim = 2 * nk * _D + nv * _D
  nc = s // _C
  k = jax.random.split(jax.random.PRNGKey(seed), 9)
  return {
      "qkv_conv": jax.random.normal(k[0], (1, s, dim), jnp.float32).astype(dtype),
      "b": jax.random.normal(k[1], (1, s, nv), jnp.float32).astype(jnp.bfloat16),
      "a": (jax.random.normal(k[2], (1, s, nv), jnp.float32) - 2.0).astype(jnp.bfloat16),
      "a_log": jax.random.normal(k[3], (nv,), jnp.float32) * 0.3,
      "dt_bias": jax.random.normal(k[4], (nv,), jnp.float32) * 0.3,
      "do": jax.random.normal(k[5], (1, s, nv, _D), jnp.float32).astype(dtype),
      "chunk_states": jax.random.normal(k[6], (1, nc, nv, _D, _D), jnp.float32) * 0.05,
      "t_inv": jnp.tril(jax.random.normal(k[7], (1, nc, nv, _C, _C), jnp.float32) * 0.05, -1) + jnp.eye(_C),
      "d_recurrent_state": jax.random.normal(k[8], (1, nv, _D, _D), jnp.float32) * 0.05,
  }


def _run_kernel(x, nv, *, two_pass, natural="off", head_tile=None, recompute=False, dtype=None):
  """The Pallas GDN backward kernel in interpret mode: (outputs, pallas_call names)."""
  fn = functools.partial(
      bwd_kernel.pallas_gdn_bwd_kernel,
      num_v_heads=nv,
      kq_head_dim=_D,
      v_head_dim=_D,
      chunk_size=_C,
      use_qk_norm_in_gdn=True,
      head_tile=head_tile,
      d_recurrent_state=x["d_recurrent_state"],
      return_dh0=True,
      interpret=True,
      recompute_t_inv=recompute,
      natural_layout=natural,
      two_pass_bf16=two_pass,
  )
  args = [x[n] for n in ("qkv_conv", "b", "a", "a_log", "dt_bias", "do", "chunk_states")]
  args.append(None if recompute else x["t_inv"])
  if dtype is not None:  # the float32-activation reference: every kernel matmul at HIGHEST
    args[0], args[5] = args[0].astype(dtype), args[5].astype(dtype)
  return jax.jit(fn)(*args), _pallas_names(jax.make_jaxpr(fn)(*args))


_KERNEL_CASES = {
    # nk, nv, head_tile, natural, recompute, seed
    "relayout_t_inv_read": (4, 16, None, "off", False, 0),
    "natural_states_t_inv_rebuilt": (4, 16, 8, "states", True, 1),
    "small_heads": (2, 8, None, "off", False, 2),
}


@pytest.mark.parametrize("case", sorted(_KERNEL_CASES))
def test_two_pass_kernel_keeps_the_high_kernel_accuracy(case):
  nk, nv, head_tile, natural, recompute, seed = _KERNEL_CASES[case]
  x = _kernel_inputs(192, nk, nv, jnp.bfloat16, seed)
  kw = {"natural": natural, "head_tile": head_tile, "recompute": recompute}
  ref, _ = _run_kernel(x, nv, two_pass=False, dtype=jnp.float32, **kw)
  high, high_names = _run_kernel(x, nv, two_pass=False, **kw)
  got, got_names = _run_kernel(x, nv, two_pass=True, **kw)
  assert high_names and not any(n.endswith("_2pass") for n in high_names)
  assert got_names == [f"{n}_2pass" for n in high_names]
  assert len(got) == len(high) == len(ref) == len(_OUTPUTS)
  # Measured on CPU, where HIGH is exact: 1.4-2.5x HIGH's error against the float32 reference (dA_log / dt_bias up to
  # 2x), cos(2-pass, HIGH) >= 0.99999. On TPU 8i (Qwen3.5-397B shapes) dA_log / dt_bias match HIGH's error and the
  # others are within 1.6x.
  for name, g, h, r in zip(_OUTPUTS, got, high, ref):
    assert g.shape == h.shape and g.dtype == h.dtype, name
    bound = 2.5 if name in ("d_a_log", "d_dt_bias") else 3.0
    assert _rel(g, r) < bound * _rel(h, r), (name, _rel(g, r), _rel(h, r))
    assert _cos(g, h) > 0.9999, name
  assert max(_rel(g, h) for g, h in zip(got[:5], high[:5])) > 1e-4  # not the HIGH kernel


def test_lhs_splits_on_the_key_matmuls_lose_the_gating_adjoint(monkeypatch):
  """Negative control for the per-call-site split: lhs splits everywhere break dA_log / dt_bias."""
  real = bwd_kernel._gdn_matmul  # pylint: disable=protected-access

  def lhs_everywhere(lhs, rhs, compute_dtype=jnp.bfloat16, keep_lhs_fp32=False, precise=False, split=None):
    return real(lhs, rhs, compute_dtype, keep_lhs_fp32, precise, "lhs" if split == "rhs" else split)

  x = _kernel_inputs(192, 4, 16, jnp.bfloat16, 0)
  ref, _ = _run_kernel(x, 16, two_pass=False, dtype=jnp.float32)
  high, _ = _run_kernel(x, 16, two_pass=False)
  monkeypatch.setattr(bwd_kernel, "_gdn_matmul", lhs_everywhere)
  jax.clear_caches()
  try:
    bad, _ = _run_kernel(x, 16, two_pass=True)
  finally:
    monkeypatch.undo()
    jax.clear_caches()
  for i in (_OUTPUTS.index("d_a_log"), _OUTPUTS.index("d_dt_bias")):
    assert _rel(bad[i], ref[i]) > 5.0 * _rel(high[i], ref[i]), _OUTPUTS[i]


def test_float32_activations_ignore_the_key():
  x = _kernel_inputs(128, 2, 8, jnp.float32, 3)
  ref, ref_names = _run_kernel(x, 8, two_pass=False)
  got, got_names = _run_kernel(x, 8, two_pass=True)
  assert got_names == ref_names == ["gdn_bwd_kernel"]
  for r, g in zip(ref, got):
    np.testing.assert_array_equal(np.asarray(g), np.asarray(r))


# ----------------------------------------------------------------------------------------------------------------------
# Wiring: gdn_decoupled_conv1d, its backward rule, the real layer
# ----------------------------------------------------------------------------------------------------------------------

_NK, _NV = 2, 4
_DIM = 2 * _NK * _D + _NV * _D


def _conv1d_inputs(s, seed=7):
  k = jax.random.split(jax.random.PRNGKey(seed), 7)
  return (
      jax.random.normal(k[0], (1, s, _DIM), jnp.float32).astype(jnp.bfloat16),
      jax.random.normal(k[1], (1, s, _NV), jnp.float32).astype(jnp.bfloat16),
      (jax.random.normal(k[2], (1, s, _NV), jnp.float32) - 4.0).astype(jnp.bfloat16),
      jax.random.normal(k[3], (_KS, 1, _DIM), jnp.float32).astype(jnp.bfloat16) * 0.1,
      jax.random.normal(k[4], (_NV,), jnp.float32) * 0.2,
      jax.random.normal(k[5], (_NV,), jnp.float32) * 0.2,
      jax.random.normal(k[6], (1, s, _NV, _D), jnp.float32).astype(jnp.bfloat16),
  )


def _gdn(two_pass, cp_axis_name=None):
  """gdn_decoupled_conv1d's output as a function of (qkv, b, a, conv weight, A_log, dt_bias)."""

  def fn(qkv, b, a, cw, a_log, dt_bias):
    out, _ = gdn_api.gdn_decoupled_conv1d(
        qkv,
        b,
        a,
        cw,
        None,
        a_log,
        dt_bias,
        None,
        None,
        _NK,
        _NV,
        _D,
        _D,
        _KS,
        _C,
        True,
        jnp.float32,
        cp_axis_name,
        bwd_2pass_bf16=two_pass,
    )
    return out

  return fn


def test_gdn_decoupled_conv1d_runs_the_two_pass_backward_only_with_the_key():
  *args, cot = _conv1d_inputs(2 * _C)
  outs, kernels = {}, {}
  for two_pass in (False, True):
    fn = _gdn(two_pass)
    grad = jax.grad(lambda *a, f=fn: jnp.sum(f(*a).astype(jnp.float32) * cot.astype(jnp.float32)), argnums=(0, 1, 2))
    outs[two_pass] = np.asarray(jax.jit(fn)(*args))
    kernels[two_pass] = _pallas_names(jax.make_jaxpr(grad)(*args))
  np.testing.assert_array_equal(outs[True], outs[False])  # the forward does not depend on the key
  assert kernels[False] == ["gdn_bwd_kernel"]
  assert kernels[True] == ["gdn_bwd_kernel_2pass"]


def test_backward_rule_positional_forms():
  """The rule's positional forms without the trailing bwd_2pass_bf16 (21 args) or fwd_head_group_size (20) trace the
  program of the full 22-argument form with the key off."""
  nondiff = (_NK, _NV, _D, _D, _KS, _C, True, jnp.float32)
  prec = cp_gdn.DEFAULT_PRECISION
  head = (None, prec, False, False, False, False, False, False, False, False)  # cp_axis_name .. conv_natural_layout
  rule = gdn_api._gdn_decoupled_conv1d_bwd  # pylint: disable=protected-access
  forms = {
      20: lambda res, cot: rule(*nondiff, *head, res, cot),
      21: lambda res, cot: rule(*nondiff, *head, 0, res, cot),
      22: lambda res, cot: rule(*nondiff, *head, 0, False, res, cot),
      "22_on": lambda res, cot: rule(*nondiff, *head, 0, True, res, cot),
  }
  *args, cot = _conv1d_inputs(2 * _C)

  def program(form):
    def f(qkv, b, a, cw, a_log, dt_bias, do):
      _, res = gdn_api._gdn_decoupled_conv1d_fwd(  # pylint: disable=protected-access
          qkv, b, a, cw, None, a_log, dt_bias, None, None, *nondiff
      )
      return forms[form](res, (do, (None, None)))

    return jax.make_jaxpr(f)(*args, cot)

  ref = program(22)
  assert str(program(20)) == str(ref)
  assert str(program(21)) == str(ref)
  assert _pallas_names(ref) == ["gdn_bwd_kernel"]
  assert _pallas_names(program("22_on")) == ["gdn_bwd_kernel_2pass"]


_LAYER_SEQ, _LAYER_EMB = 128, 64


def _layer_cfg(two_pass, **overrides):
  """A small Qwen3-Next GDN layer config on the GDN kernel path."""
  kwargs = {
      "run_name": "gdn_bwd_2pass_test",
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
      "dtype": "bfloat16",
      "weight_dtype": "float32",
      "decoder_block": "qwen3_next",
      "attention": "dot_product",
      "base_emb_dim": _LAYER_EMB,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 2,
      "head_dim": 32,
      "gdn_num_key_heads": _NK,
      "gdn_num_value_heads": _NV,
      "gdn_key_head_dim": _D,
      "gdn_value_head_dim": _D,
      "gdn_conv_kernel_dim": _KS,
      "gdn_chunk_size": _C,
      "max_target_length": _LAYER_SEQ,
      "use_gdn_kernel": True,
      "gdn_bwd_2pass_bf16": two_pass,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


def test_layer_runs_the_two_pass_backward_only_with_the_key():
  """The real layer's gradient program runs the `_2pass` backward kernel exactly when gdn_bwd_2pass_bf16 is on."""
  for two_pass in (False, True):
    cfg = _layer_cfg(two_pass)
    assert cfg.gdn_bwd_2pass_bf16 is two_pass
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape([1] * len(cfg.mesh_axes)), cfg.mesh_axes)
    layer = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, inputs_shape=(1, _LAYER_SEQ, _LAYER_EMB), rngs=nnx.Rngs(0)
    )
    graphdef, params, rest = nnx.split(layer, nnx.Param, ...)

    def loss(p, h, graphdef=graphdef, rest=rest, mesh=mesh):
      with mesh:
        return jnp.sum(nnx.merge(graphdef, p, rest)(h, model_mode=MODEL_MODE_TRAIN)[0].astype(jnp.float32))

    hidden = jnp.ones((1, _LAYER_SEQ, _LAYER_EMB), jnp.bfloat16)
    names = [n for n in _pallas_names(jax.make_jaxpr(jax.grad(loss))(params, hidden)) if "gdn_bwd" in n]
    assert names and all(n.endswith("_2pass") == two_pass for n in names), (two_pass, names)


# ----------------------------------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------------------------------


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
      "gdn_bwd_2pass_bf16": True,
  }
  fields.update(overrides)
  return SimpleNamespace(**fields)


def test_config_default_off():
  assert config_types.Qwen3Next().gdn_bwd_2pass_bf16 is False
  with open(os.path.join(os.path.dirname(config_types.__file__), "base.yml"), encoding="utf-8") as f:
    assert "\ngdn_bwd_2pass_bf16: false\n" in f.read()


def test_config_validation():
  config_types.MaxTextConfig.validate_gdn_config(_validator_cfg(), is_rl_config=False)
  config_types.MaxTextConfig.validate_gdn_config(
      _validator_cfg(use_gdn_kernel=False, gdn_bwd_2pass_bf16=False), is_rl_config=False
  )
  with pytest.raises(ValueError, match="gdn_bwd_2pass_bf16=True requires use_gdn_kernel=True"):
    config_types.MaxTextConfig.validate_gdn_config(_validator_cfg(use_gdn_kernel=False), is_rl_config=False)
  with pytest.raises(ValueError, match="gdn_bwd_2pass_bf16=True requires a decoder with Gated DeltaNet layers"):
    config_types.MaxTextConfig.validate_gdn_config(
        _validator_cfg(decoder_block=config_types.DecoderBlockType.LLAMA2), is_rl_config=False
    )


# ----------------------------------------------------------------------------------------------------------------------
# 4 devices (child process)
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_sharded_runs_match_single_device():
  """Runs `_sharding_checks` in a child process with 4 CPU devices (the device count is fixed at JAX start)."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_SHARDING_DEVICES}"
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert result.returncode == 0, report
  assert _SHARDING_SENTINEL in result.stdout, report


def _cp_checks():
  """gdn_decoupled_conv1d on 4 sequence-sharded CP ranks: key on vs key off, and vs the 1-device run with the key on."""
  from jax.sharding import PartitionSpec as P  # pylint: disable=import-outside-toplevel

  mesh = jax.sharding.Mesh(np.array(jax.devices()[:_SHARDING_DEVICES]), ("cp",))
  *args, cot = _conv1d_inputs(_SHARDING_DEVICES * 2 * _C)  # 2 chunks per rank
  tok, rep = P(None, "cp", None), P()

  def values(two_pass, cp):
    def fwd_bwd(qkv, b, a, cw, a_log, dt_bias, cot_x):
      out, vjp = jax.vjp(_gdn(two_pass, "cp" if cp else None), qkv, b, a, cw, a_log, dt_bias)
      grads = vjp(cot_x)
      if cp:  # each rank holds its tokens' part of the replicated parameters' gradients
        grads = grads[:3] + tuple(jax.lax.psum(g, "cp") for g in grads[3:])
      return (out, *grads)

    if cp:
      fwd_bwd = jax.shard_map(
          fwd_bwd,
          mesh=mesh,
          in_specs=(tok, tok, tok, rep, rep, rep, P(None, "cp", None, None)),
          out_specs=(P(None, "cp", None, None), tok, tok, tok, rep, rep, rep),
          check_vma=False,
      )
    return jax.jit(fwd_bwd)(*args, cot)

  names = ("out", "d_qkv", "d_b", "d_a", "d_conv_weight", "d_a_log", "d_dt_bias")
  off, on, on_1dev, off_1dev = values(False, True), values(True, True), values(True, False), values(False, False)
  np.testing.assert_array_equal(np.asarray(on[0]), np.asarray(off[0]))  # the forward does not depend on the key
  for name, a, b, c, d in zip(names[1:], on[1:], off[1:], on_1dev[1:], off_1dev[1:]):
    vs_off, vs_1dev, high_vs_1dev = _rel(a, b), _rel(a, c), _rel(b, d)
    print(
        f"cp: {name} rel(on, off) {vs_off:.2e} | rel(cp4, 1 device): on {vs_1dev:.2e}, off {high_vs_1dev:.2e}",
        flush=True,
    )
    assert 0.0 < vs_off < 1e-2, (name, vs_off)
    assert vs_1dev < 1e-2, (name, vs_1dev)


def _fsdp_checks():
  """The real layer with FSDP 4 under GSPMD, key on: the 1-device values to float32 summation order."""
  from flax.linen import partitioning as nn_partitioning  # pylint: disable=import-outside-toplevel
  from jax.sharding import NamedSharding, PartitionSpec as P  # pylint: disable=import-outside-toplevel
  from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
  from maxtext.utils import sharding  # pylint: disable=import-outside-toplevel

  batch = _SHARDING_DEVICES
  rng = np.random.default_rng(3)
  hidden = jnp.asarray(rng.standard_normal((batch, _LAYER_SEQ, _LAYER_EMB)), jnp.bfloat16)
  dy = jnp.asarray(rng.standard_normal((batch, _LAYER_SEQ, _LAYER_EMB)), jnp.float32)

  def values(two_pass, devices):
    cfg = _layer_cfg(two_pass, ici_fsdp_parallelism=_SHARDING_DEVICES, per_device_batch_size=1)
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
    logical = nnx.to_pure_dict(nnx.get_partition_spec(params))
    param_shardings = jax.tree.map(
        lambda spec: NamedSharding(mesh, P(*sharding.logical_to_mesh_axes(spec, mesh, rules))),
        logical,
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
      p = jax.device_put(nnx.to_pure_dict(params), param_shardings)
      h = jax.device_put(hidden, x_sharding)
      kernels = [n for n in _pallas_names(jax.make_jaxpr(step)(p, h)) if "gdn_bwd" in n]
      (_, out), (g_params, g_hidden) = step(p, h)
    assert kernels and all(n.endswith("_2pass") == two_pass for n in kernels), (two_pass, devices, kernels)
    named = [("output", out), ("d_input", g_hidden)]
    named += [(jax.tree_util.keystr(k), v) for k, v in jax.tree_util.tree_leaves_with_path(g_params)]
    return {n: np.asarray(jax.device_get(v), np.float64) for n, v in named}

  on_4dev, on_1dev = values(True, _SHARDING_DEVICES), values(True, 1)
  off_4dev, off_1dev = values(False, _SHARDING_DEVICES), values(False, 1)
  assert sorted(on_4dev) == sorted(on_1dev) == sorted(off_4dev) == sorted(off_1dev)
  for name in on_4dev:
    vs_1dev, off_vs_1dev = _rel(on_4dev[name], on_1dev[name]), _rel(off_4dev[name], off_1dev[name])
    print(
        f"fsdp: {name} rel(4 devices, 1 device): on {vs_1dev:.2e}, off {off_vs_1dev:.2e} |"
        f" rel(on, off) {_rel(on_4dev[name], off_4dev[name]):.2e}",
        flush=True,
    )
    if "proj" in name:
      # The bf16 DenseGeneral kernels' gradients are bf16 dots, reduced over the FSDP shards in bf16 (key off too).
      assert vs_1dev < max(3.0 * off_vs_1dev, 1e-5), (name, vs_1dev, off_vs_1dev)
    else:  # the GDN kernel's own outputs and their float32 parameter gradients
      assert vs_1dev < 1e-5, (name, vs_1dev)
  assert max(_rel(on_4dev[n], off_4dev[n]) for n in on_4dev if n != "output") > 1e-4  # the key reaches the kernel


def _sharding_checks():
  assert jax.device_count() >= _SHARDING_DEVICES, jax.devices()
  _cp_checks()
  _fsdp_checks()
  print(_SHARDING_SENTINEL, flush=True)


if __name__ == "__main__":
  _sharding_checks()
