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

"""fp8_fake_quant_hoist: fp8_moe_fake_quant with the absmax hoisted out of the layer scan, optionally fp8 gathers.

  - Values and programs, in a child process with 8 CPU devices (see main): a shrunk Qwen3.5 through the production
    model path (decoder layer scan under remat, RoutedMoE.sparse_matmul with prefused unsplit wi, ring of experts,
    chunk pipeline, expert-weight prefetch barrier, forced routing) on the 397B trainer's cp-as-ep rules, so every
    expert weight is FSDP-sharded on its reduced axis. With fp8_fake_quant_hoist=absmax and =gather8 the loss, the
    logits and every parameter gradient are bit-identical to =off. The compiled programs show that the hoist ran:
    with it on, no fake-quant max all-reduce is left inside the layer loop (one for the stack is outside it), and
    with gather8 the loop's expert-weight all-gathers are 8-bit floats (XLA:CPU widens them to f16, as it widens
    the bf16 gathers of =off to f32).
  - In this process: the quantizations.py functions (hoisted absmax, stacked absmax, fp8 codes with -0.0 stored as
    +0.0 and their straight-through dequantization), the mode resolution from the config key and the
    PARAMBOLE_FQ_HOIST env var, the raises for a bad env value, a key/env conflict and a mode the config cannot
    honour, and the config key itself.
"""

import os
import re
import subprocess
import sys

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path

_PASS_TOKEN = "FP8_FAKE_QUANT_HOIST_CHECKS_PASSED"
_KEY = "fp8_fake_quant_hoist"
_ENV = "PARAMBOLE_FQ_HOIST"
_NUM_DEVICES = 8
_FP8_NEG_ZERO = 0x80  # float8_e4m3fn -0.0
_BF16_NEG_ZERO = 0x8000


def _bits(x):
  """The array's raw bits as unsigned integers, so equality means bit-identical (-0.0 != +0.0)."""
  x = np.asarray(jax.device_get(x))
  return x.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[x.dtype.itemsize])


def _cfg(*argv, **overrides):
  kwargs = {
      "run_name": "fp8_fake_quant_hoist",
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path(), *argv], **kwargs)


class _EnvUnset(parameterized.TestCase):
  """Runs every test with PARAMBOLE_FQ_HOIST unset, and restores it afterwards."""

  def setUp(self):
    super().setUp()
    saved = os.environ.pop(_ENV, None)
    self.addCleanup(lambda: os.environ.pop(_ENV, None) if saved is None else os.environ.__setitem__(_ENV, saved))


def _crafted_weight(rng, e=4, d=64, c=32):
  """A bf16 (experts, in, out) weight with entries that round to the fp8 code -0.0, and exact +0.0 and -0.0."""
  w = rng.standard_normal((e, d, c)).astype(np.float32) * 0.02
  w[:, 0, :] = 0.5 * np.sign(rng.standard_normal((e, c)))  # large entries fix each channel's absmax
  tiny = rng.random((e, d, c)) < 0.05
  w[tiny] = -rng.random(int(tiny.sum())).astype(np.float32) * 1e-7  # tiny negatives -> fp8 code -0.0
  w[:, 5, :7] = 0.0
  w[:, 6, :7] = -0.0
  return jnp.asarray(w, jnp.bfloat16)


class QuantizationFunctionsTest(_EnvUnset):
  """The quantizations.py pieces of the hoist, single device: same bits as fake_quantize_weight_per_channel."""

  def test_hoisted_absmax_is_bit_identical(self):
    w = _crafted_weight(np.random.default_rng(0))
    absmax = jnp.max(jnp.abs(w), axis=(1,), keepdims=True)
    hoisted = jax.jit(lambda w, a: quantizations.fake_quantize_weight_per_channel(w, absmax=a))(w, absmax)
    inline = jax.jit(quantizations.fake_quantize_weight_per_channel)(w)
    np.testing.assert_array_equal(_bits(hoisted), _bits(inline))

  def test_absmax_of_the_wrong_shape_raises(self):
    w = _crafted_weight(np.random.default_rng(0))
    with self.assertRaisesRegex(ValueError, "absmax must be the per-channel abs-max"):
      quantizations.quantize_weight_per_channel(w, absmax=jnp.max(jnp.abs(w), axis=(2,), keepdims=True))

  def test_stacked_absmax_matches_each_layer(self):
    rng = np.random.default_rng(1)
    num_layers = 3
    stacked = jnp.asarray(rng.standard_normal((4, num_layers, 64, 32)), jnp.bfloat16)  # param_scan_axis = 1
    got = quantizations.stacked_per_channel_absmax(stacked, layer_axis=1)
    want = jnp.stack([jnp.max(jnp.abs(stacked[:, i]), axis=(1,), keepdims=True) for i in range(num_layers)])
    self.assertEqual(got.shape, (num_layers, 4, 1, 32))
    np.testing.assert_array_equal(_bits(got), _bits(want))

  def test_codes_store_negative_zero_as_positive_zero(self):
    w = _crafted_weight(np.random.default_rng(2))
    raw, raw_scale = jax.jit(quantizations.quantize_weight_per_channel)(w)
    codes, scale = jax.jit(quantizations.quantize_weight_codes_per_channel)(w)
    raw_bits, code_bits = _bits(raw), _bits(codes)
    # The input exercises the case: the plain quantization emits -0.0 codes.
    self.assertGreater(int(np.sum(raw_bits == _FP8_NEG_ZERO)), 0)
    self.assertEqual(int(np.sum(code_bits == _FP8_NEG_ZERO)), 0)
    np.testing.assert_array_equal(np.where(raw_bits == _FP8_NEG_ZERO, 0, raw_bits), code_bits)
    np.testing.assert_array_equal(_bits(scale), _bits(raw_scale))

  def test_dequantized_codes_are_the_fake_quantized_weight(self):
    w = _crafted_weight(np.random.default_rng(3))
    fake = jax.jit(quantizations.fake_quantize_weight_per_channel)(w)
    self.assertEqual(int(np.sum(_bits(fake) == _BF16_NEG_ZERO)), 0)

    def dequant(w, canonical):
      if canonical:
        codes, scale = quantizations.quantize_weight_codes_per_channel(w)
      else:
        codes, scale = quantizations.quantize_weight_per_channel(w)
      return quantizations.dequantize_weight_codes_ste(w, codes, scale)

    np.testing.assert_array_equal(_bits(jax.jit(lambda w: dequant(w, True))(w)), _bits(fake))
    # Control: the same dequantization of the raw codes keeps -0.0 and differs from the fake-quantized weight.
    raw = _bits(jax.jit(lambda w: dequant(w, False))(w))
    self.assertGreater(int(np.sum(raw == _BF16_NEG_ZERO)), 0)
    self.assertGreater(int(np.sum(raw != _bits(fake))), 0)

  def test_straight_through_gradient(self):
    rng = np.random.default_rng(4)
    w = _crafted_weight(rng)
    ct = jnp.asarray(rng.standard_normal(w.shape), jnp.bfloat16)

    def via_codes(w):
      codes, scale = quantizations.quantize_weight_codes_per_channel(jax.lax.stop_gradient(w))
      return quantizations.dequantize_weight_codes_ste(w, codes, scale)

    def loss_of(f):
      return lambda w: jnp.sum((f(w) * ct).astype(jnp.float32))

    g_codes = jax.jit(jax.grad(loss_of(via_codes)))(w)
    g_fake = jax.jit(jax.grad(loss_of(quantizations.fake_quantize_weight_per_channel)))(w)
    np.testing.assert_array_equal(_bits(g_codes), _bits(g_fake))


class _Cfg:
  """A config with some attributes overridden."""

  def __init__(self, base, **overrides):
    self._base, self._overrides = base, overrides

  def __getattr__(self, name):
    return self._overrides[name] if name in self._overrides else getattr(self._base, name)


class ModeTest(_EnvUnset):
  """fp8_fake_quant_hoist_mode: the config key, or PARAMBOLE_FQ_HOIST while the key is off."""

  @parameterized.parameters(
      ("off", None, ""),
      ("off", "0", ""),
      ("off", "absmax", "absmax"),
      ("off", "1", "gather8"),
      ("absmax", None, "absmax"),
      ("gather8", None, "gather8"),
      ("gather8", "0", "gather8"),
      ("gather8", "1", "gather8"),
      ("absmax", "absmax", "absmax"),
  )
  def test_mode(self, key, env, want):
    if env is not None:
      os.environ[_ENV] = env
    self.assertEqual(moe.fp8_fake_quant_hoist_mode(_Cfg(None, fp8_fake_quant_hoist=key)), want)

  @parameterized.parameters("2", "gather8", "yes")
  def test_bad_env_value_raises(self, env):
    os.environ[_ENV] = env
    with self.assertRaisesRegex(ValueError, _ENV):
      moe.fp8_fake_quant_hoist_mode(_Cfg(None, fp8_fake_quant_hoist="off"))

  @parameterized.parameters(("absmax", "1"), ("gather8", "absmax"))
  def test_key_and_env_conflict_raises(self, key, env):
    os.environ[_ENV] = env
    with self.assertRaisesRegex(ValueError, f"conflicts with {_ENV}"):
      moe.fp8_fake_quant_hoist_mode(_Cfg(None, fp8_fake_quant_hoist=key))

  def test_bad_key_value_raises(self):
    with self.assertRaisesRegex(ValueError, "expected off, absmax or gather8"):
      moe.fp8_fake_quant_hoist_mode(_Cfg(None, fp8_fake_quant_hoist="bogus"))


class EnabledTest(_EnvUnset):
  """fp8_fake_quant_hoist_enabled raises, at trace time, for a mode the config cannot honour."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.base = _cfg(fp8_moe_fake_quant=True, sparse_matmul=True)

  def test_off(self):
    self.assertEqual(moe.fp8_fake_quant_hoist_enabled(_Cfg(self.base, fp8_fake_quant_hoist="off")), "")

  @parameterized.parameters("absmax", "gather8")
  def test_on(self, mode):
    self.assertEqual(moe.fp8_fake_quant_hoist_enabled(_Cfg(self.base, fp8_fake_quant_hoist=mode)), mode)

  def test_absmax_does_not_need_sparse_matmul(self):
    cfg = _Cfg(self.base, fp8_fake_quant_hoist="absmax", sparse_matmul=False)
    self.assertEqual(moe.fp8_fake_quant_hoist_enabled(cfg), "absmax")

  @parameterized.parameters(
      ("absmax", {"fp8_moe_fake_quant": False}, "fp8_moe_fake_quant=true"),
      ("gather8", {"fp8_moe_fake_quant": False}, "fp8_moe_fake_quant=true"),
      ("gather8", {"sparse_matmul": False}, "sparse_matmul=true"),
      ("gather8", {"moe_fsdp_use_two_stage_all_gather": True}, "moe_fsdp_use_two_stage_all_gather=false"),
      ("gather8", {"moe_expert_weight_prefetch": "forward_backward"}, "other than forward_backward"),
      ("absmax", {"parameter_memory_host_offload": True}, "parameter_memory_host_offload=false"),
  )
  def test_unmet_requirement_raises(self, mode, overrides, message):
    with self.assertRaisesRegex(ValueError, message):
      moe.fp8_fake_quant_hoist_enabled(_Cfg(self.base, fp8_fake_quant_hoist=mode, **overrides))

  def test_env_selected_mode_is_checked_too(self):
    os.environ[_ENV] = "1"
    with self.assertRaisesRegex(ValueError, "sparse_matmul=true"):
      moe.fp8_fake_quant_hoist_enabled(_Cfg(self.base, fp8_fake_quant_hoist="off", sparse_matmul=False))


class ConfigKeyTest(_EnvUnset):
  """The fp8_fake_quant_hoist config key: off by default, three values, a bare `off` parses as 'off'."""

  def test_defined(self):
    self.assertIn(_KEY, config_types.MoEGeneral.model_fields)
    self.assertEqual(_cfg().fp8_fake_quant_hoist, "off")

  @parameterized.parameters("off", "absmax", "gather8")
  def test_accepted(self, value):
    self.assertEqual(_cfg(fp8_fake_quant_hoist=value).fp8_fake_quant_hoist, value)

  def test_bare_off_on_the_command_line(self):
    self.assertEqual(_cfg(f"{_KEY}=off").fp8_fake_quant_hoist, "off")
    self.assertEqual(_cfg(f"{_KEY}=gather8").fp8_fake_quant_hoist, "gather8")

  def test_bad_value_rejected(self):
    self.assertIn(_KEY, config_types.MoEGeneral.model_fields)  # so that an undefined key does not pass below
    with self.assertRaisesRegex(Exception, _KEY):
      _cfg(fp8_fake_quant_hoist="bogus")


def _child_env():
  env = os.environ.copy()
  # As the CPU proofs ran: no excess precision, so both programs round bf16 intermediates the same way.
  env["XLA_FLAGS"] = (
      f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_NUM_DEVICES}"
      " --xla_allow_excess_precision=false"
  ).strip()
  env["JAX_PLATFORMS"] = "cpu"
  env.pop(_ENV, None)
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  return env


@pytest.mark.cpu_only
def test_model_in_a_child_process():
  """Runs `main` below in a fresh process with 8 CPU devices: off vs absmax vs gather8 on a shrunk Qwen3.5."""
  result = subprocess.run([sys.executable, __file__], env=_child_env(), capture_output=True, text=True, check=False)
  print(result.stdout)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


# ---- child process ----

# Eight layers = two iterations of the scanned 4-layer block: XLA removes a while loop with one trip, and the
# program checks look for the fake-quant collectives inside the layer loop.
_EXPERTS, _TOP_K, _EMB, _MOE_MLP, _SEQ, _VOCAB, _LAYERS = 16, 4, 256, 128, 64, 512, 8

# A shrunk Qwen3.5-397B-A17B (3 GDN + 1 attention layer per 4-layer block) on the RL trainer's routed-MoE path, as
# far as CPU allows (megablox in interpret mode; the tokamax GMM is TPU-only).
_MODEL = {
    "model_name": "qwen3.5-397b-a17b",
    "override_model_config": True,
    "base_num_decoder_layers": _LAYERS,
    "base_emb_dim": _EMB,
    "base_num_query_heads": 4,
    "base_num_kv_heads": 2,
    "head_dim": 64,
    "vocab_size": _VOCAB,
    "base_mlp_dim": _MOE_MLP,
    "base_moe_mlp_dim": _MOE_MLP,
    "num_experts": _EXPERTS,
    "num_experts_per_tok": _TOP_K,
    "gdn_num_key_heads": 2,
    "gdn_num_value_heads": 4,
    "gdn_key_head_dim": 32,
    "gdn_value_head_dim": 32,
    "gdn_chunk_size": 16,
    "use_gdn_kernel": False,
    "use_mrope": False,
    "max_target_length": _SEQ,
    "max_prefill_predict_length": _SEQ,
    "per_device_batch_size": 0.5,
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "float32_gate_logits": True,
    "attention": "dot_product",
    "use_tokamax_splash": False,
    "prefuse_moe_weights": True,
    "moe_unsplit_prefused_wi": True,
    "fp8_moe_fake_quant": True,
    "megablox": True,
    "sparse_matmul": True,
    "use_tokamax_gmm": False,
    "use_gmm_v2": False,
    "use_ring_of_experts": True,
    "use_ragged_sort": True,
    "use_custom_sort_vjp": False,
    "ragged_buffer_factor": 2.75,
    "num_moe_token_chunks": 2,
    "moe_chunk_pipeline": True,
    "ring_of_experts_local_routing": True,
    "moe_expert_weight_prefetch": "forward",
    "moe_exclude_padding_from_routing": True,
    "custom_mesh_and_rule": "cp-as-ep",
    "ici_fsdp_parallelism": 2,
    "ici_context_parallelism": 2,
    "ici_expert_parallelism": 2,
    "context_parallel_load_balance": False,
    "scan_layers": True,
    "remat_policy": "full",
    "enable_dropout": False,
}


def _inputs(batch):
  """Packed rows (1-3 segments, then padding) and replayed routing with some unused slots."""
  rng = np.random.default_rng(1234)
  tokens = rng.integers(1, _VOCAB, size=(batch, _SEQ), dtype=np.int32)
  targets = np.roll(tokens, -1, axis=1)
  seg = np.zeros((batch, _SEQ), np.int32)
  pos = np.zeros((batch, _SEQ), np.int32)
  for b in range(batch):
    n_real = _SEQ - (5 + (7 * b) % 29)
    cuts = sorted(rng.choice(np.arange(8, n_real - 4), size=b % 3, replace=False).tolist()) if b % 3 else []
    for i, (s, e) in enumerate(zip([0] + cuts, cuts + [n_real])):
      seg[b, s:e] = i + 1
      pos[b, s:e] = np.arange(e - s)
  mask = (seg != 0).astype(np.float32)
  routing = np.stack(
      [np.stack([rng.permutation(_EXPERTS)[:_TOP_K] for _ in range(batch * _SEQ)]).reshape(batch, _SEQ, _TOP_K)
       for _ in range(_LAYERS)],
      axis=2,
  ).astype(np.int32)  # [batch, seq, layers, top_k]
  routing[:, ::7, :, _TOP_K - 1] = -1
  return tuple(jnp.asarray(v) for v in (tokens, pos, seg, routing, targets, mask))


def _census(hlo):
  """Fake-quant max all-reduces and 8-bit-float all-gathers, inside and outside the program's while loops."""
  comps, cur = {}, None
  for line in hlo.split("\n"):
    m = re.match(r"^(?:ENTRY )?%?([\w.\-]+) .*\{\s*$", line)
    if m and not line.startswith(" "):
      cur = m.group(1)
      comps[cur] = []
    elif cur is not None:
      comps[cur].append(line)
  max_regions = {c for c, lines in comps.items() if any(re.search(r"\bmaximum\(", x) for x in lines) and len(lines) <= 6}
  calls = {
      c: set(re.findall(r"(?:calls|to_apply|body|condition|branch_computations)=\{?%?([\w.\-]+)", "\n".join(lines)))
      for c, lines in comps.items()
  }
  in_loop, stack = set(), list(set(re.findall(r"body=%?([\w.\-]+)", hlo)))
  while stack:
    c = stack.pop()
    if c not in in_loop:
      in_loop.add(c)
      stack.extend(calls.get(c, ()))
  out = {"max_all_reduce_loop": 0, "max_all_reduce_top": 0, "f8_all_gather_loop": 0}
  for c, lines in comps.items():
    where = "loop" if c in in_loop else "top"
    for line in lines:
      if re.search(r"= .*\ball-reduce(-start)?\(", line):
        ta = re.search(r"to_apply=%?([\w.\-]+)", line)
        if ta and ta.group(1) in max_regions:
          out[f"max_all_reduce_{where}"] += 1
      m = re.search(r"= \(?([a-z0-9]+)\[[0-9,]*\][^=]*?\ball-gather(?:-start)?\(", line)
      # XLA:CPU runs an f8e4m3fn all-gather in f16 and a bf16 one in f32; on TPU they stay f8e4m3fn and bf16.
      if m and where == "loop" and m.group(1) in ("f8e4m3fn", "f16"):
        out["f8_all_gather_loop"] += 1
  return out


def _run(mode):
  """loss, logits, parameters and gradients (raw bits) and the program census of one mode."""
  cfg = _cfg(**_MODEL, fp8_fake_quant_hoist=mode)
  assert cfg.fp8_fake_quant_hoist == mode, cfg.fp8_fake_quant_hoist
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  create_fn, abstract_model = model_creation_utils.create_nnx_abstract_model(cfg, mesh=mesh)
  model = maxtext_utils_nnx.create_nnx_sharded_model(abstract_model, create_fn, mesh=mesh)
  batch = int(cfg.per_device_batch_size * jax.device_count())
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def loss_fn(p, tokens, pos, seg, routing, targets, mask):
      logits = nnx.merge(graphdef, p, rest, copy=True)(
          tokens, pos, decoder_segment_ids=seg, forced_routed_experts=routing, enable_dropout=False
      )
      if isinstance(logits, tuple):
        logits = logits[0]
      logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
      nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
      return jnp.sum(nll * mask) / jnp.sum(mask), logits

    compiled = jax.jit(jax.value_and_grad(loss_fn, has_aux=True)).lower(params, *_inputs(batch)).compile()
    (loss, logits), grads = compiled(params, *_inputs(batch))
  arrays = {"loss": _bits(np.asarray(loss, np.float32)), "logits": _bits(logits)}
  for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]:
    arrays["grad" + jax.tree_util.keystr(path)] = _bits(leaf)
  for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
    arrays["param" + jax.tree_util.keystr(path)] = _bits(leaf)
  census = _census(compiled.as_text())
  print(f"mode={mode} loss={float(loss):.9g} arrays={len(arrays)} census={census}", flush=True)
  return arrays, census


def main():
  os.environ.pop(_ENV, None)
  assert jax.device_count() == _NUM_DEVICES, jax.device_count()
  ok = True

  def check(cond, what):
    nonlocal ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {what}", flush=True)

  ref, ref_census = _run("off")
  check(ref_census["max_all_reduce_loop"] > 0, f"off: fake-quant max all-reduces inside the layer loop {ref_census}")
  check(ref_census["f8_all_gather_loop"] == 0, "off: no 8-bit-float all-gathers in the layer loop")
  check(sum(k.startswith("grad") and "routed_experts" in k for k in ref) > 0, "off: routed-expert gradients present")
  for mode in ("absmax", "gather8"):
    arrays, census = _run(mode)
    check(sorted(arrays) == sorted(ref), f"{mode}: same parameter and gradient tree as off ({len(arrays)} arrays)")
    differing = [k for k in sorted(ref) if k in arrays and not np.array_equal(arrays[k], ref[k])]
    check(not differing, f"{mode}: loss, logits, parameters, gradients bit-identical to off (differing: {differing[:8]})")
    check(census["max_all_reduce_loop"] == 0, f"{mode}: no fake-quant max all-reduce inside the layer loop {census}")
    check(census["max_all_reduce_top"] > 0, f"{mode}: the stack's absmax all-reduce is outside the loop")
    if mode == "gather8":
      check(census["f8_all_gather_loop"] > 0, "gather8: the layer loop all-gathers 8-bit-float expert weights")
    else:
      check(census["f8_all_gather_loop"] == 0, "absmax: no 8-bit-float all-gathers in the layer loop")
  if not ok:
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()
