# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""dense_grad_reduce on a CPU mesh: the dense-weight gradients of cp-as-ep-dense-no-cp by all-reduce.

cp-as-ep-dense-no-cp replicates the dense weights over context and maps 'embed_grad', so linears.route_dense_weight_grad
reduce-scatters each of their gradients over every token axis and then all-gathers the shard over context, across an
optimization barrier. dense_grad_reduce='all_reduce' drops the 'embed_grad' rule, so GSPMD reduces those gradients with
its own all-reduces. Only the reduction changes, so the model must compute the same function.

The model, mesh and inputs are those of cp_as_ep_dense_no_cp_test.py (the tiny hybrid Qwen3.5 on 16 CPU devices,
fsdp2 x context4 x expert2), whose helpers are reused. The checks run in a subprocess so the forced device count takes
effect before JAX initializes.

  - Config: the all_reduce rules are the reduce_scatter rules minus exactly the 'embed_grad' rule (train and eval).
  - Shardings: every parameter keeps its sharding.
  - Program: the lowered backward loses the optimization barriers of the reduce-scatter route, and the compiled
    collectives change (on TPU the 256-way reduce-scatters and context all-gathers become all-reduces; the CPU
    backend's collective choice is printed, not gated).
  - Numerics: the loss is bit-identical, and every parameter gradient agrees to reduction order with the reduce_scatter
    program (the gate of cp_as_ep_dense_no_cp_test.py: max|new - old| <= 2**-6 * max|old| per leaf for the bfloat16
    model, 1e-5 for the float32 dense variant).
  - Positive controls, each of which must be caught:
      the key set but its rule edit disabled (the program check must fail);
      the dense gradients averaged instead of summed over the 4 context ranks (the numerics gate must fail);
      one query kernel scaled by 1 + 1e-3 (the numerics gate must fail);
      'all_reduce' under cp-as-ep, whose rules do not map 'embed_grad', and a misspelled value (config must refuse both).
"""

# pylint: disable=protected-access

import collections
import contextlib
import os
import re
import subprocess
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_RULE = "cp-as-ep-dense-no-cp"
_EMBED_GRAD_RULE = ["embed_grad", ["fsdp", "expert", "context"]]
_PASS_TOKEN = "DENSE_GRAD_REDUCE_CHECKS_PASSED"
_DENSE_NAMES = (
    "['query']",
    "['key']",
    "['value']",
    "['out']",
    "shared_expert",
    "routed_experts']['gate",
)


@pytest.mark.cpu_only
def test_rl_config_and_explicit_sharding_validation():
  """RLConfig runs _apply_dense_grad_reduce and returns self; shard_mode='explicit' rejects 'embed_grad' rules."""
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  import cp_as_ep_dense_no_cp_test as base  # pylint: disable=import-outside-toplevel,import-error
  from maxtext.configs import types  # pylint: disable=import-outside-toplevel

  rules_with_embed_grad = [_EMBED_GRAD_RULE, ["embed", ["fsdp", "expert"]]]
  rl_cfg = types.RLConfig.model_validate(
      {
          "run_name": "test_rl",
          "tokenizer_path": "assets/tokenizer",
          "logical_axis_rules": list(rules_with_embed_grad),
          "dense_grad_reduce": "all_reduce",
      }
  )
  assert rl_cfg is not None
  assert _EMBED_GRAD_RULE not in rl_cfg.logical_axis_rules

  # RLConfig does not expand custom_mesh_and_rule YAMLs; setting dense_grad_reduce='all_reduce' alongside
  # custom_mesh_and_rule='cp-as-ep-dense-no-cp' (or default RL rules without 'embed_grad') must succeed cleanly.
  rl_custom = types.RLConfig.model_validate(
      {
          "run_name": "test_rl",
          "tokenizer_path": "assets/tokenizer",
          "custom_mesh_and_rule": "cp-as-ep-dense-no-cp",
          "dense_grad_reduce": "all_reduce",
      }
  )
  assert rl_custom is not None
  assert rl_custom.dense_grad_reduce == "all_reduce"

  # RLConfig with moe_wo_fsdp_on_mlp=True and custom_mesh_and_rule='cp-as-ep' also succeeds without requiring
  # RLConfig to have expanded the custom_mesh_and_rule YAML yet, while still rejecting incompatible MoE flags.
  rl_wo = types.RLConfig.model_validate(
      {
          "run_name": "test_rl",
          "tokenizer_path": "assets/tokenizer",
          "custom_mesh_and_rule": "cp-as-ep",
          "moe_wo_fsdp_on_mlp": True,
          "num_experts": 8,
      }
  )
  assert rl_wo.moe_wo_fsdp_on_mlp is True

  with pytest.raises(ValueError, match="incompatible with shard_exp_on_fsdp=True"):
    types.RLConfig.model_validate(
        {
            "run_name": "test_rl",
            "tokenizer_path": "assets/tokenizer",
            "custom_mesh_and_rule": "cp-as-ep",
            "moe_wo_fsdp_on_mlp": True,
            "shard_exp_on_fsdp": True,
        }
    )

  with pytest.raises(ValueError, match="require shard_mode='auto'"):
    types.RLConfig.model_validate(
        {
            "run_name": "test_rl",
            "tokenizer_path": "assets/tokenizer",
            "logical_axis_rules": list(rules_with_embed_grad),
            "shard_mode": "explicit",
        }
    )

  with pytest.raises(ValueError, match="require shard_mode='auto'"):
    base._config(_RULE, shard_mode="explicit")


@pytest.mark.cpu_only
def test_inference_with_physical_mesh_context():
  """DenseGeneral and GateLogit forward succeed under physical `with mesh:` (empty abstract mesh, as in MaxEngine/vLLM)."""
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  import cp_as_ep_dense_no_cp_test as base  # pylint: disable=import-outside-toplevel,import-error
  from flax import nnx  # pylint: disable=import-outside-toplevel
  from flax.linen import partitioning as nn_partitioning  # pylint: disable=import-outside-toplevel
  from maxtext.layers import linears, moe  # pylint: disable=import-outside-toplevel

  mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1, 1, 1)), ("fsdp", "context", "expert"))
  for rule in ("cp-as-ep-dense-no-cp", "cp-as-ep-dense-fsdp"):
    cfg = base._config(rule)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      assert jax.sharding.get_abstract_mesh().empty
      dense = linears.DenseGeneral(
          in_features_shape=32,
          out_features_shape=16,
          kernel_axes=("embed_attn", "mlp"),
          dtype=jnp.bfloat16,
          rngs=nnx.Rngs(0),
      )
      gate = moe.GateLogit(
          in_features_shape=32,
          out_features_shape=8,
          model_name="qwen3.5-397b-a17b",
          mesh=mesh,
          kernel_axes=("embed", None),
          dtype=jnp.bfloat16,
          rngs=nnx.Rngs(1),
      )
      x = jnp.ones((2, 4, 32), dtype=jnp.bfloat16)
      y_dense = jax.jit(lambda a, m=dense: m(a))(x)
      y_gate, _ = jax.jit(lambda a, m=gate: m(a))(x)
      assert y_dense.shape == (2, 4, 16)
      assert y_gate.shape == (2, 4, 8)


@pytest.mark.cpu_only
def test_dense_grad_reduce_on_cpu_mesh():
  """Runs `main` below in a child process with 16 CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=16"
  env["JAX_PLATFORMS"] = "cpu"
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _rules_differ_by_embed_grad(rs_rules, ar_rules) -> bool:
  """ar_rules == rs_rules with exactly the 'embed_grad' rule removed, order kept."""

  def as_list(rules):
    return [[r[0], list(r[1]) if isinstance(r[1], (list, tuple)) else r[1]] for r in rules]

  rs, ar = as_list(rs_rules), as_list(ar_rules)
  return rs.count(_EMBED_GRAD_RULE) == 1 and [r for r in rs if r != _EMBED_GRAD_RULE] == ar


def _run(base, cfg, data, cotangent, values=None, control=None):
  """base._run, also returning the lowered (pre-optimization) text of the program it compiles (its last Lowered.compile)."""
  lowered_texts = []
  real_compile = jax.stages.Lowered.compile

  def compile_and_keep_text(self, *args, **kwargs):
    lowered_texts.append(self.as_text())
    return real_compile(self, *args, **kwargs)

  with mock.patch.object(jax.stages.Lowered, "compile", compile_and_keep_text):
    result = base._run(cfg, data, cotangent, values=values, control=control)
  assert lowered_texts, "no Lowered.compile call"
  return result, lowered_texts[-1]


def _program(base, result, lowered) -> collections.Counter:
  """Optimization barriers of the lowered program and collectives (kind, replica-group size) of the compiled one."""
  counts = base._collectives(result.hlo)
  counts[("optimization_barrier (lowered)", 0)] = lowered.count("stablehlo.optimization_barrier")
  return counts


def _program_changes(base, label, reference, result) -> bool:
  """The lowered backward loses optimization barriers and the compiled collectives change."""
  (ref_result, ref_lowered), (new_result, new_lowered) = reference, result
  ref, new = _program(base, ref_result, ref_lowered), _program(base, new_result, new_lowered)
  n = base._NUM_DEVICES
  key = ("optimization_barrier (lowered)", 0)
  barriers_gone = ref[key] - new[key]
  changed = {k: (ref[k], new[k]) for k in sorted(set(ref) | set(new), key=str) if ref[k] != new[k]}
  collectives_changed = any(k != key for k in changed)
  ok = barriers_gone > 0 and collectives_changed
  print(
      f"{label}: program {'CHANGED as expected' if ok else 'NOT changed as expected'}: lowered optimization barriers"
      f" {ref[key]} -> {new[key]}; compiled {n}-device all-reduces {ref[('all-reduce', n)]} -> {new[('all-reduce', n)]},"
      f" {n}-device reduce-scatters {ref[('reduce-scatter', n)]} -> {new[('reduce-scatter', n)]}; every changed count"
      f" {changed}",
      flush=True,
  )
  return ok


def _bit_identical_by_class(reference, result):
  """(dense leaves bit-identical, dense leaves, other leaves bit-identical, other leaves)."""
  dense = [n for n in reference.grads if any(k in n for k in _DENSE_NAMES)]
  other = [n for n in reference.grads if n not in dense]

  def same(names):
    return sum(int(np.array_equal(result.grads[n], reference.grads[n])) for n in names)

  return same(dense), len(dense), same(other), len(other)


@contextlib.contextmanager
def _dense_grads_scaled(factor):
  """route_dense_weight_grad that also scales the cotangent of every 'embed'/'embed_attn' kernel by `factor`."""
  # pylint: disable=import-outside-toplevel
  from maxtext.layers import linears

  real = linears.route_dense_weight_grad

  @jax.custom_vjp
  def scaled(x):
    return x

  scaled.defvjp(lambda x: (x, None), lambda _, g: (g * jnp.asarray(factor, g.dtype),))

  def route(kernel, kernel_axes):
    kernel = real(kernel, kernel_axes)
    if any(isinstance(a, str) and a in ("embed", "embed_attn") for a in kernel_axes):
      kernel = scaled(kernel)
    return kernel

  with mock.patch.object(linears, "route_dense_weight_grad", route):
    yield


def _refused(base, label, rule, **overrides) -> bool:
  """Config construction must raise, with a message that names dense_grad_reduce."""
  try:
    base._config(rule, **overrides)
  except Exception as error:  # pylint: disable=broad-exception-caught
    msg = " ".join(str(error).split())
    named = "dense_grad_reduce" in msg
    print(
        f"{label}: refused ({type(error).__name__}), message names dense_grad_reduce: {named}: ...{msg[-200:]}",
        flush=True,
    )
    return named
  print(f"{label}: ACCEPTED (not refused)", flush=True)
  return False


def _check_no_collective_permute_and_mla_lora(cfg_rs):
  """Checks zero collective-permute on fsdp2xcontext2xexpert2 and MLA q_lora/kv_lora routing + indivisibility guard."""
  # pylint: disable=import-outside-toplevel
  from flax.linen import partitioning as nn_partitioning
  from maxtext.layers import linears

  mesh8 = jax.sharding.Mesh(np.array(jax.devices()[:8]).reshape((2, 2, 2)), ("fsdp", "context", "expert"))
  x = jax.device_put(
      jnp.arange(4 * 8 * 16, dtype=jnp.float32).reshape(4, 8, 16) * 1e-2,
      jax.sharding.NamedSharding(mesh8, jax.sharding.PartitionSpec(("fsdp", "expert"), "context", None)),
  )
  w = jax.device_put(
      jnp.arange(16 * 12, dtype=jnp.float32).reshape(16, 12) * 1e-2,
      jax.sharding.NamedSharding(mesh8, jax.sharding.PartitionSpec(("fsdp", "expert"), None)),
  )

  def probe_hlo(rules, axes=("embed_attn", "mlp")):
    with nn_partitioning.axis_rules(rules), jax.set_mesh(mesh8):
      lowered = jax.jit(jax.grad(lambda w_in: jnp.sum(x @ linears.route_dense_weight_grad(w_in, axes)))).lower(w)
      compiled = lowered.compile()
      return lowered.as_text(), compiled.as_text(), np.asarray(compiled(w))

  _, new_hlo, grad_new = probe_hlo(cfg_rs.logical_axis_rules)
  old_rules = [
      (["embed_grad", ["fsdp", "context", "expert"]] if r and r[0] == "embed_grad" else r)
      for r in cfg_rs.logical_axis_rules
  ]
  _, old_hlo, grad_old = probe_hlo(old_rules)
  perm_absent = "collective-permute" not in new_hlo and np.allclose(grad_new, grad_old, rtol=1e-5, atol=1e-6)
  perm_ctrl_present = "collective-permute" in old_hlo

  # MLA q_lora and kv_lora: divisible shapes (16 % 8 == 0) route through embed_grad with zero collective-permute.
  w_3d = jax.device_put(
      jnp.ones((16, 2, 6), dtype=jnp.float32) * 0.1,
      jax.sharding.NamedSharding(mesh8, jax.sharding.PartitionSpec(("fsdp", "expert"), None, None)),
  )
  mla_ok = True
  for axes in (("q_lora", "q_heads", "kv"), ("kv_lora", "kv_heads", "kv_head_dim")):
    with nn_partitioning.axis_rules(cfg_rs.logical_axis_rules), jax.set_mesh(mesh8):
      low = jax.jit(
          jax.grad(
              lambda w_in, ax=axes: jnp.sum(
                  jax.lax.dot_general(x, linears.route_dense_weight_grad(w_in, ax), (((2,), (0,)), ((), ())))
              )
          )
      ).lower(w_3d)
      comp = low.compile()
      mla_ok &= "stablehlo.optimization_barrier" in low.as_text() and "collective-permute" not in comp.as_text()

  # Indivisible kv_lora_rank (4 % 4 == 0 for weight sharding ('fsdp', 'expert'), 4 % 8 != 0 for 'embed_grad'):
  # skips embed_grad routing cleanly instead of failing pjit divisibility.
  x_indiv = jax.device_put(
      jnp.ones((4, 8, 4), dtype=jnp.float32),
      jax.sharding.NamedSharding(mesh8, jax.sharding.PartitionSpec(("fsdp", "expert"), "context", None)),
  )
  w_indiv = jax.device_put(
      jnp.ones((4, 2, 6), dtype=jnp.float32),
      jax.sharding.NamedSharding(mesh8, jax.sharding.PartitionSpec(("fsdp", "expert"), None, None)),
  )
  with nn_partitioning.axis_rules(cfg_rs.logical_axis_rules), jax.set_mesh(mesh8):
    low_indiv = jax.jit(
        jax.grad(
            lambda w_in: jnp.sum(
                jax.lax.dot_general(
                    x_indiv,
                    linears.route_dense_weight_grad(w_in, ("kv_lora", "kv_heads", "kv_head_dim")),
                    (((2,), (0,)), ((), ())),
                )
            )
        )
    ).lower(w_indiv)
    grad_indiv = low_indiv.compile()(w_indiv)
    mla_ok &= "stablehlo.optimization_barrier" not in low_indiv.as_text() and grad_indiv.shape == (4, 2, 6)

  # Compressed attention wq_a (`kernel_axes=('embed_attn', 'q_lora')` in attention_compressed.py): both axes are in
  # _DENSE_EMBED_AXES, and 'q_lora' wins ('fsdp', 'expert') in logical_axis_rules priority order
  # (`to = P(None, ('fsdp', 'expert'))`). route_dense_weight_grad must route only that sharded dim 1 without
  # duplicate-'embed_grad' errors or collective-permute, matching all_reduce gradients. Also check duplicate string
  # axes ('embed', 'embed') return kernel unchanged.
  from flax import nnx  # pylint: disable=import-outside-toplevel
  from maxtext.utils import sharding as sharding_utils  # pylint: disable=import-outside-toplevel

  ar_rules = [r for r in cfg_rs.logical_axis_rules if not (r and r[0] == "embed_grad")]
  to_wqa = sharding_utils.logical_to_mesh_axes(("embed_attn", "q_lora"), mesh8, rules=cfg_rs.logical_axis_rules)
  mla_ok &= tuple(to_wqa) == (None, ("fsdp", "expert"))
  x_wqa = jax.device_put(
      jnp.arange(4 * 8 * 64, dtype=jnp.float32).reshape(4, 8, 64) * 1e-3,
      jax.sharding.NamedSharding(mesh8, jax.sharding.PartitionSpec(("fsdp", "expert"), "context", None)),
  )
  with nn_partitioning.axis_rules(cfg_rs.logical_axis_rules), jax.set_mesh(mesh8):
    wqa = linears.DenseGeneral(
        in_features_shape=64,
        out_features_shape=32,
        kernel_axes=("embed_attn", "q_lora"),
        dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    wqa_gdef, wqa_params = nnx.split(wqa)

    def wqa_loss(p):
      return jnp.sum(nnx.merge(wqa_gdef, p)(x_wqa) ** 2)

    low_wqa_rs = jax.jit(jax.value_and_grad(wqa_loss)).lower(wqa_params)
    comp_wqa_rs = low_wqa_rs.compile()
    loss_wqa_rs, grads_wqa_rs = comp_wqa_rs(wqa_params)
    dup_out = linears.route_dense_weight_grad(jnp.ones((16, 16), dtype=jnp.float32), ("embed", "embed"))
    mla_ok &= dup_out.shape == (16, 16)
  with nn_partitioning.axis_rules(ar_rules), jax.set_mesh(mesh8):
    loss_wqa_ar, grads_wqa_ar = jax.jit(jax.value_and_grad(wqa_loss))(wqa_params)
  mla_ok &= (
      "stablehlo.optimization_barrier" in low_wqa_rs.as_text()
      and "collective-permute" not in comp_wqa_rs.as_text()
      and np.allclose(np.asarray(loss_wqa_rs), np.asarray(loss_wqa_ar), rtol=1e-5, atol=1e-6)
      and np.allclose(
          np.asarray(grads_wqa_rs["kernel"][...]),
          np.asarray(grads_wqa_ar["kernel"][...]),
          rtol=1e-5,
          atol=1e-6,
      )
  )

  return perm_absent, perm_ctrl_present, mla_ok


def main():
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  import cp_as_ep_dense_no_cp_test as base  # pylint: disable=import-outside-toplevel,import-error
  from maxtext.configs import types  # pylint: disable=import-outside-toplevel

  assert jax.device_count() == base._NUM_DEVICES, jax.devices()
  jax.config.update("jax_enable_compilation_cache", False)
  checks, controls = {}, {}

  cfg_rs = base._config(_RULE)
  cfg_ar = base._config(_RULE, dense_grad_reduce="all_reduce")
  print(
      f"dense_grad_reduce: {cfg_rs.dense_grad_reduce!r} (default) / {cfg_ar.dense_grad_reduce!r}",
      flush=True,
  )
  checks["config: all_reduce rules = reduce_scatter rules minus the embed_grad rule (train and eval)"] = (
      _rules_differ_by_embed_grad(cfg_rs.logical_axis_rules, cfg_ar.logical_axis_rules)
      and _rules_differ_by_embed_grad(cfg_rs.logical_axis_rules_for_eval, cfg_ar.logical_axis_rules_for_eval)
  )

  data, cotangent = base._inputs(cfg_rs)
  rs_run = _run(base, cfg_rs, data, cotangent)
  rs = rs_run[0]
  ar_run = _run(base, cfg_ar, data, cotangent, values=rs.values)
  ar = ar_run[0]
  checks["shardings: every parameter keeps its sharding"] = rs.specs == ar.specs
  checks["program: the lowered backward loses barriers and the compiled collectives change"] = _program_changes(
      base, "all_reduce", rs_run, ar_run
  )
  checks["numerics: bfloat16 model"] = base._numerics("all_reduce vs reduce_scatter", ar, rs)
  d_same, d_tot, o_same, o_tot = _bit_identical_by_class(rs, ar)
  print(
      f"bit-identical gradient leaves (dense, of; other, of): {d_same} of {d_tot}; {o_same} of {o_tot}",
      flush=True,
  )

  rs32 = base._run(base._config(_RULE, **base._F32_DENSE), data, cotangent)
  ar32 = base._run(
      base._config(_RULE, dense_grad_reduce="all_reduce", **base._F32_DENSE),
      data,
      cotangent,
      values=rs32.values,
  )
  checks["numerics: float32 dense variant"] = base._numerics(
      "float32 dense all_reduce vs reduce_scatter", ar32, rs32, tol=base._F32_TOL
  )
  d32_same, d32_tot, o32_same, o32_tot = _bit_identical_by_class(rs32, ar32)
  print(
      f"float32 bit-identical gradient leaves (dense, of; other, of): {d32_same} of {d32_tot}; {o32_same} of {o32_tot}",
      flush=True,
  )

  perm_absent, perm_ctrl_present, mla_ok = _check_no_collective_permute_and_mla_lora(cfg_rs)
  checks["fsdp2xcontext2xexpert2: ['fsdp','expert','context'] emits zero collective-permute"] = perm_absent
  controls["old ['fsdp','context','expert'] order emits collective-permute"] = perm_ctrl_present
  checks["MLA q_lora/kv_lora: routes divisible kernels and skips indivisible kv_lora_rank cleanly"] = mla_ok

  # Positive controls.
  inert = staticmethod(lambda cfg, *, require_rule: None)
  with mock.patch.object(types.MaxTextConfig, "_apply_dense_grad_reduce", inert):
    cfg_inert = base._config(_RULE, dense_grad_reduce="all_reduce")
  wrong_run = _run(base, cfg_inert, data, cotangent, values=rs.values)
  controls["the key set with its rule edit disabled"] = not _program_changes(
      base, "control: rule edit disabled", rs_run, wrong_run
  )
  wrong = base._run(cfg_ar, data, cotangent, values=rs.values, control=_dense_grads_scaled(0.25))
  controls["dense gradients averaged over the 4 context ranks"] = not base._numerics(
      "control: dense grads / 4", wrong, rs
  )
  query = next(name for name in rs.values if re.search(r"\['query'\].*\['kernel'\]", name))
  perturbed = dict(rs.values, **{query: rs.values[query] * np.float32(1 + 1e-3)})
  wrong = base._run(cfg_ar, data, cotangent, values=perturbed)
  controls[f"{query} scaled by 1 + 1e-3"] = not base._numerics("control: perturbed query", wrong, rs)
  controls["all_reduce under cp-as-ep (no embed_grad rule)"] = _refused(
      base, "control: cp-as-ep", "cp-as-ep", dense_grad_reduce="all_reduce"
  )
  controls["a misspelled value"] = _refused(base, "control: misspelled value", _RULE, dense_grad_reduce="allreduce")

  for name, ok in {
      **checks,
      **{f"positive control caught: {k}": v for k, v in controls.items()},
  }.items():
    print(f"  [{'ok' if ok else 'FAIL'}] {name}", flush=True)
  failed = [name for name, ok in checks.items() if not ok]
  undetected = [name for name, caught in controls.items() if not caught]
  if failed or undetected:
    print(
        f"checks that failed: {failed}; positive controls that were NOT detected: {undetected}",
        flush=True,
    )
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()
