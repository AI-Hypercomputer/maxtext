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

"""Unit tests for MaxText m3 conformance spec and single-device Checks 1-4."""

from collections.abc import Mapping
import dataclasses
from typing import Any
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
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


class ConformanceTest(absltest.TestCase):
  """Tests `spec` registry, `SkipDeclaration` guards, and Checks 1-4."""

  def setUp(self):
    super().setUp()
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


if __name__ == "__main__":
  absltest.main()
