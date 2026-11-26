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

"""Tests for `maxtext.m3.infra.quantization`.

The property under test is scope, stated exactly: after `maybe_quantize`, one
forward pass has one matmul in the recipe's dtype per projection layer of the
decoder stack and everything else -- attention scores, the tied-embedding
logits -- is still bf16. The parameter tree is unchanged. The checks run on
the traced jaxpr, so they hold on CPU; the TPU tests run the loop end to end
and look at the compiled program.
"""

import re
from types import SimpleNamespace

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest

from maxtext.m3 import models as m3_models
from maxtext.m3.core.sharding import create_mesh, sharding_from_config
from maxtext.m3.infra import quantization
from maxtext.m3.infra import scan as m3_scan
from maxtext.m3.train import loop
from tests.utils import m3_test_utils

_E4M3, _E5M2 = jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2)
_INT4, _E2M1 = jnp.dtype(jnp.int4), jnp.dtype(jnp.float4_e2m1fn)
_E8M0 = jnp.dtype(jnp.float8_e8m0fnu)


class _HandRolled(nnx.Module):
  """A weight applied with `jnp.einsum` directly, the way a ported `DenseGeneral` or a custom router does it."""

  def __init__(self, features, rngs):
    self.kernel = nnx.Param(jax.random.normal(rngs.params(), (4, features)))

  def __call__(self, x):
    return jnp.einsum("...i,io->...o", x, self.kernel[...])


class _Wrapped(nnx.Module):
  """A router that holds its projection, like Qwen3-Next's `Qwen3_8TopKRouter`."""

  def __init__(self, linear):
    self.linear = linear

  def __call__(self, x):
    return self.linear(x)


class _Block(nnx.Module):
  """A decoder block with one projection and one router.

  `unused` adds a projection `__call__` never uses; `handrolled` makes the
  projection a `_HandRolled` instead of an `nnx.Linear`; `twice` applies the
  projection twice.
  """

  def __init__(self, rngs, unused=False, handrolled=False, twice=False):
    self.wi = _HandRolled(4, rngs) if handrolled else nnx.Linear(4, 4, rngs=rngs)
    self.gate = nnx.Linear(4, 2, rngs=rngs)
    self.twice = twice
    if unused:
      self.unused = nnx.Linear(4, 4, rngs=rngs)

  def __call__(self, x):
    y = self.wi(self.wi(x)) if self.twice else self.wi(x)
    return y * jnp.sum(self.gate(x), axis=-1, keepdims=True)


class _Decoder(nnx.Module):
  """Both m3 layouts of the stack -- a flat `layers_0` and a `layers` container -- plus an untied head.

  The container's block has a hand-rolled router, as DeepSeek-V3's is.
  """

  def __init__(self, rngs, unused=False, handrolled=False, twice=False):
    self.layers_0 = _Block(rngs, unused, handrolled, twice)
    self.layers = nnx.List([_Block(rngs)])
    self.layers[0].gate = _HandRolled(2, rngs)
    self.logits_dense = nnx.Linear(4, 8, rngs=rngs)

  def __call__(self, x):
    return self.logits_dense(self.layers[0](self.layers_0(x)))


class _Model(nnx.Module):

  def __init__(self, rngs, unused=False, handrolled=False, twice=False):
    self.decoder = _Decoder(rngs, unused, handrolled, twice)

  def __call__(self, x):
    return self.decoder(x)


class _Flat(nnx.Module):
  """A projection with no decoder stack above it."""

  def __init__(self, rngs):
    self.proj = nnx.Linear(4, 4, rngs=rngs)

  def __call__(self, x):
    return self.proj(x)


class _RawWeightBlock(nnx.Module):
  """A layer that holds a matmul weight itself, next to a norm scale and a projection."""

  def __init__(self, rngs):
    self.kernel = nnx.Param(jax.random.normal(rngs.params(), (4, 4)))
    self.scale = nnx.Param(jnp.ones(4))
    self.wi = nnx.Linear(4, 4, rngs=rngs)

  def __call__(self, x):
    return self.wi(x @ self.kernel[...]) * self.scale[...]


class _Keywords(nnx.Module):
  """Takes an array and a Python flag by keyword."""

  def __init__(self, rngs):
    self.proj = nnx.Linear(4, 4, rngs=rngs)

  def __call__(self, x, *, y, twice=False):
    x = self.proj(x + y)
    return self.proj(x) if twice else x


class _Ragged(nnx.Module):

  def __init__(self):
    self.kernel = nnx.Param(jnp.ones((2, 4, 3)))

  def __call__(self, x, group_sizes):
    return jax.lax.ragged_dot(x, self.kernel[...], group_sizes)


class _MxStack(nnx.Module):
  """Minimal decoder stack with one Linear or Einsum projection for MX shape and parity checks."""

  def __init__(self, in_features, out_features, rngs, *, use_einsum=False, dtype=jnp.bfloat16):
    if use_einsum:
      proj = nnx.Einsum("...i,io->...o", (in_features, out_features), dtype=dtype, rngs=rngs)
    else:
      proj = nnx.Linear(in_features, out_features, use_bias=False, dtype=dtype, rngs=rngs)
    block = nnx.Module()
    block.proj = proj
    self.decoder = nnx.Module()
    self.decoder.layers = nnx.List([block])

  def __call__(self, x, *, out_sharding=None):
    proj = self.decoder.layers[0].proj
    if out_sharding is not None:
      return proj(x, out_sharding=out_sharding)
    return proj(x)


class QuantizationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = create_mesh(devices=jax.devices()[:1])
    self.enter_context(jax.set_mesh(self.mesh))

  def _model(self, **overrides):
    if "quantization" in overrides and not str(overrides["quantization"]).startswith("jax_mx"):
      overrides.setdefault("use_qwix_quantization", "true")
    config = m3_test_utils.tiny_config("qwen3-0.6b", **overrides)
    return config, m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0))

  def test_off_returns_the_same_model(self):
    config, model = self._model()
    self.assertIs(quantization.maybe_quantize(model, config, *quantization.example_inputs(config)), model)

  def test_plan_is_the_projections_of_the_decoder_stack(self):
    config, model = self._model()
    plan = quantization.plan(model)
    self.assertLen(plan.quantized, 7 * config.num_decoder_layers)
    self.assertEqual(
        {path.rsplit("/", 1)[-1] for path in plan.quantized}, {"query", "key", "value", "out", "wi_0", "wi_1", "wo"}
    )
    self.assertTrue(all(path.startswith("decoder/layers/") for path in plan.quantized), plan.quantized)
    self.assertEmpty(plan.routers)
    self.assertEmpty(plan.uncovered)

  def test_plan_keeps_routers_and_skips_the_head(self):
    plan = quantization.plan(_Model(nnx.Rngs(0)), quantize_routers=False)
    self.assertEqual(set(plan.quantized), {"decoder/layers_0/wi", "decoder/layers/0/wi"})
    # Both routers: an `nnx.Linear` and a hand-rolled one. Neither is quantized, neither is a complaint.
    self.assertEqual(set(plan.routers), {"decoder/layers_0/gate", "decoder/layers/0/gate"})
    self.assertEmpty(plan.uncovered)
    self.assertIn("2 projections", plan.describe())

  def test_plan_quantizes_routers_by_default_like_legacy(self):
    plan = quantization.plan(_Model(nnx.Rngs(0)))
    self.assertEqual(set(plan.quantized), {"decoder/layers_0/wi", "decoder/layers_0/gate", "decoder/layers/0/wi"})
    self.assertEmpty(plan.routers)
    # The hand-rolled router can't be matched by the exact-path rule, so it is reported, not skipped.
    self.assertEqual(plan.uncovered, ("decoder/layers/0/gate",))

  def test_hand_rolled_router_is_refused_with_a_hint(self):
    x = jax.ShapeDtypeStruct((2, 4), jnp.float32)
    with self.assertRaisesRegex(ValueError, r"decoder/layers/0/gate.*quantize_router_proj=false"):
      quantization.maybe_quantize(_Model(nnx.Rngs(0)), SimpleNamespace(quantization="int8"), x)
    quantized = quantization.maybe_quantize(
        _Model(nnx.Rngs(0)), SimpleNamespace(quantization="int8", quantize_router_proj=False), x
    )
    self.assertEqual(quantization.quantized_matmuls(quantized, jnp.int8, x), 2)

  def test_hand_rolled_projection_in_the_stack_is_refused(self):
    """A weight the plan cannot see must not be left in full precision silently (Qwen3-VL's `DenseGeneral` today)."""
    model = _Model(nnx.Rngs(0), handrolled=True)
    self.assertEqual(quantization.plan(model, quantize_routers=False).uncovered, ("decoder/layers_0/wi",))
    x = jax.ShapeDtypeStruct((2, 4), jnp.float32)
    with self.assertRaisesRegex(ValueError, r"can't be quantized.*decoder/layers_0/wi"):
      quantization.maybe_quantize(model, SimpleNamespace(quantization="int8", quantize_router_proj=False), x)

  @parameterized.named_parameters(("flat", False), ("scanned", True))
  def test_weight_held_by_the_layer_itself_is_refused(self, scanned):
    """The layer module sits at exactly `decoder/layers_<i>`, or `decoder/layers` once scanned."""
    rngs = nnx.Rngs(0)
    model = _Model(rngs)
    if scanned:
      model.decoder.layers = m3_scan.scan_run([_RawWeightBlock(rngs), _RawWeightBlock(rngs)], param_scan_axis=1)
      expected = "decoder/layers"
    else:
      model.decoder.layers_0 = _RawWeightBlock(rngs)
      expected = "decoder/layers_0"
    # The norm scale (1D, or 2D with the layer axis) is not a complaint.
    self.assertEqual(quantization.plan(model, quantize_routers=False).uncovered, (expected,))

  def test_census_traces_array_kwargs_and_keeps_flags_static(self):
    x = jax.ShapeDtypeStruct((2, 4), jnp.float32)
    model = _Keywords(nnx.Rngs(0))
    self.assertEqual(quantization.quantized_matmuls(model, jnp.float32, x, y=x, twice=True), 2)
    self.assertEqual(quantization.quantized_matmuls_by_module(model, ("proj",), jnp.float32, x, y=x), {"proj": 1})

  def test_census_counts_ragged_dot(self):
    x, groups = jax.ShapeDtypeStruct((8, 4), jnp.float32), jax.ShapeDtypeStruct((2,), jnp.int32)
    self.assertEqual(quantization.quantized_matmuls(_Ragged(), jnp.float32, x, groups), 1)

  def test_rule_matches_leaves_only(self):
    """qwix matches with `re.fullmatch`; a parent's own einsums must not match its children's rule."""
    the_plan = quantization.plan(_Model(nnx.Rngs(0)), quantize_routers=False)
    (rule,) = quantization.rules(the_plan, quantization.recipe("int8"))
    self.assertIsNotNone(re.fullmatch(rule.module_path, "decoder/layers_0/wi"))
    self.assertIsNotNone(re.fullmatch(rule.module_path, "decoder/layers/0/wi"))
    for path in ("decoder/layers_0", "decoder/layers_0/gate", "decoder/layers_0/wi/child", "decoder/logits_dense"):
      self.assertIsNone(re.fullmatch(rule.module_path, path), path)

  def test_int8_quantizes_each_planned_projection_and_nothing_else(self):
    config, model = self._model(quantization="int8")
    example = quantization.example_inputs(config)
    plan = quantization.plan(model)
    bf16_before = quantization.quantized_matmuls(model, jnp.bfloat16, *example)
    params_before = list(m3_test_utils.flat_params(model))

    quantized = quantization.maybe_quantize(model, config, *example)

    self.assertEqual(quantization.quantized_matmuls(quantized, jnp.int8, *example), len(plan.quantized))
    per_projection = quantization.quantized_matmuls_by_module(quantized, plan.quantized, jnp.int8, *example)
    self.assertEqual(set(per_projection.values()), {1}, per_projection)
    # The attention scores and the tied-embedding logits are the matmuls left over; they are untouched.
    self.assertEqual(quantization.quantized_matmuls(quantized, jnp.bfloat16, *example), bf16_before - len(plan.quantized))
    self.assertIsInstance(quantized, type(model))
    self.assertEqual(list(m3_test_utils.flat_params(quantized)), params_before)
    self.assertEqual({type(v).__name__ for _, v in nnx.to_flat_state(nnx.state(quantized))}, {"Param"})

  def test_int8_forward_and_backward_stay_close_to_bf16(self):
    config, model = self._model(quantization="int8")
    tokens = jax.random.randint(jax.random.key(0), (2, 9), 0, config.vocab_size)
    batch = {"inputs": tokens[:, :-1], "targets": tokens[:, 1:], "positions": jnp.broadcast_to(jnp.arange(8), (2, 8))}
    reference = model(batch["inputs"], batch["positions"])

    quantized = quantization.maybe_quantize(model, config, *quantization.example_inputs(config))
    logits = quantized(batch["inputs"], batch["positions"])
    self.assertFalse(np.array_equal(np.asarray(logits), np.asarray(reference)), "int8 had no effect")
    np.testing.assert_allclose(logits, reference, rtol=0.2, atol=0.5)

    loss, grads = nnx.value_and_grad(loop.loss_fn)(quantized, batch)
    self.assertTrue(np.isfinite(float(loss)))
    self.assertEqual(list(m3_test_utils.flat_params(grads)), list(m3_test_utils.flat_params(model)))
    self.assertTrue(all(np.all(np.isfinite(g)) for g in m3_test_utils.flat_params(grads).values()))

  @parameterized.parameters(
      ("int4", _INT4, _INT4),
      ("int8", jnp.dtype(jnp.int8), jnp.dtype(jnp.int8)),
      ("fp4", _E2M1, _E2M1),
      ("fp8", _E4M3, _E4M3),
      ("fp8_full", _E4M3, _E5M2),
      ("fp8_e4m3", _E4M3, _E4M3),
      ("fp8_e5m2", _E5M2, _E5M2),
  )
  def test_recipes_use_their_dtypes_forward_and_backward(self, name, forward, gradient):
    config, model = self._model(quantization=name)
    example = quantization.example_inputs(config)
    quantized = quantization.maybe_quantize(model, config, *example)
    planned = len(quantization.plan(model).quantized)
    self.assertEqual(quantization.quantized_matmuls(quantized, forward, *example), planned)

    graphdef, state = nnx.split(quantized)

    def loss(state, tokens, positions):
      return jnp.sum(nnx.merge(graphdef, state)(tokens, positions).astype(jnp.float32))

    pairs = quantization.matmul_dtypes(jax.grad(loss), state, *example)
    # The grad trace repeats the forward matmuls; each projection then adds two with the gradient dtype.
    with_gradient = sum(n for pair, n in pairs.items() if gradient in pair)
    self.assertEqual(with_gradient - (planned if gradient == forward else 0), 2 * planned, pairs)
    for unused in {_E4M3, _E5M2, _INT4, _E2M1} - {forward, gradient}:
      self.assertNotIn(unused, {d for pair in pairs for d in pair})

  @parameterized.parameters(
      ("quantize_logits_proj", "true", {"quantization": "fp8_full", "logits_via_embedding": "false"}),
      ("qwix_bwd_weight_grad_tile_count", "4", {}),
      ("act_quantization_calibration_method", "rms,0.8", {}),
      ("weight_sparsity_n", "2", {}),
      ("weight_block_size", "128", {}),
  )
  def test_legacy_knob_m3_does_not_implement_is_refused(self, field, value, prerequisites):
    """A real pyconfig, so the field names are checked against the config that will carry them.

    `prerequisites` are what pyconfig's own validators demand before they accept the knob.
    """
    config, model = self._model(**{"quantization": "int8", **prerequisites, field: value})
    self.assertIn(field, quantization.unsupported_settings(config))
    with self.assertRaisesRegex(ValueError, f"m3 doesn't support.*{field}"):
      quantization.maybe_quantize(model, config, *quantization.example_inputs(config))

  def test_unquantized_modules_is_refused(self):
    config = SimpleNamespace(quantization="int8", unquantized_modules=["gate"])
    self.assertEqual(quantization.unsupported_settings(config), {"unquantized_modules": ["gate"]})

  def test_aqt_default_is_refused(self):
    """Legacy trains `quantization=int8` with AQT unless `use_qwix_quantization=true`."""
    config, model = self._model(quantization="int8", use_qwix_quantization="false")
    with self.assertRaisesRegex(ValueError, "use_qwix_quantization=true"):
      quantization.maybe_quantize(model, config, *quantization.example_inputs(config))

  def test_router_that_wraps_a_linear(self):
    """Qwen3-Next's router holds `self.linear`; turning routers off must skip `gate/linear` too."""
    model = _Model(nnx.Rngs(0))
    model.decoder.layers_0.gate = _Wrapped(nnx.Linear(4, 2, rngs=nnx.Rngs(0)))
    off = quantization.plan(model, quantize_routers=False)
    self.assertNotIn("decoder/layers_0/gate/linear", off.quantized)
    self.assertIn("decoder/layers_0/gate", off.routers)
    self.assertIn("decoder/layers_0/gate/linear", quantization.plan(model).quantized)

  def test_legacy_defaults_and_absent_fields_pass(self):
    """Legacy's defaults are m3's recipe; a slim config without the fields is treated the same."""
    config, _ = self._model(quantization="int8")
    self.assertEqual(quantization.unsupported_settings(config), {})
    self.assertEqual(quantization.unsupported_settings(SimpleNamespace(quantization="int8")), {})

  def test_unknown_recipe_raises(self):
    with self.assertRaisesRegex(ValueError, "m3 supports quantization"):
      quantization.recipe("int2")

  def test_model_without_a_decoder_stack_is_refused(self):
    x = jax.ShapeDtypeStruct((2, 4), jnp.float32)
    with self.assertRaisesRegex(ValueError, "no projections under decoder/layers"):
      quantization.maybe_quantize(_Flat(nnx.Rngs(0)), SimpleNamespace(quantization="int8"), x)

  @parameterized.named_parameters(("once", False), ("other_called_twice", True))
  def test_projection_that_is_never_called_is_an_error(self, twice):
    """Checked per projection: a total count would pass when another projection runs twice."""
    x = jax.ShapeDtypeStruct((2, 4), jnp.float32)
    config = SimpleNamespace(quantization="int8", quantize_router_proj=False)
    model = _Model(nnx.Rngs(0), unused=True, twice=twice)
    with self.assertRaisesRegex(RuntimeError, r"do no int8 matmul.*\['decoder/layers_0/unused'\]"):
      quantization.maybe_quantize(model, config, x)

  def test_scanned_stack_is_quantized_inside_the_scan_body(self):
    """Scan runs first (`setup_model`); the stack is then one stacked layer, and the census walks the scan body."""
    config, model = self._model(quantization="int8", scan_layers="true")
    per_layer = len(quantization.plan(model).quantized) // config.num_decoder_layers
    example = quantization.example_inputs(config)

    scanned = m3_scan.maybe_apply_scan(model, config)
    plan = quantization.plan(scanned)
    self.assertLen(plan.quantized, per_layer)
    self.assertTrue(all(path.startswith("decoder/layers/") for path in plan.quantized), plan.quantized)

    quantized = quantization.maybe_quantize(scanned, config, *example)
    self.assertEqual(quantization.quantized_matmuls(quantized, jnp.int8, *example), per_layer)
    self.assertEqual(list(m3_test_utils.flat_params(quantized)), list(m3_test_utils.flat_params(scanned)))

  @parameterized.parameters(
      ("jax_mxfp8", "", _E4M3, _E4M3, "mxfp8", "mxfp8"),
      ("jax_mxfp4", "", _E2M1, _E2M1, "mxfp4", "mxfp4"),
      ("jax_mxfp4", "mxfp8", _E2M1, _E4M3, "mxfp4", "mxfp8"),
      ("jax_mxfp8", "mxfp4", _E4M3, _E2M1, "mxfp8", "mxfp4"),
  )
  def test_mx_recipe_resolves_modes_and_dtypes(self, name, gradient_mode, fwd, grad, mx_mode, mx_grad_mode):
    r = quantization.recipe(name, gradient_mode=gradient_mode)
    self.assertEqual((r.forward, r.gradient, r.mx_mode, r.mx_gradient_mode), (fwd, grad, mx_mode, mx_grad_mode))

  def test_mx_gradient_mode_rejected_for_non_mx_or_unknown(self):
    with self.assertRaisesRegex(ValueError, "applies only to jax_mxfp8 / jax_mxfp4"):
      quantization.recipe("fp8", gradient_mode="mxfp8")
    with self.assertRaisesRegex(ValueError, "jax_mx_gradient_mode must be in"):
      quantization.recipe("jax_mxfp8", gradient_mode="int8")
    model = _MxStack(64, 64, nnx.Rngs(0))
    x = jax.ShapeDtypeStruct((1, 32, 64), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "applies only to jax_mxfp8 / jax_mxfp4"):
      quantization.maybe_quantize(model, SimpleNamespace(quantization="", jax_mx_gradient_mode="mxfp8"), x)

  def test_mx_refuses_use_qwix_quantization_true(self):
    model = _MxStack(64, 64, nnx.Rngs(0))
    x = jax.ShapeDtypeStruct((1, 32, 64), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "does not support use_qwix_quantization=true"):
      quantization.maybe_quantize(model, SimpleNamespace(quantization="jax_mxfp8", use_qwix_quantization=True), x)

  def test_mx_refuses_multi_chip_tpu(self):
    model = _MxStack(64, 64, nnx.Rngs(0))
    x = jax.ShapeDtypeStruct((1, 32, 64), jnp.bfloat16)
    fake_mesh = jax.sharding.AbstractMesh((2,), ("fsdp",))
    with (
        mock.patch.object(jax.sharding, "get_abstract_mesh", return_value=fake_mesh),
        mock.patch.object(quantization.m3_kernels, "is_tpu_mesh", return_value=True),
    ):
      with self.assertRaisesRegex(ValueError, "multi-chip TPU"):
        quantization.maybe_quantize(model, SimpleNamespace(quantization="jax_mxfp8"), x)

  def test_mx_unsupported_jax_mode_raises_clear_error(self):
    model = _MxStack(64, 64, nnx.Rngs(0))
    x = jax.ShapeDtypeStruct((1, 32, 64), jnp.bfloat16)
    with mock.patch.object(jax.nn, "get_scaled_dot_general_config", side_effect=ValueError("Unsupported mode")):
      with self.assertRaisesRegex(ValueError, r"scaled_dot_general mode 'mxfp4'.*jax>=0\.12\.0"):
        quantization.maybe_quantize(model, SimpleNamespace(quantization="jax_mxfp4"), x)

  @parameterized.parameters(
      ("jax_mxfp8", "", _E4M3, _E4M3),
      ("jax_mxfp4", "", _E2M1, _E2M1),
      ("jax_mxfp4", "mxfp8", _E2M1, _E4M3),
  )
  def test_jax_mx_quantizes_each_planned_projection_forward_and_backward(self, name, gradient_mode, forward, gradient):
    overrides = {"quantization": name}
    if gradient_mode:
      overrides["jax_mx_gradient_mode"] = gradient_mode
    config, model = self._model(**overrides)
    example = quantization.example_inputs(config)
    the_plan = quantization.plan(model)
    bf16_before = quantization.quantized_matmuls(model, jnp.bfloat16, *example)
    params_before = list(m3_test_utils.flat_params(model))

    quantized = quantization.maybe_quantize(model, config, *example)

    planned = len(the_plan.quantized)
    self.assertEqual(quantization.quantized_matmuls(quantized, forward, *example), planned)
    self.assertEqual(quantization.quantized_matmuls(quantized, jnp.bfloat16, *example), bf16_before - planned)
    self.assertEqual(list(m3_test_utils.flat_params(quantized)), params_before)
    self.assertEqual({type(v).__name__ for _, v in nnx.to_flat_state(nnx.state(quantized))}, {"Param"})

    # Verify every quantized op is a block-scaled MX matmul with E8M0 scales.
    fwd_jaxpr = quantization._forward_jaxpr(quantized, example, {})  # pylint: disable=protected-access
    mx_eqns = [e for e in quantization._eqns(fwd_jaxpr) if e.primitive.name == "scaled_matmul_wrapper"]  # pylint: disable=protected-access
    self.assertLen(mx_eqns, planned)
    for eqn in mx_eqns:
      self.assertEqual(eqn.invars[0].aval.dtype, forward)
      self.assertEqual(eqn.invars[1].aval.dtype, forward)
      self.assertEqual(eqn.invars[2].aval.dtype, _E8M0)
      self.assertEqual(eqn.invars[3].aval.dtype, _E8M0)

    graphdef, state = nnx.split(quantized)

    def loss(s, tokens, positions):
      return jnp.sum(nnx.merge(graphdef, s)(tokens, positions).astype(jnp.float32))

    pairs = quantization.matmul_dtypes(jax.grad(loss), state, *example)
    with_gradient = sum(n for pair, n in pairs.items() if gradient in pair)
    self.assertEqual(with_gradient - (planned if gradient == forward else 0), 2 * planned, pairs)

  def test_jax_mx_scanned_stack_and_einsum(self):
    config, model = self._model(quantization="jax_mxfp8", scan_layers="true")
    per_layer = len(quantization.plan(model).quantized) // config.num_decoder_layers
    example = quantization.example_inputs(config)
    scanned = m3_scan.maybe_apply_scan(model, config)
    quantized = quantization.maybe_quantize(scanned, config, *example)
    self.assertEqual(quantization.quantized_matmuls(quantized, _E4M3, *example), per_layer)
    self.assertEqual(list(m3_test_utils.flat_params(quantized)), list(m3_test_utils.flat_params(scanned)))

    einsum_model = _MxStack(64, 128, nnx.Rngs(0), use_einsum=True)
    x = jax.ShapeDtypeStruct((1, 32, 64), jnp.bfloat16)
    quantized_einsum = quantization.maybe_quantize(einsum_model, SimpleNamespace(quantization="jax_mxfp8"), x)
    self.assertEqual(quantization.quantized_matmuls(quantized_einsum, _E4M3, x), 1)

  def test_jax_mx_unquantized_einsum_forwards_kwargs(self):
    """Unquantized `jnp.einsum` calls under an MX-quantized model must forward extra kwargs like `optimize`."""

    class _WithExtraEinsum(_MxStack):

      def __call__(self, x, *, out_sharding=None):
        y = super().__call__(x, out_sharding=out_sharding)
        return jnp.einsum("btd,bsd->bts", y, y, optimize="greedy")

    model = _WithExtraEinsum(64, 64, nnx.Rngs(0))
    x = jax.ShapeDtypeStruct((1, 32, 64), jnp.bfloat16)
    quantized = quantization.maybe_quantize(model, SimpleNamespace(quantization="jax_mxfp8"), x)
    self.assertEqual(quantization.quantized_matmuls(quantized, _E4M3, x), 1)
    self.assertEqual(quantization.quantized_matmuls(quantized, jnp.bfloat16, x), 1)

  @parameterized.named_parameters(
      ("forward_linear", "forward", 48, 64, (2, 32), False),
      ("dx_linear", "input gradient", 64, 48, (2, 32), False),
      ("dw_linear", "weight gradient", 64, 64, (2, 24), False),
      ("forward_einsum", "forward", 48, 64, (2, 32), True),
  )
  def test_jax_mx_contracting_dims_checked_per_gemm(self, gemm, in_features, out_features, tokens, use_einsum):
    model = _MxStack(in_features, out_features, nnx.Rngs(0), use_einsum=use_einsum)
    x = jax.ShapeDtypeStruct((*tokens, in_features), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, rf"the {gemm} \(.*\) GEMM contracts over 48 elements"):
      quantization.maybe_quantize(model, SimpleNamespace(quantization="jax_mxfp8"), x)

  @parameterized.parameters(False, True)
  def test_jax_mx_refuses_explicit_out_sharding(self, use_einsum):
    model = _MxStack(64, 64, nnx.Rngs(0), use_einsum=use_einsum)
    x = jax.ShapeDtypeStruct((1, 32, 64), jnp.bfloat16)
    quantized = quantization.maybe_quantize(model, SimpleNamespace(quantization="jax_mxfp8"), x)
    with self.assertRaisesRegex(ValueError, "does not support explicit out_sharding"):
      jax.make_jaxpr(lambda t: quantized(t, out_sharding=P(None, None, None)))(x)

  @parameterized.parameters(
      ("jax_mxfp8", "", False),
      ("jax_mxfp8", "", True),
      ("jax_mxfp4", "", False),
      ("jax_mxfp4", "mxfp8", False),
  )
  def test_jax_mx_matches_scaled_dot_general(self, name, gradient_mode, use_einsum):
    mode = name.removeprefix("jax_")
    grad_mode = gradient_mode or mode
    model = _MxStack(64, 128, nnx.Rngs(0), use_einsum=use_einsum)
    x = jax.random.normal(jax.random.key(1), (2, 32, 64), jnp.float32).astype(jnp.bfloat16)
    cfg = SimpleNamespace(quantization=name, jax_mx_gradient_mode=gradient_mode)
    quantized = quantization.maybe_quantize(model, cfg, jax.ShapeDtypeStruct(x.shape, x.dtype))

    kernel = model.decoder.layers[0].proj.kernel[...].astype(jnp.bfloat16)
    fwd_cfg = jax.nn.get_scaled_dot_general_config(mode)
    bwd_cfg = jax.nn.get_scaled_dot_general_config(grad_mode)

    def reference(lhs, rhs=kernel):
      return jax.nn.scaled_dot_general(
          lhs,
          rhs,
          (((2,), (0,)), ((), ())),
          preferred_element_type=jnp.bfloat16,
          configs=[fwd_cfg, fwd_cfg, bwd_cfg],
      )

    def loss(fn, lhs):
      return (fn(lhs).astype(jnp.float32) ** 2).sum()

    y = quantized(x)
    self.assertEqual(y.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(y, np.float32), np.asarray(reference(x), np.float32))
    dx = jax.grad(loss, argnums=1)(quantized, x)
    dx_ref = jax.grad(loss, argnums=1)(reference, x)
    np.testing.assert_array_equal(np.asarray(dx, np.float32), np.asarray(dx_ref, np.float32))
    dk = nnx.grad(lambda m: loss(m, x))(quantized)["decoder"]["layers"][0]["proj"]["kernel"][...]
    dk_ref = jax.grad(lambda w: loss(lambda lhs: reference(lhs, w), x))(kernel)
    np.testing.assert_array_equal(np.asarray(dk, np.float32), np.asarray(dk_ref, np.float32))


class QuantizedLoopTest(parameterized.TestCase):
  """`setup_model` composes scan and the transform; the step must compile and run on the quantized model."""

  @parameterized.parameters(False, True)
  def test_train_loop_runs_with_int8(self, scan_layers):
    config = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        quantization="int8",
        use_qwix_quantization="true",
        scan_layers=scan_layers,
        steps=2,
        base_output_directory=self.create_tempdir().full_path,
        run_name="m3_quantization_test",
    )
    self.assertTrue(np.isfinite(loop.train_loop(config)))

  @parameterized.parameters(
      ("jax_mxfp8", False),
      ("jax_mxfp8", True),
      ("jax_mxfp4", False),
  )
  def test_train_loop_runs_with_jax_mx(self, name, scan_layers):
    config = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        quantization=name,
        scan_layers=scan_layers,
        steps=2,
        base_output_directory=self.create_tempdir().full_path,
        run_name=f"m3_quantization_{name}",
    )
    self.assertTrue(np.isfinite(loop.train_loop(config)))


@pytest.mark.tpu_only
class QuantizedTrainingTpuTest(parameterized.TestCase):
  """Quantized matmuls have to survive compilation; CPU traces cannot show that."""

  @parameterized.parameters(
      ("int4", "s4"), ("int8", "s8"), ("fp4", "f4e2m1fn"), ("fp8", "f8e4m3fn"), ("fp8_e5m2", "f8e5m2")
  )
  def test_compiled_matmuls_take_quantized_operands(self, name, hlo_type):
    device = jax.devices()[0]
    # TPU7x compiles int4 dots as e4m3 (exact for int4 values); v6e keeps them s4.
    if hlo_type == "s4" and device.device_kind == "TPU7x":
      hlo_type = "f8e4m3fn"
    mesh = create_mesh(devices=[device])
    with jax.set_mesh(mesh):
      config = m3_test_utils.tiny_config("qwen3-0.6b", quantization=name, use_qwix_quantization="true")
      example = quantization.example_inputs(config)
      model = m3_models.create_model(config, mesh, rngs=nnx.Rngs(0))
      planned = len(quantization.plan(model).quantized)
      model = quantization.maybe_quantize(model, config, *example)
      graphdef, state = nnx.split(model)
      hlo = jax.jit(lambda s, t, p: nnx.merge(graphdef, s)(t, p)).lower(state, *example).compile().as_text()
    # Operands are named, so look up the type each name was defined with.
    types = dict(re.findall(r"(%[\w.\-]+) = (\w+)\[", hlo))
    quantized = 0
    for line in hlo.splitlines():
      op = next((op for op in (" convolution(", " dot(") if op in line), None)
      if op:
        operands = re.findall(r"%[\w.\-]+", line.split(op, 1)[1].split(")", 1)[0])
        quantized += bool(operands) and all(types.get(o) == hlo_type for o in operands)
    self.assertEqual(quantized, planned)

  @parameterized.parameters("int4", "int8", "fp4", "fp8")
  def test_train_loop_runs(self, name):
    config = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        quantization=name,
        use_qwix_quantization="true",
        steps=3,
        base_output_directory=self.create_tempdir().full_path,
        run_name=f"m3_quantization_{name}",
    )
    self.assertTrue(np.isfinite(loop.train_loop(config)))

  @parameterized.parameters("jax_mxfp8", "jax_mxfp4")
  def test_jax_mx_train_step_runs_single_chip(self, name):
    """Single-chip TPU lowers `scaled_dot_general` via the dequantize fallback."""
    mesh = create_mesh(devices=[jax.devices()[0]])
    with jax.set_mesh(mesh):
      config = m3_test_utils.tiny_config("qwen3-0.6b", quantization=name)
      model, optimizer = loop.setup_model(config, mesh)
      spec = sharding_from_config(config)
      for step in range(2):
        batch = loop.make_synthetic_batch(config, mesh, spec, jax.random.key(step))
        loss = loop.train_step(model, optimizer, batch)
        self.assertTrue(np.isfinite(float(loss)))

  @parameterized.parameters("jax_mxfp8", "jax_mxfp4")
  def test_jax_mx_refuses_multi_chip_tpu(self, name):
    """Multi-chip TPU fails fast instead of hitting missing `CustomSPMDPartitioning` emitter."""
    if len(jax.devices()) < 2:
      self.skipTest("requires at least 2 TPU devices")
    config = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        quantization=name,
        steps=1,
        base_output_directory=self.create_tempdir().full_path,
        run_name=f"m3_quantization_{name}_multi_chip",
    )
    with self.assertRaisesRegex(ValueError, "multi-chip TPU"):
      loop.train_loop(config)


if __name__ == "__main__":
  absltest.main()
