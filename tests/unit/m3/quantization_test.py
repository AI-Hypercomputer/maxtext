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

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.m3 import models as m3_models
from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.infra import quantization
from maxtext.m3.infra import scan as m3_scan
from maxtext.m3.train import loop
from tests.utils import m3_test_utils

_E4M3, _E5M2 = jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2)


class _HandRolled(nnx.Module):
  """A weight applied with `jnp.einsum` directly, the way a ported `DenseGeneral` or a custom router does it."""

  def __init__(self, features, rngs):
    self.kernel = nnx.Param(jax.random.normal(rngs.params(), (4, features)))

  def __call__(self, x):
    return jnp.einsum("...i,io->...o", x, self.kernel[...])


class _Block(nnx.Module):
  """A decoder block with one projection and one router.

  `unused` adds a projection `__call__` never uses; `handrolled` makes the
  projection a `_HandRolled` instead of an `nnx.Linear`.
  """

  def __init__(self, rngs, unused=False, handrolled=False):
    self.wi = _HandRolled(4, rngs) if handrolled else nnx.Linear(4, 4, rngs=rngs)
    self.gate = nnx.Linear(4, 2, rngs=rngs)
    if unused:
      self.unused = nnx.Linear(4, 4, rngs=rngs)

  def __call__(self, x):
    return self.wi(x) * jnp.sum(self.gate(x), axis=-1, keepdims=True)


class _Decoder(nnx.Module):
  """Both m3 layouts of the stack -- a flat `layers_0` and a `layers` container -- plus an untied head.

  The container's block has a hand-rolled router, as DeepSeek-V3's is.
  """

  def __init__(self, rngs, unused=False, handrolled=False):
    self.layers_0 = _Block(rngs, unused, handrolled)
    self.layers = nnx.List([_Block(rngs)])
    self.layers[0].gate = _HandRolled(2, rngs)
    self.logits_dense = nnx.Linear(4, 8, rngs=rngs)

  def __call__(self, x):
    return self.logits_dense(self.layers[0](self.layers_0(x)))


class _Model(nnx.Module):

  def __init__(self, rngs, unused=False, handrolled=False):
    self.decoder = _Decoder(rngs, unused, handrolled)

  def __call__(self, x):
    return self.decoder(x)


class _Flat(nnx.Module):
  """A projection with no decoder stack above it."""

  def __init__(self, rngs):
    self.proj = nnx.Linear(4, 4, rngs=rngs)

  def __call__(self, x):
    return self.proj(x)


class QuantizationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = create_mesh(devices=jax.devices()[:1])
    self.enter_context(jax.set_mesh(self.mesh))

  def _model(self, **overrides):
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
    plan = quantization.plan(_Model(nnx.Rngs(0)))
    self.assertEqual(set(plan.quantized), {"decoder/layers_0/wi", "decoder/layers/0/wi"})
    # Both routers: an `nnx.Linear` and a hand-rolled one. Neither is quantized, neither is a complaint.
    self.assertEqual(set(plan.routers), {"decoder/layers_0/gate", "decoder/layers/0/gate"})
    self.assertEmpty(plan.uncovered)
    self.assertIn("2 projections", plan.describe())

  def test_hand_rolled_projection_in_the_stack_is_refused(self):
    """A weight the plan cannot see must not be left in full precision silently (Qwen3-VL's `DenseGeneral` today)."""
    model = _Model(nnx.Rngs(0), handrolled=True)
    self.assertEqual(quantization.plan(model).uncovered, ("decoder/layers_0/wi",))
    x = jax.ShapeDtypeStruct((2, 4), jnp.float32)
    with self.assertRaisesRegex(ValueError, r"not NNX projection layers.*decoder/layers_0/wi"):
      quantization.maybe_quantize(model, SimpleNamespace(quantization="int8"), x)

  def test_rule_matches_leaves_only(self):
    """qwix matches with `re.fullmatch`; a parent's own einsums must not match its children's rule."""
    (rule,) = quantization.rules(quantization.plan(_Model(nnx.Rngs(0))), quantization.recipe("int8"))
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
      ("fp8", _E4M3, _E5M2),
      ("fp8_full", _E4M3, _E5M2),
      ("fp8_e4m3", _E4M3, _E4M3),
      ("fp8_e5m2", _E5M2, _E5M2),
  )
  def test_fp8_recipes_use_their_dtypes_forward_and_backward(self, name, forward, gradient):
    config, model = self._model(quantization=name)
    example = quantization.example_inputs(config)
    quantized = quantization.maybe_quantize(model, config, *example)
    planned = len(quantization.plan(model).quantized)
    self.assertEqual(quantization.quantized_matmuls(quantized, forward, *example), planned)

    graphdef, state = nnx.split(quantized)

    def loss(state, tokens, positions):
      return jnp.sum(nnx.merge(graphdef, state)(tokens, positions).astype(jnp.float32))

    dtypes = {d for pair in quantization.matmul_dtypes(jax.grad(loss), state, *example) for d in pair}
    self.assertIn(gradient, dtypes)
    for unused in {_E4M3, _E5M2} - {forward, gradient}:
      self.assertNotIn(unused, dtypes)

  @parameterized.parameters(
      ("quantize_logits_proj", "true", {"quantization": "fp8_full", "logits_via_embedding": "false"}),
      ("qwix_bwd_weight_grad_tile_count", "4", {}),
      ("act_quantization_calibration_method", "rms,0.8", {}),
      ("weight_sparsity_n", "2", {}),
  )
  def test_legacy_knob_m3_does_not_implement_is_refused(self, field, value, prerequisites):
    """A real pyconfig, so the field names are checked against the config that will carry them.

    `prerequisites` are what pyconfig's own validators demand before they accept the knob.
    """
    config, model = self._model(**{"quantization": "int8", **prerequisites, field: value})
    self.assertIn(field, quantization.unsupported_settings(config))
    with self.assertRaisesRegex(ValueError, f"m3 does not implement.*{field}"):
      quantization.maybe_quantize(model, config, *quantization.example_inputs(config))

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
    with self.assertRaisesRegex(ValueError, "nothing would be quantized"):
      quantization.maybe_quantize(_Flat(nnx.Rngs(0)), SimpleNamespace(quantization="int8"), x)

  def test_projection_that_is_never_called_is_an_error(self):
    x = jax.ShapeDtypeStruct((2, 4), jnp.float32)
    with self.assertRaisesRegex(RuntimeError, "planned 3 projections"):
      quantization.maybe_quantize(_Model(nnx.Rngs(0), unused=True), SimpleNamespace(quantization="int8"), x)

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


class QuantizedLoopTest(parameterized.TestCase):
  """`setup_model` composes scan and the transform; the step must compile and run on the quantized model."""

  @parameterized.parameters(False, True)
  def test_train_loop_runs_with_int8(self, scan_layers):
    config = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        quantization="int8",
        scan_layers=scan_layers,
        steps=2,
        base_output_directory=self.create_tempdir().full_path,
        run_name="m3_quantization_test",
    )
    self.assertTrue(np.isfinite(loop.train_loop(config)))


@pytest.mark.tpu_only
class QuantizedTrainingTpuTest(parameterized.TestCase):
  """int8 and fp8 matmuls have to survive compilation; CPU traces cannot show that."""

  @parameterized.parameters(("int8", "s8"), ("fp8", "f8e4m3fn"))
  def test_compiled_forward_carries_quantized_arrays(self, name, hlo_type):
    mesh = create_mesh(devices=jax.devices()[:1])
    with jax.set_mesh(mesh):
      config = m3_test_utils.tiny_config("qwen3-0.6b", quantization=name)
      example = quantization.example_inputs(config)
      model = quantization.maybe_quantize(m3_models.create_model(config, mesh, rngs=nnx.Rngs(0)), config, *example)
      graphdef, state = nnx.split(model)
      hlo = jax.jit(lambda s, t, p: nnx.merge(graphdef, s)(t, p)).lower(state, *example).compile().as_text()
    self.assertIn(f"{hlo_type}[", hlo)

  @parameterized.parameters("int8", "fp8")
  def test_train_loop_runs(self, name):
    config = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        quantization=name,
        steps=3,
        base_output_directory=self.create_tempdir().full_path,
        run_name=f"m3_quantization_{name}",
    )
    self.assertTrue(np.isfinite(loop.train_loop(config)))


if __name__ == "__main__":
  absltest.main()
