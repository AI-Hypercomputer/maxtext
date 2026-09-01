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

"""Guards against the NNX scan_layers=False throughput regression.

ToNNX forks the caller's nnx.Rngs into a module attribute, so it lands in the model
state and its counter is incremented on device every call. MaxText bridges one of
these per quantized DenseGeneral, so with scan_layers=False the cost is paid once per
layer: llama3-8b ended up with 672 streams and 673 extra kernel launches per step,
dropping GPU utilization from 86.0% to 76.7%.

The tests use stub backends so they run on CPU without TransformerEngine. They pin
down two properties: a backend that declares it does not draw at apply time leaves no
RNG state behind, and that state does not grow with the layer count.
"""

import types
import unittest

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import errors as flax_errors
from flax import nnx

from maxtext.layers import linears
from maxtext.layers import quantizations


class _StubDotGeneral(nn.Module):
  """Minimal stand-in for a quantized dot_general Linen module.

  Declares no variables, like TE's fp8 dense path, so the only state a bridge around
  it can contribute is the forked `Rngs` itself.
  """

  @nn.compact
  def __call__(self, inputs, kernel, dims, precision=None, **kwargs):
    del kwargs
    contracting, batch = dims
    return jax.lax.dot_general(inputs, kernel, (contracting, batch), precision=precision)


class _OptOutQuant(quantizations.Quantization):
  """A backend that never draws RNGs at apply time (as TransformerEngine does not)."""

  needs_apply_rngs = False
  quant_mode = "train"  # read by quantizations.in_serve_mode()

  def dot_general_cls(self, mesh_axes=()):
    del mesh_axes
    return _StubDotGeneral


class _DefaultQuant(quantizations.Quantization):
  """A backend that leaves `needs_apply_rngs` at its safe default of True."""

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes=()):
    del mesh_axes
    return _StubDotGeneral


class _SrRngDotGeneral(nn.Module):
  """Stand-in for NVFP4's dot_general, which draws `sr_rng` at apply time."""

  @nn.compact
  def __call__(self, inputs, kernel, dims, precision=None, **kwargs):
    del kwargs
    self.make_rng("sr_rng")
    contracting, batch = dims
    return jax.lax.dot_general(inputs, kernel, (contracting, batch), precision=precision)


class _DrawingQuant(quantizations.Quantization):
  """A backend that draws at apply time, as NVFP4 stochastic rounding does."""

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes=()):
    del mesh_axes
    return _SrRngDotGeneral


class _MisdeclaredQuant(_DrawingQuant):
  """A drawing backend that wrongly claims it does not draw."""

  needs_apply_rngs = False


def _rng_state_paths(module) -> list[str]:
  """Paths of every RNG leaf reachable in the module's state.

  Filters on `nnx.RngState`, the variable type NNX gives RNG keys and counters, so
  this keeps working if a module renames the attribute holding its `Rngs`.
  """
  return [".".join(str(p) for p in path) for path, _ in nnx.to_flat_state(nnx.state(module, nnx.RngState))]


def _make_dense(quant, seed: int = 0) -> linears.DenseGeneral:
  return linears.DenseGeneral(
      in_features_shape=8,
      out_features_shape=4,
      quant=quant,
      rngs=nnx.Rngs(params=seed, dropout=seed + 1, aqt=seed + 2),
  )


class QuantBridgeRngStateTest(unittest.TestCase):
  """The bridged quantization wrapper must not smuggle RNG state into the model."""

  def test_opt_out_backend_leaves_no_rng_state(self):
    dense = _make_dense(_OptOutQuant())
    self.assertEqual(
        _rng_state_paths(dense),
        [],
        "A backend with needs_apply_rngs=False must not retain the bridge's forked "
        "Rngs: its counters would be incremented on device every step, once per "
        "unrolled layer.",
    )

  def test_default_backend_keeps_rng_state(self):
    """Checks that the opt-out is deliberate and the default stays safe."""
    paths = _rng_state_paths(_make_dense(_DefaultQuant()))
    self.assertNotEqual(paths, [], "needs_apply_rngs defaults to True, so the Rngs must be kept.")

  def test_unquantized_dense_has_no_bridge_state(self):
    self.assertEqual(_rng_state_paths(_make_dense(None)), [])

  def test_rng_state_does_not_grow_with_layer_count(self):
    """Checks the regression's signature: state scaling with the unrolled layer count.

    A scanned decoder traces one layer body, so a per-wrapper leak stays hidden. An
    unrolled one materializes every layer, so counting across a stack catches the
    regression even if the mechanism changes.
    """
    counts = []
    for num_layers in (1, 2, 8):
      layers = [_make_dense(_OptOutQuant(), seed=i) for i in range(num_layers)]
      counts.append(sum(len(_rng_state_paths(layer)) for layer in layers))

    self.assertEqual(
        counts,
        [0, 0, 0],
        f"Bridged RNG state must not scale with the number of unrolled layers, got {counts} " "for 1/2/8 layers.",
    )

  def test_release_rngs_is_idempotent_and_keeps_the_layer_callable(self):
    """Releasing must not break apply; the wrapped module still has to run."""
    dense = _make_dense(_OptOutQuant())
    bridge = dense.quant_dot_general
    self.assertIsNotNone(bridge)
    bridge.release_rngs()  # already released during __init__; doing it again is fine

    out = dense(jnp.ones((2, 8), jnp.float32))
    self.assertEqual(out.shape, (2, 4))
    self.assertTrue(jnp.all(jnp.isfinite(out)))

  def test_opt_out_and_default_agree_numerically(self):
    """Dropping the Rngs is a state change, not a numerics change."""
    x = jnp.ones((2, 8), jnp.float32)
    opt_out = _make_dense(_OptOutQuant(), seed=7)(x)
    default = _make_dense(_DefaultQuant(), seed=7)(x)
    self.assertTrue(jnp.allclose(opt_out, default), "Releasing the bridge's Rngs must not change the output.")


class DropoutRngStateTest(unittest.TestCase):
  """A zero-rate Dropout must not carry RNG state, but must still advance the caller."""

  def test_zero_rate_dropout_holds_no_rngs(self):
    d = linears.Dropout(rate=0.0, rngs=nnx.Rngs(params=0, dropout=1, aqt=2))
    self.assertEqual(
        _rng_state_paths(d),
        [],
        "nnx.Dropout returns its input before touching self.rngs at rate 0, so holding "
        "a fork only adds (key, count) pairs to the model state.",
    )

  def test_nonzero_rate_dropout_keeps_rngs(self):
    d = linears.Dropout(rate=0.1, rngs=nnx.Rngs(params=0, dropout=1, aqt=2))
    self.assertNotEqual(_rng_state_paths(d), [], "a drawing Dropout must keep its RNGs")

  def test_zero_rate_dropout_still_advances_the_caller(self):
    """Dropping the fork must not shift later draws, or parameter init changes."""
    counts = {}
    for rate in (0.0, 0.1):
      rngs = nnx.Rngs(params=0, dropout=1, aqt=2)
      linears.Dropout(rate=rate, rngs=rngs)
      counts[rate] = {name: int(stream.count[...]) for name, stream in rngs.items()}
    self.assertEqual(
        counts[0.0],
        counts[0.1],
        "Rngs.fork() advances the caller's streams; a zero-rate Dropout must advance "
        "them identically so downstream initialization is unchanged.",
    )

  def test_zero_rate_dropout_is_a_passthrough(self):
    d = linears.Dropout(rate=0.0, rngs=nnx.Rngs(params=0, dropout=1, aqt=2))
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    self.assertTrue(jnp.array_equal(d(x, deterministic=False), x))


class QuantizationFlagTest(unittest.TestCase):
  """`needs_apply_rngs` must stay safe-by-default and correct per backend."""

  def test_base_class_defaults_to_keeping_rngs(self):
    self.assertTrue(quantizations.Quantization.needs_apply_rngs)

  def test_backends_that_may_draw_at_apply_time_keep_rngs(self):
    """AQT's config enables jax.uniform RNG, so it must never be opted out silently."""
    for cls in (
        quantizations.AqtQuantization,
        quantizations.QwixQuantization,
        quantizations.Fp8Quantization,
        quantizations.NANOOFp8Quantization,
    ):
      self.assertTrue(cls.needs_apply_rngs, f"{cls.__name__} must keep its RNGs")

  def test_every_backend_declares_the_flag(self):
    """Every backend must carry the flag, so the call sites can read it directly.

    `AqtQuantization` and `QwixQuantization` sit outside the `Quantization` hierarchy
    and declare it themselves. A backend that is missing it raises AttributeError at
    the call site rather than silently taking a default. `TransformerEngineQuantization`
    is excluded because its answer depends on the recipe, so it derives the flag per
    instance; `TransformerEngineRecipeRngTest` covers it.
    """
    for cls in (
        quantizations.AqtQuantization,
        quantizations.QwixQuantization,
        quantizations.Fp8Quantization,
        quantizations.NANOOFp8Quantization,
    ):
      self.assertIsInstance(cls.needs_apply_rngs, bool, f"{cls.__name__} must declare needs_apply_rngs")


class ApplyTimeRngTest(unittest.TestCase):
  """A backend that draws at apply time must keep the bridge's forked `Rngs`.

  This is the NVFP4 case: TE draws `sr_rng` for the DGRAD quantizer, and `ToNNX` passes
  the wrapped module only the streams it still holds, so releasing the fork leaves the
  Linen apply with no RNGs at all.
  """

  def test_drawing_backend_still_runs(self):
    out = _make_dense(_DrawingQuant())(jnp.ones((2, 8), jnp.float32))
    self.assertEqual(out.shape, (2, 4))

  def test_opting_out_a_drawing_backend_fails_loudly(self):
    """Pins the coupling that broke NVFP4, so a wrong opt-out cannot pass silently."""
    dense = _make_dense(_MisdeclaredQuant())
    with self.assertRaises(flax_errors.InvalidRngError):
      dense(jnp.ones((2, 8), jnp.float32))


class TransformerEngineRecipeRngTest(unittest.TestCase):
  """Only NVFP4 with stochastic rounding on draws at apply time."""

  # te_nvfp4 and te_nvfp4_no_rht both leave disable_stochastic_rounding at its False
  # default, so both draw; disable_rht does not affect rounding.
  EXPECTED = {
      "te_fp8_delayedscaling": False,
      "te_fp8_currentscaling": False,
      "te_mxfp8": False,
      "te_nvfp4": True,
      "te_nvfp4_no_rht": True,
  }

  def test_flag_follows_the_recipe(self):
    try:
      import transformer_engine  # pylint: disable=import-outside-toplevel,unused-import  # pytype: disable=import-error
    except ImportError:
      self.skipTest("TransformerEngine is not installed")

    for name, expected in self.EXPECTED.items():
      with self.subTest(recipe=name):
        config = types.SimpleNamespace(quantization=name, te_comm_gemm_overlap=None)
        quant = quantizations.TransformerEngineQuantization(config)
        self.assertEqual(quant.needs_apply_rngs, expected)


if __name__ == "__main__":
  unittest.main()
