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

  apply_rngs = quantizations.ApplyRngs.NONE
  quant_mode = "train"  # read by quantizations.in_serve_mode()

  def dot_general_cls(self, mesh_axes=()):
    del mesh_axes
    return _StubDotGeneral


class _DefaultQuant(quantizations.Quantization):
  """A backend that leaves `apply_rngs` at its safe default of PRIVATE."""

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

  apply_rngs = quantizations.ApplyRngs.NONE


class _SharedQuant(_DrawingQuant):
  """A drawing backend whose values do not depend on having a stream of its own."""

  apply_rngs = quantizations.ApplyRngs.SHARED


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
        "A backend with apply_rngs=NONE must not retain the bridge's forked "
        "Rngs: its counters would be incremented on device every step, once per "
        "unrolled layer.",
    )

  def test_default_backend_keeps_rng_state(self):
    """Checks that the opt-out is deliberate and the default stays safe."""
    paths = _rng_state_paths(_make_dense(_DefaultQuant()))
    self.assertNotEqual(paths, [], "apply_rngs defaults to PRIVATE, so the Rngs must be kept.")

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
  """`apply_rngs` must stay safe-by-default and correct per backend."""

  def test_base_class_defaults_to_a_private_stream(self):
    self.assertIs(quantizations.Quantization.apply_rngs, quantizations.ApplyRngs.PRIVATE)

  def test_backends_that_may_draw_at_apply_time_keep_their_own_rngs(self):
    """AQT's config enables jax.uniform RNG, so it must never be narrowed silently."""
    for cls in (quantizations.AqtQuantization, quantizations.QwixQuantization):
      self.assertIs(cls.apply_rngs, quantizations.ApplyRngs.PRIVATE, f"{cls.__name__} must keep its own RNGs")

  def test_fp8_backends_opt_out(self):
    """Flax's fp8 ops scale from amax history, so they never draw at apply time."""
    for cls in (quantizations.Fp8Quantization, quantizations.NANOOFp8Quantization):
      self.assertIs(cls.apply_rngs, quantizations.ApplyRngs.NONE, f"{cls.__name__} must release the bridge's Rngs")

  def test_every_backend_declares_the_flag(self):
    """Every backend must carry the flag, so the call sites can read it directly.

    `AqtQuantization` and `QwixQuantization` sit outside the `Quantization` hierarchy
    and declare it themselves. A backend that is missing it raises AttributeError at
    the call site rather than silently taking a default. `TransformerEngineQuantization`
    is excluded because its answer depends on the recipe, so it derives it per instance;
    `TransformerEngineRecipeRngTest` covers it.
    """
    for cls in (
        quantizations.AqtQuantization,
        quantizations.QwixQuantization,
        quantizations.Fp8Quantization,
        quantizations.NANOOFp8Quantization,
    ):
      self.assertIsInstance(cls.apply_rngs, quantizations.ApplyRngs, f"{cls.__name__} must declare apply_rngs")


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


class Fp8BackendRngStateTest(unittest.TestCase):
  """The fp8 backends run on CPU, so they can be exercised for real here.

  Asserting the flag only records what we believe about Flax; applying the layer is what
  would catch a future Flax release that starts drawing, since the released bridge would
  raise `InvalidRngError` rather than quietly lose its RNGs.
  """

  BACKENDS = (quantizations.Fp8Quantization, quantizations.NANOOFp8Quantization)

  def test_backends_hold_no_rng_state(self):
    for cls in self.BACKENDS:
      with self.subTest(backend=cls.__name__):
        self.assertEqual(_rng_state_paths(_make_dense(cls())), [])

  def test_backends_still_apply(self):
    x = jnp.ones((2, 8), jnp.float32)
    for cls in self.BACKENDS:
      with self.subTest(backend=cls.__name__):
        out = _make_dense(cls())(x)
        self.assertEqual(out.shape, (2, 4))
        self.assertTrue(jnp.all(jnp.isfinite(out)))

  def test_rng_state_does_not_grow_with_layer_count(self):
    """The unrolled decoder is where a per-wrapper leak turns into a per-layer cost."""
    counts = [
        sum(len(_rng_state_paths(_make_dense(quantizations.Fp8Quantization(), seed=i))) for i in range(n))
        for n in (1, 2, 8)
    ]
    self.assertEqual(counts, [0, 0, 0], f"fp8 RNG state must not scale with layer count, got {counts}")


class SharedRngStateTest(unittest.TestCase):
  """A backend that shares one stream must not add state per unrolled layer.

  This is what NVFP4 needs: it draws every step, so the fork cannot simply be dropped,
  but TransformerEngine folds a per-quantizer hash into whatever it draws, so one stream
  serves every wrapper. NNX stores a shared `Rngs` once however many modules hold it.
  """

  def _stack(self, quant, num_layers):
    """Builds layers the way an unrolled decoder does, from one `Rngs`."""
    rngs = nnx.Rngs(params=0, dropout=1, aqt=2)
    return [
        linears.DenseGeneral(in_features_shape=8, out_features_shape=4, quant=quant, rngs=rngs) for _ in range(num_layers)
    ]

  def _rng_leaves(self, layers) -> int:
    return len(_rng_state_paths(nnx.List(layers)))

  def test_state_does_not_grow_with_layer_count(self):
    counts = [self._rng_leaves(self._stack(_SharedQuant(), n)) for n in (1, 2, 8)]
    self.assertEqual(
        len(set(counts)),
        1,
        f"a shared stream must cost the same at any depth, got {counts} for 1/2/8 layers",
    )

  def test_a_private_backend_still_grows(self):
    """The contrast, so the test above cannot pass for the wrong reason."""
    counts = [self._rng_leaves(self._stack(_DrawingQuant(), n)) for n in (1, 2, 8)]
    self.assertEqual(len(set(counts)), 3, f"a private fork per wrapper should scale, got {counts}")

  def test_sharing_keeps_the_layer_callable(self):
    out = self._stack(_SharedQuant(), 1)[0](jnp.ones((2, 8), jnp.float32))
    self.assertEqual(out.shape, (2, 4))
    self.assertTrue(jnp.all(jnp.isfinite(out)))


class TransformerEngineRecipeRngTest(unittest.TestCase):
  """Only NVFP4 with stochastic rounding on draws at apply time, and it can share."""

  # te_nvfp4 and te_nvfp4_no_rht both leave disable_stochastic_rounding at its False
  # default, so both draw; disable_rht does not affect rounding.
  EXPECTED = {
      "te_fp8_delayedscaling": quantizations.ApplyRngs.NONE,
      "te_fp8_currentscaling": quantizations.ApplyRngs.NONE,
      "te_mxfp8": quantizations.ApplyRngs.NONE,
      "te_nvfp4": quantizations.ApplyRngs.SHARED,
      "te_nvfp4_no_rht": quantizations.ApplyRngs.SHARED,
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
        self.assertIs(quant.apply_rngs, expected)


if __name__ == "__main__":
  unittest.main()
