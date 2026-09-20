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

"""The RWKV-7 WKV7 Pallas kernels vs. the reference `wkv7_naive` scan.

Both kernel algorithms ("recurrent" and "chunked") are held to the same checks.
Most tests run the kernels in Pallas TPU interpret mode, which checks
correctness (and, via `test_lowers_to_mosaic`, that the kernels lower to Mosaic
custom calls) on any backend. The `tpu_only` tests repeat the same checks on
the compiled kernels, which is what validates Mosaic compilation, VMEM limits
and `dimension_semantics`. Neither says anything about TPU performance.
"""

import functools

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels import rwkv7_wkv
from maxtext.kernels.rwkv7_wkv import ALGORITHMS, wkv7_pallas
from maxtext.models.rwkv7 import wkv7_naive

# Relative deviation (max abs error / max abs reference). All paths are float32
# with the same math; only the evaluation order differs. Observed on CPU: the
# recurrent kernel is bit-identical on outputs and ~1e-7 on gradients; the
# chunked kernel (a different association of the same sums) is <= ~1e-6, and
# <= 2.6e-6 in the worst case below (strongest decay, 128-step chunk). 1e-5 is
# accumulation noise with margin, not a loosened bound.
TOLERANCE = 1e-5

# The strongest decay RWKV-7 can produce: w = exp(-exp(-softplus(-x) - 0.5))
# approaches exp(-e^-0.5) as x -> inf.
W_MIN = float(np.exp(-np.exp(-0.5)))

algorithms = pytest.mark.parametrize("algorithm", ALGORITHMS)


def wkv_inputs(batch, seq_len, heads, head_size, seed=0, state_scale=0.3):
  """Inputs distributed like the model's: w and a from the model's gate
  formulas, kk unit-norm per head, and a nonzero incoming state."""
  keys = jax.random.split(jax.random.key(seed), 7)
  shape = (batch, seq_len, heads, head_size)
  r, k, v = (jax.random.normal(keys[i], shape) for i in range(3))
  w = jnp.exp(-jnp.exp(-jax.nn.softplus(-jax.random.normal(keys[3], shape)) - 0.5))
  a = jax.nn.sigmoid(jax.random.normal(keys[4], shape))
  kk = jax.random.normal(keys[5], shape)
  kk = kk / jnp.linalg.norm(kk, axis=-1, keepdims=True)
  state = state_scale * jax.random.normal(keys[6], (batch, heads, head_size, head_size))
  return r, w, k, v, a, kk, state


def max_rel_dev(actual, expected):
  actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
  return float(np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1e-12))


# (batch, seq_len, heads, head_size, chunk_size). Covers: a sequence shorter
# than one chunk, an exact multiple of the chunk, a ragged tail that needs
# identity padding, single-token decode, the real 0.1B head size, and (for the
# chunked form, whose matmuls grow with it) several 64-step chunks.
SHAPES = [
    (2, 5, 3, 16, 16),
    (2, 32, 2, 16, 16),
    (2, 37, 2, 16, 16),
    (3, 1, 2, 16, 16),
    (1, 40, 2, 64, 8),
    (1, 130, 2, 64, 64),
]


# Production-sized cases for the compiled kernels: the 0.1B model's 12 heads of
# 64, a long sequence at the default and a larger chunk, and decode.
TPU_SHAPES = [
    (2, 512, 12, 64, 16),
    (2, 512, 12, 64, 64),
    (4, 1, 12, 64, 16),
]


def check_against_naive(
    args, algorithm, interpret, chunk_size=rwkv7_wkv.DEFAULT_CHUNK_SIZE, cotangent_seed=1, reset=None
):
  """Outputs, final state, and the VJP w.r.t. all seven inputs (incl. the
  incoming state) with random cotangents on both outputs, so the final-state
  cotangent path is exercised too."""
  kernel = functools.partial(wkv7_pallas, chunk_size=chunk_size, interpret=interpret, algorithm=algorithm, reset=reset)
  (out, state), pallas_vjp = jax.vjp(kernel, *args)
  (ref_out, ref_state), naive_vjp = jax.vjp(functools.partial(wkv7_naive, reset=reset), *args)
  assert out.shape == ref_out.shape and state.shape == ref_state.shape
  assert max_rel_dev(out, ref_out) < TOLERANCE
  assert max_rel_dev(state, ref_state) < TOLERANCE
  ct_keys = jax.random.split(jax.random.key(cotangent_seed), 2)
  cotangents = tuple(jax.random.normal(key, x.shape) for key, x in zip(ct_keys, (ref_out, ref_state)))
  names = ("r", "w", "k", "v", "a", "kk", "state")
  for name, got, want in zip(names, pallas_vjp(cotangents), naive_vjp(cotangents)):
    assert np.all(np.isfinite(got)), name
    assert max_rel_dev(got, want) < TOLERANCE, name


@algorithms
@pytest.mark.parametrize("batch,seq_len,heads,head_size,chunk_size", SHAPES)
def test_matches_naive(algorithm, batch, seq_len, heads, head_size, chunk_size):
  check_against_naive(wkv_inputs(batch, seq_len, heads, head_size), algorithm, True, chunk_size)


@algorithms
def test_jit_and_grad_compose(algorithm):
  args = wkv_inputs(1, 20, 2, 16)

  @jax.jit
  def loss(*x):
    out, state = wkv7_pallas(*x, interpret=True, algorithm=algorithm)
    return jnp.sum(out**2) + jnp.sum(state**2)

  def ref_loss(*x):
    out, state = wkv7_naive(*x)
    return jnp.sum(out**2) + jnp.sum(state**2)

  argnums = tuple(range(7))
  for got, want in zip(jax.grad(loss, argnums)(*args), jax.grad(ref_loss, argnums)(*args)):
    assert max_rel_dev(got, want) < TOLERANCE


@algorithms
def test_mixed_precision_inputs_match_naive(algorithm):
  """In a bf16 model config r and v arrive in bf16 while w, k, a and kk are
  promoted to float32. The kernel must return what the reference returns (a
  float32 output, not one rounded back to r's dtype) and gradients in each
  input's own dtype."""
  r, w, k, v, a, kk, state = wkv_inputs(2, 37, 2, 16)
  args = (r.astype(jnp.bfloat16), w, k, v.astype(jnp.bfloat16), a, kk, state)
  (out, final_state), pallas_vjp = jax.vjp(lambda *x: wkv7_pallas(*x, interpret=True, algorithm=algorithm), *args)
  (ref_out, ref_state), naive_vjp = jax.vjp(wkv7_naive, *args)
  assert out.dtype == ref_out.dtype == jnp.float32
  assert max_rel_dev(out, ref_out) < TOLERANCE
  assert max_rel_dev(final_state, ref_state) < TOLERANCE
  cotangents = (jnp.ones_like(out), jnp.ones_like(final_state))
  for x, got, want in zip(args, pallas_vjp(cotangents), naive_vjp(cotangents)):
    assert got.dtype == want.dtype == x.dtype
    if want.dtype == jnp.float32:
      assert max_rel_dev(got, want) < TOLERANCE
    else:
      # A bf16 gradient is rounded to 8 mantissa bits on the way out, so the
      # kernel and the reference can differ by one representable step however
      # close their float32 values are; TOLERANCE would be testing which side
      # of a rounding boundary each landed on. One bf16 ulp -- of the element,
      # or of the array's largest -- is what a bf16-typed output admits.
      got_f32, want_f32 = got.astype(jnp.float32), want.astype(jnp.float32)
      assert np.allclose(got_f32, want_f32, rtol=2**-8, atol=2**-8 * float(np.max(np.abs(want_f32))))


@algorithms
def test_strongest_decay_large_chunk(algorithm):
  """Every step at the model's strongest possible decay, 128-step chunks: the
  chunked form's rescaled factors span the widest exponent range here."""
  r, w, k, v, a, kk, state = wkv_inputs(1, 256, 2, 64, seed=3)
  args = (r, jnp.full_like(w, W_MIN), k, v, a, kk, state)
  check_against_naive(args, algorithm, True, chunk_size=128)


@algorithms
def test_aligned_keys_full_learning_rate(algorithm):
  """The same kk at every step with in-context learning rate ~1 and almost no
  decay: the delta-rule coupling matrix approaches -(strictly lower ones), the
  case that breaks a log-depth Neumann-product inverse."""
  r, w, k, v, a, kk, state = wkv_inputs(1, 128, 2, 64, seed=5)
  kk = jnp.broadcast_to(kk[:, :1], kk.shape)
  args = (r, jnp.full_like(w, 0.999), k, v, jnp.full_like(a, 0.999), kk, state)
  check_against_naive(args, algorithm, True, chunk_size=64)


@pytest.mark.parametrize("length", [1, 8, 64])
def test_unit_lower_inverse(length):
  """Forward-substitution (I - M)^-1 vs. float64, on random M and the two
  structured extremes (inverse of size ~1, and of size ~binomial(L, L/2))."""
  rng = np.random.default_rng(length)
  strict = np.tril(np.ones((length, length)), -1)
  for m in (-strict, 0.9 * strict, strict * rng.normal(size=(length, length)) * 0.3):
    exact = np.linalg.inv(np.eye(length) - m)
    got = rwkv7_wkv._unit_lower_inverse(jnp.asarray(m, jnp.float32))  # pylint: disable=protected-access
    assert max_rel_dev(got, exact) < TOLERANCE


def test_rejects_bad_arguments():
  args = wkv_inputs(1, 20, 1, 16)
  with pytest.raises(ValueError, match="multiple of 8"):
    wkv7_pallas(*args, chunk_size=12, interpret=True)
  with pytest.raises(ValueError, match="algorithm"):
    wkv7_pallas(*args, interpret=True, algorithm="wy")


@algorithms
def test_discharge_interpret_differentiable_under_remat(algorithm):
  """The model runs the kernel under `jax.checkpoint` (remat is on by default). The TPU
  interpreter's IO callbacks can't be differentiated there; the discharge interpreter,
  which the model uses off-TPU, can, and gives the same gradients as naive."""
  args = wkv_inputs(1, 20, 2, 16)
  argnums = tuple(range(7))

  def loss(interpret):
    return jax.checkpoint(lambda *x: jnp.sum(wkv7_pallas(*x, interpret=interpret, algorithm=algorithm)[0] ** 2))

  with pytest.raises(NotImplementedError, match="Effects not supported"):
    jax.grad(loss(True), argnums)(*args)
  want = jax.grad(lambda *x: jnp.sum(wkv7_naive(*x)[0] ** 2), argnums)(*args)
  for got, ref in zip(jax.grad(loss("discharge"), argnums)(*args), want):
    assert max_rel_dev(got, ref) < TOLERANCE


# Packed-sequence resets, per row: inside a chunk, exactly on chunk boundaries
# (16, 32 with 16-step chunks), at t=0 (drops a nonzero incoming state), and
# back to back.
RESETS = [[5, 16, 29, 32], [0, 7, 8, 21]]


def reset_mask(batch_rows, seq_len):
  mask = np.zeros((len(batch_rows), seq_len), bool)
  for b, row in enumerate(batch_rows):
    mask[b, row] = True
  return jnp.asarray(mask)


def test_naive_resets_equal_independent_segments():
  """The reset semantics: each segment runs as if alone, from a zero state (the first
  segment of a row continues from the incoming state unless t=0 resets it)."""
  r, w, k, v, a, kk, state = wkv_inputs(2, 40, 2, 16)
  out, final = wkv7_naive(r, w, k, v, a, kk, state, reset=reset_mask(RESETS, 40))
  for b, row in enumerate(RESETS):
    bounds = sorted(set([0] + row + [40]))
    for start, end in zip(bounds[:-1], bounds[1:]):
      seg = [x[b : b + 1, start:end] for x in (r, w, k, v, a, kk)]
      init = state[b : b + 1] if start == 0 and 0 not in row else jnp.zeros_like(state[b : b + 1])
      seg_out, seg_final = wkv7_naive(*seg, init)
      assert max_rel_dev(out[b : b + 1, start:end], seg_out) < TOLERANCE, (b, start)
    assert max_rel_dev(final[b : b + 1], seg_final) < TOLERANCE, b


@algorithms
def test_resets_match_naive(algorithm):
  """Both kernels with packed-sequence resets vs. the reference: outputs, final state and
  every gradient, with resets inside chunks and on chunk boundaries."""
  args = wkv_inputs(2, 40, 2, 16, seed=13)
  check_against_naive(args, algorithm, True, 16, reset=reset_mask(RESETS, 40))


@algorithms
def test_resets_at_strongest_decay(algorithm):
  """Resets in 128-step chunks at the strongest decay, where the chunked form's rescaled
  factors span the widest range."""
  r, w, k, v, a, kk, state = wkv_inputs(1, 256, 2, 64, seed=17)
  args = (r, jnp.full_like(w, W_MIN), k, v, a, kk, state)
  check_against_naive(args, algorithm, True, chunk_size=128, reset=reset_mask([[3, 64, 127, 128, 200]], 256))


def strongest_decay_aligned_inputs(seq_len, head_size):
  """Strongest decay at every step, one constant unit kk, a = 0.5, zero state.
  Without the size cap, a 512-step chunk of this makes the chunked kernel's
  rescaling factors overflow float32 and its output non-finite."""
  shape = (1, seq_len, 1, head_size)
  r, k, v = (jax.random.normal(key, shape) for key in jax.random.split(jax.random.key(7), 3))
  kk = jnp.zeros(shape).at[..., 0].set(1.0)
  state = jnp.zeros((1, 1, head_size, head_size))
  return r, jnp.full(shape, W_MIN), k, v, jnp.full(shape, 0.5), kk, state


def test_chunked_rejects_oversized_chunk():
  args = strongest_decay_aligned_inputs(512, 4)
  assert rwkv7_wkv.MAX_CHUNKED_CHUNK_SIZE == 128
  with pytest.raises(ValueError, match="at most 128"):
    wkv7_pallas(*args, chunk_size=512, interpret=True, algorithm="chunked")
  # The recurrent algorithm has no rescaling and so no such limit.
  check_against_naive(args, "recurrent", True, chunk_size=512)
  # The largest allowed chunk is finite and correct on the same inputs.
  check_against_naive(args, "chunked", True, chunk_size=128)


@algorithms
def test_two_core_interpret_no_races(algorithm):
  """Splits the "parallel" (batch, head) grid axes across two simulated
  TensorCores with the race detector on: the chunk axis must still see its
  state in order, and no two programs may touch the same output block."""
  # pylint: disable=import-outside-toplevel
  from jax._src.pallas.mosaic.interpret import interpret_pallas_call  # private: the only handle on race results

  params = pltpu.InterpretParams(detect_races=True, num_cores_or_threads=2)
  args = wkv_inputs(2, 37, 2, 16)
  kernel = functools.partial(wkv7_pallas, interpret=params, algorithm=algorithm)
  (out, state), vjp = jax.vjp(kernel, *args)
  assert not interpret_pallas_call.races.races_found
  grads = vjp((jnp.ones_like(out), jnp.ones_like(state)))
  assert not interpret_pallas_call.races.races_found
  ref_out, ref_state = wkv7_naive(*args)
  ref_grads = jax.vjp(wkv7_naive, *args)[1]((jnp.ones_like(out), jnp.ones_like(state)))
  assert max_rel_dev(out, ref_out) < TOLERANCE
  assert max_rel_dev(state, ref_state) < TOLERANCE
  for got, want in zip(grads, ref_grads):
    assert max_rel_dev(got, want) < TOLERANCE


@algorithms
@pytest.mark.parametrize("chunk_size", [16, 64])
def test_lowers_to_mosaic(algorithm, chunk_size):
  """Forward and backward lower to Mosaic TPU custom calls at the 0.1B model's
  head size. This runs the Pallas -> Mosaic lowering only; Mosaic's own
  compilation needs libtpu and is not exercised here."""
  sds = jax.ShapeDtypeStruct
  seq = sds((2, 128, 12, 64), jnp.float32)
  args = [seq] * 6 + [sds((2, 12, 64, 64), jnp.float32)]
  kernel = functools.partial(wkv7_pallas, interpret=False, algorithm=algorithm, chunk_size=chunk_size)

  def loss(*x):
    out, state = kernel(*x)
    return jnp.sum(out) + jnp.sum(state)

  fwd = jax.jit(kernel).trace(*args).lower(lowering_platforms=("tpu",))
  assert fwd.as_text().count("tpu_custom_call") == 1
  bwd = jax.jit(jax.grad(loss, tuple(range(7)))).trace(*args).lower(lowering_platforms=("tpu",))
  assert bwd.as_text().count("tpu_custom_call") == 2


def _tpu_compile_target():
  """A TPU v5e chip to compile for via libtpu (no TPU hardware needed), or skip."""
  from jax.experimental import topologies  # pylint: disable=import-outside-toplevel

  try:
    topology = topologies.get_topology_desc(topology_name="v5e:2x2", platform="tpu")
  except Exception as e:  # pylint: disable=broad-except
    pytest.skip(f"TPU compiler (libtpu) unavailable: {type(e).__name__}")
  return topology.devices[0]


@algorithms
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("batch,seq_len", [(2, 512), (8, 1), (1, 37)])
def test_compiles_for_tpu(algorithm, dtype, batch, seq_len):
  """Forward+backward go through Mosaic's real compiler for a TPU chip at the 0.1B model's
  head layout: training length, decode, and a ragged length, with r and v in the compute
  dtype as in the model. Catches what interpret mode and lowering can't (layout legality,
  alignment, VMEM). The recurrent kernel failed this for bf16 before its inputs were upcast."""
  device = _tpu_compile_target()
  sharding = jax.sharding.NamedSharding(jax.sharding.Mesh([device], ("x",)), jax.sharding.PartitionSpec())
  shape = (batch, seq_len, 12, 64)
  seq, f32_seq = (jax.ShapeDtypeStruct(shape, t, sharding=sharding) for t in (dtype, jnp.float32))
  state = jax.ShapeDtypeStruct((batch, 12, 64, 64), jnp.float32, sharding=sharding)
  kernel = functools.partial(wkv7_pallas, interpret=False, algorithm=algorithm, chunk_size=64)

  def loss(*x):
    out, final_state = kernel(*x)
    return jnp.sum(out**2) + jnp.sum(final_state**2)

  args = (seq, f32_seq, f32_seq, seq, f32_seq, f32_seq, state)
  jax.jit(jax.grad(loss, tuple(range(7)))).trace(*args).lower(lowering_platforms=("tpu",)).compile()


def _require_tpu():
  if jax.default_backend() != "tpu":
    pytest.skip("Requires TPU backend.")


@pytest.mark.tpu_only
@algorithms
@pytest.mark.parametrize("batch,seq_len,heads,head_size,chunk_size", TPU_SHAPES)
def test_compiled_matches_naive(algorithm, batch, seq_len, heads, head_size, chunk_size):
  _require_tpu()
  check_against_naive(wkv_inputs(batch, seq_len, heads, head_size), algorithm, False, chunk_size)


@pytest.mark.tpu_only
@algorithms
def test_compiled_strongest_decay_large_chunk(algorithm):
  _require_tpu()
  r, w, k, v, a, kk, state = wkv_inputs(2, 512, 12, 64, seed=3)
  check_against_naive((r, jnp.full_like(w, W_MIN), k, v, a, kk, state), algorithm, False, chunk_size=128)


def test_wkv7_naive_contraction_precision():
  """Verifies wkv7_naive einsums explicitly request Precision.HIGHEST."""
  r, w, k, v, a, kk, state = wkv_inputs(2, 4, 2, 8)
  jaxpr = jax.make_jaxpr(wkv7_naive)(r, w, k, v, a, kk, state)
  scan_eqn = next(e for e in jaxpr.eqns if e.primitive.name == "scan")
  sub_jaxpr = scan_eqn.params["jaxpr"].jaxpr
  dot_precisions = [e.params.get("precision") for e in sub_jaxpr.eqns if e.primitive.name == "dot_general"]
  assert len(dot_precisions) == 4, f"Expected 4 dot_general in step, found {len(dot_precisions)}"
  assert all(p == (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST) for p in dot_precisions)
