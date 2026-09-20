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

"""The RWKV-7 WKV7 Pallas GPU kernel vs. the reference `wkv7_naive` scan.

The checks are the ones `rwkv7_wkv_kernel_test.py` applies to the TPU kernels,
and reuse that module's input generator and tolerance. They run in Pallas
interpret mode on any backend; the `gpu_only` tests repeat them on the
compiled Triton kernel, which is what validates the real lowering. Neither
says anything about performance.

The kernel's backward inverts steps instead of recomputing them, so
`test_strongest_decay` matters most here: it pins the error at
w = exp(-e^-0.5), where 1/w is largest and the inversion is worst
conditioned.
"""

import functools
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels import rwkv7_wkv_gpu
from maxtext.kernels.rwkv7_wkv_gpu import MAX_CHUNK_SIZE, wkv7_pallas_gpu
from maxtext.models.rwkv7 import resolve_wkv7_impl, wkv7_naive
from tests.unit.rwkv7_wkv_kernel_test import TOLERANCE, W_MIN, max_rel_dev, wkv_inputs

# (batch, seq_len, heads, head_size, chunk_size): a sequence shorter than one
# chunk, an exact multiple, a ragged tail that needs identity padding, single
# token decode, and the 0.1B model's head size.
SHAPES = [(2, 8, 2, 16, 8), (2, 64, 3, 16, 8), (1, 37, 2, 16, 8), (3, 1, 2, 16, 8), (2, 128, 2, 64, 4)]


def kernel(*args, **kwargs):
  return wkv7_pallas_gpu(*args, interpret=True, **kwargs)


def check_matches_naive(impl, batch, seq_len, heads, head_size, chunk_size, **input_kwargs):
  """Outputs, final state and all seven gradients against `wkv7_naive`."""
  args = wkv_inputs(batch, seq_len, heads, head_size, **input_kwargs)
  run = functools.partial(impl, chunk_size=chunk_size)
  (out, state), vjp = jax.vjp(run, *args)
  (ref_out, ref_state), ref_vjp = jax.vjp(wkv7_naive, *args)
  assert out.shape == ref_out.shape and state.shape == ref_state.shape
  assert max_rel_dev(out, ref_out) < TOLERANCE
  assert max_rel_dev(state, ref_state) < TOLERANCE
  cotangents = (jax.random.normal(jax.random.key(7), out.shape), jax.random.normal(jax.random.key(8), state.shape))
  for name, got, want in zip("r w k v a kk state".split(), vjp(cotangents), ref_vjp(cotangents)):
    assert max_rel_dev(got, want) < TOLERANCE, name


@pytest.mark.cpu_only
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_naive(shape):
  check_matches_naive(kernel, *shape)


@pytest.mark.gpu_only
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_naive_on_gpu(shape):
  check_matches_naive(wkv7_pallas_gpu, *shape)


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 8])
def test_chunk_size_does_not_change_the_answer(chunk_size):
  check_matches_naive(kernel, 2, 128, 2, 16, chunk_size)


def test_chunk_size_above_the_maximum_is_rejected():
  """The bound exists because the inversion backward amplifies rounding by
  (1/w)^chunk_size; see `test_inversion_error_grows_with_the_chunk_size`."""
  args = wkv_inputs(2, 32, 2, 16)
  with pytest.raises(ValueError, match="chunk_size"):
    kernel(*args, chunk_size=MAX_CHUNK_SIZE + 1)


def test_inversion_error_grows_with_the_chunk_size(monkeypatch):
  """Why `MAX_CHUNK_SIZE` is 8 and not BlinkDL's 16: at the strongest decay the
  backward's division by w multiplies float32 rounding by up to 1.834 a step,
  so the gradients lose about a decimal digit for every four steps added. This
  is the measurement the bound comes from, so it lifts the bound to take it."""
  monkeypatch.setattr(rwkv7_wkv_gpu, "MAX_CHUNK_SIZE", 32)
  args = list(wkv_inputs(2, 128, 2, 16))
  args[1] = jnp.full_like(args[1], W_MIN)
  deviations = {}
  for chunk_size in (4, 8, 16, 32):
    run = functools.partial(wkv7_pallas_gpu, interpret=True, chunk_size=chunk_size)
    (out, state), vjp = jax.vjp(run, *args)
    (ref_out, ref_state), ref_vjp = jax.vjp(wkv7_naive, *args)
    cotangents = (jnp.ones_like(out), jnp.ones_like(state))
    deviations[chunk_size] = max(max_rel_dev(g, w) for g, w in zip(vjp(cotangents), ref_vjp(cotangents)))
    # The forward never inverts anything, so it is unaffected.
    assert max_rel_dev(out, ref_out) < TOLERANCE
    assert max_rel_dev(state, ref_state) < TOLERANCE
  assert deviations[MAX_CHUNK_SIZE] < TOLERANCE
  assert deviations[16] > 1e-4 and deviations[32] > 1.0
  assert deviations[4] < deviations[8] < deviations[16] < deviations[32]


def test_strongest_decay():
  """Every step at the model's strongest decay, where the backward's division
  by w is worst conditioned (1/w < 1.84, the bound that makes it usable)."""
  args = list(wkv_inputs(2, 256, 2, 16))
  args[1] = jnp.full_like(args[1], W_MIN)
  (out, state), vjp = jax.vjp(lambda *x: kernel(*x, chunk_size=MAX_CHUNK_SIZE), *args)
  (ref_out, ref_state), ref_vjp = jax.vjp(wkv7_naive, *args)
  assert max_rel_dev(out, ref_out) < TOLERANCE
  assert max_rel_dev(state, ref_state) < TOLERANCE
  cotangents = (jnp.ones_like(out), jnp.ones_like(state))
  for name, got, want in zip("r w k v a kk state".split(), vjp(cotangents), ref_vjp(cotangents)):
    assert max_rel_dev(got, want) < TOLERANCE, name


def test_mixed_precision_inputs_match_naive():
  """In a bf16 config r and v arrive in bf16 while w, k, a and kk are promoted
  to float32; the output is float32 and each gradient keeps its input's dtype."""
  r, w, k, v, a, kk, state = wkv_inputs(2, 37, 2, 16)
  args = (r.astype(jnp.bfloat16), w, k, v.astype(jnp.bfloat16), a, kk, state)
  (out, final_state), vjp = jax.vjp(kernel, *args)
  (ref_out, ref_state), ref_vjp = jax.vjp(wkv7_naive, *args)
  assert out.dtype == ref_out.dtype == jnp.float32
  assert max_rel_dev(out, ref_out) < TOLERANCE
  assert max_rel_dev(final_state, ref_state) < TOLERANCE
  cotangents = (jnp.ones_like(out), jnp.ones_like(final_state))
  for x, got, want in zip(args, vjp(cotangents), ref_vjp(cotangents)):
    assert got.dtype == want.dtype == x.dtype


def test_resets_are_rejected():
  """Packed-sequence resets can't be inverted, so they must fail loudly."""
  args = wkv_inputs(2, 16, 2, 16)
  reset = jnp.zeros((2, 16), jnp.float32).at[:, 8].set(1.0)
  with pytest.raises(NotImplementedError, match="resets"):
    kernel(*args, reset=reset)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "platform, segment_resets, expected",
    [
        ("tpu", False, "pallas_chunked"),
        ("tpu", True, "pallas_chunked"),
        ("gpu", False, "pallas_gpu"),
        ("gpu", True, "naive"),  # this kernel has no resets, so autoselect must not pick it
        ("cpu", False, "naive"),
    ],
)
def test_autoselected_picks_the_targets_kernel(platform, segment_resets, expected):
  """`rwkv7_wkv_impl="autoselected"` resolves from the mesh's platform, the
  same rule `attention_kernel="autoselected"` uses."""
  config = types.SimpleNamespace(rwkv7_wkv_impl="autoselected", rwkv7_segment_resets=segment_resets)
  mesh = types.SimpleNamespace(devices=np.array([types.SimpleNamespace(platform=platform)]))
  assert resolve_wkv7_impl(config, mesh) == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", ["naive", "pallas", "pallas_chunked", "pallas_gpu"])
def test_an_explicit_impl_is_never_overridden(name):
  config = types.SimpleNamespace(rwkv7_wkv_impl=name, rwkv7_segment_resets=False)
  mesh = types.SimpleNamespace(devices=np.array([types.SimpleNamespace(platform="tpu")]))
  assert resolve_wkv7_impl(config, mesh) == name
