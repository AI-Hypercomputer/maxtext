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

"""TPU tests for the 3D <-> 2D VMEM relayout helpers (bf16 and fp8)."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from maxtext.kernels.megablox import relayout
import pytest

pytestmark = pytest.mark.tpu_only  # Pallas TPU kernels.

D0 = 56
D1 = 128


def _sub(dtype):
  return pltpu.get_tpu_info().get_sublane_tiling(jnp.dtype(dtype))


def _rand(key, shape, dtype):
  """Random finite values of `dtype`."""
  x = jax.random.normal(key, shape, jnp.float32) * 4
  return x.astype(dtype)


def _bits(x):
  x = np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint8 if x.dtype.itemsize == 1 else jnp.uint16))
  return x


def _flatten_4d(x4d, gm, tile_d0, tm_load=None):
  num_g, sub, d0, d1 = x4d.shape
  m = num_g * sub
  tm = gm * sub
  tm_load = tm if tm_load is None else tm_load
  num_k = pl.cdiv(d0, tile_d0)
  tile_k = tile_d0 * d1

  def kernel(x_ref, o_ref):
    if tm_load == tm:
      o_ref[...] = relayout.load_3d_as_2d(x_ref, tm, tile_d0)
    else:
      o_ref[...] = jnp.zeros_like(o_ref)
      o_ref[pl.ds(0, tm_load), :] = relayout.load_3d_as_2d(x_ref, tm_load, tile_d0)

  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, d0 * d1), x4d.dtype),
      grid=(num_g // gm, num_k),
      in_specs=[pl.BlockSpec((gm, sub, tile_d0, d1), lambda i, k: (i, 0, k, 0))],
      out_specs=pl.BlockSpec((tm, tile_k), lambda i, k: (i, k)),
  )(x4d)


def _unflatten_to_4d(x2d, gm, tile_d0, sub):
  m, k = x2d.shape
  d0 = k // D1
  num_g = m // sub
  tm = gm * sub
  num_k = pl.cdiv(d0, tile_d0)
  tile_k = tile_d0 * D1

  def kernel(x_ref, o_ref):
    relayout.store_2d_as_3d(o_ref, x_ref[...], tm, tile_d0)

  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((num_g, sub, d0, D1), x2d.dtype),
      grid=(num_g // gm, num_k),
      in_specs=[pl.BlockSpec((tm, tile_k), lambda i, k: (i, k))],
      out_specs=pl.BlockSpec((gm, sub, tile_d0, D1), lambda i, k: (i, 0, k, 0)),
  )(x2d)


def _flatten_4d_window(x4d, gm, tile_d0):
  """Full-D0 blocks; each grid step (i, w) relayouts D0 window w (switch over static starts)."""
  num_g, sub, d0, d1 = x4d.shape
  tm = gm * sub

  def kernel(x_ref, o_ref):
    o_ref[...] = relayout.load_3d_window_as_2d(x_ref, tm, tile_d0, pl.program_id(1))

  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((num_g * sub, d0 * d1), x4d.dtype),
      grid=(num_g // gm, d0 // tile_d0),
      in_specs=[pl.BlockSpec((gm, sub, d0, d1), lambda i, w: (i, 0, 0, 0))],
      out_specs=pl.BlockSpec((tm, tile_d0 * d1), lambda i, w: (i, w)),
  )(x4d)


def _unflatten_to_4d_window(x2d, gm, tile_d0, sub):
  """Full-D0 out block resident across the D0 windows w; each step stores window w."""
  m, k = x2d.shape
  d0 = k // D1
  tm = gm * sub

  def kernel(x_ref, o_ref):
    relayout.store_2d_as_3d_window(o_ref, x_ref[...], tm, tile_d0, pl.program_id(1))

  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m // sub, sub, d0, D1), x2d.dtype),
      grid=(m // tm, d0 // tile_d0),
      in_specs=[pl.BlockSpec((tm, tile_d0 * D1), lambda i, w: (i, w))],
      out_specs=pl.BlockSpec((gm, sub, d0, D1), lambda i, w: (i, 0, 0, 0)),
  )(x2d)


DTYPES = ["bfloat16", "float8_e4m3fn"]


class RelayoutTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.devices()[0].platform, "tpu")

  @parameterized.product(dtype=DTYPES, tile_d0=[8, 16, 32, 56], gm=[4, 8])
  def test_load(self, dtype, tile_d0, gm):
    dtype = jnp.dtype(dtype)
    sub = _sub(dtype)
    m = 1024
    x = _rand(jax.random.key(0), (m, D0, D1), dtype)
    x4d = x.reshape(m // sub, sub, D0, D1)
    got = jax.jit(lambda a: _flatten_4d(a, gm, tile_d0))(x4d)
    np.testing.assert_array_equal(_bits(got), _bits(x.reshape(m, D0 * D1)))

  @parameterized.product(dtype=DTYPES, tile_d0=[8, 56], tm_load=[64, 96])
  def test_load_partial_rows(self, dtype, tile_d0, tm_load):
    dtype = jnp.dtype(dtype)
    sub = _sub(dtype)
    m, gm = 1024, 128 // sub
    tm = gm * sub
    x = _rand(jax.random.key(2), (m, D0, D1), dtype)
    x4d = x.reshape(m // sub, sub, D0, D1)
    got = jax.jit(lambda a: _flatten_4d(a, gm, tile_d0, tm_load))(x4d)
    want = np.array(x.reshape(m, D0 * D1).astype(jnp.float32)).reshape(m // tm, tm, D0 * D1)
    want[:, tm_load:, :] = 0.0
    np.testing.assert_array_equal(np.asarray(got.astype(jnp.float32)), want.reshape(m, D0 * D1))

  @parameterized.product(dtype=DTYPES, tile_d0=[8, 16, 32, 56], gm=[4, 8])
  def test_store(self, dtype, tile_d0, gm):
    dtype = jnp.dtype(dtype)
    sub = _sub(dtype)
    m = 1024
    y = _rand(jax.random.key(1), (m, D0 * D1), dtype)
    got = jax.jit(lambda a: _unflatten_to_4d(a, gm, tile_d0, sub))(y)
    np.testing.assert_array_equal(_bits(got), _bits(y.reshape(m // sub, sub, D0, D1)))

  def test_store_bf16_with_fp8_sublanes(self):
    """3D bf16 out blocks tiled by an fp8 lhs' 32-row sublane (wi dlhs)."""
    m, gm, sub = 1024, 4, 32
    y = _rand(jax.random.key(3), (m, D0 * D1), jnp.bfloat16)
    for tile_d0 in (8, 56):
      got = jax.jit(lambda a, t=tile_d0: _unflatten_to_4d(a, gm, t, sub))(y)
      np.testing.assert_array_equal(_bits(got), _bits(y.reshape(m // sub, sub, D0, D1)))

  @parameterized.product(dtype=DTYPES, tile_d0=[14, 28], gm=[4])
  def test_load_window(self, dtype, tile_d0, gm):
    """D0 windows that are not multiples of 8 (r19 tiles 1792 / 3584), incl. fp8 sub-word starts."""
    dtype = jnp.dtype(dtype)
    sub = _sub(dtype)
    m = 1024
    x = _rand(jax.random.key(4), (m, D0, D1), dtype)
    x4d = x.reshape(m // sub, sub, D0, D1)
    got = jax.jit(lambda a: _flatten_4d_window(a, gm, tile_d0))(x4d)
    np.testing.assert_array_equal(_bits(got), _bits(x.reshape(m, D0 * D1)))

  @parameterized.product(dtype=DTYPES, tile_d0=[28])
  def test_store_window(self, dtype, tile_d0):
    dtype = jnp.dtype(dtype)
    sub = _sub(dtype)
    m = 1024
    y = _rand(jax.random.key(5), (m, D0 * D1), dtype)
    got = jax.jit(lambda a: _unflatten_to_4d_window(a, 4, tile_d0, sub))(y)
    np.testing.assert_array_equal(_bits(got), _bits(y.reshape(m // sub, sub, D0, D1)))

  def test_store_window_bf16_14(self):
    m, sub = 1024, _sub(jnp.bfloat16)
    y = _rand(jax.random.key(6), (m, D0 * D1), jnp.bfloat16)
    got = jax.jit(lambda a: _unflatten_to_4d_window(a, 4, 14, sub))(y)
    np.testing.assert_array_equal(_bits(got), _bits(y.reshape(m // sub, sub, D0, D1)))


if __name__ == "__main__":
  absltest.main()
