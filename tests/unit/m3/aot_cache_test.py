# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the per-shape AOT executable cache (`maxtext.m3.infra.aot_cache`)."""

import glob
import os

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.infra import aot_cache


def _affine(x, scale, bias=0.0):
  return x * scale + bias


_cached_affine = aot_cache.cached_jit(_affine, static_argnames=("scale",))


class AotCacheTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.cache_dir = self.create_tempdir().full_path
    # The pipelines pass their mesh: it pins the device order when executables are deserialized
    # (without it they would bind to all local devices, e.g. the 4 forced host devices in CI).
    self.mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))

  def tearDown(self):
    aot_cache.install("", {}, None)  # back to the plain-jit default for other tests
    super().tearDown()

  def _files(self):
    return glob.glob(os.path.join(self.cache_dir, "*.aotx"))

  def test_behaves_like_jit_when_not_installed(self):
    x = jnp.arange(4.0)
    np.testing.assert_allclose(_cached_affine(x, scale=2.0, bias=1.0), _affine(x, 2.0, 1.0))
    self.assertEqual(aot_cache.save_pending(), 0)

  def test_save_and_reload_executable(self):
    x = jnp.arange(4.0)
    aot_cache.install(self.cache_dir, {"model": "tiny"}, mesh=self.mesh)
    np.testing.assert_allclose(_cached_affine(x, scale=3.0), _affine(x, 3.0))
    self.assertEqual(aot_cache.save_pending(), 1)
    self.assertLen(self._files(), 1)
    self.assertEqual(aot_cache.save_pending(), 0)  # nothing new recorded

    # A new install with the same metadata deserializes the executable instead of recompiling.
    aot_cache.install(self.cache_dir, {"model": "tiny"}, mesh=self.mesh)
    aot_cache.wait_for_loads()
    np.testing.assert_allclose(_cached_affine(x, scale=3.0), _affine(x, 3.0))
    self.assertEqual(aot_cache.save_pending(), 0)  # already on disk

    # Different metadata -> different fingerprint -> a separate file.
    aot_cache.install(self.cache_dir, {"model": "other"}, mesh=self.mesh)
    aot_cache.wait_for_loads()
    _cached_affine(x, scale=3.0)
    self.assertEqual(aot_cache.save_pending(), 1)
    self.assertLen(self._files(), 2)

  def test_warmup_mode_returns_zeros(self):
    x = jnp.arange(4.0) + 1.0
    self.assertFalse(aot_cache.in_warmup())
    with aot_cache.warmup_mode():  # no-op while the cache is disabled
      self.assertFalse(aot_cache.in_warmup())
      np.testing.assert_allclose(_cached_affine(x, scale=2.0), _affine(x, 2.0))

    aot_cache.install(self.cache_dir, {"model": "tiny"}, mesh=self.mesh)
    with aot_cache.warmup_mode():
      self.assertTrue(aot_cache.in_warmup())
      out = _cached_affine(x, scale=2.0)
      self.assertEqual(out.shape, x.shape)
      np.testing.assert_array_equal(out, np.zeros(4))
      with aot_cache.real_execution():
        np.testing.assert_allclose(_cached_affine(x, scale=2.0), _affine(x, 2.0))
    self.assertFalse(aot_cache.in_warmup())


if __name__ == "__main__":
  absltest.main()
