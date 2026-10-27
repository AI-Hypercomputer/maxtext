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

"""Unit test for the PRNG impl that `generate_param_only_checkpoint.main` restores with.

`train.py` writes checkpoints with `unsafe_rbg` keys, so the converter must build its
restore target with the same impl; otherwise the `nnx_aux` RNG key shapes disagree
((layers, 2) threefry vs (layers, 4) rbg) and Orbax refuses to load (b/576849994).
"""

import unittest
from unittest import mock

import jax

from maxtext.utils import generate_param_only_checkpoint


class MainPrngImplTest(unittest.TestCase):
  """`main` must switch to unsafe_rbg before the config (and the restore target) is built."""

  def setUp(self):
    super().setUp()
    self.addCleanup(jax.config.update, "jax_default_prng_impl", jax.config.jax_default_prng_impl)

  def test_main_sets_unsafe_rbg_before_initializing_the_config(self):
    # The end-to-end scripts run the converter as its own process, i.e. on JAX's default impl.
    jax.config.update("jax_default_prng_impl", "threefry2x32")
    argv = ["", "base.yml", "load_full_state_path=gs://unused"]
    impl_seen_by_initialize = []

    def fake_initialize(seen_argv):
      impl_seen_by_initialize.append(jax.config.jax_default_prng_impl)
      self.assertEqual(seen_argv, argv)
      return mock.sentinel.config

    with mock.patch.object(generate_param_only_checkpoint.pyconfig, "initialize", side_effect=fake_initialize):
      with mock.patch.object(generate_param_only_checkpoint, "generate_decode_checkpoint") as generate:
        generate_param_only_checkpoint.main(argv)

    self.assertEqual(impl_seen_by_initialize, ["unsafe_rbg"])
    self.assertEqual(jax.config.jax_default_prng_impl, "unsafe_rbg")
    generate.assert_called_once_with(mock.sentinel.config)


if __name__ == "__main__":
  unittest.main()
