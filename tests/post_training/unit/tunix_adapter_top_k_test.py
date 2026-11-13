# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`_compat_top_k` replicates a sharded reduction axis, including under jit.

`tests/conftest.py` imports JAX at collection, and a sibling post-training module assigns
`XLA_FLAGS=--xla_force_host_platform_device_count=1`, so an in-process `setdefault` leaves
`jax.device_count() == 1` and a skip would pass without exercising the traced sharding.
The collected test re-execs this file with the 4-device flag appended (last flag wins).
"""

import os
import re
import subprocess
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]

_REQUIRED_DEVICES = 4
_SENTINEL = "TUNIX_ADAPTER_TOP_K_TESTS_PASSED"
_RAN = re.compile(rf"{_SENTINEL} ran=(\d+)")

# Rank 2 so both mesh axes divide it. Maxima sit at different columns so a
# shim that returns the last index cannot match the replicated top_k.
_HOST = np.array(
    [
        [0.2, 1.5, 0.4, 3.0, 0.1, 2.2, 0.7, 1.1],
        [4.0, 0.3, 2.1, 0.8, 5.5, 1.2, 0.6, 3.3],
        [0.9, 6.0, 1.4, 2.8, 0.5, 1.7, 4.4, 0.2],
        [1.0, 0.4, 7.5, 2.0, 3.1, 0.8, 1.6, 5.0],
    ],
    dtype=np.float32,
)


@pytest.mark.post_training
@pytest.mark.cpu_only
def test_compat_top_k_on_a_four_device_cpu_mesh():
  """The only test pytest collects here; the class below runs inside the child process."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_REQUIRED_DEVICES}".strip()
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root

  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)

  report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert result.returncode == 0, report
  ran = _RAN.search(result.stdout)
  assert ran, f"the child did not report a completed run\n{report}"
  assert int(ran.group(1)) > 0, f"every test in the child skipped\n{report}"


def _assert_matches(got, reference):
  got_values, got_indices = got
  ref_values, ref_indices = reference
  np.testing.assert_array_equal(np.asarray(jax.device_get(got_values)), np.asarray(jax.device_get(ref_values)))
  np.testing.assert_array_equal(np.asarray(jax.device_get(got_indices)), np.asarray(jax.device_get(ref_indices)))


class CompatTopKTest(unittest.TestCase):
  """Traced and eager top_k through the production monkeypatch."""

  __test__ = False  # collected only via the subprocess entry point at the top of this file.

  @classmethod
  def setUpClass(cls):
    # Import before any top_k so jax.lax.top_k is the production shim.
    import maxtext.integration.tunix.tunix_adapter  # pylint: disable=unused-import,import-outside-toplevel

    if jax.lax.top_k.__name__ != "_compat_top_k":
      raise AssertionError(f"expected _compat_top_k, got {jax.lax.top_k.__name__}")
    devices = np.array(jax.devices()).reshape(2, 2)
    # Explicit axes keep the layout on the tracer (`f32[4@data,8@model]`). An auto mesh
    # drops it, so top_k would succeed without the shim and the test would not see the bug.
    cls.mesh = jax.sharding.Mesh(
        devices, ("data", "model"), axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit)
    )
    replicated_sharding = jax.sharding.NamedSharding(cls.mesh, jax.sharding.PartitionSpec(None, None))
    cls.replicated = jax.device_put(jnp.asarray(_HOST), replicated_sharding)
    cls.sharded = jax.device_put(
        jnp.asarray(_HOST), jax.sharding.NamedSharding(cls.mesh, jax.sharding.PartitionSpec("data", "model"))
    )
    cls.reference = jax.lax.top_k(cls.replicated, k=1)

  def test_jitted_sharded_top_k_matches_replicated(self):
    """Traced `…@data, …@model` top_k with the default axis, as gather_logprobs calls it."""

    @jax.jit
    def top1(x):
      return jax.lax.top_k(x, k=1)

    _assert_matches(top1(self.sharded), self.reference)

  def test_jitted_axes_and_unsharded_vocab_match_replicated(self):
    """axis=-1 and axis=1 agree, and an already-unsharded vocab axis is left alone."""

    @jax.jit
    def top1_neg(x):
      return jax.lax.top_k(x, k=1, axis=-1)

    @jax.jit
    def top1_pos(x):
      return jax.lax.top_k(x, k=1, axis=1)

    _assert_matches(top1_neg(self.sharded), self.reference)
    _assert_matches(top1_pos(self.sharded), self.reference)
    vocab_unsharded = jax.device_put(
        jnp.asarray(_HOST), jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec("data", None))
    )
    _assert_matches(top1_neg(vocab_unsharded), self.reference)

  def test_eager_sharded_top_k_matches_replicated(self):
    """Concrete `.sharding`, the path the shim already handled."""
    _assert_matches(jax.lax.top_k(self.sharded, k=1), self.reference)

  def test_oversized_k_raises_the_original_error(self):
    """k larger than the reduction axis is not swallowed or retyped."""
    k = _HOST.shape[1] + 1

    @jax.jit
    def top_k(x):
      return jax.lax.top_k(x, k=k)

    try:
      top_k(self.replicated)
    except Exception as ref_exc:  # pylint: disable=broad-exception-caught
      # Bound only for the duration of this block.
      ref_type = type(ref_exc)
      ref_message = str(ref_exc)
    else:
      self.fail("replicated top_k should raise when k exceeds the reduction axis")
    with self.assertRaises(ref_type) as got:
      top_k(self.sharded)
    self.assertIs(type(got.exception), ref_type)
    self.assertEqual(str(got.exception), ref_message)


if __name__ == "__main__":
  if jax.device_count() < _REQUIRED_DEVICES:
    raise SystemExit(
        f"needs {_REQUIRED_DEVICES} devices, got {jax.device_count()}; run this through pytest, which sets "
        f"XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}"
    )
  _loader = unittest.defaultTestLoader
  _result = unittest.TextTestRunner(verbosity=2).run(_loader.loadTestsFromTestCase(CompatTopKTest))
  if not _result.wasSuccessful():
    sys.exit(1)
  print(f"{_SENTINEL} ran={_result.testsRun - len(_result.skipped)}")
