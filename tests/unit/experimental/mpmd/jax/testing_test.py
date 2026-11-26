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

"""testing.py's gates that read the jaxlib and XLA_FLAGS -- needs_ynn_off (XLA:CPU's YNNPACK fusions) and
trips_cpu_heap_bug (jaxlib 0.12.0.x's XLA:CPU heap corruption) -- jaxlib_version's reading of jaxlib's version
string, and the device ones (stage_axes, the TPU gates), with the backend and the version stubbed: no devices, no
compiles.  One scheduled_only check holds ynn_on to what XLA compiles, a child process per XLA_FLAGS.

    python3 -m pytest -q tests/unit/experimental/mpmd/jax/testing_test.py
"""

import concurrent.futures
import contextlib
import os
import subprocess
import sys
import types
import unittest
from unittest import mock

import jax
import jaxlib.version
import pytest

from maxtext.experimental.mpmd.jax import testing


@contextlib.contextmanager
def on(backend="cpu", version=(0, 12, 0, 1), **environ):
  """jax's default backend `backend`; jaxlib's version `version`: a tuple, for testing.jaxlib_version to return, or a
  string, for jaxlib's version module's __version__ (and the __version_info__ jaxlib makes of it), which jaxlib_version
  reads; and the environment variables given (None: unset)."""
  with contextlib.ExitStack() as stack:
    stack.enter_context(mock.patch.object(jax, "default_backend", lambda: backend))
    if isinstance(version, str):
      # jaxlib's __version_info__: the components that are all digits ('0.12.1rc1' -> (0, 12))
      info = tuple(int(i) for i in version.split(".") if i.isdigit())
      stack.enter_context(mock.patch.object(jaxlib.version, "__version__", version))
      stack.enter_context(mock.patch.object(jaxlib.version, "__version_info__", info))
    else:
      stack.enter_context(mock.patch.object(testing, "jaxlib_version", lambda: version))
    stack.enter_context(mock.patch.dict(os.environ))
    for k, v in environ.items():
      if v is None:
        os.environ.pop(k, None)
      else:
        os.environ[k] = v
    yield


def outcome(gated):
  """Run the TestCase class `gated` -> its one test's outcome: 'passed', or the skip reason."""
  result = unittest.TestResult()
  unittest.defaultTestLoader.loadTestsFromTestCase(gated).run(result)
  if result.errors or result.failures:
    raise AssertionError(result.errors + result.failures)
  return result.skipped[0][1] if result.skipped else "passed"


class JaxlibVersion(unittest.TestCase):
  """jaxlib_version's reading of jaxlib's version string -- a pre-release, nightly, post-release or local build as its
  release -- and the gates that read it."""

  def test_version_tuple(self):
    """A version string's leading numeric components, whatever follows them: PEP 440's pre-release (a, b, rc),
    post-release and dev (nightly) suffixes, and a local label; () for none."""
    for version, want in (
        ("0.11.1", (0, 11, 1)),
        ("0.12.0.1", (0, 12, 0, 1)),
        ("0.12.10", (0, 12, 10)),
        ("0.12.1rc1", (0, 12, 1)),
        ("0.12.0.1rc0", (0, 12, 0, 1)),
        ("0.13.0a1", (0, 13, 0)),
        ("0.13.0b2", (0, 13, 0)),
        ("0.12.2.dev20261201", (0, 12, 2)),
        ("0.12.1.post1", (0, 12, 1)),
        ("0.12.1post1", (0, 12, 1)),
        ("0.12.1+cuda12", (0, 12, 1)),
        ("0.12.2.dev20261201+selfbuilt", (0, 12, 2)),
        ("", ()),
        ("unknown", ()),
    ):
      with self.subTest(version=version):
        self.assertEqual(testing.version_tuple(version), want)

  def test_gates_on_version_strings(self):
    """jaxlib_version reads __version__, which jaxlib's own __version_info__ cuts short at a component with a suffix
    ('0.12.1rc1' -> (0, 12)): an rc of 0.11.2 has the YNNPACK fusions on, the heap-bug gate skips under an rc or a local
    build of a jaxlib of CPU_HEAP_BUG_JAXLIBS, and a gate bounded by YNN_BUG_JAXLIBS_BEFORE runs its check under an rc
    of 0.12.2 or later, as under 0.12.2's nightlies."""

    @testing.needs_ynn_off("a jaxlib bug breaks this check", jaxlib_before=testing.YNN_BUG_JAXLIBS_BEFORE)
    class Bounded(unittest.TestCase):

      def test_it(self):
        pass

    for version, want, heap_bug, bounded_skips in (
        ("0.11.1", (0, 11, 1), False, False),
        ("0.11.2rc2", (0, 11, 2), False, True),
        ("0.12.0rc1", (0, 12, 0), True, True),
        ("0.12.0.1", (0, 12, 0, 1), True, True),
        ("0.12.1rc1", (0, 12, 1), True, True),
        ("0.12.1+cuda12", (0, 12, 1), True, True),
        ("0.12.2.dev20261201", (0, 12, 2), False, False),
        ("0.12.2rc1", (0, 12, 2), False, False),
        ("0.12.3rc1", (0, 12, 3), False, False),
        ("0.12.10rc1", (0, 12, 10), False, False),
    ):
      with self.subTest(version=version), on(version=version, XLA_FLAGS=None, MAXTEXT_MPMD_RUN_JAXLIB_CPU_CRASH=None):
        self.assertEqual(testing.jaxlib_version(), want)
        self.assertEqual(testing.cpu_heap_bug() is not None, heap_bug)
        self.assertEqual(outcome(Bounded) != "passed", bounded_skips)


class NeedsYnnOff(unittest.TestCase):
  """ynn_on's reading of the jaxlib and XLA_FLAGS, and the needs_ynn_off gate built on it."""

  def test_ynn_on(self):
    """On from jaxlib 0.11.2 on CPU unless the last --xla_cpu_experimental_ynn_fusion_type= setting in XLA_FLAGS has
    no value -- a quoted value read without its quotes, as XLA reads it; never off CPU or before 0.11.2 (whose bf16
    dots accumulate in fp32 inside the fusions too)."""
    flag = testing.YNN_FLAG
    for flags, want in (
        (None, True),
        ("--xla_force_host_platform_device_count=8", True),
        (f"--xla_force_host_platform_device_count=8 {flag}=", False),
        (f"{flag}=dot", True),
        (f"{flag}= {flag}=dot", True),
        (f"{flag}=dot {flag}=", False),
        (f'{flag}=""', False),
        (f"{flag}=''", False),
        (f'{flag}="" {flag}=dot', True),
        (f"{flag}=dot {flag}=''", False),
        (f'{flag}="dot"', True),
        (f'{flag}=" "', True),  # one space, a value
        (f"{flag}=\"'", True),  # quotes that do not pair: kept
    ):
      with self.subTest(flags=flags), on(XLA_FLAGS=flags):
        self.assertEqual(testing.ynn_on() is not None, want)
    for backend, version in (("gpu", (0, 12, 0, 1)), ("tpu", (0, 12, 0, 1)), ("cpu", (0, 11, 1))):
      with self.subTest(backend=backend, version=version), on(backend, version, XLA_FLAGS=None):
        self.assertIsNone(testing.ynn_on())
    with on(XLA_FLAGS=None):
      self.assertEqual(testing.ynn_on(), "XLA:CPU's YNNPACK fusions are on (jaxlib 0.12.0.1's default)")

  def test_gate(self):
    """A skip that says why and how to run the test: with the flag in XLA_FLAGS."""

    @testing.needs_ynn_off("they break this check")
    class Gated(unittest.TestCase):

      def test_it(self):
        pass

    with on(XLA_FLAGS="--xla_force_host_platform_device_count=8"):
      self.assertEqual(
          outcome(Gated),
          "XLA:CPU's YNNPACK fusions are on (jaxlib 0.12.0.1's default): they break this check -- run it with "
          "XLA_FLAGS='... --xla_cpu_experimental_ynn_fusion_type='",
      )
    with on(XLA_FLAGS=f"--xla_force_host_platform_device_count=8 {testing.YNN_FLAG}="):
      self.assertEqual(outcome(Gated), "passed")

  def test_jaxlib_before(self):
    """A gate with jaxlib_before skips with the fusions on only before that jaxlib (YNN_BUG_JAXLIBS_BEFORE: from
    0.12.2 on, its nightlies and rcs included), and not with them off; without it, on any jaxlib from 0.11.2 on."""

    def gated(**kw):
      @testing.needs_ynn_off("a jaxlib bug breaks this check", **kw)
      class Gated(unittest.TestCase):

        def test_it(self):
          pass

      return Gated

    self.assertEqual(testing.YNN_BUG_JAXLIBS_BEFORE, (0, 12, 2))
    bounded, unbounded = gated(jaxlib_before=testing.YNN_BUG_JAXLIBS_BEFORE), gated()
    for version, skips in (
        ((0, 11, 2), True),
        ((0, 12, 0, 1), True),
        ((0, 12, 1), True),
        ((0, 12, 1, 3), True),
        ((0, 12, 2), False),  # 0.12.2.dev20261201's and 0.12.2rc1's jaxlib_version too
        ((0, 12, 10), False),
        ((0, 13, 0), False),
    ):
      with self.subTest(version=version), on(version=version, XLA_FLAGS=None):
        self.assertEqual(outcome(bounded) != "passed", skips)
        self.assertRegex(outcome(unbounded), r"^XLA:CPU's YNNPACK fusions are on .*: a jaxlib bug breaks this check")
    for version, flags in (((0, 11, 1), None), ((0, 12, 0, 1), f"{testing.YNN_FLAG}=")):
      with self.subTest(version=version, flags=flags), on(version=version, XLA_FLAGS=flags):
        self.assertEqual((outcome(bounded), outcome(unbounded)), ("passed", "passed"))

  @pytest.mark.scheduled_only
  def test_ynn_on_is_what_xla_compiles(self):
    """ynn_on's answer against XLA's, a child process per XLA_FLAGS spelling (seconds): a bf16 dot compiles to a YNNPACK
    fusion (__ynn_fusion) exactly where ynn_on says the fusions are on.  From jaxlib 0.11.2 on: an older one fuses too,
    in fp32, which ynn_on does not count."""
    if testing.jaxlib_version() < (0, 11, 2):
      self.skipTest(f"jaxlib {testing.jaxlib_version()}: ynn_on counts no fusions before 0.11.2")
    flag = testing.YNN_FLAG
    code = (
        "import jax, jax.numpy as jnp\n"
        f"from {testing.PACKAGE} import testing\n"
        "x = jnp.ones((256, 256), jnp.bfloat16)\n"
        "hlo = jax.jit(lambda a, b: a @ b).lower(x, x).compile().as_text()\n"
        "print('__ynn_fusion' in hlo, testing.ynn_on() is not None)\n"
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=testing.child_pythonpath(__name__))

    def run(flags):
      return subprocess.run(
          [sys.executable, "-c", code],
          env=dict(env, XLA_FLAGS=flags),
          capture_output=True,
          text=True,
          timeout=300,
          check=False,
      )

    spellings = ("", f"{flag}=", f'{flag}=""', f"{flag}=''", f"{flag}=dot", f'{flag}=" "', f"{flag}=dot {flag}=''")
    with concurrent.futures.ThreadPoolExecutor(len(spellings)) as pool:
      runs = dict(zip(spellings, pool.map(run, spellings)))
    for s, r in runs.items():
      with self.subTest(XLA_FLAGS=s):
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        fused, said_on = r.stdout.split()[-2:]
        self.assertEqual(said_on, fused)


class Devices(unittest.TestCase):
  """stage_axes' stages for the devices there are, and what the TPU gates read of the platform: JAX_PLATFORMS=proxy
  (Pathways) is TPUs, as a TPU VM's own client is; the devices stubbed."""

  def test_stage_axes(self):
    """FSDP 2 x EP 2 a stage where the devices hold 4 a stage, else FSDP 1 x EP 2: two stages on MaxText's CPU CI's 4
    devices."""
    for n, stages, want in ((8, 2, (2, 2)), (16, 2, (2, 2)), (4, 2, (1, 2)), (8, 4, (1, 2)), (16, 4, (2, 2))):
      with self.subTest(devices=n, stages=stages), mock.patch.object(jax, "device_count", lambda n=n: n):
        self.assertEqual(testing.stage_axes(stages), {"fsdp": want[0], "ep": want[1]})

  def test_tpu_hint(self):
    """Where JAX_PLATFORMS names tpu or proxy (or nothing), the hint says what jax runs on; where it keeps jax off
    TPUs, how to put it on them."""
    dev = types.SimpleNamespace(platform="tpu", device_kind="TPU v6 lite")
    with mock.patch.object(jax, "devices", lambda: [dev] * 4), mock.patch.object(jax, "default_backend", lambda: "tpu"):
      for platforms in (None, "tpu", "proxy", "proxy,cpu", " Proxy"):
        with self.subTest(JAX_PLATFORMS=platforms), on("tpu", JAX_PLATFORMS=platforms):
          self.assertEqual(testing.tpu_hint(), "jax runs on tpu (4 x TPU v6 lite)")
    with on(JAX_PLATFORMS="cpu"):
      self.assertRegex(
          testing.tpu_hint(),
          r"^jax runs on cpu \(\d+ x cpu\): JAX_PLATFORMS='cpu' -- run with " r"JAX_PLATFORMS=tpu \(proxy on Pathways\)$",
      )

  def test_tpu_devices(self):
    """The devices where they report the platform 'tpu' -- a TPU VM's, or Pathways' through the proxy -- else none."""
    tpu, cpu = types.SimpleNamespace(platform="tpu"), types.SimpleNamespace(platform="cpu")
    for devs, want in (([tpu] * 4, [tpu] * 4), ([cpu] * 4, [])):
      with mock.patch.object(jax, "devices", lambda devs=devs: devs):
        self.assertEqual(testing.tpu_devices(), want)

    def no_backend():
      raise RuntimeError("Unable to initialize backend 'proxy'")

    with mock.patch.object(jax, "devices", no_backend):
      self.assertEqual(testing.tpu_devices(), [])

  def test_require_tpu(self):
    """Off the TPU it needs a class skips, saying what jax runs on -- or under MAXTEXT_MPMD_REQUIRE_TPU=1 fails."""
    with on(MAXTEXT_MPMD_REQUIRE_TPU=None):
      testing.require_tpu(True)
      with self.assertRaisesRegex(unittest.SkipTest, r"^needs TPUs of 4 devices; jax runs on cpu"):
        testing.require_tpu(False, "TPUs of 4 devices")
    with on(MAXTEXT_MPMD_REQUIRE_TPU="1"), self.assertRaisesRegex(RuntimeError, r"MAXTEXT_MPMD_REQUIRE_TPU=1, but"):
      testing.require_tpu(False)

  def test_skip_on_pathways(self):
    """A McJAX-only TPU check skips on Pathways' devices (their client's runtime_type 'proxy/...'), under
    MAXTEXT_MPMD_REQUIRE_TPU=1 too, and goes on on a TPU VM's own devices and on CPU."""
    client = lambda runtime: types.SimpleNamespace(runtime_type=runtime)
    proxy = types.SimpleNamespace(platform="tpu", client=client("proxy/pathways"))
    mcjax = types.SimpleNamespace(platform="tpu", client=client("pjrt_ifrt"))
    cpu = types.SimpleNamespace(platform="cpu")
    for devs, pathways in (([proxy] * 8, True), ([mcjax] * 8, False), ([cpu] * 4, False)):
      with (
          self.subTest(platform=devs[0].platform, pathways=pathways),
          mock.patch.object(jax, "devices", lambda devs=devs: devs),
          on(MAXTEXT_MPMD_REQUIRE_TPU="1"),
      ):
        self.assertEqual(testing.on_pathways(), pathways)
        if pathways:
          with self.assertRaisesRegex(unittest.SkipTest, r"^McJAX only, jax runs on Pathways' devices: why$"):
            testing.skip_on_pathways("why")
        else:
          testing.skip_on_pathways("why")


class Slow(unittest.TestCase):
  """The checks of several minutes skip unless MAXTEXT_MPMD_SLOW_TESTS=1, saying how to run them, and are marked
  scheduled_only."""

  def test_gate(self):
    """The switch is read where the test is decorated (at import), as unittest.skipUnless reads its condition."""

    def gated():
      class Gated(unittest.TestCase):

        @testing.slow
        def test_it(self):
          pass

      return Gated

    with on(MAXTEXT_MPMD_SLOW_TESTS=None):
      off = gated()
    with on(MAXTEXT_MPMD_SLOW_TESTS="1"):
      by_hand = gated()
    self.assertEqual(
        outcome(off), "MAXTEXT_MPMD_SLOW_TESTS=1 runs it, by hand (no CI job sets it): several minutes of CPU compiles"
    )
    self.assertEqual(outcome(by_hand), "passed")
    for g in (off, by_hand):
      self.assertIn("scheduled_only", [m.name for m in g.test_it.pytestmark])


class TripsCpuHeapBug(unittest.TestCase):
  """cpu_heap_bug's reading of the jaxlib and the switch, and the trips_cpu_heap_bug gate built on it."""

  def test_versions(self):
    """The jaxlibs of CPU_HEAP_BUG_JAXLIBS, by version prefix (0.12.10 is not 0.12.1), on CPU only;
    MAXTEXT_MPMD_RUN_JAXLIB_CPU_CRASH=1 runs the test anyway."""
    for version, want in (
        ((0, 12, 0, 1), True),
        ((0, 12, 0), True),
        ((0, 12, 1), True),
        ((0, 12, 10), False),
        ((0, 12, 2), False),
        ((0, 11, 1), False),
        ((0, 13, 0), False),
    ):
      with self.subTest(version=version), on(version=version, MAXTEXT_MPMD_RUN_JAXLIB_CPU_CRASH=None):
        self.assertEqual(testing.cpu_heap_bug() is not None, want)
    with on(MAXTEXT_MPMD_RUN_JAXLIB_CPU_CRASH="1"):
      self.assertIsNone(testing.cpu_heap_bug())
    with on("gpu", MAXTEXT_MPMD_RUN_JAXLIB_CPU_CRASH=None):
      self.assertIsNone(testing.cpu_heap_bug())

  def test_gate(self):
    """The class skips naming the bug and the switch, and runs with the switch."""

    @testing.trips_cpu_heap_bug
    class Gated(unittest.TestCase):

      def test_it(self):
        pass

    with on(MAXTEXT_MPMD_RUN_JAXLIB_CPU_CRASH=None):
      why = outcome(Gated)
    self.assertRegex(why, r"^jaxlib 0\.12\.0\.1's XLA:CPU corrupts its heap .* MAXTEXT_MPMD_RUN_JAXLIB_CPU_CRASH=1 runs")
    with on(MAXTEXT_MPMD_RUN_JAXLIB_CPU_CRASH="1"):
      self.assertEqual(outcome(Gated), "passed")


if __name__ == "__main__":
  unittest.main()
