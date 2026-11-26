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

"""Shared test gates for the mpmd/jax tests: Lineage, device counts, the hardware; and where things are.

Each gate decorates a unittest.TestCase class or a test method.  It skips (or, where a run asks for it, fails) and
applies the pytest marker, if any, that MaxText's CI selects on (its pytest.ini's: scheduled_only, tpu_only,
gpu_only, cpu_only).  The tests apply the markers themselves as @pytest.mark.<name> decorators, which
tests/utils/newly_added_detection.py reads statically (it sees no marker a gate applies: requires_gpu's gpu_only).

  slow              skip unless MAXTEXT_MPMD_SLOW_TESTS=1, which no CI job sets: a check of several minutes of CPU
                    compiles each, more than MaxText's scheduled CPU CI should carry, run by hand.  Marker
                    scheduled_only.  (A check of up to two minutes or so is a plain @pytest.mark.scheduled_only, which
                    MaxText's scheduled CI runs.)
  requires_lineage  skip unless Lineage (MaxText's experimental/lineage) imports; under MAXTEXT_MPMD_REQUIRE_LINEAGE=1
                    fail instead.  No marker.
  needs_devices(n)  skip unless jax has n devices.  No marker.  MaxText's CPU CI runs with 4; a class whose
                    pipeline stages fit either asks for 4 and takes stage_axes()'s stages.
  needs_ynn_off     skip where XLA:CPU's YNNPACK fusions are on: jaxlib >= 0.11.2 on CPU, XLA_FLAGS without
                    --xla_cpu_experimental_ynn_fusion_type= (and with jaxlib_before=, a jaxlib older than that).  No
                    marker.
  trips_cpu_heap_bug  skip on CPU under the jaxlibs whose XLA:CPU corrupts its heap on the class's programs
                    (CPU_HEAP_BUG_JAXLIBS: 0.12.0.x, 0.12.1), unless MAXTEXT_MPMD_RUN_JAXLIB_CPU_CRASH=1.  No marker.
  requires_gpu      skip unless jax runs on a GPU (of a compute capability, sm=), with the Python packages a kernel
                    needs installed (a Triton kernel's triton and jax_triton, which no requirements file brings).
                    Marker gpu_only.

Two compile-option sets, per program, for checks that hold two programs to the same bits: NO_EXCESS_PRECISION (XLA's
excess precision off) and ynn_off() (XLA:CPU's YNNPACK fusions off).

A module of CPU checks calls require_cpu() from its setUpModule.  A tpu_only class calls require_tpu(here) where it
sets up -- `here` whether jax runs on the TPUs it needs (tpu_devices(): a TPU VM's own, or Pathways' through
JAX_PLATFORMS=proxy) -- which skips, saying what jax runs on and how to put it on TPUs, or under
MAXTEXT_MPMD_REQUIRE_TPU=1 fails instead: a run meant for the TPU cannot pass as skips.  A TPU check of a TPU VM's
own libtpu (McJAX), such as which stages it offloads collectives to SparseCore on, calls skip_on_pathways(why).  A
class whose meshes take several devices asks for them with needs_devices: on CPU that is
XLA_FLAGS=--xla_force_host_platform_device_count=N (4 in MaxText's CPU CI; 8 for the classes that need them).  No
test module sets JAX_PLATFORMS or XLA_FLAGS itself: collected with other modules, a default set at import would apply
to every test of the session.

This module imports neither jax nor MaxText at import: a gate asks for devices or imports Lineage when its class sets
up or its test runs, never at collection.  assert_warns_regex stands in for unittest's assertWarnsRegex, which breaks
once transformers is imported (see its docstring).

Where things are, the same wherever the package lives (PACKAGE, its dotted name; in MaxText the tests are
tests.unit.experimental.mpmd.jax, apart from the package):
  package_dir()          the package's directory, with its data files (the launchers, libtpu_flags/, gpu.xla_flags).
  import_root([module])  the directory the package (or a test module) imports from, by its dotted name's depth: a
                         child process's working directory.
  child_pythonpath(test) PYTHONPATH for a child process that imports the package and the tests.
  is_maxtext(name)       is a module MaxText's own, not the package (nor a package above it)?
"""

import contextlib
import functools
import importlib
import importlib.util
import os
import re
import sys
import unittest
import warnings

try:
  import pytest
except ImportError:  # plain unittest without pytest installed: the gates still skip, with no markers
  pytest = None


def cpu_hint(devices=4):
  """How to run a CPU test: on `devices` fake CPU devices (MaxText's CPU CI runs with 4)."""
  return f"run with JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count={devices}"


def env_on(name):
  """Is the switch `name` (MAXTEXT_MPMD_SLOW_TESTS, MAXTEXT_MPMD_REQUIRE_LINEAGE, ...) set to 1?"""
  return os.environ.get(name) == "1"


# The package's dotted name: this module's, less its own last component.
PACKAGE = __name__.rpartition(".")[0]


def package_dir():
  """The package's directory, where its modules and data files are (the launchers, libtpu_flags/, gpu.xla_flags).  A
  test finds them here, never through its own directory: in MaxText the tests live apart from the package."""
  return os.path.dirname(os.path.abspath(__file__))


def import_root(module=None):
  """The directory a module imports from: its file's directory, one level up for each component of its package's
  dotted name.  `module` is the name of an imported module (a test's __name__; one run with `python3 -m` is __main__,
  whose spec keeps the name), by default this one: the package's root, a child process's working directory -- the
  repository root here, src/ in MaxText, whose tests import from the repository root (tests.unit.experimental...).
  None for a module run by its file's path, whose package Python does not know."""
  mod = sys.modules[module or __name__]
  spec = getattr(mod, "__spec__", None)
  if spec is None:
    return None
  root = os.path.dirname(os.path.abspath(mod.__file__))
  for _ in spec.parent.split(".") if spec.parent else ():
    root = os.path.dirname(root)
  return root


def child_pythonpath(test=None):
  """PYTHONPATH for a child process that imports the package and, `test` given (a test module's __name__), the
  tests: import_root(), the test's root where that is another one (MaxText's tests), then the PYTHONPATH inherited."""
  roots = [import_root(), import_root(test) if test else None, os.environ.get("PYTHONPATH")]
  return os.pathsep.join(filter(None, dict.fromkeys(roots)))


def mark(name):
  """The pytest marker `name` as a decorator (a no-op where pytest is not installed)."""
  if pytest is None:
    return lambda obj: obj
  return getattr(pytest.mark, name)


def _before(check):
  """A decorator that runs `check()` (which raises unittest.SkipTest, or an error) before a test method, or once
  before a TestCase class's setUpClass."""

  def decorate(obj):
    if isinstance(obj, type):
      own = obj.__dict__.get("setUpClass")

      def setUpClass(cls):
        check()
        if own is not None:
          own.__func__(cls)
        else:
          super(obj, cls).setUpClass()

      obj.setUpClass = classmethod(setUpClass)
      return obj

    @functools.wraps(obj)
    def run(*args, **kwargs):
      check()
      return obj(*args, **kwargs)

    return run

  return decorate


SLOW_TESTS = "MAXTEXT_MPMD_SLOW_TESTS"
MULTIHOST_TESTS = "MAXTEXT_MPMD_MULTIHOST_TESTS"


def require_multihost():
  """Raise unittest.SkipTest unless MAXTEXT_MPMD_MULTIHOST_TESTS=1, which no CI job sets: for a test that starts
  several JAX processes (multi-controller JAX over local CPU processes).  MaxText's CPU CI runs many test modules in
  each pytest worker, and there the out-of-memory killer took such children (exit -9).  Run them by hand --
  MAXTEXT_MPMD_MULTIHOST_TESTS=1 python3 -m pytest <the test>."""
  if not env_on(MULTIHOST_TESTS):
    raise unittest.SkipTest(
        f"{MULTIHOST_TESTS}=1 runs it, by hand (no CI job sets it): it starts several JAX processes, more memory than "
        "a CI worker running other test modules has"
    )


def slow(obj):
  """Skip the test unless MAXTEXT_MPMD_SLOW_TESTS=1, which no CI job sets, and mark it scheduled_only: a check of
  several minutes of CPU compiles (over two on a 4-device CPU replica of MaxText's CI, under load), run by hand --
  MAXTEXT_MPMD_SLOW_TESTS=1 python3 -m pytest <the test>."""
  why = f"{SLOW_TESTS}=1 runs it, by hand (no CI job sets it): several minutes of CPU compiles"
  return mark("scheduled_only")(unittest.skipUnless(env_on(SLOW_TESTS), why)(obj))


def needs_devices(n, why=""):
  """Skip the test, or the class (in its setUpClass), unless jax has `n` devices: what its meshes take, or more
  (`why` says so).  Asked when it runs: the question starts jax's backend."""

  def check():
    import jax  # pylint: disable=import-outside-toplevel

    have = jax.device_count()
    if have < n:
      note = f" ({why})" if why else ""
      raise unittest.SkipTest(f"needs {n} devices{note}, jax has {have}: on CPU, {cpu_hint(n)}")

  return _before(check)


def stage_axes(stages=2):
  """{'fsdp': ..., 'ep': ...} of each of `stages` pipeline stages on the devices jax has: FSDP 2 x EP 2 where there
  are 4 devices a stage (8 for two stages), else FSDP 1 x EP 2 (two stages on MaxText's CPU CI's 4 devices).  Both
  keep the expert axis -- its ring of experts, and the storage sharded over fsdp and expert together (ds_ep.FE), so
  the gathers and reduce-scatters move data either way; FSDP 1 has no fsdp-only collective.  A class whose stages
  take these asks for 2 * stages devices (needs_devices).  Asked when a class sets up or a test runs: the question
  starts jax's backend."""
  import jax  # pylint: disable=import-outside-toplevel

  return {"fsdp": 2, "ep": 2} if jax.device_count() >= 4 * stages else {"fsdp": 1, "ep": 2}


# XLA's excess precision (on by default) lets it drop an intermediate bf16 rounding where it fuses the producer into
# its consumer, so which roundings survive depends on the fusion: two programs doing the same bf16 arithmetic but
# fused differently part in the last bits (jax 0.12.0.1's XLA:CPU, the bf16 moments' b1 * mu in AdamW).  A test that
# holds one to the other bit for bit compiles both without it (jax.jit's or Lowered.compile's compiler_options).
NO_EXCESS_PRECISION = {"xla_allow_excess_precision": False}


def ynn_off(backend=None):
  """{'xla_cpu_experimental_ynn_fusion_type': ''} on CPU (`backend`, default jax's), else {}: per-program compile
  options without XLA:CPU's YNNPACK fusions, which are on by default.  A YNNPACK dot's summation order follows its
  operands' shapes, so a sharded program (per-device shapes) and its unsharded reference part in the last bits; and on
  jax >= 0.11.2 a bf16 dot no longer accumulates in fp32 there.  Without them a dot is XLA's own, fp32-accumulated,
  the same at any shape and thread count.  For test-only computations held to each other bit for bit; per program, so the
  process keeps its flags (no test module sets XLA_FLAGS)."""
  import jax  # pylint: disable=import-outside-toplevel

  return {"xla_cpu_experimental_ynn_fusion_type": ""} if (backend or jax.default_backend()) == "cpu" else {}


def version_tuple(version):
  """The version string `version`'s leading numeric components as a tuple of ints, () where it starts with none:
  '0.12.0.1' gives (0, 12, 0, 1).  A pre-release, post-release, nightly or local build reads as its release:
  '0.12.1rc1', '0.12.1.post1' and '0.12.1+cuda12' give (0, 12, 1), '0.12.2.dev20261201' gives (0, 12, 2) -- the dotted
  prefix that jax's own check of jaxlib's version reads (jax._src.lib.check_jaxlib_version)."""
  m = re.match(r"[0-9]+(?:\.[0-9]+)*", version)
  return tuple(int(x) for x in m.group(0).split(".")) if m else ()


def jaxlib_version():
  """jaxlib's version as a tuple of ints, e.g. (0, 12, 0, 1): its version module's __version__ by version_tuple, which
  starts no backend.  Not the module's __version_info__, which drops every component with a suffix: 0.12.1rc1's is
  (0, 12), which the gates below would take for no 0.12.1 build."""
  from jaxlib import version  # pylint: disable=import-outside-toplevel

  return version_tuple(getattr(version, "__version__", ""))


# XLA:CPU's YNNPACK fusions, which `YNN_FLAG=` (no fusion type) in XLA_FLAGS turns off for the process.  jaxlib runs
# them by default, and from 0.11.2 on a bf16 dot inside one no longer accumulates in fp32 as on 0.11.1.
YNN_FLAG = "--xla_cpu_experimental_ynn_fusion_type"


def ynn_on():
  """Why XLA:CPU's YNNPACK fusions are on in this process the way jaxlib 0.11.2 brought them, else None: jax on CPU,
  jaxlib >= 0.11.2, and XLA_FLAGS without `YNN_FLAG=` (the last of its YNN_FLAG settings counts, a quoted value
  without its quotes as XLA reads it: `YNN_FLAG=""` turns them off too; the backend read it when it started).  This
  starts jax's backend: ask where a class sets up or a test runs."""
  import jax  # pylint: disable=import-outside-toplevel

  if jax.default_backend() != "cpu" or jaxlib_version() < (0, 11, 2):
    return None
  setting = None
  for flag in os.environ.get("XLA_FLAGS", "").split():
    if flag.startswith(YNN_FLAG + "="):
      setting = flag[len(YNN_FLAG) + 1 :]
  # xla/parse_flags_from_env.cc takes a value in single or double quotes without them
  if setting is not None and len(setting) >= 2 and setting[0] == setting[-1] and setting[0] in "'\"":
    setting = setting[1:-1]
  if setting == "":
    return None
  return f"XLA:CPU's YNNPACK fusions are on (jaxlib {'.'.join(map(str, jaxlib_version()))}'s default)"


# Some checks fail with XLA:CPU's YNNPACK fusions on through bugs of particular jaxlibs, not by design -- gradients NaN
# or wrong, XLA:CPU crashing: seen on jaxlib 0.12.0.1 and 0.12.1 (2026-11-26), not on 0.11.1.  Their gates skip only
# before this jaxlib (needs_ynn_off's jaxlib_before), so that a later one runs them with the fusions on again: they
# pass once the bugs are fixed, or fail, and the bound moves up.  A nightly or rc of this jaxlib (0.12.2.dev20261201,
# 0.12.2rc1) is this jaxlib to jaxlib_version, not one before it.
YNN_BUG_JAXLIBS_BEFORE = (0, 12, 2)


def needs_ynn_off(why, jaxlib_before=None):
  """Skip the test, or the class (in its setUpClass), where XLA:CPU's YNNPACK fusions are on (ynn_on: jaxlib >= 0.11.2
  on CPU without `YNN_FLAG=` in XLA_FLAGS), saying `why` they break it and how to run it: with the flag in
  XLA_FLAGS.  For a check of the pipeline's own programs, which a test cannot compile with ynn_off()'s per-program
  options (several of them run inside others, where jax refuses compiler_options).  `jaxlib_before`, a version tuple
  (YNN_BUG_JAXLIBS_BEFORE), bounds the gate of a check the fusions break through a jaxlib bug: from that jaxlib on the
  check runs with them on."""

  def check():
    on = ynn_on()
    if on is not None and (jaxlib_before is None or jaxlib_version() < tuple(jaxlib_before)):
      raise unittest.SkipTest(f"{on}: {why} -- run it with XLA_FLAGS='... {YNN_FLAG}='")

  return _before(check)


# jaxlib 0.12.0.x's XLA:CPU corrupts its own heap compiling or running some programs -- the Qwen3.5-family MoE block's
# backward most of all (a top-k dispatch by sort / argsort, gathers, a masked grouped matmul, a scatter-add) -- and the
# process then dies somewhere later (SIGSEGV; SIGABRT from tcmalloc's "Attempt to free invalid pointer" or an absl
# CHECK in HloCostAnalysis or YnnFusionThunk) or, once seen, computes a NaN.  Not the YNNPACK fusions (it happens with
# them off), not a race in jax (one Python thread is enough), and no XLA:CPU flag tried avoids it.  `pytest
# ds_ep_qwen_test.py -k "QwenInit or QwenMoeVsPort"` on 8 CPU devices, 12 processes one at a time, crashed in 4 of 12
# on jaxlib 0.12.0.1 (6 of 12 with the YNNPACK fusions off), 10 of 12 on 0.12.1 and 0 of 12 on 0.11.1.  Under
# pytest-xdist the crash takes a worker and the tests it had left: about one worker a round of the tests, all in
# ds_ep_qwen_test, and none once its QwenMoeVsPort skips (2026-11-25).  A crash at the memory-map limit
# (vm.max_map_count, 65530 by default) is not this bug, and can come on any jaxlib.  Compiled programs' code takes the
# maps -- 27k to 44k for one case of ds_ep_test's RepeatScope.test_bitwise_vs_step_more (two tiny pipelines) on jaxlib
# 0.12.0.1 with the YNNPACK fusions off, under 10k for the processes above -- and the cure is freeing dead pipelines
# sooner (gc.collect(), as RepeatScope and BatchSplitLineage do), not a skip (2026-11-27).  In a test run's log the two
# look alike: pytest's default capture drops XLA's stderr when the process dies (the heap bug's messages above as well),
# leaving "Fatal Python error: Aborted" or "Segmentation fault" over a Python stack (at the map limit, in jax's
# backend_compile_and_load).  Rerun the test alone with -s or --capture=sys: at the limit XLA:CPU logs
# "LLVM compilation error: Cannot allocate memory" just before a SIGSEGV, or before "allocateMappedMemory failed",
# "LLVM ERROR: Unable to allocate section memory!" and a SIGABRT.  Or watch the process's map count
# (wc -l < /proc/PID/maps): it reaches vm.max_map_count just before the crash (the last sample, up to 2 s before it,
# read 64.4k to 65.5k), and a run that passes shows how near it came.  The jaxlib versions the heap corruption was seen
# on, as prefixes of jaxlib_version (a nightly or rc of one matches too):
CPU_HEAP_BUG_JAXLIBS = ((0, 12, 0), (0, 12, 1))


def cpu_heap_bug():
  """Why a test whose programs trip jaxlib's XLA:CPU heap corruption (CPU_HEAP_BUG_JAXLIBS) is not run here, else
  None: jax on CPU under one of those jaxlibs, and MAXTEXT_MPMD_RUN_JAXLIB_CPU_CRASH=1 (run it anyway) not set.  This
  starts jax's backend: ask where a class sets up or a test runs."""
  import jax  # pylint: disable=import-outside-toplevel

  if env_on("MAXTEXT_MPMD_RUN_JAXLIB_CPU_CRASH") or jax.default_backend() != "cpu":
    return None
  v = jaxlib_version()
  if not any(v[: len(p)] == p for p in CPU_HEAP_BUG_JAXLIBS):
    return None
  return (
      f"jaxlib {'.'.join(map(str, v))}'s XLA:CPU corrupts its heap compiling and running these programs (the Qwen MoE "
      "backward), and the process dies later, anywhere (testing.CPU_HEAP_BUG_JAXLIBS): "
      "MAXTEXT_MPMD_RUN_JAXLIB_CPU_CRASH=1 runs them anyway"
  )


def trips_cpu_heap_bug(obj):
  """Skip the test, or the class (in its setUpClass), where jaxlib's XLA:CPU heap corruption would take the process
  (cpu_heap_bug): a class whose programs crashed pytest-xdist workers on jaxlib 0.12.0.1 (ds_ep_qwen_test's
  QwenMoeVsPort)."""

  def check():
    why = cpu_heap_bug()
    if why is not None:
      raise unittest.SkipTest(why)

  return _before(check)(obj)


def gpu_missing(*modules, sm=None):
  """Why a GPU test cannot run here, else None: jax runs on no GPU, or on one whose compute capability's major version
  is not `sm` (10: Blackwell's sm_100 / sm_103), or a Python package it needs (`modules`) is not installed -- found
  with importlib.util.find_spec, not imported.  This starts jax's backend: ask where a class sets up or a test runs."""
  import jax  # pylint: disable=import-outside-toplevel

  d = jax.devices()[0]
  if d.platform != "gpu":
    return f"needs a GPU, jax runs on {d.platform}: run with JAX_PLATFORMS=cuda XLA_FLAGS="
  if sm is not None:
    cc = getattr(d, "compute_capability", None)
    try:
      major = int(float(cc))
    except (TypeError, ValueError):
      major = None
    if major != sm:
      return f"needs an sm_{sm}0 GPU, this one's compute capability is {cc}"
  missing = [m for m in modules if importlib.util.find_spec(m) is None]
  if missing:
    return f"needs {' and '.join(missing)} installed (pip install {' '.join(m.replace('_', '-') for m in missing)})"
  return None


def requires_gpu(*modules, sm=None, why=""):
  """Skip unless jax runs on a GPU -- of compute capability major `sm` where given -- with the Python packages
  `modules` installed (gpu_missing), checked when the class sets up or the test runs: the backend question starts
  jax's backend, never at collection.  Marks gpu_only.  @requires_gpu(), @requires_gpu(sm=10), or
  @requires_gpu('triton', 'jax_triton', why='the Triton kernel')."""

  def check():
    err = gpu_missing(*modules, sm=sm)
    if err is not None:
      raise unittest.SkipTest(f"{why}: {err}" if why else err)

  def decorate(obj):
    return mark("gpu_only")(_before(check)(obj))

  return decorate


def require_cpu(what="CPU tests"):
  """For setUpModule: skip the module unless jax runs on CPU (its bounds are CPU measurements, or it stands in for
  hardware it does not have)."""
  import jax  # pylint: disable=import-outside-toplevel

  if jax.default_backend() != "cpu":
    raise unittest.SkipTest(f"{what}: {cpu_hint()}")


# The JAX_PLATFORMS entries that put jax on TPUs: a TPU VM's own client (McJAX), and Pathways' IFRT proxy, whose
# devices report the platform 'tpu' too -- MaxText's CI runs its tpu_only tests both ways.
TPU_PLATFORMS = ("tpu", "proxy")


def tpu_devices():
  """jax's devices where they are TPUs -- a TPU VM's own, or Pathways' through the IFRT proxy, which report the
  platform 'tpu' as MaxText's tests/conftest.py reads it -- else [].  This starts jax's backend (under
  JAX_PLATFORMS=proxy after pathways.initialize()): ask where a class sets up or a test runs."""
  from maxtext.experimental.mpmd.jax import pathways  # pylint: disable=import-outside-toplevel

  pathways.initialize()
  import jax  # pylint: disable=import-outside-toplevel

  try:
    devs = jax.devices()
  except RuntimeError:  # a JAX_PLATFORMS naming a platform that does not start (a proxy that does not answer)
    return []
  return devs if devs and devs[0].platform == "tpu" else []


def tpu_hint():
  """What jax runs on (its devices: how many, of what kind) and, where JAX_PLATFORMS keeps it off TPUs, how to put
  it on them: JAX_PLATFORMS=tpu, or proxy on Pathways (TPU_PLATFORMS).  This starts jax's backend: ask where a class
  sets up or a test runs."""
  import jax  # pylint: disable=import-outside-toplevel

  try:
    devs = jax.devices()
    on = f"{jax.default_backend()} ({len(devs)} x {devs[0].device_kind})"
  except RuntimeError as e:  # a JAX_PLATFORMS naming a platform that does not start
    on = f"no backend ({e})"
  platforms = os.environ.get("JAX_PLATFORMS")
  if platforms is None or set(TPU_PLATFORMS) & {p.strip().lower() for p in platforms.split(",")}:
    return f"jax runs on {on}"
  return f"jax runs on {on}: JAX_PLATFORMS={platforms!r} -- run with JAX_PLATFORMS=tpu (proxy on Pathways)"


def require_tpu(here, what="a TPU"):
  """For a tpu_only class where it sets up (or a test): nothing if `here`, jax running on the TPU it needs (`what`);
  else unittest.SkipTest saying what jax runs on and how to put it on a TPU (tpu_hint) -- or, under
  MAXTEXT_MPMD_REQUIRE_TPU=1, set by a run meant for that TPU, an error, so that such a run cannot pass as skips (a
  JAX_PLATFORMS=cpu left over from the CPU tests, say)."""
  if here:
    return
  why = f"needs {what}; {tpu_hint()}"
  if env_on("MAXTEXT_MPMD_REQUIRE_TPU"):
    raise RuntimeError(f"MAXTEXT_MPMD_REQUIRE_TPU=1, but this {why}")
  raise unittest.SkipTest(why)


def on_pathways():
  """Are jax's devices Pathways' (the IFRT proxy's: pathways.is_proxy_device) rather than a TPU VM's own client's
  (McJAX)?  This starts jax's backend (under JAX_PLATFORMS=proxy after pathways.initialize()): ask where a class sets
  up or a test runs."""
  from maxtext.experimental.mpmd.jax import pathways  # pylint: disable=import-outside-toplevel

  pathways.initialize()
  import jax  # pylint: disable=import-outside-toplevel

  try:
    devs = jax.devices()
  except RuntimeError:  # a JAX_PLATFORMS naming a platform that does not start
    return False
  return bool(devs) and pathways.is_proxy_device(devs[0])


def skip_on_pathways(why):
  """For a TPU class where it sets up (or a test) that holds behaviour of a TPU VM's own libtpu which Pathways'
  servers do not share (`why`): unittest.SkipTest on Pathways' devices (on_pathways), MAXTEXT_MPMD_REQUIRE_TPU=1 or
  not -- such a check does not apply there; it is not missing a TPU."""
  if on_pathways():
    raise unittest.SkipTest(f"McJAX only, jax runs on Pathways' devices: {why}")


def is_maxtext(name):
  """Is the module `name` one of MaxText's own: maxtext or a module under it, but not the package or anything in it,
  nor a package above it (in MaxText the package is maxtext.experimental.mpmd.jax)."""
  return (name == "maxtext" or name.startswith("maxtext.")) and not (
      name == PACKAGE or name.startswith(PACKAGE + ".") or PACKAGE.startswith(name + ".")
  )


@functools.lru_cache(maxsize=None)
def lineage_missing():
  """Why Lineage does not import here (ds_ep_lineage.import_lineage's ImportError: the maxtext in use and what it
  lacks), else None."""
  from maxtext.experimental.mpmd.jax import ds_ep_lineage  # pylint: disable=import-outside-toplevel

  try:
    ds_ep_lineage.import_lineage()
  except ImportError as e:
    return str(e)
  return None


def requires_lineage(obj):
  """Skip unless Lineage imports, checked when the class sets up or the test runs; under MAXTEXT_MPMD_REQUIRE_LINEAGE=1
  a Lineage that does not import is an error, not a skip."""

  def check():
    err = lineage_missing()
    if err is None:
      return
    if env_on("MAXTEXT_MPMD_REQUIRE_LINEAGE"):
      raise ImportError(f"MAXTEXT_MPMD_REQUIRE_LINEAGE=1, but Lineage does not import: {err}")
    raise unittest.SkipTest(f"needs Lineage: {err}")

  return _before(check)(obj)


@contextlib.contextmanager
def assert_warns_regex(test, category, regex):
  """test.assertWarnsRegex(category, regex) without its reset of every loaded module's warning registry.  That
  reset reads an attribute of each module in sys.modules, and transformers' lazy alias modules (loaded with
  MaxText's model and trainer code) then import what they stand for, which imports torch: where torch is not
  installed, every later assertWarnsRegex in the process failed."""
  with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter("always")  # a warning given before shows again
    yield seen
  got = [str(w.message) for w in seen if issubclass(w.category, category)]
  test.assertTrue(any(re.search(regex, m) for m in got), f"no {category.__name__} matching {regex!r} in {got}")
