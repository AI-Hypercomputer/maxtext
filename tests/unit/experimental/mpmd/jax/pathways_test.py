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

"""Pathways (JAX_PLATFORMS=proxy) support: pathways.py's helpers and ds_ep_pipeline's pieces of the mode, on CPU; the
one-process --layout slices pipeline Pathways runs, on CPU; and that pipeline on TPUs, Pathways' or a TPU VM's own.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 -m pytest -q tests/unit/experimental/mpmd/jax/pathways_test.py

Under Pathways one client process drives every device, so ds_ep_pipeline runs its one-process path: every stage
local, each boundary one IFRT CopyArrays (transfer.Channel).  `OneProcessSlices` runs that path on CPU devices cut
into two fake slices, through main's Pathways-only branches (the mode's answers patched in its child process:
AS_PATHWAYS), and holds it bit for bit to the plain one-process path.  `OnTpu` (marker
tpu_only) runs the pipeline in this process on one slice of TPUs with 4 devices or more (on a job of several slices
it skips), through the IFRT proxy or a TPU VM's own client: MaxText's CI runs its tpu_only tests both ways, on a
v6e-4.  On Pathways (one client may hold a proxy at a time: run it in a pytest process of its own, and no test here
starts a child interpreter there):

    JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 MAXTEXT_MPMD_REQUIRE_TPU=1 \\
        python3 -m pytest -q tests/unit/experimental/mpmd/jax/pathways_test.py -k OnTpu

With MAXTEXT_MPMD_PATHWAYS_TRACE_DIR=gs://... it profiles a step there too and checks a Pathways worker's device
planes came back.  `SplitBackwardOnTpu` (tpu_only, scheduled_only) holds the zero-bubble schedules' split backward
to the backward it is cut from on the TPU's compiler (ds_ep_test.check_split) -- and the split last chunk
(--zb-split-last) to the fused program it is cut from (ds_ep_test.check_split_last) -- by the TPU contract
(`tpu_split_tol`), and `OnTpu`'s zero-bubble runs hold zbh1 and zbh2 to 1F1B by it (`assert_near_1f1b`).
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import glob
import inspect
import io
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import jax
import pytest

from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
from tests.unit.experimental.mpmd.jax import multihost_test
from maxtext.experimental.mpmd.jax import pathways
from maxtext.experimental.mpmd.jax import profiling
from maxtext.experimental.mpmd.jax import testing

# The zero-bubble schedules' contract on a TPU's compiler, measured on a tpu7x-8 (TPU7x; jax 0.12.0.1; McJAX with libtpu
# 0.0.50.1 and local docker Pathways 20261117) at SplitBackwardOnTpu's emb 1024 (two seeds) and on OnTpu's TINY.  A
# chunk's backward cut into I and W -- or the fused last chunk cut into AI and W -- is not the program it is cut from
# bit for bit there, as it is on CPU: XLA:TPU fuses each half by itself and, under its default options, keeps bf16
# intermediates at higher precision across the fusions it makes (excess precision).  Under the default options every
# output leaf, dx and each gradient, is within TPU_SPLIT_DEFAULT_TOL of the fused program's, relative to the leaf's
# largest value (seen: up to 1.14%, a few bf16 roundings).  With excess precision off (testing.NO_EXCESS_PRECISION) the
# halves are bit for bit -- dx, the loss and every gradient -- but for MLA's q_norm and kv_norm scales (TPU_NORM_SCALES:
# up to 1.46%), held to TPU_SPLIT_NORM_TOL, and the sum of squares the chunk's own reduce-scatter adds over its
# gradients, those among them (up to 2.5e-4), held to TPU_SPLIT_SUMSQ_TOL.  The pipeline, zbh1 / zbh2 against 1F1B: the
# first loss bit for bit (the same forward programs on the same weights; not where the last chunk splits, whose loss its
# AI computes: about 2.9e-6 apart), each loss within ZB_TPU_LOSS_RTOL and each gradient norm within ZB_TPU_GNORM_RTOL
# (OnTpu's TINY over three warmup steps: up to 4.0e-4 and 6.8e-3, the latter Pathways' at four stages with the last
# chunk split, the AdamW updates carrying the gradients' last bits into the next step); zbh2 is zbh1 bit for bit (the
# same programs, each chunk's gradients added in microbatch order), and at two stages, where nothing splits, both are
# 1F1B bit for bit.  Each bound is 2.5 to 5 times the largest seen on either backend.
TPU_SPLIT_DEFAULT_TOL = 0.03
TPU_NORM_SCALES = ("['q_norm']['scale']", "['kv_norm']['scale']")
TPU_SPLIT_NORM_TOL = 0.04
TPU_SPLIT_SUMSQ_TOL = 1e-3
ZB_TPU_LOSS_RTOL = 2e-3
ZB_TPU_GNORM_RTOL = 2e-2


def tpu_split_tol(name, mode):
  """The TPU contract of a split chunk against the program it is cut from (ds_ep_test.check_split's and
  check_split_last's `tol`): under XLA's default options every leaf within TPU_SPLIT_DEFAULT_TOL; with excess
  precision off bit for bit, but MLA's q_norm / kv_norm scales within TPU_SPLIT_NORM_TOL and the sum of squares over
  the gradients ('sumsq', rs_self's) within TPU_SPLIT_SUMSQ_TOL."""
  if mode == "default":
    return TPU_SPLIT_DEFAULT_TOL
  if name.endswith(TPU_NORM_SCALES):
    return TPU_SPLIT_NORM_TOL
  return TPU_SPLIT_SUMSQ_TOL if name == "sumsq" else 0.0


def assert_near_1f1b(t, got, ref, what, same_forward=True):
  """A zero-bubble run's warmup losses and gradient norms (dsp.main's record `got`) against 1F1B's (`ref`) on a TPU:
  every loss within ZB_TPU_LOSS_RTOL and every gradient norm within ZB_TPU_GNORM_RTOL of 1F1B's, relative, and with
  `same_forward` the first loss bit for bit -- the same forward programs on the same weights (not so where the last
  chunk splits: its loss comes from AI, cut from the fused program)."""
  if same_forward:
    t.assertEqual(got["warmup_losses"][0], ref["warmup_losses"][0], f"{what}: the first loss")
  for kind, rtol in (("losses", ZB_TPU_LOSS_RTOL), ("gnorms", ZB_TPU_GNORM_RTOL)):
    a, b = got[f"warmup_{kind}"], ref[f"warmup_{kind}"]
    rel = max(abs(x - y) / abs(y) for x, y in zip(a, b, strict=True))
    t.assertLessEqual(rel, rtol, f"{what}: {kind} {a} against 1F1B's {b}, {rel:.3g} apart")


@dataclasses.dataclass
class Client:
  """A jax client as pathways reads one: its runtime type and platform version."""

  runtime_type: str = "proxy/pathways"
  platform_version: str = "Pathways\ncl/42"


@dataclasses.dataclass(frozen=True)
class Dev:
  id: int
  client: object = None


@contextlib.contextmanager
def modules(**mods):
  """sys.modules with `mods` in place of those names for the block (dots as '__'), and only those restored after:
  mock.patch.dict on sys.modules would drop every module first imported inside the block, and the next import of jax
  then loads jaxlib's extension a second time (absl aborts the process)."""
  mods = {k.replace("__", "."): v for k, v in mods.items()}
  saved = {k: sys.modules[k] for k in mods if k in sys.modules}
  sys.modules.update(mods)
  try:
    yield
  finally:
    for k in mods:
      if k in saved:
        sys.modules[k] = saved[k]
      else:
        sys.modules.pop(k, None)


@contextlib.contextmanager
def env(**kv):
  """os.environ with `kv` set (a None value unset) for the block."""
  with mock.patch.dict(os.environ):
    for k, v in kv.items():
      if v is None:
        os.environ.pop(k, None)
      else:
        os.environ[k] = v
    yield


@contextlib.contextmanager
def redirected(k, fd):
  """This process's file descriptor `k` (1 its stdout, 2 its stderr) a copy of `fd` for the block, as a shell's
  `> file` or `2> file` makes it."""
  sys.stdout.flush()
  sys.stderr.flush()
  saved = os.dup(k)
  os.dup2(fd, k)
  try:
    yield
  finally:
    os.dup2(saved, k)
    os.close(saved)


def slices_args(axes=None):
  """multihost_test's two-slice TINY pipeline (`_slices_args`) at `axes` ({'fsdp', 'ep'}; testing.stage_axes, the
  stages the devices there are hold, by default): --layout slices, two stages of fsdp x ep devices each, one a
  slice; MPMD_FAKE_SLICES=2 cuts the devices into the two slices."""
  a = axes or testing.stage_axes()
  return multihost_test.MultiHostTest._slices_args(a["fsdp"], a["ep"])


class Mode(unittest.TestCase):
  """Which process and which devices are Pathways', asked without starting a backend."""

  def test_is_proxy(self):
    """'proxy' the first of JAX_PLATFORMS's platforms, the one jax makes its default backend: what Pipeline then
    finds its devices to be (is_proxy_device).  Further down the list another platform's devices are the default."""
    for platforms, want in (
        ("proxy", True),
        ("proxy,cpu", True),
        (" Proxy ,cpu", True),
        ("cpu,proxy", False),
        ("cpu", False),
        ("tpu", False),
        ("", False),
        ("proxyish", False),
    ):
      with self.subTest(platforms=platforms), env(JAX_PLATFORMS=platforms):
        self.assertEqual(pathways.is_proxy(platforms), want)
        self.assertEqual(pathways.is_proxy(), want)
    with env(JAX_PLATFORMS=None):
      self.assertFalse(pathways.is_proxy())

  def test_is_proxy_device(self):
    """By the device's client: 'proxy/pathways' through the IFRT proxy, 'pjrt_ifrt' on a TPU VM's own client."""
    self.assertTrue(pathways.is_proxy_device(Dev(0, Client())))
    self.assertFalse(pathways.is_proxy_device(Dev(0, Client(runtime_type="pjrt_ifrt"))))
    self.assertFalse(pathways.is_proxy_device(Dev(0)))
    self.assertFalse(pathways.is_proxy_device(object()))

  def test_multi_controller(self):
    """The one rule for the pipeline's multi-controller logic: several processes (jax.process_count() > 1) -- not
    this test's one process, never Pathways' one client (a proxy reporting several processes is refused); a device
    set by its processes, the IFRT proxy's devices one client's without a look at the others."""
    self.assertFalse(pathways.multi_controller())
    with env(JAX_PLATFORMS="proxy"):
      self.assertFalse(pathways.multi_controller())
    with mock.patch.object(jax, "process_count", lambda: 4):
      self.assertTrue(pathways.multi_controller())
      with env(JAX_PLATFORMS="proxy"), self.assertRaisesRegex(RuntimeError, r"\(Pathways\) reports 4 processes"):
        pathways.multi_controller()
    hosts = [types.SimpleNamespace(process_index=q) for q in (0, 0, 1)]
    self.assertFalse(pathways.multi_controller(hosts[:2]))
    self.assertTrue(pathways.multi_controller(iter(hosts[1:])))
    self.assertFalse(pathways.multi_controller([]))
    self.assertFalse(pathways.multi_controller([Dev(0, Client()), object()]))  # the second never looked at


class Initialize(unittest.TestCase):
  """pathwaysutils.initialize() once, under JAX_PLATFORMS=proxy only, before any backend use -- or, where the
  backends are up already, its profiler patch and its cache switch alone (its backend factory would raise)."""

  def setUp(self):
    self.calls = []
    pw = types.ModuleType("pathwaysutils")
    pw.initialize = lambda: self.calls.append("initialize")
    prof = types.ModuleType("pathwaysutils.profiling")
    prof.monkey_patch_jax = lambda: self.calls.append("monkey_patch_jax")
    pw.profiling = prof
    self.enterContext(modules(pathwaysutils=pw, pathwaysutils__profiling=prof))
    self.enterContext(mock.patch.object(pathways, "_initialized", False))

  def test_off_proxy_nothing(self):
    with env(JAX_PLATFORMS="cpu"):
      self.assertFalse(pathways.initialize())
    self.assertEqual(self.calls, [])

  def test_once_under_proxy(self):
    from jax._src import xla_bridge

    with env(JAX_PLATFORMS="proxy"), mock.patch.object(xla_bridge, "backends_are_initialized", lambda: False):
      self.assertTrue(pathways.initialize())
      self.assertTrue(pathways.initialize())
    self.assertEqual(self.calls, ["initialize"])

  def test_backends_up_already(self):
    from jax._src import xla_bridge

    updates = []
    with (
        env(JAX_PLATFORMS="proxy"),
        mock.patch.object(xla_bridge, "backends_are_initialized", lambda: True),
        mock.patch.object(jax.config, "update", lambda k, v: updates.append((k, v))),
    ):
      self.assertTrue(pathways.initialize())
    self.assertEqual(self.calls, ["monkey_patch_jax"])
    self.assertEqual(updates, [("jax_enable_compilation_cache", False)])

  def test_without_pathwaysutils_a_warning(self):
    with env(JAX_PLATFORMS="proxy"), modules(pathwaysutils=None):
      with testing.assert_warns_regex(self, RuntimeWarning, r"without pathwaysutils .* a profile records no"):
        self.assertTrue(pathways.initialize())


class ServerFlags(unittest.TestCase):
  """The flags a Pathways run gives its servers: the launchers' LIBTPU_INIT_ARGS, each flag once, as one argument
  value xpk can split on ' '."""

  def test_the_launchers_flags(self):
    """A set's flags and the MPMD flag, in the launchers' order and byte for byte their LIBTPU_INIT_ARGS but its
    trailing space (ds_ep_aot_test.composed, which the Launchers tests hold to the scripts themselves)."""
    from tests.unit.experimental.mpmd.jax import ds_ep_aot_test as aot

    for name in pathways.flag_sets():
      with self.subTest(name=name):
        self.assertEqual(pathways.read_flag_set(name), aot.flag_set(name))
        self.assertEqual(pathways.joined(pathways.server_flags(name)), aot.composed(aot.flag_set(name)).strip())
    flags = pathways.server_flags("tpu7x")
    self.assertEqual(flags, aot.TPU7X + [aot.MPMD])
    self.assertEqual(len(flags), 31)
    self.assertEqual(pathways.server_flags(None), list(pathways.MPMD_FLAGS))

  def test_each_flag_once_the_last_value_in_its_place(self):
    vmem = "--xla_tpu_scoped_vmem_limit_kib=32768"
    flags = pathways.server_flags("tpu7x", [vmem, "--xla_new=1 --xla_newer=2"])
    names = [f.split("=")[0] for f in flags]
    self.assertEqual(len(names), len(set(names)))
    self.assertEqual(flags[1], vmem)
    self.assertEqual(flags[-2:], ["--xla_new=1", "--xla_newer=2"])
    with self.assertRaisesRegex(ValueError, "'xla_x=1' is not a flag"):
      pathways.server_flags(None, ["xla_x=1"])
    with self.assertRaisesRegex(ValueError, r"no libtpu flag set 'tpu9': the sets are \['tpu7x', 'v5p'"):
      pathways.server_flags("tpu9")
    with self.assertRaisesRegex(ValueError, "no libtpu flag set"):
      pathways.read_flag_set("../libtpu_flags/tpu7x")

  def test_one_value_xpk_splits_on_single_spaces(self):
    """xpk splits each --custom-pathways-*-args value on ' ' (its pathways workload template): single spaces, no
    empty token, no flag holding whitespace; the same value for the proxy server, the resource manager and the
    workers."""
    value = pathways.joined(pathways.server_flags("tpu7x", ["  --xla_a=1   --xla_b=2 "]))
    self.assertEqual(value, value.strip())
    self.assertNotIn("  ", value)
    self.assertTrue(all(value.split(" ")))
    self.assertEqual(value.split(" "), value.split())
    args = pathways.server_args(value.split())
    self.assertEqual(list(args), list(pathways.SERVER_OPTIONS))
    self.assertEqual(set(args.values()), {value})
    for bad in ([""], ["--xla_a=1 --xla_b=2"], ["--xla_a=1\t"]):
      with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "empty or holding whitespace"):
        pathways.joined(bad)

  def test_command_line(self):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
      pathways.main(["flags", "v5p", "--extra=--xla_x=1"])
      pathways.main(["flags", "none", "--xpk"])
    line, xpk = out.getvalue().splitlines()
    self.assertEqual(line.split(" "), pathways.server_flags("v5p", ["--xla_x=1"]))
    self.assertEqual(xpk, " ".join(f"{o}={pathways.MPMD_FLAGS[0]}" for o in pathways.SERVER_OPTIONS))
    with self.assertRaisesRegex(SystemExit, "pathways flags: no libtpu flag set 'v4'"):
      pathways.main(["flags", "v4"])

  def test_client_flags_note(self):
    """Under Pathways the client's LIBTPU_INIT_ARGS configures no compiler: a note where it holds more than jax's own
    launch-barrier flag (jax's cloud TPU init adds that one wherever libtpu is installed)."""
    self.assertIsNone(pathways.client_flags_note(""))
    self.assertIsNone(pathways.client_flags_note(" --xla_tpu_use_enhanced_launch_barrier=true"))
    note = pathways.client_flags_note(pathways.joined(pathways.server_flags("tpu7x")))
    self.assertRegex(note, r"^note: LIBTPU_INIT_ARGS sets 31 flag\(s\) in this client: .* reach no compiler")


class Devices(unittest.TestCase):

  def test_hbm_table(self):
    """The TPU7x entry is the bytes_limit libtpu 0.0.50.1 reports on a tpu7x-8; v6e's and v5p's are ds_ep_aot's."""
    from maxtext.experimental.mpmd.jax import ds_ep_aot

    self.assertEqual(pathways.hbm_bytes("TPU7x"), 101_724_450_816)
    self.assertEqual(pathways.hbm_bytes("TPU v6 lite"), ds_ep_aot.TOPOLOGIES["v6e-16"][4])  # a v6e-4's bytes_limit
    self.assertEqual(pathways.hbm_bytes("TPU v5"), ds_ep_aot.TOPOLOGIES["v5p-8"][4])
    self.assertEqual(pathways.hbm_bytes("TPU v5p"), pathways.hbm_bytes("TPU v5"))
    self.assertIsNone(pathways.hbm_bytes("cpu"))

  def test_memory_stats(self):
    class Proxied:

      def memory_stats(self):
        raise RuntimeError("INVALID_ARGUMENT: MemoryStats is only supported for addressable PjRt devices.")

    class Cpu:

      def memory_stats(self):
        return None

    self.assertEqual(pathways.memory_stats(Cpu()), ({}, None))
    stats, why = pathways.memory_stats(Proxied())
    self.assertEqual(stats, {})
    self.assertEqual(why, "RuntimeError: INVALID_ARGUMENT: MemoryStats is only supported for addressable PjRt devices.")

  def test_peak_hbm_and_mem_probe_through_the_proxy(self):
    """ds_ep_pipeline.peak_hbm leaves out the devices that cannot report (all of them through the proxy: {}), and
    main's line says so; mem_probe prices against the device kind's HBM there, or --hbm-gib's."""
    from maxtext.experimental.mpmd.jax import mem_probe

    class Proxied:
      id, device_kind = 16, "TPU7x"

      def memory_stats(self):
        raise RuntimeError("INVALID_ARGUMENT: MemoryStats is only supported for addressable PjRt devices.")

    class Reports:
      id, device_kind = 3, "TPU7x"

      def memory_stats(self):
        return {"peak_bytes_in_use": 3 * 2**30, "bytes_limit": 5 * 2**30}

    with mock.patch.object(jax, "local_devices", lambda: [Proxied(), Proxied()]):
      self.assertEqual(dsp.peak_hbm(), {})
    with mock.patch.object(jax, "local_devices", lambda: [Reports(), Proxied()]):
      self.assertEqual(dsp.peak_hbm(), {3: 3.0})
    self.assertEqual(dsp.peak_hbm_line({3: 3.0}), "(GiB per device) {3: 3.0}")
    self.assertRegex(dsp.peak_hbm_line({}), r"^not reported: .* \(through the IFRT proxy, Pathways\)")
    self.assertEqual(mem_probe.device_hbm([Reports()]), (5 * 2**30, "bytes_limit"))
    hbm, src = mem_probe.device_hbm([Proxied()])
    self.assertEqual(hbm, 101_724_450_816)
    self.assertIn("pathways.HBM_BYTES", src)
    self.assertEqual(mem_probe.device_hbm([Proxied()], 2.0), (2 * 2**30, "--hbm-gib"))
    Proxied.device_kind = "TPU v9"
    with self.assertRaisesRegex(SystemExit, "give --hbm-gib"):
      mem_probe.device_hbm([Proxied()])


class Trace(unittest.TestCase):
  """A Pathways profile: gs:// only, pathwaysutils' start_trace with max_num_hosts and the options, a settle, the
  body, stop_trace whatever way the body ends."""

  def setUp(self):
    self.calls = []
    prof = types.ModuleType("pathwaysutils.profiling")
    prof.start_trace = lambda d, profiler_options=None, max_num_hosts=1: self.calls.append(
        ("start", d, profiler_options, max_num_hosts)
    )
    prof.stop_trace = lambda: self.calls.append(("stop",))
    pw = types.ModuleType("pathwaysutils")
    pw.profiling = prof
    self.enterContext(modules(pathwaysutils=pw, pathwaysutils__profiling=prof))
    self.enterContext(mock.patch.object(pathways.time, "sleep", lambda s: self.calls.append(("sleep", s))))

  def test_gs_only(self):
    for bad in ("/tmp/xprof", "xprof", "gcs://b/x", ""):
      with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "under Pathways a trace goes to a gs:// directory"):
        with pathways.trace(bad):
          pass
    self.assertEqual(self.calls, [])
    pathways.check_trace_dir("gs://bucket/traces/run")

  def test_start_settle_body_stop(self):
    opts = profiling.profile_options(profiling.EVERY_CHIP)
    with pathways.trace("gs://b/t", opts, max_num_hosts=3):
      self.calls.append(("body",))
    self.assertEqual(
        self.calls, [("start", "gs://b/t", opts, 3), ("sleep", pathways.TRACE_SETTLE_S), ("body",), ("stop",)]
    )
    self.calls.clear()
    with self.assertRaisesRegex(RuntimeError, "boom"), pathways.trace("gs://b/t"):
      raise RuntimeError("boom")
    self.assertEqual([c[0] for c in self.calls], ["start", "sleep", "stop"])

  def test_session_id(self):
    """A session's name as jax and pathwaysutils make one: XProf sorts the sessions of a logdir by it."""
    self.assertRegex(pathways.session_id(), r"^\d{4}_\d\d_\d\d_\d\d_\d\d_\d\d$")

  def test_sessions_are_named_in_local_time(self):
    """ds_ep_pipeline's multi-controller --profile names its session as session_id does, in this host's local time
    as jax and pathwaysutils name theirs (shared_profile_session: process 0 names it after the origin it picks, and
    every process takes that name): find_xplane and XProf order a logdir's sessions by name, and one host's then sort
    in time order off UTC too.  (It was named in UTC: in Los Angeles a --distributed capture sorted after a
    one-process capture up to 8 h later.)  Los Angeles time for the test; the distributed client a stub."""
    from jax._src import distributed  # pylint: disable=import-outside-toplevel

    kv = {}
    client = types.SimpleNamespace(
        wait_at_barrier=lambda name, timeout_ms: None,
        key_value_set=kv.__setitem__,
        blocking_key_value_get=lambda key, timeout_ms: kv[key],
    )
    origin = 1795730400 * 10**9  # 2026-11-26 22:00:00 UTC
    saved = os.environ.get("TZ")
    os.environ["TZ"] = "America/Los_Angeles"
    time.tzset()
    try:
      if time.localtime(0)[:6] == time.gmtime(0)[:6]:
        self.skipTest("no time zone database: TZ=America/Los_Angeles gives UTC")
      with (
          mock.patch.object(distributed.global_state, "client", client),
          mock.patch.object(dsp.time, "time_ns", lambda: origin + 10_000_000),
      ):
        for pid in (0, 1):  # process 0 names the session, process 1 takes its name
          with self.subTest(process=pid), mock.patch.object(jax, "process_index", return_value=pid):
            self.assertEqual(dsp.shared_profile_session(), ("2026_11_26_14_00_00", origin))
      self.assertEqual(pathways.session_id(origin / 1e9), "2026_11_26_14_00_00")
    finally:
      if saved is None:
        os.environ.pop("TZ", None)
      else:
        os.environ["TZ"] = saved
      time.tzset()

  def test_failures_are_trace_errors(self):
    """A trace that does not start (the body never runs, nothing to stop) or does not stop is a TraceError, which
    main records and goes on from; the body's own error stays its own, a failed stop a note on it."""
    prof = sys.modules["pathwaysutils.profiling"]

    def boom(*a, **kw):
      raise RuntimeError("no such bucket")

    with mock.patch.object(prof, "start_trace", boom):
      with self.assertRaisesRegex(pathways.TraceError, r"did not start \(RuntimeError: no such bucket\)"):
        with pathways.trace("gs://b/t"):
          self.calls.append(("body",))
    self.assertEqual(self.calls, [])
    with mock.patch.object(prof, "stop_trace", boom):
      with self.assertRaisesRegex(pathways.TraceError, r"did not stop \(RuntimeError: no such bucket\)"):
        with pathways.trace("gs://b/t"):
          self.calls.append(("body",))
      self.assertEqual([c[0] for c in self.calls], ["start", "sleep", "body"])
      with self.assertRaisesRegex(ValueError, "the body's") as cm, pathways.trace("gs://b/t"):
        raise ValueError("the body's")
    self.assertEqual(cm.exception.__notes__, ["and stopping the Pathways trace failed: RuntimeError: no such bucket"])
    with modules(pathwaysutils=None, pathwaysutils__profiling=None):
      with self.assertRaisesRegex(pathways.TraceError, "needs pathwaysutils"), pathways.trace("gs://b/t"):
        pass


class ProbeTraceDir(unittest.TestCase):
  """Before anything starts, main writes an object into the --profile directory and deletes it: a trace into a
  directory the client cannot write -- a bucket that does not exist -- never stops, and leaves the proxy refusing
  the next clients.  Local directories stand in for gs:// (xprof_align's I/O takes either)."""

  def test_written_and_deleted(self):
    from maxtext.experimental.mpmd.jax import xprof_align

    d = self.enterContext(tempfile.TemporaryDirectory())
    seen = []
    write = xprof_align.write_bytes
    with mock.patch.object(xprof_align, "write_bytes", lambda p, b: (seen.append(p), write(p, b))):
      pathways.probe_trace_dir(d + "/")
    self.assertEqual(len(seen), 1)
    self.assertEqual(os.path.dirname(seen[0]), d)
    self.assertRegex(os.path.basename(seen[0]), r"^\.mpmd_write_probe\.\d+\.\d+$")
    self.assertEqual(os.listdir(d), [])

  def test_unwritable_is_an_error(self):
    with self.assertRaisesRegex(ValueError, r"--profile /proc/x: this client cannot write there \(\w+Error: .*hang"):
      pathways.probe_trace_dir("/proc/x")

  def test_a_url_without_bucket_is_refused(self):
    """A gs:// URL that names no bucket -- gs://, gs:///traces/run (`gs://$BUCKET/...` with BUCKET unset) -- is a
    ValueError that says so, before any write (check_bucket); check_trace_dir refuses it too.  (The probe joined gs://
    into gs:/.mpmd_write_probe.*, a local name: it wrote a directory gs: in the working directory and passed, and
    gs:///x failed on google-cloud-storage's IndexError, 'string index out of range'.)"""
    from maxtext.experimental.mpmd.jax import xprof_align  # pylint: disable=import-outside-toplevel

    d = self.enterContext(tempfile.TemporaryDirectory())
    self.addCleanup(os.chdir, os.getcwd())
    os.chdir(d)
    written = []
    write = xprof_align.write_bytes

    def local_only(path, data):  # no gs:// write leaves the test
      written.append(path)
      if path.startswith("gs://"):
        raise RuntimeError("no storage in this test")
      write(path, data)

    with mock.patch.object(xprof_align, "write_bytes", local_only):
      for path in ("gs://", "gs:///", "gs:////", "gs:///x", "gs:///traces/run"):
        for check in (pathways.probe_trace_dir, pathways.check_trace_dir):
          with self.subTest(path=path, check=check.__name__):
            with self.assertRaises(ValueError) as cm:
              check(path)
            self.assertEqual(str(cm.exception), f"--profile {path}: no bucket name (gs://BUCKET/...)")
    self.assertEqual((written, os.listdir(d)), ([], []))
    with self.assertRaisesRegex(ValueError, r"^--json gs:///run\.json: no bucket name"):
      pathways.check_bucket("gs:///run.json", "--json")
    for path in ("gs://b", "gs://b/traces/run", "/tmp/x", ""):
      pathways.check_bucket(path)

  def test_storage_error_in_one_line(self):
    """A google.api_core error, whose str() is the response's whole JSON body, as its HTTP status and the service's
    message (storage_error), in the probe's error too; anything else as its type and text."""
    from maxtext.experimental.mpmd.jax import xprof_align

    body = {"error": {"code": 404, "message": "The specified bucket does not exist.", "errors": [{"reason": "notFound"}]}}

    class NotFound(Exception):  # google.api_core.exceptions.NotFound's attributes
      code = 404
      response = types.SimpleNamespace(json=lambda: body)

    e = NotFound(
        f"404 POST https://storage.googleapis.com/upload/storage/v1/b/b/o?uploadType=multipart: {json.dumps(body)}"
    )
    self.assertEqual(pathways.storage_error(e), "NotFound 404: The specified bucket does not exist.")
    self.assertEqual(pathways.storage_error(OSError(5, "I/O\nerror")), "OSError: [Errno 5] I/O error")

    def write(path, data):
      raise e

    with (
        mock.patch.object(xprof_align, "write_bytes", write),
        self.assertRaisesRegex(
            ValueError,
            r"^--profile gs://b/t: this client cannot write there \(NotFound 404: The specified bucket does not "
            r"exist\.\); a trace into it would hang when it stops",
        ),
    ):
      pathways.probe_trace_dir("gs://b/t")

  def test_unchecked_and_undeleted_warn(self):
    """Without google-cloud-storage nothing is checked, and a probe that cannot be deleted stays: warnings only."""
    from maxtext.experimental.mpmd.jax import xprof_align

    def no_storage(path, data):
      raise ImportError("No module named 'google.cloud.storage'")

    with mock.patch.object(xprof_align, "write_bytes", no_storage):
      with testing.assert_warns_regex(self, RuntimeWarning, r"not checked that this client can write there"):
        pathways.probe_trace_dir("gs://b/t")
    d = self.enterContext(tempfile.TemporaryDirectory())
    with mock.patch.object(xprof_align, "delete", lambda path: os.stat("/proc/x")):
      with testing.assert_warns_regex(self, RuntimeWarning, r"could not delete the write probe .*FileNotFoundError"):
        pathways.probe_trace_dir(d)
    self.assertEqual(len(os.listdir(d)), 1)


class Describe(unittest.TestCase):

  def test_describe_and_provenance_line(self):
    """Provenance under Pathways: the backend, the server's build in one line, the image if the launcher says it --
    and the client's libtpu flags marked as the plan's only."""
    with env(MPMD_PATHWAYS_IMAGE=None):
      self.assertEqual(
          pathways.describe(Client()), {"backend": "proxy/pathways", "server": "Pathways cl/42", "image": None}
      )
    with env(MPMD_PATHWAYS_IMAGE="20261124-jax_0.12.0.1"):
      pw = pathways.describe(Client())
    self.assertEqual(pw["image"], "20261124-jax_0.12.0.1")
    pv = {
        "maxtext": None,
        "mpmd": {"path": "/m", "sha": "abc", "branch": "b"},
        "versions": {"jax": "0.12.0.1"},
        "devices": "8 x TPU7x",
        "LIBTPU_INIT_ARGS": "--x=1",
    }
    self.assertTrue(dsp.provenance_line(pv).endswith("| 8 x TPU7x | LIBTPU_INIT_ARGS='--x=1'"))
    self.assertTrue(
        dsp.provenance_line(dict(pv, pathways=pw)).endswith(
            "| 8 x TPU7x | Pathways: proxy/pathways, server 'Pathways cl/42', image 20261124-jax_0.12.0.1 | client "
            "LIBTPU_INIT_ARGS='--x=1' (planning only)"
        )
    )


class MainUnderProxy(unittest.TestCase):
  """What main refuses or says before it builds anything, with and without Pathways (pathways.initialize stubbed:
  the real one would patch this process's profiler)."""

  def main(self, argv, proxy):
    with mock.patch.object(pathways, "initialize", lambda: proxy):
      return dsp.main(argv)

  def test_profile_goes_to_gs(self):
    with self.assertRaisesRegex(SystemExit, r"--profile /tmp/x: under Pathways a trace goes to a gs:// directory"):
      self.main(["--profile", "/tmp/x"], proxy=True)
    with env(MPMD_PROFILE_HOSTS="0"), self.assertRaisesRegex(SystemExit, r"MPMD_PROFILE_HOSTS='0': a number of hosts"):
      self.main(["--profile", "gs://b/x"], proxy=True)

  def test_profile_hosts_is_pathways_only(self):
    with self.assertRaisesRegex(SystemExit, r"--profile-hosts is Pathways' .* --profile-procs"):
      self.main(["--profile", "/tmp/x", "--profile-hosts", "2"], proxy=False)

  def test_no_distributed_under_proxy(self):
    with (
        env(JAX_PLATFORMS="proxy"),
        self.assertRaisesRegex(SystemExit, r"--distributed: JAX_PLATFORMS=proxy is Pathways"),
    ):
      dsp.init_distributed()

  def test_relay_and_xfer_refused_up_front(self):
    """The relay and xfer move arrays between processes, which a Pathways run has none of: refused before the
    backend starts, pointing to native and ppermute (Pipeline would say they need --distributed, refused too) -- with
    --distributed given as well, the transport's line first, and by mem_probe, which takes main's flags, in the same
    words (both ask pathways.check_transport, through multi_controller_only)."""
    from maxtext.experimental.mpmd.jax import mem_probe  # pylint: disable=import-outside-toplevel

    def probe(argv):
      with mock.patch.object(pathways, "initialize", lambda: True):
        return mem_probe.main(argv)

    for transport in ("relay", "xfer"):
      want = rf"^--transport {transport}: under Pathways .* native .* ppermute"
      for extra in ([], ["--distributed"]):
        with self.subTest(transport=transport, extra=extra), env(JAX_PLATFORMS="proxy"):
          said = set()
          for run in (lambda argv: self.main(argv, proxy=True), probe):
            with self.assertRaisesRegex(SystemExit, want) as cm:
              run(["--transport", transport, *extra])
            said.add(str(cm.exception))
          self.assertEqual(len(said), 1, said)

  def test_multi_controller_settings_refused_up_front(self):
    """Under Pathways what only multi-controller JAX takes is refused before the backend starts -- and before the
    write probe -- each saying what to use instead (the relay and xfer: test_relay_and_xfer_refused_up_front), several
    at once one a line, the transport's first (pathways.check_transport's message, before --distributed's, as main
    says it alone); off Pathways the helper names them (a one-host run takes them), the defaults none.
    --json-procs and --profile-procs keep their default, leaders, which changes nothing in one controller (given or
    not: the plan every process hashes and the run record name it as before); the multi-controller environment is not
    refused (it may be set cluster-wide; the import skips MPMD_CACHE_EVERY_PROCESS under the proxy)."""
    probed = []
    cases = (
        (["--distributed"], r"^--distributed: under Pathways .* one client process drives every device.*: run "),
        (["--json-procs", "all"], r"^--json-procs all: picks processes of a multi-controller run .*, --json$"),
        (["--profile-procs", "0"], r"^--profile-procs 0: .*: --profile-hosts N "),
        (["--norm-reduce", "allreduce"], r"^--norm-reduce allreduce: .* hops it to the others, a scalar copy each: "),
    )
    with mock.patch.object(pathways, "probe_trace_dir", probed.append):
      for argv, msg in cases:
        with self.subTest(argv=argv), self.assertRaisesRegex(SystemExit, msg):
          self.main(argv + ["--profile", "gs://b/x"], proxy=True)
      with self.assertRaises(SystemExit) as cm:
        self.main(["--distributed", "--transport", "relay", "--norm-reduce", "allreduce"], proxy=True)
    self.assertEqual(probed, [])
    said = [line.split(":")[0] for line in str(cm.exception).splitlines()]
    self.assertEqual(said, ["--transport relay", "--distributed", "--norm-reduce allreduce"])
    with self.assertRaises(ValueError) as transport:
      pathways.check_transport("relay")
    self.assertEqual(str(cm.exception).splitlines()[0], str(transport.exception))  # the message's one copy
    self.assertNotIn("Channel", str(cm.exception))  # under ppermute no hop is one: jax.device_put
    parse = dsp.make_parser().parse_args
    self.assertEqual((parse([]).json_procs, parse([]).profile_procs), ("leaders", "leaders"))
    self.assertEqual(dsp.multi_controller_only(parse([])), [])
    self.assertEqual(dsp.multi_controller_only(parse(["--json-procs", "leaders", "--profile-procs", "leaders"])), [])
    self.assertEqual(len(dsp.multi_controller_only(parse(["--json-procs", "1,2", "--profile-procs", "all"]))), 2)

  @testing.needs_devices(4)
  def test_multi_controller_environment_not_refused(self):
    """The multi-controller environment may be set cluster-wide: a Pathways dry run with it set goes on (the import
    skips MPMD_CACHE_EVERY_PROCESS under the proxy; the rest is read only under --distributed, the relay or xfer)."""
    keys = ("MPMD_CACHE_EVERY_PROCESS", "MPMD_COORD_PORT", "MPMD_RELAY_TIMEOUT", "MPMD_XFER_TRANSPORTS")
    with env(MPMD_FAKE_SLICES="2", **dict.fromkeys(keys, "1")):
      with mock.patch.object(pathways, "is_proxy_device", lambda d: True):
        rec = self.main(slices_args() + ["--dry-run", "--json-procs", "leaders"], proxy=True)
    self.assertIn("summary", rec)
    self.assertEqual((rec["config"]["json_procs"], rec["config"]["profile_procs"]), ("leaders", "leaders"))

  def test_mem_probe_refuses_them_too(self):
    """mem_probe (ds_ep_pipeline's flags) refuses the same under Pathways, before the backend starts, in main's words
    (both ask multi_controller_only)."""
    from maxtext.experimental.mpmd.jax import mem_probe  # pylint: disable=import-outside-toplevel

    for argv, msg in (
        (["--distributed"], r"^--distributed: under Pathways"),
        (["--transport", "xfer"], r"^--transport xfer: under Pathways .* native .* ppermute"),
        (["--distributed", "--norm-reduce", "allreduce"], r"^--distributed: .*\n--norm-reduce allreduce: "),
    ):
      with self.subTest(argv=argv):
        with mock.patch.object(pathways, "initialize", lambda: True), self.assertRaisesRegex(SystemExit, msg) as cm:
          mem_probe.main(argv)
        with self.assertRaises(SystemExit) as by_main:
          self.main(argv, proxy=True)
        self.assertEqual(str(cm.exception), str(by_main.exception))

  def test_profile_dir_probed_last(self):
    """The write probe runs after the checks that need no storage, and a directory it cannot write stops the run."""
    probed = []

    def probe(path):
      probed.append(path)
      raise ValueError(f"--profile {path}: this client cannot write there (NotFound: 404)")

    with mock.patch.object(pathways, "probe_trace_dir", probe):
      with env(MPMD_PROFILE_HOSTS="0"), self.assertRaisesRegex(SystemExit, "MPMD_PROFILE_HOSTS='0'"):
        self.main(["--profile", "gs://b/x"], proxy=True)
      self.assertEqual(probed, [])
      with self.assertRaisesRegex(SystemExit, r"^--profile gs://b/x: this client cannot write there \(NotFound"):
        self.main(["--profile", "gs://b/x"], proxy=True)
    self.assertEqual(probed, ["gs://b/x"])

  def test_profile_dir_probed_in_any_mode(self):
    """McJAX too: one process stops on a gs:// --profile this client cannot write (a trace written straight there
    hangs when it stops) -- under --distributed only once jax.distributed.initialize has run, since a process stopping
    before it would keep the others waiting, and only as the one process (a one-host slice).  Several processes go on
    with a warning: each traced one writes a local directory first, and an export error costs only the profile.  A
    local --profile is not probed.  init_distributed and build stubbed: the run ends where the pipeline would build."""
    probed, initialized = [], []

    def probe(path):
      probed.append(path)
      raise ValueError(f"--profile {path}: this client cannot write there (NotFound 404: no such bucket)")

    class Built(Exception):
      pass

    def build(a):
      raise Built

    stop = r"^--profile gs://b/x: this client cannot write there \(NotFound 404"
    out = io.StringIO()
    with (
        mock.patch.object(pathways, "probe_trace_dir", probe),
        mock.patch.object(dsp, "init_distributed", lambda: initialized.append(len(probed))),
        mock.patch.object(dsp, "build", build),
        contextlib.redirect_stdout(out),
    ):
      with self.assertRaisesRegex(SystemExit, stop):
        self.main(["--profile", "gs://b/x"], proxy=False)
      with self.assertRaisesRegex(SystemExit, stop):
        self.main(["--profile", "gs://b/x", "--distributed"], proxy=False)
      self.assertEqual(initialized, [2])  # probed first, stopped after
      self.assertEqual(out.getvalue(), "")
      with mock.patch.object(jax, "process_count", lambda: 2), self.assertRaises(Built):
        self.main(["--profile", "gs://b/x", "--distributed"], proxy=False)
      with self.assertRaises(Built):
        self.main(["--profile", "/tmp/x", "--distributed"], proxy=False)
    self.assertEqual(probed, ["gs://b/x"] * 3)
    self.assertRegex(
        out.getvalue(),
        r"^warning: --profile gs://b/x: this client cannot write there \(NotFound 404: no such bucket\) -- not across 2 "
        r"processes: a traced one writes its trace into a local directory first, and keeps it there when the export "
        r"fails\n$",
    )

  def test_a_url_without_bucket_stops_every_mode(self):
    """A gs:// --profile or --json that names no bucket (gs://, gs:///traces/run: `gs://$BUCKET/...` with BUCKET
    unset) stops the run before anything starts, with and without Pathways, a dry run too, and under --distributed
    before jax.distributed.initialize (every process parses the same flags): never the probe's warning across
    processes, after which each traced process exported its trace into a local directory gs:.  (The probe passed on
    gs://, and the trace went to the bucket 'plugins'.)  init_distributed and build stubbed: the run would end where
    the pipeline builds."""
    probed, initialized = [], []

    class Built(Exception):
      pass

    def build(a):
      raise Built

    with (
        mock.patch.object(pathways, "probe_trace_dir", probed.append),
        mock.patch.object(dsp, "init_distributed", lambda: initialized.append(True)),
        mock.patch.object(dsp, "build", build),
    ):
      for flag, path in (("--profile", "gs://"), ("--profile", "gs:///traces/run"), ("--json", "gs:///run.json")):
        for proxy, extra in ((False, []), (False, ["--distributed"]), (False, ["--dry-run"]), (True, [])):
          with self.subTest(flag=flag, path=path, proxy=proxy, extra=extra):
            with self.assertRaises(SystemExit) as cm:
              self.main([flag, path, *extra], proxy=proxy)
            self.assertEqual(str(cm.exception), f"{flag} {path}: no bucket name (gs://BUCKET/...)")
    self.assertEqual((probed, initialized), ([], []))

  def test_a_dry_run_probes_no_profile_dir(self):
    """A dry run traces nothing, so it probes no gs:// --profile, in any mode: it writes nothing into the bucket, and
    a client that cannot write there (a dev VM checking a job's command line, say) still gets the plan.
    init_distributed and build stubbed: the run ends where the pipeline would build."""
    probed = []

    def probe(path):
      probed.append(path)
      raise ValueError(f"--profile {path}: this client cannot write there (Forbidden 403: no access)")

    class Built(Exception):
      pass

    def build(a):
      raise Built

    with (
        mock.patch.object(pathways, "probe_trace_dir", probe),
        mock.patch.object(dsp, "init_distributed", lambda: None),
        mock.patch.object(dsp, "build", build),
    ):
      for proxy, extra in ((False, []), (False, ["--distributed"]), (True, [])):
        with self.subTest(proxy=proxy, extra=extra), self.assertRaises(Built):
          self.main(["--dry-run", "--profile", "gs://b/x", *extra], proxy=proxy)
    self.assertEqual(probed, [])

  @testing.needs_devices(4)
  def test_one_answer_after_the_build(self):
    """From the build on, main acts on the Pipeline's answer (its devices' client): JAX_PLATFORMS saying Pathways
    for devices that are not the proxy's -- or the other way round -- stops the run there."""
    args = slices_args() + ["--transport", "native", "--dry-run"]
    with env(MPMD_FAKE_SLICES="2"):
      with self.assertRaisesRegex(SystemExit, r"said Pathways, but the pipeline's devices are not the IFRT proxy's"):
        self.main(args, proxy=True)
      with mock.patch.object(pathways, "is_proxy_device", lambda d: True):
        with self.assertRaisesRegex(SystemExit, r"said not Pathways, but the pipeline's devices are the IFRT proxy's"):
          self.main(args, proxy=False)
      self.assertIn("summary", self.main(args, proxy=False))  # agreed: the dry run goes on

  @testing.needs_devices(4)
  def test_run_ahead_bounded_under_the_proxy(self):
    """Under Pathways --max-ahead defaults to PATHWAYS_MAX_AHEAD, 2 steps (nothing else bounds the client's run-ahead
    there: a local run queued ~90 steps, and a worker out of memory surfaced as a dropped connection); a value given
    stands; elsewhere no bound by default.  The record's config says what ran (dry runs: nothing compiled)."""
    args = slices_args() + ["--transport", "native", "--dry-run"]

    def ahead(extra, proxy):
      return self.main(args + extra, proxy=proxy)["config"]["max_ahead"]

    with env(MPMD_FAKE_SLICES="2"):
      with mock.patch.object(pathways, "is_proxy_device", lambda d: True):
        self.assertEqual((ahead([], True), ahead(["--max-ahead", "0"], True)), (dsp.PATHWAYS_MAX_AHEAD, 0))
      self.assertEqual((ahead([], False), ahead(["--max-ahead", "3"], False)), (None, 3))
    self.assertEqual(dsp.PATHWAYS_MAX_AHEAD, 2)

  @pytest.mark.scheduled_only
  @testing.needs_devices(4)
  def test_a_trace_that_fails_keeps_the_record(self):
    """A Pathways trace that does not start (or stop) loses the profile, not the run: main goes on, says so, and
    the --json record has profile_error and no profile_session.  CPU devices stand in for the proxy's
    (pathways.is_proxy_device patched), the trace and the write probe are stubs; TINY two-stage, one profiled step,
    in this process (tens of seconds)."""

    @contextlib.contextmanager
    def failing(target, popts, max_num_hosts=1):
      raise pathways.TraceError("the trace did not start (RuntimeError: no such bucket)")
      yield  # pylint: disable=unreachable

    d = self.enterContext(tempfile.TemporaryDirectory())
    args = slices_args() + ["--transport", "native", "--warmup", "1"]
    args += ["--profile", "gs://b/x", "--profile-steps", "1", "--json", f"{d}/r.json"]
    out = io.StringIO()
    with (
        env(MPMD_FAKE_SLICES="2"),
        mock.patch.object(pathways, "is_proxy_device", lambda d: True),
        mock.patch.object(pathways, "probe_trace_dir", lambda path: None),
        mock.patch.object(pathways, "trace", failing),
        contextlib.redirect_stdout(out),
    ):
      rec = self.main(args, proxy=True)
    with open(f"{d}/r.json") as f:
      written = json.load(f)
    for r in (rec, written):
      self.assertEqual(r["profile_error"], "the trace did not start (RuntimeError: no such bucket)")
      self.assertNotIn("profile_session", r)
      self.assertTrue(all(map(math.isfinite, r["warmup_losses"])))
    self.assertRegex(out.getvalue(), r"warning: no profile: --profile gs://b/x: the trace did not start")

  @pytest.mark.scheduled_only
  @testing.needs_devices(4)
  def test_the_record_is_written_before_the_profile(self):
    """--json writes the record before the profile as well as after it, so a run cut short while profiling keeps its
    warm-up and --bench numbers (a GKE job preempted during its profiled steps lost them all).  The trace stub reads
    the record as the profile starts and then stands in for the preemption: an exception main does not catch.  CPU
    devices stand in for the proxy's (pathways.is_proxy_device patched), the write probe is a stub; TINY two-stage, in
    this process (tens of seconds)."""

    class Preempted(BaseException):
      pass

    seen = []

    @contextlib.contextmanager
    def preempted(target, popts, max_num_hosts=1):
      with open(f"{d}/r.json") as f:
        seen.append(json.load(f))
      raise Preempted
      yield  # pylint: disable=unreachable

    d = self.enterContext(tempfile.TemporaryDirectory())
    args = slices_args() + ["--transport", "native", "--warmup", "2"]
    args += ["--bench", "--steps", "1", "--profile", "gs://b/x", "--json", f"{d}/r.json"]
    with (
        env(MPMD_FAKE_SLICES="2"),
        mock.patch.object(pathways, "is_proxy_device", lambda dev: True),
        mock.patch.object(pathways, "probe_trace_dir", lambda path: None),
        mock.patch.object(pathways, "trace", preempted),
        contextlib.redirect_stdout(io.StringIO()),
        self.assertRaises(Preempted),
    ):
      self.main(args, proxy=True)
    (rec,) = seen
    with open(f"{d}/r.json") as f:
      self.assertEqual(json.load(f), rec)  # nothing written after the preemption
    self.assertEqual(rec["config"]["profile"], "gs://b/x")
    self.assertEqual(len(rec["warmup_losses"]), 2)
    self.assertTrue(all(map(math.isfinite, rec["warmup_losses"] + rec["warmup_gnorms"])))
    self.assertGreater(rec["step_ms"], 0)  # --bench's
    self.assertEqual(len(rec["blocked_steps_ms"]), 3)
    self.assertFalse({"profile_session", "profile_error"} & set(rec))
    self.assertEqual(os.listdir(d), ["r.json"])  # renamed over: no temporary file beside it

  @pytest.mark.scheduled_only
  @testing.needs_devices(4)
  def test_a_named_pipe_gets_the_record_once(self):
    """A --json named pipe whose reader reads to its end and goes (cat, jq, json.load) gets one record under
    --profile, the one after the profile (with what the profile reports: profile_session), and main finishes.  Written
    before the profile as well, the first close gave the reader its end, and the second open waited for another reader
    without end: here a late reader, a minute on, lets it go.  CPU devices stand in for the proxy's
    (pathways.is_proxy_device patched), the trace and the write probe are stubs; TINY two-stage, one profiled step, in
    this process (~10 s)."""

    @contextlib.contextmanager
    def traced(*args, **kwargs):  # pathways.trace's stand-in: a trace that starts and stops
      del args, kwargs
      yield

    d = self.enterContext(tempfile.TemporaryDirectory())
    fifo = f"{d}/r.json"
    os.mkfifo(fifo)
    got, late, done = [], [], threading.Event()

    def read():  # one reader, to the pipe's end; a late one if main is still going a minute after it
      with open(fifo, encoding="utf-8") as f:
        got.append(f.read())
      if not done.wait(60):
        late.append(True)
        with open(fifo, encoding="utf-8") as f:
          f.read()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    args = slices_args() + ["--transport", "native", "--warmup", "1"]
    args += ["--profile", "gs://b/x", "--profile-steps", "1", "--json", fifo]
    try:
      with (
          env(MPMD_FAKE_SLICES="2"),
          mock.patch.object(pathways, "is_proxy_device", lambda dev: True),
          mock.patch.object(pathways, "probe_trace_dir", lambda path: None),
          mock.patch.object(pathways, "trace", traced),
          contextlib.redirect_stdout(io.StringIO()),
      ):
        rec = self.main(args, proxy=True)
    finally:
      done.set()
      reader.join(30)
      if reader.is_alive() and not got:  # main stopped before writing: a writer that writes nothing lets it go
        with contextlib.suppress(OSError):
          os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
    self.assertEqual(late, [], "the record after the profile waited for a second reader of the named pipe")
    self.assertEqual([json.loads(g) for g in got], [json.loads(json.dumps(rec, default=str))])
    self.assertTrue(rec["profile_session"].startswith("gs://b/x/plugins/profile/"), rec["profile_session"])


class JsonRecord(unittest.TestCase):
  """ds_ep_pipeline.write_json, --json's writer: a local record is replaced whole -- written beside it, then renamed
  over it -- so a write that fails or is cut short leaves the one written before as it was, a symlink keeps its link
  and a record its mode; one of this process's descriptors is written through, and a named pipe or a device in place
  (under --profile only the record after it); a gs:// one goes to xprof_align.write_bytes, whose errors it returns."""

  def setUp(self):
    super().setUp()
    self.d = self.enterContext(tempfile.TemporaryDirectory())
    self.path = os.path.join(self.d, "r.json")

  def read(self):
    with open(self.path) as f:
      return json.load(f)

  def test_replaced_whole_through_a_file_beside_it(self):
    renames = []
    replace = os.replace

    def rename(src, dst):
      renames.append((src, dst))
      replace(src, dst)

    with mock.patch.object(dsp.os, "replace", rename):
      self.assertIsNone(dsp.write_json(self.path, {"step_ms": 1.0}))
      self.assertIsNone(dsp.write_json(self.path, {"step_ms": 2.0, "devices": {0}}))
    self.assertEqual(self.read(), {"step_ms": 2.0, "devices": "{0}"})  # str() of what JSON has no type for
    self.assertEqual(len(renames), 2)
    for src, dst in renames:
      self.assertEqual((os.path.dirname(src), dst), (self.d, self.path))  # one directory: a rename, atomic
      self.assertNotEqual(src, dst)
    self.assertEqual(os.listdir(self.d), ["r.json"])

  def test_a_failed_write_leaves_the_record_before(self):
    """A write that fails halfway (a full disk: its OSError comes back) or is cut short (raised) leaves the record
    written before as it was -- open(path, "w") would have emptied it first -- and no temporary file beside it."""
    self.assertIsNone(dsp.write_json(self.path, {"step_ms": 1.0}))

    class Preempted(BaseException):
      pass

    def failing(exc):
      def opener(file, mode="r", **kwargs):
        f = open(file, mode, **kwargs)  # the builtin: the patch is ds_ep_pipeline's name only
        if "w" in mode:
          write = f.write

          def half(s):
            write(s[: len(s) // 2])
            f.flush()
            raise exc

          f.write = half
        return f

      return opener

    full = OSError(errno.ENOSPC, "No space left on device")
    with mock.patch.object(dsp, "open", failing(full), create=True):
      self.assertIs(dsp.write_json(self.path, {"step_ms": 2.0}), full)
    with mock.patch.object(dsp, "open", failing(Preempted()), create=True), self.assertRaises(Preempted):
      dsp.write_json(self.path, {"step_ms": 3.0})
    self.assertEqual(self.read(), {"step_ms": 1.0})
    self.assertEqual(os.listdir(self.d), ["r.json"])

  def test_gs_through_write_bytes(self):
    """gs:// goes to xprof_align.write_bytes (a GCS object is replaced whole), the bytes a local file gets; its
    errors (google.api_core's are no OSError) come back."""
    seen = []
    with mock.patch.object(dsp.xprof_align, "write_bytes", lambda path, data: seen.append((path, data))):
      self.assertIsNone(dsp.write_json("gs://b/r.json", {"step_ms": 1.0}))
    self.assertIsNone(dsp.write_json(self.path, {"step_ms": 1.0}))
    with open(self.path, "rb") as f:
      self.assertEqual(seen, [("gs://b/r.json", f.read())])

    class NotFound(Exception):
      pass

    def missing(path, data):
      raise NotFound("404 The specified bucket does not exist.")

    with mock.patch.object(dsp.xprof_align, "write_bytes", missing):
      self.assertIsInstance(dsp.write_json("gs://b/r.json", {}), NotFound)

  def test_a_symlinked_record_keeps_its_link(self):
    """The file a symlink names is the record replaced, beside itself: the link stays (a temp file renamed onto the
    link itself replaced it with a file of its own, and the file it named kept the old record)."""
    self.assertIsNone(dsp.write_json(self.path, {"step_ms": 1.5}))
    os.symlink("r.json", f"{self.d}/latest.json")
    self.assertIsNone(dsp.write_json(f"{self.d}/latest.json", {"step_ms": 2.5}))
    self.assertTrue(os.path.islink(f"{self.d}/latest.json"))
    self.assertEqual(self.read(), {"step_ms": 2.5})
    self.assertEqual(sorted(os.listdir(self.d)), ["latest.json", "r.json"])

  def test_a_record_keeps_its_mode(self):
    """An existing record's permission bits stay (the temp file had the umask's)."""
    self.assertIsNone(dsp.write_json(self.path, {"step_ms": 1.5}))
    os.chmod(self.path, 0o604)  # a mode no usual umask gives a new file
    self.assertIsNone(dsp.write_json(self.path, {"step_ms": 2.5}))
    self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o604)

  def test_a_descriptor_is_written_through(self):
    """One of this process's descriptors is written through, after what was written there before, and never
    resolved: with stdout or stderr a log, `> run.log` or `>> run.log`, the log keeps its file and every line, each
    record in its place.  By its name -- /dev/stdout, /dev/stderr, /dev/fd/N, /proc/self/fd/N, /proc/thread-self/fd/N,
    /proc/<pid>/fd/N -- and, stdout and stderr, by another name of the log: a link to /dev/stdout, a directory link to
    /dev, the log's own path.  (Resolved, each named run.log, a regular file, and the temp file renamed onto it left
    the log only the record, the descriptor writing on into the unlinked file; the second record, under --profile,
    went to a new file, 'run.log (deleted)'.)  A pipe so named takes the records in turn."""
    first, second = {"step_ms": 1.5}, {"step_ms": 2.5, "profile_session": "s"}
    want = f"log 1\n{json.dumps(first, indent=1)}\nlog 2\n{json.dumps(second, indent=1)}\nlog 3\n"

    def run(name, fd):  # main's log lines, its record before the profile and its record after it
      os.write(fd, b"log 1\n")
      self.assertIsNone(dsp.write_json(name, first))
      os.write(fd, b"log 2\n")
      self.assertIsNone(dsp.write_json(name, second))
      os.write(fd, b"log 3\n")

    names = (  # the record's name, and the log's descriptor: stdout (1), stderr (2) or one of its own (None)
        ("/dev/stdout", 1),
        ("/dev/stderr", 2),
        ("/dev/fd/{fd}", None),
        ("/proc/self/fd/{fd}", None),
        ("/proc/thread-self/fd/{fd}", None),
        (f"/proc/{os.getpid()}/fd/{{fd}}", None),
        ("{d}/out.json", 1),  # out.json -> /dev/stdout
        ("{d}/dev/stderr", 2),  # dev -> /dev
        ("{d}/run.log", 1),  # --json run.log > run.log
    )
    for name, k in names:
      for append in (False, True):
        with self.subTest(name=name, append=append), tempfile.TemporaryDirectory() as d:
          os.symlink("/dev/stdout", f"{d}/out.json")
          os.symlink("/dev", f"{d}/dev")
          log = f"{d}/run.log"
          with open(log, "w", encoding="utf-8") as f:
            f.write("earlier\n")
          fd = os.open(log, os.O_WRONLY | (os.O_APPEND if append else os.O_TRUNC))  # a shell's >> or >
          inode = os.fstat(fd).st_ino
          try:
            with redirected(k, fd) if k else contextlib.nullcontext():
              run(name.format(d=d, fd=fd), k or fd)
          finally:
            os.close(fd)
          with open(log, encoding="utf-8") as f:
            self.assertEqual(f.read(), ("earlier\n" if append else "") + want)
          self.assertEqual(os.stat(log).st_ino, inode)
          self.assertEqual(sorted(os.listdir(d)), ["dev", "out.json", "run.log"])
          self.assertTrue(os.path.islink(f"{d}/out.json") and os.path.islink(f"{d}/dev"))
    r, w = os.pipe()
    with os.fdopen(r, encoding="utf-8") as f:
      try:
        run(f"/dev/fd/{w}", w)
      finally:
        os.close(w)
      self.assertEqual(f.read(), want)

  def test_a_descriptor_without_python_streams(self):
    """A process started with stdout and stderr closed (`>&- 2>&-`) has sys.stdout and sys.stderr None: a record
    through a descriptor of its own still goes out, with nothing to flush before it."""
    r, w = os.pipe()
    with (
        os.fdopen(r, encoding="utf-8") as f,
        mock.patch.object(sys, "stdout", None),
        mock.patch.object(sys, "stderr", None),
    ):
      try:
        self.assertIsNone(dsp.write_json(f"/dev/fd/{w}", {"step_ms": 1.5}))
      finally:
        os.close(w)
      self.assertEqual(json.loads(f.read()), {"step_ms": 1.5})

  def test_no_regular_file_is_written_in_place(self):
    """A target that is no regular file -- a named pipe, a device -- is written in place: there is nothing to replace
    (a temp file renamed onto a named pipe replaced it, and the pipe's reader never got the record; one beside
    /dev/null could not be made)."""
    fifo = self.path
    os.mkfifo(fifo)
    r = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)  # a reader there already: the write's open does not wait
    try:
      self.assertIsNone(dsp.write_json(fifo, {"step_ms": 1.5}))
      self.assertEqual(json.loads(os.read(r, 1 << 16)), {"step_ms": 1.5})
    finally:
      os.close(r)
    self.assertTrue(stat.S_ISFIFO(os.stat(fifo).st_mode))
    self.assertEqual(os.listdir(self.d), ["r.json"])
    self.assertIsNone(dsp.write_json(os.devnull, {"step_ms": 1.5}))

  def test_a_named_pipe_takes_the_record_after_the_profile(self):
    """main's two records under --profile into a named pipe whose reader reads to its end and goes (cat, jq,
    json.load): the one before the profile is provisional, which a target written in place (a named pipe, a device)
    does not get, so the reader gets the one after it and the writes finish.  (Written both times, the first close
    gave the reader its end, and the second open waited for another reader without end.)  A regular file and a
    descriptor -- a pipe's too, which takes the records in turn -- get the provisional record as well."""
    first, second = {"step_ms": 1.5}, {"step_ms": 2.5, "profile_session": "s"}
    fifo = self.path
    os.mkfifo(fifo)
    got, wrote = [], []

    def read():  # one reader, to the pipe's end
      with open(fifo, encoding="utf-8") as f:
        got.append(f.read())

    def write():  # main's record before the profile, then its record after it
      wrote.append(dsp.write_json(fifo, first, provisional=True))
      wrote.append(dsp.write_json(fifo, second))

    reader = threading.Thread(target=read, daemon=True)
    writer = threading.Thread(target=write, daemon=True)
    reader.start()
    writer.start()
    try:
      writer.join(30)
      if writer.is_alive():  # a late reader lets the waiting write go
        with open(fifo, encoding="utf-8") as f:
          f.read()
        self.fail("the record after the profile waited for a second reader of the named pipe")
      reader.join(30)
    finally:
      if reader.is_alive():  # still waiting for a writer: one that writes nothing lets it go
        with contextlib.suppress(OSError):
          os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
    self.assertEqual(wrote, [None, None])
    self.assertEqual([json.loads(g) for g in got], [second])
    self.assertTrue(stat.S_ISFIFO(os.stat(fifo).st_mode))
    path = f"{self.d}/run.json"
    self.assertIsNone(dsp.write_json(path, first, provisional=True))
    with open(path, encoding="utf-8") as f:
      self.assertEqual(json.load(f), first)
    r, w = os.pipe()
    with os.fdopen(r, encoding="utf-8") as f:
      try:
        self.assertIsNone(dsp.write_json(f"/dev/fd/{w}", first, provisional=True))
      finally:
        os.close(w)
      self.assertEqual(json.loads(f.read()), first)


@testing.needs_devices(4)
class PathwaysHopsOnCpu(unittest.TestCase):
  """Under Pathways the gradient norm's hops are prebuilt Channels over the native transport (cheaper through the
  proxy than a device_put); under ppermute, and elsewhere in one process, device_puts.  Each is one batched copy, the
  value bit for bit: CPU devices standing in for the proxy's (pathways.is_proxy_device patched)."""

  def build(self, proxy):
    args = slices_args() + ["--transport", "native"]
    with mock.patch.object(pathways, "is_proxy_device", lambda d: proxy), env(MPMD_FAKE_SLICES="2"):
      return dsp.build(dsp.make_parser().parse_args(args))

  def test_hops(self):
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from maxtext.experimental.mpmd.jax import transfer

    plain, pw = self.build(False), self.build(True)
    self.assertEqual(plain.sq_channels, {})
    self.assertEqual(sorted(pw.sq_channels), [(0, 1), (1, 0)])
    self.assertTrue(all(type(c) is transfer.Channel and c.n_slow == 0 for c in pw.sq_channels.values()))
    self.assertEqual(sorted(pw.channels), sorted(plain.channels))
    x = jax.device_put(jnp.float32(1.25), NamedSharding(pw.meshes[0], P()))
    with transfer.no_host_bounce():
      a, b = plain._hop(0, 1, x), pw._hop(0, 1, x)
    for y in (a, b):
      self.assertEqual(y.sharding, NamedSharding(pw.meshes[1], P()))
      self.assertEqual(float(y), 1.25)

  def test_ppermute_hops_are_device_puts(self):
    """Under ppermute no hop is a prebuilt channel, Pathways or not (only the boundary copies are PermuteChannels):
    each is a jax.device_put -- what --norm-reduce allreduce's refusal promises, a scalar copy a hop.  Ahead of time
    for one Pathways client (ds_ep_aot --pathways: build's proxy) no hop is a program to compile either; for
    multi-controller JAX (ds_ep_aot's default target) each is a PermuteChannel."""
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    from jax.sharding import NamedSharding, PartitionSpec as P  # pylint: disable=import-outside-toplevel
    from maxtext.experimental.mpmd.jax import transfer  # pylint: disable=import-outside-toplevel

    args = slices_args() + ["--transport", "ppermute"]
    with mock.patch.object(pathways, "is_proxy_device", lambda d: True), env(MPMD_FAKE_SLICES="2"):
      p = dsp.build(dsp.make_parser().parse_args(args))
    self.assertEqual(p.sq_channels, {})
    with env(MPMD_FAKE_SLICES="2"):
      aot = {pw: dsp.build(dsp.make_parser().parse_args(args), aot=True, proxy=pw) for pw in (True, None)}
    self.assertEqual(aot[True].sq_channels, {})
    self.assertEqual(sorted(aot[None].sq_channels), [(0, 1), (1, 0)])
    self.assertTrue(all(isinstance(c, transfer.PermuteChannel) for c in aot[None].sq_channels.values()))
    x = jax.device_put(jnp.float32(1.25), NamedSharding(p.meshes[0], P()))
    with mock.patch.object(jax, "device_put", wraps=jax.device_put) as put:
      y = p._hop(0, 1, x)  # pylint: disable=protected-access
    put.assert_called_once()
    self.assertEqual((y.sharding, float(y)), (NamedSharding(p.meshes[1], P()), 1.25))


# Every function, method and class of the package that exists only for multi-controller JAX -- several processes,
# one per host (pathways.multi_controller) -- and jax's own entry points to it (jax.distributed's, this process's id):
# in one controller (a Pathways client, one host's process) none of them may run (`OneController`).  (owner,
# attribute): a property or class too.
MULTI_ONLY = (
    (dsp, "init_distributed"),
    (dsp, "plan_fingerprint"),
    (dsp, "host_barrier"),
    (dsp, "check_plan_agreement"),
    (dsp, "check_program_agreement"),
    (dsp, "_ranges"),
    (dsp, "missing_input_anywhere"),
    (dsp, "shared_profile_session"),
    (dsp, "proc_set"),
    (dsp, "shared_stage_hosts"),
    (dsp, "cross_slice_warning"),
    (dsp, "native_transport_aborts"),
    (dsp, "cache_every_process"),
    (dsp, "_RelayBox"),
    (dsp.Pipeline, "_do_remote"),
    (dsp.Pipeline, "_global_sumsq"),
    (dsp.Pipeline, "relay_ctx"),
    (dsp.transfer, "process_indices"),
    (dsp.transfer, "is_cross_process"),
    (dsp.transfer, "_global_index_values"),
    (dsp.transfer, "CrossProcessChannel"),
    (dsp.transfer, "_host_ip"),
    (dsp.transfer, "XferContext"),
    (dsp.transfer, "xfer_context"),
    (dsp.transfer, "_same_mesh_boundary"),
    (dsp.transfer, "_check_sources"),
    (dsp.transfer, "_host_routes"),
    (dsp.transfer, "xfer_peers"),
    (dsp.transfer, "XferChannel"),
    (dsp.transfer, "relay_routes"),
    (dsp.transfer, "host_net_settings"),
    (dsp.transfer, "RelayFuture"),
    (dsp.transfer, "resolve"),
    (dsp.transfer, "RelayContext"),
    (dsp.transfer, "relay_timing_line"),
    (dsp.transfer, "RelayChannel"),
    (dsp.xprof_align, "export_into_session"),
    (dsp.xprof_align, "local_trace_dir"),
    (dsp.profiling, "trace_label"),
    (jax.distributed, "initialize"),
    (jax, "process_index"),
)


@testing.needs_devices(4)
class OneController(unittest.TestCase):
  """In one controller -- a Pathways client, one host's own process; here this CPU process -- none of the
  multi-controller logic runs, not even as a walk that finds nothing to do: each of MULTI_ONLY is patched to record a
  call (and raise), jax.distributed's key-value client to record any use, a device's process_index to record a read
  from the package's code (jax's own reads are not recorded, nor the one predicate's: pathways.multi_controller looks
  at a device set that is not the proxy's), and once dsp.build has made the pipeline any look at its `local`,
  `placed` or `stage_procs` is recorded too.  main (and mem_probe) then run TINY pipelines end to end in this process --
  precompiled, --bench, --json, a local --profile, --mem-report, --grad-digest -- and nothing may have been recorded.
  Every PR: 1F1B (two stages), zbh2 with the last chunk split (four) and its I's paced as on one host's own TPU
  client, 1F1B over ppermute with MTP from --init-from chunk files, mem_probe; scheduled_only: breadth and
  interleaved (V=2), zbh1, zbh2, and 1F1B with the proxy's devices standing in (pathways.is_proxy_device patched,
  the trace and the write probe stubs).  Stages of FSDP n/4 x EP 2 (two) or FSDP 1 x EP n/4 (four) on n devices, 4
  or 8."""

  @contextlib.contextmanager
  def armed(self, ran, proxy=False):
    """MULTI_ONLY, the key-value client and the package's reads of a device's process_index recording into `ran`;
    dsp.build starting the watch on `local`, `placed` and `stage_procs` of the pipelines it makes; pathways.initialize
    saying `proxy`."""
    jax.devices()  # the backend up before the key-value client is patched (a backend starting would take it)
    from jax._src import distributed  # pylint: disable=import-outside-toplevel

    def raiser(what):
      def call(*args, **kwargs):
        ran.append(what)
        raise AssertionError(f"{what} ran in one controller")

      return call

    class Store:
      """jax.distributed's coordination client: any use recorded."""

      def __getattr__(self, name):
        ran.append(f"key-value store .{name}")
        raise AssertionError(f"the key-value store's {name} used in one controller")

    # jax reads a device's process_index everywhere (which devices are this process's); the package's code may not,
    # but in the predicate (and in a comprehension of it, a frame of its own before Python 3.12)
    device = type(jax.devices()[0])
    real_index = inspect.getattr_static(device, "process_index")
    pkg = os.path.dirname(dsp.__file__) + os.sep  # as the package's code objects name their files
    rule = pathways.multi_controller.__code__

    def process_index(d):
      code = inspect.currentframe().f_back.f_code
      if code.co_filename.startswith(pkg) and code is not rule and code not in rule.co_consts:
        ran.append(f"device.process_index read in {os.path.basename(code.co_filename)} {code.co_name}")
      return real_index.fget(d)

    watching = []

    def watched(name):
      def get(p):
        if watching:
          ran.append(f"Pipeline.{name} read after dsp.build")
        return p.__dict__[name]

      def put(p, v):
        p.__dict__[name] = v

      return property(get, put)

    real_build = dsp.build
    Watched = type("Watched", (dsp.Pipeline,), {name: watched(name) for name in ("local", "placed", "stage_procs")})

    def build(*args, **kwargs):
      p = real_build(*args, **kwargs)
      p.__class__ = Watched
      watching.append(True)  # what main and mem_probe do with it from here on
      return p

    with contextlib.ExitStack() as stack:
      for owner, name in MULTI_ONLY:
        stub = raiser(f"{getattr(owner, '__name__', owner).rsplit('.', 1)[-1]}.{name}")
        if isinstance(inspect.getattr_static(owner, name), property):
          stub = property(stub)
        stack.enter_context(mock.patch.object(owner, name, stub))
      stack.enter_context(mock.patch.object(device, "process_index", property(process_index)))
      stack.enter_context(mock.patch.object(distributed.global_state, "client", Store()))
      stack.enter_context(mock.patch.object(dsp, "build", build))
      stack.enter_context(mock.patch.object(pathways, "initialize", lambda: proxy))
      if proxy:
        stack.enter_context(mock.patch.object(pathways, "is_proxy_device", lambda dev: True))
        stack.enter_context(mock.patch.object(pathways, "probe_trace_dir", lambda path: None))
        stack.enter_context(mock.patch.object(pathways, "trace", lambda *a, **k: contextlib.nullcontext()))
      yield

  def shape(self, stages, vv):
    n = len(jax.devices())
    fsdp, ep = (n // 4, 2) if stages == 2 else (1, n // 4)
    per = fsdp * ep
    return f"--stages {stages} --vv {vv} --fsdp {fsdp} --ep {ep} --gbs {4 * per} --mb 4 --layers {stages * vv}".split()

  def tiny(self):
    """TINY, precompiled, two warm-up steps."""
    return [x for x in multihost_test.TINY if x not in ("--no-precompile", "--warmup", "3")] + ["--warmup", "2"]

  def check(self, extra, proxy=False, init_from=False):
    """main on TINY + `extra` (with --json and --profile; `init_from`: the weights from chunk files of that pipeline,
    written first), armed: it runs to the end and nothing was recorded."""
    with tempfile.TemporaryDirectory() as d:
      args = self.tiny() + extra
      if init_from:  # unarmed: the files a run of another setup wrote
        p = dsp.build(dsp.make_parser().parse_args(args))
        dsp.write_chunk_files(f"{d}/chunks", p.chunk_trees(p.init(jax.random.key(7))[0]))
        args += ["--init-from", f"{d}/chunks"]
      args += ["--bench", "--steps", "2", "--mem-report", "--grad-digest", "1", "--json", f"{d}/r.json"]
      args += ["--profile", "gs://b/x" if proxy else f"{d}/prof", "--profile-steps", "1"]
      ran = []
      with self.armed(ran, proxy), contextlib.redirect_stdout(io.StringIO()):
        rec = dsp.main(args)
      self.assertEqual(ran, [])
      with open(f"{d}/r.json", encoding="utf-8") as f:
        self.assertEqual(json.load(f)["warmup_losses"], rec["warmup_losses"])
      if not proxy:
        self.assertTrue(glob.glob(f"{d}/prof/plugins/profile/*/*.xplane.pb"), os.listdir(d))
    self.assertTrue(all(map(math.isfinite, rec["warmup_losses"] + rec["warmup_gnorms"])), rec)
    self.assertGreater(rec["step_ms"], 0)
    self.assertNotIn("processes", rec)  # no multi-controller record fields
    return rec

  def test_the_harness_records(self):
    """The harness itself: a registered function that runs is recorded, so are jax.process_index, a read of a
    device's process_index from the package's code (not from the predicate) and a look at `local` once dsp.build has
    made the pipeline."""
    ran = []
    with self.armed(ran):
      for call in (lambda: dsp.cross_slice_warning([], "native"), lambda: dsp.transfer.resolve(None), jax.process_index):
        with self.assertRaises(AssertionError):
          call()
      devs = jax.devices()[:2]
      self.assertFalse(pathways.multi_controller(devs))  # the predicate may look
      in_package = eval(compile("lambda d: d.process_index", dsp.__file__, "eval"))  # pylint: disable=eval-used
      self.assertEqual(in_package(devs[0]), 0)
      self.assertEqual(devs[1].process_index, 0)  # this test's own read: not the package's
      p = dsp.build(dsp.make_parser().parse_args(self.tiny() + self.shape(2, 1)))
      self.assertTrue(p.local)
    self.assertEqual(
        ran,
        [
            "ds_ep_pipeline.cross_slice_warning",
            "transfer.resolve",
            "jax.process_index",
            "device.process_index read in ds_ep_pipeline.py <lambda>",
            "Pipeline.local read after dsp.build",
        ],
    )

  def test_1f1b(self):
    self.check(self.shape(2, 1) + ["--schedule", "1f1b"])

  def test_zbh2_split_last(self):
    """zbh2 with the last chunk split, its I's (and AI's) paced as on one host's own TPU client (Pipeline.pace_i,
    forced: CPU devices allocate no output at issue): the waits look at the step's own W buffers alone."""
    waits, pace = [], dsp.Pipeline._pace  # pylint: disable=protected-access

    def paced(p, st, op):
      waits.append(st.last_i.get(op.c) is not None)
      return pace(p, st, op)

    with mock.patch.object(dsp, "allocates_at_issue", return_value=True), mock.patch.object(dsp.Pipeline, "_pace", paced):
      rec = self.check(self.shape(4, 1) + ["--schedule", "zbh2", "--zb-split-last", "on"])
    self.assertTrue(rec["zb"]["paced"])
    self.assertTrue(any(waits), waits)

  def test_ppermute_mtp_init_from(self):
    """Collective-permute boundaries (PermuteChannel: both sides this process's), MTP's tied embedding over them, and
    the weights from --init-from chunk files: every stage's read, none skipped."""
    rec = self.check(self.shape(2, 1) + ["--schedule", "1f1b", "--transport", "ppermute", "--mtp", "1"], init_from=True)
    self.assertEqual(rec["config"]["transport"], "ppermute")

  def test_mem_probe(self):
    """mem_probe prices every stage in one controller (ds_ep_aot.report over every stage, `local` not looked at),
    its lines untagged."""
    from maxtext.experimental.mpmd.jax import mem_probe  # pylint: disable=import-outside-toplevel

    ran = []
    with self.armed(ran), contextlib.redirect_stdout(io.StringIO()) as out:
      rows = mem_probe.main(self.tiny() + self.shape(2, 1) + ["--schedule", "1f1b"])
    self.assertEqual(ran, [])
    self.assertEqual([r["stage"] for r in rows], [0, 1])
    self.assertNotIn("[p", out.getvalue())

  @pytest.mark.scheduled_only
  def test_breadth_and_interleaved(self):
    for schedule in ("breadth", "interleaved"):
      with self.subTest(schedule=schedule):
        self.check(self.shape(2, 2) + ["--schedule", schedule])

  @pytest.mark.scheduled_only
  def test_zbh1_and_zbh2(self):
    """zbh1 with the last chunk split too (--zb-split-last on, unpaced; test_zbh2_split_last paces it) and zbh2 with
    it fused (the default)."""
    for schedule, extra in (("zbh1", ["--zb-split-last", "on"]), ("zbh2", [])):
      with self.subTest(schedule=schedule):
        self.check(self.shape(4, 1) + ["--schedule", schedule] + extra)

  @pytest.mark.scheduled_only
  def test_under_the_proxy(self):
    """The proxy's devices standing in: Pathways' own lines run (the norm hops as prebuilt Channels, the per-stage
    collectives' log, one read of the warm-up's gradient norm and loss), none of the multi-controller ones."""
    with env(MPMD_FAKE_SLICES="2"):
      self.check(self.shape(2, 1) + ["--schedule", "1f1b", "--layout", "slices"], proxy=True)


# A child process's program: ds_ep_pipeline.main (its flags the program's) as a Pathways run takes it, on the CPU
# devices of the child's environment -- the mode's two answers, pathways.initialize's (the process's) and
# pathways.is_proxy_device's (its devices'), patched to say Pathways, so that main and the Pipeline take every branch
# they take only there.
AS_PATHWAYS = """\
import sys
from unittest import mock
from maxtext.experimental.mpmd.jax import ds_ep_pipeline, pathways
with mock.patch.object(pathways, "initialize", lambda: True):
  with mock.patch.object(pathways, "is_proxy_device", lambda d: True):
    ds_ep_pipeline.main(sys.argv[1:])
"""


@testing.needs_devices(4)
class OneProcessSlices(unittest.TestCase):
  """The path a Pathways run takes, on CPU devices: ONE process, --layout slices over two fake slices
  (MPMD_FAKE_SLICES=2) of FSDP x EP stages (testing.stage_axes: 2 x 2 on 8 devices, 1 x 2 on 4), every boundary a
  strict transfer.Channel (--transport native) -- the single-process reference of multihost_test.py's two-slice runs
  (`_slices_args`) -- and main's and the Pipeline's Pathways-only branches taken (AS_PATHWAYS): the warm-up's one
  device_get of the gradient norm and the loss, the norm's hops through prebuilt Channels, the stages' collective
  lines -- against the same pipeline on the plain one-process path, run beside it: --layout physical, no fake slices,
  collective-permute boundaries (--transport ppermute), no Pathways branch.  Two runs, each precompiled in a child
  process of as many CPU devices as this one's stages take (with XLA:CPU's YNN fusions off: multihost_test._env),
  ~25 s.  The stages hold the same devices either way and a boundary copy or a hop moves bits, so every warmup loss
  and gradient norm is the reference's bit for bit."""

  def pipeline(self, args, devs, env, as_pathways=False):
    """One run of ds_ep_pipeline (`args`, `env` added) in a child process on `devs` CPU devices -> (its record,
    output); `as_pathways`: AS_PATHWAYS runs it, in multihost_test.run_single's environment."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    args = args + ["--json", f"{d}/r.json"]
    if as_pathways:
      child_env = multihost_test._env(devs, **env)  # pylint: disable=protected-access
      r = subprocess.run(
          [sys.executable, "-c", AS_PATHWAYS, *args],
          cwd=multihost_test.ROOT,
          env=child_env,
          capture_output=True,
          text=True,
          timeout=600,
          check=False,
      )
      rc, out = r.returncode, r.stdout + r.stderr
    else:
      rc, out = multihost_test.run_single(args, devs, env=env)
    self.assertEqual(rc, 0, out[-4000:])
    with open(f"{d}/r.json") as f:
      return json.load(f), out

  def test_one_process_two_fake_slices_native(self):
    a = testing.stage_axes()
    args, devs, per = slices_args(a), 2 * a["fsdp"] * a["ep"], a["fsdp"] * a["ep"]
    r, out = self.pipeline(args + ["--transport", "native"], devs, {"MPMD_FAKE_SLICES": "2"}, as_pathways=True)
    self.assertIn("pathways", r["provenance"])  # the Pipeline's devices answered Pathways
    self.assertRegex(out, r"stage 1: EP collectives \S+, FSDP collectives \S+")  # main's Pathways-only lines
    self.assertNotIn("processes", r)  # one process: no multi-controller record fields
    for stage, ids in enumerate((range(per), range(per, devs))):  # stage s: slice s, its devices in id order
      held = "".join(rf"{i}@\S+ " for i in ids)
      self.assertRegex(r["summary"], rf"stage {stage}: \{{'fsdp': {a['fsdp']}, 'expert': {a['ep']}\}} {held}")
    self.assertRegex(out, r"precompiled \d+ programs")
    losses, gnorms = r["warmup_losses"], r["warmup_gnorms"]
    self.assertTrue(all(map(math.isfinite, losses + gnorms)), (losses, gnorms))
    self.assertLess(losses[-1], losses[0])
    plain = [x if x != "slices" else "physical" for x in args]
    self.assertIn("physical", plain)
    ref, _ = self.pipeline(plain + ["--transport", "ppermute"], devs, {})
    self.assertEqual((losses, gnorms), (ref["warmup_losses"], ref["warmup_gnorms"]))


def one_slice_problem(devs):
  """Why OnTpu cannot run on `devs`, a job's TPU devices -> a message, or None where they are one slice: its tests cut
  the devices into two fake slices (MPMD_FAKE_SLICES=2), which ds_ep_pipeline.slice_groups refuses on the devices of
  several slices ("these devices are on slices [0, 1] of their own; unset it") -- an error where the job, not the
  code, does not fit, as on a GKE job of two slices."""
  slices = sorted({k for k in map(dsp.slice_index, devs) if k is not None})
  if len(slices) > 1:
    return f"runs on one slice, cut into two (MPMD_FAKE_SLICES=2); these {len(devs)} devices are on slices {slices}"
  return None


class OnTpuTopology(unittest.TestCase):
  """OnTpu cuts one slice into its two fake slices (MPMD_FAKE_SLICES=2), so on a job of several slices, which
  MPMD_FAKE_SLICES refuses, it skips, saying why (one_slice_problem); one slice of 4n devices, Pathways' or a TPU
  VM's, runs.  Devices faked: no backend, milliseconds."""

  @staticmethod
  def devices(n, slices=None):
    """`n` fake TPU devices of `slices` slices (None: devices without a slice_index, as CPU's are)."""
    return [types.SimpleNamespace(id=i, slice_index=None if slices is None else i * slices // n) for i in range(n)]

  def test_one_slice_runs(self):
    for n, slices in ((4, 1), (8, 1), (8, None), (16, 1), (128, 1)):
      with self.subTest(n=n, slices=slices):
        self.assertIsNone(one_slice_problem(self.devices(n, slices)))

  def test_several_slices_skip(self):
    for n in (8, 256):
      want = rf"^runs on one slice, cut into two \(MPMD_FAKE_SLICES=2\); these {n} devices are on slices \[0, 1\]$"
      with self.subTest(n=n):
        self.assertRegex(one_slice_problem(self.devices(n, 2)), want)
        with (
            mock.patch.object(testing, "tpu_devices", lambda n=n: self.devices(n, 2)),
            self.assertRaisesRegex(unittest.SkipTest, want),
        ):
          OnTpu.setUpClass()


@pytest.mark.tpu_only
class OnTpu(unittest.TestCase):
  """The TINY two-stage pipeline in THIS process on TPUs with 4 devices or more -- Pathways' through the IFRT proxy
  (one client may hold a proxy: never a child) or a TPU VM's own (McJAX) -- on the path a Pathways run takes:
  --layout slices over two fake slices of the devices (MPMD_FAKE_SLICES=2: a TPU VM's devices are one slice, and
  Pathways' all report slice 0), the EP collectives pinned (Pathways' offload probe answers where McJAX's libtpu does
  not, which would change the plan).  N devices make two stages of FSDP N/4 x EP 2: 1 x 2 on MaxText CI's v6e-4, 2 x 2
  on a tpu7x-8.  On a job of several slices, which MPMD_FAKE_SLICES refuses, it skips, saying why
  (one_slice_problem, OnTpuTopology).

  - main end to end with --bench and --json: the run record written, the step timed, the losses finite and falling;
    under Pathways the mode in the record's provenance, and no peak HBM (the proxy keeps no memory statistics); with
    MAXTEXT_MPMD_PATHWAYS_TRACE_DIR=gs://... there, a profiled step whose session holds a worker's device planes.
  - The boundaries' device-to-device copies (--transport native; one IFRT CopyArrays each under Pathways) against
    collective-permute programs (--transport ppermute): the same programs on the stages, the same bits through either
    copy -- every loss and gradient norm bit for bit, on whatever compiler the backend runs (a Pathways server
    compiles with its image's XLA and libtpu, not the client's).
  - The zero-bubble schedules against 1F1B by the TPU contract (`assert_near_1f1b`), zbh2 against zbh1 bit for bit,
    on four stages (8 devices or more), the last chunk fused (the default) and split (--zb-split-last on); on two,
    where it is the one split (scheduled_only); at two stages without it, where nothing splits, bit for bit against
    1F1B."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    devs = testing.tpu_devices()
    testing.require_tpu(len(devs) >= 4 and len(devs) % 4 == 0, "TPU devices, 4 or a multiple of 4")
    if (why := one_slice_problem(devs)) is not None:  # TPUs, but a job its fake slices do not fit
      raise unittest.SkipTest(why)
    cls.n, cls.on_pathways = len(devs), pathways.is_proxy_device(devs[0])

  def test_tiny_pipeline(self):
    d = self.enterContext(tempfile.TemporaryDirectory())
    self.enterContext(env(MPMD_FAKE_SLICES="2"))
    # TINY, precompiled (Pathways' default: precompiled programs dispatch faster through the proxy)
    args = slices_args({"fsdp": self.n // 4, "ep": 2}) + ["--ep-collectives", "xla"]
    trace = os.environ.get("MAXTEXT_MPMD_PATHWAYS_TRACE_DIR") if self.on_pathways else None
    extra = ["--profile", f'{trace.rstrip("/")}/pathways_test', "--profile-steps", "1"] if trace else []
    native = dsp.main(args + ["--transport", "native", "--bench", "--steps", "2", "--json", f"{d}/r.json", *extra])
    with open(f"{d}/r.json") as f:
      rec = json.load(f)
    self.assertEqual(rec["warmup_losses"], native["warmup_losses"])
    self.assertGreater(rec["step_ms"], 0)
    losses, gnorms = native["warmup_losses"], native["warmup_gnorms"]
    self.assertTrue(all(map(math.isfinite, losses + gnorms)), (losses, gnorms))
    self.assertLess(losses[-1], losses[0])
    if self.on_pathways:
      self.assertEqual(rec["provenance"]["pathways"]["backend"], "proxy/pathways")
      self.assertIsNone(rec["peak_hbm_gib"])
    else:
      self.assertNotIn("pathways", rec["provenance"])
    if trace:
      self.check_trace(native["profile_session"])
    permuted = dsp.main(args + ["--transport", "ppermute"])
    self.assertEqual((permuted["warmup_losses"], permuted["warmup_gnorms"]), (losses, gnorms))

  def test_zero_bubble(self):
    """zbh1 and zbh2 -- the two middle chunks' backwards split into I and W, as the command line runs them by
    default, and with the fused last chunk split into AI and W too (--zb-split-last on) -- against 1F1B on four stages
    of FSDP N/8 x EP 2, two a fake slice (--layout slices at two stages a slice, the GKE runs' layout; a chip a stage
    on a tpu7x-8), precompiled, on whatever compiler the backend runs: zbh1's warmup losses and gradient norms 1F1B's
    by the TPU contract (`assert_near_1f1b`: XLA:TPU fuses I and W unlike the backward they are cut from; the first
    loss bit for bit only where the last chunk keeps its fused program), zbh2's zbh1's bit for bit.  The split needs a
    middle chunk, three stages or more: it skips below 8 devices (MaxText CI's v6e-4)."""
    if self.n < 8:
      self.skipTest(f"needs 8 TPU devices or more (four stages of EP 2: a middle chunk to split), have {self.n}")
    self.enterContext(env(MPMD_FAKE_SLICES="2"))
    fsdp = self.n // 8
    args = [x for x in multihost_test.TINY if x != "--no-precompile"] + (
        f"--stages 4 --vv 1 --fsdp {fsdp} --ep 2 --gbs {8 * fsdp} --mb 4 --layers 4 --layout slices "
        "--transport native --ep-collectives xla"
    ).split()
    ref = dsp.main(args + ["--schedule", "1f1b"])
    self.assertTrue(all(map(math.isfinite, ref["warmup_losses"] + ref["warmup_gnorms"])), ref)
    for split_last, extra in ((False, []), (True, ["--zb-split-last", "on"])):
      with self.subTest(split_last=split_last):
        got = {name: dsp.main(args + ["--schedule", name, *extra]) for name in dsp.ZB_SCHEDULES}
        print(
            f"zero bubble on TPU (split last {split_last}): 1f1b {ref['warmup_losses']} {ref['warmup_gnorms']}; "
            + "; ".join(f"{n} {g['warmup_losses']} {g['warmup_gnorms']}" for n, g in got.items()),
            flush=True,
        )
        for name, g in got.items():
          self.assertEqual((g["zb"]["split_chunks"], g["zb"]["split_last"]), ([1, 2, 3][: 2 + split_last], split_last))
        assert_near_1f1b(self, got["zbh1"], ref, f"zbh1 (split last {split_last})", same_forward=not split_last)
        self.assertEqual(
            (got["zbh2"]["warmup_losses"], got["zbh2"]["warmup_gnorms"]),
            (got["zbh1"]["warmup_losses"], got["zbh1"]["warmup_gnorms"]),
        )

  def test_zero_bubble_two_stages(self):
    """At two stages of FSDP N/4 x EP 2, a fake slice each, nothing splits (chunk 0's backward stays whole, and the
    fused last chunk without --zb-split-last): zbh1 runs 1F1B's order, zbh2 one more warmup forward, the same programs
    and every chunk's gradients added in microbatch order -- every warmup loss and gradient norm 1F1B's bit for bit,
    on whatever compiler the backend runs.  Runs on 4 devices (MaxText CI's v6e-4)."""
    self.enterContext(env(MPMD_FAKE_SLICES="2"))
    fsdp = self.n // 4
    args = [x for x in multihost_test.TINY if x != "--no-precompile"] + (
        f"--stages 2 --vv 1 --fsdp {fsdp} --ep 2 --gbs {8 * fsdp} --mb 4 --layers 2 --layout slices "
        "--transport native --ep-collectives xla"
    ).split()
    ref = dsp.main(args + ["--schedule", "1f1b"])
    self.assertTrue(all(map(math.isfinite, ref["warmup_losses"] + ref["warmup_gnorms"])), ref)
    for name in dsp.ZB_SCHEDULES:
      with self.subTest(schedule=name):
        got = dsp.main(args + ["--schedule", name])
        self.assertNotIn("zb", got)  # no split chunk: no 'zb' block
        self.assertEqual((got["warmup_losses"], got["warmup_gnorms"]), (ref["warmup_losses"], ref["warmup_gnorms"]))

  @pytest.mark.scheduled_only
  def test_zero_bubble_split_last(self):
    """zbh1 and zbh2 with the last chunk split too (--zb-split-last on) against 1F1B on two stages of FSDP N/4 x EP
    2, a fake slice each, precompiled: the last chunk the one split (its F the forward, head, loss and input gradient,
    its W its layers' weight gradients), on whatever compiler the backend runs -- zbh1's warmup losses and gradient
    norms 1F1B's by the TPU contract (`assert_near_1f1b`), zbh2's zbh1's bit for bit.  scheduled_only:
    --zb-split-last is off by default, and the contract was measured on a tpu7x-8 only (this test runs on MaxText
    CI's v6e-4 too)."""
    self.enterContext(env(MPMD_FAKE_SLICES="2"))
    fsdp = self.n // 4
    args = [x for x in multihost_test.TINY if x != "--no-precompile"] + (
        f"--stages 2 --vv 1 --fsdp {fsdp} --ep 2 --gbs {8 * fsdp} --mb 4 --layers 2 --layout slices "
        "--transport native --ep-collectives xla"
    ).split()
    ref = dsp.main(args + ["--schedule", "1f1b"])
    self.assertTrue(all(map(math.isfinite, ref["warmup_losses"] + ref["warmup_gnorms"])), ref)
    got = {name: dsp.main(args + ["--schedule", name, "--zb-split-last", "on"]) for name in dsp.ZB_SCHEDULES}
    print(
        f"split last on TPU: 1f1b {ref['warmup_losses']} {ref['warmup_gnorms']}; "
        + "; ".join(f"{n} {g['warmup_losses']} {g['warmup_gnorms']}" for n, g in got.items()),
        flush=True,
    )
    for name, g in got.items():
      self.assertEqual((g["zb"]["split_chunks"], g["zb"]["split_last"]), ([1], True), name)
    assert_near_1f1b(self, got["zbh1"], ref, "zbh1 split last", same_forward=False)
    self.assertEqual(
        (got["zbh2"]["warmup_losses"], got["zbh2"]["warmup_gnorms"]),
        (got["zbh1"]["warmup_losses"], got["zbh1"]["warmup_gnorms"]),
    )

  def check_trace(self, session):
    """The session holds a Pathways worker's file (server-*.xplane.pb) with device planes."""
    from maxtext.experimental.mpmd.jax import xprof, xprof_align

    files = xprof_align.find_xplanes(session)
    workers = [f for f in files if os.path.basename(f).startswith("server-")]
    self.assertTrue(workers, files)
    found = []
    for w in workers:
      with tempfile.NamedTemporaryFile(suffix=".xplane.pb") as f:
        f.write(xprof_align.read_bytes(w))
        f.flush()
        if xprof.has_device_planes(f.name):
          found.append(w)
    self.assertTrue(found, workers)


@pytest.mark.tpu_only
@pytest.mark.scheduled_only
class SplitBackwardOnTpu(unittest.TestCase):
  """ds_ep_test.SplitBackward's benchmark chunk (its test_megablox) on a TPU's own compiler -- one MoE layer on
  megablox and the ragged sort, tokamax splash, bf16, on one FSDP 2 x EP 2 stage of 4 devices -- at emb 1024 (MaxText's
  SparseCore unsort kernel, ragged_gather_reduce_v2, asserts num_row_partitions <= num_simd_lanes on TPU7x below that:
  32 > 16 at tiny()'s emb 256 and at 512), in this process (a TPU VM's client or Pathways'): its backward cut into I
  and W (ds_ep.ChunkPrograms.build_split) against bwd, bwda and the B part with rs_self by the TPU contract
  (ds_ep_test.check_split with `tpu_split_tol`: within TPU_SPLIT_DEFAULT_TOL under XLA's default options, bit for bit
  with excess precision off but MLA's q_norm / kv_norm scales), and W the weight gradients' share of the flops.  The
  contract the zero-bubble schedules' TPU runs start from: CPU's bit equality says nothing of how a TPU's XLA fuses
  the halves.  scheduled_only: measured on a tpu7x-8 only."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    testing.require_tpu(len(testing.tpu_devices()) >= 4, "TPU devices, 4 or more")

  @staticmethod
  def config():
    """ds_ep_test.tiny()'s three layers on FSDP 2 x EP 2 at emb 1024, the width TPU7x's SparseCore unsort takes."""
    from tests.unit.experimental.mpmd.jax import ds_ep_test  # pylint: disable=import-outside-toplevel

    scfg = ds_ep_test.tiny(layers=3, fsdp=2, ep=2)
    return dataclasses.replace(scfg, model=scfg.model.replace(emb_dim=1024))

  def test_megablox(self):
    from maxtext.experimental.mpmd.jax import ds_ep  # pylint: disable=import-outside-toplevel
    from tests.unit.experimental.mpmd.jax import ds_ep_test  # pylint: disable=import-outside-toplevel

    ds_ep_test.check_split(self, self.config(), ds_ep.ChunkSpec(1, (1,), False, False, 0), tol=tpu_split_tol)

  def test_last_chunk(self):
    """The fused last chunk on the same kernels cut into AI and W (ds_ep.ChunkPrograms.build_split_last) against fb
    and fba (ds_ep_test.check_split_last) by the same contract: --zb-split-last's on a TPU's compiler."""
    from maxtext.experimental.mpmd.jax import ds_ep  # pylint: disable=import-outside-toplevel
    from tests.unit.experimental.mpmd.jax import ds_ep_test  # pylint: disable=import-outside-toplevel

    ds_ep_test.check_split_last(self, self.config(), ds_ep.ChunkSpec(2, (2,), False, True, 0), tol=tpu_split_tol)


if __name__ == "__main__":
  unittest.main()
