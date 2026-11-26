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

"""Pathways -- single-controller JAX, JAX_PLATFORMS=proxy -- for the MPMD pipeline.

Under Pathways one Python process, the client (on a GKE Pathways job's head pod, or any VM that reaches the proxy),
drives every device of the job through the IFRT proxy: jax.process_count() is 1 and every device is the client's.
ds_ep_pipeline then takes its one-process path as it is -- every stage local, each stage boundary one IFRT CopyArrays
(transfer.Channel), nothing of the multi-controller machinery -- and this module holds what differs from a TPU VM's
own PJRT client (McJAX), and the checks main makes in every mode, McJAX too (check_bucket, probe_trace_dir):

  is_proxy()           the mode, from JAX_PLATFORMS ('proxy' its first platform: jax's default backend), asked before
                       any backend starts; is_proxy_device(d) asks a device (its client's runtime_type,
                       'proxy/pathways') -- the same answer for the devices jax.devices() gives, which
                       ds_ep_pipeline's Pipeline asks.
  check_transport()    a --transport the mode cannot take: the relay and xfer move arrays between processes (and
                       jax.experimental.transfer, xfer's, kills the client) -- ds_ep_pipeline and mem_probe refuse
                       them through it (ds_ep_pipeline.multi_controller_only) before the backend starts, and
                       ds_ep_aot --pathways as well (pathways_problems).
  initialize()         pathwaysutils.initialize(), once, before the first backend use: its profiler patch (Pathways
                       traces go to gs:// only) and jax's persistent compile cache off (Pathways keeps its own under
                       the servers' --gcs_scratch_location; jax's is silently inert through the proxy).
  server_flags() ...   the libtpu / XLA flags: the proxy server compiles, so flags in the client's LIBTPU_INIT_ARGS
                       reach no compiler.  They go to the proxy server, the resource manager and the workers --
                       xpk's --custom-pathways-{proxy-server,server,worker}-args (server_args), or the local docker
                       containers' arguments -- and the same set stays in the client's LIBTPU_INIT_ARGS, which the
                       pipeline's planning reads (ds_ep_pipeline.libtpu_flag).
  hbm_bytes()          a device's HBM by its kind: memory_stats() raises through the proxy ("MemoryStats is only
                       supported for addressable PjRt devices").
  check_bucket()       in every mode: a gs:// --profile or --json that names no bucket (gs://, gs:///traces/run: a
                       `gs://$BUCKET/...` with BUCKET unset), refused before anything starts.
  check_trace_dir()    a Pathways profile's directory: gs:// only (pathwaysutils refuses others), with a bucket.
  probe_trace_dir()    in every mode: can this client write into a gs:// --profile (an object written there and
                       deleted)?  main asks before anything starts in a run that traces into one, since a trace
                       written straight into a bucket that does not exist hangs when it stops, in one McJAX process
                       too (under Pathways in stop_trace, leaving the proxy refusing the next clients).
  trace()              a profile: max_num_hosts worker hosts, an explicit session id, and a settle after the start (the
                       workers' TPU tracer starts late: without it a trace has no device plane); a failure to start or
                       stop it is a TraceError, which the caller can record and go on.
  describe()           what provenance records of the mode: the backend, the server build, the image (MPMD_PATHWAYS_IMAGE,
                       set by the launcher: the client cannot see it).
  multi_controller()   THE rule for the pipeline's multi-controller logic: several JAX processes, one per host
                       (jax.distributed), each driving only its own devices?  Never under Pathways (one client drives
                       every device), nor in one host's own process: what exists only for several processes -- every
                       boundary copy walked in one global order, the key-value store's agreements and barriers, the
                       relay, xfer and cross-process copies, per-process records and traces -- runs only where it says
                       yes (ds_ep_pipeline's Pipeline keeps it as `multi`), and its flags are refused under Pathways
                       before the backend starts (ds_ep_pipeline.multi_controller_only).

Also different under the proxy, with nothing to do here: host callbacks are UNIMPLEMENTED (the pipeline has none),
device ids change with every client session (the pipeline only orders by them), each new client waits ~10 s for the
worker, and only one client may hold the proxy at a time.

    python3 -m maxtext.experimental.mpmd.jax.pathways flags tpu7x        # one line: client LIBTPU_INIT_ARGS, docker args
    python3 -m maxtext.experimental.mpmd.jax.pathways flags tpu7x --xpk  # xpk's three --custom-pathways-*-args

Jax-light: imports neither jax nor pathwaysutils at import.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import sys
import time
import warnings

# The flag ds_ep_run.sh, ds_ep_aot.sh and ds_ep_multihost.sh add to any libtpu flag set: the FSDP gathers and
# reduce-scatters are collective-permutes folded into the stage programs, and libtpu's default cap on concurrent
# async permutes serializes them.
MPMD_FLAGS = ("--xla_max_concurrent_async_collective_permutes=64",)
# xpk's options that carry flags to each Pathways container (xpk splits each value on single spaces).
SERVER_OPTIONS = ("--custom-pathways-proxy-server-args", "--custom-pathways-server-args", "--custom-pathways-worker-args")
# The platform that selects Pathways (JAX_PLATFORMS=proxy), and the client's runtime_type through the IFRT proxy,
# 'proxy/pathways' ('pjrt_ifrt' on a TPU VM's own client).
PROXY = "proxy"
PROXY_RUNTIME = "proxy/"
# The settle after a trace starts: the Pathways workers' TPU tracer starts late, and a trace stopped within ~1 s of
# its start has no device plane at all (local docker Pathways on tpu7x, images 20261117 and 20261124).
TRACE_SETTLE_S = 1.0
# A device's HBM, bytes, by its device_kind: what memory_stats()'s bytes_limit reports where it can (the TPU7x one on a
# tpu7x-8, libtpu 0.0.50.1), else what ds_ep_aot.TOPOLOGIES holds (v6e: a v6e-4's bytes_limit; v5p: what XLA compiles
# a v5p-8 program against).  Through the proxy memory_stats() raises: mem_probe prices against this.
HBM_BYTES = {
    "TPU7x": 101_724_450_816,
    "TPU v6 lite": 33_546_041_344,
    "TPU v5": 102_786_660_352,  # v5p, as libtpu names it
    "TPU v5p": 102_786_660_352,  # v5p, as Pathways names it
}

_initialized = False


def is_proxy(platforms=None):
  """Does this process run on Pathways: is 'proxy' JAX_PLATFORMS's first platform (or `platforms`')?  jax makes the
  first platform listed its default backend (and fails where a listed one does not start), so this is what
  `is_proxy_device` says of the devices jax.devices() gives -- ds_ep_pipeline's Pipeline asks its meshes' devices,
  main this, and the two agree.  Asked from the environment, so it is known before any backend starts.  ('proxy'
  further down the list starts the proxy backend beside another default one: pathwaysutils, which asks whether
  'proxy' is anywhere in the list, then patches the profiler of a run whose devices are not the proxy's.)"""
  platforms = os.environ.get("JAX_PLATFORMS", "") if platforms is None else platforms
  return platforms.split(",")[0].strip().lower() == PROXY


def is_proxy_device(d):
  """Is device `d` a Pathways device: its client's runtime_type 'proxy/...' (a TPU VM's own client says 'pjrt_ifrt';
  a test's fake device may carry a client that says either)."""
  return str(getattr(getattr(d, "client", None), "runtime_type", "")).startswith(PROXY_RUNTIME)


def multi_controller(devices=None):
  """Is this run multi-controller JAX -- several processes, one per host (jax.distributed), each driving only its own
  devices: jax.process_count() > 1?  Given `devices`: are they several processes' (a multi-controller run's, or a
  compile-only topology's hosts: ds_ep_aot), not one client's?

  The one predicate for the pipeline's multi-controller logic: walking every boundary copy in one global order (an
  op of another process's stage copies, its own runs), the key-value store's agreements and barriers, the relay,
  xfer and cross-process copies, per-process records and traces.  It runs where this says yes and nowhere else.
  Under Pathways the answer is no: one client process drives every device (jax.process_count() is 1), so none of
  that runs there, not even as a walk that finds nothing to do -- and a proxy reporting several processes would
  break that premise: refused (RuntimeError).  A one-host run on its own PJRT client is single-controller too.  Ask
  it once jax.distributed is up: before jax.distributed.initialize every process counts one.  Devices: the IFRT
  proxy's are one client's (unwalked); others, by their process_index."""
  if devices is not None:
    devs = list(devices)
    return bool(devs) and not is_proxy_device(devs[0]) and len({d.process_index for d in devs}) > 1
  import jax  # pylint: disable=import-outside-toplevel

  n = jax.process_count()
  if n > 1 and is_proxy():
    raise RuntimeError(
        f"JAX_PLATFORMS=proxy (Pathways) reports {n} processes: a Pathways run is one client process driving every "
        "device, which the pipeline's single-controller path assumes (its multi-controller paths -- key-value "
        "agreements, barriers, cross-process copies -- are not for a proxy)"
    )
  return n > 1


def check_transport(transport):
  """Raise ValueError for a --transport a Pathways run cannot take: the relay and xfer move arrays between processes
  (--distributed), and under Pathways one client process drives every device.  ds_ep_pipeline.main and mem_probe ask
  it before the backend starts (through ds_ep_pipeline.multi_controller_only), and ds_ep_aot --pathways
  (pathways_problems): the Pipeline would say only after the start that they need --distributed, which Pathways
  refuses too.  The message's one copy."""
  if transport in ("relay", "xfer"):
    raise ValueError(
        f"--transport {transport}: under Pathways (JAX_PLATFORMS=proxy) one client process drives every device, "
        "and the relay and xfer move arrays between processes (--distributed): use --transport native (each "
        "boundary one IFRT CopyArrays, device to device) or ppermute (collective-permute programs)"
    )


def initialize():
  """Under JAX_PLATFORMS=proxy: pathwaysutils.initialize(), once -> True; else nothing -> False.

  Call it before the first backend use.  It patches jax's profiler (start_trace, stop_trace and trace also start the
  Pathways workers' trace; they then take only gs:// directories) and turns jax's persistent compile cache off, which
  through the proxy writes nothing anyway: Pathways caches compiled programs under the servers'
  --gcs_scratch_location.  jax >= 0.12 registers the proxy backend itself, so where the backends are up already (a
  test session that touched jax first) only the patch and the cache switch apply.  Without pathwaysutils the run goes
  on, with a warning: only a --profile then fails (trace says why)."""
  global _initialized
  if not is_proxy():
    return False
  if _initialized:
    return True
  try:
    import pathwaysutils  # pylint: disable=import-outside-toplevel
  except ImportError:
    warnings.warn(
        "JAX_PLATFORMS=proxy without pathwaysutils (pip install 'pathwaysutils>=0.1.14'): jax's own proxy backend "
        "runs, but a profile records no Pathways worker",
        RuntimeWarning,
    )
    _initialized = True
    return True
  import jax  # pylint: disable=import-outside-toplevel
  from jax._src import xla_bridge  # pylint: disable=import-outside-toplevel

  if xla_bridge.backends_are_initialized():
    from pathwaysutils import profiling as pw_profiling  # pylint: disable=import-outside-toplevel

    pw_profiling.monkey_patch_jax()
    jax.config.update("jax_enable_compilation_cache", False)
  else:
    pathwaysutils.initialize()
  _initialized = True
  return True


# ------------------------------------------------------------------------------------------------- flags
def flags_dir():
  """The package's libtpu flag sets: libtpu_flags/<set>.flags, one flag a line, '#' starting a comment."""
  return os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtpu_flags")


def flag_sets():
  """The names of the flag sets there, sorted."""
  return sorted(f[: -len(".flags")] for f in os.listdir(flags_dir()) if f.endswith(".flags"))


def read_flag_set(name):
  """The flags of set `name` (libtpu_flags/<name>.flags) -> [flag], in file order: what the launchers read
  (ds_ep_run.sh: '#' starts a comment, a flag a word)."""
  path = os.path.join(flags_dir(), f"{name}.flags")
  if "/" in name or not os.path.isfile(path):
    raise ValueError(f"no libtpu flag set {name!r}: the sets are {flag_sets()} ({flags_dir()})")
  with open(path, encoding="utf-8") as f:
    return [tok for line in f for tok in line.split("#", 1)[0].split()]


def server_flags(flag_set="tpu7x", extra=()):
  """The flags a Pathways run gives its servers -> [flag]: set `flag_set`'s (None: none), the launchers' MPMD_FLAGS,
  then `extra` (flags, or strings of them), each flag once -- a later mention of a flag (the text before its '=')
  replaces an earlier one's value in its place, as libtpu's last mention wins."""
  out = {}
  for tok in [*(read_flag_set(flag_set) if flag_set else ()), *MPMD_FLAGS, *(t for e in extra for t in str(e).split())]:
    if not tok.startswith("-"):
      raise ValueError(f"{tok!r} is not a flag (--name or --name=value)")
    out[tok.split("=", 1)[0]] = tok
  return list(out.values())


def joined(flags):
  """`flags` as one argument value: single spaces, no empty token, nothing a split on ' ' would cut wrongly -- xpk
  splits each --custom-pathways-*-args value on ' ', so a doubled or trailing space would make an empty argument."""
  for f in flags:
    if not f or f != "".join(f.split()):
      raise ValueError(f"flag {f!r}: empty or holding whitespace, which a split on ' ' would cut")
  return " ".join(flags)


def server_args(flags):
  """xpk's options for `flags` -> {option: value}: the same flags to the proxy server (it compiles every program),
  the resource manager and the workers (SERVER_OPTIONS)."""
  value = joined(flags)
  return {opt: value for opt in SERVER_OPTIONS}


def client_flags_note(flags=None):
  """A line for a Pathways run whose client sets libtpu flags (LIBTPU_INIT_ARGS, or `flags`) beyond jax's own launch
  barrier one, else None: they configure no compiler here, only the pipeline's planning reads them."""
  flags = os.environ.get("LIBTPU_INIT_ARGS", "") if flags is None else flags
  own = {"--xla_tpu_use_enhanced_launch_barrier=true"}  # jax's cloud TPU init adds it wherever libtpu is installed
  mine = [f for f in flags.split() if f not in own]
  if not mine:
    return None
  return (
      f"note: LIBTPU_INIT_ARGS sets {len(mine)} flag(s) in this client: under Pathways the proxy server compiles, so "
      "they reach no compiler; the plan reads them (ds_ep_pipeline.libtpu_flag) -- give the servers the same set "
      "(python3 -m maxtext.experimental.mpmd.jax.pathways flags SET --xpk; README.md, Pathways)"
  )


# ------------------------------------------------------------------------------------------------- devices
def hbm_bytes(device_kind):
  """HBM per device of `device_kind`, bytes (HBM_BYTES), or None for a kind not in the table."""
  return HBM_BYTES.get(str(device_kind))


def memory_stats(d):
  """d.memory_stats() -> (stats or {}, None), or ({}, why) where the backend keeps none: the IFRT proxy raises."""
  try:
    return d.memory_stats() or {}, None
  # through the IFRT proxy a JaxRuntimeError, INVALID_ARGUMENT
  except Exception as e:  # pylint: disable=broad-exception-caught
    return {}, f"{type(e).__name__}: {' '.join(str(e).split())[:160]}"


# ------------------------------------------------------------------------------------------------- profiling
class TraceError(RuntimeError):
  """A Pathways trace that did not start or did not stop (`trace`): the profile is lost, the steps it timed are not."""


def check_bucket(path, flag="--profile"):
  """Raise ValueError, naming `flag`, where `path` is a gs:// URL that names no bucket: gs://, or gs:///traces/run (a
  `gs://$BUCKET/traces/run` with BUCKET unset).  Nothing can be written there, and nothing else says so in time: jax
  exports a trace into gs:// to the bucket 'plugins' (<dir>/plugins/profile/<session>), the write probe joined onto
  gs:// wrote a local directory 'gs:' and passed, and a record fails only when it is written (google-cloud-storage's
  IndexError on an empty bucket name)."""
  if str(path).startswith("gs://") and not str(path)[5:].split("/", 1)[0]:
    raise ValueError(f"{flag} {path}: no bucket name (gs://BUCKET/...)")


def check_trace_dir(path):
  """Raise ValueError unless `path` is a gs:// directory with a bucket (check_bucket), the only kind a Pathways trace
  writes to (the workers write their device planes there, the client its own)."""
  if not str(path).startswith("gs://"):
    raise ValueError(
        f"--profile {path}: under Pathways a trace goes to a gs:// directory (the Pathways workers write their device "
        "planes there; pathwaysutils refuses any other)"
    )
  check_bucket(path)


def storage_error(e):
  """A storage error in one line: '<type> <HTTP status>: <the service's message>' for a google.api_core error, whose
  str() carries the response's whole JSON body (the message is its error.message), else '<type>: <str(e)>'."""
  msg = None
  try:
    msg = e.response.json()["error"]["message"]
  except Exception:  # pylint: disable=broad-exception-caught  (no response, not JSON, or another shape)
    pass
  code = getattr(e, "code", None)
  head = f"{type(e).__name__} {code}" if isinstance(code, int) else type(e).__name__
  return f"{head}: {' '.join(str(msg or e).split())[:400]}"


def probe_trace_dir(path):
  """Write a small object into gs:// directory `path` and delete it -> None where the write works (a failed delete
  only warns: a bucket that takes writes but not deletes still takes a trace); else raise ValueError with the storage
  error (storage_error).  A trace written straight into a directory nobody can write -- a bucket that does not exist
  -- hangs when it stops: under Pathways the client waits on stop_trace's future without end and the proxy refuses
  the next clients until it restarts (pathwaysutils 0.1.14, images 20261117 and 20261124); in one McJAX process
  jax's profiler waits in export_profile_data.  Call it before anything starts, as check_trace_dir.  It checks the
  client's access only: the Pathways servers write the device planes with credentials of their own.  Without
  google-cloud-storage it cannot check: a warning, and no error.  A gs:// URL that names no bucket is a ValueError
  before any write (check_bucket)."""
  # jax-light, as this module
  from maxtext.experimental.mpmd.jax import xprof_align  # pylint: disable=import-outside-toplevel

  check_bucket(path)
  obj = f'{str(path).rstrip("/")}/.mpmd_write_probe.{os.getpid()}.{time.time_ns()}'
  try:
    xprof_align.write_bytes(obj, b"mpmd: can this client write here?\n")
  except ImportError as e:  # google-cloud-storage missing: --json gs:// would fail too, a trace may not
    warnings.warn(f"--profile {path}: not checked that this client can write there ({e})", RuntimeWarning)
    return
  # google.api_core / google.auth errors are no OSError, and the bucket's absence is one of them
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise ValueError(
        f"--profile {path}: this client cannot write there ({storage_error(e)}); a trace into it would hang when it "
        "stops (under Pathways leaving the proxy refusing the next clients)"
    ) from None
  try:
    xprof_align.delete(obj)
  except Exception as e:  # pylint: disable=broad-exception-caught
    warnings.warn(f"--profile {path}: could not delete the write probe {obj} ({storage_error(e)})", RuntimeWarning)


def session_id(when=None):
  """A trace session's name as jax and pathwaysutils make one: `when` (seconds since the epoch; None: now) to the
  second, in this host's local time.  XProf sorts a logdir's sessions by it, and so does xprof.find_xplane: every
  session this code names (a Pathways one; the multi-controller one, ds_ep_pipeline.shared_profile_session) is named
  here, in the time zone jax names its own in -- the zone TZ or /etc/localtime names (a TZ that is a POSIX rule, not
  a zone's name -- PST8, say -- jax takes for UTC)."""
  return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(when))


@contextlib.contextmanager
def trace(log_dir, profiler_options=None, max_num_hosts=1, settle_s=TRACE_SETTLE_S):
  """A Pathways trace of the body into gs:// `log_dir`: the client's own and up to `max_num_hosts` worker hosts' --
  pathwaysutils' start_trace (whose request carries `profiler_options`' advanced configuration, the chip and
  SparseCore trims, and its session_id) -- then `settle_s` of sleep before the body, so the workers' TPU tracers are
  recording when the first program runs; stop_trace when the body ends, whatever way.  The session's files land in
  `log_dir`/plugins/profile/<session_id>/.

  A trace that does not start (the body then does not run) or does not stop is a TraceError, and the body's own
  error stays its own: where the body raised, a stop_trace failure is only a note on that error.  pathwaysutils is
  needed (a TraceError without it)."""
  check_trace_dir(log_dir)
  try:
    from pathwaysutils import profiling as pw_profiling  # pylint: disable=import-outside-toplevel
  except ImportError as e:
    raise TraceError(
        "--profile under Pathways needs pathwaysutils >= 0.1.14 (it asks the workers for their trace)"
    ) from e
  try:
    pw_profiling.start_trace(str(log_dir), profiler_options=profiler_options, max_num_hosts=max_num_hosts)
  except Exception as e:  # pylint: disable=broad-exception-caught  (a RuntimeError, or the backend's own)
    raise TraceError(f"the trace did not start ({type(e).__name__}: {e})") from e
  try:
    time.sleep(settle_s)
    yield
  except BaseException as body:
    try:
      pw_profiling.stop_trace()
    except Exception as e:  # pylint: disable=broad-exception-caught
      body.add_note(f"and stopping the Pathways trace failed: {type(e).__name__}: {e}")
    raise
  try:
    pw_profiling.stop_trace()
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise TraceError(f"the trace did not stop ({type(e).__name__}: {e})") from e


# ------------------------------------------------------------------------------------------------- provenance
def describe(backend=None):
  """The mode as provenance records it -> {'backend', 'server', 'image'}: the client's runtime type and the server's
  platform version (`backend`: the jax client, default jax's), and the image, MPMD_PATHWAYS_IMAGE (the launcher's:
  the client has no way to see it), None where unset."""
  if backend is None:
    import jax  # pylint: disable=import-outside-toplevel

    backend = jax.devices()[0].client
  return {
      "backend": str(getattr(backend, "runtime_type", "?")),
      "server": " ".join(str(getattr(backend, "platform_version", "?")).split()),
      "image": os.environ.get("MPMD_PATHWAYS_IMAGE") or None,
  }


# ------------------------------------------------------------------------------------------------- command line
def main(argv=None):
  ap = argparse.ArgumentParser(
      prog="python3 -m maxtext.experimental.mpmd.jax.pathways",
      description="The libtpu / XLA flags of a Pathways run, for its servers (and the client's LIBTPU_INIT_ARGS).",
  )
  sub = ap.add_subparsers(dest="cmd", required=True)
  f = sub.add_parser("flags", help="a flag set (libtpu_flags/<set>.flags) plus the launchers' MPMD flags")
  f.add_argument("set", help=f"the flag set: one of {flag_sets()}, or none")
  f.add_argument(
      "--extra",
      action="append",
      default=[],
      help="more flags after the set's, a string of them: --extra='--xla_a=1 --xla_b=2' (with '=': a value that "
      "starts with '--')",
  )
  f.add_argument(
      "--xpk",
      action="store_true",
      help="print xpk's three --custom-pathways-*-args options, shell-quoted, instead of one line of flags",
  )
  a = ap.parse_args(argv)
  try:
    flags = server_flags(None if a.set == "none" else a.set, a.extra)
  except ValueError as e:
    raise SystemExit(f"pathways flags: {e}") from None
  if a.xpk:
    print(" ".join(f"{opt}={shlex.quote(v)}" for opt, v in server_args(flags).items()))
  else:
    print(joined(flags))
  return flags


if __name__ == "__main__":
  main(sys.argv[1:])
