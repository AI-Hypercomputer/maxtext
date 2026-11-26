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

"""Multi-controller MPMD pipeline on CPU: N local JAX processes, each one host.

`ds_ep_pipeline --distributed` runs one process per host; a process runs the
stages holding its devices and takes part in every boundary copy, in order
(transfer.CrossProcessChannel).  Here N processes on this machine stand in for
the hosts (jax.distributed over localhost, CPU cross-host copies over the
socket transfer server), and the run must match the same pipeline in ONE
process on the same devices: same plan, same losses and gradient norms, on
every process.

    JAX_PLATFORMS=cpu python3 -m pytest -q tests/unit/experimental/mpmd/jax/multihost_test.py
    python3 -m tests.unit.experimental.mpmd.jax.multihost_test --procs 4 --devs 4 -- --stages 4 --fsdp 4 --ep 1 ...

Each pipeline test starts 2-5 JAX processes, each with CPU devices of its own (jax.distributed over localhost):
under a minute each on a workstation, several on a small CI runner, about 20 minutes in all.  So every test
that starts processes skips unless MAXTEXT_MPMD_MULTIHOST_TESTS=1 (testing.require_multihost, called by
run_multi): MaxText's CPU CI runs many test modules in each pytest worker, and there the out-of-memory killer took
these processes.  Run them by hand; the two over four fake slices on the host relay, minutes each, also need
MAXTEXT_MPMD_SLOW_TESTS=1 (testing.slow).  test_q_tiny_gives_every_model_flag, a parse, runs on every PR.  They cover transfer.py's
cross-process channels with no model (every copy direction, bystanders, a process holding no stage, and the
build-time rejections); four hosts of one device under a stage -> process map that is not the identity, for
DeepSeek-V3 layers (TINY; with their profile in one XProf session) and the Qwen3.5 family's (Q_TINY), under
1F1B-like orders and under the zero-bubble schedules (a split backward's I and W; paced as on a TPU VM's own
client too, each process its own stages);
ds_ep_multihost.sh on two hosts, a dry run; more chunks, schedules and transports, FSDP within and across
processes, the MaxText parity inputs, a process holding no stage, processes that build different plans or lack
an input file, and the Qwen3.5 family on four fake slices over the host relay.  The children run with XLA:CPU's
YNN fusions off on jax >= 0.11.2, and with no persistent compile cache unless JAX_CACHE_DIR names one (`_env`).
The reference of the two-slice runs -- one process, --layout slices over two fake slices, native Channel
boundaries: the path a Pathways run takes -- runs on every PR as pathways_test.OneProcessSlices.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import pytest

from maxtext.experimental.mpmd.jax import schedule as sched
from maxtext.experimental.mpmd.jax import testing

ROOT = testing.import_root()  # the children's working directory
# fp32 master weights and lr 1e-3: one update moves the loss by ~5% (bf16
# weights at the default lr 1e-5 moved it by ~1e-3, the order of the later
# steps' tolerance), so a stage's update gone wrong shows in the next step.
TINY = (
    "--emb 256 --heads 2 --q-lora 128 --kv-lora 128 --moe-mlp 256 --experts 8 --topk 2 --vocab 512 "
    "--seq 256 --sa-block 128 --sa-block-kv-compute 128 --gmm-impl ragged_dot --ragged-sort off "
    "--attn reference --routing topk --warmup 3 --no-precompile --weight-dtype float32 --lr 1e-3"
).split()
# TINY for the Qwen3.5 family: Qwen3.8-2.4T-A95B's layers (--model; 3 Gated DeltaNet layers, then one gated
# full-attention layer, every layer MoE with a sigmoid-gated shared expert) at toy widths -- GQA 2:1, 8 GDN value
# heads a key head as in 2.4T -- with TINY's kernels, weights and learning rate.  Every shape flag is given, and
# the rotary embedding (--qwen-rope partial: 2.4T's, which --model would default anyway), so --model picks only the
# family; the tests give --layers, a multiple of the 4-layer block.
Q_TINY = (
    "--model qwen3.8-2.4t-a95b --emb 256 --heads 4 --kv-heads 2 --head-dim 64 --gdn-key-heads 2 "
    "--gdn-value-heads 16 --gdn-key-head-dim 32 --gdn-value-head-dim 32 --qwen-rope partial --moe-mlp 256 "
    "--experts 8 --topk 2 --vocab 512 --seq 256 --sa-block 128 --sa-block-kv-compute 128 --gmm-impl ragged_dot "
    "--ragged-sort off --attn reference --routing topk --warmup 3 --no-precompile --weight-dtype float32 "
    "--lr 1e-3"
).split()


def _ports(n):
  """n free loopback ports for one run, each checked by binding it while the
  ones drawn before it are still held (so no two alike).

  A transfer server binds its port only at the run's first copy, seconds
  after the draw, and a port taken meanwhile fails that copy ('Address
  already in use') or, taken by another run's transfer server, hangs every
  copy silently.  So the ports come from below Linux's ephemeral range
  (32768-60999; macOS's starts at 49152), which no socket gets unasked --
  gloo's connections take ephemeral ports by the hundred -- and never from
  the ones a run on this machine drew in the last 15 minutes (a list in the
  temp directory, locked while drawing: pytest -n runs several at once)."""
  import fcntl

  book = os.path.join(tempfile.gettempdir(), f"mpmd_multihost_ports.{os.getuid()}")
  with open(book, "a+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0)
    now = time.time()
    recent = {}
    for line in f:  # a line cut short (a run killed mid-write) is skipped
      try:
        p, t = line.split()
        if now - float(t) < 900:
          recent[int(p)] = float(t)
      except ValueError:
        pass
    rng, socks = random.Random(), []
    try:
      while len(socks) < n:
        port = rng.randrange(10000, 32768)
        if port in recent:
          continue
        s = socket.socket()
        try:
          s.bind(("127.0.0.1", port))
        except OSError:
          s.close()
          continue
        socks.append(s)
      ports = [s.getsockname()[1] for s in socks]
    finally:
      for s in socks:
        s.close()
    recent.update((p, now) for p in ports)
    f.seek(0)
    f.truncate()
    f.writelines(f"{p} {t}\n" for p, t in recent.items())
  return ports


# XLA:CPU's YNNPACK dot fusions, which `YNN_FLAG=` (no fusion type) turns off.  jax 0.11.2's CPU backend runs them by
# default, and a bf16 matmul no longer accumulates in fp32 as on 0.11.1.  With them on, 3 of 4 runs
# of one single-process TINY pipeline had a step whose gradient norm came out ~2e3 instead of ~1.4 (jax 0.12.0.1),
# failing test_bitwise_without_stage_collectives; with them off all 4 agreed bit for bit.
YNN_FLAG = "--xla_cpu_experimental_ynn_fusion_type"


def _ynn_off():
  """[f'{YNN_FLAG}='] for children of a jaxlib >= 0.11.2 (this interpreter's), else []: an older one keeps the
  children's flags as they were."""
  try:
    from jaxlib import version  # the version module only: no XLA, no backend
  except ImportError:
    return []
  return [f"{YNN_FLAG}="] if getattr(version, "__version_info__", ()) >= (0, 11, 2) else []


def _env(devs, **extra):
  # MPMD_RELAY_IP: the address --transport relay's listeners publish, loopback here.  XLA_FLAGS: `devs` CPU devices
  # and, on jax >= 0.11.2, XLA:CPU's YNN fusions off (`_ynn_off`): the runs need no flag from the caller.
  env = dict(
      os.environ,
      JAX_PLATFORMS="cpu",
      XLA_FLAGS=" ".join([f"--xla_force_host_platform_device_count={devs}", *_ynn_off()]),
      PYTHONPATH=testing.child_pythonpath(__name__),
      PYTHONHASHSEED="0",
      MPMD_RELAY_IP="127.0.0.1",
  )
  env.update(extra)
  # ds_ep_pipeline's __main__ writes every program it compiles to JAX_CACHE_DIR, ~/jax_cache when that is unset: the
  # runs would fill the home directory of whoever runs the tests, and a later run could read an entry back under old
  # op names, which the cache key leaves out.  So a run caches nothing unless the caller names a cache directory.
  if "JAX_CACHE_DIR" not in env:
    env["JAX_ENABLE_COMPILATION_CACHE"] = "false"
  return env


def run_single(args, devs, timeout=600, env=None):
  r = subprocess.run(
      [sys.executable, "-m", "maxtext.experimental.mpmd.jax.ds_ep_pipeline", *args],
      cwd=ROOT,
      env=_env(devs, **(env or {})),
      capture_output=True,
      text=True,
      timeout=timeout,
  )
  return r.returncode, r.stdout + r.stderr


def run_multi(args, procs, devs, timeout=600, collectives=None, extra_args=None, cmd=None, env=None):
  """N processes of `ds_ep_pipeline --distributed args` -> [(returncode,
  output)] of every process.  `extra_args`: {process: more args for it};
  `cmd`: another command for all of them (a program of the test's own);
  `env`: more environment for all of them.

  Each process writes to a file of its own (a pipe nobody reads while the
  harness waits on another process would stall a chatty one).  One deadline
  covers them all, and once a process has failed the others get 30 s to
  end on their own (one waiting for it would abort only minutes later, in
  the coordination service's shutdown); a process still running then is
  killed (returncode -9), and its output so far comes back with the
  others'.  A run that lost a port to another program anyway is run again on
  fresh ports, twice at most."""
  testing.require_multihost()
  cmd = cmd or [sys.executable, "-m", "maxtext.experimental.mpmd.jax.ds_ep_pipeline", "--distributed", *args]
  for _ in range(3):
    outs = _launch(cmd, procs, devs, timeout, collectives, extra_args or {}, env or {})
    if all(rc == 0 for rc, _ in outs) or not any("Address already in use" in o for _, o in outs):
      break
  return outs


def _launch(cmd, procs, devs, timeout, collectives, extra_args, more_env):
  coord, *xfer_ports = _ports(procs + 1)
  # MPMD_XFER_IP: --transport xfer's servers on loopback.  The cross-host copy
  # gives up on a peer's metadata after 60 s by default -- on a slow machine a
  # downstream stage waits that long while its upstream is still compiling.
  extra = dict(
      more_env,
      JAX_COORDINATOR_ADDRESS=f"127.0.0.1:{coord}",
      JAX_NUM_PROCESSES=str(procs),
      MPMD_XFER_IP="127.0.0.1",
      JAX_CROSS_HOST_TRANSFER_TIMEOUT_SECONDS="1800",
  )
  if collectives:
    extra["JAX_CPU_COLLECTIVES_IMPLEMENTATION"] = collectives
  ps, logs, why = [], [], ""
  try:
    for i in range(procs):
      logs.append(tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace"))
      # Each process's transfer server on a port of its own: it publishes the
      # address as given, so '127.0.0.1:0' would leave its peers nowhere to
      # connect (every copy then hangs).
      env = _env(
          devs, JAX_PROCESS_ID=str(i), **extra, JAX_CROSS_HOST_TRANSFER_SOCKET_ADDRESS=f"127.0.0.1:{xfer_ports[i]}"
      )
      ps.append(
          subprocess.Popen([*cmd, *extra_args.get(i, ())], cwd=ROOT, env=env, stdout=logs[i], stderr=subprocess.STDOUT)
      )
    deadline, grace = time.monotonic() + timeout, None
    while any(p.poll() is None for p in ps):
      failed = [i for i, p in enumerate(ps) if p.poll() not in (None, 0)]
      if failed and grace is None:
        grace = time.monotonic() + 30
      if time.monotonic() > deadline:
        why = f"still running at the {timeout:g} s deadline"
        break
      if grace is not None and time.monotonic() > grace:
        why = f"still running 30 s after process {failed[0]} failed"
        break
      time.sleep(0.2)
  finally:
    killed = [i for i, p in enumerate(ps) if p.poll() is None]
    for i in killed:
      ps[i].kill()
    for p in ps:
      p.wait()
  out = []
  for i, (p, f) in enumerate(zip(ps, logs)):
    f.seek(0)
    out.append((p.returncode, f.read() + (f"\n[killed: {why}]" if i in killed else "")))
    f.close()
  return out


# ds_ep_pipeline's main with the zero-bubble I's paced as on a TPU VM's own client (Pipeline.pace_i, forced on: CPU
# devices allocate no output at issue): each wait printed with its op -- refused unless the op's stage is this
# process's.
PACED_MAIN = """if True:
    import sys
    from unittest import mock
    from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
    pace = dsp.Pipeline._pace
    def paced(p, st, op):
        if st.last_i.get(op.c) is not None:
            assert op.s in p.local, (op, sorted(p.local))
            print(f"paced c{op.c} m{op.m} s{op.s}", flush=True)
        return pace(p, st, op)
    with mock.patch.object(dsp, "allocates_at_issue", return_value=True), mock.patch.object(dsp.Pipeline, "_pace", paced):
        dsp.main(sys.argv[1:])
"""


def write_input_files(d, args, devs):
  """In a subprocess: d/chunk<c>.npz from a fresh init of the pipeline `args`
  builds (ds_ep_pipeline.write_chunk_files, the files --init-from reads) and
  d/tokens.npy, [gbs, seq + 1] random tokens."""
  code = """if True:
    import sys
    import jax, numpy as np
    from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
    d, args = sys.argv[1], sys.argv[2:]
    p = dsp.build(dsp.make_parser().parse_args(args))
    params, _, _ = p.init(jax.random.key(7))
    dsp.write_chunk_files(d, p.chunk_trees(params))
    toks = np.random.default_rng(1).integers(0, p.scfg.model.vocab_size, (p.gbs, p.seq + 1), dtype=np.int32)
    np.save(f'{d}/tokens.npy', toks)
  """
  r = subprocess.run(
      [sys.executable, "-c", code, d, *args], cwd=ROOT, env=_env(devs), capture_output=True, text=True, timeout=600
  )
  if r.returncode:
    raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])


def channel_check():
  """One process of `test_cross_process_channels`, under jax.distributed
  (four processes of two devices): transfer.py's cross-process channels with
  no model.  Stage s runs on process s's devices for s < 3; process 3 holds
  no stage.  Each channel kind carries a bf16 array sharded over a stage and
  a replicated f32 scalar from every stage to every other, three rounds, all
  in one order on every process: each value arrives bit for bit, on the
  destination sharding and nowhere else.  Every process watches some copies
  and makes others, so one that skipped a native copy it only watches would
  pair its next ones wrongly (it hangs).  Then the build-time rejections,
  the same on every process, and a wrong source array, which the sender
  rejects before it moves anything."""
  import jax
  import ml_dtypes
  from jax._src import distributed
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
  from maxtext.experimental.mpmd.jax import ds_ep_pipeline, transfer

  ds_ep_pipeline.init_distributed()
  pid = jax.process_index()
  own = [[d for d in jax.devices() if d.process_index == q] for q in range(jax.process_count())]
  assert [len(x) for x in own] == [2] * 4, own
  mesh = lambda devs: Mesh(np.array(devs), ("x",))
  stages = [mesh(own[s]) for s in range(3)]
  leaves = [((4, 6), ml_dtypes.bfloat16, P("x")), ((), np.float32, P())]
  pairs = [(a, b) for a in range(3) for b in range(3) if a != b]
  ctx = transfer.xfer_context()
  first = {}  # kind -> its s0->s1 channel of the bf16 leaf
  for k, kind in enumerate(("native", "ppermute", "xfer")):
    chans = []
    for j, (a, b) in enumerate(pairs):
      for i, (shape, dtype, spec) in enumerate(leaves):
        aval = jax.ShapeDtypeStruct(shape, dtype)
        src, dst = NamedSharding(stages[a], spec), NamedSharding(stages[b], spec)
        name = f"s{a}->s{b}"
        ch = (
            transfer.CrossProcessChannel(aval, src, dst, name=name)
            if kind == "native"
            else transfer.PermuteChannel(aval, src, dst, name=name)
            if kind == "ppermute"
            else transfer.XferChannel(aval, src, dst, ctx, 2 * j + i, name=name)
        )
        chans.append((a, b, i, ch, src, dst))
    first[kind] = chans[0][3]
    for r in range(3):
      for a, b, i, ch, src, dst in chans:
        shape, dtype, _ = leaves[i]
        want = np.asarray(np.random.default_rng((k, a, b, i, r)).standard_normal(shape), np.float32).astype(dtype)
        y = ch.send_flat([jax.device_put(want, src) if pid == a else None])[0]
        where = f"{kind} {ch.name} leaf {i} round {r}"
        if pid == b:
          assert y.sharding == dst, (where, y.sharding)
          np.testing.assert_array_equal(np.asarray(y, np.float32), want.astype(np.float32), err_msg=where)
        elif kind == "native":  # jax's copy: the destination array, with no shards here
          assert y.sharding == dst and not y.addressable_shards, (where, y)
        else:
          assert y is None, (where, y)
    print(f"{kind}: {3 * len(chans)} copies exact", flush=True)

  def rejects(exc, match, f, *args):
    try:
      f(*args)
    except exc as e:
      assert match in str(e), (match, str(e))
    else:
      raise AssertionError(f"no {exc.__name__} saying {match!r}")

  # xfer_peers is a function of the meshes and a process: every process checks every process
  for q in range(4):
    got = transfer.xfer_peers(stages[0], stages[1], q)
    assert got == [([0, 1], [], None, 1), ([], own[1], 0, None), ([], [], None, None)][min(q, 2)], (q, got)
  across = [mesh([own[0][x], own[1][x]]) for x in range(2)]  # two meshes over processes 0 and 1
  rejects(ValueError, "process 0 holds devices of both stages", transfer.xfer_peers, across[0], across[1], 0)
  rejects(ValueError, "would receive from processes [0, 1]", transfer.xfer_peers, across[0], stages[2], 2)
  rejects(ValueError, "would send to processes [0, 1]", transfer.xfer_peers, stages[2], across[0], 2)
  rejects(ValueError, "share devices", transfer.xfer_peers, stages[0], mesh([own[0][1], own[1][0]]), 1)
  aval = jax.ShapeDtypeStruct((4, 6), np.float32)
  xpc = lambda a, b, spec=P("x"): transfer.CrossProcessChannel(aval, NamedSharding(a, spec), NamedSharding(b, spec))
  rejects(transfer.ShardingMismatchError, "would stay on one process", xpc, across[0], mesh([own[0][1], own[2][0]]))
  rejects(ValueError, "on the same processes", xpc, across[0], across[1])
  rejects(transfer.ShardingMismatchError, "device counts differ", xpc, stages[0], mesh(own[1][:1]), P())
  other = Mesh(np.array(own[1]), ("y",))
  for make in (transfer.PermuteChannel, lambda *a: transfer.XferChannel(*a, ctx, 99)):
    rejects(
        transfer.ShardingMismatchError,
        "the stage meshes differ",
        make,
        aval,
        NamedSharding(stages[0], P("x")),
        NamedSharding(other, P("y")),
    )
    rejects(
        transfer.ShardingMismatchError,
        "the same spec on both",
        make,
        aval,
        NamedSharding(stages[0], P("x")),
        NamedSharding(stages[1], P()),
    )
  rejects(RuntimeError, "already has an XferContext", transfer.XferContext)
  assert transfer.xfer_context() is ctx
  if pid == 0:  # f32 where bf16 was declared: refused before any copy starts
    x = jax.device_put(np.zeros((4, 6), np.float32), NamedSharding(stages[0], P("x")))
    for ch in first.values():
      rejects(transfer.ShardingMismatchError, "bfloat16[4, 6]", ch.send_flat, [x])
  distributed.global_state.client.wait_at_barrier("channel_check_done", 300_000)
  print("channels ok", flush=True)


def _plan_text(summary):
  """A run's summary less what differs between its processes and one
  process: the marks on another process's stages, and the device ids --
  each becomes its rank among the summary's ids (process p's devices are
  numbered from p * 2048 on CPU), so the stage -> device map still counts."""
  s = summary.replace(" (* another process's stage: inferred)", "").replace("*", "")
  ids = sorted({int(x) for x in re.findall(r"(\d+)@", s)})
  return re.sub(r"(\d+)@", lambda m: f"#{ids.index(int(m[1]))}@", s)


class MultiHostTest(unittest.TestCase):

  def setUp(self):
    # Every test here but the flag parse starts several JAX processes: skip before any of its setup work.
    if self._testMethodName != "test_q_tiny_gives_every_model_flag":
      testing.require_multihost()

  def _check_ok(self, outs):
    """Every process exited 0; else every failed one's output, those with a
    Python traceback first (once one process dies the others abort minutes
    later in the coordination service's shutdown, which says nothing of why)."""
    bad = sorted((i for i, (rc, _) in enumerate(outs) if rc), key=lambda i: "Traceback" not in outs[i][1])
    self.assertFalse(bad, "\n".join(f"----- process {i}: exit {outs[i][0]}\n{outs[i][1][-3000:]}" for i in bad))

  def _records(self, path, outs):
    """Every process's run record (--json-procs all), in process order."""
    root, ext = os.path.splitext(path)
    names = [path] + [f"{root}.p{i}{ext}" for i in range(1, len(outs))]
    found = sorted(f for f in os.listdir(os.path.dirname(path)) if f.startswith(os.path.basename(root)))
    self.assertEqual(found, sorted(map(os.path.basename, names)))  # one record per process
    recs = []
    for i, name in enumerate(names):
      with open(name) as f:
        recs.append(json.load(f))
      self.assertEqual((recs[i]["process"], recs[i]["processes"]), (i, len(outs)))
    return recs

  def _compare(self, args, procs, devs, collectives=None, exact=True, profile=False, ref_devs=None, env=None, prog=None):
    """The multi-process run against one process holding the same devices
    (`ref_devs`, default all of them), both with `env` added to their
    environment (`prog`: the multi-process run's command but its arguments,
    instead of ds_ep_pipeline's): every process's run record (process 0
    writes one, and --json-procs all the others) at full precision.  The last
    chunk's stage's processes have the losses, the same ones, and nobody else
    any; every process holding a stage has the same gradient norms (each
    stage's update clips by them), and one holding none has none.

    `exact` (no lax collective within a stage: FSDP=1, or collective-permutes):
    the first step's loss bit for bit -- the forward, the boundary copies and
    the loss do the same arithmetic whichever process runs them -- and the
    first gradient norm to 1e-5, which a cotangent delivered to the wrong
    microbatch or chunk would move far more.  After an update the two runs
    need not agree to the bit: on the workstations this was written on they
    do, at every step, run after run (jax 0.11.1 and 0.11.2), but on GitHub's
    4-vCPU runner the CI test's second loss differed by 1.4e-5 and its third
    by 3.2e-4 -- Adam's first step is lr * sign(g), so a last-bit difference
    in a near-zero gradient flips a weight's step.  So the later losses to
    2e-3 and the later norms to 2e-2: one update moves TINY's loss by ~5%,
    and a learning rate 40% off on one stage moved the CI test's later
    losses by 2.2e-3 and 4.0e-3 and its second norm by 2.2e-2.

    Not `exact` (lax all-gathers / reduce-scatters within the stages, which
    jax.distributed runs over gloo on CPU): such a run does not even repeat
    itself.  Its forward does, so the first loss to 1e-5, but the first
    gradient norm moved by up to 1.5e-4 in 40 runs of
    test_one_stage_per_process, and at lr 1e-3 the moves grow through the
    updates (Adam's first step is lr * sign(g): a 1-ulp change flips the sign
    of a near-zero g), the later losses by up to 1.1e-4 and the later norms
    by up to 1e-2.  So the first norm and the later losses to 1e-3, the
    later norms to 5e-2 (a stage's update gone wrong, NaN, a wrong chunk).

    The plan: every stage leader logs one agreed hash, the same, and every
    process's summary is the single run's but for device ids and the marks
    on another process's stages.

    `profile`: the multi-process run also profiles a step on each stage's
    leader (the default --profile-procs), which must land in ONE XProf
    session (`_check_one_session`)."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    # the reference is always the native transport (xfer exists only across processes)
    ref_args = [x for i, x in enumerate(args) if x != "--transport" and (i == 0 or args[i - 1] != "--transport")]
    if profile:
      args = args + ["--profile", f"{d}/xprof", "--profile-steps", "1"]
    rc, single = run_single(ref_args + ["--json", f"{d}/single.json"], ref_devs or procs * devs, env=env)
    self.assertEqual(rc, 0, single[-4000:])
    with open(f"{d}/single.json") as f:
      ref = json.load(f)
    ref["_output"] = single
    n = len(ref["warmup_losses"])
    self.assertGreaterEqual(n, 2, single[-2000:])
    margs = args + ["--json", f"{d}/multi.json", "--json-procs", "all"]
    cmd = None if prog is None else [*prog, "--distributed", *margs]
    outs = run_multi(margs, procs, devs, collectives=collectives, env=env, cmd=cmd)
    self._check_ok(outs)
    recs = self._records(f"{d}/multi.json", outs)
    cfg = ref["config"]
    procs_of = {s: [r["process"] for r in recs if s in r["stages"]] for s in range(cfg["stages"])}
    self.assertTrue(all(procs_of.values()), procs_of)
    last = sched.placement(cfg["stages"], cfg["vv"], cfg["placement"])[-1]
    losses, gnorms = recs[procs_of[last][0]]["warmup_losses"], recs[procs_of[0][0]]["warmup_gnorms"]
    self.assertEqual(len(losses), n)
    for r in recs:
      self.assertEqual(r["warmup_losses"], losses if last in r["stages"] else [], f'process {r["process"]}')
      if r["stages"]:
        np.testing.assert_allclose(r["warmup_gnorms"], gnorms, rtol=1e-6, err_msg=f'process {r["process"]}')
      else:
        self.assertEqual(r["warmup_gnorms"], [None] * n)
    if exact:
      self.assertEqual(losses[0], ref["warmup_losses"][0])
      np.testing.assert_allclose(gnorms[0], ref["warmup_gnorms"][0], rtol=1e-5)
      np.testing.assert_allclose(losses, ref["warmup_losses"], rtol=2e-3)
      np.testing.assert_allclose(gnorms, ref["warmup_gnorms"], rtol=2e-2)
    else:
      np.testing.assert_allclose(losses[0], ref["warmup_losses"][0], rtol=1e-5)
      np.testing.assert_allclose(gnorms[0], ref["warmup_gnorms"][0], rtol=1e-3)
      np.testing.assert_allclose(losses, ref["warmup_losses"], rtol=1e-3)
      np.testing.assert_allclose(gnorms, ref["warmup_gnorms"], rtol=5e-2)
    leaders = sorted({min(q) for q in procs_of.values()})
    agreed = [re.findall(r"plan ([0-9a-f]{12}) agreed", outs[q][1]) for q in leaders]
    self.assertTrue(all(len(h) == 1 for h in agreed) and len({h[0] for h in agreed}) == 1, agreed)
    for r in recs:
      self.assertEqual(_plan_text(r["summary"]), _plan_text(ref["summary"]), f'process {r["process"]}')
    if profile:
      self._check_one_session(f"{d}/xprof", procs_of)
    return outs, ref

  def _check_one_session(self, logdir, procs_of):
    """One XProf session: a file per stage leader, s<stage>-p<process>-<host>,
    every one rebased onto one origin.  The origin is process 0's clock 10 ms
    before the profiled steps (shared_profile_session), so every rebased line
    starts at least 5 ms after it; a capture left unshifted starts at ~0."""
    from maxtext.experimental.mpmd.jax import xprof_align

    sessions = os.listdir(f"{logdir}/plugins/profile")
    self.assertEqual(len(sessions), 1, sessions)
    files = sorted(os.listdir(f"{logdir}/plugins/profile/{sessions[0]}"))
    self.assertEqual(
        sorted("-".join(f.split("-")[:2]) for f in files), sorted(f"s{s}-p{min(q)}" for s, q in procs_of.items()), files
    )
    spaces = [xprof_align._load(f"{logdir}/plugins/profile/{sessions[0]}/{f}") for f in files]
    self.assertEqual(len({xprof_align.start_time_ns(sp) for sp in spaces}), 1)  # one origin
    for f, sp in zip(files, spaces):
      ts = [ln.timestamp_ns for pl in sp.planes for ln in pl.lines]
      self.assertTrue(ts, f)
      self.assertGreaterEqual(min(ts), 5_000_000, f)

  @pytest.mark.scheduled_only
  def test_cross_process_channels(self):
    """transfer.py's three cross-process channels (`channel_check`) on four
    processes of two devices: native copies with bystanders, collective-permute
    programs over gloo, and jax.experimental.transfer pulls.  No model: a few
    seconds."""
    outs = run_multi(
        None,
        procs=4,
        devs=2,
        timeout=180,
        collectives="gloo",
        cmd=[sys.executable, "-c", "from tests.unit.experimental.mpmd.jax import multihost_test as m; m.channel_check()"],
    )
    self._check_ok(outs)
    for i, (_, o) in enumerate(outs):
      self.assertIn("channels ok", o, f"process {i}:\n{o[-3000:]}")

  @pytest.mark.scheduled_only
  def test_bitwise_without_stage_collectives(self):
    """PP=4 x FSDP=1, one device per process: nothing but boundary copies
    crosses processes (each with two bystander processes), so the first
    step is the single-process one bit for bit (`_compare`).  Under
    --layout physical_origin_last stage s runs on process s + 1 mod 4, so
    nothing may take a stage for its process.  Kept cheap for CI: one chunk
    per stage, three steps; and the stage leaders' profile, one XProf session."""
    args = (
        TINY
        + (
            "--stages 4 --vv 1 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 --schedule search "
            "--layout physical_origin_last"
        ).split()
    )
    self._compare(args, procs=4, devs=1, profile=True)

  @pytest.mark.scheduled_only
  def test_qwen_bitwise_without_stage_collectives(self):
    """The same on the Qwen3.5 family (Q_TINY): four one-layer chunks, three
    Gated DeltaNet ones and the gated-attention one with the head, each on a
    process of its own (stage s on process s + 1 mod 4) -- every process
    builds and inits its own stage's Qwen programs and weights, and only the
    boundary copies cross processes, so the first step is the single-process
    one bit for bit (`_compare`); the summary pins which model ran, which
    the comparison does not.  No profile: test_bitwise_without_stage_collectives
    holds that."""
    args = (
        Q_TINY
        + (
            "--stages 4 --vv 1 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 --schedule search "
            "--layout physical_origin_last"
        ).split()
    )
    _, ref = self._compare(args, procs=4, devs=1)
    self.assertIn("MPMD Qwen3.5 family: 4 MoE layers (3 Gated DeltaNet, 1 gated attention), 4 chunks", ref["summary"])

  @pytest.mark.scheduled_only
  def test_zero_bubble(self):
    """zbh2 on four processes of one device, a stage each, under --layout physical_origin_last (stage s on process
    s + 1 mod 4), precompiled: the two middle chunks' backwards split into I and W -- an I files its cotangent
    upstream, across processes, as a B does; a W copies nothing -- and every process builds and agrees one plan, the
    order of I and W ops in it, and builds every split chunk's halves (the W buffers in every process's summary); a
    stage's I and W programs are its processes' alike (check_program_agreement).  The first step is the
    single-process one bit for bit (`_compare`)."""
    args = [x for x in TINY if x != "--no-precompile"] + (
        "--stages 4 --vv 1 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 --schedule zbh2 --layout physical_origin_last"
    ).split()
    outs, ref = self._compare(args, procs=4, devs=1)
    self.assertIn("split backward (I / W) on chunks [1, 2]", ref["summary"])
    self.assertEqual(ref["zb"]["split_chunks"], [1, 2])
    split = [o for _, o in outs if re.search(r"\bbwdi_c[12]\b", o)]  # the split stages' program prints
    self.assertEqual(len(split), 2, [o[-2000:] for _, o in outs])
    self.assertTrue(
        all(re.search(r"precompiled \d+ programs in \S+, each the same on all its processes", o) for o in split)
    )

  @pytest.mark.scheduled_only
  def test_zero_bubble_split_last(self):
    """zbh1 with the last chunk split too (--zb-split-last on), four processes of one device, precompiled: the last
    chunk's AI files its cotangent across processes as its fused program does, its W copies nothing, and its
    programs -- fbi / fbai and its W's -- are its process's as every process plans them; the first step the
    single-process one bit for bit (`_compare`)."""
    args = [x for x in TINY if x != "--no-precompile"] + (
        "--stages 4 --vv 1 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 --schedule zbh1 --zb-split-last on"
    ).split()
    outs, ref = self._compare(args, procs=4, devs=1)
    self.assertIn("split backward (I / W) on chunks [1, 2, 3]", ref["summary"])
    self.assertEqual((ref["zb"]["split_chunks"], ref["zb"]["split_last"]), ([1, 2, 3], True))
    last = [o for _, o in outs if re.search(r"\bfbai_c3\b", o)]  # the last stage's program prints
    self.assertEqual(len(last), 1, [o[-2000:] for _, o in outs])
    self.assertRegex(last[0], r"precompiled \d+ programs in \S+, each the same on all its processes")

  @pytest.mark.scheduled_only
  def test_zero_bubble_paced(self):
    """zbh1 with the last chunk split, four processes of one device, its I's and AI's paced as on a TPU VM's own
    client (Pipeline.pace_i, forced on: `PACED_MAIN`): each process waits only on the W buffers of its own stage --
    every split stage's process does, stage 0's (unsplit) does not -- and the run is the unpaced single-process one,
    bit for bit in the first step (`_compare`)."""
    args = [x for x in TINY if x != "--no-precompile"] + (
        "--stages 4 --vv 1 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 --schedule zbh1 --zb-split-last on"
    ).split()
    outs, ref = self._compare(args, procs=4, devs=1, prog=[sys.executable, "-c", PACED_MAIN])
    self.assertEqual(ref["zb"]["paced"], False)  # the reference: one CPU process, unpaced
    paced = [{int(s) for s in re.findall(r"^paced c\d+ m\d+ s(\d+)$", o, re.M)} for _, o in outs]
    self.assertEqual(sorted(map(sorted, paced)), [[], [1], [2], [3]], [o[-2000:] for _, o in outs])

  @pytest.mark.scheduled_only
  def test_qwen_zero_bubble(self):
    """zbh1 on the Qwen3.5 family (Q_TINY), four processes of a one-layer chunk each: the Gated DeltaNet chunks 1
    and 2 split, and the plan agreed -- against one process, bit for bit in the first step (`_compare`)."""
    args = Q_TINY + "--stages 4 --vv 1 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 --schedule zbh1".split()
    _, ref = self._compare(args, procs=4, devs=1)
    self.assertIn("split backward (I / W) on chunks [1, 2]", ref["summary"])

  def test_q_tiny_gives_every_model_flag(self):
    """Q_TINY gives every flag --model qwen3.8-2.4t-a95b presets
    (ds_ep_pipeline.QWEN_MODELS) but --layers, which each test gives: the
    dims and the rotary embedding, so --model picks only the family and a
    change to its presets changes no test here.  A parse: no process."""
    from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp

    given = {act.dest for act in dsp.make_parser()._actions if set(act.option_strings) & set(Q_TINY)}
    self.assertEqual(set(dsp.QWEN_MODELS["qwen3.8-2.4t-a95b"]) - given, {"layers"})

  @pytest.mark.scheduled_only
  def test_flag_warning_is_tagged(self):
    """main names a flag given where it does nothing (--batch-split-impl without --batch-split) on process 0
    through its log, so a multi-process run tags the line `[s<k> p<i>] ` like every other, and a watcher of the
    logs that drops untagged lines keeps it.  A dry run, no compile: seconds."""
    args = (
        TINY + "--stages 2 --vv 1 --fsdp 1 --ep 1 --gbs 4 --mb 2 --layers 2 --batch-split-impl native --dry-run".split()
    )
    outs = run_multi(args, procs=2, devs=1, timeout=180)
    self._check_ok(outs)
    want = "warning: --batch-split-impl native: not used without --batch-split"
    lines = [ln for ln in outs[0][1].splitlines() if want in ln]
    self.assertTrue(lines, outs[0][1][-3000:])
    for ln in lines:
      self.assertRegex(ln, r"^\[s\d+ p0\] ")
    self.assertNotIn(want, outs[1][1])

  @pytest.mark.scheduled_only
  @unittest.skipUnless(shutil.which("bash") and shutil.which("timeout"), "needs bash and timeout")
  def test_multihost_launcher(self):
    """ds_ep_multihost.sh itself on two processes of one device, each a host whose environment names its worker
    (MAXTEXT_MPMD_WORKER_ID), in a dry run (no compile: seconds).  Each host writes its log and run record in a directory
    of its own, so the hosts' --json paths differ, and the plan check leaves --json out: the two agree on one plan.
    Each log lands in RUN_DIR under its host's id, and stays, alone, in its host's directory."""
    testing.require_multihost()
    d = self.enterContext(tempfile.TemporaryDirectory())
    os.makedirs(f"{d}/tmp")
    script = os.path.join(testing.package_dir(), "ds_ep_multihost.sh")
    args = TINY + "--stages 2 --vv 1 --fsdp 1 --ep 1 --gbs 4 --mb 2 --layers 2 --dry-run".split()
    cmd = ["bash", "-c", 'MAXTEXT_MPMD_WORKER_ID=$JAX_PROCESS_ID exec bash "$0" "$@"', script, *args]
    # TIMEOUT: at its deadline the harness kills the launcher, and the launcher's timeout the run
    env = dict(
        PYTHON=sys.executable,
        RUN_DIR=f"{d}/runs",
        STEP_TAG="t",
        TMPDIR=f"{d}/tmp",
        JAX_CACHE_DIR=f"{d}/cache",
        TIMEOUT="170",
    )
    # and none of what the launcher reads may come from the environment the tests run in (a multislice pod's
    # MEGASCALE_*, an agent's WID, a MODULE, a GPU platform): it reads each with `:-`, so empty is unset (the flag
    # set's LIBTPU_INIT_ARGS is not read on CPU)
    env.update(
        dict.fromkeys(
            "MEGASCALE_SLICE_ID MEGASCALE_NUM_SLICES TPU_WORKER_ID TPU_WORKER_HOSTNAMES "
            "JOB_COMPLETION_INDEX MAXTEXT_MPMD_SLICE_ID MAXTEXT_MPMD_NUM_SLICES MAXTEXT_MPMD_HOSTS_PER_SLICE WID "
            "MODULE PLATFORM MPMD_FLAG_SET EXTRA_LIBTPU_INIT_ARGS".split(),
            "",
        ),
        TPU_SKIP_MDS_QUERY="1",
    )
    outs = run_multi(None, procs=2, devs=1, timeout=180, cmd=cmd, env=env)
    self._check_ok(outs)
    agreed = [re.findall(r"plan ([0-9a-f]{12}) agreed", o) for _, o in outs]
    self.assertTrue(all(len(h) == 1 for h in agreed) and len({h[0] for h in agreed}) == 1, agreed)
    for i in range(2):
      with open(f"{d}/runs/t/logs/mpmd_t_host{i}.log") as f:
        log = f.read().splitlines()
      self.assertRegex(log[0], rf"^host {i} \([^)]+\): maxtext\.experimental\.mpmd\.jax\.ds_ep_pipeline --distributed ")
      self.assertEqual(log[-1], f"host {i} exit 0")
    self.assertEqual(
        sorted(sorted(os.listdir(os.path.join(d, "tmp", w))) for w in os.listdir(f"{d}/tmp")),
        [["mpmd_t_host0.log"], ["mpmd_t_host1.log"]],
    )

  @pytest.mark.scheduled_only
  def test_two_chunks_per_stage(self):
    """The same with V=2 circular chunks: copies from a stage to the one after
    it and from the last stage back to the first."""
    args = TINY + "--stages 4 --vv 2 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 8 --schedule search".split()
    self._compare(args, procs=4, devs=1)

  @pytest.mark.scheduled_only
  def test_one_stage_per_process(self):
    """PP=4 x FSDP=4 on 4 processes of 4 devices: every boundary crosses processes."""
    args = TINY + "--stages 4 --vv 2 --fsdp 4 --ep 1 --gbs 16 --mb 4 --layers 8 --schedule search".split()
    self._compare(args, procs=4, devs=4, exact=False)

  @pytest.mark.scheduled_only
  def test_two_device_stages(self):
    """PP=4 x FSDP=2 on 4 processes of 2 devices: every boundary copy moves
    two shards across processes.  The stages' FSDP collectives are
    collective-permutes (the default at FSDP=2), which jax.distributed runs
    repeatably, so the first step is the single-process one bit for bit (on
    the workstations every step is) -- with lax all-gathers / reduce-scatters
    (--fsdp-collectives xla) the first gradient norm moved by up to 2e-3 from
    run to run."""
    args = TINY + "--stages 4 --vv 2 --fsdp 2 --ep 1 --gbs 16 --mb 4 --layers 8 --schedule search".split()
    self._compare(args, procs=4, devs=2)

  @pytest.mark.scheduled_only
  def test_v_placement_stash(self):
    """V placement (the turn stays on a stage) with stash modes, PP=2 on 2 processes."""
    args = (
        TINY
        + (
            "--stages 2 --vv 2 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 --placement v "
            "--schedule stash --stash lite --stash-budget-gib 0.01"
        ).split()
    )
    self._compare(args, procs=2, devs=1)

  @pytest.mark.scheduled_only
  def test_interleaved_pairs(self):
    """Interleaved 1F1B's steady-state F&B pairs (Megatron-LM's combined 1F1B) on circular placement, every
    boundary crossing processes both ways."""
    args = (
        TINY
        + (
            "--stages 2 --vv 2 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 --schedule interleaved " "--pair-stages all"
        ).split()
    )
    _, ref = self._compare(args, procs=2, devs=1)
    self.assertIn("interleaved (Megatron) pairs 6 (every stage", ref["summary"])

  @pytest.mark.scheduled_only
  def test_ppermute_transport(self):
    """Boundary copies as collective-permute programs over the two stages'
    devices (--transport ppermute; gloo carries them on CPU): pure data
    movement, so the first step bit for bit the single-process run.  Precompiled (the
    other tests pass --no-precompile): each process compiles its own stages'
    programs and the copy programs of which it holds either side."""
    args = [x for x in TINY if x != "--no-precompile"] + (
        "--stages 4 --vv 2 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 8 --schedule search " "--transport ppermute"
    ).split()
    outs, ref = self._compare(args, procs=4, devs=1, collectives="gloo")
    # Each process compiles its own stage's programs (the single run: all of them, no copy programs)
    # and every copy program one of whose stages it holds: the activation copies between the stages
    # the placement links (circular: s, s+1 both ways) and the gradient-norm hops between any two.
    counts = [int(m.group(1)) for m in (re.search(r"precompiled (\d+) programs", o) for _, o in outs) if m]
    self.assertEqual(len(counts), 4, counts)
    single = int(re.search(r"precompiled (\d+) programs", ref["_output"]).group(1))
    copies = 2 * 4 + 4 * 3
    self.assertEqual(sum(counts), single + 2 * copies, counts)  # each copy program on both its sides
    self.assertLess(max(counts), single, counts)

  @staticmethod
  def _slices_args(fsdp, ep):
    """`_one_stage_per_slice`'s pipeline, its transport left out."""
    return [x for x in TINY if x != "--no-precompile"] + (
        f"--stages 2 --vv 2 --fsdp {fsdp} --ep {ep} --gbs {4 * fsdp * ep} --mb 4 --layers 4 --schedule search "
        "--layout slices"
    ).split()

  def _one_stage_per_slice(self, devs, fsdp, ep, exact, transport="ppermute", extra=()):
    """--layout slices on two fake slices (MPMD_FAKE_SLICES=2) of two
    processes of `devs` devices: each stage all of one slice, over its
    slice's two processes, and every boundary copy -- the activations between
    the slices, the gradient norm's scalars -- a collective-permute program
    over both slices (--transport ppermute, the one copy known to cross TPU
    slices), or the host relay's (`transport` 'relay': no program).  The
    single-process reference cuts its devices into the same two slices.
    Precompiled: each process compiles its own stage's programs and all four
    copy programs (it holds a side of each); the stage leaders -- processes 0
    and 2 -- say how many.  `extra`: more flags for both runs.  -> (outs, ref)
    as `_compare`."""
    args = self._slices_args(fsdp, ep) + ["--transport", transport, *extra]
    outs, ref = self._compare(args, procs=4, devs=devs, collectives="gloo", exact=exact, env={"MPMD_FAKE_SLICES": "2"})
    self.assertIn("stage leaders {0: 0, 1: 2}", outs[0][1])  # slice 0: processes 0, 1; slice 1: 2, 3
    counts = [int(m.group(1)) for m in (re.search(r"precompiled (\d+) programs", o) for _, o in outs) if m]
    single = int(re.search(r"precompiled (\d+) programs", ref["_output"]).group(1))
    self.assertEqual(len(counts), 2, counts)
    copies = 4 if transport == "ppermute" else 0
    self.assertEqual(sum(counts), single + 2 * copies, counts)  # each stage's programs, and the copies each
    return outs, ref

  @pytest.mark.scheduled_only
  def test_one_stage_per_slice(self):
    """Two-device stages, EP 2 across each slice's two processes: the
    ring-of-experts exchange, the gradients' reduce-scatter and every
    boundary copy collective-permutes, and every sum over a stage two
    partials (which add alike in either order), so the first step is the
    single-process run's bit for bit (`_one_stage_per_slice`)."""
    self._one_stage_per_slice(devs=1, fsdp=1, ep=2, exact=True)

  @pytest.mark.scheduled_only
  def test_one_stage_per_slice_fsdp_x_ep(self):
    """FSDP 2 x EP 2 stages of two processes of two devices: collective-
    permutes again, but the loss and the gradient norm sum four devices'
    partials across the stage's two processes (over gloo), in another order
    than one process does -- the first loss came out one float32 ulp off --
    so not `exact` (`_compare`)."""
    self._one_stage_per_slice(devs=2, fsdp=2, ep=2, exact=False)

  @pytest.mark.scheduled_only
  def test_one_stage_per_slice_explicit_relay(self):
    """One stage per slice, typed, over the host relay, as the multi-slice
    DeepSeek-V3 runs had them, on CPU: FSDP 2 x EP 2 stages, one on each of
    two fake slices of two processes, typed
    (--shard-mode explicit: Explicit stage meshes over two processes, the
    gathered weights reduced, the gradient accumulators unreduced), with the
    host relay between them (--transport relay) -- against the single process
    (`_one_stage_per_slice`; not `exact`: the sums over a stage cross processes
    over gloo).  That single-process explicit run is auto's bit for bit, every
    warmup loss and gradient norm."""
    outs, ref = self._one_stage_per_slice(
        devs=2, fsdp=2, ep=2, exact=False, transport="relay", extra=("--shard-mode", "explicit")
    )
    self.assertIn("shard_mode=explicit", ref["summary"])
    d = self.enterContext(tempfile.TemporaryDirectory())
    rc, out = run_single(self._slices_args(2, 2) + ["--json", f"{d}/auto.json"], 8, env={"MPMD_FAKE_SLICES": "2"})
    self.assertEqual(rc, 0, out[-4000:])
    with open(f"{d}/auto.json") as f:
      auto = json.load(f)
    self.assertNotIn("shard_mode=explicit", auto["summary"])
    self.assertEqual((ref["warmup_losses"], ref["warmup_gnorms"]), (auto["warmup_losses"], auto["warmup_gnorms"]))

  @testing.slow
  def test_repeat_scope_four_slices_relay(self):
    """The DeepSeek-V3 TPU7x target's pipeline shape on CPU: four stages,
    one on each of four fake slices (one process of two devices, EP 2), the
    host relay between them, the breadth-first order at V=2, and every
    chunk's weights gathered per visit (--fsdp-scope repeat) -- against the
    single process (`exact`: no lax collective crosses processes); and that
    single process with --fsdp-scope step, every warmup loss and gradient
    norm bit for bit."""
    args = [x for x in TINY if x != "--no-precompile"] + (
        "--stages 4 --vv 2 --fsdp 1 --ep 2 --gbs 8 --mb 4 --layers 8 --schedule breadth --fsdp-scope repeat "
        "--layout slices --transport relay"
    ).split()
    env = {"MPMD_FAKE_SLICES": "4"}
    outs, ref = self._compare(args, procs=4, devs=2, collectives="gloo", env=env)
    self.assertIn("fsdp_scope=repeat", ref["summary"])
    self.assertIn("released after", ref["summary"])
    d = self.enterContext(tempfile.TemporaryDirectory())
    step = [x for x in args if x not in ("relay", "--transport")] + ["--fsdp-scope", "step", "--json", f"{d}/step.json"]
    rc, out = run_single(step, 8, env=env)
    self.assertEqual(rc, 0, out[-4000:])
    with open(f"{d}/step.json") as f:
      rec = json.load(f)
    self.assertNotIn("fsdp_scope", rec["summary"])
    self.assertEqual((ref["warmup_losses"], ref["warmup_gnorms"]), (rec["warmup_losses"], rec["warmup_gnorms"]))

  @testing.slow
  def test_qwen_repeat_scope_four_slices_relay(self):
    """The same on the Qwen3.5 family (Q_TINY), the Qwen3.8 TPU7x target's
    pipeline shape on CPU -- a stage a fake slice, the host relay between
    them, breadth at V=2 with weights per visit -- over chunks cut unevenly as
    the target's are (2,2,2,1,1,1,1,2 layers): stage 0 holds a Gated DeltaNet
    chunk and gated-attention chunk 4, stage 1 a chunk of both kinds, stage 3
    the head's, of both.  Against the single process (`exact`), and that
    single process with --fsdp-scope step, every warmup loss and gradient
    norm bit for bit."""
    args = [x for x in Q_TINY if x != "--no-precompile"] + (
        "--stages 4 --vv 2 --fsdp 1 --ep 2 --gbs 8 --mb 4 --layers 12 --layer-split 2,2,2,1,1,1,1,2 "
        "--schedule breadth --fsdp-scope repeat --layout slices --transport relay"
    ).split()
    env = {"MPMD_FAKE_SLICES": "4"}
    _, ref = self._compare(args, procs=4, devs=2, collectives="gloo", env=env)
    self.assertIn("MPMD Qwen3.5 family: 12 MoE layers (9 Gated DeltaNet, 3 gated attention), 8 chunks", ref["summary"])
    self.assertIn("fsdp_scope=repeat", ref["summary"])
    d = self.enterContext(tempfile.TemporaryDirectory())
    step = [x for x in args if x not in ("relay", "--transport")] + ["--fsdp-scope", "step", "--json", f"{d}/step.json"]
    rc, out = run_single(step, 8, env=env)
    self.assertEqual(rc, 0, out[-4000:])
    with open(f"{d}/step.json") as f:
      rec = json.load(f)
    self.assertNotIn("fsdp_scope", rec["summary"])
    self.assertEqual((ref["warmup_losses"], ref["warmup_gnorms"]), (rec["warmup_losses"], rec["warmup_gnorms"]))

  @pytest.mark.scheduled_only
  def test_relay_timing_windows(self):
    """--bench and --profile over the relay: each process logs (and records) its relay timing in one window per
    kind of step -- warm-up (3), run-ahead (--steps 2, the benchmarked steady state), blocked (3), profiled (1) --
    each taken once every process is past the window's steps, so a window holds its own steps' transfers and no
    other's: every step sends and receives alike, so a window's counts are its steps' count times a step's."""
    args = (
        TINY
        + (
            "--stages 2 --vv 2 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 --schedule search "
            "--transport relay --bench --steps 2"
        ).split()
    )
    outs, _ = self._compare(args, procs=2, devs=1, profile=True)
    windows = {"warmup steps": 3, "the run-ahead steps": 2, "the blocked steps": 3, "the profiled steps": 1}
    for i, (_, o) in enumerate(outs):
      per_step = set()
      for what, steps in windows.items():
        m = re.search(rf"\[s{i} p{i}\] relay timing, {what}: sent (\d+) shards .*; received (\d+) ", o)
        self.assertTrue(m, f"process {i}, {what}:\n{o[-3000:]}")
        sent, received = int(m[1]), int(m[2])
        self.assertEqual((sent % steps, received % steps), (0, 0), f"process {i}, {what}: {m[0]}")
        per_step.add((sent // steps, received // steps))
      self.assertEqual(len(per_step), 1, f"process {i}: {per_step}")
      self.assertNotIn("the steps after warm-up", o)

  @pytest.mark.scheduled_only
  def test_xfer_transport(self):
    """Boundary copies as jax.experimental.transfer pulls between host pairs
    (--transport xfer, async; only the two hosts of a copy act), V placement
    with stash modes: the first step bit for bit the single-process run."""
    args = (
        TINY
        + (
            "--stages 4 --vv 2 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 8 --placement v "
            "--schedule stash --stash lite --stash-budget-gib 0.01 --transport xfer"
        ).split()
    )
    self._compare(args, procs=4, devs=1)

  @pytest.mark.scheduled_only
  def test_init_from_files_and_token_data(self):
    """--init-from DIR and --data npy:FILE, the inputs of the MaxText parity
    runs: the run is the single-process one given the same files, bit for bit
    (first step).  What it trains is the files' weights and tokens: the
    default init, or the default batch, gives another loss."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    args = TINY + ("--stages 4 --vv 1 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 --schedule search " "--warmup 2").split()
    write_input_files(d, args, 4)
    inputs = {"--init-from": d, "--data": f"npy:{d}/tokens.npy"}
    outs, ref = self._compare(args + [x for kv in inputs.items() for x in kv], procs=4, devs=1)
    self.assertTrue(any("weights from" in o for _, o in outs))
    self.assertTrue(any("data: npy:" in o for _, o in outs))
    for flag in inputs:  # one of the two inputs left at its default
      rest = [x for k, v in inputs.items() if k != flag for x in (k, v)]
      rc, out = run_single(args + rest + ["--json", f"{d}/without{flag}.json"], 4)
      self.assertEqual(rc, 0, out[-2000:])
      with open(f"{d}/without{flag}.json") as f:
        self.assertNotEqual(json.load(f)["warmup_losses"][0], ref["warmup_losses"][0], f"without {flag}")

  @pytest.mark.scheduled_only
  def test_stage_spans_two_processes(self):
    """PP=2 x FSDP=4 on 4 processes of 2 devices: each stage spans two hosts
    (its programs' collectives cross processes too, over gloo).  Only each
    stage's leader profiles (the default --profile-procs)."""
    args = TINY + "--stages 2 --vv 2 --fsdp 4 --ep 1 --gbs 16 --mb 4 --layers 4".split()
    self._compare(args, procs=4, devs=2, collectives="gloo", exact=False, profile=True)

  @pytest.mark.scheduled_only
  def test_process_holding_no_stage(self):
    """PP=2 on 3 processes of one device: process 2 holds no stage.  It runs
    nothing but walks every boundary copy, says so and exits 0, and the run
    is the single-process one on the 2 devices the stages hold."""
    args = TINY + "--stages 2 --vv 2 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 --schedule search".split()
    outs, _ = self._compare(args, procs=3, devs=1, ref_devs=2)
    self.assertIn("[idle p2] this process holds no stage", outs[2][1])
    self.assertRegex(outs[0][1], r"\b1 process(es)? holds? no stage")

  @pytest.mark.scheduled_only
  def test_plan_mismatch_stops_every_process(self):
    """Processes given different flags (one more --pf-split here) build
    different plans: every one of them stops at the plan check, naming both
    -- none hangs on a copy the others never issue."""
    args = TINY + "--stages 2 --vv 1 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 2 --schedule gpipe".split()
    outs = run_multi(args, procs=2, devs=1, timeout=60, extra_args={1: ["--pf-split", "1"]})
    for i, (rc, o) in enumerate(outs):
      self.assertGreater(rc, 0, f"process {i}:\n{o[-3000:]}")  # an error, not killed at the deadline
      err = re.search(r"the processes built 2 different plans \((.*)\)", o)
      self.assertTrue(err, f"process {i}:\n{o[-3000:]}")
      self.assertEqual(sorted(re.findall(r"processes \[(\d+)\]", err[1])), ["0", "1"], err[0])

  @pytest.mark.scheduled_only
  def test_missing_input_stops_every_process(self):
    """An --init-from file only some processes lack (each checks its stages'
    chunks, and a directory local to each host can differ): stage 1's chunk
    here.  Every process stops before compiling, each naming the process and
    the file; none waits for the others until the coordination service aborts
    them all, five minutes later, with only process 1 saying why."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    open(f"{d}/chunk0.npz", "wb").close()  # stage 0's: there, and never read (every process stops first)
    args = TINY + "--stages 2 --vv 1 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 2 --schedule gpipe".split()
    outs = run_multi(args + ["--init-from", d], procs=2, devs=1, timeout=120)
    for i, (rc, o) in enumerate(outs):
      self.assertGreater(rc, 0, f"process {i}:\n{o[-3000:]}")  # an error, not killed at the deadline
      self.assertIn(
          f"input files missing on 1 of 2 processes, so every process stops: --init-from {d}: no such file "
          f"{d}/chunk1.npz (process 1)",
          o,
      )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--procs", type=int, default=4)
  ap.add_argument("--devs", type=int, default=4)
  ap.add_argument("--collectives", default=None)
  ap.add_argument("rest", nargs=argparse.REMAINDER)
  a = ap.parse_args()
  rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
  for i, (rc, o) in enumerate(run_multi(TINY + rest, a.procs, a.devs, collectives=a.collectives)):
    print(f"===== process {i}: exit {rc}\n{o}")


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
    main()
  else:
    unittest.main()
