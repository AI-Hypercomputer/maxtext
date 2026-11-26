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

"""Verification of the cross-mesh transfer layer on 4 CPU devices (or 8): two
disjoint sub-meshes of half of them each, exact values across a batched D2D
copy, the index-map predicate, the reshard guard, and the no-D2H CI guard.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        python3 -m pytest -q tests/unit/experimental/mpmd/jax/transfer_test.py

One test runs it all (`check_transfer_layer`), section after section -- each
builds on the arrays and channels of the ones before it -- printing what each
verified.  A second makes the batched copy raise at send: the channel's
fallback to jax.device_put per leaf.

The three REGRESSION sections at the end pin the defects an adversarial review
found: (1) `send` used to hand mislabelled arrays to the batched copy and get
silently wrong numbers, (2) `strict=True` used to accept a resharding boundary
whenever the leaf was slow for an earlier reason, (3) the `set_max_inflight`
"proof" used to be satisfied by a garbage option name and a garbage value."""

import ast
import inspect
import subprocess
import sys
import time
import timeit
import unittest
from unittest import mock
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxlib import xla_client as xc

from maxtext.experimental.mpmd.jax import testing
from maxtext.experimental.mpmd.jax import transfer
from maxtext.experimental.mpmd.jax.transfer import (
    BATCHED_COPY_AVAILABLE,
    Channel,
    HostBounceError,
    MAX_INFLIGHT_OPTION,
    OPTION_AWARE_PLATFORMS,
    ShardingMismatchError,
    assert_d2d,
    block_all,
    no_host_bounce,
    option_name_evidence,
    pinned_platforms,
    same_index_map,
    set_max_inflight,
    smoke_test,
)

# The directory the package imports from (testing.import_root): the probe's child process runs there, with it on
# its PYTHONPATH.
ROOT = testing.import_root()


def setUpModule():
  testing.require_cpu()


def bench_all(fns, n=300, rounds=5, warmup=20):
  """Min over `rounds` of the mean of `n` calls, with the variants INTERLEAVED.

  The guard costs a fraction of a microsecond per leaf and this box is shared;
  measuring the variants one after the other lets drift swamp the difference
  (it can even come out negative), so every round measures every variant.
  """
  for fn in fns:
    for _ in range(warmup):
      fn()
  best = [float("inf")] * len(fns)
  for _ in range(rounds):
    for j, fn in enumerate(fns):
      keep = []
      t0 = time.perf_counter()
      for _ in range(n):
        keep.append(fn())
      best[j] = min(best[j], (time.perf_counter() - t0) / n)
      block_all(keep)
  return best


def check_transfer_layer():
  """The checks the module docstring lists, section by section, printing what each verified."""
  devs = jax.devices()
  assert len(devs) >= 4, f"need 4 CPU devices or more, got {len(devs)} (set XLA_FLAGS)"
  half = min(len(devs), 8) // 2
  mA = Mesh(np.array(devs[:half]), ("fsdp",))
  mB = Mesh(np.array(devs[half : 2 * half]), ("fsdp",))
  print(f"batched_copy available: {BATCHED_COPY_AVAILABLE}")
  print(f"meshes: A={[d.id for d in mA.devices]}  B={[d.id for d in mB.devices]}")

  # ------------------------------------------------------------------- smoke
  assert smoke_test() is True
  print("smoke_test: OK (batched_copy_array_to_devices_with_sharding round-trips)")

  # --------------------------------------------------- 2-leaf pytree A -> B
  act_spec, bias_spec = P("fsdp", None), P("fsdp")
  shA = {"y": NamedSharding(mA, act_spec), "b": NamedSharding(mA, bias_spec)}
  shB = {"y": NamedSharding(mB, act_spec), "b": NamedSharding(mB, bias_spec)}
  host = {"y": np.arange(64 * 16, dtype=np.float32).reshape(64, 16), "b": np.arange(64, dtype=np.float32) * -0.5}
  tree = jax.device_put(host, shA)
  avals = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), tree)

  # The dicts themselves are NEVER equal (keyed by Device, disjoint sub-meshes):
  im_src = shA["y"].addressable_devices_indices_map((64, 16))
  im_dst = shB["y"].addressable_devices_indices_map((64, 16))
  assert im_src != im_dst, "dict comparison unexpectedly succeeded"
  assert tuple(im_src.values()) == tuple(im_dst.values())
  print(
      f"index-map dicts equal? {im_src == im_dst}   ordered values equal? "
      f"{tuple(im_src.values()) == tuple(im_dst.values())}  <- the whole point"
  )

  ch = Channel(avals, shA, shB, name="s0->s1")
  print(ch, "reasons=", ch.reasons)
  assert ch.n_fast == 2 and ch.n_slow == 0, (ch.n_fast, ch.n_slow, ch.reasons)
  assert ch.check() is True
  out = ch.send(tree)
  block_all(out)
  for k in host:
    assert out[k].sharding == shB[k], (k, out[k].sharding)
    assert np.array_equal(np.asarray(out[k]), host[k]), k
    assert tree[k].sharding == shA[k], "source resharded in place"
  print("A->B 2-leaf pytree: values exact, n_fast=2, n_slow=0, dst sharding on B")

  # source arrays are still alive and usable after the copy (REUSE_INPUT)
  assert float(jnp.sum(tree["y"])) == float(host["y"].sum())
  print("sources still live after send (REUSE_INPUT, never DONATE_INPUT)")

  # ------------------------------------------------- predicate + assert_d2d
  transposed = NamedSharding(mB, P(None, "fsdp"))
  assert same_index_map(shA["y"], shB["y"], (64, 16)) is True
  assert same_index_map(shA["y"], transposed, (64, 16)) is False
  assert same_index_map(shA["y"], NamedSharding(mB, P()), (64, 16)) is False
  print("same_index_map: matched=True  transposed=False  replicated=False")

  try:
    assert_d2d(tree["y"], transposed, name="s0->s1[y]")
    raise SystemExit("FAIL: assert_d2d did not raise for a transposed spec")
  except ShardingMismatchError as e:
    msg = str(e)
    assert "P('fsdp', None)" in msg and "P(None, 'fsdp')" in msg, msg
    assert "->" in msg
    print("assert_d2d raised as expected:")
    print("  | " + "\n  | ".join(msg.splitlines()))

  bad = Channel(avals["y"], shA["y"], transposed, name="bad")
  assert bad.n_fast == 0 and bad.n_slow == 1 and bad.reasons[0].startswith("reshard")
  assert bad.reshards == [True]
  try:
    Channel(avals["y"], shA["y"], transposed, name="bad-strict", strict=True)
    raise SystemExit("FAIL: strict Channel did not raise")
  except ShardingMismatchError:
    print(
        "Channel(strict=True) rejects a resharding boundary; non-strict "
        f"degrades that leaf to device_put ({bad.n_slow} slow leaf)"
    )

  # --------------------------------------------- extended dtype (PRNG keys)
  keys = jax.device_put(jax.random.split(jax.random.key(0), 8), NamedSharding(mA, P("fsdp")))
  kch = Channel(jax.ShapeDtypeStruct(keys.shape, keys.dtype), keys.sharding, NamedSharding(mB, P("fsdp")), name="keys")
  assert kch.n_fast == 0 and kch.n_slow == 1 and "extended dtype" in kch.reasons[0]
  assert kch.reshards == [False]
  kout = block_all(kch.send(keys))
  assert kout.sharding == NamedSharding(mB, P("fsdp"))
  assert np.array_equal(np.asarray(jax.random.key_data(kout)), np.asarray(jax.random.key_data(keys)))
  print(f"PRNG key leaf: {kch.reasons[0]} -> device_put fallback, values exact")

  # ----------------------------------------------------- no_host_bounce()
  with no_host_bounce():
    try:
      np.asarray(tree["y"])
      raise SystemExit("FAIL: no_host_bounce did not fire on np.asarray")
    except HostBounceError as e:
      fired = str(e)
  print("no_host_bounce fired on np.asarray(sharded device array):")
  print("  | " + fired.splitlines()[0])

  with no_host_bounce():
    legit = ch.send(tree)
    block_all(legit)
  for k in host:  # values checked only after the guard exits
    assert np.array_equal(np.asarray(legit[k]), host[k])
  print("no_host_bounce did NOT fire for a matched-sharding cross-mesh send")

  with no_host_bounce():
    try:
      block_all(jax.device_put(tree["y"], transposed))
      resharded_bounced = False
    except HostBounceError:
      resharded_bounced = True
  print(f"a RESHARDING device_put bounces through the host: {resharded_bounced}")

  try:
    with no_host_bounce():
      raise ValueError("boom")
  except ValueError:
    pass
  assert np.asarray(tree["y"]).sum() == host["y"].sum(), "guard not restored"
  print("no_host_bounce restored ArrayImpl._value after an exception")

  # ======================================================================
  # REGRESSION 1: send() must reject an array that does not carry the
  # sharding the channel was resolved for.  The batched copy does not check,
  # and the result is fast-and-WRONG, not slow-and-correct.
  # ======================================================================
  print()
  print("=== regression 1: send() validates the sharding it is handed ===")
  sA1, sB1 = NamedSharding(mA, act_spec), NamedSharding(mB, act_spec)
  one = Channel(jax.ShapeDtypeStruct((64, 16), np.float32), sA1, sB1, name="r1")
  mislabelled = jax.device_put(host["y"], NamedSharding(mA, P(None, "fsdp")))
  try:
    one.send(mislabelled)
    raise SystemExit("FAIL: send() accepted a mislabelled array")
  except ShardingMismatchError as e:
    m1 = str(e)
  assert "WRONG NUMBERS" in m1 and "P(None, 'fsdp')" in m1 and "P('fsdp', None)" in m1
  assert "actual   src" in m1 and "declared src" in m1
  print("send() raised ShardingMismatchError:")
  print("  | " + "\n  | ".join(m1.splitlines()[:1] + m1.splitlines()[2:3] + m1.splitlines()[5:6]))

  # ...and this is exactly the corruption it prevents: validate=False reproduces
  # the original defect, so the guard is load-bearing, not decoration.
  unchecked = Channel(jax.ShapeDtypeStruct((64, 16), np.float32), sA1, sB1, name="r1-unchecked", validate=False)
  corrupt = unchecked.send(mislabelled)
  assert unchecked.last_fallback_error is None, "expected the fast path, silently"
  claimed = corrupt.sharding.shard_shape(corrupt.shape)
  actual = [s.data.shape for s in corrupt.addressable_shards]
  assert all(a != claimed for a in actual), (claimed, actual)
  W = np.arange(16 * 3, dtype=np.float32).reshape(16, 3)
  matmul = jax.jit(lambda a, w: a @ w)
  got = np.asarray(block_all(matmul(corrupt, jnp.asarray(W))))
  gold = host["y"] @ W
  assert not np.allclose(got, gold), "the corruption no longer reproduces"
  ctrl = np.asarray(block_all(matmul(jax.device_put(mislabelled, sB1), jnp.asarray(W))))
  assert np.allclose(ctrl, gold), "jax.device_put control is wrong?!"
  print(
      f"  validate=False still corrupts: array claims shard {claimed}, holds "
      f"{actual[0]}; matmul row0 {got[0]} != {gold[0]}"
  )
  print("  jax.device_put on the same input stays correct -> the batched copy is " "the unchecked one")

  # a DIFFERENT sharding object with the SAME index map is not a mismatch
  equivalent = jax.device_put(host["y"], NamedSharding(Mesh(np.array(devs[:half]), ("fsdp",)), P("fsdp", None)))
  assert equivalent.sharding is not sA1
  ok = block_all(one.send(equivalent))
  assert np.array_equal(np.asarray(ok), host["y"])
  print("  an equal-but-not-identical src sharding is accepted (no false " "positive)")

  # a host array has no sharding at all: rejected, not silently host-staged
  try:
    one.send(host["y"])
    raise SystemExit("FAIL: send() accepted a numpy array on the fast path")
  except ShardingMismatchError as e:
    assert "no .sharding" in str(e), str(e)
  print("  a host-numpy leaf is rejected (it has no .sharding to copy from)")

  # a different shape that still shards identically on both sides (a short last
  # microbatch) is legal and must NOT be rejected -- verified numerically
  short = jax.device_put(host["y"][:32], sA1)
  short_out = block_all(one.send(short))
  assert np.array_equal(np.asarray(short_out), host["y"][:32])
  assert short_out.sharding == sB1
  print("  a short (32, 16) microbatch through the same channel: values exact")
  # ...but a short array whose sharding no longer matches is still caught
  try:
    one.send(jax.device_put(host["y"][:32], NamedSharding(mA, P(None, "fsdp"))))
    raise SystemExit("FAIL: send() accepted a mislabelled short array")
  except ShardingMismatchError as e:
    assert "actual shape (32, 16)" in str(e), str(e)
  print("  a mislabelled short array is still caught, at its own shape")

  # the guard also fires for one bad leaf inside an otherwise-good pytree
  mixed = {"y": mislabelled, "b": tree["b"]}
  try:
    ch.send(mixed)
    raise SystemExit("FAIL: send() accepted a pytree with one mislabelled leaf")
  except ShardingMismatchError as e:
    assert "s0->s1[leaf" in str(e), str(e)
  print("  one bad leaf in a 2-leaf pytree is caught and named")
  # and nothing was enqueued for the good leaves before the raise
  assert ch.last_fallback_error is None

  # ======================================================================
  # REGRESSION 2: strict=True must reject a resharding boundary even when the
  # leaf is slow for an earlier reason (extended dtype / missing symbol).
  # ======================================================================
  print()
  print("=== regression 2: strict=True sees the reshard behind another reason ===")
  key_src, key_dst = NamedSharding(mA, P("fsdp")), NamedSharding(mB, P(None))
  assert same_index_map(key_src, key_dst, keys.shape) is False
  kbad = Channel(jax.ShapeDtypeStruct(keys.shape, keys.dtype), key_src, key_dst, name="keys-reshard")
  assert kbad.reshards == [True], kbad.reshards
  assert "extended dtype" in kbad.reasons[0] and "reshard" in kbad.reasons[0]
  print(f"  non-strict reports BOTH reasons: {kbad.reasons[0]}")
  try:
    Channel(jax.ShapeDtypeStruct(keys.shape, keys.dtype), key_src, key_dst, name="keys-strict", strict=True)
    raise SystemExit("FAIL: strict=True accepted a resharding PRNG-key boundary")
  except ShardingMismatchError as e:
    assert "would RESHARD" in str(e)
  print("  strict=True rejects it (it really does host-round-trip):")
  with no_host_bounce():
    try:
      block_all(kbad.send(keys))
      raise SystemExit("FAIL: the accepted boundary did not host-round-trip")
    except HostBounceError as e:
      print("  | " + str(e).splitlines()[0])

  saved = transfer.BATCHED_COPY_AVAILABLE
  transfer.BATCHED_COPY_AVAILABLE = False  # simulate a jaxlib without it
  try:
    degraded = Channel(avals["y"], shA["y"], transposed, name="degraded")
    assert degraded.reshards == [True] and "no batched_copy" in degraded.reasons[0]
    assert "reshard" in degraded.reasons[0]
    try:
      Channel(avals["y"], shA["y"], transposed, name="degraded-strict", strict=True)
      raise SystemExit("FAIL: strict=True accepted a reshard on a degraded jaxlib")
    except ShardingMismatchError:
      pass
  finally:
    transfer.BATCHED_COPY_AVAILABLE = saved
  print("  same on a jaxlib missing batched_copy: reason lists both, strict " "still raises")
  # and a leaf that is slow-but-genuinely-D2D is still accepted under strict
  assert Channel(
      jax.ShapeDtypeStruct(keys.shape, keys.dtype),
      keys.sharding,
      NamedSharding(mB, P("fsdp")),
      name="keys-ok",
      strict=True,
  ).reshards == [False]
  print("  a non-resharding PRNG-key boundary is still accepted under strict")

  # ======================================================================
  # REGRESSION 3: set_max_inflight must verify something.  The CPU backend
  # never reads jax_pjrt_client_create_options, so "the backend still comes
  # up" proves nothing; the old test passed with a bogus name AND value.
  # ======================================================================
  print()
  print("=== regression 3: set_max_inflight verifies, and admits what it cannot ==")
  # (a) nonsense values are a caller bug, not a silently-ignored option
  for bad_n, exc in (
      (None, TypeError),
      ("not_an_int", TypeError),
      (12.5, TypeError),
      (True, TypeError),
      (0, ValueError),
      (-1, ValueError),
  ):
    try:
      set_max_inflight(bad_n)
      raise SystemExit(f"FAIL: set_max_inflight({bad_n!r}) was accepted")
    except exc:
      pass
  print("  non-int / non-positive n raise (a str value used to sail through)")

  # (b) after backend init it is a no-op and says so
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    late = set_max_inflight(128)
  assert late is False and any("after backend initialization" in str(x.message) for x in w)
  print(f"  after backend init: returned {late} with a warning (never an " "exception)")

  # (c) structural proof that the CPU client cannot read the option at all
  src = inspect.getsource(sys.modules["jax._src.xla_bridge"])
  callers = set()
  for fn in ast.walk(ast.parse(src)):
    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
      for nd in ast.walk(fn):
        if isinstance(nd, ast.Call) and getattr(nd.func, "id", None) == "_options_from_jax_configs":
          callers.add(fn.name)
  assert callers == {"tpu_client_timer_callback", "make_pjrt_c_api_client"}, callers
  assert "options" not in inspect.signature(xc.make_cpu_client).parameters
  # This process runs on CPU (setUpModule).  Ask the backend in use, not jax_platforms: MaxText's CI runs many test
  # modules in one process, and one may leave jax_platforms set (e.g. 'tpu,cpu').  (e) checks a pinned process.
  assert jax.default_backend() == "cpu", (jax.default_backend(), pinned_platforms())
  assert "cpu" not in OPTION_AWARE_PLATFORMS
  print(
      f"  _options_from_jax_configs callers = {sorted(callers)}; " f"xla_client.make_cpu_client takes no options argument"
  )

  # (d) the option NAME is real: it is a plugin-side string, absent from jax and
  #     jaxlib python, present in the libtpu that would consume it
  found, where = option_name_evidence()
  assert found is not False, f"{MAX_INFLIGHT_OPTION} not found in {where}"
  print(f"  option name {MAX_INFLIGHT_OPTION!r}: found in plugin binary = " f"{found} ({where})")

  # (e) before backend init: the value must be observable, as an int, in the very
  #     dict the TPU client factory is handed -- and on a CPU-pinned process the
  #     call must report False rather than claim a cap it cannot apply.
  probe = (
      "import warnings;"
      "from maxtext.experimental.mpmd.jax import transfer; import jax;"
      'base = transfer.pjrt_client_options("tpu");'
      'warnings.simplefilter("always");'
      "w = warnings.catch_warnings(record=True); rec = w.__enter__();"
      "ok = transfer.set_max_inflight(128); w.__exit__(None, None, None);"
      'opts = transfer.pjrt_client_options("tpu");'
      'print("accepted", ok);'
      'print("reason", [str(m.message).split(".")[0] for m in rec]);'
      'print("tpu_option", repr(opts.get("max_inflight_computations")));'
      'print("kept_defaults", all(opts.get(k) == v for k, v in base.items()));'
      'print("devices", len(jax.devices()))'
  )
  early = subprocess.run(
      [sys.executable, "-c", probe],
      capture_output=True,
      text=True,
      cwd=ROOT,
      env={
          "JAX_PLATFORMS": "cpu",
          "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
          "PATH": "/usr/bin:/bin",
          "PYTHONPATH": testing.child_pythonpath(__name__),
      },
  )
  assert early.returncode == 0, early.stderr[-2000:]
  lines = dict(l.split(" ", 1) for l in early.stdout.strip().splitlines())
  assert lines["accepted"] == "False", early.stdout  # CPU cannot apply it
  assert "JAX_PLATFORMS=cpu pins this process" in lines["reason"], early.stdout
  assert lines["tpu_option"] == "128", early.stdout  # int, not '128'
  # Whatever defaults this jax puts in the options survive -- 0.11.1 adds
  # `ml_framework_name`, 0.11.2 adds nothing, so compare against the base dict
  # rather than a literal.
  assert lines["kept_defaults"] == "True", early.stdout  # defaults preserved
  assert lines["devices"] == "8", early.stdout
  print("  before backend init on CPU: " + " | ".join(early.stdout.split("\n")[:4]))

  # (f) the old assertion was vacuous: prove it, so nobody re-adds it.  A
  #     nonexistent option name with a non-integer value leaves the CPU backend
  #     perfectly healthy.
  vacuous = (
      "import sys, jax;"
      'jax.config.update("jax_pjrt_client_create_options",'
      ' {"totally_bogus_option": "banana",'
      '  "max_inflight_computations": "not_an_int"});'
      'print("devices", len(jax.devices()))'
  )
  junk = subprocess.run(
      [sys.executable, "-c", vacuous],
      capture_output=True,
      text=True,
      env={"JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=8", "PATH": "/usr/bin:/bin"},
  )
  assert "devices 8" in junk.stdout, junk.stderr[-2000:]
  print(
      "  control: a bogus name + non-int value ALSO gives a healthy "
      f'"{junk.stdout.strip()}" -> a live backend proves nothing on CPU'
  )
  print(
      "  (the cap itself can only be demonstrated on a TPU client; this test "
      "verifies the plumbing and refuses to fake the rest)"
  )

  # -------------------------------------------------------------- host cost
  nocheck = Channel(avals, shA, shB, name="s0->s1-nocheck", validate=False)
  # send_flat takes leaves in TREEDEF order (sorted dict keys: b, y), not in
  # whatever order the caller happens to write them.
  flat = ch.treedef.flatten_up_to(tree)
  assert [a.shape for a in flat] == [a.shape for a in ch.avals]
  flat_out = block_all(ch.send_flat(flat))
  assert np.array_equal(np.asarray(flat_out[1]), host["y"])
  assert np.array_equal(np.asarray(flat_out[0]), host["b"])
  variants = [
      lambda: [jax.device_put(tree[k], shB[k]) for k in ("y", "b")],
      lambda: jax.device_put(tree, shB),
      lambda: ch.send(tree),
      lambda: ch.send_flat(flat),
      lambda: nocheck.send_flat(flat),
  ]
  # The two cost bounds asserted below are host timings, which a busy machine (two test workers on a 2-vCPU CI
  # runner) can push past a bound once: measure again, three times at most, before one fails.
  for tries in range(1, 4):
    per_leaf, tree_put, chan, chan_flat, chan_nov = bench_all(variants)
    if per_leaf / chan_flat > 2.0 and chan_flat < 1.3 * chan_nov:
      break
  print()
  print(f"=== host-side dispatch cost, 2-leaf boundary, 8 CPU devices (measurement {tries} of at most 3) ===")
  print(f"  per-leaf jax.device_put : {per_leaf*1e6:7.1f} us/boundary " f"({per_leaf*1e6/2:6.1f} us/array)")
  print(f"  jax.device_put(pytree)  : {tree_put*1e6:7.1f} us/boundary " f"({tree_put*1e6/2:6.1f} us/array)")
  print(
      f"  Channel.send            : {chan*1e6:7.1f} us/boundary "
      f"({chan*1e6/2:6.1f} us/array)   {per_leaf/chan:4.2f}x vs per-leaf put"
  )
  print(
      f"  Channel.send_flat       : {chan_flat*1e6:7.1f} us/boundary "
      f"({chan_flat*1e6/2:6.1f} us/array)   {per_leaf/chan_flat:4.2f}x vs per-leaf put"
  )
  print(
      f"  ...validate=False       : {chan_nov*1e6:7.1f} us/boundary "
      f"({chan_nov*1e6/2:6.1f} us/array)   {per_leaf/chan_nov:4.2f}x vs per-leaf put"
  )
  # The guard is a fraction of a microsecond against a ~12 us/leaf send, i.e.
  # below the run-to-run noise of this shared box (the delta above can even come
  # out negative).  Time the guard expression itself instead -- same compares,
  # same objects, no device work -- and separately assert the end-to-end cost did
  # not blow up.
  guard = (
      timeit.timeit(
          "for i, want_s, want_shape in checks:\n"
          "  x = flat[i]\n"
          "  if x.shape != want_shape or ((s := x.sharding) is not want_s and\n"
          "                               s != want_s):\n"
          "    raise AssertionError(i)\n",
          globals={"checks": ch._checks, "flat": flat},
          number=200_000,
      )
      / 200_000
  )
  print(
      f"  guard alone             : {guard*1e6:7.3f} us/boundary "
      f"({guard*1e6/2:6.3f} us/array)   {guard/chan_flat*100:4.1f}% of a send"
  )
  assert per_leaf / chan_flat > 2.0, (per_leaf, chan_flat)  # the win survives
  assert chan_flat < 1.3 * chan_nov, (chan_flat, chan_nov)  # guard stays cheap
  assert guard / chan_flat < 0.1, (guard, chan_flat)
  print("  (CPU numbers are indicative only; the ratio is the informative part)")
  print()
  print("PASS")


@testing.needs_devices(4)
class TransferTest(unittest.TestCase):

  def test_transfer_layer(self):
    """`check_transfer_layer`: the batched D2D copy, its guards and the set_max_inflight plumbing."""
    check_transfer_layer()

  def test_failed_batched_copy_falls_back_to_device_put(self):
    """A batched copy that raises at send (an API drift) costs no step: the channel warns once, keeps the error in
    last_fallback_error, and puts every fast leaf on its destination sharding with jax.device_put, values exact, send
    after send.  A fast leaf handed back as its source array instead would feed a destination-mesh jit once and then
    fail it with INVALID_ARGUMENT.  The slow leaf (a PRNG key) is put as always.  Two meshes of half the devices each,
    as check_transfer_layer's."""
    devs = jax.devices()
    half = min(len(devs), 8) // 2
    mA, mB = Mesh(np.array(devs[:half]), ("fsdp",)), Mesh(np.array(devs[half : 2 * half]), ("fsdp",))
    specs = {"y": P("fsdp", None), "b": P("fsdp"), "key": P("fsdp")}
    src = {k: NamedSharding(mA, s) for k, s in specs.items()}
    dst = {k: NamedSharding(mB, s) for k, s in specs.items()}
    host = {"y": np.arange(64 * 16, dtype=np.float32).reshape(64, 16), "b": np.arange(64, dtype=np.float32) * -0.5}
    tree = {k: jax.device_put(v, src[k]) for k, v in host.items()}
    tree["key"] = jax.device_put(jax.random.split(jax.random.key(0), 8), src["key"])
    avals = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), tree)
    err, calls = RuntimeError("injected"), []

    def boom(srcs, *args):
      calls.append(len(srcs))
      raise err

    # The batched copy raises, whether or not this jaxlib has one.
    with mock.patch.multiple(transfer, BATCHED_COPY_AVAILABLE=True, _BATCHED_COPY=boom):
      ch = Channel(avals, src, dst, name="fallback")
      self.assertEqual((ch.n_fast, ch.n_slow), (2, 1), ch.reasons)
      warned = r"^fallback: batched copy failed \(RuntimeError: injected\); falling back to jax\.device_put per leaf"
      with testing.assert_warns_regex(self, RuntimeWarning, warned) as seen:
        outs = [block_all(ch.send(tree)) for _ in range(2)]
    self.assertEqual(sum("batched copy failed" in str(w.message) for w in seen), 1)  # once per channel
    self.assertEqual(calls, [2, 2])
    self.assertIs(ch.last_fallback_error, err)
    for out in outs:
      for k in specs:
        self.assertEqual(out[k].sharding, dst[k], k)
        self.assertEqual({s.device for s in out[k].addressable_shards}, set(devs[half : 2 * half]), k)  # the buffers
      for k, v in host.items():
        np.testing.assert_array_equal(np.asarray(out[k]), v, k)
      np.testing.assert_array_equal(jax.random.key_data(out["key"]), jax.random.key_data(tree["key"]))
