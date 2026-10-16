"""Profile-guided POST_SCHEDULER pass: move async collective starts earlier.

For every async collective in the scheduled HLO, use the PGLE FDO profile to
check whether the ops between start and done hide its latency. If not, move
the start earlier (bounded by data/control dependencies) and pull heavy
compute into the exposed window.

Steps:
  1. Rank 0 computes the new schedule; its serialized module is shared with
     all other ranks through the JAX distributed KV store (entry/exit
     barriers) so every rank compiles the same executable.
  2. Reorder, repeated until nothing changes (at most _MAX_REORDER_ITERS),
     for each computation, innermost first:
       a. Move each under-hidden async start to its earliest legal position,
          relocating the operand producers it depends on.
       b. If a blocker op prevents that, chase it earlier toward heavy
          compute; move an async done ahead of TE EP calls once its latency
          is hidden.
       c. Fill still-exposed windows with heavy compute (best of several
          fill strategies by total exposed time).
       d. In while bodies, hoist async starts to their floor.
     Each iteration ends with schedule.update()/verify().
  3. Finally, hoist host-offload copy starts to their earliest legal position.

Call ``register()`` once before the first ``jax.jit``-compiled function runs.
"""

# This pass runs inside the XLA compile callback and must fail open: any error
# means "keep XLA's schedule", so broad excepts are intentional. The reorder
# loops are deeply nested by nature.
# pylint: disable=broad-exception-caught,too-many-nested-blocks

from __future__ import annotations

import bisect
import hashlib
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Optional

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
if not _logger.handlers:
  _h = logging.StreamHandler()
  _h.setLevel(logging.DEBUG)
  _h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
  _logger.addHandler(_h)
  _logger.propagate = False

# Populated once PGLE delivers its FDO profile bytes.
_profile_costs: dict[str, float] = {}

# ---------------------------------------------------------------------------
# Proto compilation (PGLE profile + XSpace), all in one directory to avoid
# descriptor pool conflicts. TODO tixxx, remove this once jax exposes
# protobuf utilities.
# ---------------------------------------------------------------------------
_XLA_SRC = os.environ.get("COLLECTIVE_OVERLAP_XLA_PATH", "")
_TSL_SRC = f"{_XLA_SRC}/third_party/tsl"
_PROTO_OUT_DIR_BASE = "/tmp/_gpu_collective_overlap_pass_proto"

_PROTO_SOURCES = [
    "tsl/profiler/protobuf/profiled_instructions.proto",
    "tsl/profiler/protobuf/xplane.proto",
]

_PROTO_INIT_DIRS = [
    "",
    "tsl",
    "tsl/profiler",
    "tsl/profiler/protobuf",
]


def _proto_out_dir() -> str:
  """Generated-code cache dir, keyed by protobuf runtime version and user so a
  directory produced by another container's protoc (or another user) is never
  reused."""
  from google.protobuf import __version__ as pb_version  # pylint: disable=import-outside-toplevel

  return f"{_PROTO_OUT_DIR_BASE}_{pb_version}_{os.getuid()}"


_GENCODE_CHECK_RE = re.compile(
    r"_runtime_version\.ValidateProtobufRuntimeVersion\(\s*_runtime_version\.Domain\.\w+,"
    r"\s*(\d+),\s*(\d+),\s*(\d+),[^)]*\)",
    re.S,
)


def _relax_gencode_version(root: str) -> int:
  """Drop the gencode-vs-runtime version check from generated modules whose
  gencode is newer than the Python protobuf runtime; returns how many files.

  The `protoc` on PATH can be newer than the installed runtime (e.g. gencode
  7.36.2 vs runtime 6.33.6), which makes every import raise VersionError. These
  are plain messages with no newer-protoc-only features, so they load fine on an
  older runtime once the check is gone.
  """
  from google.protobuf import __version__ as pb_version  # pylint: disable=import-outside-toplevel

  runtime = tuple(int(x) for x in re.findall(r"\d+", pb_version)[:3])
  patched = 0
  for dirpath, _, files in os.walk(root):
    for name in files:
      if not name.endswith("_pb2.py"):
        continue
      path = os.path.join(dirpath, name)
      with open(path, encoding="utf-8") as f:
        text = f.read()
      new = _GENCODE_CHECK_RE.sub(
          lambda m: "pass" if tuple(int(m.group(i)) for i in (1, 2, 3)) > runtime else m.group(0),
          text,
      )
      if new != text:
        with open(path, "w", encoding="utf-8") as f:
          f.write(new)
        patched += 1
  return patched


def _ensure_protos():
  """Compile all needed proto files once into a single output directory."""
  out = _proto_out_dir()
  marker = os.path.join(out, "tsl", "profiler", "protobuf", "xplane_pb2.py")
  if not os.path.exists(marker):
    # Generate into a private directory and rename it into place, so ranks
    # racing on the same node never read a half-written tree.
    tmp = f"{out}.tmp.{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    for rel in _PROTO_INIT_DIRS:
      d = os.path.join(tmp, rel)
      os.makedirs(d, exist_ok=True)
      with open(os.path.join(d, "__init__.py"), "a", encoding="utf-8"):
        pass
    try:
      subprocess.run(
          [
              "protoc",
              f"--proto_path={_TSL_SRC}",
              "--proto_path=/usr/local/include",
              f"--python_out={tmp}",
              *_PROTO_SOURCES,
          ],
          check=True,
          capture_output=True,
      )
    except subprocess.CalledProcessError as exc:
      _logger.warning(
          "gpu_collective_overlap_pass: protoc failed: %s",
          exc.stderr.decode("utf-8", errors="replace")[-1000:],
      )
      shutil.rmtree(tmp, ignore_errors=True)
      raise
    patched = _relax_gencode_version(tmp)
    if patched:
      _logger.warning(
          "gpu_collective_overlap_pass: protoc is newer than the Python protobuf runtime; "
          "removed the gencode version check from %d generated module(s).",
          patched,
      )
    try:
      os.rename(tmp, out)
    except OSError:  # another process finished first
      shutil.rmtree(tmp, ignore_errors=True)
  if out not in sys.path:
    sys.path.insert(0, out)


def _load_costs_from_fdo(fdo_bytes: bytes) -> dict[str, float]:
  _ensure_protos()
  from tsl.profiler.protobuf import profiled_instructions_pb2 as _pi  # type: ignore  # pylint: disable=import-outside-toplevel

  profiled = _pi.ProfiledInstructionsProto()
  profiled.ParseFromString(fdo_bytes)
  return {c.name: c.cost_us for c in profiled.costs}


def _update_profile(fdo_bytes: bytes) -> None:
  """Merge an FDO profile chunk into the pooled per-instruction costs."""
  if not fdo_bytes:
    return
  try:
    costs = _load_costs_from_fdo(fdo_bytes)
    if not costs:
      return
    _profile_costs.update(costs)
    # PGLE's own te_ep entries are underestimates; keep the live corrections.
    _profile_costs.update(_te_ep_overrides)
    n_coll = sum(1 for k in costs if "all-gather" in k or "reduce-scatter" in k)
    _logger.info(
        "gpu_collective_overlap_pass: merged PGLE profile chunk: %d entries (%d collectives); pool now %d.",
        len(costs),
        n_coll,
        len(_profile_costs),
    )
  except Exception as exc:  # pylint: disable=broad-except
    _logger.warning("gpu_collective_overlap_pass: failed to parse FDO profile: %s", exc)


# TODO tixxx remove the above invocation cost once PGLE can return the
# correct aggregated cost for multi-kernel ops.
# ---------------------------------------------------------------------------
# Per-invocation cost corrections from the live PGLE XSpace
# ---------------------------------------------------------------------------
# PGLE's xplane->FDO conversion averages cost_us over every GPU kernel launch
# tagged with an HLO op's name. That is fine for single-kernel ops but wrong for
# ops that run as many kernels per execution.
# _patch_pgle_profiler intercepts the raw XSpace and
# clusters events into invocations (a new one starts after a gap of
# _INVOCATION_GAP_US) to recover the per-invocation cost.
_TE_EP_PREFIX = "te_ep"
_INVOCATION_GAP_US = 100.0
# te_ep corrections replace the PGLE values in _profile_costs. The GEMM and
# collective corrections stay separate and are used only by the stream model
# (the rest of the pass keeps PGLE's numbers).
_te_ep_overrides: dict[str, float] = {}
_gemm_overrides: dict[str, float] = {}
_GEMM_COST_PREFIXES = ("te_gemm", "te_grouped_gemm")
_collective_overrides: dict[str, float] = {}
_COLLECTIVE_COST_PREFIXES = ("all-gather.", "reduce-scatter.", "all-reduce.")


def _average_invocation_costs(by_op: dict[str, list[tuple[float, float]]]) -> dict[str, float]:
  """{hlo_op: [(ts_us, dur_us), ...]} -> {hlo_op: mean per-invocation span}."""
  costs: dict[str, float] = {}
  for hlo_op, events in by_op.items():
    if not events:
      continue
    events.sort()
    invocation_spans: list[float] = []
    group_start, group_end = events[0][0], events[0][0] + events[0][1]
    for ts, dur in events[1:]:
      if ts - group_end > _INVOCATION_GAP_US:
        invocation_spans.append(group_end - group_start)
        group_start = ts
      group_end = max(group_end, ts + dur)
    invocation_spans.append(group_end - group_start)
    costs[hlo_op] = sum(invocation_spans) / len(invocation_spans)
  return costs


def _load_costs_from_xspace_bytes(xspace_bytes: bytes, prefixes=None) -> dict[str, float]:
  """Per-invocation costs for HLO ops starting with `prefixes`, from a raw XSpace.

  Walks GPU device planes, resolves each event's `hlo_op` stat (event or event
  metadata; str_value or interned ref_value) and clusters by invocation.
  """
  _ensure_protos()
  from tsl.profiler.protobuf import xplane_pb2 as _xp  # type: ignore  # pylint: disable=import-outside-toplevel

  if prefixes is None:
    prefixes = (_TE_EP_PREFIX,)
  try:
    xspace = _xp.XSpace()
    xspace.ParseFromString(xspace_bytes)
  except Exception as exc:  # pylint: disable=broad-except
    _logger.warning("gpu_collective_overlap_pass: failed to parse live XSpace profile: %s", exc)
    return {}

  by_op: dict[str, list[tuple[float, float]]] = {}
  for plane in xspace.planes:
    if not plane.name.startswith("/device:GPU:"):
      continue
    stat_meta_name = {m.id: m.name for m in plane.stat_metadata.values()}
    event_meta_stats = {m.id: list(m.stats) for m in plane.event_metadata.values()}

    # Called only within this iteration, so capturing the loop variable is safe.
    # pylint: disable=cell-var-from-loop
    def _resolve_hlo_op(stats) -> Optional[str]:
      for stat in stats:
        if stat_meta_name.get(stat.metadata_id) != "hlo_op":
          continue
        which = stat.WhichOneof("value")
        if which == "str_value":
          return stat.str_value
        if which == "ref_value":
          return stat_meta_name.get(stat.ref_value)
      return None

    for line in plane.lines:
      for event in line.events:
        if event.WhichOneof("data") != "offset_ps":
          continue
        dur_us = event.duration_ps / 1e6
        if dur_us <= 0:
          continue
        hlo_op = _resolve_hlo_op(event_meta_stats.get(event.metadata_id, ())) or _resolve_hlo_op(event.stats)
        if not hlo_op or not hlo_op.startswith(prefixes):
          continue
        ts_us = (line.timestamp_ns * 1000 + event.offset_ps) / 1e6
        by_op.setdefault(hlo_op, []).append((ts_us, dur_us))

  return _average_invocation_costs(by_op)


# ---------------------------------------------------------------------------
# Cross-rank sync (JAX's coordination-service key-value store)
# ---------------------------------------------------------------------------
# The graph traversal here is cpu-heavy, try to avoid straining the system
# when it's running 1 process per GPU by only running the compilation on rank0
_dist_client = None
_dist_checked = False

# Count of invocations that got past the early-out; names the barriers and the
# KV key (see _gpu_collective_overlap_pass for why not a content hash).
_barrier_invocation_count = 0


def _get_distributed_client():
  """Lazily resolve jax's distributed coordination-service client."""
  global _dist_client, _dist_checked
  if not _dist_checked:
    _dist_checked = True
    try:
      from jax._src import distributed as _jax_distributed  # pylint: disable=import-outside-toplevel

      _dist_client = _jax_distributed.global_state.client
      _process_id = _jax_distributed.global_state.process_id
      _process_count = _jax_distributed.global_state.num_processes
      _logger.info(
          "gpu_collective_overlap_pass: JAX distributed client: process_id=%d process_count=%d host=%s pid=%d.%s",
          _process_id,
          _process_count,
          socket.gethostname(),
          os.getpid(),
          ""
          if (_dist_client is not None and _process_count > 1)
          else (
              " WARNING: no multi-process client; every rank will compute its own "
              "(possibly divergent) schedule. Was jax.distributed.initialize() called?"
          ),
      )
    except Exception as exc:  # pylint: disable=broad-except
      _logger.warning(
          "gpu_collective_overlap_pass: jax distributed client unavailable (%s); "
          "each rank will use its own schedule, which risks divergence.",
          exc,
      )
      _dist_client = None
  return _dist_client


def _jax_process_id() -> int:
  from jax._src import distributed as _jax_distributed  # pylint: disable=import-outside-toplevel

  return _jax_distributed.global_state.process_id


def _jax_process_count() -> int:
  from jax._src import distributed as _jax_distributed  # pylint: disable=import-outside-toplevel

  return _jax_distributed.global_state.num_processes


_SHARE_TIMEOUT_MS = int(os.environ.get("COLLECTIVE_OVERLAP_SHARE_TIMEOUT_MS", str(20 * 60 * 1000)))

# Prefix bytes distinguishing "no transformation" from module bytes in the KV store.
_SHARE_NONE = b"\x00"
_SHARE_SOME = b"\x01"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Dedicated Start opcodes. Reduce-scatter (and te_ep calls) use the generic
# async-start wrapper instead, which is async by construction.
_ASYNC_START_OPCODES = frozenset(
    {
        "all-gather-start",
        "all-reduce-start",
        "collective-permute-start",
    }
)

# COLLECTIVE_OVERLAP_WHILE_BODY_ONLY=1 skips the entry computation, which is by
# far the largest sequence and dominates compile time.
# Trades its overlap away for faster compiles.
_WHILE_BODY_ONLY = os.environ.get("COLLECTIVE_OVERLAP_WHILE_BODY_ONLY") == "1"

# COLLECTIVE_OVERLAP_HOIST_ASYNC_STARTS=0 disables the final per-while-body step
# that moves async starts (and their trivial operand chain) toward their
# earliest legal position.
_HOIST_ASYNC_STARTS = os.environ.get("COLLECTIVE_OVERLAP_HOIST_ASYNC_STARTS", "1") != "0"

# COLLECTIVE_OVERLAP_ASYNC_STREAM_MODEL=0 disables the comm-stream model that
# places async starts/dones in the hoist step (_best_async_placement). While on,
# it also owns async done placement in while bodies and
# _relocate_async_done_before_te_ep is skipped there.
_ASYNC_STREAM_MODEL = os.environ.get("COLLECTIVE_OVERLAP_ASYNC_STREAM_MODEL", "1") != "0"
# Simulated totals within this many us of the best are treated as equal when
# placing an async start, so the earliest start (largest window) wins. Without
# it a ~30 us modeled gain moved collective starts next to their dones.
_ASYNC_STREAM_TOL_US = float(os.environ.get("COLLECTIVE_OVERLAP_ASYNC_STREAM_TOL_US", "250"))
# SM gating: a launched heavy GEMM holds the SMs, so a collective kernel can't
# begin until it ends. COLLECTIVE_OVERLAP_SM_GATE=0 disables; GEMMs shorter than
# COLLECTIVE_OVERLAP_SM_GATE_MIN_US are ignored.
_SM_GATE = os.environ.get("COLLECTIVE_OVERLAP_SM_GATE", "1") != "0"
_SM_GATE_MIN_US = float(os.environ.get("COLLECTIVE_OVERLAP_SM_GATE_MIN_US", "1000"))


def _opcode_str(inst) -> str:
  """Hyphenated opcode string (e.g. "get-tuple-element") for an HloInstruction."""
  return re.sub(r"(?<!^)(?=[A-Z])", "-", inst.opcode.name[1:]).lower()


# Host-offload copies: XLA lowers a remat offload to a pair of async DMAs,
# D2H as an async dynamic-update-slice into a host-memory-space buffer (S(5)) in
# the forward pass and H2D as an async dynamic-slice out of that buffer in the
# backward pass. They are copies, not collectives, we use special handling for those.
_HOST_MEMORY_SPACE_TAG = ":S(5)"
_HOIST_HOST_OFFLOAD_COPIES = os.environ.get("COLLECTIVE_OVERLAP_HOIST_OFFLOAD_COPIES", "1") != "0"
# Names (instruction names are unique per module) of host-offload copy starts
# and their dones in the module being processed; see _register_host_offload_copies.
_host_offload_copy_names: set[str] = set()


# Unlike _CALLS_RE, allows hyphens: the wrapped computations are named
# e.g. wrapped_dynamic-update-slice_computation.
_HOST_COPY_CALLS_RE = re.compile(r"calls=%([\w.\-]+)")


def _host_offload_copy_kind(inst, comp_by_name: dict) -> Optional[str]:
  """ "d2h" for an async dynamic-update-slice writing a host-memory-space
  buffer, "h2d" for an async dynamic-slice reading one, else None.

  The scheduled module prints these as "fusion-start" with
  calls=%wrapped_dynamic-(update-)slice_computation, so follow the start's
  calls= to the computation whose root is the DUS/DS (a few hops, in case
  of nested wrappers). Parsed from text: async_wrapped_root() is not reliable
  for these.
  """
  if _opcode_str(inst) != "async-start":
    return None
  try:
    text = inst.to_string()
  except Exception:
    return None
  if _HOST_MEMORY_SPACE_TAG not in text:
    return None
  for _ in range(4):
    m = _HOST_COPY_CALLS_RE.search(text)
    comp = comp_by_name.get(m.group(1)) if m else None
    root = _computation_root(comp) if comp is not None else None
    if root is None:
      return None
    kind = {"dynamic-update-slice": "d2h", "dynamic-slice": "h2d"}.get(_opcode_str(root))
    if kind is not None:
      return kind
    try:
      text = root.to_string()
    except Exception:
      return None
  return None


def _register_host_offload_copies(module) -> None:
  """Rebuild _host_offload_copy_names (starts and their dones) from `module`."""
  names: set[str] = set()
  comp_by_name = {c.name: c for c in module.computations()}
  for comp in module.make_nonfusion_computations():
    for inst in comp.instructions():
      if _host_offload_copy_kind(inst, comp_by_name) is not None:
        names.add(inst.name)
        names.update(user.name for user in inst.users())
  _host_offload_copy_names.clear()
  _host_offload_copy_names.update(names)
  if names:
    _logger.info(
        "gpu_collective_overlap_pass [%s]: registered %d host-offload copy instruction(s) (starts and dones): %s",
        module.name,
        len(names),
        ", ".join(sorted(names)),
    )


# This tells the main loop of the pass of what the interesting ops are
# for optimization. Add additional logic here to match any future ops
# to be scheduled.
def _is_async_start(inst: object) -> bool:
  """True if `inst` is the Start half of an async collective.

  Deliberately permissive: the generic "async-start" opcode covers NCCL
  collectives (all-gather/reduce-scatter/all-reduce), te_ep call-starts and
  trivial passthrough wrappers alike, and there is no cheap reliable way to
  tell them apart by opcode/attributes. A trivial wrapper never shows a
  deficit worth acting on. Host-offload copies are the exception: they are
  recognized (see _host_offload_copy_names) and excluded.
  """
  if inst.name in _host_offload_copy_names:
    return False
  opcode = _opcode_str(inst)
  return opcode in _ASYNC_START_OPCODES or opcode == "async-start"


def _module_has_interesting_async_ops(module) -> bool:
  """True if any non-fusion computation has an async-start (cheap early-out)."""
  _register_host_offload_copies(module)
  for comp in module.make_nonfusion_computations():
    for inst in comp.instructions():
      if _is_async_start(inst):
        return True
  return False


# ---------------------------------------------------------------------------
# HLO text helpers (the HloInstruction binding exposes little beyond name/opcode/
# operands/users/to_string, so the rest is parsed from to_string())
# ---------------------------------------------------------------------------
# "parameter"/"constant" are leaves with no real dependency, so always movable.
_TRIVIAL_SINGLE_OPCODES = frozenset(
    {
        "bitcast",
        "convert",
        "transpose",
        "reshape",
        "broadcast",
        "get-tuple-element",
        "tuple",
        "copy",
        "slice",
        "parameter",
        "constant",
    }
)

_TRIVIAL_FUSED_OPCODES = frozenset(
    {
        "parameter",
        "constant",
        "iota",
        "convert",
        "bitcast",
        "reshape",
        "transpose",
        "broadcast",
        "copy",
        "concatenate",
        "dynamic-slice",
        "slice",
        "get-tuple-element",
        "tuple",
        "add",
        "subtract",
        "multiply",
        "divide",
        "negate",
        "abs",
        "maximum",
        "minimum",
        "and",
        "or",
        "not",
        "xor",
        "sqrt",
        "rsqrt",
        "exp",
        "log",
        "sign",
        "clamp",
        "floor",
        "ceil",
        "round-nearest-afz",
        "round-nearest-even",
        "compare",
        "select",
    }
)

_CALLS_RE = re.compile(r"calls=%([A-Za-z0-9_.]+)")
_TO_APPLY_RE = re.compile(r"to_apply=%([A-Za-z0-9_.]+)")
_ROOT_RE = re.compile(r"\bROOT %([A-Za-z0-9_.\-]+) = ")
_CONTROL_PRED_RE = re.compile(r"control-predecessors=\{([^}]*)\}")


def _is_trivial_fusion_body(inst, comp_by_name: dict) -> bool:
  """True if every op in inst's fused computation is trivial."""
  try:
    text = inst.to_string()
  except Exception:
    return False
  m = _CALLS_RE.search(text)
  if not m:
    return False
  comp = comp_by_name.get(m.group(1))
  if comp is None:
    return False
  try:
    for fi in comp.instructions():
      if _opcode_str(fi) not in _TRIVIAL_FUSED_OPCODES:
        return False
    return True
  except Exception:
    return False


def _is_trivially_movable_inst(inst, comp_by_name: dict) -> bool:
  """True if inst can be relocated without moving real compute."""
  opc = _opcode_str(inst)
  if opc in _TRIVIAL_SINGLE_OPCODES:
    return True
  if opc == "fusion":
    return _is_trivial_fusion_body(inst, comp_by_name)
  return False


# TODO tixxx, remove this once control_predecessors is exposed through python binding.
def _control_predecessor_names(inst) -> list[str]:
  try:
    text = inst.to_string()
  except Exception:
    return []
  m = _CONTROL_PRED_RE.search(text)
  if not m:
    return []
  return [n.strip().lstrip("%") for n in m.group(1).split(",") if n.strip()]


# TODO tixxx, remove this once root_instruciton is exposed through python binding.
def _computation_root(comp):
  """comp's ROOT instruction"""
  try:
    return comp.root_instruction()
  except Exception:
    pass
  try:
    m = _ROOT_RE.search(comp.to_string())
    if m:
      for inst in comp.instructions():
        if inst.name == m.group(1):
          return inst
  except Exception:
    pass
  try:
    for inst in comp.instructions():
      if not list(inst.users()):
        return inst
  except Exception:
    pass
  return None


_MAX_PROFILE_KEY_UNWRAP_HOPS = 8


def _unwrap_to_leaf_name(inst, comp_by_name: dict) -> str:
  """Follow nested call/async wrappers down to the instruction PGLE profiles.

  Stops at the first non-call/async op (it may simply have no profile entry,
  as for a trivial passthrough).
  """
  for _ in range(_MAX_PROFILE_KEY_UNWRAP_HOPS):
    opc = _opcode_str(inst)
    if opc not in ("call", "call-start", "async-start"):
      return inst.name
    try:
      text = inst.to_string()
    except Exception:
      return inst.name
    m = _TO_APPLY_RE.search(text) or _CALLS_RE.search(text)
    if not m:
      return inst.name
    comp = comp_by_name.get(m.group(1))
    if comp is None:
      return inst.name
    root = _computation_root(comp)
    if root is None:
      return inst.name
    inst = root
  return inst.name


def _resolve_profile_key(async_start, comp_by_name: dict) -> str:
  """Name under which async_start's latency is keyed in _profile_costs.

  Standard collectives expose their wrapped root via async_wrapped_root().
  te_ep call-starts use to_apply=%name instead (async_wrapped_root() fails or
  returns an intermediate "call.N"), so fall back to parsing it.
  """
  root = None
  try:
    root = async_start.async_wrapped_root()
  except Exception:
    root = None
  if root is None:
    try:
      text = async_start.to_string()
    except Exception:
      text = ""
    m = _TO_APPLY_RE.search(text)
    if m:
      comp = comp_by_name.get(m.group(1))
      if comp is not None:
        root = _computation_root(comp)
  if root is None:
    return async_start.name
  return _unwrap_to_leaf_name(root, comp_by_name)


def _is_te_ep_call_start(inst, comp_by_name: dict) -> bool:
  """True for a te_ep dispatch/combine call-start (the multi-ms MoE windows)."""
  if not _is_async_start(inst):
    return False
  return _resolve_profile_key(inst, comp_by_name).startswith(("te_ep_combine", "te_ep_dispatch"))


_TE_GEMM_TARGET_RE = re.compile(r'custom_call_target="(te_grouped_gemm[^"]*|te_gemm[^"]*)"')
_HEAVY_ANCHOR_TARGET_RE = re.compile(
    r'custom_call_target="(te_grouped_gemm[^"]*|te_gemm[^"]*|[^"]*cudnn[^"]*|[^"]*cublas[^"]*)"'
)


def _wraps_custom_call(inst, target_re, comp_by_name: dict) -> bool:
  """True if inst is a custom-call matching target_re, or a kind=kCustom
  fusion whose nested computation holds one (such a fusion never gets its own
  PGLE entry; the kernel is tagged with the inner custom-call's name)."""
  opc = _opcode_str(inst)
  try:
    text = inst.to_string()
  except Exception:
    return False
  if opc == "custom-call":
    return bool(target_re.search(text))
  if opc == "fusion" and "kind=kCustom" in text:
    m = _CALLS_RE.search(text)
    if not m:
      return False
    comp = comp_by_name.get(m.group(1))
    if comp is None:
      return False
    try:
      return any(_opcode_str(sub) == "custom-call" and target_re.search(sub.to_string()) for sub in comp.instructions())
    except Exception:
      return False
  return False


def _is_te_gemm_custom_call(inst, comp_by_name: dict) -> bool:
  """te_grouped_gemm/te_gemm kernel (the MoE GEMMs that dominate compute);
  prioritized when filling async collective windows."""
  return _wraps_custom_call(inst, _TE_GEMM_TARGET_RE, comp_by_name)


def _is_heavy_anchor_custom_call(inst, comp_by_name: dict) -> bool:
  """te_gemm/cudnn/cublas kernel: where _chase_async_start_blocker_toward_heavy_compute
  stops. Broader than _is_te_gemm_custom_call, kept separate so this chase
  path doesn't change the validated fill priority."""
  return _wraps_custom_call(inst, _HEAVY_ANCHOR_TARGET_RE, comp_by_name)


# Async collectives the reorder treats as relocatable; all-reduce is handled like all-gather/reduce-scatter.
_ASYNC_COLLECTIVE_PREFIXES = ("all-gather", "reduce-scatter", "all-reduce")

_MAX_RELOCATE_CHAIN = 64

# Minimum profiled cost (us) for an instruction to count as heavy compute.
_HEAVY_COMPUTE_MIN_US = 5.0


def _resolve_inst_cost(inst, comp_by_name: dict) -> float:
  """Profiled cost (us) of inst.

  A kind=kCustom fusion (e.g. dynamic-slice-fusion around te_gemm_v2_ffi plus
  a dynamic-update-slice) has no entry of its own, so sum the entries of
  every instruction in its nested computation.
  """
  if inst.name in _host_offload_copy_names:
    return 0.0  # DMA, not SM compute: hides nothing
  direct = _profile_costs.get(inst.name)
  if direct:
    return direct
  if _opcode_str(inst) != "fusion":
    return 0.0
  try:
    text = inst.to_string()
  except Exception:
    return 0.0
  if "kind=kCustom" not in text:
    return 0.0
  m = _CALLS_RE.search(text)
  if not m:
    return 0.0
  comp = comp_by_name.get(m.group(1))
  if comp is None:
    return 0.0
  try:
    return sum(_profile_costs.get(sub.name, 0.0) for sub in comp.instructions())
  except Exception:
    return 0.0


def _prefix_costs_excluding(seq: list, exclude: set, comp_by_name: dict) -> list:
  """Prefix sum of costs along `seq`, with `exclude`d instructions counted as 0."""
  prefix = [0.0] * (len(seq) + 1)
  for i, inst in enumerate(seq):
    c = 0.0 if inst in exclude else _resolve_inst_cost(inst, comp_by_name)
    prefix[i + 1] = prefix[i] + c
  return prefix


def _total_exposed_us(
    start_of_done: dict,
    positions: dict,
    seq: list,
    comp_by_name: dict,
) -> float:
  """Sum of positive deficits over every collective with a known latency, on
  the schedule as it stands: the metric that decides whether a move is a net win."""
  prefix = _prefix_costs_excluding(seq, (), comp_by_name)
  total = 0.0
  for async_done, async_start in start_of_done.items():
    if async_start not in positions or async_done not in positions:
      continue
    latency = _profile_costs.get(_resolve_profile_key(async_start, comp_by_name))
    if latency is None or latency <= 0:
      continue
    s, d = positions[async_start], positions[async_done]
    overlap = prefix[d] - prefix[s + 1] if d > s + 1 else 0.0
    total += max(0.0, latency - overlap)
  return total


def _earliest_legal_pos(
    async_start,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    cp_pins: Optional[list] = None,
) -> tuple[int, list, object]:
  """Earliest position async_start can legally move to.

  Walks async_start's operands transitively through trivially-movable
  instructions (no distance bound; trivial ops are free to move). The walk
  stops at any non-trivial instruction, which pins the floor to just after
  it. Control-predecessors outside the moving set pin it the same way, and
  are appended to `cp_pins` as (position, name) if a list is given.

  Returns (floor, movable, blocker): movable is the part of the trivial chain
  at or after the floor, in schedule order, excluding async_start (chain members
  already before the floor stay put; moving them would drag async_start's final
  position later). blocker is the non-trivial data producer pinning the
  floor, or None if the floor is 0 or pinned by a control-predecessor.
  """
  candidates: set = set()
  seen: set = set()
  floor = 0
  blocker = None
  queue = list(async_start.operands())
  while queue:
    inst = queue.pop()
    if inst in seen:
      continue
    seen.add(inst)
    if inst not in positions:
      continue
    if len(candidates) < _MAX_RELOCATE_CHAIN and _is_trivially_movable_inst(inst, comp_by_name):
      candidates.add(inst)
      queue.extend(inst.operands())
    else:
      if positions[inst] + 1 > floor:
        floor = positions[inst] + 1
        blocker = inst

  moving_names = {inst.name for inst in candidates} | {async_start.name}
  for inst in list(candidates) + [async_start]:
    for name in _control_predecessor_names(inst):
      if name in moving_names:
        continue
      cp_pos = name_to_pos.get(name)
      if cp_pos is not None and cp_pins is not None:
        cp_pins.append((cp_pos, name))
      if cp_pos is not None and cp_pos + 1 > floor:
        floor = cp_pos + 1
        blocker = None

  # The floor only grows as the walk proceeds, so early candidates may
  # already sit before the final one.
  movable = [inst for inst in candidates if positions[inst] >= floor]
  return floor, sorted(movable, key=lambda i: positions[i]), blocker


def _reinsert(seq: list, moving: list, at: int, last):
  """seq without `moving`/`last`, with `moving` then `last` inserted at `at`."""
  drop = set(moving) | {last}
  new_seq = [inst for inst in seq if inst not in drop]
  new_seq[at:at] = list(moving) + [last]
  return new_seq


def _commit_sequence(schedule, comp, new_seq: list):
  """Set comp's sequence; returns (seq, positions, name_to_pos)."""
  schedule.set_sequence(comp, new_seq)
  positions = {inst: i for i, inst in enumerate(new_seq)}
  name_to_pos = {inst.name: i for inst, i in positions.items()}
  return new_seq, positions, name_to_pos


def _relocate_async_done_before_te_ep(
    async_start,
    async_done,
    comp,
    schedule,
    seq: list,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    collective_latency: float,
    module_name: str = "",
):
  """For an already-hidden async collective, pull `done` forward to just before
  the first te_ep call-start at or after the point where its latency is
  closed, if any.

  `done` gates its consumers, so a later done than needed delays them for no
  benefit; an earlier one frees them to fill te_ep windows (the largest,
  hardest-to-fill ones). Always legal: done's only operand is start, and its
  consumers already sit after its old position. Control-predecessors are
  respected.

  Returns (changed, seq, positions, name_to_pos).
  """
  start_pos = positions[async_start]
  done_pos = positions[async_done]
  if done_pos <= start_pos + 1:
    return False, seq, positions, name_to_pos

  done_floor = start_pos + 1
  for name in _control_predecessor_names(async_done):
    cp_pos = name_to_pos.get(name)
    if cp_pos is not None and cp_pos + 1 > done_floor:
      done_floor = cp_pos + 1

  cum = 0.0
  latency_closure_pos = None
  for i in range(start_pos + 1, done_pos):
    cum += _resolve_inst_cost(seq[i], comp_by_name)
    if cum >= collective_latency:
      latency_closure_pos = i + 1
      break
  if latency_closure_pos is None:
    return False, seq, positions, name_to_pos

  search_start = max(latency_closure_pos, done_floor)
  target = None
  for i in range(search_start, done_pos):
    inst = seq[i]
    if inst is async_done or inst is async_start:
      continue
    if _is_te_ep_call_start(inst, comp_by_name):
      target = i
      break
  if target is None:
    return False, seq, positions, name_to_pos

  new_seq, new_positions, new_name_to_pos = _commit_sequence(schedule, comp, _reinsert(seq, [], target, async_done))
  _logger.info(
      "gpu_collective_overlap_pass [%s]: moved %s done from pos %d to %d, before te_ep call-start %s "
      "(latency %.1f us closed by pos %d).",
      module_name,
      async_done.name,
      done_pos,
      target,
      seq[target].name,
      collective_latency,
      latency_closure_pos,
  )
  return True, new_seq, new_positions, new_name_to_pos


# Bound on "blocker's own blocker" hops per chase. The chain is meant to be
# walked to the computation's inputs; each hop self-terminates when a blocker has
# no legal room, so this only guards against a pathological chain.
_MAX_PRODUCER_RELOCATE_HOPS = 64

# Bound on hops when chasing a fill candidate's blocker chain via
# _try_relocate_blocker_earlier (net-benefit-gated per hop, though cross-hop
# error can still accumulate). COLLECTIVE_OVERLAP_HEAVY_CHASE_HOPS=0 disables.
_MAX_HEAVY_COMPUTE_CHASE_HOPS = int(os.environ.get("COLLECTIVE_OVERLAP_HEAVY_CHASE_HOPS", "8"))

# Outer fixed-point passes in _fill_exposed_collectives_with_heavy_compute: a
# fill or chase can expose a window that was hidden when the worklist was built.
_MAX_FILL_FIXED_POINT_ITERS = 3

# Give up on a blocked candidate once it has been the first blocked thing on
# this many consecutive scan restarts, so it can't starve reachable ones.
_STUCK_CANDIDATE_GIVE_UP_STREAK = 4


def _find_best_blocker_position(
    blocker,
    chain: list,
    floor: int,
    blocker_pos: int,
    async_start,
    async_done_pos: int,
    async_latency: float,
    start_of_done: dict,
    positions: dict,
    seq: list,
    comp_by_name: dict,
) -> tuple[int, float]:
  """Position in [floor, blocker_pos] that minimizes TOTAL exposed time over
  every collective in the computation when `blocker` (and its trivial chain)
  moves there.

  Moving blocker to P changes async_start's own overlap (its new floor is P +
  len(chain) + 1) and, for every other window, removes blocker's cost if it
  currently sits inside and wouldn't at P, or adds it in the reverse case.
  Chain members are trivial (~0 cost) and ignored. This is an approximation
  (pre-move positions decide window containment) but cheap, and it catches
  the dominant effect of stealing overlap from already-hidden windows.

  Candidates are blocker's *landing* positions, so "leave it" is unambiguously
  new_pos == blocker_pos (evaluating in insertion-start units made the no-op
  baseline depend on len(chain) and allowed oscillation between two blockers).
  Ties prefer the position closest to blocker_pos.

  Returns (best insertion-start position, best total exposed us). A result of
  blocker_pos - len(chain) means don't move.
  """
  prefix = _prefix_costs_excluding(seq, set(chain) | {blocker}, comp_by_name)
  block_cost = _resolve_inst_cost(blocker, comp_by_name)

  other_windows = []  # (start_pos, done_pos, latency, overlap excluding blocker)
  for od, os_ in start_of_done.items():
    if os_ is async_start or os_ not in positions or od not in positions:
      continue
    latency2 = _profile_costs.get(_resolve_profile_key(os_, comp_by_name))
    if latency2 is None or latency2 <= 0:
      continue
    s2, d2 = positions[os_], positions[od]
    base_overlap = prefix[d2] - prefix[s2 + 1] if d2 > s2 + 1 else 0.0
    other_windows.append((s2, d2, latency2, base_overlap))

  lo = floor + len(chain)
  new_pos_candidates = {lo, blocker_pos}
  for s2, d2, _, _ in other_windows:
    if lo <= s2 < blocker_pos:
      new_pos_candidates.add(s2 + 1)
    if lo < d2 <= blocker_pos:
      new_pos_candidates.add(d2)
  new_pos_candidates = sorted(p for p in new_pos_candidates if lo <= p <= blocker_pos)

  def total_exposed(blocker_new_pos: int) -> float:
    async_new_start = blocker_new_pos + 1
    async_overlap = prefix[async_done_pos] - prefix[async_new_start] if async_done_pos > async_new_start else 0.0
    total = max(0.0, async_latency - async_overlap)
    for s2, d2, latency2, base_overlap in other_windows:
      contains_new = s2 < blocker_new_pos < d2
      overlap2 = base_overlap + (block_cost if contains_new else 0.0)
      total += max(0.0, latency2 - overlap2)
    return total

  best_new_pos, best_exposed = blocker_pos, None
  for new_pos in new_pos_candidates:
    e = total_exposed(new_pos)
    if best_exposed is None or e < best_exposed - 1e-6 or (abs(e - best_exposed) <= 1e-6 and new_pos > best_new_pos):
      best_exposed = e
      best_new_pos = new_pos
  return best_new_pos - len(chain), best_exposed


def _try_relocate_blocker_earlier(
    blocker,
    comp,
    schedule,
    seq: list,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    async_start,
    async_done_pos: int,
    async_latency: float,
    start_of_done: dict,
    module_name: str = "",
    position_margins: dict | None = None,
):
  """Relocate `blocker` (a non-trivial instruction pinning some collective's
  floor) and its trivial chain earlier, if it is heavy and one of two gates
  passes:

  1. Strict: the best position in [floor, current] strictly lowers TOTAL
     exposed time (see _find_best_blocker_position).
  2. Margin-safe fallback (only when `position_margins` is given, i.e. from
     the heavy-compute chase): move straight to the floor if blocker's cost
     fits within the slack of every collective covering its position. Path 1
     is myopic across a chain: moving a small op a few positions looks like a
     wash even when it is a necessary link to a candidate several hops away.
     Path 2 can never push a fine collective below its latency.

  Moving a producer earlier never breaks its consumers; the risk is stealing
  overlap another collective relied on, which both gates guard against.

  Returns the new (seq, positions, name_to_pos) if a move happened, else None.
  """
  if blocker not in positions:
    return None
  cost = _resolve_inst_cost(blocker, comp_by_name)
  if cost < _HEAVY_COMPUTE_MIN_US:
    return None
  floor, chain, _ = _earliest_legal_pos(blocker, positions, name_to_pos, comp_by_name)
  blocker_pos = positions[blocker]
  # The chain lands right before blocker, so its final position is
  # floor + len(chain). Comparing against the bare floor would report room to
  # move for a blocker already packed after its chain and loop forever.
  final_pos = floor + len(chain)
  if final_pos >= blocker_pos:
    return None

  best_pos, best_exposed = _find_best_blocker_position(
      blocker,
      chain,
      floor,
      blocker_pos,
      async_start,
      async_done_pos,
      async_latency,
      start_of_done,
      positions,
      seq,
      comp_by_name,
  )
  current_total = _total_exposed_us(start_of_done, positions, seq, comp_by_name)
  improves_total = best_pos < blocker_pos - len(chain) and best_exposed < current_total - 1e-6

  target = None
  if improves_total:
    target = best_pos
    _logger.debug(
        "gpu_collective_overlap_pass [%s]: TRC %s: pos %d -> %d (chain %d), total exposed %.1f -> %.1f us (path 1).",
        module_name,
        blocker.name,
        blocker_pos,
        best_pos + len(chain),
        len(chain),
        current_total,
        best_exposed,
    )
  elif position_margins is not None:
    margin = position_margins.get(blocker_pos, float("inf"))
    if cost <= margin:
      target = floor
      _logger.debug(
          "gpu_collective_overlap_pass [%s]: TRC %s: pos %d -> %d (chain %d), margin-safe "
          "(cost %.1f <= margin %.1f us) (path 2).",
          module_name,
          blocker.name,
          blocker_pos,
          floor + len(chain),
          len(chain),
          cost,
          margin,
      )

  if target is None:
    return None
  return _commit_sequence(schedule, comp, _reinsert(seq, chain, target, blocker))


def _chase_async_start_blocker_toward_heavy_compute(
    async_start,
    comp,
    schedule,
    seq: list,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    max_hops: int,
    module_name: str = "",
):
  """async-only: when async_start has no earlier legal position because its direct
  non-trivial producer (`blocker`) sits right next to it, relocate that
  blocker earlier too, repeating until the blocker is a heavy anchor
  (_is_heavy_anchor_custom_call) or has no room.

  Ungated (no net-exposure or margin check): an earlier gated version existed
  to protect other windows, but async operand producers (quantize/dbias ops)
  are specific to that collective. If validation shows another collective
  regressing, the gate needs to come back.

  Returns (changed, seq, positions, name_to_pos, floor, to_move, blocker),
  with floor/to_move/blocker being async_start's current _earliest_legal_pos.
  """
  changed = False
  floor, to_move, blocker = _earliest_legal_pos(async_start, positions, name_to_pos, comp_by_name)
  hops = 0
  while blocker is not None and not _is_heavy_anchor_custom_call(blocker, comp_by_name) and hops < max_hops:
    b_floor, b_chain, _ = _earliest_legal_pos(blocker, positions, name_to_pos, comp_by_name)
    blocker_pos = positions[blocker]
    if b_floor + len(b_chain) >= blocker_pos:
      break
    seq, positions, name_to_pos = _commit_sequence(schedule, comp, _reinsert(seq, b_chain, b_floor, blocker))
    changed = True
    hops += 1
    floor, to_move, blocker = _earliest_legal_pos(async_start, positions, name_to_pos, comp_by_name)
  return changed, seq, positions, name_to_pos, floor, to_move, blocker


def _chase_exposed_collective_blocker_earlier(
    blocker_done,
    start_of_done: dict,
    exposed_dones: set,
    window_start_pos: int,
    window_done_pos: int,
    seq: list,
    schedule,
    comp,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
):
  """If `blocker_done` is the done of ANOTHER still-exposed collective, try to
  move that collective's start earlier, so it sits between the caller's window
  and the heavy-compute candidate it is blocking.

  Narrower than the general recursive blocker chase it fires only for a
  real dependency on another exposed collective's done. The move is capped at
  `window_done_pos` and bounded by the chased start's own legal floor.

  Returns (changed, seq, positions, name_to_pos); unchanged if no move.
  """
  if blocker_done not in exposed_dones:
    return False, seq, positions, name_to_pos
  chase_start = start_of_done.get(blocker_done)
  if chase_start is None or chase_start not in positions:
    return False, seq, positions, name_to_pos
  chase_start_pos = positions[chase_start]
  if chase_start_pos <= window_start_pos:
    return False, seq, positions, name_to_pos
  chase_floor, chase_chain, _ = _earliest_legal_pos(chase_start, positions, name_to_pos, comp_by_name)
  target = min(max(chase_floor, window_start_pos + 1), window_done_pos)
  if target + len(chase_chain) >= chase_start_pos:
    return False, seq, positions, name_to_pos

  new_seq, new_positions, new_name_to_pos = _commit_sequence(
      schedule, comp, _reinsert(seq, chase_chain, target, chase_start)
  )
  return True, new_seq, new_positions, new_name_to_pos


def _find_relocatable_ancestor(
    inst,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    max_depth: int,
    skip_cheap_ancestors: bool = False,
):
  """Walk `inst`'s blocker chain to the first ancestor with structural room to
  move (its floor + chain lands before its current position).

  Purely structural unless skip_cheap_ancestors is set; the result must still
  pass _try_relocate_blocker_earlier's gating before it is moved.

  skip_cheap_ancestors also requires cost >= _HEAVY_COMPUTE_MIN_US, so the
  walk passes over near-zero-cost connector ops (dynamic-slice/select
  fusions, a collective's own call-start/done) that the cost gate would
  refuse anyway.

  Returns the ancestor (possibly `inst`), or None if the chain bottoms out,
  cycles or exhausts max_depth.
  """
  seen = set()
  cur = inst
  for _ in range(max_depth + 1):
    if cur is None or cur in seen or cur not in positions:
      return None
    seen.add(cur)
    floor, chain, sub_blocker = _earliest_legal_pos(cur, positions, name_to_pos, comp_by_name)
    has_room = floor + len(chain) < positions[cur]
    if has_room and (not skip_cheap_ancestors or _resolve_inst_cost(cur, comp_by_name) >= _HEAVY_COMPUTE_MIN_US):
      return cur
    cur = sub_blocker
  return None


def _chase_heavy_compute_blocker_chain(
    candidate,
    comp,
    schedule,
    seq: list,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    async_start,
    async_done_pos: int,
    async_latency: float,
    start_of_done: dict,
    max_hops: int,
    module_name: str = "",
    position_margins: dict | None = None,
    is_while_body: bool = False,
):
  """Relocate whatever blocks `candidate` from reaching a legal position in
  window [async_start, async_done_pos), up to `max_hops` times.

  The blocker can be any movable heavy op, e.g. dot_product_attention_fwd
  blocked by a GEMM blocked by another GEMM.
  Each hop descends via _find_relocatable_ancestor to an op that can move, and
  hands it to _try_relocate_blocker_earlier, the sole committer. Reaching an
  indirect blocker typically takes two hops: the deep ancestor first, then the
  now-movable direct blocker on the next iteration.

  `position_margins` is forwarded to enable the margin-safe path, and
  recomputed after every committed hop so several hops can't cumulatively
  draw more than a window's slack.

  Stops on a refused hop, no relocatable ancestor, max_hops, or once the
  candidate's floor is resolved.

  Returns (changed, seq, positions, name_to_pos, floor, chain, blocker),
  with floor/chain/blocker being candidate's current _earliest_legal_pos.
  """
  changed = False
  floor, chain, blocker = _earliest_legal_pos(candidate, positions, name_to_pos, comp_by_name)
  hops = 0
  while floor > async_done_pos and blocker is not None and hops < max_hops:
    target = _find_relocatable_ancestor(
        blocker,
        positions,
        name_to_pos,
        comp_by_name,
        max_hops - hops,
        skip_cheap_ancestors=is_while_body,
    )
    if target is None:
      break
    result = _try_relocate_blocker_earlier(
        target,
        comp,
        schedule,
        seq,
        positions,
        name_to_pos,
        comp_by_name,
        async_start,
        async_done_pos,
        async_latency,
        start_of_done,
        module_name,
        position_margins,
    )
    hops += 1
    if result is None:
      break
    seq, positions, name_to_pos = result
    changed = True
    if position_margins is not None:
      position_margins = _compute_position_margins(start_of_done, positions, seq, comp_by_name)
    floor, chain, blocker = _earliest_legal_pos(candidate, positions, name_to_pos, comp_by_name)
  _logger.debug(
      "gpu_collective_overlap_pass [%s]: CHC %s: %s after %d/%d hop(s), floor=%d, done_pos=%d.",
      module_name,
      candidate.name,
      "resolved" if floor <= async_done_pos else "gave up",
      hops,
      max_hops,
      floor,
      async_done_pos,
  )
  return changed, seq, positions, name_to_pos, floor, chain, blocker


def _compute_position_margins(
    start_of_done: dict,
    positions: dict,
    seq: list,
    comp_by_name: dict,
    prefix: list | None = None,
    min_pos: int = 0,
) -> dict:
  """For every position >= min_pos, the minimum slack (overlap minus latency)
  among the collectives whose [start+1, done) window covers it: how much cost
  can be pulled out of that position without dropping any covering collective
  below its latency. Uncovered positions have no entry (callers treat that as
  infinite margin).

  Over-covered time is as safe to take from as idle time, as long as no more
  than the slack is taken.

  min_pos trims the work for a caller that only queries positions >= min_pos
  (the candidate scan, done_pos + 1). A caller that queries earlier
  positions, like the margin-safe fallback at a blocker's current position,
  must keep the default 0.
  """
  if prefix is None:
    prefix = _prefix_costs_excluding(seq, (), comp_by_name)
  margins: dict[int, float] = {}
  for async_done, async_start in start_of_done.items():
    if async_start not in positions or async_done not in positions:
      continue
    latency = _profile_costs.get(_resolve_profile_key(async_start, comp_by_name))
    if latency is None or latency <= 0:
      continue
    s, d = positions[async_start], positions[async_done]
    if d <= s + 1 or d <= min_pos:
      continue
    slack = prefix[d] - prefix[s + 1] - latency
    for p in range(max(s + 1, min_pos), d):
      if p not in margins or slack < margins[p]:
        margins[p] = slack
  return margins


_GIVE_UP_MODES = ("none", "streak", "floor_aware")


class _StuckCandidateTracker:
  """Decides when a repeatedly blocked candidate is skipped so others get a
  turn. Strategies (compared per computation by
  _fill_exposed_collectives_best_of):

  - "none": never give up.
  - "streak": counts consecutive restarts blocked on the same candidate; a
    different name resets it. Cheap and safest in small while bodies.
  - "floor_aware": per-candidate, counting a repeat only if its floor failed
    to improve. Better on the large entry computation, where
    multi-hop chains converge slowly.
  """

  def __init__(self, give_up_mode: str):
    self.give_up_mode = give_up_mode
    self._last_name: str | None = None
    self._last_streak = 0
    self._streak_by_name: dict[str, int] = {}
    self._last_floor_by_name: dict[str, int] = {}

  def note_and_maybe_give_up(self, name: str, floor: int) -> tuple[int, bool]:
    """Record that `name` was found blocked at `floor`; returns (streak, give_up)."""
    if self.give_up_mode == "none":
      return 0, False
    if self.give_up_mode == "streak":
      if name == self._last_name:
        self._last_streak += 1
      else:
        self._last_name = name
        self._last_streak = 1
      streak = self._last_streak
    elif self.give_up_mode == "floor_aware":
      prev_floor = self._last_floor_by_name.get(name)
      if prev_floor is not None and floor >= prev_floor:
        self._streak_by_name[name] = self._streak_by_name.get(name, 0) + 1
      else:
        self._streak_by_name[name] = 1
      self._last_floor_by_name[name] = floor
      streak = self._streak_by_name[name]
    else:
      raise ValueError(f"unknown give_up_mode: {self.give_up_mode!r}")
    return streak, streak >= _STUCK_CANDIDATE_GIVE_UP_STREAK


def _fill_exposed_collectives_with_heavy_compute(
    seq: list,
    schedule,
    comp,
    start_of_done: dict,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    module_name: str,
    give_up_mode: str = "streak",
    is_while_body: bool = False,
) -> tuple[bool, list, dict, dict]:
  """Pull heavy compute backward into still-exposed collective windows.

  Outer fixed point (up to _MAX_FILL_FIXED_POINT_ITERS passes): each pass
  rebuilds the worklist from EVERY collective's fresh deficit, earliest start
  first, because filling or chasing one window can steal coverage from
  another that was already hidden. A window
  whose deficit is unchanged since the last pass was already scanned to
  exhaustion and is skipped. Stops when a pass changes nothing.

  For each window, scan forward from its `done` for a heavy instruction
  (cost >= _HEAVY_COMPUTE_MIN_US, not trivial) whose real dependencies
  (_earliest_legal_pos) already sit at or before `done`, and move it into the
  window. This never breaks the candidate's consumers (they already sit after
  its producer). Candidates are restricted to those whose cost fits within
  the slack of the collectives covering their position
  (_compute_position_margins), so a direct placement can never push another
  collective below its latency. A candidate whose floor lies past `done` goes
  to the chase fallbacks instead (see _chase_blocked).

  The deficit is recomputed from the real prefix sums on every iteration. An
  incremental "deficit -= cost" double-credited instructions that a chase
  displaced and then re-placed.
  """
  changed = False
  pass_changed = False
  chase_hops = 0
  # Full-range margins for the chase fallbacks (a blocker's position can be
  # anywhere, unlike the scan's done_pos+1.. queries). Computed lazily once per
  # while-iteration and discarded after a successful chase mutates positions.
  chase_margins = None
  stuck_tracker = None
  given_up_candidates: set = set()
  prev_deficit_by_name: dict[str, float] = {}

  def _chase_blocked(inst, floor, blocker, window, start_pos, done_pos, exposed_dones) -> bool:
    """`inst`'s floor lies past the window's done: record the stall and try
    the two chase fallbacks. Returns True if a chase moved something (the
    caller must restart its scan; positions have shifted).

    Fallback 1 relocates the start of another still-exposed collective whose
    done pins the candidate. Fallback 2 chases an arbitrary movable heavy
    blocker chain (_chase_heavy_compute_blocker_chain).
    """
    nonlocal seq, positions, name_to_pos, changed, pass_changed, chase_hops, chase_margins
    streak, give_up_now = stuck_tracker.note_and_maybe_give_up(inst.name, floor)
    if give_up_now:
      given_up_candidates.add(inst.name)
      _logger.debug(
          "gpu_collective_overlap_pass [%s]: window %s: giving up on %s after %d unresolved "
          "chase restarts (mode=%s, floor stuck at %d).",
          module_name,
          window["start"].name,
          inst.name,
          streak,
          give_up_mode,
          floor,
      )
    if blocker is not None and chase_hops < _MAX_PRODUCER_RELOCATE_HOPS:
      moved, seq, positions, name_to_pos = _chase_exposed_collective_blocker_earlier(
          blocker,
          start_of_done,
          exposed_dones,
          start_pos,
          done_pos,
          seq,
          schedule,
          comp,
          positions,
          name_to_pos,
          comp_by_name,
      )
      if moved:
        chase_hops += 1
        changed = pass_changed = True
        _logger.info(
            "gpu_collective_overlap_pass [%s]: chased exposed-collective blocker %s earlier "
            "to unblock %s for window %s (deficit %.1f us).",
            module_name,
            start_of_done[blocker].name,
            inst.name,
            window["start"].name,
            window["deficit"],
        )
        return True
    if _MAX_HEAVY_COMPUTE_CHASE_HOPS > 0 and chase_hops < _MAX_PRODUCER_RELOCATE_HOPS:
      if chase_margins is None:
        chase_margins = _compute_position_margins(
            start_of_done,
            positions,
            seq,
            comp_by_name,
        )
      moved, seq, positions, name_to_pos, _, _, _ = _chase_heavy_compute_blocker_chain(
          inst,
          comp,
          schedule,
          seq,
          positions,
          name_to_pos,
          comp_by_name,
          window["start"],
          done_pos,
          window["latency"],
          start_of_done,
          _MAX_HEAVY_COMPUTE_CHASE_HOPS,
          module_name,
          chase_margins,
          is_while_body=is_while_body,
      )
      if moved:
        chase_hops += 1
        changed = pass_changed = True
        return True
    return False

  for _fp_iter in range(_MAX_FILL_FIXED_POINT_ITERS):
    pass_changed = False

    windows: list[dict] = []
    cur_deficit_by_name: dict[str, float] = {}
    worklist_prefix = _prefix_costs_excluding(seq, (), comp_by_name)
    for async_done, async_start in start_of_done.items():
      if async_start not in positions or async_done not in positions:
        continue
      profile_key = _resolve_profile_key(async_start, comp_by_name)
      latency = _profile_costs.get(profile_key)
      if latency is None or latency <= 0:
        continue
      start_pos = positions[async_start]
      done_pos = positions[async_done]
      overlap = worklist_prefix[done_pos] - worklist_prefix[start_pos + 1]
      deficit = latency - overlap
      cur_deficit_by_name[async_start.name] = deficit
      if deficit > 0:
        prev = prev_deficit_by_name.get(async_start.name)
        windows.append(
            {
                "start": async_start,
                "done": async_done,
                "deficit": deficit,
                "is_async": profile_key.startswith(_ASYNC_COLLECTIVE_PREFIXES),
                "latency": latency,
                "skip": _fp_iter > 0 and prev is not None and abs(prev - deficit) < 1e-6,
            }
        )

    prev_deficit_by_name = cur_deficit_by_name

    if not windows:
      break

    windows.sort(key=lambda w: positions[w["start"]])
    excluded = set(start_of_done.keys()) | set(start_of_done.values())
    # Only genuinely under-hidden collectives are chase targets, so
    # adequately overlapped ones aren't disturbed.
    exposed_dones = {w["done"] for w in windows}

    # Windows can overlap in time; an instruction placed in a shared
    # zone counts toward both, so such placements are preferred. Best-effort
    # snapshot per pass.
    sibling_ranges: dict[str, list[tuple[int, int]]] = {}
    for w in windows:
      s, d = positions[w["start"]], positions[w["done"]]
      for other in windows:
        if other is w:
          continue
        os_, od = positions[other["start"]], positions[other["done"]]
        if os_ < d and s < od:
          sibling_ranges.setdefault(w["start"].name, []).append((os_, od))

    for window in windows:
      if window["skip"]:
        continue
      window_siblings = sibling_ranges.get(window["start"].name, [])
      chase_hops = 0
      # One window is drained fully before the next.
      stuck_tracker = _StuckCandidateTracker(give_up_mode)
      given_up_candidates = set()
      while window["deficit"] > 0:
        done_pos = positions[window["done"]]
        start_pos = positions[window["start"]]
        iter_prefix = _prefix_costs_excluding(seq, (), comp_by_name)
        window["deficit"] = max(0.0, window["latency"] - (iter_prefix[done_pos] - iter_prefix[start_pos + 1]))
        if window["deficit"] <= 0:
          break
        # The scan only queries positions >= done_pos + 1.
        position_margins = _compute_position_margins(
            start_of_done,
            positions,
            seq,
            comp_by_name,
            prefix=iter_prefix,
            min_pos=done_pos + 1,
        )
        chase_margins = None
        candidate = None
        cand_floor = None
        cand_chain: list = []
        restart = False

        # async windows prefer te_gemm candidates: of all legally
        # reachable ones take the earliest floor (it depends on the
        # dependency chain, not scan order), so the GEMM overlaps more
        # of the transfer. Falls through to the unrestricted scan if
        # none is reachable.
        if window["is_async"]:
          best_floor = None
          for i in range(done_pos + 1, len(seq)):
            inst = seq[i]
            if inst in excluded or inst.name in given_up_candidates:
              continue
            if _is_trivially_movable_inst(inst, comp_by_name):
              continue
            if not _is_te_gemm_custom_call(inst, comp_by_name):
              continue
            cost = _resolve_inst_cost(inst, comp_by_name)
            if cost < _HEAVY_COMPUTE_MIN_US:
              continue
            if cost > position_margins.get(i, float("inf")):
              continue
            floor, chain, blocker = _earliest_legal_pos(inst, positions, name_to_pos, comp_by_name)
            if floor > done_pos:
              if _chase_blocked(inst, floor, blocker, window, start_pos, done_pos, exposed_dones):
                restart = True
                break
              continue
            if best_floor is None or floor < best_floor:
              best_floor = floor
              candidate = inst
              cand_floor = floor
              cand_chain = chain
          if restart:
            continue

        if candidate is None:
          # Collect every legal candidate rather than taking the
          # first. Best fit is picked below.
          legal_candidates: list[tuple] = []
          for i in range(done_pos + 1, len(seq)):
            inst = seq[i]
            if inst in excluded or inst.name in given_up_candidates:
              continue
            if _is_trivially_movable_inst(inst, comp_by_name):
              continue
            cost = _resolve_inst_cost(inst, comp_by_name)
            if cost < _HEAVY_COMPUTE_MIN_US:
              continue
            if cost > position_margins.get(i, float("inf")):
              continue
            floor, chain, blocker = _earliest_legal_pos(inst, positions, name_to_pos, comp_by_name)
            if floor > done_pos:
              if _chase_blocked(inst, floor, blocker, window, start_pos, done_pos, exposed_dones):
                restart = True
                break
              continue
            legal_candidates.append((inst, floor, chain, cost))
          if restart:
            continue

          if legal_candidates:

            # Only called within this iteration, so the loop-variable capture is safe.
            # pylint: disable=cell-var-from-loop
            def _lands_in_sibling(floor: int) -> bool:
              return any(s <= floor < d for s, d in window_siblings)

            sufficient = [c for c in legal_candidates if c[3] >= window["deficit"]]
            if sufficient:
              # Smallest candidate that closes the deficit,
              # preferring one landing in a sibling window.
              candidate, cand_floor, cand_chain, _ = min(
                  sufficient,
                  key=lambda c: (not _lands_in_sibling(c[1]), c[3]),
              )
            else:
              candidate, cand_floor, cand_chain, _ = max(
                  legal_candidates,
                  key=lambda c: (_lands_in_sibling(c[1]), c[3]),
              )

        if candidate is None:
          break

        prev_deficit = window["deficit"]
        target = max(cand_floor, start_pos + 1)
        seq, positions, name_to_pos = _commit_sequence(schedule, comp, _reinsert(seq, cand_chain, target, candidate))
        changed = pass_changed = True
        _logger.info(
            "gpu_collective_overlap_pass [%s]: relocated heavy compute %s (cost %.1f us) into "
            "window of %s (pre-relocation deficit %.1f us).",
            module_name,
            candidate.name,
            _resolve_inst_cost(candidate, comp_by_name),
            window["start"].name,
            prev_deficit,
        )
    if not pass_changed:
      break
  return changed, seq, positions, name_to_pos


# ---------------------------------------------------------------------------
# Schedule reordering
# ---------------------------------------------------------------------------
_WHILE_CALLS_RE = re.compile(r"(?:condition|body)=%([A-Za-z0-9_.]+)")
_WHILE_BODY_RE = re.compile(r"\bbody=%([A-Za-z0-9_.]+)")


def _innermost_first_computations(module, schedule) -> tuple[list, set]:
  """Return (computations, while_body_comps).

  computations are the scheduled non-fusion computations in innermost-first
  DFS order (a while body before the computation holding its while), the
  entry computation last. while_body_comps is every computation called as a
  while body, i.e. all but the true entry.
  """
  all_comps = [c for c in module.make_nonfusion_computations() if schedule.sequence(c) is not None]
  # Restricted to this accessor call so lookups stay identity-consistent with
  # all_comps (HloComputation has no custom __eq__/__hash__).
  local_comp_by_name = {c.name: c for c in all_comps}

  children: dict = {c: [] for c in all_comps}
  all_callees: set = set()
  for comp in all_comps:
    for inst in schedule.sequence(comp):
      if _opcode_str(inst) != "while":
        continue
      try:
        text = inst.to_string()
      except Exception:
        continue
      for name in _WHILE_CALLS_RE.findall(text):
        called = local_comp_by_name.get(name)
        if called is not None and schedule.sequence(called) is not None:
          children[comp].append(called)
          all_callees.add(called)

  roots = [c for c in all_comps if c not in all_callees]
  if not roots:
    return all_comps, all_callees

  def _collect(comp, visited: set, result: list) -> None:
    if comp in visited:
      return
    visited.add(comp)
    for child in children.get(comp, []):
      _collect(child, visited, result)
    result.append(comp)

  visited: set = set()
  result: list = []
  for root in roots:
    _collect(root, visited, result)
  for comp in all_comps:
    if comp not in visited:
      result.append(comp)
  return result, all_callees


def _log_profile_coverage_gaps(module, schedule, module_name: str) -> None:
  """Log fusion/custom-call instructions in while bodies with no PGLE cost
  entry; every cost lookup silently treats a missing one as free."""
  all_comps = [c for c in module.make_nonfusion_computations() if schedule.sequence(c) is not None]
  comp_by_name = {c.name: c for c in all_comps}
  while_bodies: set = set()
  for comp in all_comps:
    for inst in schedule.sequence(comp):
      if _opcode_str(inst) != "while":
        continue
      try:
        text = inst.to_string()
      except Exception:
        continue
      for name in _WHILE_BODY_RE.findall(text):
        body = comp_by_name.get(name)
        if body is not None:
          while_bodies.add(body)

  if not while_bodies:
    return

  missing: list[str] = []
  total = 0
  for comp in while_bodies:
    for inst in schedule.sequence(comp):
      opc = _opcode_str(inst)
      if opc not in ("fusion", "custom-call"):
        continue
      total += 1
      if inst.name not in _profile_costs:
        missing.append(f"{inst.name}({opc})")

  _logger.info(
      "gpu_collective_overlap_pass [%s]: profile coverage in %d while body(ies): %d/%d "
      "fusion/custom-call instructions have a PGLE cost (%d missing).",
      module_name,
      len(while_bodies),
      total - len(missing),
      total,
      len(missing),
  )
  if missing:
    _logger.info(
        "gpu_collective_overlap_pass [%s]: missing-profile instructions (first 150 of %d): %s",
        module_name,
        len(missing),
        ", ".join(missing[:150]),
    )


# Which _GIVE_UP_MODES strategy won per computation, so the ~3x-cost comparison
# runs once per _reorder_to_fixed_point call rather than on every re-entry
# Cleared at the start of each _reorder_to_fixed_point call.
_fill_strategy_cache: dict[str, str] = {}


def _fill_exposed_collectives_best_of(
    seq: list,
    schedule,
    comp,
    start_of_done: dict,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    module_name: str,
    is_while_body: bool = False,
) -> tuple[bool, list, dict, dict]:
  """Run the fill with each _GIVE_UP_MODES strategy from the same baseline and
  keep the one with the lowest _total_exposed_us.

  No strategy dominates: "streak" wins on small while bodies,
  "floor_aware" on the large entry computation. Re-entries for
  a computation replay the cached winner with a single trial.
  """
  cached_mode = _fill_strategy_cache.get(comp.name)
  if cached_mode is not None:
    pre_total = _total_exposed_us(start_of_done, positions, seq, comp_by_name)
    changed, seq, positions, name_to_pos = _fill_exposed_collectives_with_heavy_compute(
        list(seq),
        schedule,
        comp,
        start_of_done,
        dict(positions),
        dict(name_to_pos),
        comp_by_name,
        module_name,
        give_up_mode=cached_mode,
        is_while_body=is_while_body,
    )
    _logger.info(
        "gpu_collective_overlap_pass [%s]: comp %s: cached fill strategy '%s': total exposed "
        "%.1f -> %.1f us (changed=%s).",
        module_name,
        comp.name,
        cached_mode,
        pre_total,
        _total_exposed_us(start_of_done, positions, seq, comp_by_name),
        changed,
    )
    return changed, seq, positions, name_to_pos

  baseline_seq = list(seq)
  baseline_positions = dict(positions)
  baseline_name_to_pos = dict(name_to_pos)

  best: Optional[tuple] = None  # (total_exposed, changed, seq, positions, name_to_pos, mode)
  for mode in _GIVE_UP_MODES:
    schedule.set_sequence(comp, list(baseline_seq))
    trial_changed, trial_seq, trial_positions, trial_name_to_pos = _fill_exposed_collectives_with_heavy_compute(
        list(baseline_seq),
        schedule,
        comp,
        start_of_done,
        dict(baseline_positions),
        dict(baseline_name_to_pos),
        comp_by_name,
        module_name,
        give_up_mode=mode,
        is_while_body=is_while_body,
    )
    total = _total_exposed_us(start_of_done, trial_positions, trial_seq, comp_by_name)
    _logger.info(
        "gpu_collective_overlap_pass [%s]: fill strategy '%s' for comp %s: total exposed %.1f us (changed=%s).",
        module_name,
        mode,
        comp.name,
        total,
        trial_changed,
    )
    if best is None or total < best[0] - 1e-6:  # pylint: disable=unsubscriptable-object
      best = (total, trial_changed, trial_seq, trial_positions, trial_name_to_pos, mode)

  total, changed, seq, positions, name_to_pos, mode = best
  schedule.set_sequence(comp, seq)
  _fill_strategy_cache[comp.name] = mode
  _logger.info(
      "gpu_collective_overlap_pass [%s]: comp %s: selected fill strategy '%s' (total exposed %.1f us).",
      module_name,
      comp.name,
      mode,
      total,
  )
  return changed, seq, positions, name_to_pos


def _may_hoist_control_pred(pred, positions, comp_by_name) -> bool:
  """Whether control-predecessor `pred` may move earlier.

  Anything may, except the done of an async or te_ep collective. The recursive
  hoist reached such a done through a consumer right after it and pulled it
  next to its start.
  """
  opc = _opcode_str(pred)
  if not (opc == "async-done" or opc.endswith("-done")):
    return True
  ops = list(pred.operands())
  start = ops[0] if ops else None
  if start is None or start not in positions:
    return False
  try:
    key = _resolve_profile_key(start, comp_by_name)
  except Exception:
    key = ""
  return not key.startswith(_ASYNC_COLLECTIVE_PREFIXES + (_TE_EP_PREFIX,))


def _simulate_comm_stream(order, cost: dict, lat: dict, done_start: dict, gate=None):
  """Simulate compute (in `order`) against one FIFO communication stream.

  Compute ops run back to back at their cost. Every async start is queued on
  the comm stream and begins at the later of its issue time and the end of the
  previous queued op; an async done stalls compute until its start has
  finished. Returns (total stall us, {done: stall us}).

  NCCL collectives and custom comm calls share one stream, so a
  collective issued behind a custom comm call waits for it and vice versa; the
  static window cost used elsewhere sees neither effect.

  `gate` is an optional set of SM-saturating GEMMs: a collective can't begin
  while one is running. The GEMM timeline depends on the stalls, which depend
  on the collective timeline, so it is iterated from the previous pass's GEMM
  intervals (a few passes converge).
  """
  intervals: list = []
  total = 0.0
  stalls: dict = {}
  for _ in range(3 if gate else 1):
    t = 0.0
    comm = 0.0
    end: dict = {}
    total = 0.0
    stalls = {}
    found: list = []
    for inst in order:
      start = done_start.get(inst)
      if start is not None:
        e = end.get(start)
        if e is not None and e > t:
          stalls[inst] = e - t
          total += e - t
          t = e
      elif inst in lat:
        begin = t if t > comm else comm
        for gs, ge in intervals:
          if gs < begin < ge:
            begin = ge
        comm = begin + lat[inst]
        end[inst] = comm
      else:
        c = cost.get(inst, 0.0)
        if gate and inst in gate:
          found.append((t, t + c))
        t += c
    if not gate or found == intervals:
      break
    intervals = found
  return total, stalls


def _sim_inst_cost(inst, comp_by_name: dict) -> float:
  """_resolve_inst_cost, but with the per-invocation GEMM costs (_gemm_overrides)."""
  if not _gemm_overrides:
    return _resolve_inst_cost(inst, comp_by_name)
  direct = _gemm_overrides.get(inst.name)
  if direct:
    return direct
  if _opcode_str(inst) == "fusion":
    try:
      text = inst.to_string()
      if "kind=kCustom" in text:
        m = _CALLS_RE.search(text)
        comp = comp_by_name.get(m.group(1)) if m else None
        if comp is not None:
          total = sum(_gemm_overrides.get(sub.name) or _profile_costs.get(sub.name, 0.0) for sub in comp.instructions())
          if total:
            return total
    except Exception:
      pass
  return _resolve_inst_cost(inst, comp_by_name)


def _best_async_placement(
    async_start,
    async_done,
    floor,
    to_move,
    seq,
    positions,
    cost,
    lat,
    done_start,
    ctrl_succs,
    protect=None,
    gate=None,
    diag=None,
):
  """Choose where to put an async start (and its done) to minimize simulated stalls.

  Candidates are insertion points for the start between its earliest legal
  position and the first consumer of its done. The done may stay put or move
  to just before its first consumer, so a start placed after its current done
  shifts the done along with it. Topological order is kept: the start after
  its operands and control-predecessors, the done after the start and before
  its consumers and control-successors. A start moving earlier brings its
  trivial operand chain (`to_move`); one moving later leaves it.

  Candidates within _ASYNC_STREAM_TOL_US of the best simulated total count as
  equal; among them the earliest start wins, then the latest done (largest
  window; a later done never adds a stall in the sim).

  `protect` is an optional set of dones (the async-class ones): a candidate is
  dropped if it raises their combined stall by more than the tolerance, so
  trading one async collective's exposure for another's or a te_ep stall never
  counts as a gain.

  Returns ((total, done_moved, base_idx), moved_chain, start_idx, done_idx, base)
  or None; indices are into `base` (seq without the moved instructions).
  """
  start_pos = positions[async_start]
  done_pos = positions[async_done]
  consumers = [positions[u] for u in async_done.users() if u in positions]
  consumers += [positions[x] for x in ctrl_succs.get(async_done.name, []) if x in positions]
  if not consumers:
    return None
  user_pos = min(consumers)
  p_max = min(
      [user_pos] + [positions[x] for x in ctrl_succs.get(async_start.name, []) if x in positions and x is not async_done]
  )
  cands = []
  protected_now = 0.0
  if protect:
    _, stalls_now = _simulate_comm_stream(seq, cost, lat, done_start, gate)
    protected_now = sum(stalls_now.get(d, 0.0) for d in protect)
  chain = sorted(to_move, key=lambda x: positions[x])
  chain_pos = [positions[x] for x in chain]
  prepared: dict = {}

  def prepare(k):
    # Variant k moves chain[k:] with the start; chain[:k] already sit before
    # the insertion point and stay, so nothing moves later than it is now
    # (other users of a chain member may sit in between).
    if k not in prepared:
      rem = {async_start, async_done} | set(chain[k:])
      removed_before = [0] * (len(seq) + 1)
      count = 0
      for i, inst in enumerate(seq):
        removed_before[i] = count
        if inst in rem:
          count += 1
      removed_before[len(seq)] = count
      prepared[k] = ([inst for inst in seq if inst not in rem], removed_before)
    return prepared[k]

  for early in (True, False):
    lo, hi = (floor, start_pos) if early else (start_pos + 1, p_max)
    hi = min(hi, p_max)
    for i in range(lo, hi + 1):
      # Interior points only matter next to heavy compute or an async op.
      if i < len(seq) and i not in (lo, hi, start_pos):
        inst = seq[i]
        if not (cost.get(inst, 0.0) >= _HEAVY_COMPUTE_MIN_US or inst in lat or inst in done_start):
          continue
      k = bisect.bisect_left(chain_pos, i) if early else len(chain)
      base, removed_before = prepare(k)
      moved = chain[k:] if early else []
      q_latest = user_pos - removed_before[user_pos]
      q_cur = done_pos - removed_before[done_pos]
      pb = i - removed_before[i] if i < len(seq) else len(base)
      for q in sorted({q_latest, q_cur}):
        if q < pb:
          continue
        order = base[:pb] + moved + [async_start] + base[pb:q] + [async_done] + base[q:]
        total, stalls = _simulate_comm_stream(order, cost, lat, done_start, gate)
        if protect and sum(stalls.get(d, 0.0) for d in protect) > protected_now + _ASYNC_STREAM_TOL_US:
          if diag is not None:
            diag.append((total, i, True))
          continue
        if diag is not None:
          diag.append((total, i, False))
        cands.append((total, q != q_cur, pb, k, q, moved, base))
  if not cands:
    return None
  limit = min(c[0] for c in cands) + _ASYNC_STREAM_TOL_US
  total, _moved, pb, _, q, moved, base = min((c for c in cands if c[0] <= limit), key=lambda c: (c[2], -c[4]))
  return ((round(total, 1), _moved, pb), moved, pb, q, base)


# Recursion bound when hoisting a control-predecessor whose own latest operand
# must move first.
_MAX_PRED_HOIST_DEPTH = 8


def _hoist_async_starts_to_floor(
    seq,
    schedule,
    comp,
    start_of_done,
    positions,
    name_to_pos,
    comp_by_name,
    module_name,
):
  """Move each async start in a while body to its earliest legal position.

  Runs after the fill step. The per-collective loop in _reorder_collectives skips a
  collective once its static window looks hidden, and later fill/chase moves
  can land compute ahead of it with no dependency on it.

  With the stream model on, placement is chosen by _best_async_placement
  (which may also move the done); otherwise the start goes to its floor and
  the done stays put, so the window only grows.
  """
  changed = False

  def _move_earlier(inst, depth) -> bool:
    """Move inst to just after its latest operand/control-predecessor,
    hoisting that one first (recursively, up to `depth`) if it sits right
    before inst."""
    nonlocal seq, positions, name_to_pos, changed
    pos = positions[inst]
    floor, binder = 0, None
    for op in inst.operands():
      p = positions.get(op)
      if p is not None and p + 1 > floor:
        floor, binder = p + 1, op
    for name in _control_predecessor_names(inst):
      p = name_to_pos.get(name)
      if p is not None and p + 1 > floor:
        floor, binder = p + 1, seq[p]
    if floor < pos:
      if not _may_hoist_control_pred(inst, positions, comp_by_name):
        return False
      new_seq = [i for i in seq if i is not inst]
      if len(new_seq) != len(seq) - 1:
        _logger.warning(
            "gpu_collective_overlap_pass [%s]: HOISTDIAG %s not found exactly once in sequence; skipping move.",
            module_name,
            inst.name,
        )
        return False
      new_seq.insert(floor, inst)
      seq, positions, name_to_pos = _commit_sequence(schedule, comp, new_seq)
      changed = True
      return True
    if binder is None or depth <= 0:
      return False
    if not _move_earlier(binder, depth - 1):
      return False
    return _move_earlier(inst, depth - 1)

  sim_cost: dict = {}
  sim_lat: dict = {}
  async_dones: set = set()
  sim_gate: set = set()
  n_lat_overridden = 0
  ctrl_succs: dict = {}
  if _ASYNC_STREAM_MODEL:
    for inst in seq:
      if _is_async_start(inst) or _opcode_str(inst).endswith("-done"):
        sim_cost[inst] = 0.0
      else:
        sim_cost[inst] = _sim_inst_cost(inst, comp_by_name)
        if _SM_GATE and sim_cost[inst] >= _SM_GATE_MIN_US and _is_te_gemm_custom_call(inst, comp_by_name):
          sim_gate.add(inst)
      for name in _control_predecessor_names(inst):
        ctrl_succs.setdefault(name, []).append(inst)
    for d_inst, s_inst in start_of_done.items():
      try:
        key_s = _resolve_profile_key(s_inst, comp_by_name)
        sim_lat[s_inst] = _collective_overrides.get(key_s) or _profile_costs.get(key_s, 0.0) or 0.0
        if key_s in _collective_overrides:
          n_lat_overridden += 1
        if key_s.startswith(_ASYNC_COLLECTIVE_PREFIXES):
          async_dones.add(d_inst)
      except Exception:
        sim_lat[s_inst] = 0.0
    _logger.info(
        "gpu_collective_overlap_pass [%s]: stream model: %s simulated total exposed %.1f us before async placement "
        "(%d collective latencies from per-invocation clustering).",
        module_name,
        comp.name,
        _simulate_comm_stream(seq, sim_cost, sim_lat, start_of_done, sim_gate)[0],
        n_lat_overridden,
    )
  for async_done, async_start in sorted(start_of_done.items(), key=lambda kv: positions[kv[1]]):
    profile_key = _resolve_profile_key(async_start, comp_by_name)
    if not profile_key.startswith(_ASYNC_COLLECTIVE_PREFIXES):
      continue
    orig_pos = positions[async_start]
    for _ in range(_MAX_PRODUCER_RELOCATE_HOPS):
      cp_pins: list = []
      floor, to_move, _ = _earliest_legal_pos(
          async_start,
          positions,
          name_to_pos,
          comp_by_name,
          cp_pins,
      )
      if not cp_pins:
        break
      pin_pos, pin_name = max(cp_pins)
      if pin_pos + 1 < floor or pin_pos + 1 < 1:
        break  # a data dependency pins the floor, not a control-predecessor
      if not _move_earlier(seq[pin_pos], _MAX_PRED_HOIST_DEPTH):
        _logger.debug(
            "gpu_collective_overlap_pass [%s]: %s floor %d pinned by control-predecessor %s "
            "(pos %d) which could not be hoisted.",
            module_name,
            async_start.name,
            floor,
            pin_name,
            pin_pos,
        )
        break
    floor, to_move, _ = _earliest_legal_pos(async_start, positions, name_to_pos, comp_by_name)
    start_pos = positions[async_start]
    if _ASYNC_STREAM_MODEL and sim_lat.get(async_start, 0.0) > 0:
      before = _simulate_comm_stream(seq, sim_cost, sim_lat, start_of_done, sim_gate)[0]
      cand_diag: list = []
      best = _best_async_placement(
          async_start,
          async_done,
          floor,
          to_move,
          seq,
          positions,
          sim_cost,
          sim_lat,
          start_of_done,
          ctrl_succs,
          protect=async_dones,
          gate=sim_gate,
          diag=cand_diag,
      )
      if cand_diag:
        top = sorted(cand_diag)[:4]
        later = sorted((c for c in cand_diag if c[1] > start_pos))[:2]

        def fmt(cs):
          return ", ".join(f"seq idx {c[1]}: {c[0]:.1f} us{' [dropped]' if c[2] else ''}" for c in cs) or "none"

        _logger.debug(
            "gpu_collective_overlap_pass [%s]: STREAMCANDS %s: %d candidates (%d dropped by async-stall guard); "
            "current %.1f us; lowest %s; lowest later than current %s.",
            module_name,
            async_start.name,
            len(cand_diag),
            sum(1 for c in cand_diag if c[2]),
            before,
            fmt(top),
            fmt(later),
        )
      _logger.debug(
          "gpu_collective_overlap_pass [%s]: STREAMDIAG %s: start pos %d, done pos %d, floor %d, to_move %d, "
          "latency %.1f us, simulated total %.1f us; best %s.",
          module_name,
          async_start.name,
          start_pos,
          positions[async_done],
          floor,
          len(to_move),
          sim_lat.get(async_start, 0.0),
          before,
          "none"
          if best is None
          else f"total {best[0][0]:.1f} us at base idx {best[2]}, done idx {best[3]}, " f"{len(best[1])} chain ops moved",
      )
      if best is not None:
        key, moved, pb, q, base = best
        if key[0] <= before + _ASYNC_STREAM_TOL_US:
          done_pos_old = positions[async_done]
          new_seq = base[:pb] + list(moved) + [async_start] + base[pb:q] + [async_done] + base[q:]
          if len(new_seq) == len(seq) and set(new_seq) == set(seq) and new_seq != seq:
            seq, positions, name_to_pos = _commit_sequence(schedule, comp, new_seq)
            changed = True
            _logger.info(
                "gpu_collective_overlap_pass [%s]: stream model placed %s: start pos %d -> %d, done pos %d -> %d "
                "(simulated total exposed %.1f -> %.1f us).",
                module_name,
                async_start.name,
                orig_pos,
                positions[async_start],
                done_pos_old,
                positions[async_done],
                before,
                key[0],
            )
        continue
    if floor + len(to_move) >= start_pos:
      continue
    seq, positions, name_to_pos = _commit_sequence(schedule, comp, _reinsert(seq, to_move, floor, async_start))
    changed = True
    _logger.info(
        "gpu_collective_overlap_pass [%s]: hoisted %s from pos %d to %d with %d relocated operand(s).",
        module_name,
        async_start.name,
        orig_pos,
        positions[async_start],
        len(to_move),
    )
  return changed, seq, positions, name_to_pos


def _reorder_collectives(module, schedule, module_name: str) -> bool:
  """Move async collective starts earlier where latency is under-hidden.

  Returns True if the schedule changed.
  """
  changed = False
  # Includes fusion sub-computations, unlike make_nonfusion_computations().
  comp_by_name = {c.name: c for c in module.computations()}

  ordered_comps, while_body_comps = _innermost_first_computations(module, schedule)
  for comp in ordered_comps:
    is_while_body = comp in while_body_comps
    if _WHILE_BODY_ONLY and not is_while_body:
      continue
    seq = list(schedule.sequence(comp))

    start_of_done: dict = {}
    for inst in seq:
      if _is_async_start(inst):
        users = list(inst.users())
        if len(users) == 1:
          start_of_done[users[0]] = inst
        else:
          _logger.debug(
              "gpu_collective_overlap_pass [%s]: async-start %s has %d users (expected its done only); skipping.",
              module_name,
              inst.name,
              len(users),
          )

    if not start_of_done:
      continue

    positions = {inst: i for i, inst in enumerate(seq)}
    name_to_pos = {inst.name: i for inst, i in positions.items()}

    for async_done, async_start in start_of_done.items():
      profile_key = _resolve_profile_key(async_start, comp_by_name)
      collective_latency = _profile_costs.get(profile_key)
      if collective_latency is None or collective_latency <= 0:
        _logger.debug(
            "gpu_collective_overlap_pass [%s]: no profile entry for %s (resolved key: %s).",
            module_name,
            async_start.name,
            profile_key,
        )
        continue

      async_start_pos = positions[async_start]
      async_done_pos = positions[async_done]
      is_async = profile_key.startswith(_ASYNC_COLLECTIVE_PREFIXES)
      stream_owns_done = _ASYNC_STREAM_MODEL and is_while_body

      current_overlap = sum(_resolve_inst_cost(seq[i], comp_by_name) for i in range(async_start_pos + 1, async_done_pos))
      if current_overlap >= collective_latency:
        if is_async and not stream_owns_done:
          done_moved, seq, positions, name_to_pos = _relocate_async_done_before_te_ep(
              async_start,
              async_done,
              comp,
              schedule,
              seq,
              positions,
              name_to_pos,
              comp_by_name,
              collective_latency,
              module_name,
          )
          changed = changed or done_moved
        continue

      deficit = collective_latency - current_overlap
      _logger.debug(
          "gpu_collective_overlap_pass [%s]: %s deficit=%.1f us (latency=%.1f us, overlap=%.1f us).",
          module_name,
          async_start.name,
          deficit,
          collective_latency,
          current_overlap,
      )

      # Always move all the way to the floor: earlier only adds headroom
      # and puts the collective ahead of heavy compute it doesn't depend on.
      floor, to_move, _ = _earliest_legal_pos(async_start, positions, name_to_pos, comp_by_name)
      orig_async_start_pos = async_start_pos

      if is_async:
        # The floor stops at the first non-trivial producer; try moving
        # that producer out of the way too.
        (
            chased_blocker,
            seq,
            positions,
            name_to_pos,
            floor,
            to_move,
            _,
        ) = _chase_async_start_blocker_toward_heavy_compute(
            async_start,
            comp,
            schedule,
            seq,
            positions,
            name_to_pos,
            comp_by_name,
            _MAX_PRODUCER_RELOCATE_HOPS,
            module_name,
        )
        if chased_blocker:
          changed = True
          async_start_pos = positions[async_start]
          async_done_pos = positions[async_done]

      # The chain lands right before async_start, so its post-move position
      # is floor + len(to_move); comparing the bare floor would treat an
      # already-packed start as movable.
      if floor + len(to_move) >= async_start_pos:
        continue

      seq, positions, name_to_pos = _commit_sequence(schedule, comp, _reinsert(seq, to_move, floor, async_start))
      changed = True

      new_overlap = sum(
          _resolve_inst_cost(seq[i], comp_by_name) for i in range(positions[async_start] + 1, positions[async_done])
      )
      _logger.info(
          "gpu_collective_overlap_pass: moving %s from pos %d to %d with %d relocated operand(s) "
          "(overlap %.1f -> %.1f us, latency=%.1f us).",
          async_start.name,
          orig_async_start_pos,
          positions[async_start],
          len(to_move),
          current_overlap,
          new_overlap,
          collective_latency,
      )

      if new_overlap >= collective_latency and is_async and not stream_owns_done:
        done_moved, seq, positions, name_to_pos = _relocate_async_done_before_te_ep(
            async_start,
            async_done,
            comp,
            schedule,
            seq,
            positions,
            name_to_pos,
            comp_by_name,
            collective_latency,
            module_name,
        )
        changed = changed or done_moved

    heavy_changed, seq, positions, name_to_pos = _fill_exposed_collectives_best_of(
        seq,
        schedule,
        comp,
        start_of_done,
        positions,
        name_to_pos,
        comp_by_name,
        module_name,
        is_while_body=is_while_body,
    )
    changed = changed or heavy_changed

    if is_while_body and _HOIST_ASYNC_STARTS:
      hoist_changed, seq, positions, name_to_pos = _hoist_async_starts_to_floor(
          seq,
          schedule,
          comp,
          start_of_done,
          positions,
          name_to_pos,
          comp_by_name,
          module_name,
      )
      changed = changed or hoist_changed

  return changed


# ---------------------------------------------------------------------------
# Top-level POST_SCHEDULER pass
# ---------------------------------------------------------------------------
# Cap on reorder re-runs against a freshly update()'d module. Normal runs
# converge in 1-2 iterations.
_MAX_REORDER_ITERS = 4


def _log_final_exposed_summary(module, schedule, module_name: str, label: str) -> None:
  """Log total and per-collective exposed time computed directly from the live
  schedule: ground truth, since intermediate totals go stale after a later
  schedule.update()/verify()."""
  comp_by_name = {c.name: c for c in module.computations()}
  per_comp = []
  per_collective = []  # (comp_name, collective_name, deficit_us)
  for comp in module.make_nonfusion_computations():
    seq = schedule.sequence(comp)
    if seq is None:
      continue
    seq = list(seq)
    positions = {inst: i for i, inst in enumerate(seq)}
    start_of_done: dict = {}
    for inst in seq:
      if _is_async_start(inst):
        users = list(inst.users())
        if len(users) == 1:
          start_of_done[users[0]] = inst
    if not start_of_done:
      continue

    prefix = _prefix_costs_excluding(seq, (), comp_by_name)
    comp_total = 0.0
    for async_done, async_start in start_of_done.items():
      if async_start not in positions or async_done not in positions:
        continue
      latency = _profile_costs.get(_resolve_profile_key(async_start, comp_by_name))
      if latency is None or latency <= 0:
        continue
      s, d = positions[async_start], positions[async_done]
      overlap = prefix[d] - prefix[s + 1] if d > s + 1 else 0.0
      deficit = max(0.0, latency - overlap)
      if deficit > 0:
        per_collective.append((comp.name, async_start.name, deficit))
        comp_total += deficit
    if comp_total > 0:
      per_comp.append((comp.name, comp_total))
  per_comp.sort(key=lambda x: -x[1])
  per_collective.sort(key=lambda x: -x[2])
  _logger.info(
      "gpu_collective_overlap_pass [%s]: FINAL exposed summary [%s]: grand_total=%.1f us across %d computation(s): %s",
      module_name,
      label,
      sum(t for _, t in per_comp),
      len(per_comp),
      ", ".join(f"{name}={total:.1f}us" for name, total in per_comp),
  )
  _logger.info(
      "gpu_collective_overlap_pass [%s]: FINAL per-collective exposed [%s]: %s",
      module_name,
      label,
      ", ".join(f"{comp_name}/{name}={deficit:.1f}us" for comp_name, name, deficit in per_collective),
  )


def _hoist_host_offload_copy_starts(module, schedule, module_name: str) -> bool:
  """Move every host-offload copy start to its earliest legal position.

  Runs last, after all other relocation, so nothing leaves a copy later than
  its data allows. A D2H copy updates its host buffer in place, so a start
  whose host-buffer operand has other users is left alone (those readers must
  stay ordered before the in-place write). H2D copies only read it.
  """
  if not _HOIST_HOST_OFFLOAD_COPIES or not _host_offload_copy_names:
    return False
  comp_by_name = {c.name: c for c in module.computations()}
  changed = False
  for comp in module.make_nonfusion_computations():
    seq = schedule.sequence(comp)
    if seq is None:
      continue
    seq = list(seq)
    starts = [i for i in seq if i.name in _host_offload_copy_names and _opcode_str(i) == "async-start"]
    if not starts:
      continue
    positions = {inst: i for i, inst in enumerate(seq)}
    name_to_pos = {inst.name: i for inst, i in positions.items()}
    for start in starts:
      operands = list(start.operands())
      # Only the D2H copy writes its host buffer in place; the H2D slice
      # just reads it.
      if _host_offload_copy_kind(start, comp_by_name) == "d2h" and operands and len(list(operands[0].users())) > 1:
        _logger.info(
            "gpu_collective_overlap_pass [%s]: host-offload copy %s: host buffer %s has other users; not hoisting.",
            module_name,
            start.name,
            operands[0].name,
        )
        continue
      floor, movable, _ = _earliest_legal_pos(start, positions, name_to_pos, comp_by_name)
      old_pos = positions[start]
      if floor >= old_pos:
        continue
      new_seq = _reinsert(seq, movable, floor, start)
      # The start lands after its relocated operand chain, so a "move" can
      # leave it at the same index; commit only real moves.
      if new_seq.index(start) >= old_pos:
        continue
      seq, positions, name_to_pos = _commit_sequence(schedule, comp, new_seq)
      changed = True
      _logger.info(
          "gpu_collective_overlap_pass [%s]: hoisted host-offload copy %s in %s from pos %d to %d "
          "with %d relocated operand(s).",
          module_name,
          start.name,
          comp.name,
          old_pos,
          positions[start],
          len(movable),
      )
  return changed


def _reorder_to_fixed_point(module, schedule, module_name: str) -> bool:
  """Run _reorder_collectives to a fixed point; returns True if anything changed.

  schedule.update()/verify() (needed so XLA can canonicalize the schedule after
  raw seq mutations) can itself relocate an instruction whose dependency one
  of our earlier moves invalidated, without _reorder_collectives's bookkeeping
  noticing, so a window scored hidden can end up exposed. Re-running on the
  post-update schedule re-fills it.
  """
  global _fill_strategy_cache
  _fill_strategy_cache = {}
  changed = False
  for _ in range(_MAX_REORDER_ITERS):
    if not _reorder_collectives(module, schedule, module_name):
      break
    changed = True
    schedule.update()
    schedule.verify()
    module.set_schedule(schedule)
  else:
    # Cap reached: the last update() was never rechecked, so run one more pass.
    _logger.warning(
        "gpu_collective_overlap_pass [%s]: reorder fixed point not reached after %d iterations; "
        "running one final pass.",
        module_name,
        _MAX_REORDER_ITERS,
    )
    if _reorder_collectives(module, schedule, module_name):
      changed = True
      schedule.update()
      schedule.verify()
      module.set_schedule(schedule)
  if _hoist_host_offload_copy_starts(module, schedule, module_name):
    changed = True
    schedule.update()
    schedule.verify()
    module.set_schedule(schedule)
  return changed


def _compute_collective_overlap(serialized_hlo: bytes) -> Optional[bytes]:
  """Reorder the schedule using the PGLE profile. Runs on rank 0 only; see
  _gpu_collective_overlap_pass."""
  if not _profile_costs:
    return None

  from jax._src.lib import hlo as _hlo  # pylint: disable=import-outside-toplevel

  module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
  schedule = module.schedule()
  if schedule is None:
    return None

  _register_host_offload_copies(module)
  module_name = module.name

  _log_profile_coverage_gaps(module, schedule, module_name)

  if not _reorder_to_fixed_point(module, schedule, module_name):
    return None
  _log_final_exposed_summary(module, schedule, module_name, "final")
  return module.as_serialized_hlo_module_proto()


def _gpu_collective_overlap_pass(serialized_hlo: bytes) -> Optional[bytes]:
  """Run the pass on rank 0 and share its output bytes with every rank.

  Every rank must compile a byte-identical executable, so only rank 0
  computes and everyone returns the same bytes, through the coordination
  service key-value store.
  """
  global _barrier_invocation_count

  # First statement, as cheap as possible, so the log shows the callback was
  # entered even if a hang happens later in XLA.
  _t_enter = time.monotonic()
  _logger.info(
      "gpu_collective_overlap_pass: ENTER host=%s pid=%d input_bytes=%d",
      socket.gethostname(),
      os.getpid(),
      len(serialized_hlo),
  )

  # Early-out before touching the distributed client: modules with no async op
  # need no synchronization at all.
  from jax._src.lib import hlo as _hlo  # pylint: disable=import-outside-toplevel

  module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
  if not _module_has_interesting_async_ops(module):
    _logger.info(
        "gpu_collective_overlap_pass: EXIT host=%s pid=%d module=%s result=no-op (no async ops; %.1f ms).",
        socket.gethostname(),
        os.getpid(),
        module.name,
        (time.monotonic() - _t_enter) * 1000,
    )
    return None

  client = _get_distributed_client()
  _in_rank = _jax_process_id() if client is not None else 0
  dump_modules = os.environ.get("COLLECTIVE_OVERLAP_DUMP_MODULE", "1") == "1"

  if dump_modules:
    # Every rank's raw input, so ranks' modules can be diffed.
    _module_text = module.to_string()
    _in_host = socket.gethostname()
    _in_pid = os.getpid()
    _content_digest = hashlib.sha256(_module_text.encode()).hexdigest()
    _logger.info(
        "gpu_collective_overlap_pass: PRE-PASS INPUT HASH rank=%d host=%s pid=%d module=%s sha256=%s bytes=%d text_bytes=%d",
        _in_rank,
        _in_host,
        _in_pid,
        module.name,
        _content_digest,
        len(serialized_hlo),
        len(_module_text),
    )
    sys.stderr.write(
        f"[gpu_collective_overlap_pass] === BEGIN PRE-PASS INPUT MODULE "
        f"(rank={_in_rank} host={_in_host} pid={_in_pid} "
        f"module={module.name} sha256={_content_digest}) ===\n"
        f"{_module_text}\n"
        f"[gpu_collective_overlap_pass] === END PRE-PASS INPUT MODULE "
        f"(rank={_in_rank} module={module.name}) ===\n"
    )

  is_root = client is None or _in_rank == 0
  multi_process = client is not None and _jax_process_count() > 1

  _entry_name = _exit_name = None
  if multi_process:
    _barrier_invocation_count += 1
    _entry_name = f"gpu_collective_overlap_pass_entry_{_barrier_invocation_count}"
    _exit_name = f"gpu_collective_overlap_pass_exit_{_barrier_invocation_count}"
    try:
      client.wait_at_barrier(_entry_name, _SHARE_TIMEOUT_MS)
    except Exception as exc:  # pylint: disable=broad-except
      _logger.warning(
          "gpu_collective_overlap_pass: entry barrier %s failed (%s); proceeding without it.",
          _entry_name,
          exc,
      )

  result_bytes: Optional[bytes] = None
  if is_root:
    _t_compute = time.monotonic()
    result_bytes = _compute_collective_overlap(serialized_hlo)
    _logger.info(
        "gpu_collective_overlap_pass: _compute_collective_overlap done (%s, %.1f ms).",
        "no-op" if result_bytes is None else f"{len(result_bytes)} bytes",
        (time.monotonic() - _t_compute) * 1000,
    )

  if multi_process:
    _key = f"gpu_collective_overlap_pass_kv_{_barrier_invocation_count}"
    try:
      if is_root:
        _payload = _SHARE_NONE if result_bytes is None else _SHARE_SOME + result_bytes
        client.key_value_set_bytes(_key, _payload)
      else:
        _payload = client.blocking_key_value_get_bytes(_key, _SHARE_TIMEOUT_MS)
        result_bytes = None if _payload == _SHARE_NONE else _payload[len(_SHARE_SOME) :]
    except Exception as exc:  # pylint: disable=broad-except
      _logger.warning(
          "gpu_collective_overlap_pass: sharing the scheduled module via the JAX distributed client "
          "failed (%s); using this rank's own result, which risks divergent schedules.",
          exc,
      )
      if not is_root:
        result_bytes = _compute_collective_overlap(serialized_hlo)

    try:
      client.wait_at_barrier(_exit_name, _SHARE_TIMEOUT_MS)
    except Exception as exc:  # pylint: disable=broad-except
      _logger.warning(
          "gpu_collective_overlap_pass: exit barrier %s failed (%s); proceeding without it.",
          _exit_name,
          exc,
      )

  # What this rank returns.
  if result_bytes is not None and dump_modules:
    _rank = _jax_process_id()
    _host = socket.gethostname()
    _pid = os.getpid()
    _digest = hashlib.sha256(result_bytes).hexdigest()
    _final_module = _hlo.HloModule.from_serialized_hlo_module_proto(result_bytes)
    _module_name = _final_module.name
    _logger.info(
        "gpu_collective_overlap_pass: POST-SHARE HASH rank=%d host=%s pid=%d module=%s sha256=%s bytes=%d",
        _rank,
        _host,
        _pid,
        _module_name,
        _digest,
        len(result_bytes),
    )
    sys.stderr.write(
        f"[gpu_collective_overlap_pass] === BEGIN POST-SHARE MODULE "
        f"(rank={_rank} host={_host} pid={_pid} module={_module_name} "
        f"sha256={_digest}) ===\n"
        f"{_final_module.to_string()}\n"
        f"[gpu_collective_overlap_pass] === END POST-SHARE MODULE "
        f"(rank={_rank} module={_module_name}) ===\n"
    )

  _logger.info(
      "gpu_collective_overlap_pass: EXIT host=%s pid=%d result=%s (total %.1f ms).",
      socket.gethostname(),
      os.getpid(),
      "no-op" if result_bytes is None else f"{len(result_bytes)} bytes",
      (time.monotonic() - _t_enter) * 1000,
  )
  return result_bytes


# TODO tixxx remove the monkey patch once fdo profile can be passed to
# custom pass interface.
# ---------------------------------------------------------------------------
# PGLE profile interception
# ---------------------------------------------------------------------------
_patched = False


def _patch_pgle_profiler() -> None:
  """Hook PGLE to capture its profile and the raw XSpace behind it."""
  global _patched
  if _patched:
    return
  _patched = True
  import jax._src.profiler as _jax_profiler  # pylint: disable=import-outside-toplevel

  _original_consume = _jax_profiler.PGLEProfiler.consume_fdo_profile

  def _consume_and_capture(self):
    result = _original_consume(self)
    if result:
      _update_profile(result)
    return result

  _jax_profiler.PGLEProfiler.consume_fdo_profile = _consume_and_capture

  # PGLEProfiler.trace() calls _profiler.get_fdo_profile(xspace) on each
  # retry's raw XSpace and then discards it.
  _original_get_fdo_profile = _jax_profiler._profiler.get_fdo_profile  # pylint: disable=protected-access

  def _get_fdo_profile_and_capture(xspace):
    try:
      all_costs = _load_costs_from_xspace_bytes(
          bytes(xspace),
          (_TE_EP_PREFIX,) + _GEMM_COST_PREFIXES + _COLLECTIVE_COST_PREFIXES,
      )
      te_ep_costs = {k: v for k, v in all_costs.items() if k.startswith(_TE_EP_PREFIX)}
      gemm_costs = {k: v for k, v in all_costs.items() if k.startswith(_GEMM_COST_PREFIXES)}
      coll_costs = {k: v for k, v in all_costs.items() if k.startswith(_COLLECTIVE_COST_PREFIXES)}
      _gemm_overrides.update(gemm_costs)
      _collective_overrides.update(coll_costs)
      _te_ep_overrides.update(te_ep_costs)
      _profile_costs.update(te_ep_costs)
      _logger.info(
          "gpu_collective_overlap_pass: captured live per-invocation corrections: %d te_ep, %d GEMM, %d collective.",
          len(te_ep_costs),
          len(gemm_costs),
          len(coll_costs),
      )
    except Exception as exc:  # pylint: disable=broad-except
      _logger.warning(
          "gpu_collective_overlap_pass: failed to extract per-invocation costs from live PGLE data: %s",
          exc,
      )
    return _original_get_fdo_profile(xspace)

  _jax_profiler._profiler.get_fdo_profile = _get_fdo_profile_and_capture  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_registered = False


def register() -> None:
  """Register the collective-overlap POST_SCHEDULER pass and PGLE hook.

  Idempotent: elastic retries re-run initialize(), and each extra registration
  would queue another copy of the pass.
  """
  global _registered
  if _registered:
    return
  if os.environ.get("COLLECTIVE_OVERLAP_DISABLE", "1") == "1":
    _logger.info("gpu_collective_overlap_pass: disabled (set COLLECTIVE_OVERLAP_DISABLE=0 to enable)")
    return
  if not _XLA_SRC:
    _logger.warning(
        "gpu_collective_overlap_pass: not registered; the XLA source path is needed to compile "
        "the profile protos. Set COLLECTIVE_OVERLAP_XLA_PATH to an XLA checkout."
    )
    return
  import jax.extend.xla as jex_xla  # pylint: disable=import-outside-toplevel

  _patch_pgle_profiler()
  jex_xla.register_hlo_module_transformation(
      _gpu_collective_overlap_pass,
      name="gpu_collective_overlap_pass",
      stage=jex_xla.PipelineStage.POST_SCHEDULER,
  )
  _registered = True
  _logger.info("gpu_collective_overlap_pass: registered POST_SCHEDULER pass 'gpu_collective_overlap_pass'.")
