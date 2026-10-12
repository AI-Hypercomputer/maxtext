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

"""Custom XLA post-scheduler transformation for DSv3.

Maximizes async overlap by "bubbling" every async start as early as possible and
every async done as late as possible within a computation's schedule.

The module is split into two layers:

  * A pure core that operates on `ScheduleGraph`, a plain-data description of
  one
    computation's schedule. Every algorithmic helper is a pure function of a
    `ScheduleGraph`, which makes them unit-testable without building HLO.
  * A thin HLO adapter that builds a `ScheduleGraph` from an `HloComputation`,
    runs the core, validates the result, and writes it back to the schedule.

The algorithm, in order:

  1. Bundle each async start with the transitive closure of its data-formatting
     producers, and each async done with the transitive closure of its
     data-formatting consumers. Bundles may overlap, and they stay that way: an
     op claimed by several async ops travels with every one of them.
  2. Probe each bundle against the *original* schedule to find the earliest slot
     a start bundle could occupy and the latest slot a done bundle could occupy.
  3. Derive the final done order (farthest-from-the-end first) and the start
     bubbling order (which is the reverse of the final start order).
  4. Raise each start to its FIFO floor, so that starts sharing a hardware queue
     issue in the same order their dones complete.
  5. Defer every glued op to the back of its bubbling order. A glue edge says
     "schedule me immediately before that op", which only works once the op it
     points at has settled.
  6. Bubble the start bundles left one at a time, then the done bundles right
  one
     at a time, mutating the sequence as we go. Each bundle is re-probed and
     re-pruned against the live schedule, so a member another async op already
     moved is simply picked up from wherever it now sits. A glued bundle ignores
     its own probe and tucks against its anchor instead, unless that position is
     out of its legal reach.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import json
import re
from typing import Any

from jax._src.lib import hlo as _hlo
import jax.extend.xla as jex_xla
from tensorflow.compiler.xla.service import hlo_pb2  # pylint: disable=g-direct-tensorflow-import

_DEFAULT_TRANSFORM_NAME = "max_async_overlap_transform"

# Async start opcode -> matching async done opcode. copy-start/copy-done are
# deliberately absent: they are treated as data-formatting ops instead, so that
# a copy-start -> copy-done -> async-start chain travels with the async start.
_ASYNC_PAIRS: Mapping[str, str] = {
    "kAsyncStart": "kAsyncDone",
    "kAllGatherStart": "kAllGatherDone",
    "kAllReduceStart": "kAllReduceDone",
    "kCollectivePermuteStart": "kCollectivePermuteDone",
    "kSend": "kSendDone",
    "kRecv": "kRecvDone",
}
_ASYNC_START_OPCODES = frozenset(_ASYNC_PAIRS)
_ASYNC_DONE_OPCODES = frozenset(_ASYNC_PAIRS.values())
_DONE_TO_START_OPCODE: Mapping[str, str] = {done: start for start, done in _ASYNC_PAIRS.items()}

# Cheap "plumbing" ops that are pulled along with the async op they feed (or are
# fed by). A fusion counts too, if its fused computation contains nothing else.
_DATA_FORMATTING_OPCODES = frozenset(
    {
        "kAdd",
        # A token carries no data. Every send/recv hangs off one, so leaving it
        # pinned would nail the transfer to wherever the token happens to sit.
        "kAfterAll",
        "kBitcast",
        "kBitcastConvert",
        "kBroadcast",
        # Static fp8 quantization of a collective's payload: multiply by the scale,
        # clamp to the fp8 range, convert. An elementwise dtype change, like the
        # convert alone.
        "kClamp",
        "kConcatenate",
        "kConstant",
        "kConvert",
        "kCopy",
        "kCopyDone",
        "kCopyStart",
        "kDynamicSlice",
        "kDynamicUpdateSlice",
        "kGetTupleElement",
        "kIota",
        # A barrier is pure ordering: it computes nothing and its operands are its
        # only real constraint. Pinning it makes a wall out of whatever it happens
        # to be tupled with -- `program_order` tuples a phase token with values read
        # straight from the loop carry, which are available immediately. Carrying
        # the barrier along with the async op it gates keeps every genuine
        # dependency (they are still operands) while dropping the false ones.
        "kOptimizationBarrier",
        "kPad",
        "kReshape",
        "kSlice",
        "kSubtract",
        "kTranspose",
        "kTuple",
        # Offset arithmetic for a collective's send/recv descriptors: partition-id
        # bit twiddling, capacity clamping, and prefix sums over the expert-count
        # histogram. All operate on a handful of s32 elements.
        "kAnd",
        "kCompare",
        "kMultiply",
        "kPartitionId",
        "kReduceWindow",
        "kSelect",
        "kShiftRightLogical",
    }
)

# Custom calls that are pure plumbing. `AllocateBuffer` reserves an output
# buffer and reads nothing, so it can travel with the op that consumes it.
# `ZeroCrop` is the pad/crop the megascale cross-slice rewrite inserts on a DCN
# transfer's payload; it reshapes storage in place and carries no compute.
_DATA_FORMATTING_CUSTOM_CALLS = frozenset({"AllocateBuffer", "ZeroCrop"})

# Opcodes allowed inside a fused computation for the fusion to still count as
# data formatting. kReduce is here rather than in _DATA_FORMATTING_OPCODES
# because the only reduces that matter are the ones fused with a dynamic-slice
# to compute a collective's offsets; a standalone reduce is real compute.
_FUSION_INTERNAL_EXTRA_OPCODES = frozenset({"kParameter", "kReduce"})

_CONTROL_PREDECESSORS_RE = re.compile(r"control-predecessors=\{([^}]*)\}")
_CALLS_RE = re.compile(r"\bcalls=(%?[\w.\-]+)")
_CUSTOM_CALL_TARGET_RE = re.compile(r'custom_call_target="([^"]*)"')
_BODY_RE = re.compile(r"\bbody=(%?[\w.\-]+)")
_INSTRUCTION_NAME_RE = re.compile(r"%([\w.\-]+)")
_ASYNC_THREAD_RE = re.compile(r'async_execution_thread="([^"]+)"')
_CORE_IDS_RE = re.compile(r'"core_ids":\s*\[([^\]]*)\]')

# Frontend attribute handler that marks a send/recv as megascale (DCN) traffic
# rather than a transfer to the host. Megascale lowers a cross-slice collective
# into send/recv pairs carrying is_host_transfer=true, so the handler name --
# not the host transfer bit -- tells them apart from real host offload.
_MEGASCALE_HANDLER = "xla_megascale_runtime"

# DCN transfers print neither an execution thread nor core ids, but sends share
# one FIFO with each other and recvs share another, so they get synthetic ids.
_DCN_SEND_QUEUE = "dcn:send"
_DCN_RECV_QUEUE = "dcn:recv"

# Host offload (memory space S(5)) prefetches and stores. They print neither an
# execution thread nor core ids either, but they share the host-transfer DMA
# resources with each other, one per direction.
_HOST_H2D_QUEUE = "host:h2d"
_HOST_D2H_QUEUE = "host:d2h"
_HOST_MEMORY_SPACE = "S(5)"

# Queues whose starts may not be hoisted above a done that preceded them in the
# original schedule. XLA's scheduler bounds how many host offloads are in flight
# and places each prefetch only once an earlier one has retired; hoisting past
# that retirement piles every transfer onto the shared host-command stream.
_SERIALIZED_QUEUES = frozenset({_HOST_H2D_QUEUE, _HOST_D2H_QUEUE})

# AggregateSendRecv stamps each megascale send/recv with one of these. The
# backend sweeps every DEFER into the next ISSUE in schedule order and fires a
# single host interrupt for the lot.
_AGGREGATE_DEFER = "AGGREGATE_STATUS_DEFER"
_AGGREGATE_ISSUE = "AGGREGATE_STATUS_ISSUE"
_AGGREGATED_SEND_RECV_CONFIG = "aggregated_send_recv_config"

# MaxSendRecvAggregation() for TPU v4 and later, bounding the sync flags one
# aggregation group may use.
_MAX_SEND_RECV_AGGREGATION = 40


class ScheduleTransformError(Exception):
  """Raised when the transform cannot produce a valid schedule."""


# ---------------------------------------------------------------------------
# Pure core: data model
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ScheduleGraph:
  """One computation's schedule as plain data.

  Attributes:
    order: Instruction names in schedule order.
    opcodes: Instruction name -> opcode name (e.g. "kAsyncStart").
    operands: Instruction name -> its operands, in operand order.
    users: Inverse of `operands`, in schedule order.
    control_preds: Instruction name -> its control predecessors.
    control_succs: Inverse of `control_preds`.
    data_formatting: Names of instructions eligible for bundling.
    pinned: Names that must never move or be bundled (parameters and the root).
    param_prefix_len: Length of the leading run of parameter instructions.
    root: Name of the root instruction, which must stay last.
    queues: Async start name -> the hardware FIFO it is issued on. Starts absent
      from this map are not queue-ordered against anything.
    glue: Async op name -> the async op it must sit immediately before. Used to
      keep a DCN recv against the send it shares a token with, and that send's
      done against the recv's done. Both members of a glue edge must bubble in
      the same phase, and the anchor must not itself be glued.
  """

  order: tuple[str, ...]
  opcodes: Mapping[str, str]
  operands: Mapping[str, tuple[str, ...]]
  users: Mapping[str, tuple[str, ...]]
  control_preds: Mapping[str, tuple[str, ...]]
  control_succs: Mapping[str, tuple[str, ...]]
  data_formatting: frozenset[str]
  pinned: frozenset[str]
  param_prefix_len: int
  root: str
  queues: Mapping[str, str]
  glue: Mapping[str, str]

  def positions(self) -> dict[str, int]:
    """Returns the name -> index map for the original order."""
    return {name: i for i, name in enumerate(self.order)}


def make_schedule_graph(
    order: Sequence[str],
    opcodes: Mapping[str, str],
    operands: Mapping[str, Sequence[str]],
    control_preds: Mapping[str, Sequence[str]] | None = None,
    data_formatting: Iterable[str] = (),
    queues: Mapping[str, str] | None = None,
    glue: Mapping[str, str] | None = None,
    serialized_queues: Iterable[str] = (),
) -> ScheduleGraph:
  """Builds a `ScheduleGraph`, deriving users, control successors, and pins.

  Args:
    order: Instruction names in schedule order.
    opcodes: Instruction name -> opcode name.
    operands: Instruction name -> operands. Names outside `order` are dropped.
    control_preds: Instruction name -> control predecessors.
    data_formatting: Names eligible for bundling.
    queues: Async start name -> the hardware FIFO it is issued on.
    glue: Async op name -> the async op it must sit immediately before. Edges
      with either end outside `order` are dropped.
    serialized_queues: Queues whose starts must stay behind every same-queue
      done that precedes them in `order`. Each such done becomes a control
      predecessor of the start, so every later step honors it for free.

  Returns:
    A fully populated `ScheduleGraph`.

  Raises:
    ScheduleTransformError: If `order` is empty or contains duplicates.
  """
  order = tuple(order)
  if not order:
    raise ScheduleTransformError("Cannot build a graph from an empty schedule.")
  present = set(order)
  if len(present) != len(order):
    raise ScheduleTransformError("Schedule contains duplicate instructions.")

  control_preds = control_preds or {}
  queues = {k: v for k, v in (queues or {}).items() if k in present}
  serialized_queues = frozenset(serialized_queues)
  operand_map: dict[str, tuple[str, ...]] = {}
  control_pred_map: dict[str, tuple[str, ...]] = {}
  user_lists: dict[str, list[str]] = {name: [] for name in order}
  control_succ_lists: dict[str, list[str]] = {name: [] for name in order}

  retired: dict[str, list[str]] = {}
  for name in order:
    operand_map[name] = tuple(o for o in operands.get(name, ()) if o in present)
    preds = [p for p in control_preds.get(name, ()) if p in present]
    queue = queues.get(name)
    if queue is not None and queue in serialized_queues:
      preds += [d for d in retired.get(queue, ()) if d not in preds]
    control_pred_map[name] = tuple(preds)
    if opcodes.get(name) in _ASYNC_DONE_OPCODES and operand_map[name]:
      done_queue = queues.get(operand_map[name][0])
      if done_queue is not None and done_queue in serialized_queues:
        retired.setdefault(done_queue, []).append(name)
  for name in order:
    for operand in operand_map[name]:
      user_lists[operand].append(name)
    for pred in control_pred_map[name]:
      control_succ_lists[pred].append(name)

  root = order[-1]
  param_prefix_len = 0
  for name in order:
    if opcodes.get(name) != "kParameter":
      break
    param_prefix_len += 1
  pinned = frozenset([name for name in order if opcodes.get(name) == "kParameter"] + [root])

  return ScheduleGraph(
      order=order,
      opcodes=dict(opcodes),
      operands=operand_map,
      users={name: tuple(user_lists[name]) for name in order},
      control_preds=control_pred_map,
      control_succs={name: tuple(control_succ_lists[name]) for name in order},
      data_formatting=frozenset(n for n in data_formatting if n in present),
      pinned=pinned,
      param_prefix_len=param_prefix_len,
      root=root,
      queues=queues,
      glue={k: v for k, v in (glue or {}).items() if k in present and v in present},
  )


@dataclasses.dataclass(frozen=True)
class AsyncPair:
  """An async start paired with its async done."""

  start: str
  done: str


@dataclasses.dataclass(frozen=True)
class Probe:
  """Where a bundle could move to, and what stopped it."""

  slot: int
  blocker: str | None


@dataclasses.dataclass(frozen=True)
class RescheduleResult:
  """Everything the core computed, for both application and debugging."""

  order: tuple[str, ...]
  pairs: tuple[AsyncPair, ...]
  start_bundles: Mapping[str, frozenset[str]]
  done_bundles: Mapping[str, frozenset[str]]
  start_probes: Mapping[str, Probe]
  done_probes: Mapping[str, Probe]
  start_bubble_order: tuple[str, ...]
  done_bubble_order: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class FifoViolation:
  """A pair of async ops whose starts and dones complete out of order."""

  group: str
  earlier_start: str
  later_start: str
  earlier_done: str
  later_done: str


# ---------------------------------------------------------------------------
# Pure core: classification and bundling
# ---------------------------------------------------------------------------


def find_async_pairs(graph: ScheduleGraph) -> list[AsyncPair]:
  """Finds async start/done pairs, matching a done to `done.operands[0]`.

  Args:
    graph: The schedule graph.

  Returns:
    Pairs in schedule order of the done instruction.

  Raises:
    ScheduleTransformError: If an async done cannot be matched to a start, or if
      an unsupported kAsyncUpdate is present.
  """
  for name in graph.order:
    if graph.opcodes[name] == "kAsyncUpdate":
      raise ScheduleTransformError(f"kAsyncUpdate is not supported (instruction {name}).")

  pairs: list[AsyncPair] = []
  for name in graph.order:
    opcode = graph.opcodes[name]
    if opcode not in _ASYNC_DONE_OPCODES:
      continue
    operands = graph.operands[name]
    expected = _DONE_TO_START_OPCODE[opcode]
    if not operands or graph.opcodes.get(operands[0]) != expected:
      raise ScheduleTransformError(
          f"Async done {name} ({opcode}) has no {expected} as operand 0;" f" operands={operands}."
      )
    pairs.append(AsyncPair(start=operands[0], done=name))
  return pairs


def bundle_for_start(graph: ScheduleGraph, start: str) -> frozenset[str]:
  """Returns `start` plus its transitive data-formatting producers."""
  return _closure(graph, start, graph.operands)


def bundle_for_done(graph: ScheduleGraph, done: str) -> frozenset[str]:
  """Returns `done` plus its transitive data-formatting consumers."""
  return _closure(graph, done, graph.users)


def _closure(
    graph: ScheduleGraph,
    seed: str,
    edges: Mapping[str, tuple[str, ...]],
) -> frozenset[str]:
  """Grows a bundle from `seed` along `edges` through data-formatting ops."""
  bundle = {seed}
  stack = [seed]
  while stack:
    current = stack.pop()
    for neighbor in edges[current]:
      if neighbor in bundle:
        continue
      if neighbor in graph.pinned:
        continue
      if neighbor not in graph.data_formatting:
        continue
      bundle.add(neighbor)
      stack.append(neighbor)
  return frozenset(bundle)


# ---------------------------------------------------------------------------
# Pure core: probing
# ---------------------------------------------------------------------------


def earliest_slot(
    graph: ScheduleGraph,
    bundle: frozenset[str],
    positions: Mapping[str, int],
) -> Probe:
  """Returns the lowest index `bundle` can start at, and what blocked it.

  A bundle may move up to just after the latest-scheduled instruction it depends
  on (by data or control edge) that is not itself a bundle member, and never
  above the leading run of parameters.

  Args:
    graph: The schedule graph.
    bundle: The instructions moving together.
    positions: Current name -> index map (may differ from the original order).

  Returns:
    The target slot and the blocking instruction, if any.
  """
  slot = graph.param_prefix_len
  blocker: str | None = None
  for member in bundle:
    for dep in (*graph.operands[member], *graph.control_preds[member]):
      if dep in bundle:
        continue
      if positions[dep] + 1 > slot:
        slot = positions[dep] + 1
        blocker = dep
  return Probe(slot=slot, blocker=blocker)


def latest_slot(
    graph: ScheduleGraph,
    bundle: frozenset[str],
    positions: Mapping[str, int],
) -> Probe:
  """Returns the highest index `bundle` can end at, and what blocked it.

  Mirror image of `earliest_slot`: a bundle may move down to just before the
  earliest-scheduled instruction that depends on it, and never past the root.

  Args:
    graph: The schedule graph.
    bundle: The instructions moving together.
    positions: Current name -> index map.

  Returns:
    The target slot for the *last* bundle member, and the blocking instruction.
  """
  slot = positions[graph.root] - 1
  blocker: str | None = graph.root
  for member in bundle:
    for dep in (*graph.users[member], *graph.control_succs[member]):
      if dep in bundle:
        continue
      if positions[dep] - 1 < slot:
        slot = positions[dep] - 1
        blocker = dep
  return Probe(slot=slot, blocker=blocker)


def prune_bundle_for_start(
    graph: ScheduleGraph,
    seed: str,
    bundle: frozenset[str],
    positions: Mapping[str, int],
    min_slot: int = 0,
) -> tuple[frozenset[str], Probe]:
  """Shrinks a start bundle until no member would have to move later.

  Bundles are not contiguous: members can be scattered through the schedule.
  `earliest_slot` is the maximum over *all* members' dependencies, so a member
  sitting before that bound would be dragged forward when the bundle is
  compacted, possibly past its own users. Such members are dropped (they simply
  stay put and act as ordinary producers), which can only lower the bound, so
  the loop terminates.

  `min_slot` is the FIFO floor the bundle will actually be placed at, which can
  exceed the dependency bound. Members are pruned against the larger of the two,
  because that is where they will really land. The seed is exempt: raising the
  seed to the floor is the whole point of the floor.

  Args:
    graph: The schedule graph.
    seed: The async start.
    bundle: Candidate bundle.
    positions: Current name -> index map.
    min_slot: Lower bound on the slot the bundle will be moved to.

  Returns:
    The pruned bundle and its probe. The probe reports the dependency bound
    only; the caller still applies `min_slot`.

  Raises:
    ScheduleTransformError: If the seed precedes its own dependency bound.
  """
  current = set(bundle)
  while True:
    probe = earliest_slot(graph, frozenset(current), positions)
    target = max(probe.slot, min_slot)
    dropped = {m for m in current if positions[m] < target}
    if seed in dropped:
      if positions[seed] < probe.slot:
        raise ScheduleTransformError(
            f"Async start {seed} at {positions[seed]} precedes its own bundle" f" lower bound {probe.slot}."
        )
      dropped.discard(seed)
    if not dropped:
      return frozenset(current), probe
    current -= dropped


def prune_bundle_for_done(
    graph: ScheduleGraph,
    seed: str,
    bundle: frozenset[str],
    positions: Mapping[str, int],
) -> tuple[frozenset[str], Probe]:
  """Shrinks a done bundle until no member would have to move earlier.

  Mirror image of `prune_bundle_for_start`.

  Args:
    graph: The schedule graph.
    seed: The async done.
    bundle: Candidate bundle.
    positions: Current name -> index map.

  Returns:
    The pruned bundle and its probe.

  Raises:
    ScheduleTransformError: If the seed itself would be dropped.
  """
  current = set(bundle)
  while True:
    probe = latest_slot(graph, frozenset(current), positions)
    dropped = {m for m in current if positions[m] > probe.slot}
    if not dropped:
      return frozenset(current), probe
    if seed in dropped:
      raise ScheduleTransformError(
          f"Async done {seed} at {positions[seed]} follows its own bundle" f" upper bound {probe.slot}."
      )
    current -= dropped


# ---------------------------------------------------------------------------
# Pure core: ordering
# ---------------------------------------------------------------------------


def order_dones(
    dones: Sequence[str],
    done_probes: Mapping[str, Probe],
    start_probes: Mapping[str, Probe],
    start_of_done: Mapping[str, str],
    total: int,
    glue: Mapping[str, str] | None = None,
) -> list[str]:
  """Orders async dones as they should appear in the final schedule.

  Farthest from the end first; ties broken by the earliest position of the
  corresponding start, then by name.

  A glued cluster is tied to the *latest* start in it, not to each done's own
  start. A DCN recv start depends on nothing and so probes to the very front,
  which would rank the recv dones by name; the FIFO constraint would then drag
  every send up to match that arbitrary order. Ranking the cluster by its send
  instead lets the transfers complete in the order they can be issued, which
  costs no overlap because the dones of a cluster share a ceiling anyway.

  Args:
    dones: Async done names.
    done_probes: Probe results for the done bundles.
    start_probes: Probe results for the start bundles.
    start_of_done: Done name -> its start name.
    total: Number of instructions in the schedule.
    glue: Done name -> the done it must sit immediately before.

  Returns:
    Done names in final schedule order.
  """
  followers: dict[str, list[str]] = {}
  for done in dones:
    anchor = (glue or {}).get(done)
    if anchor is not None:
      followers.setdefault(anchor, []).append(done)

  def cluster_start_slot(done: str) -> int:
    slots = [-1]
    for member in (done, *followers.get(done, ())):
      start = start_of_done.get(member)
      if start in start_probes:
        slots.append(start_probes[start].slot)
    return max(slots)

  def key(done: str) -> tuple[int, int, str]:
    distance_from_end = total - 1 - done_probes[done].slot
    return (-distance_from_end, cluster_start_slot(done), done)

  return sorted(dones, key=key)


def fifo_constrained_slots(
    start_probes: Mapping[str, Probe],
    done_order: Sequence[str],
    start_of_done: Mapping[str, str],
    queues: Mapping[str, str],
) -> dict[str, int]:
  """Raises start slots so each hardware queue issues in completion order.

  An async queue is FIFO: whatever is issued first completes first. The done
  order is fixed by how late each done can sink, so on a shared queue the issue
  order is fully determined by it — a start whose done must complete later
  cannot be issued before one whose done must complete sooner.

  Where the raw probes disagree, the *start* yields. Delaying a start is always
  legal (a start may sit anywhere at or after its earliest slot), whereas a done
  cannot be pushed past its first consumer. The cost is some overlap on the
  transfer that could not have completed first anyway.

  A start is only delayed when its slot is *strictly* before its queue
  predecessor's. On a tie no delay is needed: `order_starts` already breaks ties
  by latest done first, which bubbles the later-completing start first and so
  lands it after its predecessor. Bumping on a tie would instead push the start
  past every unrelated op that shares the slot.

  Args:
    start_probes: Probe results for the start bundles.
    done_order: Final order of async dones.
    start_of_done: Done name -> its start name.
    queues: Async start name -> its queue. Starts absent here are unconstrained.

  Returns:
    Start name -> its FIFO-respecting earliest slot.
  """
  slots = {start: probe.slot for start, probe in start_probes.items()}
  last_on_queue: dict[str, int] = {}
  for done in done_order:
    start = start_of_done.get(done)
    if start is None or start not in slots:
      continue
    queue = queues.get(start)
    if queue is None:
      continue
    floor = last_on_queue.get(queue)
    if floor is not None and slots[start] < floor:
      slots[start] = floor
    last_on_queue[queue] = slots[start]
  return slots


def order_starts(
    starts: Sequence[str],
    start_slots: Mapping[str, int],
    done_order: Sequence[str],
    done_of_start: Mapping[str, str],
) -> list[str]:
  """Orders async starts for the bubbling phase.

  This is the *reverse* of the desired final order, because bubbling each start
  as far left as possible pushes previously-placed starts to the right.

  Sorted by earliest position descending; ties broken by latest corresponding
  done first, then by name descending (so the final schedule is name-ascending).

  Args:
    starts: Async start names.
    start_slots: Start name -> its earliest slot, after FIFO constraints.
    done_order: Final order of async dones.
    done_of_start: Start name -> its done name.

  Returns:
    Start names in bubbling order.
  """
  done_rank = {done: i for i, done in enumerate(done_order)}

  def key(start: str) -> tuple[int, int]:
    done = done_of_start.get(start)
    rank = done_rank.get(done, -1) if done is not None else -1
    return (-start_slots[start], -rank)

  # Stable sort over a name-descending list yields name-ascending final order.
  return sorted(sorted(starts, reverse=True), key=key)


def defer_glued(names: Sequence[str], glue: Mapping[str, str]) -> list[str]:
  """Moves glued followers to the back of a bubbling order.

  A glued op is placed relative to its anchor rather than to a slot, so the
  anchor has to be sitting at its final position first. Anchors are never
  themselves glued, so one partition is enough.

  Args:
    names: Async ops in their natural bubbling order.
    glue: Follower -> anchor. Names absent from this map are not glued.

  Returns:
    The same names, unglued ones first, each group keeping its relative order.
  """
  return [n for n in names if n not in glue] + [n for n in names if n in glue]


def apply_glue_order(names: Sequence[str], glue: Mapping[str, str]) -> list[str]:
  """Reorders `names` so each glued follower sits directly before its anchor.

  This is where the bubbler will actually leave them, which is not where their
  own probes ranked them. Anything reasoning about the final order -- the FIFO
  analysis above all, since a queue completes in schedule order -- has to use
  this rather than the raw probe ranking.

  Args:
    names: Async ops in probe order.
    glue: Follower -> anchor. Names absent from this map are not glued.

  Returns:
    The same names with every follower moved to just before its anchor.
  """
  followers: dict[str, list[str]] = {}
  for name in names:
    anchor = glue.get(name)
    if anchor is not None:
      followers.setdefault(anchor, []).append(name)
  ordered: list[str] = []
  for anchor in names:
    if anchor in glue:
      continue
    ordered.extend(followers.get(anchor, ()))
    ordered.append(anchor)
  return ordered


# ---------------------------------------------------------------------------
# Pure core: bubbling
# ---------------------------------------------------------------------------


def move_bundle_to_start_slot(order: Sequence[str], bundle: frozenset[str], slot: int) -> list[str]:
  """Moves `bundle` so its first member sits at index `slot`.

  Bundle members keep their relative order. Because every instruction before
  `slot` is a non-member in a valid schedule, `slot` indexes the same position
  before and after the members are lifted out.

  Args:
    order: Current schedule.
    bundle: Instructions to move together.
    slot: Target index of the first bundle member.

  Returns:
    The new schedule.
  """
  members = [name for name in order if name in bundle]
  rest = [name for name in order if name not in bundle]
  return rest[:slot] + members + rest[slot:]


def move_bundle_to_end_slot(order: Sequence[str], bundle: frozenset[str], slot: int) -> list[str]:
  """Moves `bundle` so its last member sits at index `slot`.

  Args:
    order: Current schedule.
    bundle: Instructions to move together.
    slot: Target index of the last bundle member.

  Returns:
    The new schedule.
  """
  members = [name for name in order if name in bundle]
  rest = [name for name in order if name not in bundle]
  tail = len(order) - 1 - slot
  split = len(rest) - tail
  return rest[:split] + members + rest[split:]


def move_bundle_before(order: Sequence[str], bundle: frozenset[str], anchor: str) -> list[str]:
  """Moves `bundle` so its last member sits immediately before `anchor`.

  The anchor's index is resolved *after* the members are lifted out, so this
  lands correctly whether the bundle currently sits before or after the anchor.
  That is the difference from `move_bundle_to_start_slot`, whose slot is an
  index into the original order.

  Args:
    order: Current schedule.
    bundle: Instructions to move together. Must not contain `anchor`.
    anchor: The instruction to park the bundle against.

  Returns:
    The new schedule.

  Raises:
    ScheduleTransformError: If `anchor` is a bundle member.
  """
  if anchor in bundle:
    raise ScheduleTransformError(f"Cannot park a bundle against {anchor}, which is one of its members.")
  members = [name for name in order if name in bundle]
  rest = [name for name in order if name not in bundle]
  index = rest.index(anchor)
  return rest[:index] + members + rest[index:]


def bubble_starts_left(
    graph: ScheduleGraph,
    order: Sequence[str],
    bubble_order: Sequence[str],
    bundles: Mapping[str, frozenset[str]],
    floors: Mapping[str, int] | None = None,
) -> tuple[list[str], dict[str, frozenset[str]]]:
  """Bubbles each start bundle as far left as legal, one at a time.

  Bundles are re-pruned against live positions, since earlier moves change them.
  `bundles` may overlap, and no attempt is made to make them disjoint: a
  data-formatting op reachable from several starts travels with all of them.
  Each mover can only pull it further left, and every start bubbled afterwards
  lands further left still, so the op ends up as early as any of its consumers
  can use it. Re-pruning keeps each move legal.

  A glued start ignores its floor and parks immediately before its anchor
  instead. If its dependencies will not reach that far the glue is dropped for
  that start and it bubbles normally, which is why `glue` is best-effort.

  Args:
    graph: The schedule graph.
    order: Starting schedule.
    bubble_order: Starts to move, in order. Glued starts must come after their
      anchors, which `defer_glued` arranges.
    bundles: Start -> its (possibly overlapping) bundle.
    floors: Start -> a lower bound on its slot, from the FIFO constraint.

  Returns:
    The rewritten schedule, and each start's actually-moved bundle.
  """
  floors = floors or {}
  current = list(order)
  taken: dict[str, frozenset[str]] = {}
  for start in bubble_order:
    positions = {name: i for i, name in enumerate(current)}
    anchor = graph.glue.get(start)
    floor = positions[anchor] if anchor is not None else floors.get(start, 0)
    bundle, probe = prune_bundle_for_start(graph, start, bundles[start], positions, floor)
    if anchor is not None and probe.slot <= positions[anchor]:
      current = move_bundle_before(current, bundle, anchor)
    else:
      current = move_bundle_to_start_slot(current, bundle, max(probe.slot, floor))
    taken[start] = bundle
  return current, taken


def bubble_dones_right(
    graph: ScheduleGraph,
    order: Sequence[str],
    bubble_order: Sequence[str],
    bundles: Mapping[str, frozenset[str]],
) -> tuple[list[str], dict[str, frozenset[str]]]:
  """Bubbles each done bundle as far right as legal, one at a time.

  Mirror image of `bubble_starts_left`, including the overlapping bundles.

  A glued done parks immediately before its anchor rather than at its own
  latest slot. Because the anchor has already sunk as far right as it can, the
  follower is pulled towards it rather than the other way around, which costs
  the follower some overlap but none of the anchor's.

  Args:
    graph: The schedule graph.
    order: Starting schedule.
    bubble_order: Dones to move, in order. Glued dones must come after their
      anchors, which `defer_glued` arranges.
    bundles: Done -> its (possibly overlapping) bundle.

  Returns:
    The rewritten schedule, and each done's actually-moved bundle.
  """
  current = list(order)
  taken: dict[str, frozenset[str]] = {}
  for done in bubble_order:
    positions = {name: i for i, name in enumerate(current)}
    bundle, probe = prune_bundle_for_done(graph, done, bundles[done], positions)
    anchor = graph.glue.get(done)
    if anchor is not None and positions[anchor] - 1 <= probe.slot:
      current = move_bundle_before(current, bundle, anchor)
    else:
      current = move_bundle_to_end_slot(current, bundle, probe.slot)
    taken[done] = bundle
  return current, taken


def reschedule(graph: ScheduleGraph) -> RescheduleResult:
  """Runs the full bubbling algorithm over `graph`.

  Args:
    graph: The schedule graph for one computation.

  Returns:
    The new order plus all intermediate state, for debugging.
  """
  pairs = find_async_pairs(graph)
  done_of_start = {p.start: p.done for p in pairs}
  start_of_done = {p.done: p.start for p in pairs}
  starts = [p.start for p in pairs]
  dones = [p.done for p in pairs]

  positions = graph.positions()

  # Probe against the original schedule. Pruning happens here too, so that the
  # recorded position is one the bundle can actually reach.
  start_bundles: dict[str, frozenset[str]] = {}
  start_probes: dict[str, Probe] = {}
  for start in starts:
    bundle, probe = prune_bundle_for_start(graph, start, bundle_for_start(graph, start), positions)
    start_bundles[start] = bundle
    start_probes[start] = probe

  done_bundles: dict[str, frozenset[str]] = {}
  done_probes: dict[str, Probe] = {}
  for done in dones:
    bundle, probe = prune_bundle_for_done(graph, done, bundle_for_done(graph, done), positions)
    done_bundles[done] = bundle
    done_probes[done] = probe

  done_order = order_dones(
      dones,
      done_probes,
      start_probes,
      start_of_done,
      len(graph.order),
      graph.glue,
  )
  # Gluing relocates the followers, so this -- not `done_order` -- is the
  # completion order each queue will actually see.
  final_done_order = apply_glue_order(done_order, graph.glue)
  start_slots = fifo_constrained_slots(start_probes, final_done_order, start_of_done, graph.queues)
  start_bubble_order = defer_glued(
      order_starts(starts, start_slots, final_done_order, done_of_start),
      graph.glue,
  )
  done_bubble_order = defer_glued(final_done_order, graph.glue)
  order, moved_starts = bubble_starts_left(graph, graph.order, start_bubble_order, start_bundles, start_slots)
  order, moved_dones = bubble_dones_right(graph, order, done_bubble_order, done_bundles)

  return RescheduleResult(
      order=tuple(order),
      pairs=tuple(pairs),
      start_bundles=moved_starts,
      done_bundles=moved_dones,
      start_probes=start_probes,
      done_probes=done_probes,
      start_bubble_order=tuple(start_bubble_order),
      done_bubble_order=tuple(done_bubble_order),
  )


# ---------------------------------------------------------------------------
# Pure core: validation
# ---------------------------------------------------------------------------


def validate_order(graph: ScheduleGraph, order: Sequence[str]) -> None:
  """Checks that `order` is a legal schedule for `graph`.

  Args:
    graph: The schedule graph.
    order: The candidate schedule.

  Raises:
    ScheduleTransformError: On any violation.
  """
  if len(order) != len(graph.order) or set(order) != set(graph.order):
    missing = set(graph.order) - set(order)
    extra = set(order) - set(graph.order)
    raise ScheduleTransformError(
        f"Schedule changed instruction set (missing={sorted(missing)[:5]},"
        f" extra={sorted(extra)[:5]}, len {len(order)} vs {len(graph.order)})."
    )
  positions = {name: i for i, name in enumerate(order)}
  for name in order:
    for operand in graph.operands[name]:
      if positions[operand] >= positions[name]:
        raise ScheduleTransformError(f"{name} is scheduled before its operand {operand}.")
    for pred in graph.control_preds[name]:
      if positions[pred] >= positions[name]:
        raise ScheduleTransformError(f"{name} is scheduled before its control predecessor {pred}.")
  if order[-1] != graph.root:
    raise ScheduleTransformError(f"Root {graph.root} is not last; {order[-1]} is.")
  prefix = tuple(order[: graph.param_prefix_len])
  if prefix != graph.order[: graph.param_prefix_len]:
    raise ScheduleTransformError(
        "Leading parameter run was disturbed:" f" {prefix[:5]} vs {graph.order[:graph.param_prefix_len][:5]}."
    )


def fifo_violations(
    order: Sequence[str],
    pairs: Iterable[AsyncPair],
    group_of: Callable[[AsyncPair], str | None],
) -> list[FifoViolation]:
  """Finds async pairs on one queue that complete out of issue order.

  Hardware async queues are FIFO: if `start(A)` precedes `start(B)` on the same
  queue then `done(A)` must precede `done(B)`. Nested completion forces the
  backend to re-sink starts, undoing the overlap this transform creates.

  Args:
    order: The schedule.
    pairs: Async pairs to check.
    group_of: Maps a pair to its queue id, or None to skip the pair.

  Returns:
    One violation per adjacent out-of-order pair, in group order.
  """
  positions = {name: i for i, name in enumerate(order)}
  grouped: dict[str, list[AsyncPair]] = {}
  for pair in pairs:
    group = group_of(pair)
    if group is None:
      continue
    grouped.setdefault(group, []).append(pair)

  violations: list[FifoViolation] = []
  for group in sorted(grouped):
    members = sorted(grouped[group], key=lambda p: positions[p.start])
    for earlier, later in zip(members, members[1:]):
      if positions[earlier.done] > positions[later.done]:
        violations.append(
            FifoViolation(
                group=group,
                earlier_start=earlier.start,
                later_start=later.start,
                earlier_done=earlier.done,
                later_done=later.done,
            )
        )
  return violations


# ---------------------------------------------------------------------------
# Pure core: send/recv aggregation
# ---------------------------------------------------------------------------


def aggregation_statuses(
    order: Sequence[str],
    operands: Mapping[str, Sequence[str]],
    control_preds: Mapping[str, Sequence[str]],
    transfers: Iterable[str],
    dones: Iterable[str],
    free: Iterable[str],
    channel_ids: Mapping[str, int],
    max_group_size: int = _MAX_SEND_RECV_AGGREGATION,
) -> dict[str, str]:
  """Recomputes AggregateSendRecv's DEFER/ISSUE stamps for a final schedule.

  XLA stamps these before post-scheduler transforms run, against a schedule the
  transform then rewrites, and a second XLA run would skip every op it already
  stamped. This replays the pass's grouping on the final order instead, without
  its reordering: only ops that are already adjacent are grouped. That matches
  the production flags, which skip no small instructions between members.

  A group is a run of transfers, with their dones and operand-free tokens
  allowed in between. Anything else ends it, as does a transfer that reads a
  group member, a control edge from a group done, a repeated channel, or a full
  group. The last transfer of a group ISSUEs and the rest DEFER.

  Args:
    order: Instruction names in schedule order.
    operands: Instruction name -> operands.
    control_preds: Instruction name -> control predecessors.
    transfers: Megascale send/recv ops eligible for aggregation.
    dones: Their send-done/recv-done ops.
    free: Operand-free parameters and tokens, which AggregateSendRecv hoists out
      of the way and so never break a group.
    channel_ids: Instruction name -> channel id, for transfers and dones.
    max_group_size: Most transfers one group may hold.

  Returns:
    Transfer name -> its status. Every transfer in `order` is present.
  """
  transfers = frozenset(transfers)
  dones = frozenset(dones)
  free = frozenset(free)
  statuses: dict[str, str] = {}
  group: set[str] = set()
  members: list[str] = []
  channels: set[int] = set()

  def issue() -> None:
    for i, name in enumerate(members):
      last = i == len(members) - 1
      statuses[name] = _AGGREGATE_ISSUE if last else _AGGREGATE_DEFER
    group.clear()
    members.clear()
    channels.clear()

  for name in order:
    if name in free:
      continue
    if name not in transfers and name not in dones:
      issue()
      continue
    if any(p in group and p in dones for p in control_preds.get(name, ())):
      issue()
    channel = channel_ids.get(name)
    if name in transfers:
      if any(o in group for o in operands.get(name, ())) or len(members) >= max_group_size or channel in channels:
        issue()
      members.append(name)
    group.add(name)
    if channel is not None:
      channels.add(channel)
  issue()
  return statuses


def stranded_defers(order: Sequence[str], statuses: Mapping[str, str]) -> list[str]:
  """Returns the DEFER'd ops with no ISSUE after them.

  The backend flushes a DEFER into the next ISSUE in schedule order. With none
  left, its host interrupt never fires and its done waits forever.

  Args:
    order: Instruction names in schedule order.
    statuses: Instruction name -> its aggregation status.

  Returns:
    The stranded names, in schedule order.
  """
  stranded: list[str] = []
  for name in order:
    status = statuses.get(name)
    if status == _AGGREGATE_ISSUE:
      stranded.clear()
    elif status == _AGGREGATE_DEFER:
      stranded.append(name)
  return stranded


# ---------------------------------------------------------------------------
# HLO adapter
# ---------------------------------------------------------------------------


def _parse_control_predecessors(instruction_text: str) -> tuple[str, ...]:
  """Extracts control predecessor names from an instruction's to_string()."""
  match = _CONTROL_PREDECESSORS_RE.search(instruction_text)
  if match is None:
    return ()
  return tuple(_INSTRUCTION_NAME_RE.findall(match.group(1)))


def _fusion_is_data_formatting(
    instruction_text: str,
    computations_by_name: Mapping[str, "_hlo.HloComputation"],
    cache: dict[str, bool],
) -> bool:
  """Returns True if a fusion's body contains only data-formatting ops.

  Recurses into nested fusions: a fusion whose body is all formatting is still
  formatting however deeply it is wrapped.

  Args:
    instruction_text: The fusion instruction's textual form, which names the
      fused computation it calls.
    computations_by_name: All non-fusion computations, keyed by name.
    cache: Memo of callee name -> verdict, shared across calls and mutated here.
  """
  match = _CALLS_RE.search(instruction_text)
  if match is None:
    return False
  callee = match.group(1).lstrip("%")
  if callee in cache:
    return cache[callee]
  computation = computations_by_name.get(callee)
  if computation is None:
    cache[callee] = False
    return False
  # Fused computations cannot be recursive, but seed the cache anyway so a
  # malformed module cannot spin here.
  cache[callee] = False
  allowed = _DATA_FORMATTING_OPCODES | _FUSION_INTERNAL_EXTRA_OPCODES
  result = True
  for inner in computation.instructions():
    opcode = inner.opcode.name
    if opcode in allowed:
      continue
    if opcode == "kFusion" and _fusion_is_data_formatting(inner.to_string(), computations_by_name, cache):
      continue
    if opcode == "kCustomCall" and _custom_call_is_data_formatting(inner.to_string()):
      continue
    result = False
    break
  cache[callee] = result
  return result


def _custom_call_is_data_formatting(instruction_text: str) -> bool:
  """Returns True for custom calls that only reserve or reshape storage."""
  match = _CUSTOM_CALL_TARGET_RE.search(instruction_text)
  return match is not None and match.group(1) in _DATA_FORMATTING_CUSTOM_CALLS


def async_queue(opcode: str, instruction_text: str) -> str | None:
  """Returns the hardware FIFO an async start is issued on, or None.

  Queued async ops name their execution thread and the core they run on; ops
  sharing both share a FIFO.

  Megascale DCN transfers print neither field, but they are queue ordered:
  every send shares one FIFO and every recv shares another. They are recognised
  by their handler name rather than by the fields, hence the opcode.

  Host offloads print neither field either. A prefetch (a dynamic-slice out of
  host memory) and a store (a dynamic-update-slice into it) each share one
  host-transfer queue per direction.

  Args:
    opcode: The instruction's opcode name, e.g. "kSend".
    instruction_text: The instruction's `to_string()`.

  Returns:
    A queue id, or None if the instruction is not queue-ordered.
  """
  if _MEGASCALE_HANDLER in instruction_text:
    if opcode == "kSend":
      return _DCN_SEND_QUEUE
    if opcode == "kRecv":
      return _DCN_RECV_QUEUE
  if opcode == "kAsyncStart" and _HOST_MEMORY_SPACE in instruction_text:
    if " dynamic-slice-start(" in instruction_text:
      return _HOST_H2D_QUEUE
    if " dynamic-update-slice-start(" in instruction_text:
      return _HOST_D2H_QUEUE
  thread = _ASYNC_THREAD_RE.search(instruction_text)
  cores = _CORE_IDS_RE.search(instruction_text)
  if thread is None or cores is None:
    return None
  return f"{thread.group(1)}:{cores.group(1).strip()}"


def megascale_glue(
    sequence: Sequence["_hlo.HloInstruction"],
) -> dict[str, str]:
  """Pairs each megascale DCN recv with its send, and their dones.

  Megascale lowers a cross-slice collective into a send and a recv that share
  an `after-all` token. That token is the only link between the two: these
  instructions carry no channel id. A recv must be posted before its send goes
  out, and the data it receives must not be waited on until the send has
  drained, so the recv is glued before the send and the send's done before the
  recv's done.

  Args:
    sequence: The computation's scheduled instructions.

  Returns:
    Follower -> anchor, suitable for `make_schedule_graph(glue=...)`.
  """
  recv_by_token: dict[str, str] = {}
  send_by_token: dict[str, str] = {}
  done_of: dict[str, str] = {}
  for inst in sequence:
    opcode = inst.opcode.name
    if opcode in ("kSendDone", "kRecvDone"):
      operands = inst.operands()
      if operands:
        done_of[operands[0].name] = inst.name
      continue
    if opcode not in ("kSend", "kRecv"):
      continue
    if _MEGASCALE_HANDLER not in inst.to_string():
      continue
    tokens = [o.name for o in inst.operands() if o.opcode.name == "kAfterAll"]
    if len(tokens) != 1:
      continue
    by_token = recv_by_token if opcode == "kRecv" else send_by_token
    by_token[tokens[0]] = inst.name

  glue: dict[str, str] = {}
  for token, recv in recv_by_token.items():
    send = send_by_token.get(token)
    if send is None:
      continue
    glue[recv] = send
    send_done = done_of.get(send)
    recv_done = done_of.get(recv)
    if send_done is not None and recv_done is not None:
      glue[send_done] = recv_done
  return glue


def _build_graph(
    sequence: Sequence["_hlo.HloInstruction"],
    computations_by_name: Mapping[str, "_hlo.HloComputation"],
    fusion_cache: dict[str, bool],
) -> ScheduleGraph:
  """Builds a `ScheduleGraph` from a scheduled instruction sequence."""
  order = [inst.name for inst in sequence]
  opcodes = {inst.name: inst.opcode.name for inst in sequence}
  operands = {inst.name: [o.name for o in inst.operands()] for inst in sequence}

  control_preds: dict[str, tuple[str, ...]] = {}
  data_formatting: list[str] = []
  queues: dict[str, str] = {}
  for inst in sequence:
    text = inst.to_string()
    control_preds[inst.name] = _parse_control_predecessors(text)
    opcode = inst.opcode.name
    if opcode in _DATA_FORMATTING_OPCODES:
      data_formatting.append(inst.name)
    elif opcode == "kFusion" and _fusion_is_data_formatting(text, computations_by_name, fusion_cache):
      data_formatting.append(inst.name)
    elif opcode == "kCustomCall" and _custom_call_is_data_formatting(text):
      data_formatting.append(inst.name)
    if opcode in _ASYNC_START_OPCODES:
      queue = async_queue(opcode, text)
      if queue is not None:
        queues[inst.name] = queue

  return make_schedule_graph(
      order=order,
      opcodes=opcodes,
      operands=operands,
      control_preds=control_preds,
      data_formatting=data_formatting,
      queues=queues,
      glue=megascale_glue(sequence),
      serialized_queues=_SERIALIZED_QUEUES,
  )


def while_body_names(module: "_hlo.HloModule") -> set[str]:
  """Returns the names of every computation used as a while body."""
  names: set[str] = set()
  for computation in module.make_nonfusion_computations():
    for inst in computation.instructions():
      if inst.opcode.name != "kWhile":
        continue
      match = _BODY_RE.search(inst.to_string())
      if match is not None:
        names.add(match.group(1).lstrip("%"))
  return names


def default_computation_filter(
    module: "_hlo.HloModule",
) -> Callable[["_hlo.HloComputation"], bool]:
  """Returns a predicate selecting while-body computations."""
  bodies = while_body_names(module)
  return lambda computation: computation.name in bodies


def transform_module(
    module: "_hlo.HloModule",
    computation_filter: Callable[["_hlo.HloModule"], Callable[["_hlo.HloComputation"], bool]] | None = None,
) -> list[tuple[str, ScheduleGraph, RescheduleResult]]:
  """Rewrites the schedule of every selected computation in `module`.

  Args:
    module: The module to rewrite in place.
    computation_filter: Factory returning a predicate over computations.
      Defaults to selecting while bodies.

  Returns:
    One (computation name, graph, result) triple per rewritten computation.

  Raises:
    ScheduleTransformError: If a rewritten schedule fails validation.
  """
  schedule = module.schedule()
  if schedule is None:
    return []

  filter_factory = computation_filter or default_computation_filter
  selects = filter_factory(module)
  computations_by_name = {c.name: c for c in module.computations()}
  fusion_cache: dict[str, bool] = {}
  results: list[tuple[str, ScheduleGraph, RescheduleResult]] = []

  for computation in module.make_nonfusion_computations():
    if not selects(computation):
      continue
    try:
      sequence = schedule.sequence(computation)
    except Exception:  # pylint: disable=broad-except
      continue
    if not sequence:
      continue

    graph = _build_graph(sequence, computations_by_name, fusion_cache)
    result = reschedule(graph)
    validate_order(graph, result.order)

    by_name = {inst.name: inst for inst in sequence}
    schedule.set_sequence(computation, [by_name[name] for name in result.order])
    results.append((computation.name, graph, result))

  schedule.verify()
  module.set_schedule(schedule)
  return results


def restamp_send_recv_aggregation(serialized_hlo: bytes, computation_names: Iterable[str]) -> bytes:
  """Recomputes AggregateSendRecv's DEFER/ISSUE stamps on rewritten schedules.

  XLA runs AggregateSendRecv before post-scheduler transforms, so its stamps
  describe groups in a schedule that no longer exists. Only ops XLA already
  stamped are touched: being stamped is what marks one as an eligible megascale
  transfer. The HLO bindings cannot write backend configs, hence the proto.

  Args:
    serialized_hlo: Serialized HloModuleProto bytes.
    computation_names: Computations whose schedule was rewritten.

  Returns:
    Serialized HloModuleProto bytes.

  Raises:
    ScheduleTransformError: If a DEFER would be left with no ISSUE after it.
  """
  names = frozenset(computation_names)
  proto = hlo_pb2.HloModuleProto.FromString(serialized_hlo)
  for computation in proto.computations:
    if computation.name not in names:
      continue
    if computation.id not in proto.schedule.sequences:
      continue
    by_id = {inst.id: inst for inst in computation.instructions}
    sequence = [by_id[i] for i in proto.schedule.sequences[computation.id].instruction_ids]
    name_of = {inst.id: inst.name for inst in computation.instructions}

    configs: dict[str, dict[str, Any]] = {}
    transfers: set[str] = set()
    for inst in sequence:
      if inst.opcode not in ("send", "recv") or not inst.backend_config:
        continue
      config = json.loads(inst.backend_config)
      if _AGGREGATED_SEND_RECV_CONFIG in config:
        configs[inst.name] = config
        transfers.add(inst.name)
    dones = {
        inst.name
        for inst in sequence
        if inst.opcode in ("send-done", "recv-done")
        and inst.operand_ids
        and name_of.get(inst.operand_ids[0]) in transfers
    }
    free = {
        inst.name
        for inst in sequence
        if inst.opcode in ("parameter", "after-all") and not inst.operand_ids and not inst.control_predecessor_ids
    }
    order = [inst.name for inst in sequence]
    statuses = aggregation_statuses(
        order=order,
        operands={inst.name: [name_of[i] for i in inst.operand_ids] for inst in sequence},
        control_preds={inst.name: [name_of[i] for i in inst.control_predecessor_ids] for inst in sequence},
        transfers=transfers,
        dones=dones,
        free=free,
        # A proto channel_id of 0 means unset; host channels are only assigned
        # later, in the backend.
        channel_ids={
            inst.name: inst.channel_id
            for inst in sequence
            if (inst.name in transfers or inst.name in dones) and inst.channel_id
        },
    )
    stranded = stranded_defers(order, statuses)
    if stranded:
      raise ScheduleTransformError(f"DEFER'd transfers {stranded} in {computation.name} have no ISSUE" " after them.")
    for inst in sequence:
      status = statuses.get(inst.name)
      config = configs.get(inst.name)
      if status is None or config is None:
        continue
      if config[_AGGREGATED_SEND_RECV_CONFIG].get("status") == status:
        continue
      config[_AGGREGATED_SEND_RECV_CONFIG]["status"] = status
      inst.backend_config = json.dumps(config, separators=(",", ":")).encode()
  return proto.SerializeToString()


def max_async_overlap_transform(serialized_hlo: bytes) -> bytes:
  """XLA post-scheduler transformation pass.

  Bubbles async starts as early as possible and async dones as late as possible
  in every while-body computation, to maximize overlap, then restamps send/recv
  aggregation to match the new schedules.

  Args:
    serialized_hlo: Serialized HloModuleProto bytes.

  Returns:
    Serialized HloModuleProto bytes.
  """
  from_proto = getattr(_hlo.HloModule, "from_serialized_hlo_module_proto")
  module: _hlo.HloModule = from_proto(serialized_hlo)
  results = transform_module(module)
  return restamp_send_recv_aggregation(
      module.as_serialized_hlo_module_proto(),
      [name for name, _, _ in results],
  )


def _blocker_label(graph: ScheduleGraph, blocker: str | None) -> str:
  """Renders a blocking instruction as "name (opcode)" for the debug report."""
  if blocker is None:
    return "-"
  return f"{blocker} ({graph.opcodes.get(blocker, '?')})"


def debug_report(serialized_hlo: bytes, max_bundle_members: int = 6) -> str:
  """Returns a human-readable trace of what the transform did.

  Args:
    serialized_hlo: Serialized HloModuleProto bytes.
    max_bundle_members: How many bundle members to list per async op.

  Returns:
    A multi-line report, one section per rewritten computation.
  """
  from_proto = getattr(_hlo.HloModule, "from_serialized_hlo_module_proto")
  module: _hlo.HloModule = from_proto(serialized_hlo)
  results = transform_module(module)

  lines: list[str] = []
  for name, graph, result in results:
    original = graph.positions()
    final = {n: i for i, n in enumerate(result.order)}
    lines.append("=" * 100)
    lines.append(f"COMPUTATION {name}: {len(graph.order)} instructions," f" {len(result.pairs)} async pairs")
    lines.append("=" * 100)

    lines.append("-- ASYNC STARTS (bubble order; final order is the reverse)")
    lines.append(f"{'#':<4}{'start':<40}{'orig':>7}{'probe':>7}{'final':>7}" f"  {'blocker':<34}bundle")
    for i, start in enumerate(result.start_bubble_order):
      probe = result.start_probes[start]
      members = sorted(result.start_bundles[start] - {start})
      shown = ", ".join(members[:max_bundle_members])
      if len(members) > max_bundle_members:
        shown += f", (+{len(members) - max_bundle_members} more)"
      lines.append(
          f"{i:<4}{start:<40}{original[start]:>7}{probe.slot:>7}{final[start]:>7}"
          f"  {_blocker_label(graph, probe.blocker):<40}{shown}"
      )

    lines.append("")
    lines.append("-- ASYNC DONES (final order)")
    lines.append(f"{'#':<4}{'done':<40}{'orig':>7}{'probe':>7}{'final':>7}" f"  {'blocker':<34}bundle")
    for i, done in enumerate(result.done_bubble_order):
      probe = result.done_probes[done]
      members = sorted(result.done_bundles[done] - {done})
      shown = ", ".join(members[:max_bundle_members])
      if len(members) > max_bundle_members:
        shown += f", (+{len(members) - max_bundle_members} more)"
      lines.append(
          f"{i:<4}{done:<40}{original[done]:>7}{probe.slot:>7}"
          f"{final[done]:>7}  {_blocker_label(graph, probe.blocker):<40}{shown}"
      )
    lines.append("")
  return "\n".join(lines)


def register_transform(
    name: str = _DEFAULT_TRANSFORM_NAME,
    platforms: Sequence[str] | str | None = None,
) -> None:
  """Registers the post-scheduler transformation with JAX/XLA."""
  jex_xla.register_hlo_module_transformation(
      max_async_overlap_transform,
      name=name,
      stage=jex_xla.PipelineStage.POST_SCHEDULER,
      platforms=platforms,
  )


def unregister_transform(
    name: str = _DEFAULT_TRANSFORM_NAME,
    platforms: Sequence[str] | str | None = None,
) -> bool:
  """Unregisters the post-scheduler transformation from JAX/XLA."""
  return jex_xla.clear_hlo_module_transformation(
      name=name,
      stage=jex_xla.PipelineStage.POST_SCHEDULER,
      platforms=platforms,
  )
