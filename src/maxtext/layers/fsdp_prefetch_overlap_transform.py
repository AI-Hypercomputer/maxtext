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

"""Post-scheduler XLA HLO transform for FSDP weight prefetching.

Hoists async collective starts (weight all-gathers and deferred weight-gradient
reduce-scatters) to the start of each layer and sinks their matching async dones
to the end of each layer in the entry computation and every scan `while` loop
body, while preserving per-SparseCore FIFO queue ordering and leaving host-offload
transfers in their existing XLA schedule.
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
import dataclasses
import re

from jax._src.lib import hlo as _hlo
import jax.extend.xla as jex_xla

_DEFAULT_TRANSFORM_NAME = "fsdp_prefetch_overlap_transform"

_ASYNC_PAIRS: Mapping[str, str] = {
    "kAsyncStart": "kAsyncDone",
    "kAllGatherStart": "kAllGatherDone",
    "kAllReduceStart": "kAllReduceDone",
    "kCollectivePermuteStart": "kCollectivePermuteDone",
}
_ASYNC_START_OPCODES = frozenset(_ASYNC_PAIRS)
_ASYNC_DONE_OPCODES = frozenset(_ASYNC_PAIRS.values())
_DONE_TO_START_OPCODE: Mapping[str, str] = {done: start for start, done in _ASYNC_PAIRS.items()}

# Lightweight slicing, view, and barrier ops pulled along with the async start
# they feed or the async done they consume.
_DATA_FORMATTING_OPCODES = frozenset(
    {
        "kAdd",
        "kBitcast",
        "kBitcastConvert",
        "kBroadcast",
        "kClamp",
        "kCompare",
        "kConcatenate",
        "kConstant",
        "kConvert",
        "kCopy",
        "kDynamicSlice",
        "kDynamicUpdateSlice",
        "kGetTupleElement",
        "kIota",
        "kOptimizationBarrier",
        "kPad",
        "kReshape",
        "kSelect",
        "kSlice",
        "kSubtract",
        "kTranspose",
        "kTuple",
    }
)
_DATA_FORMATTING_CUSTOM_CALLS = frozenset({"AllocateBuffer"})
_FUSION_INTERNAL_EXTRA_OPCODES = frozenset({"kParameter", "kReduce"})

_CONTROL_PREDECESSORS_RE = re.compile(r"control-predecessors=\{([^}]*)\}")
_CALLS_RE = re.compile(r"\bcalls=(%?[\w.\-]+)")
_CUSTOM_CALL_TARGET_RE = re.compile(r'custom_call_target="([^"]*)"')
_BODY_RE = re.compile(r"\bbody=(%?[\w.\-]+)")
_INSTRUCTION_NAME_RE = re.compile(r"%([\w.\-]+)")
_ASYNC_THREAD_RE = re.compile(r'async_execution_thread="([^"]+)"')
_CORE_IDS_RE = re.compile(r'"core_ids":\s*\[([^\]]*)\]')
_ENTRY_RE = re.compile(r"^ENTRY %([\w.\-]+)", re.MULTILINE)

_HOST_MEMORY_SPACE = "S(5)"


class ScheduleTransformError(Exception):
  """Raised when the transform cannot produce a valid schedule."""


@dataclasses.dataclass(frozen=True)
class ScheduleGraph:
  """One computation's schedule and dependency graph."""

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

  def positions(self) -> dict[str, int]:
    return {name: i for i, name in enumerate(self.order)}


@dataclasses.dataclass(frozen=True)
class AsyncPair:
  """An async start paired with its matching async done."""

  start: str
  done: str


def make_schedule_graph(
    order: Sequence[str],
    opcodes: Mapping[str, str],
    operands: Mapping[str, Sequence[str]],
    *,
    control_preds: Mapping[str, Sequence[str]] | None = None,
    data_formatting: Iterable[str] = (),
    queues: Mapping[str, str] | None = None,
    fences: Iterable[str] = (),
) -> ScheduleGraph:
  """Builds a `ScheduleGraph`, deriving users, control successors, and fences."""
  order = tuple(order)
  present = set(order)
  if not order or len(present) != len(order):
    raise ScheduleTransformError("Schedule must be non-empty without duplicate instructions.")

  control_preds = control_preds or {}
  queues = {k: v for k, v in (queues or {}).items() if k in present}
  fences = frozenset(fences)
  operand_map: dict[str, tuple[str, ...]] = {}
  control_pred_lists: dict[str, list[str]] = {}
  user_lists: dict[str, list[str]] = {name: [] for name in order}
  control_succ_lists: dict[str, list[str]] = {name: [] for name in order}

  fence: str | None = None
  for name in order:
    operand_map[name] = tuple(o for o in operands.get(name, ()) if o in present)
    preds = [p for p in control_preds.get(name, ()) if p in present]
    if opcodes.get(name) in _ASYNC_START_OPCODES and fence is not None and fence not in preds:
      preds.append(fence)
    control_pred_lists[name] = preds
    if name in fences:
      fence = name

  fence = None
  for name in reversed(order):
    if name in fences:
      fence = name
    elif fence is not None and opcodes.get(name) in _ASYNC_DONE_OPCODES:
      if name not in control_pred_lists[fence]:
        control_pred_lists[fence].append(name)

  control_pred_map = {name: tuple(control_pred_lists[name]) for name in order}
  for name in order:
    for operand in operand_map[name]:
      user_lists[operand].append(name)
    for pred in control_pred_map[name]:
      control_succ_lists[pred].append(name)

  root = order[-1]
  param_prefix_len = next((i for i, n in enumerate(order) if opcodes.get(n) != "kParameter"), len(order))
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
  )


def find_async_pairs(graph: ScheduleGraph) -> list[AsyncPair]:
  """Finds async start/done pairs in schedule order of the done instruction."""
  pairs: list[AsyncPair] = []
  for name in graph.order:
    opcode = graph.opcodes[name]
    if opcode == "kAsyncUpdate":
      raise ScheduleTransformError(f"kAsyncUpdate is not supported (instruction {name}).")
    if opcode not in _ASYNC_DONE_OPCODES:
      continue
    operands = graph.operands[name]
    expected = _DONE_TO_START_OPCODE[opcode]
    if not operands or graph.opcodes.get(operands[0]) != expected:
      raise ScheduleTransformError(f"Async done {name} ({opcode}) has no {expected} as operand 0.")
    pairs.append(AsyncPair(start=operands[0], done=name))
  return pairs


def _closure(graph: ScheduleGraph, seed: str, edges: Mapping[str, tuple[str, ...]]) -> frozenset[str]:
  """Grows a bundle from `seed` along `edges` through data-formatting ops."""
  bundle, stack = {seed}, [seed]
  while stack:
    for neighbor in edges[stack.pop()]:
      if neighbor not in bundle and neighbor not in graph.pinned and neighbor in graph.data_formatting:
        bundle.add(neighbor)
        stack.append(neighbor)
  return frozenset(bundle)


def earliest_slot(graph: ScheduleGraph, bundle: frozenset[str], positions: Mapping[str, int]) -> int:
  """Returns the lowest schedule index `bundle` can start at."""
  slot = graph.param_prefix_len
  for member in bundle:
    for dep in (*graph.operands[member], *graph.control_preds[member]):
      if dep not in bundle and positions[dep] + 1 > slot:
        slot = positions[dep] + 1
  return slot


def latest_slot(graph: ScheduleGraph, bundle: frozenset[str], positions: Mapping[str, int]) -> int:
  """Returns the highest schedule index `bundle` can end at (before `root`)."""
  slot = positions[graph.root] - 1
  for member in bundle:
    for dep in (*graph.users[member], *graph.control_succs[member]):
      if dep not in bundle and positions[dep] - 1 < slot:
        slot = positions[dep] - 1
  return slot


def prune_bundle_for_start(
    graph: ScheduleGraph, seed: str, bundle: frozenset[str], positions: Mapping[str, int], min_slot: int = 0
) -> tuple[frozenset[str], int]:
  """Shrinks a start bundle until no member would be dragged later."""
  current = set(bundle)
  while True:
    slot = earliest_slot(graph, frozenset(current), positions)
    dropped = {m for m in current if positions[m] < max(slot, min_slot)}
    if seed in dropped:
      if positions[seed] < slot:
        raise ScheduleTransformError(f"Async start {seed} precedes its own lower bound {slot}.")
      dropped.discard(seed)
    if not dropped:
      return frozenset(current), slot
    current -= dropped


def prune_bundle_for_done(
    graph: ScheduleGraph, seed: str, bundle: frozenset[str], positions: Mapping[str, int], max_slot: int | None = None
) -> tuple[frozenset[str], int]:
  """Shrinks a done bundle until no member would be dragged earlier."""
  current = set(bundle)
  while True:
    slot = latest_slot(graph, frozenset(current), positions)
    target = slot if max_slot is None else min(slot, max_slot)
    dropped = {m for m in current if positions[m] > target}
    if not dropped:
      return frozenset(current), slot
    if seed in dropped:
      alone = frozenset({seed})
      if earliest_slot(graph, alone, positions) > target:
        raise ScheduleTransformError(f"Async done {seed} cannot move back to its queue cap {target}.")
      return alone, latest_slot(graph, alone, positions)
    current -= dropped


def _move_to_start_slot(order: Sequence[str], bundle: frozenset[str], slot: int) -> list[str]:
  members = [name for name in order if name in bundle]
  rest = [name for name in order if name not in bundle]
  index = slot - sum(1 for name in order[:slot] if name in bundle)
  return rest[:index] + members + rest[index:]


def _move_to_end_slot(order: Sequence[str], bundle: frozenset[str], slot: int) -> list[str]:
  members = [name for name in order if name in bundle]
  rest = [name for name in order if name not in bundle]
  split = len(rest) - (len(order) - 1 - slot)
  return rest[:split] + members + rest[split:]


def _move_before(order: Sequence[str], bundle: frozenset[str], anchor: str) -> list[str]:
  members = [name for name in order if name in bundle]
  rest = [name for name in order if name not in bundle]
  index = rest.index(anchor)
  return rest[:index] + members + rest[index:]


def reschedule(graph: ScheduleGraph) -> tuple[str, ...]:
  """Schedules async starts at the start of each layer and async dones at the end."""
  pairs = find_async_pairs(graph)
  if not pairs:
    return graph.order
  done_of_start = {p.start: p.done for p in pairs}
  start_of_done = {p.done: p.start for p in pairs}
  starts = [p.start for p in pairs]
  dones = [p.done for p in pairs]
  positions = graph.positions()

  start_bundles, raw_start_slots = {}, {}
  for s in starts:
    start_bundles[s], raw_start_slots[s] = prune_bundle_for_start(graph, s, _closure(graph, s, graph.operands), positions)
  done_bundles, done_slots = {}, {}
  for d in dones:
    done_bundles[d], done_slots[d] = prune_bundle_for_done(graph, d, _closure(graph, d, graph.users), positions)

  # Rank shared-queue completions by (start_slot + done_slot, done_slot, start_slot, done)
  # and raise start floors so each FIFO queue issues in completion order.
  completion_order = sorted(
      dones,
      key=lambda d: (
          raw_start_slots.get(start_of_done[d], -1) + done_slots[d],
          done_slots[d],
          raw_start_slots.get(start_of_done[d], -1),
          d,
      ),
  )
  start_slots = dict(raw_start_slots)
  last_on_queue: dict[str, int] = {}
  for d in completion_order:
    s = start_of_done[d]
    q = graph.queues.get(s)
    if q is not None:
      if s in start_slots and q in last_on_queue and start_slots[s] < last_on_queue[q]:
        start_slots[s] = last_on_queue[q]
      last_on_queue[q] = start_slots[s]

  done_rank = {d: i for i, d in enumerate(completion_order)}
  start_bubble_order = sorted(
      sorted(starts, reverse=True),
      key=lambda s: (-start_slots[s], -done_rank.get(done_of_start[s], -1)),
  )
  current = list(graph.order)
  for s in start_bubble_order:
    pos = {name: i for i, name in enumerate(current)}
    floor = pos[graph.order[start_slots[s] - 1]] + 1 if start_slots.get(s, 0) > 0 else 0
    bundle, slot = prune_bundle_for_start(graph, s, start_bundles[s], pos, floor)
    current = _move_to_start_slot(current, bundle, max(slot, floor))

  # Re-probe dones against the hoisted schedule and sink them right, respecting FIFO caps.
  pos = {name: i for i, name in enumerate(current)}
  for d in dones:
    done_bundles[d], done_slots[d] = prune_bundle_for_done(graph, d, _closure(graph, d, graph.users), pos)
  total = len(graph.order)
  done_bubble_order = sorted(dones, key=lambda d: (-(total - 1 - done_slots[d]), pos[start_of_done[d]], d))
  issue = dict(pos)
  placed: list[str] = []
  for d in done_bubble_order:
    pos = {name: i for i, name in enumerate(current)}
    q = graph.queues.get(start_of_done[d])
    cappers = [
        other
        for other in placed
        if q is not None
        and graph.queues.get(start_of_done[other]) == q
        and issue[start_of_done[other]] > issue[start_of_done[d]]
    ]
    capper = min(cappers, key=pos.__getitem__, default=None)
    cap = None if capper is None else pos[capper] - 1
    bundle, probe_slot = prune_bundle_for_done(graph, d, done_bundles[d], pos, cap)
    slot = probe_slot if cap is None else min(probe_slot, cap)
    if capper is not None and slot < probe_slot:
      current = _move_before(current, bundle, capper)
    else:
      current = _move_to_end_slot(current, bundle, slot)
    placed.append(d)
  return tuple(current)


def validate_order(graph: ScheduleGraph, order: Sequence[str]) -> None:
  """Verifies that `order` is a legal topological schedule for `graph`."""
  if len(order) != len(graph.order) or set(order) != set(graph.order):
    raise ScheduleTransformError(f"Schedule changed instruction set (len {len(order)} vs {len(graph.order)}).")
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
  if tuple(order[: graph.param_prefix_len]) != graph.order[: graph.param_prefix_len]:
    raise ScheduleTransformError("Leading parameter run was disturbed.")


def _fusion_is_data_formatting(
    instruction_text: str,
    computations_by_name: Mapping[str, "_hlo.HloComputation"],
    cache: dict[str, bool],
) -> bool:
  """Returns True if a fusion's body contains only data-formatting ops."""
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
  cache[callee] = False
  allowed = _DATA_FORMATTING_OPCODES | _FUSION_INTERNAL_EXTRA_OPCODES
  for inner in computation.instructions():
    opcode = inner.opcode.name
    if opcode in allowed:
      continue
    if opcode == "kFusion" and _fusion_is_data_formatting(inner.to_string(), computations_by_name, cache):
      continue
    target = _CUSTOM_CALL_TARGET_RE.search(inner.to_string())
    if opcode == "kCustomCall" and target is not None and target.group(1) in _DATA_FORMATTING_CUSTOM_CALLS:
      continue
    return False
  cache[callee] = True
  return True


def _is_host_async_start(opcode: str, instruction_text: str) -> bool:
  """Returns True if an async-start is a host DMA transfer or host call."""
  if opcode != "kAsyncStart":
    return False
  if _HOST_MEMORY_SPACE in instruction_text:
    return True
  thread = _ASYNC_THREAD_RE.search(instruction_text)
  return thread is not None and thread.group(1) == "host"


def async_queue(opcode: str, instruction_text: str) -> str | None:
  """Returns the hardware FIFO queue an async collective start runs on, or None."""
  if opcode not in _ASYNC_START_OPCODES or _is_host_async_start(opcode, instruction_text):
    return None
  thread = _ASYNC_THREAD_RE.search(instruction_text)
  cores = _CORE_IDS_RE.search(instruction_text)
  if thread is None or cores is None:
    return None
  return f"{thread.group(1)}:{cores.group(1).strip()}"


def _build_graph(
    sequence: Sequence["_hlo.HloInstruction"],
    computations_by_name: Mapping[str, "_hlo.HloComputation"],
    fusion_cache: dict[str, bool],
    async_bodies: Collection[str] = (),
) -> ScheduleGraph:
  """Builds a `ScheduleGraph` from a scheduled HLO instruction sequence."""
  order = [inst.name for inst in sequence]
  opcodes: dict[str, str] = {}
  operands = {inst.name: [o.name for o in inst.operands()] for inst in sequence}
  control_preds: dict[str, tuple[str, ...]] = {}
  data_formatting: list[str] = []
  queues: dict[str, str] = {}
  fences: list[str] = []
  for inst in sequence:
    text = inst.to_string()
    cp_match = _CONTROL_PREDECESSORS_RE.search(text)
    control_preds[inst.name] = () if cp_match is None else tuple(_INSTRUCTION_NAME_RE.findall(cp_match.group(1)))
    opcode = inst.opcode.name
    if _is_host_async_start(opcode, text):
      opcode = "kHostAsyncStart"
    elif opcode == "kAsyncDone" and operands[inst.name] and opcodes.get(operands[inst.name][0]) == "kHostAsyncStart":
      opcode = "kHostAsyncDone"
    opcodes[inst.name] = opcode
    target = _CUSTOM_CALL_TARGET_RE.search(text) if opcode == "kCustomCall" else None
    if (
        opcode in _DATA_FORMATTING_OPCODES
        or (opcode == "kFusion" and _fusion_is_data_formatting(text, computations_by_name, fusion_cache))
        or (target is not None and target.group(1) in _DATA_FORMATTING_CUSTOM_CALLS)
    ):
      data_formatting.append(inst.name)
    if opcode in _ASYNC_START_OPCODES:
      queue = async_queue(opcode, text)
      if queue is not None:
        queues[inst.name] = queue
    elif opcode == "kWhile":
      body_match = _BODY_RE.search(text)
      if body_match is not None and body_match.group(1).lstrip("%") in async_bodies:
        fences.append(inst.name)

  return make_schedule_graph(
      order=order,
      opcodes=opcodes,
      operands=operands,
      control_preds=control_preds,
      data_formatting=data_formatting,
      queues=queues,
      fences=fences,
  )


def _while_and_async_body_names(module: "_hlo.HloModule") -> tuple[set[str], set[str]]:
  """Returns (all while-body names, while-body names that issue async work)."""
  issues_directly: dict[str, bool] = {}
  nested_bodies: dict[str, list[str]] = {}
  for computation in module.make_nonfusion_computations():
    issues_directly[computation.name] = False
    nested_bodies[computation.name] = []
    for inst in computation.instructions():
      opcode = inst.opcode.name
      if opcode in _ASYNC_START_OPCODES:
        issues_directly[computation.name] = True
      elif opcode == "kWhile":
        match = _BODY_RE.search(inst.to_string())
        if match is not None:
          nested_bodies[computation.name].append(match.group(1).lstrip("%"))

  memo: dict[str, bool] = {}

  def issues(name: str) -> bool:
    if name not in memo:
      memo[name] = False
      memo[name] = issues_directly.get(name, False) or any(issues(n) for n in nested_bodies.get(name, ()))
    return memo[name]

  all_bodies = set().union(*nested_bodies.values()) if nested_bodies else set()
  return all_bodies, {b for b in all_bodies if issues(b)}


def transform_module(module: "_hlo.HloModule") -> list[str]:
  """Rewrites the schedule of the entry computation and every while body in `module`."""
  schedule = module.schedule()
  if schedule is None:
    return []

  options = _hlo.HloPrintOptions()
  for attr in (
      "print_backend_config",
      "print_extra_attributes",
      "print_metadata",
      "print_operand_shape",
      "print_program_shape",
      "print_result_shape",
  ):
    setattr(options, attr, False)
  entry_match = _ENTRY_RE.search(module.to_string(options))
  if entry_match is None:
    raise ScheduleTransformError(f"Module {module.name} has no entry.")

  while_bodies, async_bodies = _while_and_async_body_names(module)
  selected = while_bodies | {entry_match.group(1)}
  computations_by_name = {c.name: c for c in module.computations()}
  fusion_cache: dict[str, bool] = {}
  rewritten: list[str] = []

  for computation in module.make_nonfusion_computations():
    if computation.name not in selected:
      continue
    try:
      sequence = schedule.sequence(computation)
    except Exception:  # pylint: disable=broad-except
      continue
    if not sequence:
      continue
    graph = _build_graph(sequence, computations_by_name, fusion_cache, async_bodies)
    new_order = reschedule(graph)
    validate_order(graph, new_order)
    by_name = {inst.name: inst for inst in sequence}
    schedule.set_sequence(computation, [by_name[name] for name in new_order])
    rewritten.append(computation.name)

  schedule.verify()
  module.set_schedule(schedule)
  return rewritten


def fsdp_prefetch_overlap_transform(serialized_hlo: bytes) -> bytes:
  """XLA post-scheduler transformation pass for FSDP weight prefetching."""
  from_proto = getattr(_hlo.HloModule, "from_serialized_hlo_module_proto")
  module: _hlo.HloModule = from_proto(serialized_hlo)
  transform_module(module)
  return module.as_serialized_hlo_module_proto()


def register_transform(
    name: str = _DEFAULT_TRANSFORM_NAME,
    platforms: Sequence[str] | str | None = None,
) -> None:
  """Registers the FSDP prefetch post-scheduler transformation with JAX/XLA."""

  def _transform_if_applicable(serialized_hlo: bytes) -> bytes | None:
    if b"fsdp_prefetch" not in serialized_hlo:
      return None
    return fsdp_prefetch_overlap_transform(serialized_hlo)

  unregister_transform(name=name, platforms=platforms)
  jex_xla.register_hlo_module_transformation(
      _transform_if_applicable,
      name=name,
      stage=jex_xla.PipelineStage.POST_SCHEDULER,
      platforms=platforms,
  )


def unregister_transform(
    name: str = _DEFAULT_TRANSFORM_NAME,
    platforms: Sequence[str] | str | None = None,
) -> bool:
  """Unregisters the FSDP prefetch post-scheduler transformation from JAX/XLA."""
  return jex_xla.clear_hlo_module_transformation(
      name=name,
      stage=jex_xla.PipelineStage.POST_SCHEDULER,
      platforms=platforms,
  )
