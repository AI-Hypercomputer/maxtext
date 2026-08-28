# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helpers for asserting on the collectives in HLO text.

The helpers work on HLO text lines such as:

  ppermute.1 = f32[2,16]{1,0} collective-permute(shard_map.2), ...
  all_gather.1 = s32[64]{0} all-gather(broadcast.1), ...

Compiled HLO may rewrite a collective into its asynchronous -start/-done
pair; the -start instruction is counted and the -done instruction is not, so
each collective is counted once.
"""

import re

_FLOAT_TYPES = ("bf16", "f16", "f32", "f64")

# `%name (p: f32[8]) -> f32[8] {`, optionally prefixed with ENTRY, at column 0.
_COMPUTATION_HEADER = re.compile(r"^(ENTRY )?%?([\w.\-]+)\s*\(.*\)\s*->\s*.*\{\s*$")
# Computation operands named by a single attribute, e.g. `body=%loop_body`.
_CALLEE = re.compile(r"(?:body|condition|to_apply|true_computation|false_computation|select|scatter)=%?([\w.\-]+)")
# Computation operands named by a braced list, e.g. `called_computations={%a, %b}`.
_CALLEE_LIST = re.compile(r"(?:called_computations|branch_computations)=\{([^}]*)\}")


def collective_lines(hlo_text, collective):
  """Returns the HLO instruction lines that call the given collective op."""
  pattern = re.compile(rf"\b{re.escape(collective)}(-start)?\(")
  return [line for line in hlo_text.splitlines() if "=" in line and pattern.search(line)]


def _result_shapes(line, collective):
  """Parses the result shapes of a collective instruction line.

  Everything bracketed before the op call belongs to the result; an
  asynchronous -start instruction has a tuple result whose elements include
  both the sharded operand buffer and the full output buffer.
  """
  call = re.search(rf"\b{re.escape(collective)}(-start)?\(", line)
  result_part = line[: call.start()] if call else line
  shapes = []
  for match in re.finditer(r"([a-z][a-z0-9]*)\[([0-9,]*)\]", result_part):
    dims = tuple(int(dim) for dim in match.group(2).split(",") if dim)
    shapes.append((match.group(1), dims))
  return shapes


def _collective_dimensions(line):
  """Parses the dimensions={...} attribute of a collective instruction line."""
  match = re.search(r"dimensions=\{([0-9,]*)\}", line)
  if not match:
    return ()
  return tuple(int(dim) for dim in match.group(1).split(",") if dim)


def _computations(hlo_text):
  """Splits HLO text into {computation name: instruction lines} and the entry name."""
  computations, entry, current = {}, None, None
  for line in hlo_text.splitlines():
    header = _COMPUTATION_HEADER.match(line)
    if header and not line.startswith(" "):
      current = header.group(2)
      computations[current] = []
      if header.group(1):
        entry = current
    elif line == "}":
      current = None
    elif current is not None:
      computations[current].append(line)
  return computations, entry


def _callees(line):
  """Names the computations an HLO instruction line invokes."""
  names = set(_CALLEE.findall(line))
  for group in _CALLEE_LIST.findall(line):
    names.update(name.strip().lstrip("%") for name in group.split(",") if name.strip())
  return names


def _reachable(computations, roots):
  """Names of `roots` plus every computation they transitively invoke."""
  seen, pending = set(), list(roots)
  while pending:
    name = pending.pop()
    if name in seen or name not in computations:
      continue
    seen.add(name)
    pending.extend(_callees(" ".join(computations[name])))
  return seen


def split_by_entry_loop(hlo_text, collective):
  """Splits a collective's instruction lines by whether they run inside an entry-level loop.

  Returns `(inside, outside)`. `inside` holds the lines belonging to computations
  reachable from the body of a `while` that the entry computation calls directly, so
  those instructions run once per loop iteration; `outside` holds the rest, which run
  once per call. A `jax.lax.scan` lowers to exactly such a `while`, which makes this the
  discriminator for "is this collective hoisted out of the scan or not".
  """
  computations, entry = _computations(hlo_text)
  if entry is None:
    raise ValueError("HLO text has no ENTRY computation")
  loop_bodies = [match.group(1) for line in computations[entry] for match in re.finditer(r"body=%?([\w.\-]+)", line)]
  in_loop = _reachable(computations, loop_bodies)
  lines = collective_lines(hlo_text, collective)
  in_loop_lines = {line for name in in_loop for line in computations[name]}
  inside = [line for line in lines if line in in_loop_lines]
  outside = [line for line in lines if line not in in_loop_lines]
  return inside, outside


def cross_replica_all_reduce_sizes(lines):
  """Element counts of the buffers reduced by each SPMD-generated all-reduce line.

  Only lines whose `replica_groups` name a mesh axis are counted. Those are the
  collectives SPMD partitioning inserts to reduce over a mesh axis; a `jax.shard_map`
  body instead emits manual-mode collectives, which spell their replica groups out as
  literal device ids and are not partitioning decisions the caller controls.
  """
  sizes = []
  for line in lines:
    if not re.search(r"replica_groups=mesh\[", line):
      continue
    for _, dims in _result_shapes(line, "all-reduce"):
      size = 1
      for dim in dims:
        size *= dim
      sizes.append(size)
  return sizes


def attention_sequence_all_gather_lines(hlo_text, sequence_lengths, dtypes=_FLOAT_TYPES):
  """Returns all-gather lines whose gathered dimension is a full-sequence dimension.

  A line matches only when the dimension named in its dimensions={...}
  attribute has a full-sequence result size, so a sequence-sized size on a
  non-gathered dimension does not match. dtypes restricts matches by result
  element type. The float default excludes the intended int32 segment-ID
  gathers, so a match means a full-sequence gather of activations or
  gradients; pass ("s32",) to count the segment-ID gathers instead.
  """
  lines = []
  for line in collective_lines(hlo_text, "all-gather"):
    gather_dims = _collective_dimensions(line)
    for result_type, dims in _result_shapes(line, "all-gather"):
      if result_type in dtypes and any(
          gather_dim < len(dims) and dims[gather_dim] in sequence_lengths for gather_dim in gather_dims
      ):
        lines.append(line)
        break
  return lines
