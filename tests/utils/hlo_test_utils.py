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
"""Helpers for asserting on the collectives in attention HLO text.

The helpers work on HLO text lines such as:

  ppermute.1 = f32[2,16]{1,0} collective-permute(shard_map.2), ...
  all_gather.1 = s32[64]{0} all-gather(broadcast.1), ...

Compiled HLO may rewrite a collective into its asynchronous -start/-done
pair; the -start instruction is counted and the -done instruction is not, so
each collective is counted once.
"""

import re

_FLOAT_TYPES = ("bf16", "f16", "f32", "f64")


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
