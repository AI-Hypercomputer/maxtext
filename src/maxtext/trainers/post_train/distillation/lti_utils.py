# Copyright 2023–2026 Google LLC
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

"""Utilities for Learn-to-Init phase of the distillation"""

from flax import nnx
from maxtext.utils import max_logging


def _nav_to_attr(root, parts):
  """Walk `root.parts[0].parts[1]...` and return the terminal Python object.

  Handles both attribute access and mapping/sequence access (integer-looking
  components are tried as sequence indices as a last resort) so string paths
  from `nnx.graph.iter_graph` round-trip across scanned and unscanned models.
  """
  obj = root
  for p in parts:
    if hasattr(obj, p):
      obj = getattr(obj, p)
      continue
    try:
      obj = obj[p]
      continue
    except (KeyError, TypeError, IndexError):
      pass
    try:
      obj = obj[int(p)]
      continue
    except (ValueError, TypeError, KeyError, IndexError):
      pass
    raise AttributeError(
        f"prepare_student_weights: cannot resolve path component {p!r} on "
        f"{type(obj).__name__}. Make sure copy/share map paths match "
        f"nnx.graph.iter_graph output for this model."
    )
  return obj


def prepare_student_weights(
    student_model: nnx.Module,
    teacher_model: nnx.Module,
    teacher_weights_copy_map: dict[str, str],
    student_weights_share_map: dict[str, str],
):
  """Injects weights from a teacher model into a student model in-place as well as
  shares specific pairs of weights inside the student model - might be useful for learn-to-init experiments.

  This function iterates through the provided mapping dictionaries and
  copies the corresponding weights from the teacher.

  It works by matching the graph path of a module in the student to the same
  path in the teacher.

  Args:
    student_model: The student model (NNX Module), which will be modified.
    teacher_model: The teacher model (NNX Module), used as the source.
    teacher_weights_copy_map: A dictionary mapping teacher parameter paths (as strings)
      to student parameter paths.
    student_weights_share_map: A dictionary mapping student parameter paths to be shared.
  """
  max_logging.log("Starting teacher weight injection...")

  # Get a dictionary view of the teacher graph for efficient lookups
  teacher_graph = {"/".join(map(str, path)): node for path, node in nnx.graph.iter_graph(teacher_model)}
  student_graph = {"/".join(map(str, path)): node for path, node in nnx.graph.iter_graph(student_model)}

  # --- Weight sharing (alias destination -> source Variable) ---
  for source_path, dest_path in student_weights_share_map.items():
    source_node = student_graph.get(source_path)
    dest_node = student_graph.get(dest_path)
    assert (
        source_node is not None
    ), f"Student parameter sharing: Could not find source_node model parameter at path: {source_path}"
    assert (
        dest_node is not None
    ), f"Student parameter sharing: Could not find dest_node model parameter at path: {dest_path}"

    assert source_node.value.shape == dest_node.value.shape, (
        f"Shape mismatch for sharing parameter between {source_path} and {dest_path}: "
        f"{source_node.value.shape} vs {dest_node.value.shape}"
    )

    max_logging.info(f"Sharing parameter {source_path} with {dest_path}")
    dest_parts = dest_path.split("/")

    dest_parent = _nav_to_attr(student_model, dest_parts[:-1])
    dest_attr = dest_parts[-1]

    if hasattr(dest_parent, dest_attr):
      setattr(dest_parent, dest_attr, source_node)
    else:
      dest_parent[dest_attr] = source_node

  for teacher_path, student_path in teacher_weights_copy_map.items():
    teacher_node = teacher_graph.get(teacher_path)
    student_node = student_graph.get(student_path)
    assert teacher_node is not None, f"Could not find teacher model parameter at path: {teacher_path}"
    assert student_node is not None, f"Could not find student model parameter at path: {student_path}"
    assert (
        student_node.value.shape == teacher_node.value.shape
    ), f"Shape mismatch for {teacher_path}. Teacher: {teacher_node.value.shape}, Student: {student_node.value.shape}"
    student_node.value = teacher_node.value
    max_logging.info(f"Inserted teacher weight parameter {teacher_path} to the student at {student_path}")

  max_logging.info("Teacher weight injection complete.")
