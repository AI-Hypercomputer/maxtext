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

from typing import Callable

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.utils import max_logging
import re


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


def _set_child(parent, name: str, value) -> None:
  if hasattr(parent, name):
    setattr(parent, name, value)
  else:
    parent[name] = value


def share_frozen_params(
    student_model: nnx.Module,
    teacher_model: nnx.Module,
    is_frozen: Callable[[tuple], bool],
) -> int:
  """Makes frozen student params reference the identical teacher params, so they are stored once.

  A student `nnx.Param` is replaced by the teacher's `nnx.Param` Variable at the
  same graph path only if `is_frozen(path)` is True and both arrays have the same
  shape, dtype, sharding and values. Afterwards both models hold the same
  Variable, so the array is stored once in device memory. Trainable params
  (`is_frozen(path)` False) are never shared, and params that differ from the
  teacher's (e.g. pruned or sparsified) keep their own value.

  Args:
    student_model: The student model, modified in place.
    teacher_model: The teacher model whose params are reused.
    is_frozen: Returns True for student param paths (tuples of path components)
      that are not trained.

  Returns:
    The number of shared params.
  """
  teacher_params = {path: node for path, node in nnx.graph.iter_graph(teacher_model) if isinstance(node, nnx.Param)}
  candidates = [
      (path, node)
      for path, node in nnx.graph.iter_graph(student_model)
      if isinstance(node, nnx.Param) and is_frozen(path) and path in teacher_params
  ]
  num_shared = 0
  num_skipped = 0
  for path, student_node in candidates:
    teacher_node = teacher_params[path]
    if student_node is teacher_node:
      continue
    s_val, t_val = student_node.get_value(), teacher_node.get_value()
    same_layout = (
        s_val.shape == t_val.shape
        and s_val.dtype == t_val.dtype
        and getattr(s_val, "sharding", None) == getattr(t_val, "sharding", None)
    )
    if not same_layout or not bool(jax.jit(jnp.array_equal)(s_val, t_val)):
      num_skipped += 1
      continue
    _set_child(_nav_to_attr(student_model, [str(p) for p in path[:-1]]), str(path[-1]), teacher_node)
    num_shared += 1
  max_logging.log(
      f"share_frozen_params: shared {num_shared} frozen student params with the teacher; {num_skipped} candidates "
      "kept their own copy because shape, dtype, sharding or values differ."
  )
  return num_shared


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

  for source_pattern, dest_pattern in student_weights_share_map.items():
    matched_any = False
    for source_path, source_node in student_graph.items():
      match = re.fullmatch(source_pattern, source_path)
      if match:
        matched_any = True
        dest_path = source_path if source_pattern == dest_pattern else match.expand(dest_pattern)
        dest_node = student_graph.get(dest_path)
        assert (
            dest_node is not None
        ), f"Student parameter sharing: Could not find dest_node model parameter at path: {dest_path}"
        assert source_node.get_value().shape == dest_node.get_value().shape, (
            f"Shape mismatch for sharing parameter between {source_path} and {dest_path}: "
            f"{source_node.get_value().shape} vs {dest_node.get_value().shape}"
        )
        max_logging.info(f"Sharing parameter {source_path} with {dest_path}")
        dest_parts = dest_path.split("/")
        dest_parent = _nav_to_attr(student_model, dest_parts[:-1])
        dest_attr = dest_parts[-1]
        if hasattr(dest_parent, dest_attr):
          setattr(dest_parent, dest_attr, source_node)
        else:
          dest_parent[dest_attr] = source_node

    if not matched_any:
      raise ValueError(f"Student parameter sharing: No paths matched the source pattern: {source_pattern}")

  # --- teacher to student weights copying  ---
  for teacher_pattern, student_pattern in teacher_weights_copy_map.items():
    matched_any = False
    for teacher_path, teacher_node in teacher_graph.items():
      match = re.fullmatch(teacher_pattern, teacher_path)
      if match:
        matched_any = True
        student_path = teacher_path if teacher_pattern == student_pattern else match.expand(student_pattern)
        student_path_parts = student_path.split("/")
        student_node = _nav_to_attr(student_model, student_path_parts)  # student_graph.get(student_path)

        assert (
            student_node is not None
        ), f"Could not find student model parameter at path: {student_path} for teacher parameter {teacher_path}"
        assert student_node.get_value().shape == teacher_node.get_value().shape, (
            f"Shape mismatch for {teacher_path}. Teacher: {teacher_node.get_value().shape},"
            f" Student: {student_node.get_value().shape}"
        )
        student_node.set_value(teacher_node.get_value())
        max_logging.info(f"Inserted teacher weight parameter {teacher_path} to the student at {student_path}")

    if not matched_any:
      raise ValueError(f"Teacher weight injection: No paths matched the source pattern: {teacher_pattern}")

  max_logging.info("Teacher weight injection complete.")
