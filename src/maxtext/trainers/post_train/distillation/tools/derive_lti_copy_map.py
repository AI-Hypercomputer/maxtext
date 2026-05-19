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

"""Derive distill_weights_copy_map for an LTI distillation run.

Loads the same student/teacher abstract configs that train_distill.py builds,
walks both graphs with nnx.graph.iter_graph, and emits a copy_map YAML snippet
listing every teacher path whose shape exactly matches the student path. Uses
nnx.eval_shape so no weights are materialized.

Usage:
  python -m maxtext.trainers.post_train.distillation.tools.derive_lti_copy_map <distillation_config.yml>
"""

import sys
from collections import defaultdict

from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.configs import pyconfig
from maxtext.utils import model_creation_utils


def _abstract_state_paths(config):
  """Return {path -> shape} for every parameter in an abstract model."""
  _, abs_model = model_creation_utils.create_nnx_abstract_model(config)
  paths = {}
  for path, node in nnx.graph.iter_graph(abs_model):
    if not isinstance(node, nnx.Variable):
      continue
    try:
      val = node.value
    except Exception:  # pylint: disable=broad-exception-caught
      continue
    shape = getattr(val, "shape", None)
    if shape is None:
      continue
    paths["/".join(map(str, path))] = tuple(shape)
  return paths


def main(argv):
  if len(argv) != 2:
    sys.stderr.write(f"usage: {argv[0]} <distillation_config.yml>\n")
    sys.exit(2)

  config_path = argv[1]

  global_config = pyconfig.initialize([argv[0], config_path])
  student_overrides = dict(global_config.student_overrides or {})
  teacher_overrides = dict(global_config.teacher_overrides or {})

  # Teacher built from a sanitized argv (no CLI flags) -- mirrors train_distill.py:898.
  teacher_argv = [argv[0], config_path]
  with nn_partitioning.axis_rules(global_config.logical_axis_rules):
    student_config = pyconfig.initialize([argv[0], config_path], **student_overrides)
    teacher_config = pyconfig.initialize(teacher_argv, **teacher_overrides)

  print("Building abstract student model...", file=sys.stderr)
  student_paths = _abstract_state_paths(student_config)
  print(f"  student params: {len(student_paths)}", file=sys.stderr)

  print("Building abstract teacher model...", file=sys.stderr)
  teacher_paths = _abstract_state_paths(teacher_config)
  print(f"  teacher params: {len(teacher_paths)}", file=sys.stderr)

  matches = []  # (path, shape)
  missing_in_student = []
  shape_mismatch = []  # (path, teacher_shape, student_shape)
  for path, t_shape in teacher_paths.items():
    if path not in student_paths:
      missing_in_student.append(path)
      continue
    s_shape = student_paths[path]
    if s_shape == t_shape:
      matches.append((path, t_shape))
    else:
      shape_mismatch.append((path, t_shape, s_shape))

  print(f"\nMatching paths (will be copied): {len(matches)}", file=sys.stderr)
  print(f"Shape-mismatch (skipped):        {len(shape_mismatch)}", file=sys.stderr)
  print(f"Only in teacher (skipped):       {len(missing_in_student)}", file=sys.stderr)

  if shape_mismatch:
    print("\n--- Skipped due to shape mismatch (expected for LTI: attn projections, q/k_norm) ---", file=sys.stderr)
    for p, t, s in shape_mismatch[:20]:
      print(f"  {p}  teacher={t}  student={s}", file=sys.stderr)
    if len(shape_mismatch) > 20:
      print(f"  ... and {len(shape_mismatch) - 20} more", file=sys.stderr)

  if missing_in_student:
    print("\n--- Only in teacher (skipped: no student counterpart) ---", file=sys.stderr)
    for p in missing_in_student[:20]:
      print(f"  {p}", file=sys.stderr)
    if len(missing_in_student) > 20:
      print(f"  ... and {len(missing_in_student) - 20} more", file=sys.stderr)

  # Group matches by parent module so the YAML stays human-readable while still
  # mapping 1:1 (we escape regex metachars and use exact-match patterns).
  print("\n# --- Paste into the YAML under `distill_weights_copy_map:` ---")
  print("distill_weights_copy_map:")
  groups = defaultdict(list)
  for path, _ in sorted(matches):
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    groups[parent].append(path)
  for parent, paths in groups.items():
    if parent:
      print(f"  # {parent}")
    for p in paths:
      # exact-match regex; copy teacher->student same path
      escaped = p.replace(".", r"\.").replace("[", r"\[").replace("]", r"\]")
      print(f'  "{escaped}": "{p}"')


if __name__ == "__main__":
  main(sys.argv)
