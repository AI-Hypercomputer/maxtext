# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure NNX model parameter replacement and interpolation utilities for DiLoCo."""

from typing import Any

from flax import nnx
import jax
from maxtext.common.train_state_nnx import TrainStateNNX
from maxtext.trainers.diloco.utils.fragmenter import FragmentedTreeManipulator


def replace_nnx_model_params(s: Any, new_params: Any) -> Any:
  """Replaces model parameters in an NNX TrainState or dictionary structure."""
  s_model = s["model"] if hasattr(s, "keys") else s.model
  s_opt = s["optimizer"] if hasattr(s, "keys") else s.optimizer

  graphdef, _, non_param_state = nnx.split(s_model, nnx.Param, ...)
  new_model = nnx.merge(graphdef, new_params, non_param_state)

  if isinstance(s_model, nnx.State):
    new_model = nnx.state(new_model)
  elif isinstance(s_model, dict):
    new_model = nnx.to_pure_dict(new_model)

  if hasattr(s, "keys"):
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(s)
    new_model_iter = iter(jax.tree_util.tree_leaves(new_model))

    def _is_model_leaf(path):
      if not path:
        return False
      k = path[0]
      return getattr(k, "key", None) == "model" or getattr(k, "name", None) == "model"

    new_leaves = [next(new_model_iter) if _is_model_leaf(p) else leaf for p, leaf in leaves_with_paths]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)
  else:
    return TrainStateNNX(new_model, s_opt)


def replace_nnx_model_params_frag(
    s: Any,
    manipulator: FragmentedTreeManipulator,
    frag_idx: int,
    outer_frag_replica: dict[str, Any],
    alpha: float = 0.0,
) -> Any:
  """Replaces a single parameter fragment in an NNX TrainState with optional alpha interpolation."""
  s_model = s["model"] if hasattr(s, "keys") else s.model
  graphdef, full_params, non_param_state = nnx.split(s_model, nnx.Param, ...)
  full_params_dict = full_params.to_pure_dict()
  if alpha > 0.0:
    inner_frag = manipulator.get_flat_fragment(full_params_dict, frag_idx, has_replica_dim=False)
    merged_frag = jax.tree.map(lambda i, o: alpha * i + (1 - alpha) * o, inner_frag, outer_frag_replica)
  else:
    merged_frag = outer_frag_replica

  new_full_params = manipulator.apply_flat_fragment(full_params_dict, frag_idx, merged_frag, has_replica_dim=False)
  new_model = nnx.merge(graphdef, new_full_params, non_param_state)

  if isinstance(s_model, nnx.State):
    new_model = nnx.state(new_model)
  elif isinstance(s_model, dict):
    new_model = nnx.to_pure_dict(new_model)

  if hasattr(s, "keys"):
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(s)
    new_model_iter = iter(jax.tree_util.tree_leaves(new_model))

    def _is_model_leaf(path):
      if not path:
        return False
      k = path[0]
      return getattr(k, "key", None) == "model" or getattr(k, "name", None) == "model"

    new_leaves = [next(new_model_iter) if _is_model_leaf(p) else leaf for p, leaf in leaves_with_paths]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)
  else:
    s_opt = s["optimizer"] if hasattr(s, "keys") else s.optimizer
    return TrainStateNNX(new_model, s_opt)
