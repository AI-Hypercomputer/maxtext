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

"""SPMD utilities for DiLoCo and Streaming DiLoCo."""

from maxtext.trainers.diloco.utils.spmd.checkpoint_utils import (
    from_diloco_checkpoint_dict,
    is_diloco_checkpoint,
    to_diloco_checkpoint_dict,
)
from maxtext.trainers.diloco.utils.spmd.fragment_utils import (
    FragmentedTreeManipulator,
    apply_fragment_to_inner_state,
    get_streaming_schedule,
    replace_nnx_model_params_frag,
    synchronize_fragment_state,
)
from maxtext.trainers.diloco.utils.spmd.state_utils import (
    add_diloco_to_sharding,
    extract_per_island_metrics,
    extract_replica_0,
    replace_nnx_model_params,
    reshape_first_axis_with_diloco,
    setup_diloco_initial_state,
    synchronize_full_state,
)

__all__ = [
    "FragmentedTreeManipulator",
    "apply_fragment_to_inner_state",
    "get_streaming_schedule",
    "replace_nnx_model_params_frag",
    "synchronize_fragment_state",
    "add_diloco_to_sharding",
    "extract_per_island_metrics",
    "extract_replica_0",
    "replace_nnx_model_params",
    "reshape_first_axis_with_diloco",
    "setup_diloco_initial_state",
    "synchronize_full_state",
    "from_diloco_checkpoint_dict",
    "is_diloco_checkpoint",
    "to_diloco_checkpoint_dict",
]
