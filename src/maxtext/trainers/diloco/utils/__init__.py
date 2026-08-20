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

"""DiLoCo utilities package."""

from maxtext.trainers.diloco.utils.fragmenter import (
    FragmentedTreeManipulator,
    get_streaming_schedule,
)
from maxtext.trainers.diloco.utils.nnx_state_utils import (
    replace_nnx_model_params,
    replace_nnx_model_params_frag,
)
from maxtext.trainers.diloco.utils.spmd_diloco_checkpointing import (
    from_diloco_checkpoint_dict,
    is_diloco_checkpoint,
    restore_diloco_checkpoint,
    to_diloco_checkpoint_dict,
)
from maxtext.utils.diloco_sharding import (
    add_diloco_to_sharding,
    extract_per_island_metrics,
    extract_replica_0,
    reshape_first_axis_with_diloco,
)
from maxtext.trainers.diloco.utils.spmd_diloco_sync import (
    apply_fragment_to_inner_state,
    setup_diloco_initial_state,
    synchronize_fragment_state,
    synchronize_full_state,
)

__all__ = [
    # Common PyTree & Schedule Utilities
    "FragmentedTreeManipulator",
    "get_streaming_schedule",
    # Common NNX State Utilities
    "replace_nnx_model_params",
    "replace_nnx_model_params_frag",
    # SPMD Sharding & Metric Extraction
    "add_diloco_to_sharding",
    "reshape_first_axis_with_diloco",
    "extract_replica_0",
    "extract_per_island_metrics",
    # SPMD Synchronization
    "synchronize_full_state",
    "synchronize_fragment_state",
    "apply_fragment_to_inner_state",
    "setup_diloco_initial_state",
    # SPMD Checkpointing
    "from_diloco_checkpoint_dict",
    "is_diloco_checkpoint",
    "restore_diloco_checkpoint",
    "to_diloco_checkpoint_dict",
]
