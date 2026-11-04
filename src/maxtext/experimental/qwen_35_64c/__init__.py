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

"""Pure-JAX Qwen3.5-397B-A17B package with Dual-SparseCore 2-microbatch batch-split."""

from maxtext.experimental.qwen_35_64c.config import Qwen35Config, full_config, tiny_config
from maxtext.experimental.qwen_35_64c.mesh import create_mesh
from maxtext.experimental.qwen_35_64c.model import (
    build_remat_policy,
    create_optimizer,
    data_pspec,
    get_abstract_opt_state,
    init_opt_state,
    init_synthetic_batch,
    make_train_step,
    make_value_and_grad_fn,
)
from maxtext.experimental.qwen_35_64c.params import (
    AttnSublayerParams,
    GdnSublayerParams,
    ModelParams,
    MoeParams,
    SuperblockParams,
    get_abstract_params,
    get_param_pspecs,
    init_random_params,
)

__all__ = [
    "AttnSublayerParams",
    "GdnSublayerParams",
    "ModelParams",
    "MoeParams",
    "Qwen35Config",
    "SuperblockParams",
    "build_remat_policy",
    "create_mesh",
    "create_optimizer",
    "data_pspec",
    "full_config",
    "get_abstract_opt_state",
    "get_abstract_params",
    "get_param_pspecs",
    "init_opt_state",
    "init_random_params",
    "init_synthetic_batch",
    "make_train_step",
    "make_value_and_grad_fn",
    "tiny_config",
]
