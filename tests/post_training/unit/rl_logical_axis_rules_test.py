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

"""rl.yml needs a logical-axis rule for every axis of the RL actor's parameters (CPU-only).

rl.yml replaces base.yml's `logical_axis_rules` instead of extending it. MaxText places parameters
with its own resolver, which leaves an axis without a rule unsharded, so a missing rule goes unnoticed
until Tunix shards the trainer state: `PeftTrainer._shard_optimizer` derives the shardings of the
optimizer state and of the gradient accumulator from `nnx.get_partition_spec` under these rules. That
keeps any name it has no rule for, and `tunix.sft.sharding_utils.get_sharding` then replicates every
buffer whose PartitionSpec names an axis that is not on the mesh. With `local_layers` missing, that
replicated the float32 gradients of all of Qwen3.5-35B-A3B's nested-scanned linear-attention layers
(94 GiB) and ran the Pathways head out of host memory before the first train step.
"""

import sys
import unittest

import pytest
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.configs import pyconfig, types
from maxtext.utils import maxtext_utils, model_creation_utils
from tests.utils.test_helpers import get_test_config_path

pytestmark = [pytest.mark.post_training]

# A toy Qwen3.5: two cycles of three Gated DeltaNet layers and one full-attention layer. With
# scan_layers=True the linear-attention layers of each block are stacked in a nested
# `local_layers` scan inside the `layers` block scan (the layout Qwen3-Next and Gemma4 share).
_QWEN3_5_OVERRIDES = {
    "model_name": "qwen3.5-35b-a3b",
    "override_model_config": True,
    "scan_layers": True,
    "enable_checkpointing": False,
    "max_target_length": 8,
    "base_emb_dim": 64,
    "base_num_decoder_layers": 8,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "head_dim": 32,
    "base_mlp_dim": 128,
    "base_moe_mlp_dim": 32,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 32,
    "gdn_num_key_heads": 2,
    "gdn_num_value_heads": 4,
    "gdn_key_head_dim": 16,
    "gdn_value_head_dim": 16,
    "gdn_chunk_size": 4,
    # MRoPE sections must cover rotary_dim / 2, which is 4 here, not the shipped [11, 11, 10].
    "mrope_section": [2, 1, 1],
    "sparse_matmul": True,
    "megablox": False,
    "attention": "dot_product",
}


class RLLogicalAxisRulesTest(unittest.TestCase):
  """Every axis the RL actor's params are annotated with must resolve to mesh axes under rl.yml's rules."""

  @pytest.mark.cpu_only
  def test_scanned_qwen3_5_param_axes_resolve_to_mesh_axes(self):
    config = pyconfig.initialize_pydantic(
        [sys.argv[0], get_test_config_path("post_train/rl.yml")],
        config_class=types.RLConfig,
        run_name="rl_logical_axis_rules_test",
        tokenizer_path="Qwen/Qwen3.5-35B-A3B",
        **_QWEN3_5_OVERRIDES,
    )
    mesh = maxtext_utils.get_mesh_from_config(config)
    _, abstract_model = model_creation_utils.create_nnx_abstract_model(config, mesh)
    self.assertIn("local_layers", nnx.state(abstract_model, nnx.Param)["decoder"]["layers"])

    # Tunix's RLCluster runs the actor trainer under these rules, as `nn_partitioning.axis_rules`.
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      pspecs = nnx.get_partition_spec(nnx.state(abstract_model, nnx.Param))

    unresolved = {}
    for path, spec in nnx.to_flat_state(pspecs):
      spec = spec.get_value() if isinstance(spec, nnx.Variable) else spec
      for entry in spec or ():
        for name in entry if isinstance(entry, tuple) else (entry,):
          if name is not None and name not in mesh.axis_names:
            unresolved.setdefault(name, "/".join(map(str, path)))
    self.assertEqual(
        unresolved,
        {},
        "rl.yml has no logical_axis_rules entry for these axes (first parameter using each is shown); "
        "Tunix would replicate the optimizer state and gradients of every such parameter on all devices.",
    )


if __name__ == "__main__":
  unittest.main()
