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

"""Unit tests for the pipeline BSW partition-spec helpers in maxtext.utils.pipeline_utils."""

import unittest

from jax.sharding import PartitionSpec as P

from maxtext.utils import pipeline_utils

# The axes from_repeat_weights_to_bsw and get_current_weights_from_bsw remove ("expert" is added for non-MoE weights).
AXES_TO_REMOVE = ["fsdp", "fsdp_transpose", "context"]
# cp-as-ep: 'context' co-sharded with 'expert' on the routed-expert dim (exp -> ['context', 'expert']).
CP_AS_EP = ("context", "expert")


def _moe_layer_specs():
  """Full physical specs ([circular_repeats, stage, ...]) of one DeepSeek MoE decoder layer under cp-as-ep."""
  return {
      "DeepSeekMoeBlock_0": {
          "MoeBlock_0": {
              "wi_0": P(None, "stage", CP_AS_EP, "fsdp", "tensor"),
              "wo": P(None, "stage", CP_AS_EP, "tensor", ("fsdp", "fsdp_transpose")),
          },
          "shared_experts": {"wi_0": {"kernel": P(None, "stage", ("fsdp", "context", "expert"), "tensor")}},
      },
      "self_attention": {"out": {"kernel": P(None, "stage", "tensor", None, "expert")}},
  }


class RemoveGatheredMeshAxesTest(unittest.TestCase):

  def test_moe_block_keeps_context_co_sharded_with_expert(self):
    self.assertEqual(
        pipeline_utils.remove_gathered_mesh_axes(P(CP_AS_EP, ("fsdp",), "tensor"), True, list(AXES_TO_REMOVE)),
        P(CP_AS_EP, None, "tensor"),
    )

  def test_moe_block_removes_context_not_co_sharded_with_expert(self):
    self.assertEqual(
        pipeline_utils.remove_gathered_mesh_axes(P("context", ("fsdp", "context"), "expert"), True, list(AXES_TO_REMOVE)),
        P(None, None, "expert"),
    )

  def test_non_moe_removes_expert_and_context(self):
    self.assertEqual(
        pipeline_utils.remove_gathered_mesh_axes(
            P(("fsdp", "context", "expert"), "tensor", "expert"), False, list(AXES_TO_REMOVE)
        ),
        P(None, "tensor", None),
    )

  def test_does_not_mutate_axes_to_remove(self):
    axes_to_remove = list(AXES_TO_REMOVE)
    pipeline_utils.remove_gathered_mesh_axes(P("expert"), False, axes_to_remove)
    self.assertEqual(axes_to_remove, AXES_TO_REMOVE)


class DeriveStageWeightPartitionSpecsTest(unittest.TestCase):

  def test_moe_weights_after_non_moe_weights_keep_expert_sharding(self):
    # A sequential multi-layer stage visits layers_0's non-MoE weights before layers_1's routed experts: the one
    # axes_to_remove list shared by every leaf must not pick up "expert" along the way.
    expected_layer = {
        "DeepSeekMoeBlock_0": {
            "MoeBlock_0": {
                "wi_0": P("stage", CP_AS_EP, None, "tensor"),
                "wo": P("stage", CP_AS_EP, "tensor", None),
            },
            "shared_experts": {"wi_0": {"kernel": P("stage", None, "tensor")}},
        },
        "self_attention": {"out": {"kernel": P("stage", "tensor", None, None)}},
    }
    self.assertEqual(
        pipeline_utils.derive_stage_weight_partition_specs(
            {"layers_0": _moe_layer_specs(), "layers_1": _moe_layer_specs()}, list(AXES_TO_REMOVE)
        ),
        {"layers_0": expected_layer, "layers_1": expected_layer},
    )


if __name__ == "__main__":
  unittest.main()
