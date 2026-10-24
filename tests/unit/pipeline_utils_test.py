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

"""Unit tests for maxtext.utils.pipeline_utils: the BSW partition-spec and explicit-sharding stage-mesh helpers."""

import unittest

import jax
from jax.sharding import AxisType, Mesh, PartitionSpec as P
import numpy as np

from maxtext.common.common_types import ShardMode
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


def _one_device_mesh(axis_types):
  return Mesh(np.asarray(jax.devices()[:1]).reshape((1, 1, 1)), ("data", "stage", "fsdp"), axis_types=axis_types)


class PipelineStageMeshTest(unittest.TestCase):
  """pipeline_stage_mesh and the PartitionSpec helpers the pipeline uses under shard_mode=explicit."""

  def test_only_stage_becomes_auto_under_explicit(self):
    mesh = _one_device_mesh((AxisType.Explicit,) * 3)
    stage_mesh = pipeline_utils.pipeline_stage_mesh(mesh, ShardMode.EXPLICIT)
    self.assertEqual(stage_mesh.axis_types, (AxisType.Explicit, AxisType.Auto, AxisType.Explicit))
    self.assertEqual(stage_mesh.axis_names, mesh.axis_names)
    self.assertTrue((stage_mesh.devices == mesh.devices).all())

  def test_unchanged_under_auto(self):
    mesh = _one_device_mesh((AxisType.Auto,) * 3)
    self.assertIs(pipeline_utils.pipeline_stage_mesh(mesh, ShardMode.AUTO), mesh)

  def test_remove_mesh_axis(self):
    self.assertIsNone(pipeline_utils.remove_mesh_axis(None, "stage"))
    self.assertIsNone(pipeline_utils.remove_mesh_axis("stage", "stage"))
    self.assertEqual(pipeline_utils.remove_mesh_axis("fsdp", "stage"), "fsdp")
    self.assertEqual(pipeline_utils.remove_mesh_axis(("stage", "fsdp"), "stage"), ("fsdp",))
    self.assertIsNone(pipeline_utils.remove_mesh_axis(("stage",), "stage"))

  def test_remove_mesh_axis_from_spec_keeps_reduced(self):
    spec = P("stage", ("stage", "fsdp"), None, reduced={"data"})
    self.assertEqual(pipeline_utils.remove_mesh_axis_from_spec(spec, "stage"), P(None, ("fsdp",), None, reduced={"data"}))


class TagWeightGradReducedTest(unittest.TestCase):
  """tag_weight_grad_reduced, which pipeline_unreduced_weight_grads applies to the gathered BSW stage weights."""

  def test_skips_axes_the_weight_is_sharded_on(self):
    token_axes = frozenset({"fsdp", "expert"})
    self.assertEqual(
        pipeline_utils.tag_weight_grad_reduced(P("stage", None, None), token_axes),
        P("stage", None, None, reduced={"fsdp", "expert"}),
    )
    # An expert weight is sharded over 'expert'; its gradient is only partial over the remaining token axes.
    self.assertEqual(
        pipeline_utils.tag_weight_grad_reduced(P("stage", "expert", None), token_axes),
        P("stage", "expert", None, reduced={"fsdp"}),
    )
    self.assertEqual(
        pipeline_utils.tag_weight_grad_reduced(P("stage", ("fsdp", "expert")), token_axes), P("stage", ("fsdp", "expert"))
    )
    self.assertIsNone(pipeline_utils.tag_weight_grad_reduced(None, token_axes))
    self.assertEqual(pipeline_utils.tag_weight_grad_reduced(P("stage"), frozenset()), P("stage"))


if __name__ == "__main__":
  unittest.main()
