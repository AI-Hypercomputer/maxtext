from typing import Any

from jax.sharding import PartitionSpec, Mesh

from MaxText.sharding import MeshSharding, ACT, WT
# from MaxText.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, MODEL_MODE_AUTOREGRESSIVE


class Llama2TensorShardingTraining(MeshSharding):

  def __init__(self, mesh: Mesh):
    super().__init__(mesh)

  def get(self, *args: Any, **kwargs) -> tuple[str | None, ...] | None:
    tensor = kwargs["t"]
    tensor_type = kwargs["tt"]

    mesh_axes = []
    match tensor, tensor_type:
      case "mlp_pre_norm", WT:
        mesh_axes = ("tensor", "tensor_transpose", "tensor_sequence")
      case ("inputs" | "post_norm" | "post_attn" | "post_mlp" | "post_dropout" | "mlp_pre_norm" | "mlp_dropout"), ACT:
        mesh_axes = (
            ("data", "fsdp", "fsdp_transpose", "expert"),
            ("tensor_sequence", "context", "sequence"),
            ("tensor", "tensor_transpose"),
        )
      case "mlp_wi_fused", WT:
        mesh_axes = (
            ("fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context", "expert"),
            None,
            ("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"),
        )
      case ("mlp_wi_unfused" | "mlp_wo"), WT:
        mesh_axes = (
            ("fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context", "expert"),
            ("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"),
        )

    return mesh_axes


class Llama2AxisShardingTraining(MeshSharding):

  def __init__(self, mesh: Mesh):
    super().__init__(mesh)

  def get(self, *args: Any, **kwargs) -> tuple[str | None, ...] | None:
    axes = kwargs["a"]
    tensor_type = kwargs["tt"]

    mesh_axes = []
    for axis in axes:
      match axis, tensor_type:
        case "batch", _:
          mesh_axes.append(("data", "fsdp", "fsdp_transpose", "expert"))
        case "length", _:
          mesh_axes.append(("tensor_sequence", "context", "sequence"))
        case "embed", ACT:
          mesh_axes.append(("tensor", "tensor_transpose"))
        case "embed", WT:
          mesh_axes.append(("fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context", "expert"))
        case "norm":
          mesh_axes.append(("tensor", "tensor_transpose", "tensor_sequence"))
        case "num_activations", _:
          mesh_axes.append(None)
        case "mlp":
          mesh_axes.append(("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"))

    return PartitionSpec(*mesh_axes)


# thought there was a difference but there's not
#
# if model_mode == MODEL_MODE_PREFILL:
#   activation_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
# else:
#   activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")
#
# class Llama2TensorShardingInference(MeshProtocol):
#   def __init__(self, model_mode: str):
#     self.model_mode = model_mode

#     def __call__(self, *args: Any, **kwargs) -> tuple[str | None, ...] | None:
#       tensor = kwargs['t']
#       mode = self.model_mode
#       assert mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE), "This sharding is for inference modes only."

#       match tensor, mode:
#         case 'inputs' | 'post_norm' | 'post_attention' | 'post_mlp' | 'post_dropout', MODEL_MODE_PREFILL:
#           return (['data', 'fsdp', 'fsdp_transpose', 'expert'],
#                    ['sequence', 'context'],
#                    ['tensor', 'tensor_transpose'])
#         case 'inputs' | 'post_norm' | 'post_attention' | 'post_mlp' | 'post_dropout', MODEL_MODE_AUTOREGRESSIVE:
#           return (['data', 'fsdp', 'fsdp_transpose', 'expert'],
#                   ['sequence', 'context', 'expert'],
#                   ['tensor', 'tensor_transpose'])


# class Llama2AxisShardingInference(MeshProtocol):
#   def __init__(self, model_mode: str):
#     self.model_mode = model_mode

#     def __call__(self, *args: Any, **kwargs) -> tuple[str | None, ...] | None:
#       axes = kwargs['a']
#       mode = self.model_mode
#       assert mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE), "This sharding is for inference modes only."

#       mesh_axes = []
#       for axis in axes:
#         match axis, mode:
#           case 'batch':
#             mesh_axes.append(['data', 'fsdp', 'fsdp_transpose', 'expert'])
#           case 'length':
#             mesh_axes.append(['tensor_sequence', 'context', 'sequence'])
#           case 'embed':
#             mesh_axes.append(['tensor', 'tensor_transpose'])

#       return tuple(mesh_axes)


"""
# TODO:
- waiting on cristian to explain nnx.Param taking sharding as a parameter
- why did we not do rms_norm?
- we could revisit splitting into two classes above and instead key on model_mode
- either way inference rules will need to take model_mode for prefill vs. decode
- also went through attentions to the point of paged attention which does have nnx.Cache so will want to address that
- and then go on from that point of attentions.py

WHAT HAVE WE DONE?
- Applied sharding constraints in llama2.py
- Passed through to attentions

DO WE UNDERSTAND WHAT'S HAPPENING?
- Linen only applies as metadata - can see it if print the object
- It gets applied in get_partition_spec call (or maybe get_param_spec in some cases unless he was wrong about that name)
- sharding= passed to NNX is a tuple of strings and so not a match for jax.lax.with_sharding_constraint

WHAT DO WE WANT TO SHOW?
1. which sharding class to use is driven by optional config...
2. but sensible default such that e.g. llama2 uses llama2tensor
3. llama2 can depend on a class (e.g. paged attn or attn in general) which hasn't been ported (in which case it uses old system)
4. a class that has been ported (e.g. rms_norm) can be used by a class that hasn't been ported
5. linen fallback can be used explicitly for a class that has been ported
"""
