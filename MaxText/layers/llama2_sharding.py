from typing import Any

from jax.sharding import PartitionSpec, Mesh

from MaxText.sharding import MeshSharding, TensorType as TT
# from MaxText.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, MODEL_MODE_AUTOREGRESSIVE


class Llama2TensorShardingTraining(MeshSharding):

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    tensor = kwargs["t"]
    tensor_type = kwargs["tt"]
    # FIXME: it's quite likely we have inverted the necessary conditional in several places
    ep_as_context = kwargs.get("ep_ctx", False)

    mesh_axes = []
    match tensor, tensor_type:
      case "embedding", TT.WT:
        mesh_axes = (
          ('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'),
          ('fsdp', 'fsdp_transpose', 'sequence', 'context', 'expert')
        )
      case "embed_output", TT.ACT:
        mesh_axes = (
          ('data', 'stage', 'fsdp', 'fsdp_transpose', 'expert'),
          ("tensor_sequence", "context", "sequence"),
          ("tensor", "tensor_transpose"),
        )
      case "rms_norm", TT.WT:
        mesh_axes = (
          ('tensor', 'tensor_transpose', 'tensor_sequence'),
        )
      case ("inputs_q", "inputs_kv"), TT.ACT:
        mesh_axes = (
          ('data', 'fsdp', 'fsdp_transpose', 'expert') if ep_as_context else ('data', 'fsdp', 'fsdp_transpose'),
          # TODO: there are multiple matches for this axis in the current axis_rules. figure out which is correct
          #       (see also bigger note on axis rules, below)
          ('sequence', 'context') if ep_as_context else ('sequence', 'context', 'expert'),
          ("tensor", "tensor_transpose")
        )
      case ("query" | "key" | "value"), TT.ACT:
        mesh_axes = (
          ('data', 'fsdp', 'fsdp_transpose', 'expert') if ep_as_context else ('data', 'fsdp', 'fsdp_transpose'),
          ('sequence', 'context') if ep_as_context else ('sequence', 'context', 'expert'),  # TODO: as above
          ('tensor', 'tensor_transpose', 'sequence','tensor_sequence')
          ('tensor', 'tensor_transpose', 'tensor_sequence')
        )
      case "out", TT.ACT:
        mesh_axes = (("data", "fsdp", "fsdp_transpose", "expert") if ep_as_context else ("data", "fsdp", "fsdp_transpose"),
          ('sequence', 'context') if ep_as_context else ('sequence', 'context', 'expert'),
          ('tensor', 'tensor_transpose', 'sequence','tensor_sequence','autoregressive'),
          ('tensor', 'tensor_transpose', 'tensor_sequence')
        )
      case ("query" | "kv"), TT.WT:
        mesh_axes = (
          ("fsdp", "fsdp_transpose", "sequence", "context", "expert"),
          ('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'),
          (None)
        )
      case "qkv", TT.WT:
        mesh_axes = (
          ("fsdp", "fsdp_transpose", "sequence", "context", "expert"),
          (None),
          ('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'),
        )
      case "out", TT.WT:
        mesh_axes = (
          ('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'),
          (None),
          ("fsdp", "fsdp_transpose", "sequence", "context", "expert")
        )
      case "mlp_pre_norm", TT.WT:
        mesh_axes = ("tensor", "tensor_transpose", "tensor_sequence")
      case ("inputs" | "attn_lnx" | "mlp_lnx" | "mlp_pre_norm" | "layer_output" | "hidden"), TT.ACT:
        mesh_axes = (
            ("data", "fsdp", "fsdp_transpose", "expert"),
            ("tensor_sequence", "context", "sequence"),
            ("tensor", "tensor_transpose"),
        )
      case "mlp_wi_fused", TT.WT:
        mesh_axes = (
            ("fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context", "expert"),
            None,
            ("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"),
        )
      case ("mlp_wi_unfused" | "mlp_wo"), TT.WT:
        mesh_axes = (
            ("fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context", "expert"),
            ("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"),
        )
      case _, _:
        assert False, "Unexpected tensor name for sharding"

    return PartitionSpec(*mesh_axes)


# NOTE: in current logical_axis_rules, embed has multiple mappings and the first is:
# ("fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context", "expert") which is not what we return here.
# The reason is that in Llama the embed axis (only) shares a tensor with the vocab axis and -- as below -- that uses
# tensor_transpose, which means the mapping we return is not the first mapping from the current axis rules (above) but
# is the mapping that is chosen. We are therefore able to replicate this behavior easily here but we would have an issue
# if the embed axis also appeared on another tensor where e.g. tensor_transpose was **not** already taken
# in which case we could only choose the correct mapping by also including the name of the tensor, or by
# minicking the behavior of the logical_axis_rules implementation (e.g. storing already matches axes in a set)
#
class Llama2AxisShardingTraining(MeshSharding):

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_type = kwargs["tt"]

    mesh_axes = []
    for axis in axes:
      match axis, tensor_type:
        case "activation_batch", TT.ACT:
          mesh_axes.append(("data", "fsdp", "fsdp_transpose", "expert"))
        case "activation_batch_no_exp", TT.ACT:
          mesh_axes.append(('data', 'fsdp', 'fsdp_transpose'))
        case "activation_embed_and_logits_batch", TT.ACT:
          mesh_axes.append(('data', 'stage', 'fsdp', 'fsdp_transpose', 'expert'))
        case "activation_embed", TT.ACT:
          mesh_axes.append(("tensor", "tensor_transpose"))
        case "embed", TT.WT:
          mesh_axes.append(("fsdp", "fsdp_transpose", "sequence", "context", "expert"))
        case "vocab", TT.WT:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'))
        case "norm", TT.WT:
          mesh_axes.append(("tensor", "tensor_transpose", "tensor_sequence"))
        case ("activation_length" | "activation_norm_length"), TT.ACT:
          mesh_axes.append(('sequence', 'context', 'expert'))  # but has multiple rules
        case "activation_length_no_exp", TT.ACT:
          mesh_axes.append(('sequence', 'context'))
        case "activation_kv", TT.ACT:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence'))
        case "activation_kv_batch", TT.ACT:
          mesh_axes.append(('data', 'fsdp', 'fsdp_transpose', 'expert'))
        case "activation_kv_batch_no_exp", TT.ACT:
          mesh_axes.append(('data', 'fsdp', 'fsdp_transpose')),
        case "activation_kv_heads", TT.ACT:
          mesh_axes.append(('tensor', 'tensor_transpose', 'sequence','tensor_sequence'))
        case "activation_kv_head_dim", TT.ACT:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence'))
        case "activation_heads", TT.ACT:
          mesh_axes.append(('tensor', 'tensor_transpose', 'sequence','tensor_sequence','autoregressive'))
        case  ("heads" | "q_heads" | "kv_heads"), TT.WT:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'))
        case ("kv", "kv_head_dim", "qkv", "num_activations"), TT.WT:
          mesh_axes.append((None))  # TODO: if this blows up it should have been ()
        case "mlp", TT.WT:
          mesh_axes.append(("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"))
        case _, _:
          assert False, "Unexpected logical axis name for sharding"

    return PartitionSpec(*mesh_axes)


# NOTE: this is a potential target state of rules (modulo todos) but we do not yet use it, as it's not
#       backwards-compatible with current logical_axis_rules (since we've changed axis names)
#
class Llama2AxisShardingTrainingV2(MeshSharding):

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_type = kwargs["tt"]
    ep_as_context = kwargs.get("ep_ctx", False)

    mesh_axes = []
    for axis in axes:
      match axis, tensor_type, ep_as_context:
        case "batch", TT.ACT, True:
          mesh_axes.append(("data", "fsdp", "fsdp_transpose", "expert"))
        case "batch", TT.ACT, False:
          mesh_axes.append(('data', 'fsdp', 'fsdp_transpose'))
        case "embed_and_logits_batch", TT.ACT:
          mesh_axes.append(('data', 'stage', 'fsdp', 'fsdp_transpose', 'expert'))
        case "embed", TT.ACT:
          mesh_axes.append(("tensor", "tensor_transpose"))
        case "embed", TT.WT, _:
          mesh_axes.append(("fsdp", "fsdp_transpose", "sequence", "context", "expert"))
        case "vocab", TT.WT, _:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'))
        case "norm", TT.WT, _:
          mesh_axes.append(("tensor", "tensor_transpose", "tensor_sequence"))
        case ("length", "norm_length"), TT.ACT, True:
          mesh_axes.append(('sequence', 'context', 'expert'))  # TODO: but has multiple rules
        case "length", TT.ACT, False:
          mesh_axes.append(('sequence', 'context'))
        case "kv", TT.ACT, _:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence'))
        case "kv_batch", TT.ACT, True:
          mesh_axes.append(('data', 'fsdp', 'fsdp_transpose', 'expert'))
        case "kv_batch", TT.ACT, False:
          mesh_axes.append(('data', 'fsdp', 'fsdp_transpose')),
        case "kv_heads", TT.ACT, _:
          mesh_axes.append(('tensor', 'tensor_transpose', 'sequence','tensor_sequence'))
        case "kv_head_dim", TT.ACT, _:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence'))
        case "heads", TT.ACT, _:
          mesh_axes.append(('tensor', 'tensor_transpose', 'sequence','tensor_sequence','autoregressive'))
        case  ("heads" | "q_heads" | "kv_heads"), TT.WT, _:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'))
        case ("kv", "kv_head_dim", "qkv", "num_activations"), TT.WT, _:
          mesh_axes.append((None))  # TODO: if this blows up it should have been ()
        case "mlp", TT.WT, _:
          mesh_axes.append(("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"))
        case _, _, _:
          assert False, "Unexpected logical axis name for sharding"

    return PartitionSpec(*mesh_axes)


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
