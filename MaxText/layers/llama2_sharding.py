from typing import Any

from jax.sharding import PartitionSpec

from MaxText.sharding import MeshSharding, TensorType as TT, assert_matches_logical_axis_rules


class Llama2TensorShardingTraining(MeshSharding):

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    tensor = kwargs["t"]
    tensor_type = kwargs.get("tensor_type", TT.Weight)
    # FIXME: it's quite likely we have inverted the necessary conditional in several places
    ep_as_context = kwargs.get("ep_ctx", False)

    mesh_axes = []
    match tensor, tensor_type:
      case "token_embedder" | "logits_dense", TT.Weight:
        mesh_axes = (
          ('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'),
          ('fsdp', 'fsdp_transpose', 'sequence', 'context', 'expert')
        )
        if tensor == "logits_dense":
          mesh_axes = tuple(mesh_axes[1], mesh_axes[2])
      case "embed_output", TT.Activation:
        mesh_axes = (
          ('data', 'stage', 'fsdp', 'fsdp_transpose', 'expert'),
          ("tensor_sequence", "context", "sequence"),
          ("tensor", "tensor_transpose"),
        )
      case ("pre_self_attention_layer_norm" | "post_self_attention_layer_norm" | "decoder_norm" | "rms_norm"), TT.Weight:
        mesh_axes = (
          ('tensor', 'tensor_transpose', 'tensor_sequence'),
        )
      case ("inputs_q", "inputs_kv"), TT.Activation:
        mesh_axes = (
          ('data', 'fsdp', 'fsdp_transpose'),
          # TODO: there are multiple matches for this axis in the current axis_rules. figure out which is correct
          #       (see also bigger note on axis rules, below)
          ('sequence', 'context', 'expert'),
          ("tensor", "tensor_transpose")
        )
      case ("query" | "key" | "value"), TT.Activation:
        mesh_axes = (
          ('data', 'fsdp', 'fsdp_transpose'),
          ('sequence', 'context') if ep_as_context else ('sequence', 'context', 'expert'),  # TODO: as above
          ('tensor', 'tensor_transpose', 'sequence','tensor_sequence')
          ('tensor', 'tensor_transpose', 'tensor_sequence')
        )
      case "out", TT.Activation:
        mesh_axes = (
          ("data", "fsdp", "fsdp_transpose"),
          ('sequence', 'context', 'expert'),
          ('tensor', 'tensor_transpose', 'sequence','tensor_sequence','autoregressive'),
          ('tensor', 'tensor_transpose', 'tensor_sequence')
        )
      case ("query" | "kv"), TT.Weight:
        mesh_axes = (
          ("fsdp", "fsdp_transpose", "sequence", "context", "expert"),
          ('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'),
          (None)
        )
      case "qkv", TT.Weight:
        mesh_axes = (
          ("fsdp", "fsdp_transpose", "sequence", "context", "expert"),
          (None),
          ('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'),
        )
      case "out", TT.Weight:
        mesh_axes = (
          ('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'),
          (None),
          ("fsdp", "fsdp_transpose", "sequence", "context", "expert")
        )
      case "mlp_pre_norm", TT.Weight:
        mesh_axes = ("tensor", "tensor_transpose", "tensor_sequence")
      case ("inputs" | "lnx", "attn_lnx" | "mlp_lnx" | "mlp_pre_norm" | "layer_output" | "hidden"), TT.Activation:
        mesh_axes = (
            ("data", "fsdp", "fsdp_transpose", "expert"),
            ("tensor_sequence", "context", "sequence"),
            ("tensor", "tensor_transpose"),
        )
      case "mlp_wi_fused", TT.Weight:
        mesh_axes = (
            ("fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context", "expert"),
            (None),
            ("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"),
        )
      case ("mlp_wi_unfused" | "mlp_wo"), TT.Weight:
        mesh_axes = (
            ("fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context", "expert"),
            ("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"),
        )
      case "mlp_wo", TT.Weight:
        mesh_axes = (
            ("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"),
            ("fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context", "expert"),
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
# TODO: when creating a clean version take a param for expert abuse
#
class Llama2AxisShardingTraining(MeshSharding):

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_type = kwargs.get("tensor_type", TT.Weight)

    mesh_axes = []
    for axis in axes:
      match axis, tensor_type:
        case "activation_batch", TT.Activation:
          mesh_axes.append(("data", "fsdp", "fsdp_transpose", "expert"))
        case "activation_batch_no_exp", TT.Activation:
          mesh_axes.append(('data', 'fsdp', 'fsdp_transpose'))
        case "activation_embed_and_logits_batch", TT.Activation:
          mesh_axes.append(('data', 'stage', 'fsdp', 'fsdp_transpose', 'expert'))
        case "activation_embed", TT.Activation:
          mesh_axes.append(("tensor", "tensor_transpose"))
        case "embed", TT.Weight:
          mesh_axes.append(("fsdp", "fsdp_transpose", "sequence", "context", "expert"))
        case "vocab", TT.Weight:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'))
        case "norm", TT.Weight:
          mesh_axes.append(("tensor", "tensor_transpose", "tensor_sequence"))
        case ("activation_length" | "activation_norm_length"), TT.Activation:
          mesh_axes.append(('sequence', 'context', 'expert'))  # but has multiple rules
        case "activation_length_no_exp", TT.Activation:
          mesh_axes.append(('sequence', 'context'))
        case "activation_kv", TT.Activation:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence'))
        case "activation_kv_batch", TT.Activation:
          mesh_axes.append(('data', 'fsdp', 'fsdp_transpose', 'expert'))
        case "activation_kv_batch_no_exp", TT.Activation:
          mesh_axes.append(('data', 'fsdp', 'fsdp_transpose')),
        case "activation_kv_heads", TT.Activation:
          mesh_axes.append(('tensor', 'tensor_transpose', 'sequence','tensor_sequence'))
        case "activation_kv_head_dim", TT.Activation:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence'))
        case "activation_heads", TT.Activation:
          mesh_axes.append(('tensor', 'tensor_transpose', 'sequence','tensor_sequence','autoregressive'))
        case  ("heads" | "q_heads" | "kv_heads"), TT.Weight:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'))
        case ("kv", "kv_head_dim", "qkv", "num_activations"), TT.Weight:
          mesh_axes.append((None))  # TODO: if this blows up it should have been ()
        case "mlp", TT.Weight:
          mesh_axes.append(("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"))
        case _, _:
          assert False, "Unexpected logical axis name for sharding"

      assert_matches_logical_axis_rules(mesh_axes[-1], axis)

    return PartitionSpec(*mesh_axes)


# NOTE: this is a potential target state of rules but we do not yet use it, as it's not
#       backwards-compatible with current logical_axis_rules (since we've changed axis names)
#       we will need to check cases where
#
class Llama2AxisShardingTrainingV2(MeshSharding):
  def __init__(self, config):
    self.config = config

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_name = kwargs["t"]
    tensor_type = kwargs.get("tensor_type", TT.Weight)
    tensor_transpose, fsdp_transpose = self.config.tensor_transpose, self.config.fsdp_transpose

    mesh_axes = []
    for axis in axes:
      match axis, tensor_type:
        case "batch", TT.Activation:
          mesh_axes.append(('data', 'fsdp', 'fsdp_transpose'))
        case "embed_and_logits_batch", TT.Activation:
          mesh_axes.append(('data', 'stage', 'fsdp', 'fsdp_transpose', 'expert'))
        case "embed", TT.Activation:
          mesh_axes.append(("tensor", "tensor_transpose"))
        case "embed", TT.Weight:
          axis_mappings = ["fsdp", "sequence", "context", "expert"]
          if tensor_name in ("mlp_wi_fused", "mlp_wi_unfused", "mlp_wo") and tensor_transpose:
            axis_mappings.append("tensor")
          mesh_axes.append(tuple(axis_mappings))
        case "mlp", TT.Weight:
          axis_mappings = ["tensor", "tensor_sequence", "autoregressive"]
          if tensor_name in ("mlp_wi_fused", "mlp_wi_unfused", "mlp_wo") and fsdp_transpose:
            axis_mappings.append("fsdp")
          mesh_axes.append(tuple(axis_mappings))
        case "vocab", TT.Weight:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'))
        case "norm", TT.Weight:
          mesh_axes.append(("tensor", "tensor_transpose", "tensor_sequence"))
        case ("length" | "norm_length"), TT.Activation:
          mesh_axes.append(('sequence', 'context'))
        case "kv", TT.Activation:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence'))
        case "kv_batch", TT.Activation:
          mesh_axes.append(('data', 'fsdp', 'fsdp_transpose')),
        case "kv_heads", TT.Activation:
          mesh_axes.append(('tensor', 'tensor_transpose', 'sequence','tensor_sequence'))
        case "kv_head_dim", TT.Activation:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence'))
        case "heads", TT.Activation:
          mesh_axes.append(('tensor', 'tensor_transpose', 'sequence','tensor_sequence','autoregressive'))
        case  ("heads" | "q_heads" | "kv_heads"), TT.Weight:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'))
        case ("kv", "kv_head_dim", "qkv", "num_activations"), TT.Weight:
          mesh_axes.append((None))
        case _, _, _:
          assert False, "Unexpected logical axis name for sharding"

    return PartitionSpec(*mesh_axes)
