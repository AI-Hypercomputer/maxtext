from typing import Any

from jax.sharding import PartitionSpec, Mesh

from MaxText.sharding import MeshSharding, TensorType as TT, assert_matches_logical_axis_rules


# NOTE: very different to llame rules due to many more activation constraints in moe.py
# TODO: rms_norm or possibly decoder_norm might not be used
#
class Qwen3TensorShardingTraining(MeshSharding):

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    tensor = kwargs["t"]
    # when called by MeshSharding.shard this will be set to Activation. otherwise we assume Weight
    tensor_type = kwargs.get("tensor_type", TT.Weight)

    # TODO: it's quite likely we have inverted the necessary conditional in several places
    ep_as_context = kwargs.get("ep_ctx", False)
    # TODO: the logic to determine these two bools would ideally live here and not in the caller (moe.py)
    batch_sharded_by_expert = kwargs.get("batch_sharded_by_expert", False)
    tensor_transpose = kwargs.get("tensor_transpose", False)

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
          ('data', 'fsdp', 'fsdp_transpose', 'expert') if ep_as_context else ('data', 'fsdp', 'fsdp_transpose'),
          # TODO: there are multiple matches for this axis in the current axis_rules. figure out which is correct
          #       (see also bigger note on axis rules, below)
          ('sequence', 'context') if ep_as_context else ('sequence', 'context', 'expert'),
          ("tensor", "tensor_transpose")
        )
      case ("query" | "key" | "value"), TT.Activation:
        mesh_axes = (
          ('data', 'fsdp', 'fsdp_transpose', 'expert') if ep_as_context else ('data', 'fsdp', 'fsdp_transpose'),
          ('sequence', 'context') if ep_as_context else ('sequence', 'context', 'expert'),  # TODO: as above
          ('tensor', 'tensor_transpose', 'sequence','tensor_sequence')
          ('tensor', 'tensor_transpose', 'tensor_sequence')
        )
      case "out", TT.Activation:
        mesh_axes = (("data", "fsdp", "fsdp_transpose", "expert") if ep_as_context else ("data", "fsdp", "fsdp_transpose"),
          ('sequence', 'context') if ep_as_context else ('sequence', 'context', 'expert'),
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
      case ("inputs" | "lnx", "attn_lnx" | "mlp_lnx" | "mlp_pre_norm" | "layer_output" | "hidden" | "mlp_output"), TT.Activation:
        mesh_axes = (
            ("data", "fsdp", "fsdp_transpose", "expert"),
            ("tensor_sequence", "context", "sequence"),
            ("tensor", "tensor_transpose"),
        )
      case "mlp_wi_fused", TT.Weight:
        mesh_axes = (
            ("fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context", "expert"),
            None,
            ("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"),
        )
      case ("mlp_wi_unfused" | "mlp_wo"), TT.Weight:
        mesh_axes = (
            ("fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context", "expert"),
            ("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"),
        )
      case "gate_logit", TT.Weight:
        mesh_axes = (
          ("fsdp", "fsdp_transpose", "sequence", "context", "expert"),
          (None)
        )
      case ("moe_wi_0" | "moe_wi_1"), TT.Weight:
        mesh_axes = (
          ("expert",),
          ('fsdp', 'fsdp_transpose', 'sequence', 'tensor_transpose', 'context'),
          ("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive")
        )
      case "moe_wo", TT.Weight:
        mesh_axes = (
          (("expert",),
          ("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"),
          ('fsdp', 'fsdp_transpose', 'sequence', 'tensor_transpose', 'context'))
        )
      case "expert_mask_fused", TT.Activation:
        mesh_axes = (
          ("data", "fsdp", "fsdp_transpose", "expert"),
          None,
          None,
          None,
        )
      case "expert_token_count", TT.Activation:
        mesh_axes = (
          ("data", "fsdp", "fsdp_transpose", "expert"),
          ('sequence', 'context', 'expert'),
          None,
          None,
          None,
        ),
      case ("w0_kernel" | "w1_kernel"), TT.Weight:
        mesh_axes = (
          ('expert',),
          None,
          ("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"),
        )
      case "wo_kernel", TT.Activation:
        mesh_axes = (
          ('expert',),
          ("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"),
          None,
        )
      case ("gate_logits" | "pre_bias_logits"), TT.Activation:
        mesh_axes = (
          ("data", "fsdp", "fsdp_transpose", "expert"),
          ('sequence', 'context', 'expert'),
          None,
        )
      case ("dispatch_mask" | "combine_mask"), TT.Activation:
        mesh_axes = (
          ("data", "fsdp", "fsdp_transpose", "expert"),
          ('sequence', 'context', 'expert'),
          None,
          None,
        )
      case "dispatch", TT.Activation:
        mesh_axes = (
          ('expert',),
          ('data', 'fsdp', 'fsdp_transpose'),
          None,
         ("tensor", "tensor_transpose")
        )
      case ("layer_w0" | "layer_w1"), TT.Activation:
        mesh_axes = (
          ('expert',),
          ('data', 'fsdp', 'fsdp_transpose'),
          None,
          ('tensor', 'tensor_transpose', 'tensor_sequence')
        )
      case "intermediate_layer", TT.Activation:
        mesh_axes = (
          ('expert',),
          ('data', 'fsdp', 'fsdp_transpose'),
          None,
          ("tensor", "tensor_transpose"),
        )
      case "no_cap_inputs", TT.Activation:
        mesh_axes = (
          ("data", "fsdp", "fsdp_transpose", "expert"),
          ('sequence', 'context', 'expert'),
          ("tensor", "tensor_transpose")
        )
      case "sparse_inputs", TT.Weight:
        if batch_sharded_by_expert:
          batch_axis = ("data", "fsdp", "fsdp_transpose", "expert")
        else:
          batch_axis = ("data", "fsdp", "fsdp_transpose")

        if tensor_transpose:
          embed_axis = ("tensor", "tensor_transpose")
        else:
          embed_axis = None

        mesh_axes = (
          batch_axis,
          ('sequence', 'context', 'expert'),
          embed_axis
        )
      case ("sparse_gate_logits" | "sparse_pre_bias_logits"), TT.Weight:
        if batch_sharded_by_expert:
          batch_axis = ("data", "fsdp", "fsdp_transpose", "expert")
        else:
          batch_axis = ("data", "fsdp", "fsdp_transpose")

        mesh_axes = (
          batch_axis,
          ('sequence', 'context', 'expert'),
          None
        )
      case ("sparse_w0" | "sparse_w1"), TT.Weight:
        mesh_axes = (
          ("expert",),
          ("tensor_transpose",),
          ("tensor", "tensor_sequence", "autoregressive")
        )
      case "sparse_wo", TT.Weight:
        mesh_axes = (
          ("expert",),
          ("tensor", "tensor_sequence", "autoregressive"),
          ("expert",)
        )
      case "sparse_shard_map", TT.Activation:
        if batch_sharded_by_expert:
          batch_axis = ("data", "fsdp", "fsdp_transpose", "expert")
        else:
          batch_axis = ("data", "fsdp", "fsdp_transpose")

        mesh_axes = (
          batch_axis,
          ('sequence', 'context', 'expert')
          ("tensor", "tensor_transpose")
        )
      case _, _:
        assert False, "Unexpected tensor name for sharding"

    return PartitionSpec(*mesh_axes)

# As Llama3 but plus exp and embed_no_exp, activation_exp and activation_mlp
class Qwen3AxisShardingTraining(MeshSharding):

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
          # TODO: has multiple rules in current logical axis_rules
          mesh_axes.append(("fsdp", "fsdp_transpose", "sequence", "context", "expert"))
        case "vocab", TT.Weight:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'))
        case "norm", TT.Weight:
          mesh_axes.append(("tensor", "tensor_transpose", "tensor_sequence"))
        case ("activation_length" | "activation_norm_length"), TT.Activation:
          # TODO: has multiple rules for activation_length (not norm_length)
          mesh_axes.append(('sequence', 'context', 'expert'))
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
          mesh_axes.append((None,))
        case "mlp", TT.Weight:
          mesh_axes.append(("fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"))
        case "exp", TT.Weight:
          mesh_axes.append(("expert",))
        case "embed_no_exp", TT.Weight:
          # TODO: has multiple rules in current logical axis_rules
          mesh_axes.append(('fsdp', 'fsdp_transpose', 'sequence', 'tensor_transpose', 'context'))
        case "activation_exp", TT.Activation:
          mesh_axes.append(('expert',))
        case "activation_mlp", TT.Activation:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence'))
        case "embed_tensor_transpose", TT.Weight:
          mesh_axes.append(("tensor_transpose",))
        case _, _:
          assert False, "Unexpected logical axis name for sharding"

      assert_matches_logical_axis_rules(mesh_axes[-1], axis)

    return PartitionSpec(*mesh_axes)


class Qwen3AxisShardingTrainingV2(MeshSharding):

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_type = kwargs.get("tensor_type", TT.Weight)
    # TODO: this isn't correct - we need to check:
    # - whether ep is active
    # - if it is then whether the tensor is an expert or whether we'll be using the expert axis for something else
    # - if we'll be using for something else, whether we'll be using it for context or batch
    ep_as_context = kwargs.get("ep_ctx", False)
    tensor_transpose, fsdp_transpose = self.config.tensor_transpose, self.config.fsdp_transpose

    mesh_axes = []
    for axis in axes:
      match axis, tensor_type, ep_as_context:
        case "batch", TT.Activation, True:
          mesh_axes.append(("data", "fsdp", "fsdp_transpose", "expert"))
        case "batch", TT.Activation, False:
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
        case "vocab", TT.Weight, _:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'))
        case "norm", TT.Weight, _:
          mesh_axes.append(("tensor", "tensor_transpose", "tensor_sequence"))
        case ("length" | "norm_length"), TT.Activation, True:
          mesh_axes.append(('sequence', 'context', 'expert'))
        case ("length" | "norm_length"), TT.Activation, False:
          mesh_axes.append(('sequence', 'context'))
        case "kv", TT.Activation, _:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence'))
        case "kv_batch", TT.Activation, True:
          mesh_axes.append(('data', 'fsdp', 'fsdp_transpose', 'expert'))
        case "kv_batch", TT.Activation, False:
          mesh_axes.append(('data', 'fsdp', 'fsdp_transpose')),
        case "kv_heads", TT.Activation, _:
          mesh_axes.append(('tensor', 'tensor_transpose', 'sequence','tensor_sequence'))
        case "kv_head_dim", TT.Activation, _:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence'))
        case "heads", TT.Activation, _:
          mesh_axes.append(('tensor', 'tensor_transpose', 'sequence','tensor_sequence','autoregressive'))
        case  ("heads" | "q_heads" | "kv_heads"), TT.Weight, _:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive'))
        case ("kv", "kv_head_dim", "qkv", "num_activations"), TT.Weight, _:
          mesh_axes.append((None))
        case _, _, _:
          assert False, "Unexpected logical axis name for sharding"

    return PartitionSpec(*mesh_axes)
