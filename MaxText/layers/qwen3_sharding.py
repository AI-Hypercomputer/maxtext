from typing import Any

from jax.sharding import PartitionSpec, Mesh

from MaxText.sharding import MeshSharding, TensorType as TT


# NOTE: differences to Llamna2/3
#       addition of mlp_output, moe_wi_0, moe_wi_1, moe_wo, gate_logit
#       we also added ctx_ar_parallel
#
# TODO: rms_norm or possibly decoder_norm might not be used
class Qwen3TensorShardingTraining(MeshSharding):

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    tensor = kwargs["t"]
    tensor_type = kwargs["tt"]
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
      case _, _:
        assert False, "Unexpected tensor name for sharding"

    return PartitionSpec(*mesh_axes)

# As Llama3 but plus exp and embed_no_exp, activation_exp and activation_mlp
class Qwen3AxisShardingTraining(MeshSharding):

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_type = kwargs["tt"]

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
          mesh_axes.append(('sequence', 'context', 'expert'))  # TODO: has multiple rules for activation_length (not norm_length)
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
        case "exp", TT.Weight:
          mesh_axes.append(("expert",))
        case "embed_no_exp", TT.Weight:
          # TODO: has multiple rules
          mesh_axes.append(('fsdp', 'fsdp_transpose', 'sequence', 'tensor_transpose', 'context'))
        case "activation_exp", TT.Activation:
          mesh_axes.append(('expert',))
        case "activation_mlp", TT.Activation:
          mesh_axes.append(('tensor', 'tensor_transpose', 'tensor_sequence'))
        case _, _:
          assert False, "Unexpected logical axis name for sharding"

    return PartitionSpec(*mesh_axes)
