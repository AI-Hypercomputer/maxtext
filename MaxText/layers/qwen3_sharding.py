from typing import Any

from jax.sharding import PartitionSpec, Mesh

from MaxText.sharding import MeshSharding, TensorType as TT, assert_matches_logical_axis_rules, Axis
dp, fsdp, tp, pp, fsdp_t, sp, cp, cp_ar, tp_t, tp_s, ep, ar = Axis


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
    tensor_transpose = kwargs.get(tp_t, False)

    mesh_axes = []
    match tensor, tensor_type:
      case "token_embedder" | "logits_dense", TT.Weight:
        mesh_axes = (
          (tp, tp_t, tp_s, ar),
          (fsdp, fsdp_t, sp, cp, ep)
        )
        if tensor == "logits_dense":
          mesh_axes = tuple(mesh_axes[1], mesh_axes[2])
      case "embed_output", TT.Activation:
        mesh_axes = (
          (dp, pp, fsdp, fsdp_t, ep),
          (tp_s, cp, sp),
          (tp, tp_t),
        )
      case ("pre_self_attention_layer_norm" | "post_self_attention_layer_norm" | "decoder_norm" | "rms_norm"), TT.Weight:
        mesh_axes = (
          (tp, tp_t, tp_s),
        )
      case ("inputs_q", "inputs_kv"), TT.Activation:
        mesh_axes = (
          (dp, fsdp, fsdp_t, ep) if ep_as_context else (dp, fsdp, fsdp_t),
          # TODO: there are multiple matches for this axis in the current axis_rules. figure out which is correct
          #       (see also bigger note on axis rules, below)
          (sp, cp) if ep_as_context else (sp, cp, ep),
          (tp, tp_t)
        )
      case ("query" | "key" | "value"), TT.Activation:
        mesh_axes = (
          (dp, fsdp, fsdp_t, ep) if ep_as_context else (dp, fsdp, fsdp_t),
          (sp, cp) if ep_as_context else (sp, cp, ep),  # TODO: as above
          (tp, tp_t, sp,tp_s)
          (tp, tp_t, tp_s)
        )
      case "out", TT.Activation:
        mesh_axes = ((dp, fsdp, fsdp_t, ep) if ep_as_context else (dp, fsdp, fsdp_t),
          (sp, cp) if ep_as_context else (sp, cp, ep),
          (tp, tp_t, sp,tp_s,ar),
          (tp, tp_t, tp_s)
        )
      case ("query" | "kv"), TT.Weight:
        mesh_axes = (
          (fsdp, fsdp_t, sp, cp, ep),
          (tp, tp_t, tp_s, ar),
          (None)
        )
      case "qkv", TT.Weight:
        mesh_axes = (
          (fsdp, fsdp_t, sp, cp, ep),
          (None),
          (tp, tp_t, tp_s, ar),
        )
      case "out", TT.Weight:
        mesh_axes = (
          (tp, tp_t, tp_s, ar),
          (None),
          (fsdp, fsdp_t, sp, cp, ep)
        )
      case "mlp_pre_norm", TT.Weight:
        mesh_axes = (tp, tp_t, tp_s)
      case ("inputs" | "lnx", "attn_lnx" | "mlp_lnx" | "mlp_pre_norm" | "layer_output" | "hidden" | "mlp_output"), TT.Activation:
        mesh_axes = (
            (dp, fsdp, fsdp_t, ep),
            (tp_s, cp, sp),
            (tp, tp_t),
        )
      case "mlp_wi_fused", TT.Weight:
        mesh_axes = (
            (fsdp, fsdp_t, sp, tp_t, cp, ep),
            None,
            (fsdp_t, tp, tp_s, ar),
        )
      case ("mlp_wi_unfused" | "mlp_wo"), TT.Weight:
        mesh_axes = (
            (fsdp, fsdp_t, sp, tp_t, cp, ep),
            (fsdp_t, tp, tp_s, ar),
        )
      case "gate_logit", TT.Weight:
        mesh_axes = (
          (fsdp, fsdp_t, sp, cp, ep),
          (None)
        )
      case ("moe_wi_0" | "moe_wi_1"), TT.Weight:
        mesh_axes = (
          (ep,),
          (fsdp, fsdp_t, sp, tp_t, cp),
          (fsdp_t, tp, tp_s, ar)
        )
      case "moe_wo", TT.Weight:
        mesh_axes = (
          ((ep,),
          (fsdp_t, tp, tp_s, ar),
          (fsdp, fsdp_t, sp, tp_t, cp))
        )
      case "expert_mask_fused", TT.Activation:
        mesh_axes = (
          (dp, fsdp, fsdp_t, ep),
          None,
          None,
          None,
        )
      case "expert_token_count", TT.Activation:
        mesh_axes = (
          (dp, fsdp, fsdp_t, ep),
          (sp, cp, ep),
          None,
          None,
          None,
        ),
      case ("w0_kernel" | "w1_kernel"), TT.Weight:
        mesh_axes = (
          (ep,),
          None,
          (fsdp_t, tp, tp_s, ar),
        )
      case "wo_kernel", TT.Activation:
        mesh_axes = (
          (ep,),
          (fsdp_t, tp, tp_s, ar),
          None,
        )
      case ("gate_logits" | "pre_bias_logits"), TT.Activation:
        mesh_axes = (
          (dp, fsdp, fsdp_t, ep),
          (sp, cp, ep),
          None,
        )
      case ("dispatch_mask" | "combine_mask"), TT.Activation:
        mesh_axes = (
          (dp, fsdp, fsdp_t, ep),
          (sp, cp, ep),
          None,
          None,
        )
      case "dispatch", TT.Activation:
        mesh_axes = (
          (ep,),
          (dp, fsdp, fsdp_t),
          None,
         (tp, tp_t)
        )
      case ("layer_w0" | "layer_w1"), TT.Activation:
        mesh_axes = (
          (ep,),
          (dp, fsdp, fsdp_t),
          None,
          (tp, tp_t, tp_s)
        )
      case "intermediate_layer", TT.Activation:
        mesh_axes = (
          (ep,),
          (dp, fsdp, fsdp_t),
          None,
          (tp, tp_t),
        )
      case "no_cap_inputs", TT.Activation:
        mesh_axes = (
          (dp, fsdp, fsdp_t, ep),
          (sp, cp, ep),
          (tp, tp_t)
        )
      case "sparse_inputs", TT.Weight:
        if batch_sharded_by_expert:
          batch_axis = (dp, fsdp, fsdp_t, ep)
        else:
          batch_axis = (dp, fsdp, fsdp_t)

        if tensor_transpose:
          embed_axis = (tp, tp_t)
        else:
          embed_axis = None

        mesh_axes = (
          batch_axis,
          (sp, cp, ep),
          embed_axis
        )
      case ("sparse_gate_logits" | "sparse_pre_bias_logits"), TT.Weight:
        if batch_sharded_by_expert:
          batch_axis = (dp, fsdp, fsdp_t, ep)
        else:
          batch_axis = (dp, fsdp, fsdp_t)

        mesh_axes = (
          batch_axis,
          (sp, cp, ep),
          None
        )
      case ("sparse_w0" | "sparse_w1"), TT.Weight:
        mesh_axes = (
          (ep,),
          (tp_t,),
          (tp, tp_s, ar)
        )
      case "sparse_wo", TT.Weight:
        mesh_axes = (
          (ep,),
          (tp, tp_s, ar),
          (ep,)
        )
      case "sparse_shard_map", TT.Activation:
        if batch_sharded_by_expert:
          batch_axis = (dp, fsdp, fsdp_t, ep)
        else:
          batch_axis = (dp, fsdp, fsdp_t)

        mesh_axes = (
          batch_axis,
          (sp, cp, ep)
          (tp, tp_t)
        )
      case _, _:
        assert False, "Unexpected tensor name for sharding"

    self.maybe_check_valid_mesh_axes(mesh_axes)
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
          mesh_axes.append((dp, fsdp, fsdp_t, ep))
        case "activation_batch_no_exp", TT.Activation:
          mesh_axes.append((dp, fsdp, fsdp_t))
        case "activation_embed_and_logits_batch", TT.Activation:
          mesh_axes.append((dp, pp, fsdp, fsdp_t, ep))
        case "activation_embed", TT.Activation:
          mesh_axes.append((tp, tp_t))
        case "embed", TT.Weight:
          # TODO: has multiple rules in current logical axis_rules
          mesh_axes.append((fsdp, fsdp_t, sp, cp, ep))
        case "vocab", TT.Weight:
          mesh_axes.append((tp, tp_t, tp_s, ar))
        case "norm", TT.Weight:
          mesh_axes.append((tp, tp_t, tp_s))
        case ("activation_length" | "activation_norm_length"), TT.Activation:
          # TODO: has multiple rules for activation_length (not norm_length)
          mesh_axes.append((sp, cp, ep))
        case "activation_length_no_exp", TT.Activation:
          mesh_axes.append((sp, cp))
        case "activation_kv", TT.Activation:
          mesh_axes.append((tp, tp_t, tp_s))
        case "activation_kv_batch", TT.Activation:
          mesh_axes.append((dp, fsdp, fsdp_t, ep))
        case "activation_kv_batch_no_exp", TT.Activation:
          mesh_axes.append((dp, fsdp, fsdp_t)),
        case "activation_kv_heads", TT.Activation:
          mesh_axes.append((tp, tp_t, sp,tp_s))
        case "activation_kv_head_dim", TT.Activation:
          mesh_axes.append((tp, tp_t, tp_s))
        case "activation_heads", TT.Activation:
          mesh_axes.append((tp, tp_t, sp,tp_s,ar))
        case  ("heads" | "q_heads" | "kv_heads"), TT.Weight:
          mesh_axes.append((tp, tp_t, tp_s, ar))
        case ("kv", "kv_head_dim", "qkv", "num_activations"), TT.Weight:
          mesh_axes.append((None,))
        case "mlp", TT.Weight:
          mesh_axes.append((fsdp_t, tp, tp_s, ar))
        case "exp", TT.Weight:
          mesh_axes.append((ep,))
        case "embed_no_exp", TT.Weight:
          # TODO: has multiple rules in current logical axis_rules
          mesh_axes.append((fsdp, fsdp_t, sp, tp_t, cp))
        case "activation_exp", TT.Activation:
          mesh_axes.append((ep,))
        case "activation_mlp", TT.Activation:
          mesh_axes.append((tp, tp_t, tp_s))
        case "embed_tensor_transpose", TT.Weight:
          mesh_axes.append((tp_t,))
        case _, _:
          assert False, "Unexpected logical axis name for sharding"

      assert_matches_logical_axis_rules(mesh_axes[-1], axis)

    self.maybe_check_valid_mesh_axes(mesh_axes)
    return PartitionSpec(*mesh_axes)


class Qwen3AxisShardingTrainingV2(MeshSharding):
  def __init__(self, config):
    super().__init__()
    self.config = config

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_name = kwargs["t"]
    tensor_type = kwargs.get("tensor_type", TT.Weight)
    # TODO: this isn't correct - we need to check:
    # - whether ep is active
    # - if it is then whether the tensor is an expert or whether we'll be using the expert axis for something else
    # - if we'll be using for something else, whether we'll be using it for context or batch
    ep_attn_type = self.config.expert_shard_attention_option
    tensor_transpose, fsdp_transpose = self.config.tensor_transpose, self.config.fsdp_transpose

    mesh_axes = []
    for axis in axes:
      match axis, tensor_type:
        case "batch", TT.Activation, :
          axis_mappings = [dp, fsdp, fsdp_t, ep]
          if ep_attn_type == "batch":
            axis_mappings.append(ep)
          mesh_axes.append(tuple(axis_mappings))
        case "embed_and_logits_batch", TT.Activation:
          axis_mappings = [dp, pp, fsdp, fsdp_t]
          if ep_attn_type == "batch":
            axis_mappings.append(ep)
          mesh_axes.append(tuple(axis_mappings))
        case "embed", TT.Activation:
          mesh_axes.append((tp, tp_t))
        case "embed", TT.Weight:
          axis_mappings = [fsdp, sp, cp, ep]
          if tensor_name in ("mlp_wi_fused", "mlp_wi_unfused", "mlp_wo") and tensor_transpose:
            axis_mappings.append(tp)
          mesh_axes.append(tuple(axis_mappings))
        case "mlp", TT.Weight:
          axis_mappings = [tp, tp_s, ar]
          if tensor_name in ("mlp_wi_fused", "mlp_wi_unfused", "mlp_wo") and fsdp_transpose:
            axis_mappings.append(fsdp)
          mesh_axes.append(tuple(axis_mappings))
        case "vocab", TT.Weight, _:
          mesh_axes.append((tp, tp_t, tp_s, ar))
        case "norm", TT.Weight, _:
          mesh_axes.append((tp, tp_t, tp_s))
        case ("length" | "norm_length"), TT.Activation:
          axis_mappings = [sp, cp]
          if ep_attn_type == "context":
            axis_mappings.append(ep)
          mesh_axes.append(tuple(axis_mappings))
        case "kv", TT.Activation, _:
          mesh_axes.append((tp, tp_t, tp_s))
        case "kv_batch", TT.Activation:
          axis_mappings = [dp, fsdp, fsdp_t]
          if ep_attn_type == "batch":
            axis_mappings.append(ep)
          mesh_axes.append(tuple(axis_mappings))
        case "kv_heads", TT.Activation, _:
          mesh_axes.append((tp, tp_t, sp,tp_s))
        case "kv_head_dim", TT.Activation, _:
          mesh_axes.append((tp, tp_t, tp_s))
        case "heads", TT.Activation, _:
          mesh_axes.append((tp, tp_t, sp,tp_s,ar))
        case  ("heads" | "q_heads" | "kv_heads"), TT.Weight, _:
          mesh_axes.append((tp, tp_t, tp_s, ar))
        case ("kv", "kv_head_dim", "qkv", "num_activations"), TT.Weight, _:
          mesh_axes.append((None,))
        case _, _, _:
          assert False, "Unexpected logical axis name for sharding"

    self.maybe_check_valid_mesh_axes(mesh_axes)
    return PartitionSpec(*mesh_axes)
