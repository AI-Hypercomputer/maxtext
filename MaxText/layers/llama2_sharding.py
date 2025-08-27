from typing import Any

from jax.sharding import PartitionSpec

from MaxText.sharding import MeshSharding, TensorType as TT, assert_matches_logical_axis_rules, Axis
dp, fsdp, tp, pp, sp, cp, cp_ar, tp_s, ep, ar = Axis


class Llama2hardingTrainingV2(MeshSharding):
  def __init__(self, config):
    super().__init__()
    self.config = config

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_name = kwargs["t"]
    tensor_type = kwargs.get("tensor_type", TT.Weight)
    ep_attn_type = self.config.expert_shard_attention_option
    tensor_transpose, fsdp_transpose = self.config.tensor_transpose, self.config.fsdp_transpose

    mesh_axes = []
    for axis in axes:
      match axis, tensor_type:
        case "batch", TT.Activation:
          axis_mappings = [dp, fsdp]
          if ep_attn_type == "batch":
            axis_mappings.append(ep)
          mesh_axes.append(tuple(axis_mappings))
        case "embed_and_logits_batch", TT.Activation:
          axis_mappings = [dp, pp, fsdp, ep]
          if ep_attn_type == "batch":
            axis_mappings.append(ep)
          mesh_axes.append(tuple(axis_mappings))
        case "embed", TT.Activation:
          mesh_axes.append((tp))
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
        case "vocab", TT.Weight:
          mesh_axes.append((tp, tp_s, ar))
        case "norm", TT.Weight:
          mesh_axes.append((tp, tp_s))
        case ("length" | "norm_length"), TT.Activation:
          axis_mappings = [sp, cp]
          if ep_attn_type == "context":
            axis_mappings.append(ep)
          mesh_axes.append(tuple(axis_mappings))
        case "kv", TT.Activation:
          mesh_axes.append((tp, tp_s))
        case "kv_batch", TT.Activation:
          axis_mappings = [dp, fsdp]
          if ep_attn_type == "batch":
            axis_mappings.append(ep)
          mesh_axes.append(tuple(axis_mappings))
        case "kv_heads", TT.Activation:
          mesh_axes.append((tp, sp,tp_s))
        case "kv_head_dim", TT.Activation:
          mesh_axes.append((tp, tp_s))
        case "heads", TT.Activation:
          mesh_axes.append((tp, sp,tp_s,ar))
        case  ("heads" | "q_heads" | "kv_heads"), TT.Weight:
          mesh_axes.append((tp, tp_s, ar))
        case ("kv", "kv_head_dim", "qkv", "num_activations"), TT.Weight:
          mesh_axes.append((None))
        case _, _, _:
          assert False, "Unexpected logical axis name for sharding"

    self.maybe_check_valid_mesh_axes(mesh_axes)
    return PartitionSpec(*mesh_axes)
