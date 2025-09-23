import enum
from typing import Any

from jax.sharding import PartitionSpec, Mesh

from MaxText.common_types import MODEL_MODE_PREFILL, MODEL_MODE_DECODE, MODEL_MODE_TRAINING
from MaxText.sharding import MeshSharding, TensorType as TT, assert_matches_logical_axis_rules, Axis
# TODO: something less fragile
dp, fsdp, tp, pp, sp, cp, cp_ar, tp_s, ep, ar, tp_t, fsdp_t = Axis


# NOTE: a few rules are currently missing since we have not yet migrated attention_op (e.g. attention_q_length)
#
class MoEShardingTrainingV2(MeshSharding):
  def __init__(self, config):
    super().__init__()
    self.config = config

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_name = kwargs["t"]
    tensor_type = kwargs.get("tt", TT.Weight)
    ep_attn_type = self.config.expert_shard_attention_option
    # TODO: we could get this from config or the mesh
    tp_t_active = kwargs.get("tp_t_active", False)

    if self.config.ici_context_autoregressive_parallelism > 1:
      raise Exception("Context autoregressive parallelism not supported for training")

    mesh_axes = []
    for axis in axes:
     mesh_axes.append(self.map_axis(axis, tensor_name, tensor_type, ep_attn_type, tp_t_active))

    self.maybe_check_valid_mesh_axes(mesh_axes)
    return PartitionSpec(*mesh_axes)

  def map_axis(axis, tensor_name, tensor_type, ep_attn_type, tp_t_active):
    match axis, tensor_type:
      case "batch", TT.Activation:
                                                    axis_mappings = [dp, fsdp, fsdp_t]
                                                    # for attention, we re-use the ep axis
                                                    if (ep_attn_type == "batch" and
                                                        tensor_name not in ("dispatch", "layer_w0", "layer_w1",
                                                                            "intermediate_layer")):
                                                      axis_mappings.append(ep)
                                                    return axis_mappings
      case "embed_and_logits_batch", TT.Activation:
                                                    axis_mappings = [dp, pp, fsdp, fsdp_t]
                                                    if ep_attn_type == "batch":
                                                      axis_mappings.append(ep)
                                                    return tuple(axis_mappings)
      case "embed", TT.Activation:
                                                    if tensor_name == "sparse_inputs" and not tp_t_active:
                                                      return ()
                                                    else:
                                                      return (tp, tp_t)
      case "embed", TT.Weight:
                                                    axis_mappings = [fsdp, fsdp_t, sp, cp]
                                                    # for attention, we re-use the ep axis
                                                    if tensor_name not in ("moe_wi_0", "moe_wi_1", "moe_wo"):
                                                      axis_mappings.append(ep)
                                                    return tuple(axis_mappings)
      case "mlp", TT.Weight:
                                                    return (fsdp_t, tp, tp_s)
      case "mlp", TT.Activation:
                                                    return (tp, tp_t, tp_s)
      case "vocab", TT.Weight:
                                                    return (tp, tp_t, tp_s, ar)
      case "norm", TT.Weight:
                                                    return (tp, tp_t, tp_s)
      case "length", TT.Activation:
                                                    axis_mappings = [sp, cp]
                                                    if ep_attn_type == "context":
                                                      axis_mappings.append(ep)
                                                    return tuple(axis_mappings)
      case "norm_length", TT.Activation:
                                                    return (tp_s, sp, cp)
      case "kv", TT.Activation:
                                                    return (tp, tp_s)
      case "kv_batch", TT.Activation:
                                                    axis_mappings = [dp, fsdp, fsdp_t]
                                                    if ep_attn_type == "batch":
                                                      axis_mappings.append(ep)
                                                    return tuple(axis_mappings)
      case ("kv_heads" | "heads"), TT.Activation, _:
                                                    return (tp, tp_t, sp, tp_s)
      case "kv_head_dim", TT.Activation:
                                                    return (tp, tp_t, tp_s)
      case  ("heads" | "q_heads" | "kv_heads"), TT.Weight, _:
                                                    return (tp, tp_t, tp_s)
      case ("kv", "kv_head_dim", "qkv", "num_activations"), TT.Weight, _:
                                                    return (None,)
      case "exp", _:
                                                    return (ep,)
      case "embed_tensor_transpose", TT.Weight:
                                                    return (tp_t)
      case _, _:
                                                    assert False, "Unexpected logical axis name for sharding"
                                                    return (())

  def map_axis_juan(axis, tensor_name, tensor_type, ep_attn_type, tp_t_active):
    moe_tensors = ("dispatch", "layer_w0", "layer_w1", "intermediate_layer", "moe_wi_0", "moe_wi_1", "moe_wo", "sparse_inputs")
    moe_tensor = "moe" if tensor_name in moe_tensors else "non-moe"
    tp_t = "tp_t" if tp_t_active else "non-tp_t"
    class C: elb = "embed_and_logits_batch"

    match axis, tensor_type, moe_tensor, ep_attn_type, tp_t_active:

      case "batch",                                           TT.Activation, "non-moe", "batch", _: return (dp, fsdp, fsdp_t, ep)
      case "batch",                                           TT.Activation, _, _, _:               return (dp, fsdp, fsdp_t)
      case C.elb, _,                                          TT.Activation, "batch", _:            return (dp, pp, fsdp, fsdp_t, ep)
      case C.elb,                                             TT.Activation, _, "context", _:       return (dp, pp, fsdp, fsdp_t)
      case "embed",                                           TT.Activation, "moe", _, "non-tp_t":  return ()
      case "embed",                                           TT.Activation, _, _, _:               return (tp, tp_t)
      case "embed",                                           TT.Weight,     "moe", _, _:           return (fsdp, fsdp_t, sp, cp)
      case "embed",                                           TT.Weight,     _, _, _:               return (fsdp, fsdp_t, sp, cp, ep)
      case "mlp",                                             TT.Weight,     _, _, _:               return (fsdp_t, tp, tp_s)
      case "mlp",                                             TT.Activation, _, _, _:               return (tp, tp_t, tp_s)
      case "vocab",                                           TT.Weight,     _, _, _:               return (tp, tp_t, tp_s, ar)
      case "norm",                                            TT.Weight,     _, _, _:               return (tp, tp_t, tp_s)
      case "length",                                          TT.Activation, _, "context":          return (sp, cp, ep)
      case "length",                                          TT.Activation, _, _, _:               return (sp, cp)
      case "norm_length",                                     TT.Activation, _, _, _:               return (tp_s, sp, cp)
      case "kv",                                              TT.Activation, _, _, _:               return (tp, tp_s)
      case "kv_batch",                                        TT.Activation, _, "batch", _:         return (dp, fsdp, fsdp_t, ep)
      case "kv_batch",                                        TT.Activation, _, _, _:               return (dp, fsdp, fsdp_t)
      case ("kv_heads" | "heads"),                            TT.Activation, _, _, _:               return (tp, tp_t, sp, tp_s)
      case "kv_head_dim",                                     TT.Activation, _, _, _:               return (tp, tp_t, tp_s)
      case  ("heads" | "q_heads" | "kv_heads"),               TT.Weight,     _, _, _:               return (tp, tp_t, tp_s)
      case ("kv", "kv_head_dim", "qkv", "num_activations"),   TT.Weight,     _, _, _:               return ()
      case "exp",                                             TT.Weight,     _, _, _, _:            return (ep,)
      case "embed_tensor_transpose",                          TT.Weight,     _, _, _:               return (tp_t)
      case _, __, _, _, :
        assert False, "Unexpected logical axis name for sharding"
        return (())

  def is_batch_sharded_by_expert(self):
    return True


class Qwen3VariantTrainingSharding(MeshSharding):b
  def __init__(self, config):
    super().__init__()
    self.qwen3_sharding = Qwen3ShardingTrainingV2(config)

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_name = kwargs["t"]
    tensor_type = kwargs.get("tt", TT.Weight)
    ep_attn_type = self.config.expert_shard_attention_option
    # TODO: we could get this from config or the mesh
    tp_t_active = kwargs.get("tp_t_active", False)

    mesh_axes = []
    for axis in axes:
      match axis, tensor_type:
        case "length", TT.Activation: mesh_axes.append(())
        case _, _:
          mesh_axes.append(self.qwen3_sharding.map_axis(axis, tensor_name, tensor_type, ep_attn_type, tp_t_active))

    self.maybe_check_valid_mesh_axes(mesh_axes)
    return PartitionSpec(*mesh_axes)


class ModelMode(enum.Enum):
  Train = MODEL_MODE_TRAINING
  Prefill = MODEL_MODE_PREFILL
  Decode = MODEL_MODE_DECODE


# Incomplete pseudo-code of inference specific sharding rules
class Qwen3ShardingInferenceV2(MeshSharding):
  def __init__(self, config):
    super().__init__()
    self.config = config

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_name = kwargs["t"]
    tensor_type = kwargs.get("tt", TT.Weight)
    mode = kwargs.get("mode", MODEL_MODE_TRAINING)
    ep_attn_type = self.config.expert_shard_attention_option
    context_ar = self.config.ici_context_autoregressive_parallelism > 1

    mesh_axes = []
    for axis in axes:
      match tensor_name, axis, tensor_type, mode:
        case _, "length", TT.Activation, TT.Prefill:
          mesh_axes.append((sp, cp))
        case _, "length", TT.Activation, TT.Decode:
          mesh_axes.append((sp))
        case _, "embed_and_logits_batch", TT.Activation, _:
          mesh_axes.append((dp, pp, fsdp))
        case _, "embed", TT.Activation, _:
          mesh_axes.append((tp))
        case _, "mlp", TT.Activation:
          mesh_axes.append((tp, tp_s))
        case _, "embed", TT.Weight, _:
          mesh_axes.append((tp) if not context_ar and tensor_name not in ("query", "kv", "qkv", "out") else ())
        case _, "q_heads", TT.Weight, _:
          mesh_axes.append((tp, tp_s, ar) if not context_ar and tensor_name not in ("query", "kv", "qkv", "out") else ())
        case _, "kv", TT.Weight, _:
          mesh_axes.append(())
        case ("length" | "norm_length"), TT.Activation:
          axis_mappings = [sp, cp]
          if ep_attn_type == "context":
            axis_mappings.append(ep)
          mesh_axes.append(tuple(axis_mappings))
        case "batch", TT.Activation, :
          axis_mappings = [dp, fsdp, ep]
          if ep_attn_type == "batch":
            axis_mappings.append(ep)
          mesh_axes.append(tuple(axis_mappings))
