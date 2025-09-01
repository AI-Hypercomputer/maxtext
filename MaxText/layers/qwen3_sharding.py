import enum
from typing import Any

from jax.sharding import PartitionSpec, Mesh

from MaxText.common_types import MODEL_MODE_PREFILL, MODEL_MODE_DECODE, MODEL_MODE_TRAINING
from MaxText.sharding import MeshSharding, TensorType as TT, assert_matches_logical_axis_rules, Axis
# TODO: something less fragile
dp, fsdp, tp, pp, sp, cp, cp_ar, tp_s, ep, ar = Axis


class Qwen3ShardingTrainingV2(MeshSharding):
  def __init__(self, config):
    super().__init__()
    self.config = config

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_name = kwargs["t"]
    tensor_type = kwargs.get("tensor_type", TT.Weight)
    ep_attn_type = self.config.expert_shard_attention_option
    tensor_transpose, fsdp_transpose = self.config.tensor_transpose, self.config.fsdp_transpose

    if self.config.ici_context_autoregressive_parallelism > 1:
      raise Exception("Context autoregressive parallelism not supported for training")

    mesh_axes = []
    for axis in axes:
      match axis, tensor_type:
        case "batch", TT.Activation:
                                                      axis_mappings = [dp, fsdp]
                                                      if ep_attn_type == "batch":
                                                        axis_mappings.append(ep)
                                                      mesh_axes.append(tuple(axis_mappings))
        case "embed_and_logits_batch", TT.Activation:
                                                      axis_mappings = [dp, pp, fsdp]
                                                      if ep_attn_type == "batch":
                                                        axis_mappings.append(ep)
                                                      mesh_axes.append(tuple(axis_mappings))
        case "embed", TT.Activation:
                                                      mesh_axes.append((tp))
        case "embed", TT.Weight:
                                                      axis_mappings = [fsdp, sp, cp]
                                                      if tensor_name not in ("moe_wi_0", "moe_wi_1", "moe_wo"):
                                                        axis_mappings.append(ep)
                                                      if (tensor_name in ("mlp_wi_fused", "mlp_wi_unfused", "mlp_wo",
                                                                         "moe_wi_0", "moe_wi_1", "moe_wo")
                                                                         and tensor_transpose):
                                                        axis_mappings.append(tp)
                                                      mesh_axes.append(tuple(axis_mappings))
        case "mlp", TT.Weight:
                                                      axis_mappings = [tp, tp_s]
                                                      if (tensor_name in ("mlp_wi_fused", "mlp_wi_unfused", "mlp_wo",
                                                                         "moe_wi_0", "moe_wi_1", "moe_wo")
                                                                         and fsdp_transpose):
                                                        axis_mappings.append(fsdp)
                                                      mesh_axes.append(tuple(axis_mappings))
        case "mlp", TT.Activation:
                                                      mesh_axes.append((tp, tp_s))
        case "vocab", TT.Weight, _:
                                                      mesh_axes.append((tp, tp_s, ar))
        case "norm", TT.Weight, _:
                                                      mesh_axes.append((tp, tp_s))
        case "length", TT.Activation:
                                                      axis_mappings = [sp, cp]
                                                      if ep_attn_type == "context":
                                                        axis_mappings.append(ep)
                                                      mesh_axes.append(tuple(axis_mappings))
        case "norm_length", TT.Activation:
                                                      mesh_axes.append((tp_s, sp, cp))
        case "kv", TT.Activation, _:
                                                      mesh_axes.append((tp, tp_s))
        case "kv_batch", TT.Activation:
                                                      axis_mappings = [dp, fsdp]
                                                      if ep_attn_type == "batch":
                                                        axis_mappings.append(ep)
                                                      mesh_axes.append(tuple(axis_mappings))
        case "kv_heads", TT.Activation, _:
                                                      mesh_axes.append((tp, sp,tp_s))
        case "kv_head_dim", TT.Activation, _:
                                                      mesh_axes.append((tp, tp_s))
        case "heads", TT.Activation, _:
                                                      mesh_axes.append((tp, sp,tp_s))
        case  ("heads" | "q_heads" | "kv_heads"), TT.Weight, _:
                                                      mesh_axes.append((tp, tp_s))
        case ("kv", "kv_head_dim", "qkv", "num_activations"), TT.Weight, _:
                                                      mesh_axes.append((None,))
        case "exp", _:
                                                      mesh_axes.append((ep,))
        case "embed_tensor_transpose", TT.Weight:
                                                      # TODO: this previously mapped directly to tensor_transpose.
                                                      # should instead  map to fsdp in the case that
                                                      # tensor_transpose is enabled
                                                      mesh_axes.append(())
        case _, _, _:
          assert False, "Unexpected logical axis name for sharding"

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
    tensor_type = kwargs.get("tensor_type", TT.Weight)
    mode = kwargs.get("mode", MODEL_MODE_TRAINING)
    ep_attn_type = self.config.expert_shard_attention_option
    tensor_transpose, fsdp_transpose = self.config.tensor_transpose, self.config.fsdp_transpose
    context_ar = self.config.ici_context_autoregressive_parallelism > 1

    mesh_axes = []
    for axis in axes:
      match tensor_name, axis, tensor_type, mode:
        case _, "embed_and_logits_batch", TT.Activation, _:
          mesh_axes.append((dp, pp, fsdp))
        case _, "length", TT.Activation, TT.Prefill:
          # actually resolves to the same as for Decode but was at least intended to be mapped separately in current code
          mesh_axes.append((sp, cp))
        case _, "length", TT.Activation, TT.Decode:
          mesh_axes.append((sp, cp))
        case _, "embed", TT.Activation, _:
          mesh_axes.append((tp))
        case _, "mlp", TT.Activation:
          mesh_axes.append((tp, tp_s))
        case _, "embed", TT.Weight, _:
          # NOTE: here we fix a bug in the original code where sharding still happened in qkv in context_ar unlike kv
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
        # TODO: support these cases
        # prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, EMBED),
        # decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, EMBED),

        # prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
        # decode:  query = self.sharding.shard(query, t="query", a=(DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))

        # prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
        # decode:  key = self.sharding.shard(key, t="key", a=(DECODE_BATCH, DECODE_LENGTH, KV_HEAD, D_KV))

        # prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
        # decode:  value = self.sharding.shard(value, t="value", a=(DECODE_BATCH, DECODE_LENGTH, KV_HEAD, D_KV))

        # prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV),
        # decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV),
        #
        # TODO: more code here:
