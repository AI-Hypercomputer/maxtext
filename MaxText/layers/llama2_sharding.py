from typing import Any

from jax.sharding import PartitionSpec

from MaxText.sharding import MeshSharding, TensorType as TT, assert_matches_logical_axis_rules, Axis
dp, fsdp, tp, pp, fsdp_t, sp, cp, cp_ar, tp_t, tp_s, ep, ar = Axis


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
          (dp, fsdp, fsdp_t),
          (sp, cp, ep),
          (tp, tp_t)
        )
      case ("query" | "key" | "value"), TT.Activation:
        mesh_axes = (
          (dp, fsdp, fsdp_t),
          (sp, cp) if ep_as_context else (sp, cp, ep),
          (tp, tp_t, sp,tp_s),
          (None,)
        )
      case "out", TT.Activation:
        mesh_axes = (
          (dp, fsdp, fsdp_t),
          (sp, cp, ep),
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
      case ("inputs" | "lnx", "attn_lnx" | "mlp_lnx" | "mlp_pre_norm" | "layer_output" | "hidden"), TT.Activation:
        mesh_axes = (
            (dp, fsdp, fsdp_t, ep),
            (tp_s, cp, sp),
            (tp, tp_t),
        )
      case "mlp_wi_fused", TT.Weight:
        mesh_axes = (
            (fsdp, fsdp_t, sp, tp_t, cp, ep),
            (None),
            (fsdp_t, tp, tp_s, ar),
        )
      case ("mlp_wi_unfused" | "mlp_wo"), TT.Weight:
        mesh_axes = (
            (fsdp, fsdp_t, sp, tp_t, cp, ep),
            (fsdp_t, tp, tp_s, ar),
        )
      case "mlp_wo", TT.Weight:
        mesh_axes = (
            (fsdp_t, tp, tp_s, ar),
            (fsdp, fsdp_t, sp, tp_t, cp, ep),
        )
      case _, _:
        assert False, "Unexpected tensor name for sharding"

    self.maybe_check_valid_mesh_axes(mesh_axes)
    return PartitionSpec(*mesh_axes)


class Llama2ShardingTraining(MeshSharding):

  def __init__(self):
    super().__init__()
    self.all_axes = set([
      "activation_batch", "activation_batch_no_exp", "activation_embed_and_logits_batch", "activation_embed", "embed",
      "vocab", "norm", "activation_length", "activation_norm_length", "activation_length_no_exp", "activation_kv",
      "activation_kv_batch", "activation_kv_batch_no_exp", "activation_kv_heads", "activation_kv_head_dim",
      "activation_heads", "heads", "q_heads", "kv_heads", "kv", "kv_head_dim", "qkv", "num_activations", "mlp",
      "activation_mlp",
    ])
    self.unused_axes = self.all_axes.copy()

  def mark_axis_used(self, axis):
    assert axis in self.all_axes, "Axis not present in list of all axes - either it wasn't there or has been used twice"
    if axis in self.unused_axes:
      self.unused_axes.remove(axis)

  def check_final_state(self):
    assert not self.unused_axes, f"Axes not used: {self.unused_axes}"

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_name = kwargs["t"]
    tensor_type = kwargs.get("tensor_type", TT.Weight)

    mesh_axes = []
    for axis in axes:
      self.mark_axis_used(axis)

      match axis, tensor_type:
        case "activation_batch", TT.Activation:                   axis_mapping = (dp, fsdp, fsdp_t, ep)
        case "activation_embed_and_logits_batch", TT.Activation:  axis_mapping = (dp, pp, fsdp, fsdp_t, ep)
        case "activation_embed", TT.Activation:                   axis_mapping = (tp, tp_t)
        case "embed", TT.Weight:                                  axis_mapping = (fsdp, fsdp_t, sp, cp, ep)
        case "vocab", TT.Weight:                                  axis_mapping = (tp, tp_t, tp_s, ar)
        case "norm", TT.Weight:                                   axis_mapping = (tp, tp_t, tp_s)
        case "activation_norm_length", TT.Activation:             axis_mapping = (tp_s, cp, sp)
        case "activation_length", TT.Activation:                  axis_mapping = (sp, cp, ep) if tensor_name != "mlp_pre_out" else ()
        case "activation_kv", TT.Activation:                      axis_mapping = (tp, tp_t, tp_s) if tensor_name != "out" else ()
        case "activation_kv_batch", TT.Activation:                axis_mapping = (dp, fsdp, fsdp_t, ep)
        case "activation_kv_heads", TT.Activation:                axis_mapping = (tp, tp_t, sp,tp_s)
        case "activation_kv_head_dim", TT.Activation:             axis_mapping = (tp, tp_t, tp_s) if tensor_name not in ("query", "key", "value") else ()
        case "activation_heads", TT.Activation:                   axis_mapping = (tp, tp_t, sp,tp_s,ar)
        case  ("heads" | "q_heads" | "kv_heads"), TT.Weight:      axis_mapping = (tp, tp_t, tp_s, ar)
        case ("kv", "kv_head_dim", "qkv",
              "num_activations"), TT.Weight:                      axis_mapping = (None)
        case "mlp", TT.Weight:                                    axis_mapping = (fsdp_t, tp, tp_s, ar)
        case "activation_mlp", TT.Activation:                     axis_mapping = (tp, tp_t, tp_s)
        case _, _:
                                                                  assert False, "Unexpected logical axis name for sharding"
                                                                  axis_mapping = None

      axis_mapping_values = [axis.value for axis in axis_mapping]
      mesh_axes.append(axis_mapping_values)

    # NOTE: this has to be done on all axes at once as they're not mapped independently
    assert_matches_logical_axis_rules(mesh_axes, axes)
    self.maybe_check_valid_mesh_axes(mesh_axes)
    return PartitionSpec(*mesh_axes)
