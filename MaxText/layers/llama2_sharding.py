from typing import Any

from jax.sharding import PartitionSpec

from MaxText.sharding import MeshSharding, TensorType as TT, Axis
dp, fsdp, tp, pp, sp, cp, cp_ar, tp_s, ep, ar, tp_t, fsdp_t = Axis


class Llama2hardingTrainingV2(MeshSharding):
  def __init__(self, config):
    super().__init__()
    self.config = config

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_name = kwargs["t"]
    tensor_type = kwargs.get("tt", TT.Weight)

    mesh_axes = []
    for axis in axes:
      match axis, tensor_type:
        case "batch", TT.Activation:                        mesh_axes.append((dp, fsdp))
        case "embed_and_logits_batch", TT.Activation:       mesh_axes.append((dp, pp, fsdp))
        case "embed", TT.Activation:                        mesh_axes.append((tp, tp_t))
        case "embed", TT.Weight:                            mesh_axes.append((fsdp, fsdp_t, sp, cp))
        case "mlp", TT.Weight:                              mesh_axes.append((fsdp_t, tp, tp_s))
        case "mlp", TT.Activation:                          mesh_axes.append((tp, tp_t, tp_s))
        case "vocab", TT.Weight:                            mesh_axes.append((tp, tp_s, ar))
        case "norm", TT.Weight:                             mesh_axes.append((tp, tp_s))
        case "length", TT.Activation:                       mesh_axes.append((sp, cp)) if tensor_name != "mlp_pre_out" else ()
        case "norm_length", TT.Activation:                  mesh_axes.append((tp_s, sp, cp))
        case "kv", TT.Activation:                           mesh_axes.append((tp, tp_s))  if tensor_name != "out" else ()
        case "kv_batch", TT.Activation:                     mesh_axes.append((dp, fsdp, fsdp_t))
        case ("kv_heads" | "heads"), TT.Activation:         mesh_axes.append((tp, tp_t, sp, tp_s))
        case "kv_head_dim", TT.Activation:                  mesh_axes.append((tp, tp_t, tp_s) if tensor_name not in ("query", "key", "value") else ())
        case ("heads" | "q_heads" | "kv_heads"), TT.Weight: mesh_axes.append((tp, tp_t, tp_s))
        case ("kv", "kv_head_dim", "qkv"), TT.Weight:       mesh_axes.append(())
        case "num_activations", TT.Weight:                  mesh_axes.append(())
        case _, _, _:
                                                            assert False, "Unexpected logical axis name for sharding"
                                                            mesh_axes.append(())

    self.maybe_check_valid_mesh_axes(mesh_axes)
    return PartitionSpec(*mesh_axes)
