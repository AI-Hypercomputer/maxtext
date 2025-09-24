from typing import Any

from jax.sharding import PartitionSpec

from MaxText.sharding import MeshSharding, TensorType as TT, Axis
dp, fsdp, tp, pp, sp, cp, cp_ar, tp_s, ep, ar, tp_t, fsdp_t = Axis


class DenseShardingTrainingV2(MeshSharding):
  def __init__(self, config):
    super().__init__()
    self.config = config

  def __map_axis_tensor_sequence__(self, axis, tensor_type, tensor_name):
    tensor_seq = self.config.tensor_sequence
    tp_type = "tp" if tensor_seq else "tp_s"

    match axis, tensor_name, tp_type, tensor_type:
      case "batch",                            _,                            _,      TT.Activation: return (dp, fsdp)
      case "elb",                              _,                            _,      TT.Activation: return (dp, pp, fsdp)
      case "embed",                            _,                            "tp_s", TT.Activation: return ()
      case "embed",                            _,                            "tp",   TT.Activation: return (tp, tp_t)
      case "embed",                            _,                            _,      TT.Weight:     return (fsdp, fsdp_t, sp, cp)
      case "mlp",                              _,                            _,      TT.Weight:     return (fsdp_t, tp)
      case "mlp",                              _,                            _,      TT.Activation: return (tp, tp_t)
      case "vocab",                            _,                            _,      TT.Weight:     return (tp, ar)
      case "vocab",                            _,                            _,      TT.Activation: return (tp, tp_t, sp)
      case "norm",                             _,                            _,      TT.Weight:     return (tp)
      case "length",                           "mlp_pre_out",                _,      TT.Activation: return ()
      case "length",                           _,                            _,      TT.Activation: return (sp, cp)
      case "norm_length",                      _,                            "tp_s", TT.Activation: return (tp, sp, cp)
      case "norm_length",                      _,                            "tp",   TT.Activation: return (sp, cp)
      case "kv",                               "out",                        _,      TT.Activation: return ()
      case "kv",                               _,                            _,      TT.Activation: return (tp)
      case "kv_batch",                         _,                            _,      TT.Activation: return (dp, fsdp, fsdp_t)
      case ("kv_heads" | "heads"),             _,                            _,      TT.Activation: return (tp, tp_t, sp)
      case "kv_head_dim",                      ("query" | "key" | "value"),  _,      TT.Activation: return ()
      case "kv_head_dim",                      _,                            _,      TT.Activation: return (tp, tp_t, tp_s)
      case ("heads" | "q_heads" | "kv_heads"), _,                            _,      TT.Weight:     return (tp, tp_t)
      case ("kv" | "kv_head_dim" | "qkv"),     _,                            _,      TT.Weight:     return ()
      case "num_activations",                  _,                            _,      TT.Weight:     return ()
      case _, _, _:
        assert False, "Unexpected logical axis name for sharding"
        return ()

  def __map_axis_juan__(self, axis, tensor_type, tensor_name):

    # TODO: activation_vocab is missing
    match axis, tensor_name, tensor_type:
      case "batch",                            _,                            TT.Activation: return (dp, fsdp)
      case "elb",                              _,                            TT.Activation: return (dp, pp, fsdp)
      case "embed",                            _,                            TT.Activation: return (tp, tp_t)
      case "embed",                            _,                            TT.Weight:     return (fsdp, fsdp_t, sp, cp)
      case "mlp",                              _,                            TT.Weight:     return (fsdp_t, tp, tp_s)
      case "mlp",                              _,                            TT.Activation: return (tp, tp_t, tp_s)
      case "vocab",                            _,                            TT.Weight:     return (tp, tp_s, ar)
      case "norm",                             _,                            TT.Weight:     return (tp, tp_s)
      case "length",                           "mlp_pre_out",                TT.Activation: return ()
      case "length",                            _,                           TT.Activation: return (sp, cp)
      case "norm_length",                       _,                           TT.Activation: return (tp_s, sp, cp)
      case "kv",                                "out",                       TT.Activation: return ()
      case "kv",                                _,                           TT.Activation: return (tp, tp_s)
      case "kv_batch",                          _,                           TT.Activation: return (dp, fsdp, fsdp_t)
      case ("kv_heads" | "heads"),              _,                           TT.Activation: return (tp, tp_t, sp, tp_s)
      case "kv_head_dim",                       ("query" | "key" | "value"), TT.Activation: return ()
      case "kv_head_dim",                       _,                           TT.Activation: return (tp, tp_t, tp_s)
      case ("heads" | "q_heads" | "kv_heads"),  _,                           TT.Weight:     return (tp, tp_t, tp_s)
      case ("kv" | "kv_head_dim" | "qkv"),      _,                           TT.Weight:     return ()
      case "num_activations",                   _,                           TT.Weight:     return ()
      case _, _, _:
        assert False, "Unexpected logical axis name for sharding"
        return ()

  def map_axis(self, axis, tensor_type, tensor_name):
    match axis, tensor_type:
      case "batch", TT.Activation:                        return (dp, fsdp)
      case "embed_and_logits_batch", TT.Activation:       return (dp, pp, fsdp)
      case "embed", TT.Activation:                        return (tp, tp_t)
      case "embed", TT.Weight:                            return (fsdp, fsdp_t, sp, cp)
      case "mlp", TT.Weight:                              return (fsdp_t, tp, tp_s)
      case "mlp", TT.Activation:                          return (tp, tp_t, tp_s)
      case "vocab", TT.Weight:                            return (tp, tp_s, ar)
      case "norm", TT.Weight:                             return (tp, tp_s)
      case "length", TT.Activation:                       return (sp, cp) if tensor_name != "mlp_pre_out" else ()
      case "norm_length", TT.Activation:                  return (tp_s, sp, cp)
      case "kv", TT.Activation:                           return (tp, tp_s)if tensor_name != "out" else ()
      case "kv_batch", TT.Activation:                     return (dp, fsdp, fsdp_t)
      case ("kv_heads" | "heads"), TT.Activation:         return (tp, tp_t, sp, tp_s)
      case "kv_head_dim", TT.Activation:                  return (tp, tp_t, tp_s) if tensor_name not in ("query", "key", "value") else ()
      case ("heads" | "q_heads" | "kv_heads"), TT.Weight: return (tp, tp_t, tp_s)
      case ("kv", "kv_head_dim", "qkv"), TT.Weight:       return ()
      case "num_activations", TT.Weight:                  return ()
      case _, _, _:
        assert False, "Unexpected logical axis name for sharding"
        return ()


  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_name = kwargs["t"]
    tensor_type = kwargs.get("tt", TT.Weight)

    mesh_axes = []
    for axis in axes:
      mesh_axes.append(map_axis(axis, tensor_type, tensor_name))

    self.maybe_check_valid_mesh_axes(mesh_axes)
    return PartitionSpec(*mesh_axes)
