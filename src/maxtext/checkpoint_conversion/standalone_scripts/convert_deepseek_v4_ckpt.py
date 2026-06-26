import argparse
import gc
import psutil
import torch
import numpy as np
from tqdm import tqdm
from absl import logging

import maxtext.max_logging as max_logging
from maxtext.checkpoint_conversion.utils.utils import save_weights_to_checkpoint, get_state_dict_from_model
from maxtext.checkpoint_conversion.utils.hf_model_configs import MODEL_PARAMS_DICT

def _get_expert_stack(chkpt_vars, hf_prefix, num_experts, weight_name):
  stacked_weights = []
  for expert_idx in range(num_experts):
    hf_key = f"{hf_prefix}.ffn.experts.{expert_idx}.{weight_name}.weight"
    if hf_key not in chkpt_vars:
      return None
    pt_tensor = chkpt_vars[hf_key]
    stacked_weights.append(pt_tensor.to(torch.float16).numpy().transpose())
  return np.stack(stacked_weights)

def _convert_huggingface_to_jax_weights(base_model_path, model_params, mem_info) -> dict:
  max_logging.log("Starting conversion to JAX weights")
  
  jax_weights = {
      "decoder": {
          "pre_layers": {},
          "scanned_blocks": {"layers_0": {}, "layers_1": {}}
      }
  }

  chkpt_vars = get_state_dict_from_model(base_model_path)
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # 1. Embeddings and LM Head
  max_logging.log("Processing embeddings and LM head")
  if "embed_tokens.weight" in chkpt_vars:
    jax_weights["token_embedder"] = {
        "embedding": chkpt_vars["embed_tokens.weight"].to(torch.float16).numpy()
    }
  if "norm.weight" in chkpt_vars:
    jax_weights["decoder"]["decoder_norm"] = {
        "scale": chkpt_vars["norm.weight"].to(torch.float16).numpy()
    }
  if "lm_head.weight" in chkpt_vars:
    jax_weights["decoder"]["logits_dense"] = {
        "kernel": chkpt_vars["lm_head.weight"].to(torch.float16).numpy().transpose()
    }
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # DeepSeek V4 splits layers into pre_layers (0,1,2) and scanned layers (3 to N)
  # But the scanned layers are interleaved into layers_0 (odd) and layers_1 (even).
  num_pre_layers = model_params["num_pre_layers"]
  layers_per_block = model_params["layers_per_block"]
  num_experts = model_params["num_experts"]

  layers_config = {
      "pre_layers": num_pre_layers,
      "layers_0": layers_per_block,
      "layers_1": layers_per_block
  }

  def t(arr):
    if arr is None: return arr
    axes = list(range(len(arr.shape)))
    axes[0], axes[1] = axes[1], axes[0]
    return np.transpose(arr, axes=tuple(axes))

  for layer_key, layer_value in layers_config.items():
    max_logging.log(f"Processing {layer_key}")
    
    self_attention = {
        "q_norm": {"scale": None},
        "kv_norm": {"scale": None},
        "wq_a": {"kernel": None},
        "wq_b": {"kernel": None},
        "wkv": {"kernel": None},
        "o_a_proj": {"kernel": None},
        "o_b_proj": {"kernel": None}
    }
    # V4 specific attention properties
    if layer_key != "pre_layers":
      self_attention["sinks"] = None
      if layer_key == "layers_0":
        self_attention["hca_compressor"] = {
            "position_bias": None, "kv_norm": {"scale": None}, 
            "gate_proj": {"kernel": None}, "kv_proj": {"kernel": None}
        }
      elif layer_key == "layers_1":
        self_attention["csa_compressor"] = {
            "position_bias": None, "kv_norm": {"scale": None}, 
            "gate_proj": {"kernel": None}, "kv_proj": {"kernel": None},
            "indexer": {
                "position_bias": None, "kv_norm": {"scale": None},
                "gate_proj": {"kernel": None}, "kv_proj": {"kernel": None},
                "weights_proj": {"kernel": None}, "q_proj": {"kernel": None}
            }
        }
    
    pre_self_attention_layer_norm = {"scale": None}
    post_self_attention_layer_norm = {"scale": None}
    
    moe = {
        "MoeBlock_0": {
            "gate": {"kernel": None, "bias": None},
            "wi_0": None, "wo": None, "wi_1": None,
            "tid2eid": None
        },
        "shared_experts": {
            "wi_0": {"kernel": None}, "wo": {"kernel": None}, "wi_1": {"kernel": None}
        }
    }

    for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
      if layer_key == "pre_layers":
        hf_layer_idx = layer_idx
      elif layer_key == "layers_0":
        hf_layer_idx = num_pre_layers + 2 * layer_idx
      elif layer_key == "layers_1":
        hf_layer_idx = num_pre_layers + 2 * layer_idx + 1

      hf_prefix = f"layers.{hf_layer_idx}"
      
      if f"{hf_prefix}.attn.q_norm.weight" not in chkpt_vars:
        continue
      
      # Extract vars
      q_norm = chkpt_vars[f"{hf_prefix}.attn.q_norm.weight"].to(torch.float16).numpy()
      kv_norm = chkpt_vars[f"{hf_prefix}.attn.kv_norm.weight"].to(torch.float16).numpy()
      wq_a = chkpt_vars[f"{hf_prefix}.attn.wq_a.weight"].to(torch.float16).numpy().transpose()
      wq_b = chkpt_vars[f"{hf_prefix}.attn.wq_b.weight"].to(torch.float16).numpy().transpose().reshape(1024, 64, 512)
      wkv = chkpt_vars[f"{hf_prefix}.attn.wkv.weight"].to(torch.float16).numpy().transpose().reshape(4096, 1, 512)
      o_a_proj = chkpt_vars[f"{hf_prefix}.attn.wo_a.weight"].to(torch.float16).numpy().transpose().reshape(8, 4096, 1024)
      o_b_proj = chkpt_vars[f"{hf_prefix}.attn.wo_b.weight"].to(torch.float16).numpy().transpose()
      pre_attn_norm = chkpt_vars[f"{hf_prefix}.attn_norm.weight"].to(torch.float16).numpy()
      post_attn_norm = chkpt_vars[f"{hf_prefix}.ffn_norm.weight"].to(torch.float16).numpy()

      if layer_key == "pre_layers":
        # No stacking needed for pre_layers, just direct assignment
        jax_weights["decoder"]["pre_layers"][f"layers_{layer_idx}"] = {
            "self_attention": {
                "q_norm": {"scale": q_norm},
                "kv_norm": {"scale": kv_norm},
                "wq_a": {"kernel": wq_a},
                "wq_b": {"kernel": wq_b},
                "wkv": {"kernel": wkv},
                "o_a_proj": {"kernel": o_a_proj},
                "o_b_proj": {"kernel": o_b_proj},
            },
            "pre_self_attention_layer_norm": {"scale": pre_attn_norm},
            "post_self_attention_layer_norm": {"scale": post_attn_norm},
            "mlp": {
                "MoeBlock_0": {
                    "gate": {
                        "kernel": chkpt_vars[f"{hf_prefix}.ffn.gate.weight"].to(torch.float16).numpy().transpose(),
                        "bias": chkpt_vars[f"{hf_prefix}.ffn.gate.bias"].to(torch.float16).numpy() if f"{hf_prefix}.ffn.gate.bias" in chkpt_vars else None,
                    },
                    "tid2eid": chkpt_vars[f"{hf_prefix}.ffn.gate.tid2eid"].to(torch.float16).numpy() if f"{hf_prefix}.ffn.gate.tid2eid" in chkpt_vars else None,
                    "wi_0": _get_expert_stack(chkpt_vars, hf_prefix, num_experts, "w1"),
                    "wo": _get_expert_stack(chkpt_vars, hf_prefix, num_experts, "w2"),
                    "wi_1": _get_expert_stack(chkpt_vars, hf_prefix, num_experts, "w3")
                },
                "shared_experts": {
                    "wi_0": {"kernel": chkpt_vars[f"{hf_prefix}.ffn.shared_experts.w1.weight"].to(torch.float16).numpy().transpose()},
                    "wo": {"kernel": chkpt_vars[f"{hf_prefix}.ffn.shared_experts.w2.weight"].to(torch.float16).numpy().transpose()},
                    "wi_1": {"kernel": chkpt_vars[f"{hf_prefix}.ffn.shared_experts.w3.weight"].to(torch.float16).numpy().transpose()}
                }
            }
        }
        # cleanup nulls
        if jax_weights["decoder"]["pre_layers"][f"layers_{layer_idx}"]["mlp"]["MoeBlock_0"]["gate"]["bias"] is None:
            del jax_weights["decoder"]["pre_layers"][f"layers_{layer_idx}"]["mlp"]["MoeBlock_0"]["gate"]["bias"]
        if jax_weights["decoder"]["pre_layers"][f"layers_{layer_idx}"]["mlp"]["MoeBlock_0"]["tid2eid"] is None:
            del jax_weights["decoder"]["pre_layers"][f"layers_{layer_idx}"]["mlp"]["MoeBlock_0"]["tid2eid"]
        continue

      # For scanned layers: allocate if first time
      sinks = chkpt_vars[f"{hf_prefix}.attn.attn_sink"].to(torch.float16).numpy()
      
      if self_attention["q_norm"]["scale"] is None:
        stack_shape = (layer_value,)
        self_attention["sinks"] = np.zeros(stack_shape + sinks.shape, dtype=np.float16)
        self_attention["q_norm"]["scale"] = np.zeros(stack_shape + q_norm.shape, dtype=np.float16)
        self_attention["kv_norm"]["scale"] = np.zeros(stack_shape + kv_norm.shape, dtype=np.float16)
        self_attention["wq_a"]["kernel"] = np.zeros(stack_shape + wq_a.shape, dtype=np.float16)
        self_attention["wq_b"]["kernel"] = np.zeros(stack_shape + wq_b.shape, dtype=np.float16)
        self_attention["wkv"]["kernel"] = np.zeros(stack_shape + wkv.shape, dtype=np.float16)
        self_attention["o_a_proj"]["kernel"] = np.zeros(stack_shape + o_a_proj.shape, dtype=np.float16)
        self_attention["o_b_proj"]["kernel"] = np.zeros(stack_shape + o_b_proj.shape, dtype=np.float16)
        pre_self_attention_layer_norm["scale"] = np.zeros(stack_shape + pre_attn_norm.shape, dtype=np.float16)
        post_self_attention_layer_norm["scale"] = np.zeros(stack_shape + post_attn_norm.shape, dtype=np.float16)

      self_attention["sinks"][layer_idx, ...] = sinks
      self_attention["q_norm"]["scale"][layer_idx, ...] = q_norm
      self_attention["kv_norm"]["scale"][layer_idx, ...] = kv_norm
      self_attention["wq_a"]["kernel"][layer_idx, ...] = wq_a
      self_attention["wq_b"]["kernel"][layer_idx, ...] = wq_b
      self_attention["wkv"]["kernel"][layer_idx, ...] = wkv
      self_attention["o_a_proj"]["kernel"][layer_idx, ...] = o_a_proj
      self_attention["o_b_proj"]["kernel"][layer_idx, ...] = o_b_proj
      pre_self_attention_layer_norm["scale"][layer_idx, ...] = pre_attn_norm
      post_self_attention_layer_norm["scale"][layer_idx, ...] = post_attn_norm

      if layer_key == "layers_0":
        if f"{hf_prefix}.attn.compressor.ape" in chkpt_vars:
          pos_bias = chkpt_vars[f"{hf_prefix}.attn.compressor.ape"].to(torch.float16).numpy()
          c_norm = chkpt_vars[f"{hf_prefix}.attn.compressor.norm.weight"].to(torch.float16).numpy()
          c_gate = chkpt_vars[f"{hf_prefix}.attn.compressor.wgate.weight"].to(torch.float16).numpy().transpose()
          c_kv = chkpt_vars[f"{hf_prefix}.attn.compressor.wkv.weight"].to(torch.float16).numpy().transpose()
          
          hca = self_attention["hca_compressor"]
          if hca["position_bias"] is None:
            stack_shape = (layer_value,)
            hca["position_bias"] = np.zeros(stack_shape + pos_bias.shape, dtype=np.float16)
            hca["kv_norm"]["scale"] = np.zeros(stack_shape + c_norm.shape, dtype=np.float16)
            hca["gate_proj"]["kernel"] = np.zeros(stack_shape + c_gate.shape, dtype=np.float16)
            hca["kv_proj"]["kernel"] = np.zeros(stack_shape + c_kv.shape, dtype=np.float16)
          
          hca["position_bias"][layer_idx, ...] = pos_bias
          hca["kv_norm"]["scale"][layer_idx, ...] = c_norm
          hca["gate_proj"]["kernel"][layer_idx, ...] = c_gate
          hca["kv_proj"]["kernel"][layer_idx, ...] = c_kv

      elif layer_key == "layers_1":
        if f"{hf_prefix}.attn.compressor.ape" in chkpt_vars:
          pos_bias = chkpt_vars[f"{hf_prefix}.attn.compressor.ape"].to(torch.float16).numpy()
          c_norm = chkpt_vars[f"{hf_prefix}.attn.compressor.norm.weight"].to(torch.float16).numpy()
          c_gate = chkpt_vars[f"{hf_prefix}.attn.compressor.wgate.weight"].to(torch.float16).numpy().transpose()
          c_kv = chkpt_vars[f"{hf_prefix}.attn.compressor.wkv.weight"].to(torch.float16).numpy().transpose()
          
          csa = self_attention["csa_compressor"]
          if csa["position_bias"] is None:
            stack_shape = (layer_value,)
            csa["position_bias"] = np.zeros(stack_shape + pos_bias.shape, dtype=np.float16)
            csa["kv_norm"]["scale"] = np.zeros(stack_shape + c_norm.shape, dtype=np.float16)
            csa["gate_proj"]["kernel"] = np.zeros(stack_shape + c_gate.shape, dtype=np.float16)
            csa["kv_proj"]["kernel"] = np.zeros(stack_shape + c_kv.shape, dtype=np.float16)
          
          csa["position_bias"][layer_idx, ...] = pos_bias
          csa["kv_norm"]["scale"][layer_idx, ...] = c_norm
          csa["gate_proj"]["kernel"][layer_idx, ...] = c_gate
          csa["kv_proj"]["kernel"][layer_idx, ...] = c_kv

          if f"{hf_prefix}.attn.indexer.compressor.ape" in chkpt_vars:
            i_pos_bias = chkpt_vars[f"{hf_prefix}.attn.indexer.compressor.ape"].to(torch.float16).numpy()
            i_norm = chkpt_vars[f"{hf_prefix}.attn.indexer.compressor.norm.weight"].to(torch.float16).numpy()
            i_gate = chkpt_vars[f"{hf_prefix}.attn.indexer.compressor.wgate.weight"].to(torch.float16).numpy().transpose()
            i_kv = chkpt_vars[f"{hf_prefix}.attn.indexer.compressor.wkv.weight"].to(torch.float16).numpy().transpose()
            i_wproj = chkpt_vars[f"{hf_prefix}.attn.indexer.weights_proj.weight"].to(torch.float16).numpy().transpose()
            i_qproj = chkpt_vars[f"{hf_prefix}.attn.indexer.wq_b.weight"].to(torch.float16).numpy().transpose()

            idxr = csa["indexer"]
            if idxr["position_bias"] is None:
              idxr["position_bias"] = np.zeros(stack_shape + i_pos_bias.shape, dtype=np.float16)
              idxr["kv_norm"]["scale"] = np.zeros(stack_shape + i_norm.shape, dtype=np.float16)
              idxr["gate_proj"]["kernel"] = np.zeros(stack_shape + i_gate.shape, dtype=np.float16)
              idxr["kv_proj"]["kernel"] = np.zeros(stack_shape + i_kv.shape, dtype=np.float16)
              idxr["weights_proj"]["kernel"] = np.zeros(stack_shape + i_wproj.shape, dtype=np.float16)
              idxr["q_proj"]["kernel"] = np.zeros(stack_shape + i_qproj.shape, dtype=np.float16)
              
            idxr["position_bias"][layer_idx, ...] = i_pos_bias
            idxr["kv_norm"]["scale"][layer_idx, ...] = i_norm
            idxr["gate_proj"]["kernel"][layer_idx, ...] = i_gate
            idxr["kv_proj"]["kernel"][layer_idx, ...] = i_kv
            idxr["weights_proj"]["kernel"][layer_idx, ...] = i_wproj
            idxr["q_proj"]["kernel"][layer_idx, ...] = i_qproj

      # --- MoE ---
      if f"{hf_prefix}.ffn.gate.weight" in chkpt_vars:
        gate_w = chkpt_vars[f"{hf_prefix}.ffn.gate.weight"].to(torch.float16).numpy().transpose()
        gate_b = chkpt_vars[f"{hf_prefix}.ffn.gate.bias"].to(torch.float16).numpy() if f"{hf_prefix}.ffn.gate.bias" in chkpt_vars else None
        
        w1 = _get_expert_stack(chkpt_vars, hf_prefix, num_experts, "w1")
        w2 = _get_expert_stack(chkpt_vars, hf_prefix, num_experts, "w2")
        w3 = _get_expert_stack(chkpt_vars, hf_prefix, num_experts, "w3")

        sw1 = chkpt_vars[f"{hf_prefix}.ffn.shared_experts.w1.weight"].to(torch.float16).numpy().transpose()
        sw2 = chkpt_vars[f"{hf_prefix}.ffn.shared_experts.w2.weight"].to(torch.float16).numpy().transpose()
        sw3 = chkpt_vars[f"{hf_prefix}.ffn.shared_experts.w3.weight"].to(torch.float16).numpy().transpose()

        if moe["MoeBlock_0"]["gate"]["kernel"] is None:
          stack_shape = (layer_value,)
          moe["MoeBlock_0"]["gate"]["kernel"] = np.zeros(stack_shape + gate_w.shape, dtype=np.float16)
          if gate_b is not None:
            moe["MoeBlock_0"]["gate"]["bias"] = np.zeros(stack_shape + gate_b.shape, dtype=np.float16)
          
          moe["MoeBlock_0"]["wi_0"] = np.zeros(stack_shape + w1.shape, dtype=np.float16)
          moe["MoeBlock_0"]["wo"] = np.zeros(stack_shape + w2.shape, dtype=np.float16)
          moe["MoeBlock_0"]["wi_1"] = np.zeros(stack_shape + w3.shape, dtype=np.float16)
          
          moe["shared_experts"]["wi_0"]["kernel"] = np.zeros(stack_shape + sw1.shape, dtype=np.float16)
          moe["shared_experts"]["wo"]["kernel"] = np.zeros(stack_shape + sw2.shape, dtype=np.float16)
          moe["shared_experts"]["wi_1"]["kernel"] = np.zeros(stack_shape + sw3.shape, dtype=np.float16)

        moe["MoeBlock_0"]["gate"]["kernel"][layer_idx, ...] = gate_w
        if gate_b is not None:
          moe["MoeBlock_0"]["gate"]["bias"][layer_idx, ...] = gate_b
        moe["MoeBlock_0"]["wi_0"][layer_idx, ...] = w1
        moe["MoeBlock_0"]["wo"][layer_idx, ...] = w2
        moe["MoeBlock_0"]["wi_1"][layer_idx, ...] = w3
        
        moe["shared_experts"]["wi_0"]["kernel"][layer_idx, ...] = sw1
        moe["shared_experts"]["wo"]["kernel"][layer_idx, ...] = sw2
        moe["shared_experts"]["wi_1"]["kernel"][layer_idx, ...] = sw3


    if layer_key != "pre_layers":
      # RE-ORDER manually (transpose)
      self_attention["sinks"] = t(self_attention["sinks"])
      self_attention["q_norm"]["scale"] = t(self_attention["q_norm"]["scale"])
      self_attention["kv_norm"]["scale"] = t(self_attention["kv_norm"]["scale"])
      self_attention["wq_a"]["kernel"] = t(self_attention["wq_a"]["kernel"])
      self_attention["wq_b"]["kernel"] = t(self_attention["wq_b"]["kernel"])
      self_attention["wkv"]["kernel"] = t(self_attention["wkv"]["kernel"])
      self_attention["o_a_proj"]["kernel"] = t(self_attention["o_a_proj"]["kernel"])
      self_attention["o_b_proj"]["kernel"] = t(self_attention["o_b_proj"]["kernel"])
      pre_self_attention_layer_norm["scale"] = t(pre_self_attention_layer_norm["scale"])
      post_self_attention_layer_norm["scale"] = t(post_self_attention_layer_norm["scale"])

      if layer_key == "layers_0":
        hca = self_attention["hca_compressor"]
        hca["position_bias"] = t(hca["position_bias"])
        hca["kv_norm"]["scale"] = t(hca["kv_norm"]["scale"])
        hca["gate_proj"]["kernel"] = t(hca["gate_proj"]["kernel"])
        hca["kv_proj"]["kernel"] = t(hca["kv_proj"]["kernel"])
      elif layer_key == "layers_1":
        csa = self_attention["csa_compressor"]
        csa["position_bias"] = t(csa["position_bias"])
        csa["kv_norm"]["scale"] = t(csa["kv_norm"]["scale"])
        csa["gate_proj"]["kernel"] = t(csa["gate_proj"]["kernel"])
        csa["kv_proj"]["kernel"] = t(csa["kv_proj"]["kernel"])
        
        idxr = csa["indexer"]
        idxr["position_bias"] = t(idxr["position_bias"])
        idxr["kv_norm"]["scale"] = t(idxr["kv_norm"]["scale"])
        idxr["gate_proj"]["kernel"] = t(idxr["gate_proj"]["kernel"])
        idxr["kv_proj"]["kernel"] = t(idxr["kv_proj"]["kernel"])
        idxr["weights_proj"]["kernel"] = t(idxr["weights_proj"]["kernel"])
        idxr["q_proj"]["kernel"] = t(idxr["q_proj"]["kernel"])
        
      moe["MoeBlock_0"]["gate"]["kernel"] = t(moe["MoeBlock_0"]["gate"]["kernel"])
      moe["MoeBlock_0"]["gate"]["bias"] = t(moe["MoeBlock_0"]["gate"]["bias"])
      # Remove tid2eid from scanned layers since it's unused
      if "tid2eid" in moe["MoeBlock_0"]:
        del moe["MoeBlock_0"]["tid2eid"]
      
      moe["MoeBlock_0"]["wi_0"] = t(moe["MoeBlock_0"]["wi_0"])
      moe["MoeBlock_0"]["wo"] = t(moe["MoeBlock_0"]["wo"])
      moe["MoeBlock_0"]["wi_1"] = t(moe["MoeBlock_0"]["wi_1"])
      
      moe["shared_experts"]["wi_0"]["kernel"] = t(moe["shared_experts"]["wi_0"]["kernel"])
      moe["shared_experts"]["wo"]["kernel"] = t(moe["shared_experts"]["wo"]["kernel"])
      moe["shared_experts"]["wi_1"]["kernel"] = t(moe["shared_experts"]["wi_1"]["kernel"])

      jax_weights["decoder"]["scanned_blocks"][layer_key] = {
          "self_attention": self_attention,
          "pre_self_attention_layer_norm": pre_self_attention_layer_norm,
          "post_self_attention_layer_norm": post_self_attention_layer_norm,
          "mlp": moe
      }

  del chkpt_vars
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return jax_weights

def _convert_to_jax_weights(base_model_path, model_size, mem_info) -> dict:
  model_params = {
      "num_pre_layers": 3,
      "layers_per_block": 20,
      "num_experts": 256,
  }
  return _convert_huggingface_to_jax_weights(base_model_path, model_params, mem_info)

def main() -> None:
  parser = argparse.ArgumentParser(description="Convert DeepSeek V4 model weights.")
  parser.add_argument("--base_model_path", type=str, required=True)
  parser.add_argument("--maxtext_model_path", type=str, required=True)
  parser.add_argument("--model_size", type=str, required=True)
  parser.add_argument("--dry_run", action="store_true", help="Run without saving the checkpoint")
  args = parser.parse_args()

  mem_info = psutil.Process()
  jax_weights = _convert_to_jax_weights(args.base_model_path, args.model_size, mem_info)

  if args.dry_run:
    max_logging.log("Dry run mode: weights mapped successfully in memory.")
    
    def print_shapes(d, indent=""):
      for k, v in d.items():
        if isinstance(v, dict):
          print(f"{indent}{k}:")
          print_shapes(v, indent + "  ")
        elif isinstance(v, np.ndarray):
          print(f"{indent}{k}: {v.shape}")
          
    print("\n--- MaxText PyTree Shapes ---")
    print("decoder/pre_layers/layers_0:")
    if "layers_0" in jax_weights["decoder"]["pre_layers"]:
      print_shapes(jax_weights["decoder"]["pre_layers"]["layers_0"], "  ")
    print("\ndecoder/scanned_blocks/layers_1 (CSA):")
    if "layers_1" in jax_weights["decoder"]["scanned_blocks"]:
      print_shapes(jax_weights["decoder"]["scanned_blocks"]["layers_1"], "  ")
    return

  save_weights_to_checkpoint(args.maxtext_model_path, jax_weights, 16, True, True)

if __name__ == "__main__":
  main()
