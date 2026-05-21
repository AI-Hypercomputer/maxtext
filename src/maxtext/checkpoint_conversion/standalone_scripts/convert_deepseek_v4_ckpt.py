import sys
import pprint
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.common import checkpointing
import torch
import numpy as np
from typing import List
import os
import json

def get_path_string(path):
  keys = []
  for p in path:
    if hasattr(p, "key"):
      keys.append(str(p.key))
    elif hasattr(p, "idx"):
      keys.append(str(p.idx))
    else:
      keys.append(str(p).strip("'\""))
  if len(keys) > 0 and keys[-1][0] == '.':
    keys.pop()
  return "-".join(keys)

def valid_match(hf_param):
  """Identical shapes"""
  if isinstance(hf_param, torch.Tensor):
    return hf_param.to(torch.float32).cpu().numpy()
  return hf_param

def transposed_match(hf_param):
  """Transposed shapes"""
  return valid_match(hf_param).transpose()

def split_attention_projection_match(hf_param, num_heads):
  """Reshape and transpose for attention projections (e.g. wo_a)."""
  hf_param = hf_param.to(torch.float32)
  total_out, embed_dim = hf_param.shape
  head_dim = total_out // num_heads
  # hf_param: (num_heads * head_dim, embed_dim)
  # reshape to (num_heads, head_dim, embed_dim)
  # transpose to (num_heads, embed_dim, head_dim)
  jax_param = hf_param.reshape(num_heads, head_dim, embed_dim).permute(0, 2, 1).cpu().numpy()
  return jax_param

def stacked_moe_match(weight_dict, hf_prefix, num_experts, suffix):
  """Stacks all expert weights from hf weight dict into one array."""
  params = []
  for i in range(num_experts):
    key = f"{hf_prefix}.experts.{i}.{suffix}"
    p = weight_dict[key].to(torch.float32).cpu().numpy()
    # DeepSeek expert weights are (out, in) in HF.
    # w1, w3 are (2048, 4096) -> MT (4096, 2048)
    # w2 is (4096, 2048) -> MT (2048, 4096)
    # Both need transpose.
    params.append(p.transpose())
  return np.stack(params, axis=0)

def convert_weights(abstract_params, weight_dict, config):
  """Maps from MaxText abstract parameters to actual weights from HF."""
  num_experts = config.num_experts
  num_query_heads = config.num_query_heads if config.num_query_heads is not None else config.num_attention_heads

  def map_fn(path, abstract_var):
    path_str = get_path_string(path)
    
    # 1. Global params
    if path_str == "token_embedder-embedding" and "embed.weight" in weight_dict:
      return valid_match(weight_dict["embed.weight"])
    if path_str == "decoder-logits_dense-kernel" and "head.weight" in weight_dict:
      return transposed_match(weight_dict["head.weight"])
    if path_str == "decoder-hc_head-hc_fn" and "hc_head_fn" in weight_dict:
      return transposed_match(weight_dict["hc_head_fn"])
    if path_str == "decoder-hc_head-hc_base" and "hc_head_base" in weight_dict:
      return valid_match(weight_dict["hc_head_base"])
    if path_str == "decoder-hc_head-hc_scale" and "hc_head_scale" in weight_dict:
      return valid_match(weight_dict["hc_head_scale"])
      
    # 2. Layer-specific params
    parts = path_str.split("-")
    if len(parts) >= 3 and parts[0] == "decoder" and parts[1] == "layers":
      layer_idx = int(parts[2])
      hf_prefix = f"layers.{layer_idx}"
      suffix = "-".join(parts[3:])
      
      # RMSNorms
      if suffix == "pre_self_attention_layer_norm-scale" and f"{hf_prefix}.attn_norm.weight" in weight_dict:
        return valid_match(weight_dict[f"{hf_prefix}.attn_norm.weight"])
      if suffix == "post_self_attention_layer_norm-scale" and f"{hf_prefix}.ffn_norm.weight" in weight_dict:
        return valid_match(weight_dict[f"{hf_prefix}.ffn_norm.weight"])
        
      # mHC Attention
      if suffix.startswith("mhc_attention"):
        hf_key_fn = f"{hf_prefix}.hc_attn_fn"
        hf_key_base = f"{hf_prefix}.hc_attn_base"
        hf_key_scale = f"{hf_prefix}.hc_attn_scale"
        if hf_key_fn in weight_dict:
          if suffix == "mhc_attention-pre_alpha": return valid_match(weight_dict[hf_key_fn][0:4, :].T)
          if suffix == "mhc_attention-post_alpha": return valid_match(weight_dict[hf_key_fn][4:8, :].T)
          if suffix == "mhc_attention-res_alpha": return valid_match(weight_dict[hf_key_fn][8:24, :].T)
        if hf_key_base in weight_dict:
          if suffix == "mhc_attention-pre_beta": return valid_match(weight_dict[hf_key_base][0:4])
          if suffix == "mhc_attention-post_beta": return valid_match(weight_dict[hf_key_base][4:8])
          if suffix == "mhc_attention-res_beta": return valid_match(weight_dict[hf_key_base][8:24].reshape(4, 4))
        if hf_key_scale in weight_dict:
          if suffix == "mhc_attention-pre_alpha_scale": return valid_match(weight_dict[hf_key_scale][0:1])
          if suffix == "mhc_attention-post_alpha_scale": return valid_match(weight_dict[hf_key_scale][1:2])
          if suffix == "mhc_attention-res_alpha_scale": return valid_match(weight_dict[hf_key_scale][2:3])

      # mHC MLP
      if suffix.startswith("mhc_mlp"):
        hf_key_fn = f"{hf_prefix}.hc_ffn_fn"
        hf_key_base = f"{hf_prefix}.hc_ffn_base"
        hf_key_scale = f"{hf_prefix}.hc_ffn_scale"
        if hf_key_fn in weight_dict:
          if suffix == "mhc_mlp-pre_alpha": return valid_match(weight_dict[hf_key_fn][0:4, :].T)
          if suffix == "mhc_mlp-post_alpha": return valid_match(weight_dict[hf_key_fn][4:8, :].T)
          if suffix == "mhc_mlp-res_alpha": return valid_match(weight_dict[hf_key_fn][8:24, :].T)
        if hf_key_base in weight_dict:
          if suffix == "mhc_mlp-pre_beta": return valid_match(weight_dict[hf_key_base][0:4])
          if suffix == "mhc_mlp-post_beta": return valid_match(weight_dict[hf_key_base][4:8])
          if suffix == "mhc_mlp-res_beta": return valid_match(weight_dict[hf_key_base][8:24].reshape(4, 4))
        if hf_key_scale in weight_dict:
          if suffix == "mhc_mlp-pre_alpha_scale": return valid_match(weight_dict[hf_key_scale][0:1])
          if suffix == "mhc_mlp-post_alpha_scale": return valid_match(weight_dict[hf_key_scale][1:2])
          if suffix == "mhc_mlp-res_alpha_scale": return valid_match(weight_dict[hf_key_scale][2:3])

      # Self Attention
      if suffix == "self_attention-q_a_proj-kernel" and f"{hf_prefix}.attn.wq_a.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.wq_a.weight"])
      if suffix == "self_attention-q_b_proj-kernel" and f"{hf_prefix}.attn.wq_b.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.wq_b.weight"])
      if suffix == "self_attention-q_a_norm-scale" and f"{hf_prefix}.attn.q_norm.weight" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.attn.q_norm.weight"])
      if suffix == "self_attention-kv_proj-kernel" and f"{hf_prefix}.attn.wkv.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.wkv.weight"])
      if suffix == "self_attention-kv_norm-scale" and f"{hf_prefix}.attn.kv_norm.weight" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.attn.kv_norm.weight"])
      if suffix == "self_attention-o_a_proj-kernel" and f"{hf_prefix}.attn.wo_a.weight" in weight_dict: 
        o_groups = getattr(config, "o_groups", 8)
        return split_attention_projection_match(weight_dict[f"{hf_prefix}.attn.wo_a.weight"], o_groups)
      if suffix == "self_attention-o_b_proj-kernel" and f"{hf_prefix}.attn.wo_b.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.wo_b.weight"])
      if suffix == "self_attention-sinks" and f"{hf_prefix}.attn.attn_sink" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.attn.attn_sink"])

      # Compressor / Indexer
      if suffix == "self_attention-compressor-position_bias" and f"{hf_prefix}.attn.compressor.ape" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.attn.compressor.ape"])
      if suffix == "self_attention-compressor-kv_norm-scale" and f"{hf_prefix}.attn.compressor.norm.weight" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.attn.compressor.norm.weight"])
      if suffix == "self_attention-compressor-gate_proj-kernel" and f"{hf_prefix}.attn.compressor.wgate.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.compressor.wgate.weight"])
      if suffix == "self_attention-compressor-kv_proj-kernel" and f"{hf_prefix}.attn.compressor.wkv.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.compressor.wkv.weight"])
      if suffix == "self_attention-compressor-indexer-q_b_proj-kernel" and f"{hf_prefix}.attn.indexer.wq_b.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.indexer.wq_b.weight"])
      if suffix == "self_attention-compressor-indexer-position_bias" and f"{hf_prefix}.attn.indexer.compressor.ape" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.attn.indexer.compressor.ape"])
      if suffix == "self_attention-compressor-indexer-kv_norm-scale" and f"{hf_prefix}.attn.indexer.compressor.norm.weight" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.attn.indexer.compressor.norm.weight"])
      if suffix == "self_attention-compressor-indexer-gate_proj-kernel" and f"{hf_prefix}.attn.indexer.compressor.wgate.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.indexer.compressor.wgate.weight"])
      if suffix == "self_attention-compressor-indexer-kv_proj-kernel" and f"{hf_prefix}.attn.indexer.compressor.wkv.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.indexer.compressor.wkv.weight"])
      if suffix == "self_attention-compressor-indexer-weights_proj-kernel" and f"{hf_prefix}.attn.indexer.weights_proj.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.indexer.weights_proj.weight"])

      # experts
      if suffix == "mlp-MoeBlock_0-gate-kernel" and f"{hf_prefix}.ffn.gate.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.ffn.gate.weight"])
      if suffix == "mlp-MoeBlock_0-gate-tid2eid" and f"{hf_prefix}.ffn.gate.tid2eid" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.ffn.gate.tid2eid"])
      if suffix == "mlp-MoeBlock_0-gate-e_score_correction_bias" and f"{hf_prefix}.ffn.gate.e_score_correction_bias" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.ffn.gate.e_score_correction_bias"])
      if suffix == "mlp-MoeBlock_0-wi_0":
        try: return stacked_moe_match(weight_dict, f"{hf_prefix}.ffn", num_experts, "w1.weight")
        except KeyError: pass
      if suffix == "mlp-MoeBlock_0-wi_1":
        try: return stacked_moe_match(weight_dict, f"{hf_prefix}.ffn", num_experts, "w3.weight")
        except KeyError: pass
      if suffix == "mlp-MoeBlock_0-wo":
        try: return stacked_moe_match(weight_dict, f"{hf_prefix}.ffn", num_experts, "w2.weight")
        except KeyError: pass

      # shared
      if suffix == "mlp-shared_experts-wi-kernel" and f"{hf_prefix}.ffn.shared_experts.w1.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.ffn.shared_experts.w1.weight"])
      if suffix == "mlp-shared_experts-wo-kernel" and f"{hf_prefix}.ffn.shared_experts.w2.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.ffn.shared_experts.w2.weight"])

    return np.zeros(abstract_var.shape, dtype=np.float32)

  params = jax.tree_util.tree_map_with_path(map_fn, abstract_params)
  
  def finalize(p, v):
    arr = jnp.asarray(p, dtype=v.dtype)
    if hasattr(v, 'sharding') and v.sharding is not None:
      arr = jax.device_put(arr, v.sharding)
    return arr
    
  return jax.tree_util.tree_map(finalize, params, abstract_params)

def main():
  # Initialize the configuration for deepseek_v4-flash
  argv = sys.argv
  if len(argv) < 2:
    argv = ['', 'src/maxtext/configs/base.yml', 'model_name=deepseek_v4-flash', 'override_model_config=True', 'attention=dot_product', 'skip_jax_distributed_system=True', 'weight_dtype=bfloat16', 'scan_layers=False', 'num_experts=256']
  
  print("Initializing configuration...")
  config = pyconfig.initialize(argv)
  
  print("Creating device mesh...")
  mesh = maxtext_utils.get_mesh_from_config(config)
  
  print("Creating model architecture...")
  model = model_creation_utils.create_model(config, mesh)
  
  print("Getting abstract parameters...")
  abstract_vars = maxtext_utils.get_abstract_param(model, config)
  abstract_params = abstract_vars["params"]

  hf_weights_dir = "/mnt/disks/external_disk/ds-v4-bf16/"
  index_path = os.path.join(hf_weights_dir, "model.safetensors.index.json")
  
  if os.path.exists(index_path):
    from safetensors.torch import load_file as load_safetensors
    print(f"Loading HF weights from {hf_weights_dir}...")
    with open(index_path, "r") as f:
      index = json.load(f)
    
    weight_map = index["weight_map"]
    all_files = sorted(list(set(weight_map.values())))
    
    print(f"Found {len(all_files)} files. Loading all weights into memory (Warning: High memory usage)...")
    full_weight_dict = {}
    
    for i, filename in enumerate(all_files):
      print(f"[{i+1}/{len(all_files)}] Loading {filename}...")
      file_path = os.path.join(hf_weights_dir, filename)
      weights = load_safetensors(file_path)
      full_weight_dict.update(weights)

    print("Converting weights...")
    converted_params = convert_weights(abstract_params, full_weight_dict, config)
    
    print(f"\nSaving converted checkpoint to {config.checkpoint_dir}...")
    checkpointing.save_params_to_path(
        config.checkpoint_dir,
        converted_params,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3
    )
    print("Conversion complete.")
  else:
    print(f"Index file not found at {index_path}. Skipping actual conversion.")
  
  print("Script finished.")

if __name__ == "__main__":
  main()
