import sys
import pprint
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.common import checkpointing
from maxtext.models import models
import torch
import numpy as np
from typing import List
import os
import json
import gc
from maxtext.checkpoint_conversion.standalone_scripts import llama_or_mistral_ckpt
from maxtext.layers import quantizations
from maxtext.checkpoint_conversion.utils.hf_shape import DEEPSEEK_V4_HF_WEIGHTS_TO_SHAPE

class MockWeightDict:
  def __init__(self, hf_shapes):
    self.hf_shapes = hf_shapes
    self.cache = {}

  def __getitem__(self, key):
    shape = self.hf_shapes[key]
    shape_tuple = tuple(shape)
    if shape_tuple not in self.cache:
      self.cache[shape_tuple] = torch.zeros(shape, dtype=torch.bfloat16)
    return self.cache[shape_tuple]

  def __contains__(self, key):
    return key in self.hf_shapes


Transformer = models.transformer_as_linen

# ==============================================================================
# SCANNED LAYER CONFIGURATION
# ==============================================================================
SCANNED_LAYER_MAPPING = {
    # 1. Standalone/Unscanned layers
    "decoder-layers_0": [0],
    "decoder-layers_1": [1],
    "decoder-layers_2": [2],

    # 2. Scanned Blocks
    # scanned_blocks.layers_0 corresponds to HCA (odd indices 3 to 41)
    "decoder-scanned_blocks-layers_0": list(range(3, 42, 2)),

    # scanned_blocks.layers_1 corresponds to CSA (even indices 4 to 42)
    "decoder-scanned_blocks-layers_1": list(range(4, 43, 2)),
}

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

def get_path_string2(path):
  key_parts = [k.key for k in path if hasattr(k, "key")]
  param_key = ".".join(key_parts)
  return param_key

def get_block_prefix(path_str):
  """Extracts the high-level block prefix for chunking."""
  if path_str.startswith("Tid2EidVar-"):
    path_str = path_str[len("Tid2EidVar-"):]
  elif path_str.startswith("params-"):
    path_str = path_str[len("params-"):]

  if path_str.startswith("decoder-scanned_blocks-layers_"):
    parts = path_str.split("-")
    if len(parts) >= 3:
      return f"{parts[0]}-{parts[1]}-{parts[2]}"
  elif path_str.startswith("decoder-layers_"):
    parts = path_str.split("-")
    if len(parts) >= 2:
      return f"{parts[0]}-{parts[1]}"
  return "GLOBAL"

def valid_match(hf_param):
  if isinstance(hf_param, torch.Tensor):
    return hf_param.to(torch.float32).cpu().numpy()
  return hf_param

def transposed_match(hf_param):
  return valid_match(hf_param).transpose()

def split_attention_projection_match(hf_param, num_heads):
  hf_param = hf_param.to(torch.float32)
  total_out, embed_dim = hf_param.shape
  head_dim = total_out // num_heads
  jax_param = hf_param.reshape(num_heads, head_dim, embed_dim).permute(0, 2, 1).cpu().numpy()
  return jax_param

def stacked_moe_match(weight_dict, hf_prefix, num_experts, suffix):
  params = []
  for i in range(num_experts):
    key = f"{hf_prefix}.experts.{i}.{suffix}"
    p = valid_match(weight_dict[key])
    params.append(p.transpose())
  return np.stack(params, axis=0)

def get_unscanned_weight(hf_prefix, suffix, weight_dict, config):
  """Extracts a single un-scanned layer weight. Extracted from original map_fn."""
  num_experts = config.num_experts

  # Normalize DeepSeek V4 compressor prefixes to legacy format
  if suffix.startswith("self_attention-csa_compressor-"):
    suffix = suffix.replace("self_attention-csa_compressor-", "self_attention-compressor-")
  elif suffix.startswith("self_attention-hca_compressor-"):
    suffix = suffix.replace("self_attention-hca_compressor-", "self_attention-compressor-")

  # Normalize q_proj to q_b_proj for indexer
  if "indexer-q_proj-kernel" in suffix:
    suffix = suffix.replace("indexer-q_proj-kernel", "indexer-q_b_proj-kernel")

  # mHC Norms (unweighted in HF, so we initialize to ones)
  if suffix == "mhc_attention-mhc_norm-scale":
    return np.ones((config.base_emb_dim * config.mhc_expansion_rate,), dtype=np.float32)
  if suffix == "mhc_mlp-mhc_norm-scale":
    return np.ones((config.base_emb_dim * config.mhc_expansion_rate,), dtype=np.float32)

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
  if suffix in ("self_attention-wq_a-kernel", "self_attention-q_a_proj-kernel") and f"{hf_prefix}.attn.wq_a.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.wq_a.weight"])
  if suffix in ("self_attention-wq_b-kernel", "self_attention-q_b_proj-kernel") and f"{hf_prefix}.attn.wq_b.weight" in weight_dict:
    w = transposed_match(weight_dict[f"{hf_prefix}.attn.wq_b.weight"])
    return w.reshape(config.q_lora_rank, config.num_query_heads, config.head_dim)
  if suffix in ("self_attention-q_norm-scale", "self_attention-q_a_norm-scale") and f"{hf_prefix}.attn.q_norm.weight" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.attn.q_norm.weight"])
  if suffix in ("self_attention-wkv-kernel", "self_attention-kv_proj-kernel") and f"{hf_prefix}.attn.wkv.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.wkv.weight"])
  if suffix == "self_attention-kv_norm-scale" and f"{hf_prefix}.attn.kv_norm.weight" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.attn.kv_norm.weight"])
  if suffix in ("self_attention-wo_a-kernel", "self_attention-o_a_proj-kernel") and f"{hf_prefix}.attn.wo_a.weight" in weight_dict:
    o_groups = getattr(config, "o_groups", 8)
    return split_attention_projection_match(weight_dict[f"{hf_prefix}.attn.wo_a.weight"], o_groups)
  if suffix in ("self_attention-wo_b-kernel", "self_attention-o_b_proj-kernel") and f"{hf_prefix}.attn.wo_b.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.wo_b.weight"])
  if suffix == "self_attention-sinks" and f"{hf_prefix}.attn.attn_sink" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.attn.attn_sink"])

  # Compressor / Indexer
  if suffix == "self_attention-compressor-position_bias" and f"{hf_prefix}.attn.compressor.ape" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.attn.compressor.ape"])
  if suffix == "self_attention-compressor-kv_norm-scale" and f"{hf_prefix}.attn.compressor.norm.weight" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.attn.compressor.norm.weight"])
  if suffix == "self_attention-compressor-gate_proj-kernel" and f"{hf_prefix}.attn.compressor.wgate.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.compressor.wgate.weight"])
  if suffix == "self_attention-compressor-kv_proj-kernel" and f"{hf_prefix}.attn.compressor.wkv.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.compressor.wkv.weight"])
  if suffix == "self_attention-compressor-indexer-q_b_proj-kernel" and f"{hf_prefix}.attn.indexer.wq_b.weight" in weight_dict:
    w = transposed_match(weight_dict[f"{hf_prefix}.attn.indexer.wq_b.weight"])
    return w.reshape(config.q_lora_rank, config.indexer_n_heads, config.indexer_head_dim)
  if suffix == "self_attention-compressor-indexer-position_bias" and f"{hf_prefix}.attn.indexer.compressor.ape" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.attn.indexer.compressor.ape"])
  if suffix == "self_attention-compressor-indexer-kv_norm-scale" and f"{hf_prefix}.attn.indexer.compressor.norm.weight" in weight_dict: return valid_match(weight_dict[f"{hf_prefix}.attn.indexer.compressor.norm.weight"])
  if suffix == "self_attention-compressor-indexer-gate_proj-kernel" and f"{hf_prefix}.attn.indexer.compressor.wgate.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.indexer.compressor.wgate.weight"])
  if suffix == "self_attention-compressor-indexer-kv_proj-kernel" and f"{hf_prefix}.attn.indexer.compressor.wkv.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.indexer.compressor.wkv.weight"])
  if suffix == "self_attention-compressor-indexer-weights_proj-kernel" and f"{hf_prefix}.attn.indexer.weights_proj.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.attn.indexer.weights_proj.weight"])

  # MLP Experts
  if suffix == "mlp-MoeBlock_0-gate-kernel" and f"{hf_prefix}.ffn.gate.weight" in weight_dict:
    return transposed_match(weight_dict[f"{hf_prefix}.ffn.gate.weight"])
  if suffix == "mlp-MoeBlock_0-gate-e_score_correction_bias" and f"{hf_prefix}.ffn.gate.bias" in weight_dict:
    return valid_match(weight_dict[f"{hf_prefix}.ffn.gate.bias"])
  if suffix == "mlp-MoeBlock_0-wi_0":
    return stacked_moe_match(weight_dict, f"{hf_prefix}.ffn", num_experts, "w3.weight")
  if suffix == "mlp-MoeBlock_0-wi_1":
    return stacked_moe_match(weight_dict, f"{hf_prefix}.ffn", num_experts, "w1.weight")
  if suffix == "mlp-MoeBlock_0-wo":
    return stacked_moe_match(weight_dict, f"{hf_prefix}.ffn", num_experts, "w2.weight")

  # MLP Shared Experts
  if suffix == "mlp-shared_experts-wi_0-kernel" and f"{hf_prefix}.ffn.shared_experts.w1.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.ffn.shared_experts.w1.weight"])
  if suffix == "mlp-shared_experts-wi_1-kernel" and f"{hf_prefix}.ffn.shared_experts.w3.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.ffn.shared_experts.w3.weight"])
  if suffix == "mlp-shared_experts-wi-kernel" and f"{hf_prefix}.ffn.shared_experts.w3.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.ffn.shared_experts.w3.weight"])
  if suffix == "mlp-shared_experts-wi_up-kernel" and f"{hf_prefix}.ffn.shared_experts.w1.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.ffn.shared_experts.w1.weight"])
  if suffix == "mlp-shared_experts-wo-kernel" and f"{hf_prefix}.ffn.shared_experts.w2.weight" in weight_dict: return transposed_match(weight_dict[f"{hf_prefix}.ffn.shared_experts.w2.weight"])

  return None

def convert_weights(abstract_params, weight_dict, config, block_to_convert=None):
  """Maps from MaxText abstract parameters to actual weights from HF."""

  def map_fn(path, abstract_var):
    path_str = get_path_string(path)
    
    is_tid2eid = False
    if path_str.startswith("Tid2EidVar-"):
      is_tid2eid = True
      path_str = path_str[len("Tid2EidVar-"):]
    elif path_str.startswith("params-"):
      path_str = path_str[len("params-"):]

    block_prefix = get_block_prefix(path_str)

    # Filter out paths that don't belong to the current block
    if block_to_convert is not None and block_prefix != block_to_convert:
      return None

    if is_tid2eid:
      return np.zeros(abstract_var.shape, dtype=np.float32)

    # 1. Global / Unscanned base params
    if block_prefix == "GLOBAL":
      if path_str == "token_embedder-embedding" and "embed.weight" in weight_dict: return valid_match(weight_dict["embed.weight"])
      if path_str == "decoder-logits_dense-kernel" and "head.weight" in weight_dict: return transposed_match(weight_dict["head.weight"])
      if path_str == "decoder-decoder_norm-scale" and "norm.weight" in weight_dict: return valid_match(weight_dict["norm.weight"])
      if path_str == "decoder-hc_head-hc_fn" and "hc_head_fn" in weight_dict: return transposed_match(weight_dict["hc_head_fn"])
      if path_str == "decoder-hc_head-hc_base" and "hc_head_base" in weight_dict: return valid_match(weight_dict["hc_head_base"])
      if path_str == "decoder-hc_head-hc_scale" and "hc_head_scale" in weight_dict: return valid_match(weight_dict["hc_head_scale"])
      raise ValueError(f"CRITICAL: Unmapped Global MaxText parameter '{path_str}'.")

    # 2. Scanned Layer params
    if block_prefix in SCANNED_LAYER_MAPPING:
      hf_indices = SCANNED_LAYER_MAPPING[block_prefix]
      suffix = path_str[len(block_prefix)+1:] # Strip prefix (e.g. "decoder-layers-layers_1-")

      stacked_weights = []
      for layer_idx in hf_indices:
        print(f"handling {block_prefix} layer {layer_idx}")
        hf_prefix = f"layers.{layer_idx}"
        w = get_unscanned_weight(hf_prefix, suffix, weight_dict, config)
        if w is None:
          raise ValueError(f"CRITICAL: Unmapped suffix '{suffix}' for HF layer {layer_idx}")
        stacked_weights.append(w)

      # Initially stack along 0-th dimension
      stacked_np = np.stack(stacked_weights, axis=0)
      if len(hf_indices) == 1:
        stacked_np = np.squeeze(stacked_np, axis=0)

      # DYNAMIC PERMUTATION: MaxText scan axes aren't always axis 0.
      # e.g., kv_proj-kernel might expect [4096, 20, 512].
      target_shape = abstract_var.shape
      if stacked_np.shape != target_shape:
        u_shape = stacked_np.shape[1:]
        N = stacked_np.shape[0]
        matched = False

        # Find the correct insertion point for the scan dimension
        for i in range(len(u_shape) + 1):
          proposed_shape = u_shape[:i] + (N,) + u_shape[i:]
          if proposed_shape == target_shape:
            stacked_np = np.moveaxis(stacked_np, 0, i)
            matched = True
            break

        if not matched:
          # Fallback to pure reshape if moveaxis fails
          if stacked_np.size == np.prod(target_shape):
            stacked_np = stacked_np.reshape(target_shape)
          else:
            raise ValueError(f"Shape mismatch for {path_str}. Stacked: {stacked_np.shape}, Target: {target_shape}")
      print("done with scanned weight stacking")
      return stacked_np

    raise ValueError(f"CRITICAL: Unmapped Scanned MaxText prefix '{block_prefix}' in '{path_str}'. Ensure it is added to SCANNED_LAYER_MAPPING.")

  params = jax.tree_util.tree_map_with_path(map_fn, abstract_params)

  def finalize(p, v):
    if p is None:
      return None
    return jax.device_put(jnp.asarray(p, dtype=v.dtype), jax.devices("cpu")[0])

  return jax.tree_util.tree_map(finalize, params, abstract_params, is_leaf=lambda x: x is None)

def main():
  argv = sys.argv
  if len(argv) < 2:
    argv = ['', 'src/maxtext/configs/base.yml', 'model_name=deepseek4-284b', 'override_model_config=True', 'attention=dot_product', 'skip_jax_distributed_system=True', 'weight_dtype=bfloat16', 'scan_layers=True', 'base_output_directory=/home/snehalv_google_com/mock_scanned_checkpoint/']

  print("Initializing configuration...")
  config = pyconfig.initialize(argv)
  print(f"\n--- Running Mock MaxText Conversion: {config.model_name} (Scan: {config.scan_layers}) ---")
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh=mesh, quant=quant)

  abstract_params = maxtext_utils.get_abstract_param(model, config)

  hf_weights_dir = "/home/snehalv_google_com/test-ckpt-bf16"
  config_path = os.path.join(hf_weights_dir, "config.json")

  if os.path.exists(config_path):
    print(f"Mocking HF weights from {config_path} using zeros...")
    from maxtext.checkpoint_conversion.utils.hf_model_configs import deepseek_v4_flash_dict
    with open(config_path, "r") as f:
      raw_config = json.load(f)
    
    hf_config = {**deepseek_v4_flash_dict, **raw_config}
    
    if "n_routed_experts" in hf_config:
      hf_config["num_experts"] = hf_config["n_routed_experts"]
    if "n_shared_experts" in hf_config:
      hf_config["shared_experts"] = hf_config["n_shared_experts"]

    # Overwrite HF expert count with JAX config for fast testing
    hf_config["num_experts"] = config.num_experts

    hf_shapes = DEEPSEEK_V4_HF_WEIGHTS_TO_SHAPE(hf_config)
    full_in_memory_dict = MockWeightDict(hf_shapes)

    block_prefixes = set()
    def find_blocks(path, var):
      block_prefixes.add(get_block_prefix(get_path_string(path)))
    jax.tree_util.tree_map_with_path(find_blocks, abstract_params)
    sorted_blocks = sorted(list(block_prefixes))

    print(f"Found {len(sorted_blocks)} logical blocks to convert: {sorted_blocks}")
    final_params = abstract_params

    def process_block(block_prefix):
      print(f"Starting block {block_prefix}...")
      converted = convert_weights(abstract_params, full_in_memory_dict, config, block_to_convert=block_prefix)
      print(f"Finished {block_prefix} conversion")
      return block_prefix, converted

    print("Starting sequential conversion on Zeros...")

    for block_prefix in sorted_blocks:
      try:
        _, converted_tree = process_block(block_prefix)
        print(f"Merging block {block_prefix} into final checkpoint tree...")

        def merge_fn(path, f_val, c_val):
          if get_block_prefix(get_path_string(path)) == block_prefix:
            return c_val
          return f_val

        final_params = jax.tree_util.tree_map_with_path(merge_fn, final_params, converted_tree)
        del converted_tree
        gc.collect()

      except Exception as exc:
        print(f"Block {block_prefix} generated an exception: {exc}")
        raise exc

    flat_params = jax.tree_util.tree_flatten_with_path(final_params)[0]
    nested_zeros_dict = {}

    for path, x in flat_params:
      name = get_path_string2(path)
      parts = name.split('.')
      current_level = nested_zeros_dict
      for part in parts[:-1]:
        if part not in current_level:
          current_level[part] = {}
        current_level = current_level[part]
      current_level[parts[-1]] = x
    
    print(f"\nSaving converted checkpoint to {config.checkpoint_dir}...")
    llama_or_mistral_ckpt.save_weights_to_checkpoint(
        config.checkpoint_dir,
        nested_zeros_dict,
        mesh.size,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
        checkpoint_storage_concurrent_gb=config.checkpoint_storage_concurrent_gb,
    )
    print("Conversion complete.")
  else:
    print(f"HF config file not found at {config_path}. Skipping.")

  print("Script finished.")

if __name__ == "__main__":
  main()
