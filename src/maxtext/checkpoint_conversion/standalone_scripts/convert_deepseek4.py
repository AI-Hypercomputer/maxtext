import os
import sys
import numpy as np
import jax.numpy as jnp
from safetensors.numpy import load_file
import orbax.checkpoint as ocp

def DEEPSEEKV4_MAXTEXT_TO_HF_PARAM_MAPPING(n_layers, num_experts):
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "head.weight",
  }
  
  for i in range(n_layers):
      mapping.update({
          f"params-decoder-layers_{i}-pre_self_attention_layer_norm-scale": f"model.layers.{i}.input_layernorm.weight",
          f"params-decoder-layers_{i}-post_self_attention_layer_norm-scale": f"model.layers.{i}.post_attention_layernorm.weight",
          
          # Attention
          f"params-decoder-layers_{i}-self_attention-wq_a-kernel": f"model.layers.{i}.self_attn.q_a_proj.weight",
          f"params-decoder-layers_{i}-self_attention-q_norm-scale": f"model.layers.{i}.self_attn.q_a_norm.weight",
          f"params-decoder-layers_{i}-self_attention-wq_b-kernel": f"model.layers.{i}.self_attn.q_b_proj.weight",
          f"params-decoder-layers_{i}-self_attention-wkv-kernel": f"model.layers.{i}.self_attn.kv_proj.weight",
          f"params-decoder-layers_{i}-self_attention-kv_norm-scale": f"model.layers.{i}.self_attn.kv_norm.weight",
          f"params-decoder-layers_{i}-self_attention-sinks": f"model.layers.{i}.self_attn.sinks",
          f"params-decoder-layers_{i}-self_attention-o_a_proj-kernel": f"model.layers.{i}.self_attn.o_a_proj.weight",
          f"params-decoder-layers_{i}-self_attention-o_b_proj-kernel": f"model.layers.{i}.self_attn.o_b_proj.weight",
          
          # mHC Attention
          f"params-decoder-layers_{i}-mhc_attention-mhc_norm-scale": None,
          f"params-decoder-layers_{i}-mhc_attention-pre_alpha": f"model.layers.{i}.attn_hc.fn",
          f"params-decoder-layers_{i}-mhc_attention-post_alpha": f"model.layers.{i}.attn_hc.fn",
          f"params-decoder-layers_{i}-mhc_attention-res_alpha": f"model.layers.{i}.attn_hc.fn",
          f"params-decoder-layers_{i}-mhc_attention-pre_beta": f"model.layers.{i}.attn_hc.base",
          f"params-decoder-layers_{i}-mhc_attention-post_beta": f"model.layers.{i}.attn_hc.base",
          f"params-decoder-layers_{i}-mhc_attention-res_beta": f"model.layers.{i}.attn_hc.base",
          f"params-decoder-layers_{i}-mhc_attention-pre_alpha_scale": f"model.layers.{i}.attn_hc.scale",
          f"params-decoder-layers_{i}-mhc_attention-post_alpha_scale": f"model.layers.{i}.attn_hc.scale",
          f"params-decoder-layers_{i}-mhc_attention-res_alpha_scale": f"model.layers.{i}.attn_hc.scale",
          
          # mHC MLP
          f"params-decoder-layers_{i}-mhc_mlp-mhc_norm-scale": None,
          f"params-decoder-layers_{i}-mhc_mlp-pre_alpha": f"model.layers.{i}.ffn_hc.fn",
          f"params-decoder-layers_{i}-mhc_mlp-post_alpha": f"model.layers.{i}.ffn_hc.fn",
          f"params-decoder-layers_{i}-mhc_mlp-res_alpha": f"model.layers.{i}.ffn_hc.fn",
          f"params-decoder-layers_{i}-mhc_mlp-pre_beta": f"model.layers.{i}.ffn_hc.base",
          f"params-decoder-layers_{i}-mhc_mlp-post_beta": f"model.layers.{i}.ffn_hc.base",
          f"params-decoder-layers_{i}-mhc_mlp-res_beta": f"model.layers.{i}.ffn_hc.base",
          f"params-decoder-layers_{i}-mhc_mlp-pre_alpha_scale": f"model.layers.{i}.ffn_hc.scale",
          f"params-decoder-layers_{i}-mhc_mlp-post_alpha_scale": f"model.layers.{i}.ffn_hc.scale",
          f"params-decoder-layers_{i}-mhc_mlp-res_alpha_scale": f"model.layers.{i}.ffn_hc.scale",
      })
      
      # MoE Block
      if i < 3:
          mapping[f"Tid2EidVar-decoder-layers_{i}-mlp-MoeBlock_0-tid2eid"] = f"model.layers.{i}.mlp.gate.tid2eid"
      
      mapping.update({
          f"params-decoder-layers_{i}-mlp-MoeBlock_0-gate-kernel": f"model.layers.{i}.mlp.gate.weight",
          
          # Shared Experts
          f"params-decoder-layers_{i}-mlp-shared_experts-wi_0-kernel": f"model.layers.{i}.mlp.shared_experts.gate_proj.weight",
          f"params-decoder-layers_{i}-mlp-shared_experts-wi_1-kernel": f"model.layers.{i}.mlp.shared_experts.up_proj.weight",
          f"params-decoder-layers_{i}-mlp-shared_experts-wo-kernel": f"model.layers.{i}.mlp.shared_experts.down_proj.weight",
          
          # Stacked Experts
          f"params-decoder-layers_{i}-mlp-MoeBlock_0-wi_0": [f"model.layers.{i}.mlp.experts.{e}.w1.weight" for e in range(num_experts)],
          f"params-decoder-layers_{i}-mlp-MoeBlock_0-wi_1": [f"model.layers.{i}.mlp.experts.{e}.w3.weight" for e in range(num_experts)],
          f"params-decoder-layers_{i}-mlp-MoeBlock_0-wo": [f"model.layers.{i}.mlp.experts.{e}.w2.weight" for e in range(num_experts)],
      })
      
      if i >= 3:
          mapping[f"params-decoder-layers_{i}-mlp-MoeBlock_0-gate-bias"] = f"model.layers.{i}.mlp.gate.e_score_correction_bias"
          
      # Attention Compressor
      if i >= 2:
          if i % 2 == 0 or i == 2:
              # CSA
              mapping.update({
                  f"params-decoder-layers_{i}-self_attention-csa_compressor-kv_proj-kernel": f"model.layers.{i}.self_attn.compressor.kv_proj.weight",
                  f"params-decoder-layers_{i}-self_attention-csa_compressor-gate_proj-kernel": f"model.layers.{i}.self_attn.compressor.gate_proj.weight",
                  f"params-decoder-layers_{i}-self_attention-csa_compressor-position_bias": f"model.layers.{i}.self_attn.compressor.position_bias",
                  f"params-decoder-layers_{i}-self_attention-csa_compressor-kv_norm-scale": f"model.layers.{i}.self_attn.compressor.kv_norm.weight",
                  
                  # CSA Indexer
                  f"params-decoder-layers_{i}-self_attention-csa_compressor-indexer-gate_proj-kernel": f"model.layers.{i}.self_attn.compressor.indexer.gate_proj.weight",
                  f"params-decoder-layers_{i}-self_attention-csa_compressor-indexer-kv_proj-kernel": f"model.layers.{i}.self_attn.compressor.indexer.kv_proj.weight",
                  f"params-decoder-layers_{i}-self_attention-csa_compressor-indexer-q_proj-kernel": f"model.layers.{i}.self_attn.compressor.indexer.q_b_proj.weight",
                  f"params-decoder-layers_{i}-self_attention-csa_compressor-indexer-weights_proj-kernel": f"model.layers.{i}.self_attn.compressor.indexer.scorer.weights_proj.weight",
                  f"params-decoder-layers_{i}-self_attention-csa_compressor-indexer-position_bias": f"model.layers.{i}.self_attn.compressor.indexer.position_bias",
                  f"params-decoder-layers_{i}-self_attention-csa_compressor-indexer-kv_norm-scale": f"model.layers.{i}.self_attn.compressor.indexer.kv_norm.weight",
              })
          else:
              # HCA
              mapping.update({
                  f"params-decoder-layers_{i}-self_attention-hca_compressor-kv_proj-kernel": f"model.layers.{i}.self_attn.compressor.kv_proj.weight",
                  f"params-decoder-layers_{i}-self_attention-hca_compressor-gate_proj-kernel": f"model.layers.{i}.self_attn.compressor.gate_proj.weight",
                  f"params-decoder-layers_{i}-self_attention-hca_compressor-position_bias": f"model.layers.{i}.self_attn.compressor.position_bias",
                  f"params-decoder-layers_{i}-self_attention-hca_compressor-kv_norm-scale": f"model.layers.{i}.self_attn.compressor.kv_norm.weight",
              })
              
  return mapping

def DEEPSEEKV4_MAXTEXT_TO_HF_PARAM_HOOK_FN(n_heads):
  def transpose(input_tensor, target_shape=None):
    return np.transpose(input_tensor)

  def transpose_stack(input_tensor, target_shape=None):
    stacked = np.stack(input_tensor, axis=0) # [E, out, in]
    return np.transpose(stacked, (0, 2, 1))  # [E, in, out]
    
  def ones_norm(input_tensor, target_shape=None):
    # mhc_norm_scale target shape is k * d = 4 * 4096 = 16384
    return np.ones(16384, dtype=np.float32)

  def reshape_transpose_wq_b(input_tensor, target_shape=None):
      tensor = np.transpose(input_tensor)
      in_dim = tensor.shape[0]
      out_dim = tensor.shape[1]
      return tensor.reshape((in_dim, n_heads, out_dim // n_heads))
      
  def reshape_transpose_wkv(input_tensor, target_shape=None):
      tensor = np.transpose(input_tensor)
      return tensor.reshape((4096, 1, 512))
      
  def reshape_transpose_o_a(input_tensor, target_shape=None):
      # input_tensor is (8192, 4096)
      tensor = input_tensor.reshape((8, 1024, 4096))
      return np.transpose(tensor, (0, 2, 1))
      
  def mhc_split_fn_pre(input_tensor):
      return np.transpose(input_tensor[0:4, :])
  def mhc_split_fn_post(input_tensor):
      return np.transpose(input_tensor[4:8, :])
  def mhc_split_fn_res(input_tensor):
      return np.transpose(input_tensor[8:24, :])

  def mhc_split_base_pre(input_tensor):
      return input_tensor[0:4]
  def mhc_split_base_post(input_tensor):
      return input_tensor[4:8]
  def mhc_split_base_res(input_tensor):
      return input_tensor[8:24].reshape((4, 4))

  def mhc_split_scale_pre(input_tensor):
      return np.array([input_tensor[0]])
  def mhc_split_scale_post(input_tensor):
      return np.array([input_tensor[1]])
  def mhc_split_scale_res(input_tensor):
      return np.array([input_tensor[2]])

  mapping = {}
  for key, hf_key in DEEPSEEKV4_MAXTEXT_TO_HF_PARAM_MAPPING(7, 8).items():
      if hf_key is None: mapping[key] = ones_norm
      elif "-wkv-kernel" in key: mapping[key] = reshape_transpose_wkv
      elif "-wq_b-kernel" in key: mapping[key] = reshape_transpose_wq_b
      elif "-o_a_proj-kernel" in key: mapping[key] = reshape_transpose_o_a
      elif "mhc" in key:
          if "pre_alpha" in key and "scale" not in key: mapping[key] = mhc_split_fn_pre
          elif "post_alpha" in key and "scale" not in key: mapping[key] = mhc_split_fn_post
          elif "res_alpha" in key and "scale" not in key: mapping[key] = mhc_split_fn_res
          elif "pre_beta" in key: mapping[key] = mhc_split_base_pre
          elif "post_beta" in key: mapping[key] = mhc_split_base_post
          elif "res_beta" in key: mapping[key] = mhc_split_base_res
          elif "pre_alpha_scale" in key: mapping[key] = mhc_split_scale_pre
          elif "post_alpha_scale" in key: mapping[key] = mhc_split_scale_post
          elif "res_alpha_scale" in key: mapping[key] = mhc_split_scale_res
      elif type(hf_key) == list:
          mapping[key] = transpose_stack
      elif ("-kernel" in key or "-embedding" in key or "-sinks" in key) and "-token_embedder-embedding" not in key:
          mapping[key] = transpose

  return mapping

def restructure_scanned_weights(converted_weights):
    restructured = {}
    
    restructured["token_embedder"] = converted_weights["token_embedder"]
    restructured["Tid2EidVar"] = converted_weights["Tid2EidVar"]
    
    restructured["decoder"] = {}
    # Copy prefix layers (0, 1, 2)
    for i in range(3): # first_num_hash_layers
        restructured["decoder"][f"layers_{i}"] = converted_weights["decoder"][f"layers_{i}"]
        
    # Copy logits_dense
    restructured["decoder"]["logits_dense"] = converted_weights["decoder"]["logits_dense"]
    
    # Copy decoder_norm
    restructured["decoder"]["decoder_norm"] = converted_weights["decoder"]["decoder_norm"]
    
    restructured["decoder"]["scanned_blocks"] = {
        "layers_0": {},
        "layers_1": {}
    }
    
    def stack_recursive(d1, d2, axis=1):
        result = {}
        for k in d1.keys():
            if isinstance(d1[k], dict):
                result[k] = stack_recursive(d1[k], d2[k], axis)
            else:
                result[k] = jnp.stack([d1[k], d2[k]], axis=axis)
        return result

    # Stack layers_3 and layers_5 into scanned_blocks.layers_0
    restructured["decoder"]["scanned_blocks"]["layers_0"] = stack_recursive(
        converted_weights["decoder"]["layers_3"],
        converted_weights["decoder"]["layers_5"],
        axis=1
    )
    
    # Stack layers_4 and layers_6 into scanned_blocks.layers_1
    restructured["decoder"]["scanned_blocks"]["layers_1"] = stack_recursive(
        converted_weights["decoder"]["layers_4"],
        converted_weights["decoder"]["layers_6"],
        axis=1
    )
    
    return restructured

def restructure_unscanned_weights(converted_weights):
    restructured = {}
    restructured["token_embedder"] = converted_weights["token_embedder"]
    restructured["Tid2EidVar"] = converted_weights["Tid2EidVar"]
    restructured["decoder"] = converted_weights["decoder"]
    return restructured

def convert(hf_model_path, orbax_path, scan_layers=True):
    n_layers = 7
    num_experts = 8
    n_heads = 64
    
    weights = {}
    for filename in os.listdir(hf_model_path):
        if filename.endswith(".safetensors"):
            filepath = os.path.join(hf_model_path, filename)
            weights.update(load_file(filepath))
            
    print(f"Loaded {len(weights)} weights from safetensors.")

    mapping = DEEPSEEKV4_MAXTEXT_TO_HF_PARAM_MAPPING(n_layers, num_experts)
    hooks = DEEPSEEKV4_MAXTEXT_TO_HF_PARAM_HOOK_FN(n_heads)

    
    print("Applying mapping...")
    converted_weights = {}
    for mt_key, hf_key in mapping.items():
        try:
            if hf_key is None:
                val = hooks[mt_key](None) if mt_key in hooks else np.ones(1, dtype=np.float32)
            elif isinstance(hf_key, list):
                val = [weights[k] for k in hf_key]
                if mt_key in hooks:
                    val = hooks[mt_key](val)
            else:
                val = weights[hf_key]
                if mt_key in hooks:
                    val = hooks[mt_key](val)
            
            # Convert to jnp array
            val = jnp.array(val)
                    
            keys = mt_key.split("-")
            if keys[0] == "params":
                keys = keys[1:]
            
            d = converted_weights
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = val
        except Exception as e:
            print(f"Error processing {mt_key}: {e}")
            raise e
        
    print(f"Converted. Saving to {orbax_path}...")
    
    import jax
    # Ensure no torch is imported
    assert 'torch' not in sys.modules, "Torch should not be loaded!"
    
    if scan_layers:
        restructured = restructure_scanned_weights(converted_weights)
    else:
        restructured = restructure_unscanned_weights(converted_weights)
    params_collection = {
        "decoder": restructured["decoder"],
        "token_embedder": restructured["token_embedder"]
    }
    tid2eid_collection = restructured["Tid2EidVar"]
    
    save_dict = {
        "params": {
            "params": params_collection,
            "Tid2EidVar": tid2eid_collection
        }
    }
    checkpointer = ocp.PyTreeCheckpointer(use_ocdbt=True, use_zarr3=True)
    checkpointer.save(os.path.abspath(orbax_path), save_dict)
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert DeepSeek4 HF checkpoint to MaxText Orbax.")
    parser.add_argument("hf_model_path", type=str, help="Path to HF checkpoint directory.")
    parser.add_argument("orbax_path", type=str, help="Path to output Orbax checkpoint directory.")
    parser.add_argument("--scan_layers", action="store_true", default=True, help="Whether layers are scanned (default: True).")
    parser.add_argument("--no_scan_layers", action="store_false", dest="scan_layers", help="Disable layer scanning (flat layers layout).")
    
    args = parser.parse_args()
    convert(args.hf_model_path, args.orbax_path, scan_layers=args.scan_layers)
