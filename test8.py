# %%
import numpy as np
import torch
import jax.numpy as jnp
from typing import Dict, Any

import os
os.chdir('/home/shuningjin_google_com/maxtext')

os.environ["SKIP_JAX_PRECOMPILE"] = "1"
os.environ["JAX_RANDOM_WEIGHTS"] = "False"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
#os.environ["HF_TOKEN"] = ""
os.environ["TPU_MIN_LOG_LEVEL"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TPU_STDERR_LOG_LEVEL"] = "0"
os.environ["VLLM_MLA_DISABLE"] = "1"
os.environ["MODEL_IMPL_TYPE"] = "vllm"
os.environ["JAX_PLATFORMS"] = "tpu"


from transformers import AutoModelForCausalLM
from vllm import LLM
import numpy as np
import torch
import numpy as np
import jax.numpy as jnp

# %%
model_name = "unsloth/gpt-oss-20b-BF16"
golden_llm = LLM(model_name, max_model_len=128, tensor_parallel_size=4)
golden_state = golden_llm.llm_engine.model_executor.driver_worker.model_runner.state

# %%

# local path
# model_name = "/home/shuningjin/qwen3-30b-a3b/hf-bf16"
# or repo id
# model_name = "Qwen/Qwen3-30B-A3B"
model_name = "unsloth/gpt-oss-20b-BF16"

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="bfloat16")
print(len(list(model.named_parameters())))
for name, val in model.named_parameters():
  print(name, val.shape)

# %%
# Define the target data type (bfloat16 for vLLM/JAX compatibility)
TARGET_DTYPE = jnp.bfloat16

def to_np_array(weight: Any) -> np.ndarray:
  """
  Safely converts the input (Tensor or Array) to a NumPy array and
  explicitly casts it to TARGET_DTYPE (np.bfloat16).
  """
  if isinstance(weight, torch.Tensor):
    # Convert PyTorch Tensor to NumPy array
    weight = weight.detach().cpu().to(torch.float32).numpy()

  # Ensure it is a NumPy array and cast to the target dtype
  if isinstance(weight, np.ndarray):
    # The .astype(TARGET_DTYPE) performs the bfloat16 casting
    return weight.astype(TARGET_DTYPE)

  # If the input is already a JAX/Numpy bfloat16 array, this is safe.
  # Otherwise, this handles conversion from PyTorch/other NumPy dtypes.
  return np.array(weight).astype(TARGET_DTYPE)


def convert_gptoss_to_vllm_moe_jax(hf_state_dict: Dict[str, Any], num_layers: int = 24, num_experts: int = 32) -> Dict[str, np.ndarray]:
    """
    Converts GPT-OSS HuggingFace checkpoint keys and weights/biases to vLLM format, 
    with fixes for QKV key naming and explicit handling of MoE expert biases.
    
    The TypeError (2880 vs 5760) is likely a runtime error in vLLM's MoE kernel, 
    but the mapping itself below is correct based on the key names provided. 
    The fix for the QKV naming from 'qkv_proj' to 'qkv' is included.

    Args:
        hf_state_dict: Dictionary containing weights in the GPT-OSS HF format.
        num_layers: The total number of decoder layers.
        num_experts: The total number of experts.

    Returns:
        A new dictionary with weights in the vLLM format, as NumPy arrays
        with dtype set to np.bfloat16.
    """
    vllm_state_dict = {}
    
    print(f"--- Starting GPT-OSS-MoE to vLLM Conversion (L={num_layers}, E={num_experts}) ---")

    # --- 1. Handle Embedding, Head, and Final LayerNorm ---
    vllm_state_dict['vllm_model.lm_head.weight'] = to_np_array(hf_state_dict['lm_head.weight'])
    vllm_state_dict['vllm_model.model.embedding.weight'] = to_np_array(hf_state_dict['model.embed_tokens.weight'])
    vllm_state_dict['vllm_model.model.norm.weight'] = to_np_array(hf_state_dict['model.norm.weight'])
    
    # --- 2. Iterate through Layers ---
    for l in range(num_layers):
        hf_prefix = f'model.layers.{l}'
        vllm_prefix = f'vllm_model.model.layers.{l}'
        
        # --- Direct Copies (LayerNorms, O_proj, Router) ---
        vllm_state_dict[f'{vllm_prefix}.input_layernorm.weight'] = to_np_array(hf_state_dict[f'{hf_prefix}.input_layernorm.weight'])
        vllm_state_dict[f'{vllm_prefix}.post_attention_layernorm.weight'] = to_np_array(hf_state_dict[f'{hf_prefix}.post_attention_layernorm.weight'])
        
        # O_proj (weight and bias)
        vllm_state_dict[f'{vllm_prefix}.attn.o_proj.weight'] = to_np_array(hf_state_dict[f'{hf_prefix}.self_attn.o_proj.weight'])
        vllm_state_dict[f'{vllm_prefix}.attn.o_proj.bias'] = to_np_array(hf_state_dict[f'{hf_prefix}.self_attn.o_proj.bias'])
        
        # Router (weight and bias)
        vllm_state_dict[f'{vllm_prefix}.mlp.router.weight'] = to_np_array(hf_state_dict[f'{hf_prefix}.mlp.router.weight'])
        vllm_state_dict[f'{vllm_prefix}.mlp.router.bias'] = to_np_array(hf_state_dict[f'{hf_prefix}.mlp.router.bias'])

        # Sinks (Direct Copy for Gpt-oss specific state)
        if f'{hf_prefix}.self_attn.sinks' in hf_state_dict:
             vllm_state_dict[f'{vllm_prefix}.attn.sinks'] = to_np_array(hf_state_dict[f'{hf_prefix}.self_attn.sinks'])
        
        # --- Fused Attention (QKV) ---
        # Weights (Concatenate [Q, K, V] along output dimension 0)
        q_weight = to_np_array(hf_state_dict[f'{hf_prefix}.self_attn.q_proj.weight'])
        k_weight = to_np_array(hf_state_dict[f'{hf_prefix}.self_attn.k_proj.weight'])
        v_weight = to_np_array(hf_state_dict[f'{hf_prefix}.self_attn.v_proj.weight'])
        qkv_fused_weight = np.concatenate([q_weight, k_weight, v_weight], axis=0)
        
        # FIX: Renamed from 'qkv_proj' to 'qkv'
        vllm_state_dict[f'{vllm_prefix}.attn.qkv.weight'] = qkv_fused_weight
        
        # Biases (Concatenate [Q_bias, K_bias, V_bias] along dimension 0)
        q_bias = to_np_array(hf_state_dict[f'{hf_prefix}.self_attn.q_proj.bias'])
        k_bias = to_np_array(hf_state_dict[f'{hf_prefix}.self_attn.k_proj.bias'])
        v_bias = to_np_array(hf_state_dict[f'{hf_prefix}.self_attn.v_proj.bias'])
        qkv_fused_bias = np.concatenate([q_bias, k_bias, v_bias], axis=0)
        
        # FIX: Renamed from 'qkv_proj' to 'qkv'
        vllm_state_dict[f'{vllm_prefix}.attn.qkv.bias'] = qkv_fused_bias
        
        # # --- Fused MoE Experts ---
        # # W1/W3 Fusion: HF's 'gate_up_proj' (W1 + W3) maps directly to vLLM's 'w13_weight'
        # # These shapes (32, 5760, 2880) and (32, 5760) are correct for vLLM's internal MoE representation.
        # vllm_state_dict[f'{vllm_prefix}.mlp.experts.w13_weight'] = to_np_array(hf_state_dict[f'{hf_prefix}.mlp.experts.gate_up_proj'])
        # vllm_state_dict[f'{vllm_prefix}.mlp.experts.w13_bias'] = to_np_array(hf_state_dict[f'{hf_prefix}.mlp.experts.gate_up_proj_bias'])
        
        # # W2: HF's 'down_proj' maps directly to vLLM's 'w2_weight'
        # # These shapes (32, 2880, 2880) and (32, 2880) are correct for vLLM's internal MoE representation.
        # vllm_state_dict[f'{vllm_prefix}.mlp.experts.w2_weight'] = to_np_array(hf_state_dict[f'{hf_prefix}.mlp.experts.down_proj'])
        # vllm_state_dict[f'{vllm_prefix}.mlp.experts.w2_bias'] = to_np_array(hf_state_dict[f'{hf_prefix}.mlp.experts.down_proj_bias'])


        # --- Fused MoE Experts ---

        # W1/W3 Fusion: HF's 'gate_up_proj'
        w13_weight = to_np_array(hf_state_dict[f'{hf_prefix}.mlp.experts.gate_up_proj'])

        # **FIX APPLIED HERE: Transpose the last two dimensions (5760 and 2880) **
        # Input shape is (E, H_out, H_in). We need (E, H_in, H_out) based on the error.
        # The original shape from HF is likely (E, Output, Input) -> (32, 5760, 2880)
        # The vLLM expects (32, 5760, 2880). YOUR conversion yielded (32, 2880, 5760).
        # Therefore, you need to transpose the dimensions 1 and 2.
        vllm_state_dict[f'{vllm_prefix}.mlp.experts.w13_weight'] = np.transpose(w13_weight, (0, 2, 1)) 

        # W1/W3 Bias (No transpose needed for 1D/2D bias)
        vllm_state_dict[f'{vllm_prefix}.mlp.experts.w13_bias'] = to_np_array(hf_state_dict[f'{hf_prefix}.mlp.experts.gate_up_proj_bias'])

        # W2 (down_proj): Check if it also needs transposing
        # HF key: down_proj (32, 2880, 2880) -> vLLM key: w2_weight (32, 2880, 2880)
        # Since the shape is (E, H, H), transposing might be necessary if the internal convention differs. 
        # Let's start by transposing W2 as well, as this is common for all linear layers.
        w2_weight = to_np_array(hf_state_dict[f'{hf_prefix}.mlp.experts.down_proj'])
        vllm_state_dict[f'{vllm_prefix}.mlp.experts.w2_weight'] = np.transpose(w2_weight, (0, 2, 1)) 

        # W2 Bias
        vllm_state_dict[f'{vllm_prefix}.mlp.experts.w2_bias'] = to_np_array(hf_state_dict[f'{hf_prefix}.mlp.experts.down_proj_bias'])



    print("--- GPT-OSS-MoE to vLLM Conversion Complete ---")
    return vllm_state_dict


converted_state = convert_gptoss_to_vllm_moe_jax(model.state_dict())

# %%
missing_keys = golden_state.keys() - converted_state.keys()
print(missing_keys)


for key, val in converted_state.items():
  print(
      f"{golden_state[key].dtype}, {val.dtype} -------- converted state shape:{key} {val.shape} and golden_state[key] shape:  {golden_state[key].shape}"
  )
  assert val.shape == golden_state[key].shape, f"{key}, {val.shape} {golden_state[key].shape}"
  golden_state[key] = val


del model

print(golden_llm.generate("what is the capital of France?"))

# %%



