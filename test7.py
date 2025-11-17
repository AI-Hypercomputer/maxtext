import numpy as np
import torch
import jax.numpy as jnp
from typing import Dict, Any

# Define the target data type (bfloat16 for vLLM/JAX compatibility)
TARGET_DTYPE = jnp.bfloat16


import os


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


def to_np_array(weight: Any) -> np.ndarray:
  """
  Safely converts the input (Tensor or Array) to a NumPy array and
  explicitly casts it to TARGET_DTYPE (np.bfloat16).
  """
  if isinstance(weight, torch.Tensor):
    # Convert PyTorch Tensor to NumPy array
    weight = weight.detach().cpu().numpy()

  # Ensure it is a NumPy array and cast to the target dtype
  if isinstance(weight, np.ndarray):
    # The .astype(TARGET_DTYPE) performs the bfloat16 casting
    return weight.astype(TARGET_DTYPE)

  # If the input is already a JAX/Numpy bfloat16 array, this is safe.
  # Otherwise, this handles conversion from PyTorch/other NumPy dtypes.
  return np.array(weight).astype(TARGET_DTYPE)


def convert_gptoss_to_vllm_moe_jax(
    hf_state_dict: Dict[str, Any], num_layers: int = 24, num_experts: int = 32
) -> Dict[str, np.ndarray]:
  """
  Converts GPT-OSS (Mistral-based MoE architecture) HuggingFace checkpoint
  keys and weights/biases to vLLM format, including QKV fusion and MoE expert handling.

  Based on the provided key lists:
  HF uses: q_proj, k_proj, v_proj (separate)
  HF MoE uses: gate_up_proj (fused W1/W3), down_proj (W2)
  vLLM uses: qkv_proj (fused), w13_weight (fused W1/W3), w2_weight (W2)

  Args:
      hf_state_dict: Dictionary containing weights in the GPT-OSS HF format.
      num_layers: The total number of decoder layers. (Inferred as 24 layers
                  from the max layer index in the key list, but set to 24 for a typical 20B model).
      num_experts: The total number of experts. (Inferred as 32 from key shapes).

  Returns:
      A new dictionary with weights in the vLLM format, as NumPy arrays
      with dtype set to np.bfloat16.
  """
  vllm_state_dict = {}

  print(f"--- Starting GPT-OSS-MoE to vLLM Conversion (L={num_layers}, E={num_experts}) ---")

  # --- 1. Handle Embedding, Head, and Final LayerNorm ---
  vllm_state_dict["vllm_model.lm_head.weight"] = to_np_array(hf_state_dict["lm_head.weight"])
  vllm_state_dict["vllm_model.model.embedding.weight"] = to_np_array(hf_state_dict["model.embed_tokens.weight"])
  vllm_state_dict["vllm_model.model.norm.weight"] = to_np_array(hf_state_dict["model.norm.weight"])

  # --- 2. Iterate through Layers ---
  for l in range(num_layers):
    hf_prefix = f"model.layers.{l}"
    vllm_prefix = f"vllm_model.model.layers.{l}"

    # --- Direct Copies (LayerNorms, O_proj, Router) ---
    vllm_state_dict[f"{vllm_prefix}.input_layernorm.weight"] = to_np_array(
        hf_state_dict[f"{hf_prefix}.input_layernorm.weight"]
    )
    vllm_state_dict[f"{vllm_prefix}.post_attention_layernorm.weight"] = to_np_array(
        hf_state_dict[f"{hf_prefix}.post_attention_layernorm.weight"]
    )

    # O_proj (weight and bias)
    vllm_state_dict[f"{vllm_prefix}.attn.o_proj.weight"] = to_np_array(
        hf_state_dict[f"{hf_prefix}.self_attn.o_proj.weight"]
    )
    vllm_state_dict[f"{vllm_prefix}.attn.o_proj.bias"] = to_np_array(hf_state_dict[f"{hf_prefix}.self_attn.o_proj.bias"])

    # Router (weight and bias)
    vllm_state_dict[f"{vllm_prefix}.mlp.router.weight"] = to_np_array(hf_state_dict[f"{hf_prefix}.mlp.router.weight"])
    vllm_state_dict[f"{vllm_prefix}.mlp.router.bias"] = to_np_array(hf_state_dict[f"{hf_prefix}.mlp.router.bias"])

    # Sinks (Direct Copy for Gpt-oss specific state)
    # Note: vLLM's internal handling of sinks might not use this key directly,
    # but we copy it for completeness based on the vLLM key list.
    if f"{hf_prefix}.self_attn.sinks" in hf_state_dict:
      vllm_state_dict[f"{vllm_prefix}.attn.sinks"] = to_np_array(hf_state_dict[f"{hf_prefix}.self_attn.sinks"])

    # --- Fused Attention (QKV) ---
    # Weights (Concatenate [Q, K, V] along output dimension 0)
    q_weight = to_np_array(hf_state_dict[f"{hf_prefix}.self_attn.q_proj.weight"])
    k_weight = to_np_array(hf_state_dict[f"{hf_prefix}.self_attn.k_proj.weight"])
    v_weight = to_np_array(hf_state_dict[f"{hf_prefix}.self_attn.v_proj.weight"])
    qkv_fused_weight = np.concatenate([q_weight, k_weight, v_weight], axis=0)
    vllm_state_dict[f"{vllm_prefix}.attn.qkv_proj.weight"] = qkv_fused_weight

    # Biases (Concatenate [Q_bias, K_bias, V_bias] along dimension 0)
    q_bias = to_np_array(hf_state_dict[f"{hf_prefix}.self_attn.q_proj.bias"])
    k_bias = to_np_array(hf_state_dict[f"{hf_prefix}.self_attn.k_proj.bias"])
    v_bias = to_np_array(hf_state_dict[f"{hf_prefix}.self_attn.v_proj.bias"])
    qkv_fused_bias = np.concatenate([q_bias, k_bias, v_bias], axis=0)
    vllm_state_dict[f"{vllm_prefix}.attn.qkv_proj.bias"] = qkv_fused_bias

    # --- Fused MoE Experts ---
    # W1/W3 Fusion: HF's 'gate_up_proj' (W1 + W3) maps directly to vLLM's 'w13_weight'
    vllm_state_dict[f"{vllm_prefix}.mlp.experts.w13_weight"] = to_np_array(
        hf_state_dict[f"{hf_prefix}.mlp.experts.gate_up_proj"]
    )
    vllm_state_dict[f"{vllm_prefix}.mlp.experts.w13_bias"] = to_np_array(
        hf_state_dict[f"{hf_prefix}.mlp.experts.gate_up_proj_bias"]
    )

    # W2: HF's 'down_proj' maps directly to vLLM's 'w2_weight'
    vllm_state_dict[f"{vllm_prefix}.mlp.experts.w2_weight"] = to_np_array(
        hf_state_dict[f"{hf_prefix}.mlp.experts.down_proj"]
    )
    vllm_state_dict[f"{vllm_prefix}.mlp.experts.w2_bias"] = to_np_array(
        hf_state_dict[f"{hf_prefix}.mlp.experts.down_proj_bias"]
    )

  print("--- GPT-OSS-MoE to vLLM Conversion Complete ---")
  return vllm_state_dict


# local path
# model_name = "/home/shuningjin/qwen3-30b-a3b/hf-bf16"
# or repo id
# model_name = "Qwen/Qwen3-30B-A3B"
model_name = "unsloth/gpt-oss-20b-BF16"

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="bfloat16")
print(len(list(model.named_parameters())))
for name, val in model.named_parameters():
  print(name, val.shape)



golden_llm = LLM(model_name, max_model_len=128, tensor_parallel_size=4)
golden_state = golden_llm.llm_engine.model_executor.driver_worker.model_runner.state
gold_state = golden_state.flat_state()


converted_state = convert_gptoss_to_vllm_moe_jax(model.state_dict())

missing_keys = gold_state.keys() - converted_state.keys()
print(missing_keys)


for key, val in converted_state.items():
  print(
      f"{golden_state[key].dtype}, {val.dtype} -------- converted state shape:{key} {val.shape} and golden_state[key] shape:  {golden_state[key].shape}"
  )
  golden_state[key] = val

print(golden_llm.generate("what is the capital of France?"))
