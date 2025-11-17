from dataclasses import dataclass
import jax.numpy as jnp
from flax import nnx
import re


@dataclass
class GptOssMaxTextMapping:
  """Mapping from MaxText GPT-OSS to VLLM JAX NNX."""

  @staticmethod
  def to_hf_hook_fns():
    return {}

  @staticmethod
  def to_hf_transpose_keys():
    return {}

  @staticmethod
  def to_hf_mapping(layer_cycle_interval: int = 2):
    mapping = {
        "base.token_embedder.embedding": ("embedder.input_embedding_table_VD", (("data", "model"), None)),
        "base.decoder.decoder_norm.scale": ("final_norm.scale", (None,)),
        "base.decoder.logits_dense.kernel": ("lm_head.input_embedding_table_DV", (None, ("data", "model"))),
    }

    for block_idx in range(layer_cycle_interval):
      src_block = f"base.decoder.layers.layers_{block_idx}"
      mapping.update(
          {
              f"{src_block}.pre_self_attention_layer_norm.scale": ("layers.*.pre_attention_norm.scale", (None, "layer")),
              f"{src_block}.post_self_attention_layer_norm.scale": ("layers.*.pre_mlp_norm.scale", (None, "layer")),
              # Attention
              f"{src_block}.GptOssAttention.query.kernel": ("layers.*.attn.kernel_q_DNH", (None, "layer", "model", None)),
              f"{src_block}.GptOssAttention.key.kernel": ("layers.*.attn.kernel_k_DKH", (None, "layer", "model", None)),
              f"{src_block}.GptOssAttention.value.kernel": ("layers.*.attn.kernel_v_DKH", (None, "layer", "model", None)),
              f"{src_block}.GptOssAttention.out.kernel": (
                  "layers.*.attn.kernel_o_proj_NHD",
                  ("model", "layer", None, None),
              ),
              f"{src_block}.GptOssAttention.query.bias": ("layers.*.attn.bias_q_NH", (None, "layer", None)),
              f"{src_block}.GptOssAttention.key.bias": ("layers.*.attn.bias_k_KH", (None, "layer", None)),
              f"{src_block}.GptOssAttention.value.bias": ("layers.*.attn.bias_v_KH", (None, "layer", None)),
              f"{src_block}.GptOssAttention.out.bias": ("layers.*.attn.bias_o_D", (None, "layer")),
              f"{src_block}.GptOssAttention.sinks": ("layers.*.attn.sinks_N", (None, "layer")),
              # MoE
              f"{src_block}.GptOssMlp.gate.kernel": ("layers.*.custom_module.router.kernel_DE", (None, "layer", "model")),
              f"{src_block}.GptOssMlp.gate.bias": ("layers.*.custom_module.router.bias_E", (None, "layer")),
              # Target Unquantized Keys (Assuming you disabled quantization)
              f"{src_block}.GptOssMlp.gate_up_proj.kernel": (
                  "layers.*.custom_module.mlp1_weight_EDF2",
                  ("expert", "layer", "model", None),
              ),
              f"{src_block}.GptOssMlp.gate_up_proj_bias": (
                  "layers.*.custom_module.mlp1_bias_EF2",
                  ("expert", "layer", None),
              ),
              f"{src_block}.GptOssMlp.wo.kernel": (
                  "layers.*.custom_module.mlp2_weight_EFD",
                  ("expert", "layer", "model", None),
              ),
              f"{src_block}.GptOssMlp.wo.bias": ("layers.*.custom_module.mlp2_bias_ED", ("expert", "layer", None)),
          }
      )
    return mapping


def preprocess_maxtext_nnx_state(maxtext_state: nnx.State) -> DictionaryState:
  """
  1. Flattens nnx.State using flat_state()
  2. Merges MoE weights.
  3. Wraps arrays in SimpleParam (Fixes 'ArrayImpl has no attribute value').
  """
  print("Preprocessing MaxText NNX State...")

  # Temporary dict to hold raw values
  raw_values = {}
  keys_to_merge = []

  # 1. Iterate using the correct flat_state() API
  for path_tuple, param in maxtext_state.flat_state():
    # Convert path tuple to dot-separated string key
    key = ".".join(str(p) for p in path_tuple)

    # Unwrap Param to get the raw array
    value = param.value if hasattr(param, "value") else param

    # Identify split MoE weights (Gate)
    if re.search(r"\.GptOssMlp\.wi_0", key):
      key_wi1 = key.replace("wi_0", "wi_1")
      # We defer checking if wi_1 exists until the merge loop
      keys_to_merge.append((key, key_wi1))

    raw_values[key] = value

  # 2. Merge MoE Weights (Interleave Gate & Up)
  for key_wi0, key_wi1 in keys_to_merge:
    # Only merge if both parts exist
    if key_wi0 in raw_values and key_wi1 in raw_values:
      val_0 = raw_values[key_wi0]  # Gate
      val_1 = raw_values[key_wi1]  # Up

      print(f"  Merging: {key_wi0} + {key_wi1}")

      # Stack along last dim and interleave
      # Shape: [..., Hidden, Intermed] -> [..., Hidden, Intermed * 2]
      combined = jnp.stack((val_0, val_1), axis=-1)
      new_shape = combined.shape[:-2] + (combined.shape[-2] * 2,)
      merged_value = combined.reshape(new_shape)

      # Create new key: ...gate_up_proj
      new_key = key_wi0.replace("wi_0", "gate_up_proj")
      raw_values[new_key] = merged_value

      # Handle Bias merging if present
      key_b0 = key_wi0.replace("kernel", "bias") if "kernel" in key_wi0 else key_wi0 + "_bias"
      key_b1 = key_wi1.replace("kernel", "bias") if "kernel" in key_wi1 else key_wi1 + "_bias"

      if key_b0 in raw_values and key_b1 in raw_values:
        b0 = raw_values[key_b0]
        b1 = raw_values[key_b1]
        b_comb = jnp.stack((b0, b1), axis=-1)
        b_new_shape = b_comb.shape[:-2] + (b_comb.shape[-2] * 2,)

        # Determine new bias key name
        if "kernel" in new_key:
          new_bias_key = new_key.replace("kernel", "bias")
        else:
          new_bias_key = new_key.replace("gate_up_proj", "gate_up_proj_bias")

        raw_values[new_bias_key] = b_comb.reshape(b_new_shape)

        del raw_values[key_b0]
        del raw_values[key_b1]

      # Remove original split weights to clean up
      del raw_values[key_wi0]
      del raw_values[key_wi1]

  # 3. Wrap every array in SimpleParam
  wrapped_output = {k: SimpleParam(v) for k, v in raw_values.items()}

  # Return the custom state object
  return DictionaryState(wrapped_output)
