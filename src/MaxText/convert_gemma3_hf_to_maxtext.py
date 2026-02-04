# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Gemma 3 HuggingFace checkpoints to MaxText format.
This script converts Gemma 3 model weights from HuggingFace safetensors format
to MaxText-compatible Orbax checkpoint format.
Usage:
    python convert_gemma3_hf_to_maxtext.py \
        --input_path=/path/to/gemma3-27b-hf \
        --output_path=gs://bucket/gemma3-27b-maxtext \
        --model_size=27b
Supported models:
    - google/gemma-3-27b-it (27B parameters)
    - google/gemma-3-12b-it (12B parameters)
    - google/gemma-3-4b-it (4B parameters)
    - google/gemma-3-1b-it (1B parameters)
"""
import argparse
import gc
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import ml_dtypes
import numpy as np
import orbax.checkpoint as ocp
import torch
from safetensors.torch import load_file
# Model configurations for different Gemma 3 sizes
MODEL_CONFIGS = {
    "1b": {
        "num_layers": 26,
        "num_heads": 8,
        "num_kv_heads": 4,
        "head_dim": 256,
        "embed_dim": 2048,
        "hidden_dim": 8192,
    },
    "4b": {
        "num_layers": 34,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 256,
        "embed_dim": 3072,
        "hidden_dim": 12288,
    },
    "12b": {
        "num_layers": 48,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 256,
        "embed_dim": 4096,
        "hidden_dim": 16384,
    },
    "27b": {
        "num_layers": 62,
        "num_heads": 32,
        "num_kv_heads": 16,
        "head_dim": 128,
        "embed_dim": 5376,
        "hidden_dim": 21504,
    },
}
def to_bf16_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to NumPy bfloat16 array.
    
    Args:
        tensor: PyTorch tensor in any dtype.
        
    Returns:
        NumPy array in bfloat16 format.
    """
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return tensor.to(torch.bfloat16).view(torch.int16).numpy().view(ml_dtypes.bfloat16)
def load_hf_weights(input_path: str) -> dict:
    """Load HuggingFace safetensors weights.
    
    Args:
        input_path: Path to the HuggingFace model directory.
        
    Returns:
        Dictionary of weight tensors converted to bfloat16 NumPy arrays.
    """
    weights = {}
    print("Loading weights...")
    
    for filename in sorted(os.listdir(input_path)):
        if filename.endswith(".safetensors"):
            print(f"  Loading {filename}...")
            filepath = os.path.join(input_path, filename)
            tensors = load_file(filepath)
            for key, tensor in tensors.items():
                weights[key] = to_bf16_numpy(tensor)
            del tensors
            gc.collect()
    
    print(f"Loaded {len(weights)} tensors.")
    return weights
def convert_to_maxtext(weights: dict, config: dict) -> dict:
    """Convert HuggingFace weights to MaxText format.
    
    Args:
        weights: Dictionary of HuggingFace weight tensors.
        config: Model configuration dictionary.
        
    Returns:
        Dictionary of weights in MaxText format.
    """
    print("Converting to MaxText format...")
    
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    embed_dim = config["embed_dim"]
    hidden_dim = config["hidden_dim"]
    
    # Embedding and output layers
    embed = weights["language_model.model.embed_tokens.weight"]
    final_norm = weights["language_model.model.norm.weight"] + 1.0
    lm_head = embed.T  # Weight tying
    
    # Initialize layer arrays
    q = np.zeros((num_layers, embed_dim, num_heads, head_dim), dtype=ml_dtypes.bfloat16)
    k = np.zeros((num_layers, embed_dim, num_kv_heads, head_dim), dtype=ml_dtypes.bfloat16)
    v = np.zeros((num_layers, embed_dim, num_kv_heads, head_dim), dtype=ml_dtypes.bfloat16)
    o = np.zeros((num_layers, num_heads, head_dim, embed_dim), dtype=ml_dtypes.bfloat16)
    pre = np.zeros((num_layers, embed_dim), dtype=ml_dtypes.bfloat16)
    post = np.zeros((num_layers, embed_dim), dtype=ml_dtypes.bfloat16)
    gate = np.zeros((num_layers, embed_dim, hidden_dim), dtype=ml_dtypes.bfloat16)
    up = np.zeros((num_layers, embed_dim, hidden_dim), dtype=ml_dtypes.bfloat16)
    down = np.zeros((num_layers, hidden_dim, embed_dim), dtype=ml_dtypes.bfloat16)
    
    # Convert each layer
    for i in range(num_layers):
        if i % 10 == 0:
            print(f"  Layer {i}/{num_layers}...")
        
        prefix = f"language_model.model.layers.{i}"
        
        # Attention weights (reshape from [out, in] to MaxText layout)
        q[i] = weights[f"{prefix}.self_attn.q_proj.weight"].T.reshape(
            embed_dim, num_heads, head_dim
        )
        k[i] = weights[f"{prefix}.self_attn.k_proj.weight"].T.reshape(
            embed_dim, num_kv_heads, head_dim
        )
        v[i] = weights[f"{prefix}.self_attn.v_proj.weight"].T.reshape(
            embed_dim, num_kv_heads, head_dim
        )
        o[i] = weights[f"{prefix}.self_attn.o_proj.weight"].T.reshape(
            num_heads, head_dim, embed_dim
        )
        
        # RMSNorm weights (add +1.0 offset for Gemma)
        pre[i] = weights[f"{prefix}.input_layernorm.weight"] + 1.0
        post[i] = weights[f"{prefix}.post_attention_layernorm.weight"] + 1.0
        
        # MLP weights
        gate[i] = weights[f"{prefix}.mlp.gate_proj.weight"].T
        up[i] = weights[f"{prefix}.mlp.up_proj.weight"].T
        down[i] = weights[f"{prefix}.mlp.down_proj.weight"].T
    
    # Build MaxText checkpoint structure
    maxtext_weights = {
        "token_embedder": {"embedding": embed},
        "decoder": {
            "decoder_norm": {"scale": final_norm},
            "logits_dense": {"kernel": lm_head},
            "layers": {
                "self_attention": {
                    "query": {"kernel": q},
                    "key": {"kernel": k},
                    "value": {"kernel": v},
                    "out": {"kernel": o},
                },
                "pre_self_attention_layer_norm": {"scale": pre},
                "post_self_attention_layer_norm": {"scale": post},
                "mlp": {
                    "wi_0": {"kernel": gate},
                    "wi_1": {"kernel": up},
                    "wo": {"kernel": down},
                },
            },
        },
    }
    
    return maxtext_weights
def save_checkpoint(weights: dict, output_path: str) -> None:
    """Save weights as Orbax checkpoint.
    
    Args:
        weights: MaxText-formatted weight dictionary.
        output_path: Output path (local or gs://).
    """
    print(f"Saving checkpoint to {output_path}...")
    ocp.PyTreeCheckpointer().save(output_path, weights)
    print("Done!")
def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemma 3 HuggingFace checkpoints to MaxText format."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the HuggingFace model directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for MaxText checkpoint (local or gs://).",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["1b", "4b", "12b", "27b"],
        required=True,
        help="Gemma 3 model size.",
    )
    
    args = parser.parse_args()
    
    config = MODEL_CONFIGS[args.model_size]
    print(f"Converting Gemma 3 {args.model_size.upper()} model...")
    print(f"  Input: {args.input_path}")
    print(f"  Output: {args.output_path}")
    
    weights = load_hf_weights(args.input_path)
    maxtext_weights = convert_to_maxtext(weights, config)
    
    del weights
    gc.collect()
    
    save_checkpoint(maxtext_weights, args.output_path)
if __name__ == "__main__":
    main()