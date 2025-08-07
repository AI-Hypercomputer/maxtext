"""
Copyright 2025 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

r"""Convert weights from a Qwen3-MoE style model to a MaxText one.

Example cmd:

python3 -m MaxText.convert_qwen3_moe_ckpt --base_model_path <path/to/hf/ckpt> \
    --maxtext_model_path <GCS/path/to/save/new/maxtext/ckpt> --model_size qwen3-moe
"""

import argparse
import pathlib
import os
import gc
import json
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from MaxText import max_logging, llama_or_mistral_ckpt

def hf_to_maxtext_mapping(layer_idx: int, hf_config: dict) -> dict:
    """
    Generates a mapping between Hugging Face (HF) and MaxText model weight names for Qwen3-MoE.

    Args:
        layer_idx: The index of the current layer.
        hf_config: The Hugging Face model configuration dictionary.

    Returns:
        A dictionary mapping HF weight names to MaxText weight names.
    """
    num_experts = hf_config.get("num_experts", 128)
    
    mapping = {
        "model.embed_tokens.weight": "token_embedder.embedding",
        "model.norm.weight": "decoder.decoder_norm.scale",
        "lm_head.weight": "decoder.logits_dense.kernel",
    }

    # Layer-specific mappings
    layer_mapping = {
        # Attention block
        f"model.layers.{layer_idx}.input_layernorm.weight": f"decoder.layers.{layer_idx}.pre_self_attention_layer_norm.scale",
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"decoder.layers.{layer_idx}.post_self_attention_layer_norm.scale",
        f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"decoder.layers.{layer_idx}.self_attention.query.kernel",
        f"model.layers.{layer_idx}.self_attn.k_proj.weight": f"decoder.layers.{layer_idx}.self_attention.key.kernel",
        f"model.layers.{layer_idx}.self_attn.v_proj.weight": f"decoder.layers.{layer_idx}.self_attention.value.kernel",
        f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"decoder.layers.{layer_idx}.self_attention.out.kernel",
        f"model.layers.{layer_idx}.self_attn.q_norm.weight": f"decoder.layers.{layer_idx}.self_attention.query_norm.scale",
        f"model.layers.{layer_idx}.self_attn.k_norm.weight": f"decoder.layers.{layer_idx}.self_attention.key_norm.scale",
        
        # MoE block
        f"model.layers.{layer_idx}.mlp.gate.weight": f"decoder.layers.{layer_idx}.moe_block.gate.kernel",
    }
    
    # Add expert mappings
    for i in range(num_experts):
        layer_mapping[f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj.weight"] = f"decoder.layers.{layer_idx}.moe_block.wi_0.{i}"
        layer_mapping[f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj.weight"] = f"decoder.layers.{layer_idx}.moe_block.wi_1.{i}"
        layer_mapping[f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj.weight"] = f"decoder.layers.{layer_idx}.moe_block.wo.{i}"

    mapping.update(layer_mapping)
    return mapping

def convert_hf_to_maxtext(base_model_path: str, hf_config: dict) -> dict:
    """Converts a Hugging Face Qwen3-MoE checkpoint to a MaxText compatible format."""
    
    num_layers = hf_config["num_hidden_layers"]
    num_experts = hf_config["num_experts"]
    
    # Load all safetensors files into a single dictionary
    hf_weights = {}
    ckpt_paths = sorted(pathlib.Path(base_model_path).glob("*.safetensors"))
    for i, ckpt_path in enumerate(ckpt_paths):
        max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)}...")
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                hf_weights[key] = f.get_tensor(key)

    maxtext_weights = {}

    # Convert non-layer-specific weights
    maxtext_weights["token_embedder"] = {"embedding": hf_weights["model.embed_tokens.weight"].to(torch.float16).numpy()}
    maxtext_weights["decoder"] = {
        "decoder_norm": {"scale": hf_weights["model.norm.weight"].to(torch.float16).numpy()},
        "logits_dense": {"kernel": hf_weights["lm_head.weight"].to(torch.float16).numpy().transpose()}
    }
    
    # Prepare layer structure
    maxtext_weights["decoder"]["layers"] = [{} for _ in range(num_layers)]

    for l in tqdm(range(num_layers), desc="Converting layers"):
        maxtext_layer = {}
        
        # Attention
        maxtext_layer["self_attention"] = {
            "query": {"kernel": hf_weights[f"model.layers.{l}.self_attn.q_proj.weight"].to(torch.float16).numpy().transpose()},
            "key": {"kernel": hf_weights[f"model.layers.{l}.self_attn.k_proj.weight"].to(torch.float16).numpy().transpose()},
            "value": {"kernel": hf_weights[f"model.layers.{l}.self_attn.v_proj.weight"].to(torch.float16).numpy().transpose()},
            "out": {"kernel": hf_weights[f"model.layers.{l}.self_attn.o_proj.weight"].to(torch.float16).numpy().transpose()},
            "query_norm": {"scale": hf_weights[f"model.layers.{l}.self_attn.q_norm.weight"].to(torch.float16).numpy()},
            "key_norm": {"scale": hf_weights[f"model.layers.{l}.self_attn.k_norm.weight"].to(torch.float16).numpy()},
        }
        
        # Layer norms
        maxtext_layer["pre_self_attention_layer_norm"] = {"scale": hf_weights[f"model.layers.{l}.input_layernorm.weight"].to(torch.float16).numpy()}
        maxtext_layer["post_self_attention_layer_norm"] = {"scale": hf_weights[f"model.layers.{l}.post_attention_layernorm.weight"].to(torch.float16).numpy()}
        
        # MoE
        gate = hf_weights[f"model.layers.{l}.mlp.gate.weight"].to(torch.float16).numpy().transpose()
        
        wi_0_experts = [hf_weights[f"model.layers.{l}.mlp.experts.{i}.gate_proj.weight"].to(torch.float16).numpy().transpose() for i in range(num_experts)]
        wi_1_experts = [hf_weights[f"model.layers.{l}.mlp.experts.{i}.up_proj.weight"].to(torch.float16).numpy().transpose() for i in range(num_experts)]
        wo_experts = [hf_weights[f"model.layers.{l}.mlp.experts.{i}.down_proj.weight"].to(torch.float16).numpy().transpose() for i in range(num_experts)]
        
        maxtext_layer["moe_block"] = {
            "gate": {"kernel": gate},
            "wi_0": np.stack(wi_0_experts, axis=0),
            "wi_1": np.stack(wi_1_experts, axis=0),
            "wo": np.stack(wo_experts, axis=0)
        }
        
        maxtext_weights["decoder"]["layers"][l] = maxtext_layer

    return maxtext_weights

def main(args):
    # Load HF config
    with open(os.path.join(args.base_model_path, "config.json"), "r") as f:
        hf_config = json.load(f)

    max_logging.log(f"Starting conversion for Qwen3-MoE model...")
    
    jax_weights = convert_hf_to_maxtext(args.base_model_path, hf_config)
    
    max_logging.log(f"Conversion complete. Saving MaxText checkpoint to {args.maxtext_model_path}")
    
    llama_or_mistral_ckpt.save_weights_to_checkpoint(
        args.maxtext_model_path,
        jax_weights
    )
    
    max_logging.log("Checkpoint saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Qwen3-MoE HF weights to MaxText.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the HF Qwen3-MoE checkpoint files.")
    parser.add_argument("--maxtext_model_path", type=str, required=True, help="Path to save the MaxText checkpoint.")
    
    args = parser.parse_args()
    main(args)