"""Convert weights from a Qwen3 MoE HuggingFace model to a MaxText checkpoint.

This script provides a skeleton for converting the recently released
Qwen3 Mixture of Experts model. The detailed parameter mapping is not
yet implemented and must be filled in before use.
"""

import argparse
import os
import pathlib

import numpy as np
import torch
from safetensors import safe_open
from transformers import Qwen3MoeConfig

from MaxText import max_logging
from MaxText import llama_or_mistral_ckpt
from MaxText.inference_utils import str2bool


def _convert_huggingface_to_jax_weights(base_model_path: str) -> dict:
    """Convert HuggingFace Qwen3 MoE weights to a MaxText-compatible format."""
    config = Qwen3MoeConfig.from_pretrained(base_model_path)
    max_logging.log("Loading Qwen3 MoE checkpoint")
    ckpt_paths = sorted(pathlib.Path(base_model_path).glob("*.safetensors"))
    chkpt_vars = {}
    for ckpt_path in ckpt_paths:
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                chkpt_vars[key] = f.get_tensor(key)

    jax_weights = {
        "decoder": {
            "decoder_norm": {"scale": chkpt_vars["model.norm.weight"].numpy()},
            "logits_dense": {"kernel": chkpt_vars["lm_head.weight"].numpy().T},
            "layers": {},
        },
        "token_embedder": {"embedding": chkpt_vars["model.embed_tokens.weight"].numpy()},
    }

    for layer_idx in range(config.num_hidden_layers):
        layer = {}
        layer["pre_self_attention_layer_norm"] = {
            "scale": chkpt_vars[f"model.layers.{layer_idx}.input_layernorm.weight"].numpy()
        }
        layer["post_self_attention_layer_norm"] = {
            "scale": chkpt_vars[f"model.layers.{layer_idx}.post_attention_layernorm.weight"].numpy()
        }
        layer["self_attention"] = {
            "query": {
                "kernel": chkpt_vars[f"model.layers.{layer_idx}.self_attn.q_proj.weight"].numpy().T
                / np.sqrt(config.head_dim)
            },
            "key": {"kernel": chkpt_vars[f"model.layers.{layer_idx}.self_attn.k_proj.weight"].numpy().T},
            "value": {"kernel": chkpt_vars[f"model.layers.{layer_idx}.self_attn.v_proj.weight"].numpy().T},
            "out": {"kernel": chkpt_vars[f"model.layers.{layer_idx}.self_attn.o_proj.weight"].numpy().T},
        }

        moe = {
            "gate": {"kernel": chkpt_vars[f"model.layers.{layer_idx}.mlp.gate.weight"].numpy().T},
            "wi_0": [],
            "wi_1": [],
            "wo": [],
        }
        for expert_idx in range(config.num_experts):
            moe["wi_0"].append(
                chkpt_vars[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"].numpy().T
            )
            moe["wi_1"].append(
                chkpt_vars[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"].numpy().T
            )
            moe["wo"].append(
                chkpt_vars[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"].numpy().T
            )
        moe["wi_0"] = np.stack(moe["wi_0"], axis=0)
        moe["wi_1"] = np.stack(moe["wi_1"], axis=0)
        moe["wo"] = np.stack(moe["wo"], axis=0)

        layer["MoeBlock_0"] = moe
        jax_weights["decoder"]["layers"][f"layer_{layer_idx}"] = layer

    return jax_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen3 MoE weights.")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--maxtext_model_path", type=str, required=True)
    parser.add_argument("--simulated_cpu_devices_count", type=int, default=16)
    parser.add_argument("--use-ocdbt", type=str2bool, default=True)
    parser.add_argument("--use-zarr3", type=str2bool, default=True)
    args = parser.parse_args()

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"

    weights = _convert_huggingface_to_jax_weights(args.base_model_path)
    llama_or_mistral_ckpt.save_weights_to_checkpoint(
        args.maxtext_model_path,
        weights,
        args.simulated_cpu_devices_count,
        args.use_ocdbt,
        args.use_zarr3,
    )


if __name__ == "__main__":
    main()
