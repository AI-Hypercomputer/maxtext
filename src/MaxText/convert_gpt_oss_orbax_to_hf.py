# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Convert weights from a MaxText GPT-OSS model to a HuggingFace model.

Usage:

This script is meant to be run from the root of the MaxText repository.

Example cmd:
python3 MaxText/convert_gpt_oss_orbax_to_hf.py src/MaxText/configs/base.yml \
    load_parameters_path=/path/to/maxtext/checkpoint/step_.../ \
    model_name=gpt-oss-20b \
    hf_model_path=/local/path/to/save/HF/model/
"""

import json
import os
from typing import Sequence

import numpy as np
import torch
from absl import app
from jax.sharding import Mesh
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from MaxText import checkpointing, max_logging, maxtext_utils, pyconfig
from MaxText.generate_param_only_checkpoint import _read_train_checkpoint

# Model parameters to map MaxText config to GPT-OSS HF config
MODEL_PARAMS_DICT = {
    "gpt-oss-20b": {
        "base_emb_dim": 2880,
        "base_num_query_heads": 64,
        "base_num_kv_heads": 8,
        "head_dim": 64,
        "base_num_decoder_layers": 24,
        "intermediate_dim": 8192,
        "num_experts": 8,
    },
    "gpt-oss-120b": {
        "base_emb_dim": 2880,
        "base_num_query_heads": 64,
        "base_num_kv_heads": 8,
        "head_dim": 64,
        "base_num_decoder_layers": 36,
        "intermediate_dim": 8192,
        "num_experts": 16, # Assuming this scales from 20b
    },
}

# Configuration used to create a `config.json` for the HF model.
# The architecture is assumed to be Mixtral-like based on the MoE structure.
HF_CONFIG_MAP = {
    "gpt-oss-20b": {
        "hidden_size": 2880,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_hidden_layers": 24,
        "intermediate_size": 8192,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-5,
        "model_type": "mixtral",
    },
    "gpt-oss-120b": {
        "hidden_size": 2880,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_hidden_layers": 36,
        "intermediate_size": 8192,
        "num_local_experts": 16,
        "num_experts_per_tok": 2,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-5,
        "model_type": "mixtral",
    },
}


def load_model_state(config):
    """Loads the MaxText model's TrainState from the Orbax checkpoint."""
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        config.checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing,
        config.checkpoint_period,
    )

    max_logging.log(f"Read training checkpoint from: {config.load_parameters_path}")
    training_state, _ = _read_train_checkpoint(
        config=config,
        checkpoint_manager=checkpoint_manager,
        mesh=mesh,
        # We need the raw unscanned checkpoint, not the scanned one
        scanned_params=False,
    )
    return training_state


def convert_state_to_hf(training_state, model_size):
    """
    Ports the parameters from the Orbax training_state into a
    Hugging Face compatible state dictionary.
    """
    if model_size not in MODEL_PARAMS_DICT:
        raise NotImplementedError(f"Model size {model_size} not supported.")

    model_params = MODEL_PARAMS_DICT[model_size]
    base_num_decoder_layers = model_params["base_num_decoder_layers"]
    base_emb_dim = model_params["base_emb_dim"]
    base_num_query_heads = model_params["base_num_query_heads"]
    base_num_kv_heads = model_params["base_num_kv_heads"]
    head_dim = model_params["head_dim"]
    num_experts = model_params["num_experts"]

    hf_state_dict = {}
    maxtext_params = training_state.params["params"]

    # Embedding
    hf_state_dict["model.embed_tokens.weight"] = torch.tensor(
        np.asarray(maxtext_params["token_embedder"]["embedding"]), dtype=torch.bfloat16
    )

    # Decoder Layers
    for idx in tqdm(range(base_num_decoder_layers), desc="Porting layers"):
        layer_params = maxtext_params["decoder"][f"layers_{idx}"]

        # LayerNorms
        hf_state_dict[f"model.layers.{idx}.input_layernorm.weight"] = torch.tensor(
            np.asarray(layer_params["pre_self_attention_layer_norm"]["scale"]), dtype=torch.bfloat16
        )
        hf_state_dict[f"model.layers.{idx}.post_attention_layernorm.weight"] = torch.tensor(
            np.asarray(layer_params["post_self_attention_layer_norm"]["scale"]), dtype=torch.bfloat16
        )

        # Attention
        attn = layer_params["GptOssAttention"]
        q_proj = np.asarray(attn["query"]["kernel"]).reshape(base_emb_dim, base_num_query_heads * head_dim).T
        k_proj = np.asarray(attn["key"]["kernel"]).reshape(base_emb_dim, base_num_kv_heads * head_dim).T
        v_proj = np.asarray(attn["value"]["kernel"]).reshape(base_emb_dim, base_num_kv_heads * head_dim).T
        o_proj = np.asarray(attn["out"]["kernel"]).reshape(base_num_query_heads * head_dim, base_emb_dim).T

        hf_state_dict[f"model.layers.{idx}.self_attn.q_proj.weight"] = torch.tensor(q_proj, dtype=torch.bfloat16)
        hf_state_dict[f"model.layers.{idx}.self_attn.k_proj.weight"] = torch.tensor(k_proj, dtype=torch.bfloat16)
        hf_state_dict[f"model.layers.{idx}.self_attn.v_proj.weight"] = torch.tensor(v_proj, dtype=torch.bfloat16)
        hf_state_dict[f"model.layers.{idx}.self_attn.o_proj.weight"] = torch.tensor(o_proj, dtype=torch.bfloat16)

        # Attention Biases
        q_bias = np.asarray(attn["query"]["bias"]).reshape(base_num_query_heads * head_dim)
        k_bias = np.asarray(attn["key"]["bias"]).reshape(base_num_kv_heads * head_dim)
        v_bias = np.asarray(attn["value"]["bias"]).reshape(base_num_kv_heads * head_dim)

        hf_state_dict[f"model.layers.{idx}.self_attn.q_proj.bias"] = torch.tensor(q_bias, dtype=torch.bfloat16)
        hf_state_dict[f"model.layers.{idx}.self_attn.k_proj.bias"] = torch.tensor(k_bias, dtype=torch.bfloat16)
        hf_state_dict[f"model.layers.{idx}.self_attn.v_proj.bias"] = torch.tensor(v_bias, dtype=torch.bfloat16)
        hf_state_dict[f"model.layers.{idx}.self_attn.o_proj.bias"] = torch.tensor(np.asarray(attn["out"]["bias"]), dtype=torch.bfloat16)
        
        # Sinks are not standard in HF models, so we skip them.
        
        # MoE MLP
        mlp = layer_params["GptOssMlp"]
        hf_state_dict[f"model.layers.{idx}.mlp.router.weight"] = torch.tensor(np.asarray(mlp["gate"]["kernel"]).T, dtype=torch.bfloat16)
        hf_state_dict[f"model.layers.{idx}.mlp.router.bias"] = torch.tensor(np.asarray(mlp["gate"]["bias"]), dtype=torch.bfloat16)

        # Interleave wi_0 and wi_1 to form the combined gate_up_proj tensor
        wi_0 = np.asarray(mlp["wi_0"])
        wi_1 = np.asarray(mlp["wi_1"])
        
        # The forward script slices on the last dimension, so we interleave on the last dimension
        s = wi_0.shape # (num_experts, hidden_dim, intermediate_dim)
        gate_up_proj = np.empty((s[0], s[1], s[2] * 2), dtype=wi_0.dtype)
        gate_up_proj[..., ::2] = wi_0
        gate_up_proj[..., 1::2] = wi_1
        hf_state_dict[f"model.layers.{idx}.mlp.experts.gate_up_proj"] = torch.tensor(gate_up_proj, dtype=torch.bfloat16)
        
        # Interleave biases
        wi_0_bias = np.asarray(mlp["wi_0_bias"])
        wi_1_bias = np.asarray(mlp["wi_1_bias"])
        s_bias = wi_0_bias.shape # (num_experts, intermediate_dim)
        gate_up_proj_bias = np.empty((s_bias[0], s_bias[1] * 2), dtype=wi_0_bias.dtype)
        gate_up_proj_bias[..., ::2] = wi_0_bias
        gate_up_proj_bias[..., 1::2] = wi_1_bias
        hf_state_dict[f"model.layers.{idx}.mlp.experts.gate_up_proj_bias"] = torch.tensor(gate_up_proj_bias, dtype=torch.bfloat16)
        
        # Down projection
        hf_state_dict[f"model.layers.{idx}.mlp.experts.down_proj"] = torch.tensor(np.asarray(mlp["wo"]), dtype=torch.bfloat16)
        hf_state_dict[f"model.layers.{idx}.mlp.experts.down_proj_bias"] = torch.tensor(np.asarray(mlp["wo_bias"]), dtype=torch.bfloat16)

    # Final LayerNorm and LM Head
    hf_state_dict["model.norm.weight"] = torch.tensor(
        np.asarray(maxtext_params["decoder"]["decoder_norm"]["scale"]), dtype=torch.bfloat16
    )
    hf_state_dict["lm_head.weight"] = torch.tensor(
        np.asarray(maxtext_params["decoder"]["logits_dense"]["kernel"]).T, dtype=torch.bfloat16
    )

    return hf_state_dict


def convert_orbax_to_hf(hf_model_path, config):
    """
    Main function to convert a MaxText checkpoint to HuggingFace format.
    It loads the state, converts the weights, and saves the model.
    """
    max_logging.log("Loading MaxText training state...")
    training_state = load_model_state(config)

    max_logging.log("Converting weights to Hugging Face format...")
    hf_state_dict = convert_state_to_hf(training_state, config.model_name)

    max_logging.log(f"Saving HuggingFace model to path: {hf_model_path}")
    os.makedirs(hf_model_path, exist_ok=True)

    # Save weights
    save_file(hf_state_dict, os.path.join(hf_model_path, "model.safetensors"))

    # Create and save HF config file
    hf_config_data = HF_CONFIG_MAP.get(config.model_name)
    if not hf_config_data:
        raise ValueError(f"No HF config available for model: {config.model_name}")

    with open(os.path.join(hf_model_path, "config.json"), "w") as f:
        json.dump(hf_config_data, f, indent=2)

    max_logging.log("Successfully saved model weights and config.")


def main(argv: Sequence[str]):
    # Pop the hf_model_path argument before initializing pyconfig
    hf_model_path_arg = None
    for i, arg in enumerate(argv):
        if arg.startswith("hf_model_path"):
            hf_model_path_arg = argv.pop(i)
            break

    if not hf_model_path_arg:
        raise ValueError("You must specify the `hf_model_path` argument.")

    hf_model_path = hf_model_path_arg.split("=")[1]
    
    pyconfig.initialize(argv)
    config = pyconfig.config

    convert_orbax_to_hf(hf_model_path, config)


if __name__ == "__main__":
    app.run(main)