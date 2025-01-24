"""
 Copyright 2024 Google LLC
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

r"""Convert weights from a MaxText model to a HuggingFace model.

Usage:

python3 MaxText/gemma2_orbax_to_hf.py MaxText/configs/base.yml \
            base_output_directory=path/to/saving/intermediate_MaxText_files \
            load_parameters_path=/path/to/MaxText/checkpoint run_name=<your run name> model_name=<gemma2-variant> \
            async_checkpointing=false hf_model_path=/local/path/to/save/HF/model/to

"""


from typing import Sequence
import torch
from tqdm import tqdm
from absl import app
import numpy as np
import pyconfig
import max_utils
from jax.sharding import Mesh
import max_logging
import checkpointing
from generate_param_only_checkpoint import _read_train_checkpoint
from transformers import AutoModelForCausalLM, AutoConfig


def load_hf_model(model_size):
  """
  Load the model that we are interested in from HuggingFace

  """
  if model_size == "gemma2-2b":
    config = AutoConfig.from_pretrained("google/gemma-2-2b")
    model = AutoModelForCausalLM.from_config(config)
  elif model_size == "gemma2-9b":
    config = AutoConfig.from_pretrained("google/gemma-2-9b")
    model = AutoModelForCausalLM.from_config(config)
  elif model_size == "gemma2-27b":
    config = AutoConfig.from_pretrained("google/gemma-2-27b-it")
    model = AutoModelForCausalLM.from_config(config)
  else:
    raise NotImplementedError
  return model


def load_model_state(config):
  """
  Loads the MaxText model's TrainState from the Orbax checkpoint
  """
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Create a checkpoint manager to load decode checkpoint at config.checkpoint_dir
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      config.checkpoint_dir,
      config.enable_checkpointing,
      config.async_checkpointing,
      config.checkpoint_period,
  )

  # Read training state from config.load_paramaters_path
  max_logging.log(f"Read training checkpoint from: {config.load_full_state_path}")
  training_state, _ = _read_train_checkpoint(config, checkpoint_manager, mesh)
  return training_state


def convert_state_to_hf(training_state, model_size):
  """
  Port the parameters from the Orbax training_state into the hf_model
  """
  if model_size not in ("gemma2-2b", "gemma2-9b", "gemma2-27b"):
    raise NotImplementedError
  num_layers = training_state.params["params"]["decoder"]["layers"]["mlp_global"]["wi_0"]["kernel"].shape[1]
  _, _, embed_dim = training_state.params["params"]["decoder"]["layers"]["mlp_global"]["wo"]["kernel"].shape
  _, _, query_head, head_dim = training_state.params["params"]["decoder"]["layers"]["self_attention_local"]["query"][
      "kernel"
  ].shape
  kv_head = training_state.params["params"]["decoder"]["layers"]["self_attention_local"]["key"]["kernel"].shape[2]

  query_pre_attn_scalar = None
  if model_size in ("gemma2-2b", "gemma2-9b"):
    query_pre_attn_scalar = head_dim**-0.5
  elif model_size in ("gemma2-27b"):
    query_pre_attn_scalar = (embed_dim // query_head) ** -0.5

  hf_model_params = {}

  # removing last 128 tokens from vocab as HF considers them special tokens and is not part of the training
  hf_model_params["model.embed_tokens.weight"] = torch.tensor(
      np.asarray(training_state.params["params"]["token_embedder"]["embedding"])[:-128, :], dtype=torch.float32
  )

  for layer_int in tqdm(range(num_layers), desc="Porting parameters layerwise"):
    print(f"Converting weights for layer {layer_int}")
    # local attention layer
    hf_layer = layer_int * 2
    hf_model_params[f"model.layers.{hf_layer}.self_attn.q_proj.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["layers"]["self_attention_local"]["query"]["kernel"][
                :, layer_int, :, :
            ]
        )
        .reshape(embed_dim, query_head * head_dim)
        .T
        / query_pre_attn_scalar,
        dtype=torch.float32,
    )
    hf_model_params[f"model.layers.{hf_layer}.self_attn.k_proj.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["layers"]["self_attention_local"]["key"]["kernel"][:, layer_int, :, :]
        )
        .reshape(embed_dim, kv_head * head_dim)
        .T,
        dtype=torch.float32,
    )
    hf_model_params[f"model.layers.{hf_layer}.self_attn.v_proj.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["layers"]["self_attention_local"]["value"]["kernel"][
                :, layer_int, :, :
            ]
        )
        .reshape(embed_dim, kv_head * head_dim)
        .T,
        dtype=torch.float32,
    )
    hf_model_params[f"model.layers.{hf_layer}.self_attn.o_proj.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["layers"]["self_attention_local"]["out"]["kernel"][:, layer_int, :, :]
        )
        .transpose((2, 0, 1))
        .reshape(embed_dim, query_head * head_dim),
        dtype=torch.float32,
    )
    # local mlp layer
    hf_model_params[f"model.layers.{hf_layer}.mlp.gate_proj.weight"] = torch.tensor(
        np.asarray(training_state.params["params"]["decoder"]["layers"]["mlp_local"]["wi_0"]["kernel"][:, layer_int, :]).T
    )
    hf_model_params[f"model.layers.{hf_layer}.mlp.up_proj.weight"] = torch.tensor(
        np.asarray(training_state.params["params"]["decoder"]["layers"]["mlp_local"]["wi_1"]["kernel"][:, layer_int, :]).T
    )
    hf_model_params[f"model.layers.{hf_layer}.mlp.down_proj.weight"] = torch.tensor(
        np.asarray(training_state.params["params"]["decoder"]["layers"]["mlp_local"]["wo"]["kernel"][:, layer_int, :]).T
    )
    # local layer norms
    hf_model_params[f"model.layers.{hf_layer}.input_layernorm.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["layers"]["pre_self_attention_norm_local"]["scale"][:, layer_int]
        )
        - 1,
        dtype=torch.float32,
    )
    hf_model_params[f"model.layers.{hf_layer}.post_attention_layernorm.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["layers"]["post_self_attention_norm_local"]["scale"][:, layer_int]
        )
        - 1,
        dtype=torch.float32,
    )
    hf_model_params[f"model.layers.{hf_layer}.pre_feedforward_layernorm.weight"] = torch.tensor(
        np.asarray(training_state.params["params"]["decoder"]["layers"]["pre_ffw_norm_local"]["scale"][:, layer_int]) - 1,
        dtype=torch.float32,
    )
    hf_model_params[f"model.layers.{hf_layer}.post_feedforward_layernorm.weight"] = torch.tensor(
        np.asarray(training_state.params["params"]["decoder"]["layers"]["post_ffw_norm_local"]["scale"][:, layer_int]) - 1,
        dtype=torch.float32,
    )
    # Global Attention and MLP layers
    hf_layer = layer_int * 2 + 1
    hf_model_params[f"model.layers.{hf_layer}.self_attn.q_proj.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["layers"]["self_attention_global"]["query"]["kernel"][
                :, layer_int, :, :
            ]
        )
        .reshape(embed_dim, query_head * head_dim)
        .T
        / query_pre_attn_scalar,
        dtype=torch.float32,
    )
    hf_model_params[f"model.layers.{hf_layer}.self_attn.k_proj.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["layers"]["self_attention_global"]["key"]["kernel"][
                :, layer_int, :, :
            ]
        )
        .reshape(embed_dim, kv_head * head_dim)
        .T,
        dtype=torch.float32,
    )
    hf_model_params[f"model.layers.{hf_layer}.self_attn.v_proj.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["layers"]["self_attention_global"]["value"]["kernel"][
                :, layer_int, :, :
            ]
        )
        .reshape(embed_dim, kv_head * head_dim)
        .T,
        dtype=torch.float32,
    )
    hf_model_params[f"model.layers.{hf_layer}.self_attn.o_proj.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["layers"]["self_attention_global"]["out"]["kernel"][
                :, layer_int, :, :
            ]
        )
        .transpose((2, 0, 1))
        .reshape(embed_dim, query_head * head_dim),
        dtype=torch.float32,
    )
    # local mlp layer
    hf_model_params[f"model.layers.{hf_layer}.mlp.gate_proj.weight"] = torch.tensor(
        np.asarray(training_state.params["params"]["decoder"]["layers"]["mlp_global"]["wi_0"]["kernel"][:, layer_int, :]).T
    )
    hf_model_params[f"model.layers.{hf_layer}.mlp.up_proj.weight"] = torch.tensor(
        np.asarray(training_state.params["params"]["decoder"]["layers"]["mlp_global"]["wi_1"]["kernel"][:, layer_int, :]).T
    )
    hf_model_params[f"model.layers.{hf_layer}.mlp.down_proj.weight"] = torch.tensor(
        np.asarray(training_state.params["params"]["decoder"]["layers"]["mlp_global"]["wo"]["kernel"][:, layer_int, :]).T
    )
    # local layer norms
    hf_model_params[f"model.layers.{hf_layer}.input_layernorm.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["layers"]["pre_self_attention_norm_global"]["scale"][:, layer_int]
        )
        - 1,
        dtype=torch.float32,
    )
    hf_model_params[f"model.layers.{hf_layer}.post_attention_layernorm.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["layers"]["post_self_attention_norm_global"]["scale"][:, layer_int]
        )
        - 1,
        dtype=torch.float32,
    )
    hf_model_params[f"model.layers.{hf_layer}.pre_feedforward_layernorm.weight"] = torch.tensor(
        np.asarray(training_state.params["params"]["decoder"]["layers"]["pre_ffw_norm_global"]["scale"][:, layer_int]) - 1,
        dtype=torch.float32,
    )
    hf_model_params[f"model.layers.{hf_layer}.post_feedforward_layernorm.weight"] = torch.tensor(
        np.asarray(training_state.params["params"]["decoder"]["layers"]["post_ffw_norm_global"]["scale"][:, layer_int]) - 1,
        dtype=torch.float32,
    )

  # LM head and layernorm
  hf_model_params["lm_head.weight"] = torch.tensor(
      np.asarray(training_state.params["params"]["token_embedder"]["embedding"][:-128, :]), dtype=torch.float32
  )
  hf_model_params["model.norm.weight"] = torch.tensor(
      np.asarray(training_state.params["params"]["decoder"]["decoder_norm"]["scale"]),
      dtype=torch.float32,
  )
  return hf_model_params


def convert_orbax_hf(hf_model_path, config):
  """
  Landing function to convert MaxText model's checkpoint to HuggingFace format
  """
  hf_model = load_hf_model(config.model_name)
  training_state = load_model_state(config)
  new_hf_model_params = convert_state_to_hf(training_state, config.model_name)
  print(f"Saving HuggingFace model to path = {hf_model_path}")
  hf_model.save_pretrained(hf_model_path, state_dict=new_hf_model_params)


def main(argv: Sequence[str]):
  pyconfig.initialize(argv[:-1])
  hf_model_path = argv[-1].split("=")[1]
  print(f"Will save converted HuggingFace checkpoint to path = {hf_model_path}")

  convert_orbax_hf(hf_model_path, pyconfig.config)


if __name__ == "__main__":
  app.run(main)
