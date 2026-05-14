"""
Module for converting Qwen 2.5 checkpoints to NNX format.
"""

from flax import nnx
import orbax.checkpoint as ocp
from etils import epath
import torch
from transformers import AutoModelForCausalLM
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils, model_creation_utils
import numpy as np


def convert():
  """
  Main conversion function to map HF weights to MaxText NNX structure.
  """
  hf_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
  output_path = "gs://igorts_europe/converted_checkpoints/qwen2.5-1.5b-instruct_nnx_v1"

  print(f"Loading HF model config: {hf_model_id}")

  # Initialize MaxText config
  argv = ["src/maxtext/configs/base.yml", "model_name=qwen2.5-1.5b", "scan_layers=False", "attention=dot_product"]
  mt_config = pyconfig.initialize(argv)
  mesh = maxtext_utils.get_mesh_from_config(mt_config)

  # Inject logical axes
  mt_config.mesh_axes.append("norm")
  mt_config.ici_parallelism.append(1)
  mt_config.dcn_parallelism.append(1)

  print("Creating MaxText model (abstract)...")
  with mesh:

    def _create_model():
      return model_creation_utils.create_model(mt_config, mesh, rngs=nnx.Rngs(0))

    abstract_model = nnx.eval_shape(_create_model)
    state = nnx.state(abstract_model)

  print(f"Loading HF weights from {hf_model_id}")
  # We use torch to load weights as it is easiest for HF format
  hf_model = AutoModelForCausalLM.from_pretrained(
      hf_model_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
  )
  hf_state = hf_model.state_dict()

  def pt_to_np(pt_tensor):
    return pt_tensor.detach().cpu().to(torch.float32).numpy().astype(np.float32)

  print("Mapping weights...")
  new_state_dict = {}

  # Mapping logic for Qwen 2.5
  # Token embedder
  new_state_dict[("token_embedder", "embedding", "value")] = pt_to_np(hf_state["model.embed_tokens.weight"])

  # Decoder norm
  new_state_dict[("decoder", "decoder_norm", "scale", "value")] = pt_to_np(hf_state["model.norm.weight"])

  # Layers
  n_layers = mt_config.num_decoder_layers
  for i in range(n_layers):
    prefix_hf = f"model.layers.{i}."
    prefix_mt = ("decoder", "layers", i)

    # Self attention
    new_state_dict[prefix_mt + ("self_attention", "query", "kernel", "value")] = pt_to_np(
        hf_state[prefix_hf + "self_attn.q_proj.weight"]
    ).transpose()
    new_state_dict[prefix_mt + ("self_attention", "key", "kernel", "value")] = pt_to_np(
        hf_state[prefix_hf + "self_attn.k_proj.weight"]
    ).transpose()
    new_state_dict[prefix_mt + ("self_attention", "value", "kernel", "value")] = pt_to_np(
        hf_state[prefix_hf + "self_attn.v_proj.weight"]
    ).transpose()
    new_state_dict[prefix_mt + ("self_attention", "out", "kernel", "value")] = pt_to_np(
        hf_state[prefix_hf + "self_attn.o_proj.weight"]
    ).transpose()

    new_state_dict[prefix_mt + ("self_attention", "query", "bias", "value")] = pt_to_np(
        hf_state[prefix_hf + "self_attn.q_proj.bias"]
    )
    new_state_dict[prefix_mt + ("self_attention", "key", "bias", "value")] = pt_to_np(
        hf_state[prefix_hf + "self_attn.k_proj.bias"]
    )
    new_state_dict[prefix_mt + ("self_attention", "value", "bias", "value")] = pt_to_np(
        hf_state[prefix_hf + "self_attn.v_proj.bias"]
    )

    # MLP
    new_state_dict[prefix_mt + ("mlp", "wi_0", "kernel", "value")] = pt_to_np(
        hf_state[prefix_hf + "mlp.gate_proj.weight"]
    ).transpose()
    new_state_dict[prefix_mt + ("mlp", "wi_1", "kernel", "value")] = pt_to_np(
        hf_state[prefix_hf + "mlp.up_proj.weight"]
    ).transpose()
    new_state_dict[prefix_mt + ("mlp", "wo", "kernel", "value")] = pt_to_np(
        hf_state[prefix_hf + "mlp.down_proj.weight"]
    ).transpose()

    # Norms
    new_state_dict[prefix_mt + ("pre_self_attention_norm", "scale", "value")] = pt_to_np(
        hf_state[prefix_hf + "input_layernorm.weight"]
    )
    new_state_dict[prefix_mt + ("post_self_attention_norm", "scale", "value")] = pt_to_np(
        hf_state[prefix_hf + "post_attention_layernorm.weight"]
    )

  # Final logic: check if anything missing
  print("Verification...")
  missing = []
  for path, _ in state.items():
    if path not in new_state_dict:
      # Some variables might be RNG or states we don't need to restore
      if "rng" not in str(path):
        missing.append(path)

  if missing:
    print(f"Warning: {len(missing)} variables missing from mapping (e.g. {missing[0]})")

  # Save using Orbax
  print(f"Saving to {output_path}...")

  # NNX Checkpoint structure: just the state items
  checkpoint_data = {}
  for path, value in new_state_dict.items():
    # Reconstruct nested dict for Orbax
    curr = checkpoint_data
    for p in path[:-1]:
      if p not in curr:
        curr[p] = {}
      curr = curr[p]
    curr[path[-1]] = value

  # Add step
  checkpoint_data["step"] = 0

  # Modern Orbax save
  ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))
  save_path = epath.Path(output_path) / "0" / "items"
  ckptr.save(save_path, checkpoint_data)

  # Create commit_success.txt to mark as valid
  with epath.Path(output_path) / "0" / "commit_success.txt" as f:
    f.write_text("")

  print("Done!")


if __name__ == "__main__":
  convert()
