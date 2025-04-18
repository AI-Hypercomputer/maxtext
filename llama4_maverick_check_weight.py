from transformers import AutoModelForCausalLM

model_id = "/mnt/disks/jacobplatin/models/llama4/maverick/4-layer-debug-hf/HF-4layers/"
# model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct-Original"
hf_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="float32",
)
# pass
# load in HF and check token embedding
# load in PT weights as debug and playaround to see if match

# import torch
# import pathlib
# chkpt_vars = {}
# base_model_path = "/mnt/disks/jacobplatin/models/llama4/maverick/hf-non-instruct/"
# ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.pth"))
# for i, ckpt_path in enumerate(ckpt_paths):
#   print(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
#   # NOTE: starting in PT2.6, `weights_only` was switched from the default of `False` to `True`
#   # thus we need to specify this or else loading will fail
#   chkpt_vars[int(ckpt_path.name.split(".", maxsplit=2)[1])] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
# chkpt_vars = [chkpt_vars[i] for i in sorted(list(chkpt_vars.keys()))]


# Llama4 JAX cmd:
# JAX_PLATFORMS=cuda,cpu python -m MaxText.tests.llama4_logit_verification_script jax MaxText/configs/base.yml hardware=gpu
# scan_layers=false base_output_directory=llama4 run_name=temp-testing-only model_name=llama4-17b-16e
# force_unroll=false weight_dtype=float32 sparse_matmul=false megablox=false tokenizer_path="meta-llama/Llama-4-Scout-17B-16E"
# max_target_length=16 max_prefill_predict_length=4 per_device_batch_size=1 dtype=float32
# load_parameters_path=...

import jax
import jax.numpy as jnp
import numpy as np

import MaxText.layers.models as models
import MaxText.layers.quantizations as quantizations
from MaxText import pyconfig
from MaxText import max_utils

import argparse
import sys

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("backend_type", type=str, choices=["jax", "pt", "hf"])
  test_args, _ = parser.parse_known_args()

  model_args = sys.argv
  backend_type = model_args.pop(1).lower()
  print(model_args)
  config = pyconfig.initialize(model_args)

  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  init_rng, rng1 = jax.random.split(init_rng)
  devices_array = max_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  model = models.Transformer(config, mesh=mesh, quant=quant)
  state, _ = max_utils.setup_decode_state(model, config, rng1, mesh, None)
  pass
