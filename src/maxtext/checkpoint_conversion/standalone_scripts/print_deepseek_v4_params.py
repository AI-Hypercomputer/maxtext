import sys
import jax
import os
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.models import models
from maxtext.layers import quantizations

Transformer = models.transformer_as_linen

def get_path_string2(path):
  key_parts = [k.key for k in path if hasattr(k, "key")]
  param_key = "params." + ".".join(key_parts)
  return param_key

def main():
  argv = sys.argv
  if len(argv) < 2:
    argv = [
        '',
        'src/maxtext/configs/base.yml',
        'model_name=deepseek4',
        'override_model_config=True',
        'attention=dot_product',
        'skip_jax_distributed_system=True',
        'weight_dtype=bfloat16',
        'scan_layers=True',
    ]

  print("Initializing configuration...")
  config = pyconfig.initialize(argv)
  print(f"\n--- Inspecting MaxText Architecture: {config.model_name} (Scan: {config.scan_layers}) ---")
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh=mesh, quant=quant)

  # Get abstract params (no memory/compute)
  abstract_params = maxtext_utils.get_abstract_param(model, config)["params"]
  
  flat_params = jax.tree_util.tree_flatten_with_path(abstract_params)[0]
  
  output_file = "deepseek_v4_params_and_shapes.txt"
  print(f"Writing parameter names and shapes to {output_file}...")
  
  with open(output_file, "w") as f:
    for path, x in flat_params:
      name = get_path_string2(path)
      # x is a jax.ShapeDtypeStruct, so we can access x.shape and x.dtype
      f.write(f"{name}: shape={x.shape}, dtype={x.dtype}\n")
      
  print(f"Successfully wrote {len(flat_params)} parameters to {output_file}.")

if __name__ == "__main__":
  main()
