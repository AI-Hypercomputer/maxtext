import os
import sys

# Add maxtext to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext import models
from maxtext.layers import quantizations
import functools
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

def main():
  # Setup config
  argv = [
      "check_sharding.py",
      "src/maxtext/configs/base.yml",
      "model_name=qwen1.5-moe-a2.7b",
      "ici_fsdp_parallelism=4",
      "sparse_matmul=True",
      "run_name=test_sharding",
      "base_output_directory=/tmp/test_sharding",
      "skip_jax_distributed_system=True"
  ]
  config = pyconfig.initialize(argv)
  print(f"logical_axis_rules: {config.logical_axis_rules}")
  print(f"config.ici_parallelism: {config.ici_parallelism}")
  print(f"config.mesh_axes: {config.mesh_axes}")

  # Create mesh
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  print(f"Mesh shape: {devices_array.shape}")
  print(f"Mesh axes: {config.mesh_axes}")

  # Create model
  quant = quantizations.configure_quantization(config)
  maxtext_model = models.transformer_as_linen(config, mesh=mesh, quant=quant, model_mode="train")

  # Get abstract state
  rng = jax.random.PRNGKey(0)
  init_state_fn = functools.partial(maxtext_utils.init_initial_state, maxtext_model, None, config, False, rng)
  
  # Replicate get_abstract_state logic to get logical annotations
  with nn_partitioning.axis_rules(config.logical_axis_rules):
    abstract_state = jax.eval_shape(init_state_fn)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  
  # Also get mesh annotations for comparison
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)

  # Helper to print trees
  def print_tree(tree, name, path="", indent=0):
    if isinstance(tree, dict) or hasattr(tree, 'keys'):
      for k in tree.keys():
        new_path = f"{path}/{k}" if path else str(k)
        print("  " * indent + str(k))
        print_tree(tree[k], name, new_path, indent + 1)
    else:
      print("  " * indent + f"{name}: {tree}")
      if "moe_block" in path and "wi_0" in path:
         print("  " * indent + f"--> TARGET FOUND: {path}")

  print("\nLogical Annotations Tree:")
  print_tree(state_logical_annotations.params, "Logical", "", 0)

  print("\nMesh Annotations Tree:")
  print_tree(state_mesh_annotations.params, "Mesh", "", 0)

if __name__ == "__main__":
  main()
