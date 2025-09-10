import jax
from jax import numpy as jnp
import os



debug_dir = os.path.join(os.environ["HOME"], "debug-oss")
debug_jax = os.path.join(debug_dir, "jax")
debug_pt = os.path.join(debug_dir, "pt")



def load_array(debug_jax):
  loaded_arrays = {}
  # Check if the directory exists before proceeding
  if os.path.isdir(debug_jax):
      print(debug_jax)
      print(os.listdir(debug_jax))
      # Loop through each item in the specified directory
      for filename in os.listdir(debug_jax):
          # We only want to load .npy or .npz files
          if filename.endswith('.npy'):
              file_path = os.path.join(debug_jax, filename)
              try:
                  # Load the array from the file
                  array_data = jnp.load(file_path)
                  # Add the array to the dictionary with its filename as the key
                  loaded_arrays[filename[:-4]] = array_data
                  print(f"✅ Loaded '{filename}'")
              except Exception as e:
                  print(f"❌ Failed to load '{filename}': {e}")
  else:
      print(f"Directory not found: {debug_jax}")
  return loaded_arrays



jax_dict = load_array(debug_jax)
print("jax", jax_dict)

pt_dict = load_array(debug_pt)
print("pt", pt_dict)
