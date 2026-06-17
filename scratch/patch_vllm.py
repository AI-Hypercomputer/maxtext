"""Runtime integration patching script for MaxText/vLLM on GKE."""

import os
import sys


def patch_file(filepath, target_str, replacement_str):
  """Replaces the first occurrence of target_str with replacement_str in filepath."""
  if not os.path.exists(filepath):
    print(f"File not found: {filepath}")
    return False
  with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()
  if target_str not in content:
    print(f"Target string not found in {filepath}")
    return False
  # Only patch if not already patched
  if replacement_str in content:
    print(f"File {filepath} already patched")
    return True
  new_content = content.replace(target_str, replacement_str, 1)
  with open(filepath, "w", encoding="utf-8") as f:
    f.write(new_content)
  print(f"Patched {filepath} successfully")
  return True


# Find package installation paths
vllm_path = None
tpu_inf_path = None
tunix_path = None

for path in sys.path:
  if not vllm_path and os.path.exists(os.path.join(path, "vllm")):
    vllm_path = os.path.join(path, "vllm")
  if not tpu_inf_path and os.path.exists(os.path.join(path, "tpu_inference")):
    tpu_inf_path = os.path.join(path, "tpu_inference")
  if not tunix_path and os.path.exists(os.path.join(path, "tunix")):
    tunix_path = os.path.join(path, "tunix")

# Fallbacks for standard docker locations if sys.path check missed them
common_locations = ["/usr/local/lib/python3.12/site-packages", "/usr/local/lib/python3.11/site-packages"]
for loc in common_locations:
  if not vllm_path and os.path.exists(os.path.join(loc, "vllm")):
    vllm_path = os.path.join(loc, "vllm")
  if not tpu_inf_path and os.path.exists(os.path.join(loc, "tpu_inference")):
    tpu_inf_path = os.path.join(loc, "tpu_inference")
  if not tunix_path and os.path.exists(os.path.join(loc, "tunix")):
    tunix_path = os.path.join(loc, "tunix")

print(f"vLLM path: {vllm_path}")
print(f"tpu_inference path: {tpu_inf_path}")
print(f"tunix path: {tunix_path}")

# 1. Patch tpu-inference dp_scheduler.py
if tpu_inf_path:
  dp_scheduler_path = os.path.join(tpu_inf_path, "core", "sched", "dp_scheduler.py")
  target_dp = """    if "hash_block_size" in sig.parameters:
        scheduler_kwargs["hash_block_size"] = hash_block_size"""
  replacement_dp = """    has_kwargs = any(
        p.kind.name == 'VAR_KEYWORD' for p in sig.parameters.values()
    )
    if "hash_block_size" in sig.parameters or has_kwargs:
        scheduler_kwargs["hash_block_size"] = hash_block_size"""
  patch_file(dp_scheduler_path, target_dp, replacement_dp)

# 2. Patch tunix generate/utils.py
if tunix_path:
  utils_path = os.path.join(tunix_path, "generate", "utils.py")

  target_utils_1 = """  # Get flat target state
  tgt_flat_list = dst_state.flat_state()"""

  replacement_utils_1 = """  is_dict_dst = isinstance(dst_state, dict)
  if is_dict_dst:
      class DictVariable:
          def __init__(self, d, k):
              self.d = d
              self.k = k
          @property
          def value(self):
              return self.d[self.k]
          @value.setter
          def value(self, val):
              self.d[self.k] = val
          @property
          def sharding(self):
              return getattr(self.d[self.k], "sharding", None)
      tgt_flat_list = []
      for k in dst_state.keys():
          path_tuple = tuple(k.split("."))
          tgt_flat_list.append((path_tuple, DictVariable(dst_state, k)))
  else:
      tgt_flat_list = dst_state.flat_state()"""

  patch_file(utils_path, target_utils_1, replacement_utils_1)

  target_utils_2 = """  return dst_state.from_flat_path(tgt_flat_list)"""
  replacement_utils_2 = """  if isinstance(dst_state, dict):
      return dst_state
  return dst_state.from_flat_path(tgt_flat_list)"""
  patch_file(utils_path, target_utils_2, replacement_utils_2)

# 3. Patch tunix generate/vllm_sampler.py
if tunix_path:
  sampler_path = os.path.join(tunix_path, "generate", "vllm_sampler.py")
  target_sampler = """    args = config._processed_engine_kwargs.copy()"""
  replacement_sampler = """    args = config._processed_engine_kwargs.copy()
    args["limit_mm_per_prompt"] = {"image": 0, "video": 0}"""
  patch_file(sampler_path, target_sampler, replacement_sampler)

print("All integration patches processed.")
