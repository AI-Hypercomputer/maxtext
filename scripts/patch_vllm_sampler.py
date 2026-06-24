"""Hotfix to disable prompt_logprobs in tunix vLLM Sampler and bypass vLLM V1 scheduler AssertionError."""

import os
import sys

# --- Patch 1: tunix vllm_sampler.py ---
print("Applying hotfix to tunix/generate/vllm_sampler.py...")
p = "/usr/local/lib/python3.12/site-packages/tunix/generate/vllm_sampler.py"
if not os.path.exists(p):
  try:
    import tunix.generate.vllm_sampler as vs

    p = vs.__file__
  except ImportError:
    print("Error: tunix not found.")
    sys.exit(1)

print(f"Target file: {p}")
with open(p, "r", encoding="utf-8") as f:
  c = f.read()
target = (
    "      if self.config.return_logprobs:\n"
    "        sampling_params.logprobs = 1  # b/428730696\n"
    "        sampling_params.prompt_logprobs = 1  # b/428730696"
)
replacement = (
    "      if self.config.return_logprobs:\n"
    "        sampling_params.logprobs = 1  # b/428730696\n"
    "        sampling_params.prompt_logprobs = 0"
)

if target in c:
  c = c.replace(target, replacement)
  with open(p, "w", encoding="utf-8") as f:
    f.write(c)
  print("Hotfix successfully applied to tunix/generate/vllm_sampler.py.")
else:
  if "sampling_params.prompt_logprobs = 0" in c:
    print("Hotfix already applied.")
  else:
    print("Warning: target string not found.")

# --- Patch 2: vllm scheduler.py ---
print("Applying hotfix to vllm/v1/core/sched/scheduler.py...")
p2 = "/usr/local/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"
if os.path.exists(p2):
  with open(p2, "r", encoding="utf-8") as f:
    c2 = f.read()
  target2 = (
      "                # Invariant: EngineCore returns no partial prefill"
      " outputs.\n                assert not prompt_logprobs_tensors"
  )
  replacement2 = "                # Invariant: EngineCore returns no partial prefill" " outputs.\n                pass"
  if target2 in c2:
    c2 = c2.replace(target2, replacement2)
    with open(p2, "w", encoding="utf-8") as f:
      f.write(c2)
    print("Hotfix successfully applied to vllm/v1/core/sched/scheduler.py.")
  else:
    # Check if the target is slightly different (e.g. fewer spaces or different lines)
    # We can also do a simpler string replacement
    target2_alt = "assert not prompt_logprobs_tensors"
    if target2_alt in c2:
      c2 = c2.replace(target2_alt, "pass")
      with open(p2, "w", encoding="utf-8") as f:
        f.write(c2)
      print("Hotfix successfully applied to vllm/v1/core/sched/scheduler.py (alt match).")
    else:
      print("Warning: target string not found in scheduler.py.")
else:
  print("Warning: scheduler.py path not found.")

# --- Patch 3: tpu_inference utils.py ---
print("Applying hotfix to tpu_inference/runner/utils.py...")
p3 = "/usr/local/lib/python3.12/site-packages/tpu_inference/runner/utils.py"
if os.path.exists(p3):
  with open(p3, "r", encoding="utf-8") as f:
    c3 = f.read()

  # Replace the ints with strings in advanced_configuration dicts
  targets_and_replacements = [
      (
          '            "tpu_num_sparse_cores_to_trace": 1,\n' '            "tpu_num_sparse_core_tiles_to_trace": 1,',
          '            "tpu_num_sparse_cores_to_trace": "1",\n' '            "tpu_num_sparse_core_tiles_to_trace": "1",',
      ),
      (
          '                "tpu_num_chips_to_profile_per_task": 1,\n'
          '                "tpu_num_sparse_cores_to_trace": 1,\n'
          '                "tpu_num_sparse_core_tiles_to_trace": 1,',
          '                "tpu_num_chips_to_profile_per_task": "1",\n'
          '                "tpu_num_sparse_cores_to_trace": "1",\n'
          '                "tpu_num_sparse_core_tiles_to_trace": "1",',
      ),
  ]

  modified = False
  for target3, replacement3 in targets_and_replacements:
    if target3 in c3:
      c3 = c3.replace(target3, replacement3)
      modified = True

  if modified:
    with open(p3, "w", encoding="utf-8") as f:
      f.write(c3)
    print("Hotfix successfully applied to tpu_inference/runner/utils.py.")
  else:
    print("Warning: target strings not found or already patched in tpu_inference/runner/utils.py.")
else:
  print("Warning: tpu_inference/runner/utils.py path not found.")

# --- Patch 4: pathwaysutils profiling.py ---
print("Applying hotfix to pathwaysutils/profiling.py...")
p4 = "/usr/local/lib/python3.12/site-packages/pathwaysutils/profiling.py"
if os.path.exists(p4):
  with open(p4, "r", encoding="utf-8") as f:
    c4 = f.read()

  target4 = "      _, result_future = _profile_state.executable.call()"
  replacement4 = (
      "      import jax\n"
      "      import jax.numpy as jnp\n"
      "      out_avals = [jax.core.ShapedArray((1,), jnp.object_)]\n"
      "      out_shardings = [\n"
      "          getattr(\n"
      "              jax.sharding,\n"
      "              'make_single_device_sharding',\n"
      "              jax.sharding.SingleDeviceSharding,\n"
      "          )(jax.devices()[0])\n"
      "      ]\n"
      "      _, result_future = _profile_state.executable.call(\n"
      "          out_avals=out_avals, out_shardings=out_shardings\n"
      "      )"
  )

  if target4 in c4:
    c4 = c4.replace(target4, replacement4)
    with open(p4, "w", encoding="utf-8") as f:
      f.write(c4)
    print("Hotfix successfully applied to pathwaysutils/profiling.py.")
  else:
    print("Warning: target string not found in pathwaysutils/profiling.py.")
else:
  print("Warning: pathwaysutils/profiling.py path not found.")
