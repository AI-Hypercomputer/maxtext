# Copyright 2026 Google LLC
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

"""
Phase 3 Modular Block 3: Parity Comparative Auditor.
Loads cached PyTorch and MaxText dumped activations, evaluates MAE element-by-element,
and asserts strict convergence constraints.
"""

import sys
import numpy as np
import jax # Auto-registers bfloat16 data type in numpy


def compare(py_val, jax_val, name, threshold=1e-4):
  """Calculates Mean Absolute Error (MAE) between PyTorch and JAX arrays."""
  print(f"\n=== Audit Step: Comparing {name} ===")

  if py_val.shape != jax_val.shape:
    if py_val.size == jax_val.size:
      print(f"Notice: Shapes differ but sizes match. Reshaping JAX from {jax_val.shape} to {py_val.shape}.")
      jax_val = jax_val.reshape(py_val.shape)
    else:
      print(f"CRITICAL ERROR: Shape mismatch for {name}: PyTorch {py_val.shape} vs JAX {jax_val.shape}", file=sys.stderr)
      return False

  mae = np.mean(np.abs(py_val - jax_val))
  max_diff = np.max(np.abs(py_val - jax_val))

  print(f"PyTorch {name} first token slice:\n{py_val[0, :5]}")
  print(f"JAX {name} first token slice:\n{jax_val[0, :5]}")
  print(f"Mean Absolute Error (MAE): {mae:.8f}")
  print(f"Max Absolute Difference: {max_diff:.8f}")

  if mae > threshold:
    print(f"ALERT: {name} Parity DIVERGES! MAE exceeds tolerance threshold of {threshold:.8f}", file=sys.stderr)
    return False

  print(f"SUCCESS: {name} Parity holds cleanly.")
  return True


def main():
  print("=== Logit Parity Multi-Layered Comparative Audit ===")

  # Sequential intermediate audit steps
  steps = [
      ("chosen_embeds", 1e-5),
      ("chosen_q_proj", 1e-4),
      ("rejected_q_proj", 1e-4),
      ("chosen_k_proj", 1e-4),
      ("rejected_k_proj", 1e-4),
      ("chosen_v_proj", 1e-4),
      ("rejected_v_proj", 1e-4),
      ("chosen_o_proj", 1e-4),
      ("rejected_o_proj", 1e-4),
      ("chosen_mlp_out", 1e-4),
      ("rejected_mlp_out", 1e-4),
      ("chosen_logits", 1e-4),
      ("rejected_logits", 1e-4),
  ]

  all_pass = True
  for step, threshold in steps:
    try:
      py_arr = np.load(f"quals/logs/pytorch_{step}.npy")
      jax_arr = np.load(f"quals/logs/maxtext_{step}.npy")

      success = compare(py_arr, jax_arr, step.upper(), threshold=threshold)
      if not success:
        all_pass = False
    except FileNotFoundError:
      print(f"Warning: Missing audit dump file for step: {step.upper()}", file=sys.stderr)
      all_pass = False

  if all_pass:
    print("\n=== FINAL PARITY VERIFICATION SUCCESSFUL ===")
    sys.exit(0)
  else:
    print("\n=== FINAL PARITY VERIFICATION FAILED ===")
    sys.exit(1)


if __name__ == "__main__":
  main()
