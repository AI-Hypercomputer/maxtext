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
Phase 3.5 Modular Block 3: Full DPO Stack comparative auditor.
Compares tokenizer padding token IDs, loss masks, next-token log-probabilities,
and final preference losses mathematically.
"""

import sys
import numpy as np

def compare_tensors(py_val, jax_val, name, threshold=1e-5, check_exact=False):
  """Compares PyTorch and JAX tensors mathematically and prints Mean Absolute Error (MAE)."""
  print(f"\n=== Audit Step: Comparing DPO {name} ===")
  
  if py_val.shape != jax_val.shape:
    print(f"CRITICAL ERROR: Shape mismatch for {name}: PyTorch {py_val.shape} vs JAX {jax_val.shape}", file=sys.stderr)
    return False

  if check_exact:
    mismatch = np.sum(py_val != jax_val)
    print(f"Strict Equality Check: {mismatch} mismatches out of {py_val.size} elements.")
    if mismatch > 0:
      print(f"PyTorch {name} sample:\n{py_val[..., -10:]}")
      print(f"JAX {name} sample:\n{jax_val[..., -10:]}")
      return False
    print(f"SUCCESS: {name} is exactly identical.")
    return True

  mae = np.mean(np.abs(py_val - jax_val))
  max_diff = np.max(np.abs(py_val - jax_val))
  
  print(f"PyTorch {name} values:\n{py_val}")
  print(f"JAX {name} values:\n{jax_val}")
  print(f"Mean Absolute Error (MAE): {mae:.8f}")
  print(f"Max Absolute Difference: {max_diff:.8f}")
  
  if mae > threshold:
    print(f"ALERT: {name} Parity DIVERGES! MAE exceeds tolerance threshold of {threshold:.8f}", file=sys.stderr)
    return False
  
  print(f"SUCCESS: {name} Parity holds cleanly.")
  return True


def main():
  print("=== DPO Full-Stack Cross-Framework Parity Audit ===")
  
  # 1. Input Preprocessing & Padding Parity Checks (Strict Equality Checks)
  input_steps = [
      ('dpo_chosen_ids', "Token IDs (Chosen)"),
      ('dpo_rejected_ids', "Token IDs (Rejected)"),
      ('dpo_chosen_mask', "Response Loss Mask (Chosen)"),
      ('dpo_rejected_mask', "Response Loss Mask (Rejected)")
  ]
  
  all_pass = True
  for step, label in input_steps:
    try:
      py_arr = np.load(f"quals/logs/pytorch_{step}.npy")
      jax_arr = np.load(f"quals/logs/maxtext_{step}.npy")
      
      # Squeeze batch dimensions if present to ensure identical rank
      py_arr = np.squeeze(py_arr)
      jax_arr = np.squeeze(jax_arr)
      
      success = compare_tensors(py_arr, jax_arr, label.upper(), check_exact=True)
      if not success:
        all_pass = False
    except FileNotFoundError:
      print(f"Warning: Missing input audit dump file for step: {step.upper()}", file=sys.stderr)
      all_pass = False

  # 2. Mathematical Logps, Losses, & Metrics Parity Checks (MAE Checks)
  math_steps = [
      ('dpo_chosen_logps', "Policy Next-Token Log-Probabilities (Chosen)", 1e-4),
      ('dpo_rejected_logps', "Policy Next-Token Log-Probabilities (Rejected)", 1e-4),
      ('dpo_loss', "Standard DPO Preference Loss Value", 1e-4),
      ('dpo_margin', "Rewards Implicit Margin Value", 1e-4),
      ('dpo_accuracy', "Implicit Rewards Margin Accuracy", 1e-5)
  ]

  for step, label, threshold in math_steps:
    try:
      py_arr = np.load(f"quals/logs/pytorch_{step}.npy")
      jax_arr = np.load(f"quals/logs/maxtext_{step}.npy")
      
      success = compare_tensors(py_arr, jax_arr, label.upper(), threshold=threshold)
      if not success:
        all_pass = False
    except FileNotFoundError:
      print(f"Warning: Missing math audit dump file for step: {step.upper()}", file=sys.stderr)
      all_pass = False

  if all_pass:
    print("\n=== DPO FULL-STACK PARITY VERIFICATION SUCCESSFUL ===")
    sys.exit(0)
  else:
    print("\n=== DPO FULL-STACK PARITY VERIFICATION FAILED ===")
    sys.exit(1)

if __name__ == "__main__":
  main()
