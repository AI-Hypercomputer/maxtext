"""Standalone diagnostic script for SFT prompt token backward pass activations."""

import sys
import os

sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import jax
import jax.numpy as jnp
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.trainers.pre_train import train
from maxtext.utils import maxtext_utils


def build_sft_batch(batch_size=1, seq_len=128, prompt_len=40, completion_len=50, vocab_size=32000, seed=42):
  """Builds a fixed SFT batch supporting multi-device / multi-TPU FSDP sharding."""
  np.random.seed(seed)
  total_valid = prompt_len + completion_len
  assert total_valid <= seq_len

  inputs = np.random.randint(100, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
  inputs[:, total_valid:] = 0  # padding

  targets = inputs.copy()
  targets[:, :-1] = inputs[:, 1:]
  targets[:, -1] = 0

  # Mask prompt tokens in targets (unk_id = 0)
  targets[:, :prompt_len] = 0

  inputs_segmentation = np.zeros((batch_size, seq_len), dtype=np.int32)
  inputs_segmentation[:, :total_valid] = 1

  targets_segmentation = np.zeros((batch_size, seq_len), dtype=np.int32)
  targets_segmentation[:, prompt_len:total_valid] = 1

  inputs_position = np.tile(np.arange(seq_len, dtype=np.int32), (batch_size, 1))

  batch = {
      "inputs": jnp.array(inputs),
      "targets": jnp.array(targets),
      "inputs_position": jnp.array(inputs_position),
      "inputs_segmentation": jnp.array(inputs_segmentation),
      "targets_segmentation": jnp.array(targets_segmentation),
  }
  return batch


def compute_global_metrics(tree_a, tree_b):
  """Computes global absolute norms, relative L2 diff and cosine similarity streaming to CPU host memory."""
  leaves_a = jax.tree.leaves(tree_a)
  leaves_b = jax.tree.leaves(tree_b)
  diff_sq_sum = 0.0
  norm_a_sq_sum = 0.0
  norm_b_sq_sum = 0.0
  dot_sum = 0.0
  for x_a, x_b in zip(leaves_a, leaves_b):
    a_arr = np.asarray(x_a, dtype=np.float32)
    b_arr = np.asarray(x_b, dtype=np.float32)
    diff = a_arr - b_arr
    diff_sq_sum += np.sum(diff * diff)
    norm_a_sq_sum += np.sum(a_arr * a_arr)
    norm_b_sq_sum += np.sum(b_arr * b_arr)
    dot_sum += np.sum(a_arr * b_arr)
  norm_a = float(np.sqrt(norm_a_sq_sum))
  norm_b = float(np.sqrt(norm_b_sq_sum))
  diff_norm = float(np.sqrt(diff_sq_sum))
  rel_l2 = float(diff_norm / (norm_a + 1e-12))
  cos_sim = float(dot_sum / (norm_a * norm_b + 1e-12))
  return norm_a, norm_b, diff_norm, rel_l2, cos_sim


def run_experiment():
  sys.argv = ["scratch/prompt_grad_ab.py"]
  base_argv = [
      "",
      "src/maxtext/configs/base.yml",
      "run_name=prompt_grad_ab",
      "enable_nnx=false",
      "scan_layers=false",
      "remat_policy=minimal_with_context",
      "num_decoder_layers=4",
      "per_device_batch_size=1",
      "enable_dropout=false",
      "dtype=float32",
      "weight_dtype=float32",
      "num_vocab_tiling=1",
      "sft_train_on_completion_only=true",
      "max_target_length=128",
      "attention=dot_product",
      "log_config=false",
  ]

  # Initialize config FIRST before any JAX backend query
  cfg_ref = pyconfig.initialize(base_argv)

  num_devices = len(jax.devices())
  print(f"JAX default backend: {jax.default_backend()}, devices count: {num_devices}, devices: {jax.devices()}")

  # Build SFT batch matched to device count
  batch = build_sft_batch(
      batch_size=num_devices,
      seq_len=cfg_ref.max_target_length,
      prompt_len=40,
      completion_len=50,
      vocab_size=cfg_ref.vocab_size,
  )

  # Step 1: Prompt mask sanity check
  prompt_mask = (batch["targets_segmentation"] == 0) & (batch["inputs_segmentation"] != 0)
  completion_mask = batch["targets_segmentation"] != 0
  padding_mask = batch["inputs_segmentation"] == 0

  print("\n=== STEP 1: PROMPT MASK SANITY CHECK ===")
  print(f"Batch size:            {num_devices}")
  print(f"Total sequence length: {cfg_ref.max_target_length}")
  print(f"Prompt tokens sum:     {int(prompt_mask.sum())}")
  print(f"Completion tokens sum: {int(completion_mask.sum())}")
  print(f"Padding tokens sum:    {int(padding_mask.sum())}")
  assert prompt_mask.sum() > 0, "ERROR: prompt_mask.sum() is 0! SFT completion-only masking is not working."
  assert (
      prompt_mask.sum() + completion_mask.sum() + padding_mask.sum() == cfg_ref.max_target_length * num_devices
  ), "ERROR: Partition does not sum to total sequence length!"
  print("Sanity check passed successfully!\n")

  # Create Mesh & Model
  devices_array = maxtext_utils.create_device_mesh(cfg_ref)
  mesh = Mesh(devices_array, cfg_ref.mesh_axes)
  quant = quantizations.configure_quantization(cfg_ref)
  model_init = models.TransformerLinenPure(config=cfg_ref, mesh=mesh, quant=quant)

  # Init parameters with fixed seed
  init_rng = jax.random.PRNGKey(42)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
    init_vars = model_init.init(
        {"params": init_rng, "dropout": init_rng},
        batch["inputs"],
        batch["inputs_position"],
        decoder_segment_ids=batch["inputs_segmentation"],
        decoder_target_tokens=batch["targets"],
        decoder_target_mask=batch["targets_segmentation"],
    )

  dropout_rng = jax.random.PRNGKey(0)

  def compute_grads(config):
    model = models.TransformerLinenPure(config=config, mesh=mesh, quant=quant)

    def loss_wrapper(p):
      l, _ = train.loss_fn(
          model=model,
          config=config,
          data=batch,
          dropout_rng=dropout_rng,
          params=p,
          is_train=True,
      )
      return l

    l_val, g = jax.value_and_grad(loss_wrapper)(init_vars)
    return l_val, g

  # -------------------------------------------------------------
  # DELIVERABLE A: Noise floor & Gradient A/B comparison
  # -------------------------------------------------------------
  print("=== DELIVERABLE A: GRADIENT A/B COMPARISON ===")

  # 1. Baseline 1: stop_grad=False
  cfg_base1 = pyconfig.initialize(base_argv + ["debug_stop_grad_prompt=false", "debug_tap_prompt_grads=false"])
  loss_b1, grads_b1 = compute_grads(cfg_base1)

  # 2. Baseline 2 (for noise floor): stop_grad=False
  cfg_base2 = pyconfig.initialize(base_argv + ["debug_stop_grad_prompt=false", "debug_tap_prompt_grads=false"])
  loss_b2, grads_b2 = compute_grads(cfg_base2)

  # 3. Test: stop_grad=True
  cfg_test = pyconfig.initialize(base_argv + ["debug_stop_grad_prompt=true", "debug_tap_prompt_grads=false"])
  loss_test, grads_test = compute_grads(cfg_test)

  noise_norm_a, noise_norm_b, noise_diff, noise_rel_l2, noise_cos = compute_global_metrics(grads_b1, grads_b2)
  norm_b1, norm_test, diff_norm, ab_rel_l2, ab_cos = compute_global_metrics(grads_b1, grads_test)

  print(f"Noise floor (baseline vs baseline):")
  print(f"  Baseline 1 Global Grad L2: {noise_norm_a:.6e}")
  print(f"  Baseline 2 Global Grad L2: {noise_norm_b:.6e}")
  print(f"  Abs L2 Difference:        {noise_diff:.6e}")
  print(f"  Relative L2:               {noise_rel_l2:.6e}")
  print(f"  Cosine Sim:                {noise_cos:.8f}")

  print(f"\nA/B Test (Full vs stop_gradient prompt activations):")
  print(f"  Baseline Loss:             {float(loss_b1):.6f}")
  print(f"  Test Loss:                 {float(loss_test):.6f}")
  print(f"  Abs Loss Difference:       {abs(float(loss_b1) - float(loss_test)):.6e}")
  print(f"  Baseline Global Grad L2:   {norm_b1:.6e}")
  print(f"  Test Global Grad L2:       {norm_test:.6e}")
  print(f"  Abs Global Grad L2 Diff:   {diff_norm:.6e}")
  print(f"  Global Relative L2 Diff:   {ab_rel_l2:.6e} ({ab_rel_l2 * 100:.2f}%)")
  print(f"  Global Cosine Similarity:  {ab_cos:.8f}")

  print("\n--- PER-PARAMETER GRADIENT BREAKDOWN ---")

  param_metrics = []

  def compare_param(path, p_ref, p_test):
    path_str = "/".join(str(k.key if hasattr(k, "key") else k) for k in path)
    r_ref = np.asarray(p_ref, dtype=np.float32).ravel()
    r_test = np.asarray(p_test, dtype=np.float32).ravel()
    diff_norm_p = np.linalg.norm(r_ref - r_test)
    ref_norm = np.linalg.norm(r_ref)
    test_norm = np.linalg.norm(r_test)
    diff = float(diff_norm_p / (ref_norm + 1e-12))
    cos = float(np.dot(r_ref, r_test) / (ref_norm * test_norm + 1e-12))
    param_metrics.append({
        "path": path_str,
        "norm_ref": float(ref_norm),
        "norm_test": float(test_norm),
        "abs_l2": float(diff_norm_p),
        "rel_l2": diff,
        "cos": cos,
    })

  jax.tree_util.tree_map_with_path(compare_param, grads_b1, grads_test)

  param_metrics.sort(key=lambda item: item["norm_ref"], reverse=True)

  print(f"{'Parameter Path':<55} | {'Ref Grad L2':<12} | {'Test Grad L2':<12} | {'Abs L2 Diff':<12} | {'Rel L2 Diff':<12} | {'Cosine Sim':<10}")
  print("-" * 125)
  for m in param_metrics[:35]:
    print(f"{m['path']:<55} | {m['norm_ref']:<12.5e} | {m['norm_test']:<12.5e} | {m['abs_l2']:<12.5e} | {m['rel_l2']:<12.5e} | {m['cos']:<10.6f}")

  # -------------------------------------------------------------
  # DELIVERABLE B: Per-layer backward tap trace
  # -------------------------------------------------------------
  print("\n=== DELIVERABLE B: PER-LAYER BACKWARD GRADIENT-NORM TRACE ===")
  cfg_tap = pyconfig.initialize(base_argv + ["debug_stop_grad_prompt=false", "debug_tap_prompt_grads=true"])
  _, _ = compute_grads(cfg_tap)


if __name__ == "__main__":
  run_experiment()
