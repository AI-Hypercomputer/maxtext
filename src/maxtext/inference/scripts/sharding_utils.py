# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sharding related utilities."""

import pprint
import warnings
from typing import Sequence

import numpy as np

import matplotlib.pyplot as plt


def latency_bound_comms(comm: float, latency=1e-6):
  return max(comm, latency)


def calculate_matmul_resources(
    activations_shape: tuple[int, ...],
    weights_shape: tuple[int, ...],
    ici_bandwidth: float,
    peak_flops: float,
    sD: int = 1,
    sK: int = 1,
    sW: int = 1,
    sF: int = 1,
    sE: int = 1,
    activation_size_bytes: int = 2,
    weight_size_bytes: int = 2,
    ici_latency: float = 1e-6,
    all_gather_axes: Sequence[str] = tuple(),
    debug=True,
) -> dict[str, float]:
  """
  Calculates estimated FLOPs, communication volume, and memory for a distributed matrix multiplication.

  The multiplication is ``A @ W``.
  A (activations) has shape (M, K).
  W (weights) has shape (G, K, F).

  Sharding strategy assumed:

  * Data Parallelism: ``sD`` shards the M dim of A.
  * Embedding Parallelism: ``sK`` shards on the embedding dim of A.
  * Tensor Parallelism for W dim: ``sK`` shards the W dimension of W.
  * Tensor Parallelism for F dim: ``sF`` shards the second weight dim of W.

  Args:
    activations_shape: Shape of the activations tensor (M, K).
    weights_shape: Shape of the weights tensor (G, K, F).
      G is the number of groups if this is a GMM (e.g in MoE layer).
    sD: Number of data parallel shards (sD). Must be >= 1.
    sK: Sharding factor for the activation embedding dimension.
    sW: Sharding factor for the first weight dimension.
    sF: Sharding factor for the second weight dimension.
    sE: Sharding factor to split up expert weights.
    activation_size_bytes: Size of a single element in bytes for the activations.
    weight_size_bytes: Size of a single element in bytes for the weights.
    ici_latency: The latency overhead of communicating between TPUs.
    all_gather_axes: Optional additional output axes that need to be
      all-gathered (e.g. "M", "F").
    debug: Whether to print intermediate resource calculations.

  Returns:
    A dictionary with keys

    * ``t_flops``: Estimated FLOPs latency.
    * ``t_comms``: Estimated communication latency.
    * ``memory``: Estimated memory footprint per device for storing local shards
      of activations, weights, and output (bytes).
  """

  M, K_act = activations_shape[0], activations_shape[-1]
  # Intermediate activation shape
  I = np.prod(np.array(activations_shape[1:-1]))
  if len(weights_shape) == 3:
    G, K_w, F = weights_shape
  elif len(weights_shape) == 2:
    K_w, F = weights_shape
    G = 1
  else:
    raise ValueError(f"weights_shape={weights_shape} is not supported!.")

  def _gather_dim_to_shard():
    # # Used to map all-gather arguments to the respective shardings.
    return {"D": sD, "K": sK, "W": sW, "F": sF, "E": sE}

  gather_dim_to_shard = _gather_dim_to_shard()

  def _validate_shardings_and_shapes():
    if not (sD >= 1 and sK >= 1 and sW >= 1 and sF >= 1 and sE >= 1):
      raise ValueError("All sharding amounts must be >= 1.")
    if sK > 1 and sF > 1:
      raise ValueError("Cannot have both sK & sF > 1!")
    if K_act != K_w:
      raise ValueError(f"K dimension of activations ({K_act}) must match K dimension of weights ({K_w})")
    if sK > 1 and sW > 1 and sK != sW:
      raise ValueError("Sharding amounts between embedding dim and first weight matricx dim are different!.")
    # Warnings for non-divisibility. Calculations proceed with float division,
    # implying an average or approximation if not perfectly divisible.
    if M % sD != 0:
      print(
          f"Warning: Activations M dimension ({M}) is not perfectly divisible by sharding amount {sD}.",
          "Results are approximate.",
      )
    if K_act % sK != 0:
      print(
          f"Warning: Common K dimension ({K_act}) is not perfectly divisible by sharding amount {sK}.",
          "Results are approximate.",
      )
    if K_w % sW != 0:
      print(
          f"Warning: Common W dimension ({K_w}) is not perfectly divisible by sharding amount {sW}. Results are approximate."
      )
    if F % sF != 0:
      print(
          f"Warning: Weights F dimension ({F}) is not perfectly divisible by sharding amount {sF}. Results are approximate."
      )
    if G % sE != 0:
      print(
          f"Warning: Experts G dimension ({G}) is not perfectly divisible by sharding amount {sE}. Results are approximate."
      )

  _validate_shardings_and_shapes()
  K = K_act

  # Implied all-gather flags
  is_fsdp_act = sK > 1 and sW == 1
  is_fsdp_weight = sK == 1 and sW > 1

  # Local device dimensions
  local_M_dim = M // sD
  local_K_dim = K // sK
  local_W_dim = K // sW
  local_G_dim = G // sE
  local_F_dim = F // sF

  # 1. Total FLOPs
  # For A(M,K) @ W(K,F), FLOPs = 2 * M * K * F
  total_flops = 2.0 * np.prod(activations_shape) * G * F / (sF * sE * sD * sK * sW)
  if debug:
    print(f"Total GFlops = {total_flops/1e9}")
  if is_fsdp_act:
    total_flops *= sK
    if debug:
      print(f"Total GFlops after activation all-gather = {total_flops/1e9}")
  elif is_fsdp_weight:
    total_flops *= sW
    if debug:
      print(f"Total GFlops after weights all-gather = {total_flops/1e9}")
  t_flops = total_flops / peak_flops

  # 2. Memory per device
  # A_local: (M/sD, K/sK)
  # W_local: (G/gE, K/sK, N/sF)
  # Out_local: (M/sD, N/sF) (buffer for local output)
  mem_activations_bytes = local_M_dim * I * local_K_dim * activation_size_bytes
  mem_weights_bytes = local_G_dim * local_W_dim * local_F_dim * weight_size_bytes
  if debug:
    print(f"Activation memory (GB): {mem_activations_bytes/1e9}")
    print(f"Weights memory (GB): {mem_weights_bytes/1e9}")
  # All-gather
  if is_fsdp_act:
    mem_activations_bytes *= sK
    if debug:
      print(f"Activation memory (GB) after all-gather: {mem_activations_bytes/1e9}")
  elif is_fsdp_weight:
    mem_weights_bytes *= sW
    if debug:
      print(f"Weight memory (GB) after all-gather: {mem_weights_bytes/1e9}")

  local_output_bytes = local_M_dim * I * local_G_dim * local_F_dim * max(activation_size_bytes, weight_size_bytes)
  if debug:
    print(f"Output memory (GB): {local_output_bytes/1e9}")

  gathered_output_bytes = local_output_bytes * np.prod([gather_dim_to_shard[axis] for axis in all_gather_axes])
  if debug:
    print(f"Output memory (GB) after additional axes gathers: {gathered_output_bytes/1e9}")
  memory_per_TPU_bytes = mem_activations_bytes + mem_weights_bytes + gathered_output_bytes
  if debug:
    print(f"Total memory (GB): {memory_per_TPU_bytes/1e9}")

  # 3. Communication Volume per TPU
  t_comms = 0.0

  # For FSDP-style comms, all-gather the tensor.
  if is_fsdp_act:
    communication_volume_per_TPU_bytes = np.prod(np.array(activations_shape)) / sK * activation_size_bytes
    t_comms += latency_bound_comms(communication_volume_per_TPU_bytes / ici_bandwidth, ici_latency) * (sK - 1)
    if debug:
      print(f"Per-TPU comms for activation all-gather (GB): {communication_volume_per_TPU_bytes/1e9}")

  elif is_fsdp_weight:
    communication_volume_per_TPU_bytes = np.prod(np.array(weights_shape)) / sW * weight_size_bytes
    t_comms += latency_bound_comms(communication_volume_per_TPU_bytes / ici_bandwidth, ici_latency) * (sW - 1)
    if debug:
      print(f"Per-TPU comms for weights all-gather (GB): {communication_volume_per_TPU_bytes/1e9}")

  elif sK > 1 and sW > 1:
    # Perform reduce-scatter on the output.
    t_comms = latency_bound_comms(local_output_bytes / ici_bandwidth, ici_latency) * (sK - 1)
    if debug:
      print(f"Per-TPU comms for all-reduce (GB): {local_output_bytes/1e9}")

  # All-to-all on the output during expert parallelism (assuming equal loads. i.e. 1/4 * comms(all-gather))
  if sE > 1:
    t_comms += latency_bound_comms(local_output_bytes / ici_bandwidth, ici_latency) * (sE - 1) / 4
    if debug:
      print(f"Per-TPU comms for all-to-all (GB): {local_output_bytes/1e9}")

  for axis in all_gather_axes:
    current_output_bytes = local_output_bytes
    current_sharding = gather_dim_to_shard[axis]
    t_comms += latency_bound_comms(current_output_bytes / ici_bandwidth, ici_latency) * (current_sharding - 1)
    if debug:
      print(f"Per-TPU comms for axis {axis} all-gather (GB): {current_output_bytes/1e9}")
    current_output_bytes *= current_sharding

  return {
      "t_flops": t_flops,
      "t_comms": t_comms,
      "memory_per_TPU_bytes": memory_per_TPU_bytes,
  }


def plot_sharding_scheme_comparison(
    calc_resource_func,
    activations_shape,
    weights_shape,
    sharding_schemes: list[dict],
):
  """
  Generates plots comparing different sharding schemes:

  1. Communication latency vs. FLOPs latency
  2. Communication latency / memory per device
  3. Memory & Communication Latency

  Args:
    activations_shape: Shape of the activations tensor (M, K).
    weights_shape: Shape of the weights tensor (G, K, F).
    sharding_schemes: A list of dictionaries. Each dictionary must contain:

      * ``label``: A string label for the scheme (e.g., "DP=8").
      * ``shard_settings``: A dictionary with sharding parameters used for
        ``calc_resource_func()``. E.g::

        [
            {
                "label": "DP=8", # Pure Data Parallelism
                "shard_settings": {
                    "sD": 8,
                    "all_gather_axes": ("D",)
                }
            },
        ]
    element_size_bytes: Size of a single element in bytes.
  """
  results = []
  valid_schemes_labels = []

  print("Calculating resources for sharding schemes...")
  for scheme in sharding_schemes:
    label = scheme.get("label", "Unknown Scheme")
    shard_settings = scheme.get("shard_settings")

    print(f"\n--- Scheme: {label} ---")
    try:
      # Clear previous warnings for divisibility for cleaner output per iteration
      with warnings.catch_warnings(record=True) as caught_warnings:
        del caught_warnings
        warnings.simplefilter("always")  # Catch all warnings

        # Call the resource calculation function
        res = calc_resource_func(activations_shape, weights_shape, **shard_settings)
        print("Workload stats:\n")
        pprint.PrettyPrinter(indent=4).pprint(res)

      results.append(res)
      valid_schemes_labels.append(label)
    except ValueError as e:
      print(f"Error calculating resources for scheme '{label}': {e}. Skipping.")
    except (TypeError, KeyError, ZeroDivisionError, AttributeError) as e:
      print(f"An unexpected error occurred for scheme '{label}': {e}. Skipping.")

  if not results:
    print("No valid data points generated. Cannot create plots.")
    return

  # Extract data for plotting
  t_flops_list = np.array([r["t_flops"] for r in results])
  t_comms_list = np.array([r["t_comms"] for r in results])
  mem_list = np.array([r["memory_per_TPU_bytes"] for r in results]) / (1024**3)  # GB
  title_suffix_context = f": A{activations_shape} @ W{weights_shape}"
  num_schemes = len(valid_schemes_labels)  # Number of successfully processed schemes
  colors = plt.cm.viridis(np.linspace(0, 1, num_schemes)) if num_schemes > 0 else []

  # Calculate FLOPs/Communication ratio
  flops_per_comm_ratio = np.zeros(num_schemes)
  has_infinite_ratio = [False] * num_schemes
  for i in range(num_schemes):
    if t_comms_list[i] > 1e-9:  # Threshold to avoid near-zero division issues
      flops_per_comm_ratio[i] = t_flops_list[i] / t_comms_list[i]
    elif t_flops_list[i] > 1e-9:  # Positive FLOPs and zero/tiny communication
      flops_per_comm_ratio[i] = np.inf
      has_infinite_ratio[i] = True
    else:  # Zero FLOPs and zero/tiny communication
      flops_per_comm_ratio[i] = 0

  finite_ratios = flops_per_comm_ratio[np.isfinite(flops_per_comm_ratio)]
  placeholder_for_inf = 0
  if finite_ratios.size > 0:
    placeholder_for_inf = np.max(finite_ratios) * 1.5 if np.max(finite_ratios) > 0 else 1000
  elif np.any(has_infinite_ratio):
    placeholder_for_inf = 1000

  plot_ratios = np.array(
      [placeholder_for_inf if r_inf else r_val for r_val, r_inf in zip(flops_per_comm_ratio, has_infinite_ratio)]
  )
  plot_ratios = np.nan_to_num(plot_ratios, nan=0.0, posinf=placeholder_for_inf, neginf=-placeholder_for_inf)

  # --- Create Plots ---
  categorical_x = np.arange(num_schemes)  # For categorical x-axis

  # Plot 1: FLOPs & Communication (Grouped Bar Plot)
  grouped_bar_width_fc = 0.35
  fig_flops_comm_grouped, ax_flops_comm_grouped = plt.subplots(figsize=(max(10, num_schemes * 1.7), 7))

  rects_flops = ax_flops_comm_grouped.bar(
      categorical_x - grouped_bar_width_fc / 2,
      t_flops_list,
      grouped_bar_width_fc,
      label="T_flops",
      color="mediumseagreen",
  )
  rects_comms_grouped = ax_flops_comm_grouped.bar(
      categorical_x + grouped_bar_width_fc / 2, t_comms_list, grouped_bar_width_fc, label="T_comms", color="deepskyblue"
  )

  ax_flops_comm_grouped.set_xlabel("Sharding Scheme")
  ax_flops_comm_grouped.set_ylabel("Seconds")
  ax_flops_comm_grouped.set_title(f"T_flops & T_comms by Sharding Scheme{title_suffix_context}", fontsize=14)
  ax_flops_comm_grouped.set_xticks(categorical_x)
  ax_flops_comm_grouped.set_xticklabels(valid_schemes_labels, rotation=45, ha="right", fontsize=10)
  if num_schemes > 0:
    ax_flops_comm_grouped.legend(fontsize=10)

  ax_flops_comm_grouped.bar_label(rects_flops, padding=3, fmt="%.2e", fontsize=9)
  ax_flops_comm_grouped.bar_label(rects_comms_grouped, padding=3, fmt="%.2e", fontsize=9)

  ax_flops_comm_grouped.grid(True, axis="y", linestyle="--", alpha=0.7)

  max_y_val_fc = 0
  if t_flops_list.size > 0:
    max_y_val_fc = max(max_y_val_fc, np.max(t_flops_list))
  if t_comms_list.size > 0:
    max_y_val_fc = max(max_y_val_fc, np.max(t_comms_list))
  print(f"max_y_val_fc = {max_y_val_fc}")
  ax_flops_comm_grouped.set_ylim(0, max_y_val_fc * 1.15)

  fig_flops_comm_grouped.tight_layout()
  plt.show()

  # Plot 2: FLOPs/Communication Ratio
  fig_ratio, ax_ratio = plt.subplots(figsize=(max(10, num_schemes * 1.1), 7))

  bars_ratio = ax_ratio.bar(categorical_x, plot_ratios, width=0.6, color=colors, alpha=0.9)

  ax_ratio.set_xlabel("Sharding Scheme")
  ax_ratio.set_ylabel("T_flops / T_comms")
  ax_ratio.set_title(f"Roofline (T_flops vs. T_comms) for {title_suffix_context}", fontsize=14)
  ax_ratio.set_xticks(categorical_x)
  ax_ratio.set_xticklabels(valid_schemes_labels, rotation=45, ha="right", fontsize=10)
  ax_ratio.grid(True, axis="y", linestyle="--", alpha=0.7)

  for i, bar in enumerate(bars_ratio):
    yval = bar.get_height()
    label_text = f"{yval:.2f}"
    if has_infinite_ratio[i] and yval == placeholder_for_inf:
      label_text = f"> {np.max(finite_ratios):.2f}\n(Effectively Inf)" if finite_ratios.size > 0 else "Very High"
    ax_ratio.text(bar.get_x() + bar.get_width() / 2.0, yval, label_text, va="bottom", ha="center", fontsize=9)

  if plot_ratios.size > 0:
    max_ratio_plot_val = np.max(plot_ratios)
    ax_ratio.set_ylim(0, max_ratio_plot_val * 1.15)

  fig_ratio.tight_layout()
  plt.show()

  # Plot 3: Memory vs. Communication (Bars positioned by Communication Volume)
  fig_mem, ax_mem = plt.subplots(figsize=(max(10, num_schemes * 1.3), 7))  # Slightly wider for labels
  bar_width_mem = 0.6

  ax_mem.bar(
      categorical_x,
      mem_list,
      width=bar_width_mem,
      color=colors,
      alpha=0.85,
      edgecolor=[np.array(c[:3]) * 0.6 for c in colors],
  )

  ax_mem.set_xlabel("Sharding Scheme")
  ax_mem.set_ylabel("Memory per TPU (GB)")
  ax_mem.set_title(f"Memory & Comm. by Sharding Scheme{title_suffix_context}", fontsize=14)  # Updated title
  ax_mem.set_xticks(categorical_x)
  ax_mem.set_xticklabels(valid_schemes_labels, rotation=45, ha="right", fontsize=10)
  ax_mem.grid(True, axis="y", linestyle="--", alpha=0.7)

  # Add custom labels in scientific notation
  for i in range(num_schemes):
    mem_val = mem_list[i]
    comm_val = t_comms_list[i]  # This is assumed to be in MB

    # Format the label string as requested
    # Using \n for a new line to make it more readable on the plot
    label_text = f"mem: {mem_val:.2e} GB\nt_comms: {comm_val:.2e} sec"

    ax_mem.text(
        categorical_x[i],  # x-position: center of the bar
        mem_val,  # y-position: top of the bar
        label_text,
        ha="center",  # Horizontal alignment
        va="bottom",  # Vertical alignment (anchor at bottom of text, so text is above y)
        fontsize=8,
        rotation=0,
        bbox={"facecolor": "white", "alpha": 0.6, "pad": 2, "boxstyle": "round,pad=0.3"},  # Added bbox
    )

  if mem_list.size > 0:
    max_mem_val = np.max(mem_list)
    # Adjust y-limit to accommodate multi-line labels; factor might need tuning
    ax_mem.set_ylim(0, max_mem_val * 1.35)  # Increased padding for multi-line labels
  else:
    ax_mem.set_ylim(0, 1)

  fig_mem.tight_layout()
  plt.show()
