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

from MaxText.inference.scripts.sharding_utils import *
import unittest

# Common test parameters
M, K, F = 64, 128, 256
activations_shape_2d = (M, K)
weights_shape_2d = (K, F)
G = 1  # Default for 2D weights

ici_bandwidth_val = 4.50e10  # 45 GB/s
peak_flops_val = 1.97e14  # 1 TFLOP/s
activation_size_bytes_val = 2  # BF16
weight_size_bytes_val = 2  # BF16
ici_latency_val = 1e-6  # 1 microsecond
TOLERANCE = 1e-9  # For floating point comparisons


class ShardingTests(unittest.TestCase):

  def test_no_sharding(self):
    """
    Tests the basic case with no sharding.
    """
    sD, sK, sW, sF, sE = 1, 1, 1, 1, 1

    result = calculate_matmul_resources(
        activations_shape=activations_shape_2d,
        weights_shape=weights_shape_2d,
        ici_bandwidth=ici_bandwidth_val,
        peak_flops=peak_flops_val,
        sD=sD,
        sK=sK,
        sW=sW,
        sF=sF,
        sE=sE,
        activation_size_bytes=activation_size_bytes_val,
        weight_size_bytes=weight_size_bytes_val,
        ici_latency=ici_latency_val,
        debug=False,
    )

    # Expected FLOPs
    # Total FLOPs = 2 * M * K * F
    expected_t_flops = (2.0 * M * K * F) / peak_flops_val
    assert abs(result["t_flops"] - expected_t_flops) < TOLERANCE

    # Expected comms
    expected_t_comms = 0.0
    assert abs(result["t_comms"] - expected_t_comms) < TOLERANCE

    # Expected Memory per TPU
    # Activations
    expected_mem_activations = M * K * activation_size_bytes_val
    # Weights
    expected_mem_weights = K * F * weight_size_bytes_val
    # Output
    expected_mem_output = M * F * max(activation_size_bytes_val, weight_size_bytes_val)
    # Total
    expected_memory_per_TPU = expected_mem_activations + expected_mem_weights + expected_mem_output
    assert abs(result["memory_per_TPU_bytes"] - expected_memory_per_TPU) < TOLERANCE

  def test_output_feature_parallelism_sF(self):
    """
    Tests sharding on the F dimension of weights (sF > 1).
    """
    sF = 4
    sD, sK, sW, sE = 1, 1, 1, 1

    result = calculate_matmul_resources(
        activations_shape=activations_shape_2d,
        weights_shape=weights_shape_2d,  # (K, F)
        ici_bandwidth=ici_bandwidth_val,
        peak_flops=peak_flops_val,
        sD=sD,
        sK=sK,
        sW=sW,
        sF=sF,
        sE=sE,
        activation_size_bytes=activation_size_bytes_val,
        weight_size_bytes=weight_size_bytes_val,
        ici_latency=ici_latency_val,
        debug=False,
    )

    # Expected FLOPs
    expected_t_flops = (2.0 * M * K * (F / sF)) / peak_flops_val
    assert (
        abs(result["t_flops"] - expected_t_flops) < TOLERANCE
    ), f"FLOPs mismatch: got {result['t_flops']}, expected {expected_t_flops}"

    # Expeted comms
    expected_t_comms = 0.0
    assert (
        abs(result["t_comms"] - expected_t_comms) < TOLERANCE
    ), f"Comms mismatch: got {result['t_comms']}, expected {expected_t_comms}"

    # Expected Memory per TPU
    # Activations
    expected_mem_activations = M * K * activation_size_bytes_val
    # Weights
    expected_mem_weights = K * (F / sF) * weight_size_bytes_val
    # Output
    expected_mem_output = M * (F / sF) * max(activation_size_bytes_val, weight_size_bytes_val)
    # Total
    expected_memory_per_TPU = expected_mem_activations + expected_mem_weights + expected_mem_output
    assert (
        abs(result["memory_per_TPU_bytes"] - expected_memory_per_TPU) < TOLERANCE
    ), f"Memory mismatch: got {result['memory_per_TPU_bytes']}, expected {expected_memory_per_TPU}"

  def test_data_parallelism_sD(self):
    """
    Tests sharding on the M dimension of activations (sD).
    """
    sD = 4
    sK, sW, sF, sE = 1, 1, 1, 1

    result = calculate_matmul_resources(
        activations_shape=activations_shape_2d,
        weights_shape=weights_shape_2d,
        ici_bandwidth=ici_bandwidth_val,
        peak_flops=peak_flops_val,
        sD=sD,
        sK=sK,
        sW=sW,
        sF=sF,
        sE=sE,
        activation_size_bytes=activation_size_bytes_val,
        weight_size_bytes=weight_size_bytes_val,
        ici_latency=ici_latency_val,
        debug=False,
    )

    # Expected FLOPs:
    expected_t_flops = (2.0 * M * K * F) / (peak_flops_val * sD)
    assert abs(result["t_flops"] - expected_t_flops) < TOLERANCE

    # Expected comms
    expected_t_comms = 0.0
    assert abs(result["t_comms"] - expected_t_comms) < TOLERANCE

    # Expected Memory per TPU
    # Activations
    expected_mem_activations = (M / sD) * K * activation_size_bytes_val
    # Weights
    expected_mem_weights = K * F * weight_size_bytes_val
    # Output
    expected_mem_output = (M / sD) * F * max(activation_size_bytes_val, weight_size_bytes_val)
    # Total
    expected_memory_per_TPU = expected_mem_activations + expected_mem_weights + expected_mem_output
    assert abs(result["memory_per_TPU_bytes"] - expected_memory_per_TPU) < TOLERANCE

  def test_fsdp_activation_sharding_sK(self):
    """
    Tests FSDP-style sharding on the K dimension of activations (sK).
    Weights are not sharded (sW=1).
    """
    sK = 4
    sD, sW, sF, sE = 1, 1, 1, 1

    result = calculate_matmul_resources(
        activations_shape=activations_shape_2d,
        weights_shape=weights_shape_2d,
        ici_bandwidth=ici_bandwidth_val,
        peak_flops=peak_flops_val,
        sD=sD,
        sK=sK,
        sW=sW,
        sF=sF,
        sE=sE,
        activation_size_bytes=activation_size_bytes_val,
        weight_size_bytes=weight_size_bytes_val,
        ici_latency=ici_latency_val,
        debug=False,
    )

    # Expected FLOPs
    expected_t_flops = (2.0 * M * K * F) / peak_flops_val
    assert abs(result["t_flops"] - expected_t_flops) < TOLERANCE

    # Expected comms
    # per TPU
    comm_data_size = (M * K / sK) * activation_size_bytes_val
    # t_comms
    expected_t_comms = latency_bound_comms(comm_data_size / ici_bandwidth_val, ici_latency_val) * (sK - 1)
    assert abs(result["t_comms"] - expected_t_comms) < TOLERANCE

    # Expected Memory per TPU
    # Activations
    expected_mem_activations = M * K * activation_size_bytes_val
    # Weights
    expected_mem_weights = K * F * weight_size_bytes_val
    # Output
    expected_mem_output = M * F * max(activation_size_bytes_val, weight_size_bytes_val)
    # Total
    expected_memory_per_TPU = expected_mem_activations + expected_mem_weights + expected_mem_output
    assert abs(result["memory_per_TPU_bytes"] - expected_memory_per_TPU) < TOLERANCE

  def test_fsdp_weight_sharding_sW(self):
    """
    Tests FSDP-style sharding on the W dimension of weights (sW).
    Activations are not sharded (sK=1).
    """
    sW = 4
    sD, sK, sF, sE = 1, 1, 1, 1

    result = calculate_matmul_resources(
        activations_shape=activations_shape_2d,
        weights_shape=weights_shape_2d,
        ici_bandwidth=ici_bandwidth_val,
        peak_flops=peak_flops_val,
        sD=sD,
        sK=sK,
        sW=sW,
        sF=sF,
        sE=sE,
        activation_size_bytes=activation_size_bytes_val,
        weight_size_bytes=weight_size_bytes_val,
        ici_latency=ici_latency_val,
        debug=False,
    )

    # Expected FLOPs
    expected_t_flops = (2.0 * M * K * F) / peak_flops_val
    assert abs(result["t_flops"] - expected_t_flops) < TOLERANCE

    # Expected comms
    # per TPU
    comm_data_size = (K * F / sW) * weight_size_bytes_val
    # t_comms
    expected_t_comms = latency_bound_comms(comm_data_size / ici_bandwidth_val, ici_latency_val) * (sW - 1)
    assert abs(result["t_comms"] - expected_t_comms) < TOLERANCE

    # Expected Memory per TPU
    # Activations
    expected_mem_activations = M * K * activation_size_bytes_val
    # Weights
    expected_mem_weights = K * F * weight_size_bytes_val
    # Output
    expected_mem_output = M * F * max(activation_size_bytes_val, weight_size_bytes_val)
    # Total
    expected_memory_per_TPU = expected_mem_activations + expected_mem_weights + expected_mem_output
    assert abs(result["memory_per_TPU_bytes"] - expected_memory_per_TPU) < TOLERANCE

  def test_tensor_parallel_sK_sW(self):
    """
    Tests tensor parallelism where both sK (on K_act) and sW (on K_w) are used.
    Assumes sK == sW and reduce-scatter for partial results.
    """
    sK = 2
    sW = 2  # Must be equal to sK for this path
    sD, sF, sE = 1, 1, 1

    result = calculate_matmul_resources(
        activations_shape=activations_shape_2d,
        weights_shape=weights_shape_2d,
        ici_bandwidth=ici_bandwidth_val,
        peak_flops=peak_flops_val,
        sD=sD,
        sK=sK,
        sW=sW,
        sF=sF,
        sE=sE,
        activation_size_bytes=activation_size_bytes_val,
        weight_size_bytes=weight_size_bytes_val,
        ici_latency=ici_latency_val,
        debug=False,
    )

    # Expected FLOPs
    expected_t_flops = (2.0 * M * (K / (sK * sW)) * F) / peak_flops_val
    assert abs(result["t_flops"] - expected_t_flops) < TOLERANCE

    # Expected comms
    # per TPU
    local_output_bytes = M * F * max(activation_size_bytes_val, weight_size_bytes_val)
    # t_comms
    expected_t_comms = latency_bound_comms(local_output_bytes / ici_bandwidth_val, ici_latency_val) * (sK - 1)
    assert abs(result["t_comms"] - expected_t_comms) < TOLERANCE

    # Expected Memory per TPU
    # Activations
    expected_mem_activations = M * (K / sK) * activation_size_bytes_val
    # Weights
    expected_mem_weights = (K / sW) * F * weight_size_bytes_val
    # Output
    expected_mem_output = M * F * max(activation_size_bytes_val, weight_size_bytes_val)
    # Total
    expected_memory_per_TPU = expected_mem_activations + expected_mem_weights + expected_mem_output
    assert abs(result["memory_per_TPU_bytes"] - expected_memory_per_TPU) < TOLERANCE

  def test_output_feature_parallelism_sF_with_all_gather_F(self):
    """
    Tests sharding on the F dimension of weights (sF > 1)
    AND all-gathering the output along the F dimension.
    """
    sF = 4  # Shard the output feature dimension
    sD, sK, sW, sE = 1, 1, 1, 1  # Isolate sF effect
    all_gather_axes = ["F"]

    result = calculate_matmul_resources(
        activations_shape=activations_shape_2d,
        weights_shape=weights_shape_2d,  # (K, F)
        ici_bandwidth=ici_bandwidth_val,
        peak_flops=peak_flops_val,
        sD=sD,
        sK=sK,
        sW=sW,
        sF=sF,
        sE=sE,
        activation_size_bytes=activation_size_bytes_val,
        weight_size_bytes=weight_size_bytes_val,
        ici_latency=ici_latency_val,
        all_gather_axes=all_gather_axes,
        debug=False,
    )

    # Expected FLOPs
    expected_t_flops = (2.0 * M * K * (F / sF)) / peak_flops_val
    assert (
        abs(result["t_flops"] - expected_t_flops) < TOLERANCE
    ), f"FLOPs mismatch: got {result['t_flops']}, expected {expected_t_flops}"

    # Expected comms
    # per TPU
    local_output_bytes_for_gather = M * (F / sF) * max(activation_size_bytes_val, weight_size_bytes_val)
    # t_comms
    expected_t_comms = latency_bound_comms(local_output_bytes_for_gather / ici_bandwidth_val, ici_latency_val) * (sF - 1)
    assert (
        abs(result["t_comms"] - expected_t_comms) < TOLERANCE
    ), f"Comms mismatch: got {result['t_comms']}, expected {expected_t_comms}"

    # Expected Memory per TPU:
    # Activations
    expected_mem_activations = M * K * activation_size_bytes_val
    # Weights
    expected_mem_weights = K * (F / sF) * weight_size_bytes_val
    # Outputs
    expected_mem_output_gathered = M * F * max(activation_size_bytes_val, weight_size_bytes_val)
    # Total
    expected_memory_per_TPU = expected_mem_activations + expected_mem_weights + expected_mem_output_gathered
    assert (
        abs(result["memory_per_TPU_bytes"] - expected_memory_per_TPU) < TOLERANCE
    ), f"Memory mismatch: got {result['memory_per_TPU_bytes']}, expected {expected_memory_per_TPU}"

  def test_expert_parallelism_sE(self):
    """
    Tests expert parallelism sharding on the G dimension (sE).
    """
    G_val = 8
    weights_shape_3d = (G_val, K, F)
    sE = 4
    sD, sK, sW, sF = 1, 1, 1, 1

    result = calculate_matmul_resources(
        activations_shape=activations_shape_2d,
        weights_shape=weights_shape_3d,
        ici_bandwidth=ici_bandwidth_val,
        peak_flops=peak_flops_val,
        sD=sD,
        sK=sK,
        sW=sW,
        sF=sF,
        sE=sE,
        activation_size_bytes=activation_size_bytes_val,
        weight_size_bytes=weight_size_bytes_val,
        ici_latency=ici_latency_val,
        debug=False,
    )

    # Expected FLOPs
    expected_t_flops = (2.0 * M * K * F * G_val) / (peak_flops_val * sE)
    assert abs(result["t_flops"] - expected_t_flops) < TOLERANCE

    # Expected comms
    # per TPU
    local_output_bytes = M * (G_val / sE) * F * max(activation_size_bytes_val, weight_size_bytes_val)
    # t_comms
    expected_t_comms = latency_bound_comms(local_output_bytes / ici_bandwidth_val, ici_latency_val) * (sE - 1) / 4
    assert abs(result["t_comms"] - expected_t_comms) < TOLERANCE

    # Expected Memory per TPU
    # Activations
    expected_mem_activations = M * K * activation_size_bytes_val
    # Weights
    expected_mem_weights = (G_val / sE) * K * F * weight_size_bytes_val
    # Output
    expected_mem_output = M * (G_val / sE) * F * max(activation_size_bytes_val, weight_size_bytes_val)
    # Total
    expected_memory_per_TPU = expected_mem_activations + expected_mem_weights + expected_mem_output
    assert abs(result["memory_per_TPU_bytes"] - expected_memory_per_TPU) < TOLERANCE

  def test_mixed_sharding_sD_sK_sW(self):
    """
    Tests a mix of data parallelism and tensor parallelism (reduce-scatter).
    """
    sD = 2
    sK = 2
    sW = 2  # sK == sW
    sF, sE = 1, 1

    result = calculate_matmul_resources(
        activations_shape=activations_shape_2d,
        weights_shape=weights_shape_2d,
        ici_bandwidth=ici_bandwidth_val,
        peak_flops=peak_flops_val,
        sD=sD,
        sK=sK,
        sW=sW,
        sF=sF,
        sE=sE,
        activation_size_bytes=activation_size_bytes_val,
        weight_size_bytes=weight_size_bytes_val,
        ici_latency=ici_latency_val,
        debug=False,
    )

    # Expected FLOPs
    expected_t_flops = (2.0 * (M / sD) * (K / (sK * sW)) * F) / peak_flops_val
    assert abs(result["t_flops"] - expected_t_flops) < TOLERANCE

    # Expected comms
    # per TPU
    local_output_bytes = (M / sD) * F * max(activation_size_bytes_val, weight_size_bytes_val)
    # t_comms
    expected_t_comms = latency_bound_comms(local_output_bytes / ici_bandwidth_val, ici_latency_val) * (sK - 1)
    assert abs(result["t_comms"] - expected_t_comms) < TOLERANCE

    # Expected Memory per TPU
    # Activations
    expected_mem_activations = (M / sD) * (K / sK) * activation_size_bytes_val
    # Weights
    expected_mem_weights = (K / sW) * F * weight_size_bytes_val
    # Output
    expected_mem_output = (M / sD) * F * max(activation_size_bytes_val, weight_size_bytes_val)
    # Total
    expected_memory_per_TPU = expected_mem_activations + expected_mem_weights + expected_mem_output
    assert abs(result["memory_per_TPU_bytes"] - expected_memory_per_TPU) < TOLERANCE

  def test_additional_all_gather_axes_D(self):
    """
    Tests additional all-gather on the 'D' dimension of the output.
    """
    sD = 2
    sK, sW, sF, sE = 1, 1, 1, 1
    all_gather_axes = ["D"]

    result = calculate_matmul_resources(
        activations_shape=activations_shape_2d,
        weights_shape=weights_shape_2d,
        ici_bandwidth=ici_bandwidth_val,
        peak_flops=peak_flops_val,
        sD=sD,
        sK=sK,
        sW=sW,
        sF=sF,
        sE=sE,
        activation_size_bytes=activation_size_bytes_val,
        weight_size_bytes=weight_size_bytes_val,
        ici_latency=ici_latency_val,
        all_gather_axes=all_gather_axes,
        debug=False,
    )

    # Expected FLOPs
    expected_t_flops = (2.0 * M * K * F) / (peak_flops_val * sD)
    assert abs(result["t_flops"] - expected_t_flops) < TOLERANCE

    # Expected comms
    # per TPU
    local_output_bytes_base = (M / sD) * F * max(activation_size_bytes_val, weight_size_bytes_val)
    # t_comms
    expected_t_comms = latency_bound_comms(local_output_bytes_base / ici_bandwidth_val, ici_latency_val) * (sD - 1)
    assert abs(result["t_comms"] - expected_t_comms) < TOLERANCE

    # Expected Memory per TPU
    # Activations
    expected_mem_activations = (M / sD) * K * activation_size_bytes_val
    # Weights
    expected_mem_weights = K * F * weight_size_bytes_val
    # Output
    expected_mem_output = M * F * max(activation_size_bytes_val, weight_size_bytes_val)
    # Total
    expected_memory_per_TPU = expected_mem_activations + expected_mem_weights + expected_mem_output
    assert abs(result["memory_per_TPU_bytes"] - expected_memory_per_TPU) < TOLERANCE
