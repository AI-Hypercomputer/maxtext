import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.profiler import start_trace, stop_trace
import numpy as np
import time
from typing import Tuple, Callable, Dict, Any, List, Sequence, Optional, NamedTuple
from functools import partial
import os
import tempfile

# Constants
GLOBAL_BATCH = 8 * 1024
EMBED = 8192
MLP = 28672
NUM_ITERATIONS = 100

TRACE_DIR = "traces"


class MeshConfig(NamedTuple):
    shape: Tuple[int, ...]
    axis_names: Tuple[str, ...]
    creator: Callable
    name: str

def standard_mesh(shape: Tuple[int, ...], axis_names: Tuple[str, ...]) -> MeshConfig:
    return MeshConfig(shape, axis_names, create_mesh, "Standard")

def balanced_2d_mesh(shape: Tuple[int, ...], axis_names: Tuple[str, ...]) -> MeshConfig:
    def balanced_mesh_creator(mesh_shape):
        return make_nested_balanced_2d_devices(jax.devices(), mesh_shape)
    
    return MeshConfig(shape, axis_names, 
                      partial(create_mesh, mesh_creator=balanced_mesh_creator),
                      "Balanced 2D")

def create_trace_name(mesh_shape: Tuple[int, ...], batch_size: int, embed_size: int, mlp_size: int, mesh_axis_names: Tuple[str, ...], explicit_layout: Optional[List[List[int]]] = None) -> str:
    mesh_str = "x".join(map(str, mesh_shape))
    axes_str = "_".join(mesh_axis_names)
    layout_str = "custom" if explicit_layout else "standard"
    return f"matmul_b{batch_size}_e{embed_size}_m{mlp_size}_mesh_{mesh_str}_axes_{axes_str}_layout_{layout_str}"

def create_mesh(mesh_shape: Tuple[int, ...], mesh_axis_names: Tuple[str, ...], mesh_creator: Callable = mesh_utils.create_device_mesh) -> Mesh:
    devices = mesh_creator(mesh_shape)
    return Mesh(devices=devices, axis_names=mesh_axis_names)

def create_arrays(batch_size: int, embed_size: int, mlp_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key = jax.random.PRNGKey(0)
    key_A, key_W1 = jax.random.split(key)
    A = jax.random.normal(key_A, (batch_size, embed_size), dtype=jnp.bfloat16)
    W1 = jax.random.normal(key_W1, (embed_size, mlp_size), dtype=jnp.bfloat16)
    return A, W1

def shard_arrays(mesh: Mesh, A: jnp.ndarray, W1: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    A_sharded = jax.device_put(A, NamedSharding(mesh, P(None, "model")))
    W1_sharded = jax.device_put(W1, NamedSharding(mesh, P("model", None)))
    return A_sharded, W1_sharded

def create_matmul_function(mesh: jax.sharding.Mesh):
    activation_spec = P(None, "model")
    weight_spec = P("model", None)
    activation_sharding = NamedSharding(mesh, activation_spec)
    weight_sharding = NamedSharding(mesh, weight_spec)
    
    @partial(jax.jit, static_argnums=(2,))
    def matmul(A, W1, dummy_arg):
        A = jax.lax.with_sharding_constraint(A, activation_sharding)
        W1 = jax.lax.with_sharding_constraint(W1, weight_sharding)
        result = A @ W1
        return jax.lax.with_sharding_constraint(result, activation_sharding)
    
    return matmul


def run_experiment(mesh: Mesh, A: jnp.ndarray, W1: jnp.ndarray, num_iterations: int, trace_name: str) -> Tuple[float, float, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    A_sharded, W1_sharded = shard_arrays(mesh, A, W1)
    matmul_fn = create_matmul_function(mesh)
    dummy = object()
    
    # Warmup
    for _ in range(10):
        result = matmul_fn(A_sharded, W1_sharded, dummy)
        jax.block_until_ready(result)
    
    # Ensure the trace directory exists
    os.makedirs(TRACE_DIR, exist_ok=True)
    
    # Benchmark with profiling
    trace_loc = os.path.join(TRACE_DIR, trace_name)
    start_trace(trace_loc)
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = matmul_fn(A_sharded, W1_sharded, dummy)
        jax.block_until_ready(result)
        end = time.perf_counter()
        times.append(end - start)
    
    stop_trace()
    print(f"Profiler trace saved to: {trace_loc}")
    
    return np.mean(times), np.std(times, ddof=1), A_sharded, W1_sharded, result
    

def memory_usage_profile() -> Dict[str, Any]:
    memory_stats = jax.device_get(jax.devices()[0].memory_stats())
    total_memory = memory_stats['bytes_limit']
    used_memory = memory_stats['bytes_in_use']
    return {
        "total_memory": total_memory,
        "used_memory": used_memory,
        "memory_utilization": used_memory / total_memory if total_memory > 0 else 0,
        "total_memory_formatted": format_bytes(total_memory),
        "used_memory_formatted": format_bytes(used_memory)
    }

def format_bytes(size_in_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} PB"

def compute_efficiency(time: float, A: jnp.ndarray, W1: jnp.ndarray) -> float:
    flops = 2 * A.shape[0] * A.shape[1] * W1.shape[1]
    return flops / (time * 1e12)  # TFLOP/s

def print_results(config_name: str, mesh: Mesh, mean_time: float, std_dev: float, num_iterations: int, A: jnp.ndarray, W1: jnp.ndarray, A_sharded: jnp.ndarray, W1_sharded: jnp.ndarray, result: jnp.ndarray, visualize: bool = False) -> str:
    memory_profile = memory_usage_profile()
    efficiency = compute_efficiency(mean_time, A, W1)
    
    output = f"""
Experiment: {config_name}
Mesh Shape: {tuple(mesh.shape.values())}
Mesh Axes: {mesh.axis_names}
Matrix Multiplication Time (avg of {num_iterations} runs):
  Mean: {mean_time:.6f} seconds
  Std Dev: {std_dev:.6f} seconds
Memory Utilization: {memory_profile['memory_utilization']:.2%}
Total Memory: {memory_profile['total_memory_formatted']}
Used Memory: {memory_profile['used_memory_formatted']}
Computational Efficiency: {efficiency:.2f} TFLOP/s

{compare_device_placement(mesh)}
"""

    print(output)
    if visualize:
        print("\nArray Sharding Visualization:\n")
        print("A sharding:\n")
        jax.debug.visualize_array_sharding(A_sharded)
        print("\nW1 sharding:\n")
        jax.debug.visualize_array_sharding(W1_sharded)
        print("\nResult sharding:\n")
        jax.debug.visualize_array_sharding(result)
    print('=' * 50)

def compare_device_placement(mesh: Mesh) -> str:
    mesh_shape = tuple(mesh.shape.values())
    devices_array = np.array(mesh.devices).reshape(mesh_shape)
    compact_mesh = np.array([f"D{device.id}" for device in devices_array.flat]).reshape(mesh_shape)
    return f"Compact Mesh Representation:\n{compact_mesh}\n"

def make_nested_balanced_2d_devices(devices: Sequence[jax.Device], ici_mesh_shape: Sequence[int]) -> Sequence[jax.Device]:
    log_len = np.array(devices).size.bit_length() - 1
    arr = np.arange(log_len)[::-1]
    midpoint = len(arr) // 2
    first_half = arr[:midpoint]
    second_half = arr[midpoint:]
    new_axis_order = []
    for pair in zip(second_half, first_half):
        new_axis_order.extend(pair)
    if len(arr) % 2 == 1:
        new_axis_order.append(second_half[-1])
    ordered_flat_devices = sorted(np.array(devices).flatten(), key=lambda x: x.id)
    result = np.reshape(ordered_flat_devices, (2,) * log_len).transpose(new_axis_order[::-1]).reshape(ici_mesh_shape)
    return result

def create_explicit_layout_mesh(device_layout: List[List[int]], mesh_axis_names: Tuple[str, ...]) -> Mesh:
    devices = jax.devices()
    if max(max(row) for row in device_layout) >= len(devices):
        raise ValueError(f"Device layout contains indices that exceed the number of available devices ({len(devices)})")

    new_devices = [[devices[idx] for idx in row] for row in device_layout]
    mesh_shape = (len(device_layout), len(device_layout[0]))
    
    if len(mesh_axis_names) != len(mesh_shape):
        raise ValueError(f"Number of axis names ({len(mesh_axis_names)}) does not match mesh shape ({mesh_shape})")

    return Mesh(np.array(new_devices), mesh_axis_names)

def main():
    A, W1 = create_arrays(GLOBAL_BATCH, EMBED, MLP)
    
    mesh_configs = [
        standard_mesh((4, 2), ("model", "sequence")),
        balanced_2d_mesh((4, 2), ("model", "sequence")),
        standard_mesh((1, 4, 2), ("placeholder", "model", "sequence")),
        balanced_2d_mesh((1, 4, 2), ("placeholder", "model", "sequence")),
    ]

    visualize = False

    results = {}
    for config in mesh_configs:
        mesh = config.creator(config.shape, config.axis_names)
        trace_name = create_trace_name(config.shape, GLOBAL_BATCH, EMBED, MLP, config.axis_names)
        mean_time, std_dev, A_sharded, W1_sharded, result = run_experiment(mesh, A, W1, NUM_ITERATIONS, trace_name)
        config_name = f"{config.name}_{config.shape}"
        results[config_name] = (mesh, mean_time, std_dev)
        print_results(config_name, mesh, mean_time, std_dev, NUM_ITERATIONS, A, W1, A_sharded, W1_sharded, result, visualize)


    """
    Assumed device layout is
      [ 0 | 1 ]       [ 4 | 5 ]
      [ --+-- ] <---> [ --+-- ]
      [ 2 | 3 ]       [ 6 | 7 ]
    """
    explicit_layouts = [
        [[0, 4], [1, 5], [3, 7], [2, 6]],   # 0.004309
        [[1, 5], [3, 7], [2, 6], [0, 4]],   # 0.004309
        [[0, 4], [1, 5], [2, 6], [3, 7]],   # 0.005691
        [[0, 1], [2, 3], [4, 5], [6, 7]],   # 0.005632
        [[0, 7], [1, 6], [2, 5], [3, 4]],   # 0.005687
        [[0, 4], [1, 7], [2, 6], [3, 5]],   # 0.005680
        [[0, 2], [1, 3], [4, 6], [5, 7]],   # 0.006855
        [[0, 5], [1, 6], [2, 7], [3, 4]],   # 0.005683
        [[0, 1], [3, 2], [4, 5], [7, 6]],   # 0.005679
        [[0, 5], [1, 4], [2, 7], [3, 6]],   # 0.005673
    ]

    for i, explicit_layout in enumerate(explicit_layouts):
        explicit_mesh_shape = (len(explicit_layout), len(explicit_layout[0]))
        mesh = create_explicit_layout_mesh(explicit_layout, ("model", "sequence"))
        trace_name = create_trace_name(explicit_mesh_shape, GLOBAL_BATCH, EMBED, MLP, ("model", "sequence"), explicit_layout)
        mean_time, std_dev, A_sharded, W1_sharded, result = run_experiment(mesh, A, W1, NUM_ITERATIONS, trace_name)
        config_name = f"explicit_layout_{i}"
        results[config_name] = (mesh, mean_time, std_dev)
        print_results(config_name, mesh, mean_time, std_dev, NUM_ITERATIONS, A, W1, A_sharded, W1_sharded, result, visualize)
    
    best_config = min(results, key=lambda x: results[x][1])
    print("\nBest Configuration:")
    print_results(best_config, *results[best_config], NUM_ITERATIONS, A, W1, A_sharded, W1_sharded, result, visualize)

if __name__ == "__main__":
    main()