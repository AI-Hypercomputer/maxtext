import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.profiler import trace, save_device_memory_profile
import itertools
import datetime
import string
import tempfile
import json
import time
from typing import Tuple, Callable, Dict, Any, List, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt
import random
from functools import partial
import os


def create_trace_name(
  mesh_shape: Tuple[int, ...],
  batch_size: int,
  embed_size: int,
  mlp_size: int,
  mesh_axis_names: Tuple[str, ...],
  explicit_layout: Optional[List[List[int]]] = None
) -> str:
  mesh_str = "x".join(map(str, mesh_shape))
  axes_str = "_".join(mesh_axis_names)
  layout_str = "custom" if explicit_layout else "standard"
  return f"matmul_mesh_{mesh_str}_axes_{axes_str}_layout_{layout_str}"


def create_mesh(mesh_shape: Tuple[int, int], mesh_axis_names: Tuple[str, str]) -> Mesh:
  devices = mesh_utils.create_device_mesh(mesh_shape)
  return Mesh(devices=devices, axis_names=mesh_axis_names)


def matmul(A: jnp.ndarray, W1: jnp.ndarray, mesh: jax.sharding.Mesh) -> jnp.ndarray:
  activation_spec = P(None, "model")
  activation_sharding = NamedSharding(mesh, activation_spec)
  
  @partial(jax.jit, out_shardings=activation_sharding)
  def f(_A, _weights):
    result = _A @ _weights
    return jax.lax.with_sharding_constraint(result, activation_sharding)
  
  return f(A, W1)


def create_arrays(batch_size: int, embed_size: int, mlp_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
  key = jax.random.PRNGKey(0)  # You can change the seed if desired
  key_A, key_W1 = jax.random.split(key)

  A = jax.random.normal(key_A, (batch_size, embed_size), dtype=jnp.bfloat16)
  W1 = jax.random.normal(key_W1, (embed_size, mlp_size), dtype=jnp.bfloat16)

  return A, W1


def shard_arrays(mesh: Mesh, A: jnp.ndarray, W1: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  # For A (batch_size, embed_size)
  activation_spec = P(None, "model")
  activation_sharding = NamedSharding(mesh, activation_spec)
  A_sharded = jax.device_put(A, activation_sharding)

  # For W1 (embed_size, mlp_size)
  weight_spec = P("model", None)
  weight_sharding = NamedSharding(mesh, weight_spec)
  W1_sharded = jax.device_put(W1, weight_sharding)
  
  return A_sharded, W1_sharded


def visualize_sharding(A_sharded: jnp.ndarray, W1_sharded: jnp.ndarray, mesh_shape: Tuple[int, ...]):
  print(f"Sharding A: {A_sharded.shape=}, {mesh_shape=}")
  jax.debug.visualize_array_sharding(A_sharded)
  print("\n")
  print(f"Sharding W1: {W1_sharded.shape=}, {mesh_shape=}")
  jax.debug.visualize_array_sharding(W1_sharded)
  print("\n")


def create_standard_mesh(mesh_shape: Tuple[int, ...], mesh_axis_names: Tuple[str, ...]) -> Mesh:
  devices = mesh_utils.create_device_mesh(mesh_shape)
  return Mesh(devices=devices, axis_names=mesh_axis_names)


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
  ordered_flat_devices = sorted(
    np.array(devices).flatten(), key=lambda x: x.id
  )
  result = np.reshape(ordered_flat_devices, (2,) * log_len).transpose(new_axis_order[::-1]).reshape(ici_mesh_shape)
  return result


def create_balanced_2d_mesh(mesh_shape: Tuple[int, ...], mesh_axis_names: Tuple[str, ...]) -> Mesh:
  nested_balanced_2d_devices = make_nested_balanced_2d_devices(jax.devices(), mesh_shape)
  return Mesh(nested_balanced_2d_devices, mesh_axis_names)


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


def format_bytes(size_in_bytes: int) -> str:
  """Convert bytes to human-readable format."""
  for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
    if size_in_bytes < 1024.0:
      return f"{size_in_bytes:.2f} {unit}"
    size_in_bytes /= 1024.0
  return f"{size_in_bytes:.2f} PB"


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


def flop_estimate(A: jnp.ndarray, W1: jnp.ndarray) -> int:
  return 2 * A.shape[0] * A.shape[1] * W1.shape[1]


def compute_efficiency(time: float, flops: int) -> float:
  return flops / (time * 1e12)  # TFLOP/s


def run_sharding_experiment(
  mesh_shape: Tuple[int, ...],
  batch_size: int,
  embed_size: int,
  mlp_size: int,
  num_iterations: int = 100,
  use_visuals: bool = False,
  mesh_creator: Callable = create_standard_mesh,
  mesh_axis_names: Tuple[str, ...] = ("model", "sequence"),
  explicit_layout: Optional[List[List[int]]] = None
) -> Tuple[Dict[str, Any], Mesh]:
  if explicit_layout:
    mesh = create_explicit_layout_mesh(explicit_layout, mesh_axis_names)
  else:
    mesh = mesh_creator(mesh_shape, mesh_axis_names)
  
  A, W1 = create_arrays(batch_size, embed_size, mlp_size)
  A_sharded, W1_sharded = shard_arrays(mesh, A, W1)
  
  if use_visuals:
    visualize_sharding(A_sharded, W1_sharded, mesh_shape)
  
  trace_name = create_trace_name(mesh_shape, batch_size, embed_size, mlp_size, mesh_axis_names, explicit_layout)
  with tempfile.TemporaryDirectory() as tmpdir:
    trace_loc = os.path.join(tmpdir, trace_name)
    
    matmul_fn = create_matmul_function(mesh)
    
    dummy = object()  # Use a dummy object to prevent constant folding
    for _ in range(10):
      result = matmul_fn(A_sharded, W1_sharded, dummy)
      jax.block_until_ready(result)
    
    # Main benchmark loop
    jax.profiler.start_trace(trace_loc)
    times = []
    for _ in range(num_iterations):
      start = time.perf_counter()
      result = matmul_fn(A_sharded, W1_sharded, dummy)
      jax.block_until_ready(result)
      end = time.perf_counter()
      times.append(end - start)
    
    jax.profiler.stop_trace()
    
    print(f"Profiler trace saved to: {trace_loc}")
  
  mean_time = np.mean(times)
  std_dev = np.std(times, ddof=1) 
  
  memory_profile = memory_usage_profile()
  flops = flop_estimate(A, W1)
  efficiency = compute_efficiency(mean_time, flops)

  return {
    "mean_time": mean_time,
    "std_dev": std_dev,
    "memory_utilization": memory_profile['memory_utilization'],
    "total_memory": memory_profile['total_memory'],
    "used_memory": memory_profile['used_memory'],
    "total_memory_formatted": memory_profile['total_memory_formatted'],
    "used_memory_formatted": memory_profile['used_memory_formatted'],
    "efficiency": efficiency,
    "trace_name": trace_name
  }, mesh


def benchmark_mesh_configurations(
  mesh_configs: List[Tuple[Tuple[int, ...], Callable, Tuple[str, ...]]],
  batch_size: int,
  embed_size: int,
  mlp_size: int,
  num_iterations: int = 100,
  use_visuals: bool = False
) -> Tuple[Dict[Tuple[int, ...], Dict[str, Any]], Dict[str, Mesh]]:
  results = {}
  meshes = {}
  for mesh_shape, mesh_creator, mesh_axis_names in mesh_configs:
    print(f"Running experiment for mesh shape: {mesh_shape}")
    result, mesh = run_sharding_experiment(
      mesh_shape, batch_size, embed_size, mlp_size, 
      num_iterations, use_visuals, mesh_creator, mesh_axis_names
    )
    results[mesh_shape] = result
    meshes[f"{mesh_creator.__name__}_{mesh_shape}"] = mesh
  return results, meshes


def format_experiment_results(config_name: str, results: Dict[str, Any], mesh: Mesh, num_iterations: int) -> str:
  return f"""
Experiment: {config_name}
Mesh Shape: {tuple(mesh.shape.values())}
Mesh Axes: {mesh.axis_names}
Matrix Multiplication Time (avg of {num_iterations} runs):
  Mean: {results['mean_time']:.6f} seconds
  Std Dev: {results['std_dev']:.6f} seconds
Memory Utilization: {results['memory_utilization']:.2%}
Total Memory: {results['total_memory_formatted']}
Used Memory: {results['used_memory_formatted']}
Computational Efficiency: {results['efficiency']:.2f} TFLOP/s

{compare_device_placement(mesh)}
{'=' * 50}
"""

def compare_device_placement(mesh: Mesh) -> str:
  mesh_shape = tuple(mesh.shape.values())
  devices_array = np.array(mesh.devices).reshape(mesh_shape)
  
  compact_mesh = np.empty(mesh_shape, dtype=object)
  for index in np.ndindex(mesh_shape):
    device = devices_array[index]
    compact_mesh[index] = f"D{device.id}"
  
  return f"Compact Mesh Representation:\n{compact_mesh}\n"


def create_explicit_layout_mesh(device_layout: List[List[int]], mesh_axis_names: Tuple[str, ...]) -> Mesh:
  devices = jax.devices()
  if max(max(row) for row in device_layout) >= len(devices):
    raise ValueError(f"Device layout contains indices that exceed the number of available devices ({len(devices)})")

  new_devices = [[devices[idx] for idx in row] for row in device_layout]
  mesh_shape = (len(device_layout), len(device_layout[0]))
  
  if len(mesh_axis_names) != len(mesh_shape):
    raise ValueError(f"Number of axis names ({len(mesh_axis_names)}) does not match mesh shape ({mesh_shape})")

  return Mesh(np.array(new_devices), mesh_axis_names)


if __name__ == "__main__":
  GLOBAL_BATCH = 8 * 1024
  EMBED = 8192
  MLP = 28672
  num_iterations = 100

  mesh_configs = [
    # ((1, 8), create_standard_mesh, ("model", "sequence")),
    # ((2, 4), create_standard_mesh, ("sequence", "model")),
    ((4, 2), create_standard_mesh, ("model", "sequence")),
    ((4, 2), create_balanced_2d_mesh, ("model", "sequence")),
    # ((8, 1), create_standard_mesh, ("model", "sequence")),
    ((1, 2, 4), create_standard_mesh, ("placeholder", "sequence", "model")),
    ((1, 4, 2), create_balanced_2d_mesh, ("placeholder", "model", "sequence")),
    # ((2, 1, 4), create_standard_mesh, ("sequence", "placeholder", "model")),
    # ((2, 4, 1), create_standard_mesh, ("sequence", "model", "placeholder")),
    # ((4, 1, 2), create_standard_mesh, ("model", "placeholder", "sequence")),
    # ((4, 2, 1), create_standard_mesh, ("model", "sequence", "placeholder")),
  ]

  results = {}
  meshes = {}

  for mesh_shape, mesh_creator, mesh_axis_names in mesh_configs:
    config_name = f"{mesh_creator.__name__}_{mesh_shape}"
    result, mesh = run_sharding_experiment(
      mesh_shape, GLOBAL_BATCH, EMBED, MLP,
      num_iterations=num_iterations, use_visuals=False,
      mesh_creator=mesh_creator, mesh_axis_names=mesh_axis_names
    )
    results[config_name] = result
    meshes[config_name] = mesh
    print(format_experiment_results(config_name, result, mesh, num_iterations))

  explicit_layouts = [
    [[0, 4], [1, 5], [3, 7], [2, 6]],
    [[0, 1], [2, 3], [4, 5], [6, 7]],
    [[0, 1, 2, 3], [4, 5, 6, 7]],
  ]

  for i, explicit_layout in enumerate(explicit_layouts):
    explicit_mesh_shape = (len(explicit_layout), len(explicit_layout[0]))
    config_name = f"explicit_layout_{i}"
    
    explicit_result, explicit_mesh = run_sharding_experiment(
      explicit_mesh_shape, GLOBAL_BATCH, EMBED, MLP,
      num_iterations=num_iterations, use_visuals=False,
      mesh_axis_names=("model", "sequence"),
      explicit_layout=explicit_layout
    )
    
    results[config_name] = explicit_result
    meshes[config_name] = explicit_mesh
    print(format_experiment_results(config_name, explicit_result, explicit_mesh, num_iterations))

  best_config = min(results, key=lambda x: results[x]['mean_time'])
  print("\nBest Configuration:")
  print(format_experiment_results(best_config, results[best_config], meshes[best_config], num_iterations))
