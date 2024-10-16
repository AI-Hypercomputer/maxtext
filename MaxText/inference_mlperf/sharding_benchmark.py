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


def simple_timeit(f, *args, tries = 10, task = None):
    '''Simple utility to time a function for multiple runs'''
    assert task is not None

    trace_name = f"t_{task}_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    trace_dir = f"/tmp/{trace_name}"

    outcomes_ms = []
    jax.block_until_ready(f(*args)) #warm it up!
    jax.profiler.start_trace(trace_dir)

    for _ in range(tries):
        s = datetime.datetime.now()
        jax.block_until_ready(f(*args))
        e = datetime.datetime.now()
        outcomes_ms.append(1000*(e-s).total_seconds())
    jax.profiler.stop_trace()

    average_time_ms = sum(outcomes_ms)/len(outcomes_ms)
    print(f"{task}: average time milliseconds: {average_time_ms:.2f}, trace {trace_dir}")
    return average_time_ms

def create_mesh(mesh_shape: Tuple[int, int], mesh_axis_names: Tuple[str, str]) -> Mesh:
  """
  Creates a JAX mesh with the specified shape and axis names.

  Args:
    mesh_shape: A tuple representing the shape of the mesh.
    mesh_axis_names: A tuple of strings representing the names of the mesh axes.

  Returns:
    A JAX Mesh object.
  """
  devices = mesh_utils.create_device_mesh(mesh_shape)
  return Mesh(devices=devices, axis_names=mesh_axis_names)

def create_arrays(batch_size: int, embed_size: int, mlp_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """
  Creates two NumPy arrays for use in matrix multiplication.

  Args:
    batch_size: The size of the batch dimension.
    embed_size: The size of the embedding dimension.
    mlp_size: The size of the MLP dimension.

  Returns:
    A tuple containing two NumPy arrays: A and W1.
  """
  A = jnp.ones((batch_size, embed_size), dtype=jnp.bfloat16)
  W1 = jnp.ones((embed_size, mlp_size), dtype=jnp.bfloat16)
  return A, W1

def shard_arrays(mesh: Mesh, A: jnp.ndarray, W1: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """
  Shards the input arrays A and W1 according to the specified mesh.

  Args:
    mesh: The JAX Mesh object.
    A: The activation array.
    W1: The weight array.

  Returns:
    A tuple containing the sharded activation array (A_sharded) and the sharded weight array (W1_sharded).
  """
  activation_sharding = NamedSharding(mesh, P(None, "model"))
  weight_sharding = NamedSharding(mesh, P("model",))
  A_sharded = jax.device_put(A, activation_sharding)
  W1_sharded = jax.device_put(W1, weight_sharding)
  return A_sharded, W1_sharded

def visualize_sharding(A_sharded: jnp.ndarray, W1_sharded: jnp.ndarray, mesh_shape: Tuple[int, int]):
  """
  Visualizes the sharding of the arrays A and W1.

  Args:
    A_sharded: The sharded activation array.
    W1_sharded: The sharded weight array.
    mesh_shape: The shape of the mesh.
  """
  print(f"Sharding A: {A_sharded.shape=}, {mesh_shape=}")
  jax.debug.visualize_array_sharding(A_sharded)
  print("\n")
  print(f"Sharding W1: {W1_sharded.shape=}, {mesh_shape=}")
  jax.debug.visualize_array_sharding(W1_sharded)
  print("\n")

def create_standard_mesh(mesh_shape: Tuple[int, ...], mesh_axis_names: Tuple[str, ...]) -> Mesh:
  """
  Creates a standard JAX mesh with devices assigned in a sequential order.

  Args:
    mesh_shape: A tuple representing the shape of the mesh.
    mesh_axis_names: A tuple of strings representing the names of the mesh axes.

  Returns:
    A JAX Mesh object.
  """
  devices = mesh_utils.create_device_mesh(mesh_shape)
  return Mesh(devices=devices, axis_names=mesh_axis_names)

def make_nested_balanced_2d_devices(devices: Sequence[jax.Device], ici_mesh_shape: Sequence[int]) -> Sequence[jax.Device]:
  """
  Arranges devices in a balanced 2D mesh, prioritizing locality.

  Args:
    devices: A sequence of JAX devices.
    ici_mesh_shape: The desired shape of the 2D mesh.

  Returns:
    A nested sequence of devices arranged in a balanced 2D mesh.
  """
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
  """
  Creates a balanced 2D JAX mesh with devices arranged for locality.

  Args:
    mesh_shape: A tuple representing the shape of the mesh.
    mesh_axis_names: A tuple of strings representing the names of the mesh axes.

  Returns:
    A JAX Mesh object.
  """
  nested_balanced_2d_devices = make_nested_balanced_2d_devices(jax.devices(), mesh_shape)
  return Mesh(nested_balanced_2d_devices, mesh_axis_names)

def create_random_mesh(mesh_shape: Tuple[int, ...], mesh_axis_names: Tuple[str, ...]) -> Mesh:
  """
  Creates a JAX mesh with devices assigned randomly.

  Args:
    mesh_shape: A tuple representing the shape of the mesh.
    mesh_axis_names: A tuple of strings representing the names of the mesh axes.

  Returns:
    A JAX Mesh object.
  """
  num_devices = np.prod(mesh_shape)
  devices = jax.devices()[:num_devices]
  np.random.shuffle(devices)
  devices = np.array(devices).reshape(mesh_shape)
  return Mesh(devices=devices, axis_names=mesh_axis_names)

def time_matrix_multiply(A: jnp.ndarray, W1: jnp.ndarray, num_iterations: int = 100) -> Tuple[float, float]:
  """
  Times the execution of a matrix multiplication operation.

  Args:
    A: The activation array.
    W1: The weight array.
    num_iterations: The number of iterations to run.

  Returns:
    A tuple containing the mean execution time and standard deviation.
  """
  @jax.jit
  def matmul(A, W1):
    result = jnp.dot(A, W1)
    return result

  
  # Warm up
  for _ in range(10):
    jax.block_until_ready(matmul(A, W1))
  
  times = []
  for _ in range(num_iterations):
    start = time.perf_counter()
    jax.block_until_ready(matmul(A, W1))
    end = time.perf_counter()
    times.append(end - start)
  
  mean_time = np.mean(times)
  std_dev = np.std(times)
  
  return mean_time, std_dev

def memory_usage_profile() -> Dict[str, Any]:
  """
  Profiles the memory usage of the current JAX process.

  Returns:
    A dictionary containing the total memory, used memory, and memory utilization.
  """
  memory_stats = jax.device_get(jax.devices()[0].memory_stats())
  total_memory = memory_stats['bytes_limit'] 
  used_memory = memory_stats['bytes_in_use'] 
  
  return {
    "total_memory": total_memory,
    "used_memory": used_memory,
    "memory_utilization": used_memory / total_memory if total_memory > 0 else 0
  }

def flop_estimate(A: jnp.ndarray, W1: jnp.ndarray) -> int:
  """
  Estimates the number of floating-point operations (FLOPs) for a matrix multiplication.

  Args:
    A: The activation array.
    W1: The weight array.

  Returns:
    The estimated number of FLOPs.
  """
  return 2 * A.shape[0] * A.shape[1] * W1.shape[1]

def compute_efficiency(time: float, flops: int) -> float:
  """
  Calculates the computational efficiency in TFLOP/s.

  Args:
    time: The execution time in seconds.
    flops: The number of FLOPs.

  Returns:
    The computational efficiency in TFLOP/s.
  """
  return flops / (time * 1e12)  # TFLOP/s

def run_sharding_experiment(
    mesh_shape: Tuple[int, ...],
    batch_size: int,
    embed_size: int,
    mlp_size: int,
    num_iterations: int = 100,
    use_visuals: bool = True,
    mesh_creator: Callable = create_standard_mesh,
    mesh_axis_names: Tuple[str, ...] = ("model", "seq"),
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
    
    mean_time, std_dev = time_matrix_multiply(A_sharded, W1_sharded, num_iterations)
    
    memory_profile = memory_usage_profile()
    flops = flop_estimate(A, W1)
    efficiency = compute_efficiency(mean_time, flops)

    return {
        "mean_time": mean_time,
        "std_dev": std_dev,
        "memory_utilization": memory_profile['memory_utilization'],
        "total_memory": memory_profile['total_memory'],
        "used_memory": memory_profile['used_memory'],
        "efficiency": efficiency
    }, mesh


def benchmark_mesh_configurations(
    mesh_configs: List[Tuple[Tuple[int, ...], Callable, Tuple[str, ...]]],
    batch_size: int,
    embed_size: int,
    mlp_size: int,
    num_iterations: int = 100,
    use_visuals: bool = False
) -> Tuple[Dict[Tuple[int, ...], Dict[str, Any]], Dict[str, Mesh]]:
  """
  Benchmarks different mesh configurations.

  Args:
    mesh_configs: A list of tuples containing mesh shape, mesh creator function, and mesh axis names.
    batch_size: The size of the batch dimension.
    embed_size: The size of the embedding dimension.
    mlp_size: The size of the MLP dimension.
    num_iterations: The number of iterations to run.
    use_visuals: Whether to visualize the sharding.

  Returns:
    A tuple containing a dictionary of results and a dictionary of meshes.
  """
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
    """
    Formats the experiment results into a readable string.
    """
    return f"""
Experiment: {config_name}
Mesh Shape: {tuple(mesh.shape.values())}
Matrix Multiplication Time (avg of {num_iterations} runs):
  Mean: {results['mean_time']:.6f} seconds
  Std Dev: {results['std_dev']:.6f} seconds
Memory Utilization: {results['memory_utilization']:.2%}
Total Memory: {results['total_memory']:,} bytes
Used Memory: {results['used_memory']:,} bytes
Computational Efficiency: {results['efficiency']:.2f} TFLOP/s

Device Placement:
{compare_device_placement(mesh)}
{'=' * 50}
"""

def compare_device_placement(mesh: Mesh) -> str:
    """
    Generates a string representation of the device placement for a given mesh.
    """
    mesh_shape = tuple(mesh.shape.values())
    devices_array = np.array(mesh.devices).reshape(mesh_shape)
    
    compact_mesh = np.empty(mesh_shape, dtype=object)
    for index in np.ndindex(mesh_shape):
        device = devices_array[index]
        compact_mesh[index] = f"D{device.id}"
    
    return f"Compact Mesh Representation:\n{compact_mesh}\n"

def get_device_topology() -> Dict[int, Dict[str, Any]]:
  """
  Gets the topology of the available JAX devices.

  Returns:
    A dictionary mapping device IDs to device information.
  """
  devices = jax.devices()
  topology = {}
  for device in devices:
    if hasattr(device, 'client'):
      # For TPUs
      topology[device.id] = {
        'platform': device.platform,
        'device_kind': device.device_kind,
        'client': device.client.platform_version
      }
    else:
      # For GPUs and CPUs
      topology[device.id] = {
        'platform': device.platform,
        'device_kind': device.device_kind,
      }
    # Add more hardware-specific information if available
    if hasattr(device, 'core_on_chip'):
      topology[device.id]['core_on_chip'] = device.core_on_chip
    if hasattr(device, 'slice_index'):
      topology[device.id]['slice_index'] = device.slice_index
  return topology

def generate_permutations(shape: Tuple[int, int]) -> List[List[List[int]]]:
    """
    Generates all possible permutations for a given mesh shape.

    Args:
        shape: A tuple representing the shape of the mesh (rows, cols).

    Returns:
        A list of permutations, where each permutation is a list of lists of integers.
    """
    total_devices = shape[0] * shape[1]
    all_devices = list(range(total_devices))
    all_permutations = list(itertools.permutations(all_devices))
    
    return [list(zip(*[iter(perm)]*shape[1])) for perm in all_permutations]

def generate_unique_permutations(shape: Tuple[int, int]) -> List[List[List[int]]]:
    """
    Generates unique permutations for a given mesh shape, exploiting symmetry.

    Args:
        shape: A tuple representing the shape of the mesh (rows, cols).

    Returns:
        A list of unique permutations, where each permutation is a list of lists of integers.
    """
    total_devices = shape[0] * shape[1]
    all_devices = list(range(total_devices))
    
    # Generate all permutations of the first row
    first_row_perms = list(itertools.permutations(all_devices, shape[1]))
    
    unique_perms = []
    for first_row in first_row_perms:
        remaining_devices = set(all_devices) - set(first_row)
        remaining_perms = list(itertools.permutations(remaining_devices, total_devices - shape[1]))
        
        for perm in remaining_perms:
            full_perm = list(first_row) + list(perm)
            unique_perms.append(list(zip(*[iter(full_perm)]*shape[1])))
    
    return unique_perms

def generate_heuristic_permutations(shape: Tuple[int, int], num_samples: int) -> List[List[List[int]]]:
    """
    Generates a set of permutations using heuristics.

    Args:
        shape: A tuple representing the shape of the mesh (rows, cols).
        num_samples: Number of permutations to generate.

    Returns:
        A list of permutations generated using heuristics.
    """
    total_devices = shape[0] * shape[1]
    all_devices = list(range(total_devices))
    
    heuristic_perms = []
    
    # Include some "natural" orderings
    heuristic_perms.append(list(zip(*[iter(all_devices)]*shape[1])))  # Standard order
    heuristic_perms.append(list(zip(*[iter(all_devices[::-1])]*shape[1])))  # Reverse order
    
    # Generate some random permutations
    for _ in range(num_samples - 2):
        random_perm = random.sample(all_devices, total_devices)
        heuristic_perms.append(list(zip(*[iter(random_perm)]*shape[1])))
    
    return heuristic_perms

def run_optimized_permutation_experiments(
    mesh_shapes: List[Tuple[int, int]],
    batch_size: int,
    embed_size: int,
    mlp_size: int,
    num_iterations: int = 100,
    top_k: int = 5,
    max_permutations: int = 1000,
    use_heuristics: bool = True
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    results = {}

    for shape in mesh_shapes:
        print(f"\nRunning experiments for mesh shape {shape}")
        
        if use_heuristics:
            permutations = generate_heuristic_permutations(shape, max_permutations)
        else:
            permutations = generate_unique_permutations(shape)[:max_permutations]
        
        print(f"Testing {len(permutations)} permutations")
        
        shape_results = []
        for i, perm in enumerate(permutations):
            print(f"Testing permutation {i+1}/{len(permutations)}")
            try:
                result, mesh = run_sharding_experiment(
                    shape, batch_size, embed_size, mlp_size,
                    num_iterations=num_iterations, use_visuals=False,
                    mesh_creator=lambda shape, axis_names: create_custom_permuted_mesh(shape, perm, axis_names),
                    mesh_axis_names=("model", "seq")
                )
                shape_results.append({"permutation": perm, **result})
            except Exception as e:
                print(f"Error occurred for permutation {i+1}: {str(e)}")
                continue
        
        # Sort results by mean_time and get top K
        top_results = sorted(shape_results, key=lambda x: x['mean_time'])[:top_k]
        results[shape] = top_results

    return results

def run_permutation_experiments(
    mesh_shapes: List[Tuple[int, int]],
    batch_size: int,
    embed_size: int,
    mlp_size: int,
    num_iterations: int = 100,
    top_k: int = 5
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """
    Runs experiments for all permutations of given mesh shapes and returns the top K results for each shape.

    Args:
        mesh_shapes: List of mesh shapes to test.
        batch_size: The size of the batch dimension.
        embed_size: The size of the embedding dimension.
        mlp_size: The size of the MLP dimension.
        num_iterations: The number of iterations for each experiment.
        top_k: The number of top results to return for each shape.

    Returns:
        A dictionary mapping mesh shapes to lists of top K results.
    """
    results = {}

    for shape in mesh_shapes:
        print(f"\nRunning experiments for mesh shape {shape}")
        shape_results = []
        permutations = generate_permutations(shape)
        
        for i, perm in enumerate(permutations):
            print(f"Testing permutation {i+1}/{len(permutations)}")
            result, _ = run_custom_permuted_experiment(
                shape, perm, batch_size, embed_size, mlp_size,
                num_iterations=num_iterations, use_visuals=False
            )
            shape_results.append({"permutation": perm, **result})
        
        # Sort results by mean_time and get top K
        top_results = sorted(shape_results, key=lambda x: x['mean_time'])[:top_k]
        results[shape] = top_results

    return results

def create_custom_permuted_mesh(mesh_shape: Tuple[int, ...], permutation: List[List[int]], mesh_axis_names: Tuple[str, ...]) -> Mesh:
    """
    Creates a JAX mesh with a custom permuted device layout.

    Args:
        mesh_shape: A tuple representing the shape of the mesh.
        permutation: A list of lists representing the permutation of device indices.
        mesh_axis_names: A tuple of strings representing the names of the mesh axes.

    Returns:
        A JAX Mesh object with the custom permuted device layout.
    """
    devices = jax.devices()
    if len(devices) < np.prod(mesh_shape):
        raise ValueError(f"Not enough devices. Required: {np.prod(mesh_shape)}, Available: {len(devices)}")

    if len(permutation) != mesh_shape[0] or any(len(row) != mesh_shape[1] for row in permutation):
        raise ValueError(f"Permutation shape {len(permutation)}x{len(permutation[0])} does not match mesh shape {mesh_shape}")

    new_devices = [[devices[idx] for idx in row] for row in permutation]
    return Mesh(np.array(new_devices), mesh_axis_names)

def create_explicit_layout_mesh(device_layout: List[List[int]], mesh_axis_names: Tuple[str, ...]) -> Mesh:
    """
    Creates a JAX mesh with an explicitly specified device layout.

    Args:
        device_layout: A list of lists representing the desired device layout.
                       Each inner list represents a row in the mesh, and each
                       integer represents the index of the device in jax.devices().
        mesh_axis_names: A tuple of strings representing the names of the mesh axes.

    Returns:
        A JAX Mesh object with the specified device layout.

    Raises:
        ValueError: If there are not enough devices or if the layout is invalid.
    """
    devices = jax.devices()
    if max(max(row) for row in device_layout) >= len(devices):
        raise ValueError(f"Device layout contains indices that exceed the number of available devices ({len(devices)})")

    new_devices = [[devices[idx] for idx in row] for row in device_layout]
    mesh_shape = (len(device_layout), len(device_layout[0]))
    
    if len(mesh_axis_names) != len(mesh_shape):
        raise ValueError(f"Number of axis names ({len(mesh_axis_names)}) does not match mesh shape ({mesh_shape})")

    return Mesh(np.array(new_devices), mesh_axis_names)

def run_custom_permuted_experiment(
    mesh_shape: Tuple[int, ...],
    permutation: List[List[int]],
    batch_size: int,
    embed_size: int,
    mlp_size: int,
    num_iterations: int = 100,
    use_visuals: bool = True,
    mesh_axis_names: Tuple[str, ...] = ("model", "seq")
) -> Tuple[Dict[str, Any], Mesh]:
    """
    Runs a sharding experiment with a custom permuted mesh layout.

    Args:
        mesh_shape: The shape of the mesh.
        permutation: A list of lists representing the permutation of device indices.
        batch_size: The size of the batch dimension.
        embed_size: The size of the embedding dimension.
        mlp_size: The size of the MLP dimension.
        num_iterations: The number of iterations to run.
        use_visuals: Whether to visualize the sharding.
        mesh_axis_names: The names of the mesh axes.

    Returns:
        A tuple containing a dictionary of results and the created mesh.
    """
    mesh = create_custom_permuted_mesh(mesh_shape, permutation, mesh_axis_names)
    print(f"\nRunning experiment for custom permuted mesh with shape {mesh_shape}")
    print("Mesh:")
    print(mesh)
    print()

    A, W1 = create_arrays(batch_size, embed_size, mlp_size)
    A_sharded, W1_sharded = shard_arrays(mesh, A, W1)
    
    if use_visuals:
        visualize_sharding(A_sharded, W1_sharded, mesh_shape)
    
    mean_time, std_dev = time_matrix_multiply(A_sharded, W1_sharded, num_iterations)
    print(f"Matrix multiplication time (average of {num_iterations} runs):")
    print(f"  Mean: {mean_time:.6f} seconds")
    print(f"  Standard Deviation: {std_dev:.6f} seconds")

    memory_profile = memory_usage_profile()
    print(f"Memory utilization: {memory_profile['memory_utilization']:.2%}")
    print(f"Total memory: {memory_profile['total_memory']} bytes")
    print(f"Used memory: {memory_profile['used_memory']} bytes")

    flops = flop_estimate(A, W1)
    efficiency = compute_efficiency(mean_time, flops)
    print(f"Computational efficiency: {efficiency:.2f} TFLOP/s")

    compare_device_placement({"custom_permuted_mesh": mesh})

    print("\n" + "="*50 + "\n")

    return {
        "mean_time": mean_time,
        "std_dev": std_dev,
        "memory_utilization": memory_profile['memory_utilization'],
        "total_memory": memory_profile['total_memory'],
        "used_memory": memory_profile['used_memory'],
        "efficiency": efficiency
    }, mesh

if __name__ == "__main__":
    GLOBAL_BATCH = 8 * 1024
    EMBED = 8192
    MLP = 28672
    num_iterations = 40

    mesh_configs = [
        ((8, 1), create_standard_mesh, ("model", "seq")),
        ((4, 2), create_standard_mesh, ("model", "seq")),
        ((2, 4), create_standard_mesh, ("model", "seq")),
        ((1, 8), create_standard_mesh, ("model", "seq")),
        ((4, 2), create_balanced_2d_mesh, ("model", "seq")),
        ((2, 4), create_balanced_2d_mesh, ("model", "seq")),
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

    # Run experiment with explicit device layout
    explicit_layout = [[0, 4], [1, 5], [3, 7], [2, 6]]
    explicit_mesh_shape = (4, 2)
    explicit_result, explicit_mesh = run_sharding_experiment(
        explicit_mesh_shape, GLOBAL_BATCH, EMBED, MLP,
        num_iterations=num_iterations, use_visuals=False,
        mesh_axis_names=("model", "seq"),
        explicit_layout=explicit_layout
    )
    config_name = "explicit_layout"
    results[config_name] = explicit_result
    meshes[config_name] = explicit_mesh
    print(format_experiment_results(config_name, explicit_result, explicit_mesh, num_iterations))

    # print("Starting optimized permutation testing...")
    # permuted_mesh_shapes = [(4, 2), (2, 4)]
    # permutation_results = run_optimized_permutation_experiments(
    #     permuted_mesh_shapes, GLOBAL_BATCH, EMBED, MLP,
    #     num_iterations=40, top_k=5, max_permutations=100, use_heuristics=True
    # )

    # print("\nTop 5 results for each mesh shape:")
    # for shape, top_results in permutation_results.items():
    #     print(f"\nMesh shape: {shape}")
    #     for i, result in enumerate(top_results, 1):
    #         print(f"  Rank {i}:")
    #         print(f"    Permutation: {result['permutation']}")
    #         print(f"    Mean time: {result['mean_time']:.6f} seconds")
    #         print(f"    Efficiency: {result['efficiency']:.2f} TFLOP/s")
    #         print(f"    Memory utilization: {result['memory_utilization']:.2%}")

    # # Find the overall best configuration
    # all_results = [(shape, result) for shape, shape_results in permutation_results.items() for result in shape_results]
    # best_permuted_config = min(all_results, key=lambda x: x[1]['mean_time'])
    # best_permuted_name = f"best_permuted_{best_permuted_config[0]}"
    # results[best_permuted_name] = best_permuted_config[1]
    # meshes[best_permuted_name] = create_custom_permuted_mesh(best_permuted_config[0], best_permuted_config[1]['permutation'], ("model", "seq"))

    
    # print("\nOverall best permuted configuration:")
    # print(f"  Mesh shape: {best_permuted_config[0]}")
    # print(f"  Permutation: {best_permuted_config[1]['permutation']}")
    # print(f"  Mean time: {best_permuted_config[1]['mean_time']:.6f} seconds")
    # print(f"  Efficiency: {best_permuted_config[1]['efficiency']:.2f} TFLOP/s")
    # print(f"  Memory utilization: {best_permuted_config[1]['memory_utilization']:.2%}")

    # Find and print the best configuration among all experiments
    best_config = min(results, key=lambda x: results[x]['mean_time'])
    best_config = min(results, key=lambda x: results[x]['mean_time'])
    print("\nBest Configuration Among All Experiments (Including Permuted):")
    print(format_experiment_results(best_config, results[best_config], meshes[best_config], num_iterations))
