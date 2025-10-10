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

"""Accelerator Resource Management Utilities for MaxText.

This module provides utilities for managing accelerator resources (TPU/GPU/CPU),
including initialization and cleanup of JAX distributed systems.

This is the SINGLE SOURCE OF TRUTH for JAX distributed system management.
All initialization and cleanup should go through this module to ensure
proper resource tracking and management.

Supports: TPU, GPU, and CPU backends.

INTER-PROCESS RESOURCE SHARING:
This module supports sharing resource information across multiple programs
using the same MaxText installation on one machine. Programs can check if
resources are currently in use by another process.
"""

import os
import gc
import json
import socket
import sys
import threading
import time
import fcntl
import psutil
import jax
import orbax.checkpoint as ocp
from etils import epath
from pathlib import Path
from MaxText import max_logging


# Global state tracking for resource management
_resource_lock = threading.Lock()
_initialization_metadata = {
    "is_initialized": False,
    "backend": None,
    "coordinator_address": None,
    "num_processes": None,
    "process_id": None,
    "initialization_time": None,
    "program_name": None,
}

# Inter-process communication paths
_LOCK_DIR = Path(os.getenv("MAXTEXT_RESOURCE_LOCK_DIR", "/tmp/maxtext_resources"))
_LOCK_FILE = _LOCK_DIR / "resource.lock"
_STATE_FILE = _LOCK_DIR / "resource_state.json"


def _ensure_lock_dir():
  """Ensure the lock directory exists."""
  _LOCK_DIR.mkdir(parents=True, exist_ok=True)


def _is_process_alive(pid):
  """Check if a process is still running.

  Args:
    pid: Process ID to check.

  Returns:
    bool: True if process exists and is running.
  """
  try:
    process = psutil.Process(pid)
    return process.is_running()
  except (psutil.NoSuchProcess, psutil.AccessDenied):
    return False


def _read_resource_state():
  """Read the current resource state from file.

  Returns:
    dict or None: Resource state or None if file doesn't exist or is invalid.
  """
  if not _STATE_FILE.exists():
    return None

  try:
    with open(_STATE_FILE, "r", encoding="utf-8") as f:
      state = json.load(f)
      # Check if the process is still alive
      if "pid" in state and not _is_process_alive(state["pid"]):
        max_logging.log(f"Stale resource lock detected (PID {state['pid']} is dead), cleaning up...")
        _STATE_FILE.unlink(missing_ok=True)
        return None
      return state
  except (json.JSONDecodeError, IOError):
    return None


def _write_resource_state(state):
  """Write resource state to file.

  Args:
    state: Dictionary containing resource state.
  """
  _ensure_lock_dir()
  with open(_STATE_FILE, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)


def _acquire_resource_lock(timeout=0):
  """Try to acquire the inter-process resource lock.

  Args:
    timeout: How long to wait for lock (seconds). 0 means non-blocking.

  Returns:
    file object or None: Lock file handle if acquired, None otherwise.
  """
  _ensure_lock_dir()
  lock_file = open(_LOCK_FILE, "w", encoding="utf-8")

  start_time = time.time()
  while True:
    try:
      fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
      return lock_file
    except IOError:
      if timeout == 0:
        lock_file.close()
        return None

      if time.time() - start_time >= timeout:
        lock_file.close()
        return None

      time.sleep(0.1)


def _release_resource_lock(lock_file):
  """Release the inter-process resource lock.

  Args:
    lock_file: Lock file handle from _acquire_resource_lock.
  """
  if lock_file:
    try:
      fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
      lock_file.close()
    except Exception as e:
      max_logging.log(f"Warning: Error releasing resource lock: {e}")


def is_resource_available():
  """Check if accelerator resources are available (not in use by another process).

  Returns:
    bool: True if resources are available, False if blocked by another process.

  Example:
    >>> if is_resource_available():
    >>>     initialize_jax_distributed()
    >>> else:
    >>>     print("Resources are in use by another process")
  """
  lock_file = _acquire_resource_lock(timeout=0)
  if lock_file is None:
    return False

  try:
    state = _read_resource_state()
    if state is not None and state.get("is_initialized", False):
      # Resources are in use
      _release_resource_lock(lock_file)
      return False

    _release_resource_lock(lock_file)
    return True
  except Exception:
    _release_resource_lock(lock_file)
    return True  # Assume available on error


def get_resource_holder_info():
  """Get information about which process is currently holding the resources.

  Returns:
    dict or None: Information about the process holding resources, or None if available.

  Example:
    >>> info = get_resource_holder_info()
    >>> if info:
    >>>     print(f"Resources held by PID {info['pid']}, started at {info['start_time']}")
  """
  state = _read_resource_state()
  if state is None or not state.get("is_initialized", False):
    return None
  return state


def wait_for_resources(timeout=300, poll_interval=5):
  """Wait for resources to become available.

  Args:
    timeout: Maximum time to wait in seconds.
    poll_interval: How often to check for availability in seconds.

  Returns:
    bool: True if resources became available, False if timeout.

  Example:
    >>> if wait_for_resources(timeout=60):
    >>>     initialize_jax_distributed()
    >>> else:
    >>>     print("Timeout waiting for resources")
  """
  start_time = time.time()
  while time.time() - start_time < timeout:
    if is_resource_available():
      return True

    holder_info = get_resource_holder_info()
    if holder_info:
      max_logging.log(
          f"Waiting for resources held by PID {holder_info.get('pid')} "
          f"(uptime: {time.time() - holder_info.get('initialization_time', time.time()):.1f}s)..."
      )

    time.sleep(poll_interval)

  return False


def _update_metadata(backend=None, coordinator_address=None, num_processes=None, process_id=None, program_name=None):
  """Update initialization metadata (thread-safe and persists to disk)."""
  with _resource_lock:
    _initialization_metadata["is_initialized"] = jax.distributed.is_initialized()
    _initialization_metadata["backend"] = backend or jax.default_backend()
    _initialization_metadata["coordinator_address"] = coordinator_address
    _initialization_metadata["num_processes"] = num_processes
    _initialization_metadata["process_id"] = process_id
    _initialization_metadata["initialization_time"] = time.time()
    _initialization_metadata["program_name"] = program_name or os.path.basename(sys.argv[0])

    # Persist to disk for inter-process sharing
    lock_file = _acquire_resource_lock(timeout=5)
    if lock_file:
      try:
        state = {
            "pid": os.getpid(),
            "is_initialized": _initialization_metadata["is_initialized"],
            "backend": _initialization_metadata["backend"],
            "coordinator_address": coordinator_address,
            "num_processes": num_processes,
            "process_id": process_id,
            "initialization_time": _initialization_metadata["initialization_time"],
            "hostname": os.uname().nodename,
            "program_name": _initialization_metadata["program_name"],
            "command_line": " ".join(sys.argv),
        }
        _write_resource_state(state)
      finally:
        _release_resource_lock(lock_file)


def _clear_metadata():
  """Clear initialization metadata (thread-safe and removes disk state)."""
  with _resource_lock:
    _initialization_metadata["is_initialized"] = False
    _initialization_metadata["backend"] = None
    _initialization_metadata["coordinator_address"] = None
    _initialization_metadata["num_processes"] = None
    _initialization_metadata["process_id"] = None
    _initialization_metadata["initialization_time"] = None
    _initialization_metadata["program_name"] = None

    # Clear disk state
    lock_file = _acquire_resource_lock(timeout=5)
    if lock_file:
      try:
        _STATE_FILE.unlink(missing_ok=True)
      finally:
        _release_resource_lock(lock_file)


# Helper functions for backend detection and coordinator management


def is_cpu_backend(raw_keys):
  """Determine whether MaxText is intended to run on a CPU backend."""
  return raw_keys.get("hardware") == "cpu"


def is_gpu_backend(raw_keys):
  """Determine whether MaxText is intended to run on a GPU backend."""
  return raw_keys.get("hardware") == "gpu"


def get_coordinator_ip_address():
  """Get coordinator IP Address with retries."""
  coordinator_address = ""
  coordinator_ip_address = ""
  if os.environ.get("JAX_COORDINATOR_ADDRESS") is not None:
    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    coordinator_found = False
    lookup_attempt = 1
    max_coordinator_lookups = 50
    while not coordinator_found and lookup_attempt <= max_coordinator_lookups:
      try:
        coordinator_ip_address = socket.gethostbyname(coordinator_address)
        coordinator_found = True
      except socket.gaierror:
        max_logging.log(
            f"Failed to recognize coordinator address {coordinator_address} on attempt {lookup_attempt}, retrying..."
        )
        lookup_attempt += 1
        time.sleep(5)
  max_logging.log(f"Coordinator IP address: {coordinator_ip_address}")
  return coordinator_ip_address


def get_initialization_metadata():
  """Get current initialization metadata (thread-safe).

  Returns:
    dict: Copy of initialization metadata.
  """
  with _resource_lock:
    return dict(_initialization_metadata)


# JAX initialization functions for specific backends


def _retrieve_jax_init_info(raw_keys):
  """Retrieve JAX init info from a local file."""
  JAX_INIT_INFO_FILE = "jax-init-info.txt"
  local_jax_init_info_file = epath.Path(raw_keys["local_checkpoint_directory"]) / JAX_INIT_INFO_FILE
  # Allow time for the JAX init info file to be populated by GKE. This is needed because the file is
  # only populated when the worker with process id of 0 is determined. After a disruption, although some
  # workers might be up and running, the init info file won't be populated until the node with process id
  # of 0 is known and this could take time. Using 900 seconds for now and it needs to be increased if the
  # "repair" time is longer.
  for i in range(900):
    if local_jax_init_info_file.exists():
      return local_jax_init_info_file.read_text().split("\n")[:2]
    max_logging.log(f"Unable to locate {JAX_INIT_INFO_FILE} after {i} seconds, sleeping for 1 second before retrying...")
    time.sleep(1)
  max_logging.log(
      f"Unable to locate {JAX_INIT_INFO_FILE} after 900 seconds," "returning empty process id and coordinator address."
  )
  return "", ""


def initialize_jax_for_gpu(raw_keys, program_name=None):
  """JAX distributed initialize for GPUs."""
  if os.environ.get("JAX_COORDINATOR_IP") is not None:
    coordinator_ip = str(os.getenv("JAX_COORDINATOR_IP"))
    coordinator_port = str(os.getenv("JAX_COORDINATOR_PORT"))
    coordinator_address = f"{coordinator_ip}:{coordinator_port}"
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=int(os.getenv("NNODES", "1")),
        process_id=int(os.getenv("NODE_RANK", "0")),
        initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
    )
    max_logging.log(f"JAX global devices: {jax.devices()}")

    # Update metadata with program name
    _update_metadata(
        backend="gpu",
        coordinator_address=coordinator_address,
        num_processes=int(os.getenv("NNODES", "1")),
        process_id=int(os.getenv("NODE_RANK", "0")),
        program_name=program_name,
    )


def initialize_jax_for_cpu(raw_keys, program_name=None):
  """JAX distributed initialize for CPUs. Includes retries until the coordinator is ready."""
  coordinator_ip_address = get_coordinator_ip_address()
  coordinator_address = coordinator_ip_address + ":1234"  # JAX coordinator port used in XPK
  # Env variables to be set in XPK or otherwise
  job_index = int(os.environ.get("JOB_INDEX", "0"))
  job_completion_index = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
  processes_in_job = int(os.environ.get("PROCESSES_IN_JOB", "1"))
  pid = job_index * processes_in_job + job_completion_index
  max_logging.log(f" Jax process id is {pid} ")
  # Explicit initialize is needed only for CPUs
  jax.distributed.initialize(
      coordinator_address=coordinator_address,
      process_id=pid,
      num_processes=int(os.environ.get("JAX_PROCESS_COUNT", "1")),
      initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
  )

  # Update metadata with program name
  _update_metadata(
      backend="cpu",
      coordinator_address=coordinator_address,
      num_processes=int(os.environ.get("JAX_PROCESS_COUNT", "1")),
      process_id=pid,
      program_name=program_name,
  )


def initialize_jax_for_tpu_with_emergency_checkpointing(raw_keys, program_name=None):
  """Initialize JAX distributed runtime for TPUs when emergency checkpointing is used.
  The information required to initialize JAX distributed runtime will be written by GKE to
  the local checkpoint directory. This function retrieves that information and initializes
  JAX distributed runtime.
  """
  process_id, coordinator_address = _retrieve_jax_init_info(raw_keys)

  if process_id != "" and coordinator_address != "":
    max_logging.log(
        f"Using {process_id} as the process_id and {coordinator_address} as the"
        " coordinator_address to initialize JAX distributed runtime..."
    )
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        process_id=int(process_id),
        initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
    )

    ocp.multihost.initialize_runtime_to_distributed_ids()
    ocp.multihost.initialize_distributed_to_device_ids()

    # Update metadata with program name
    _update_metadata(
        backend="tpu",
        coordinator_address=coordinator_address,
        process_id=int(process_id),
        program_name=program_name,
    )


def maybe_initialize_jax_distributed_system(raw_keys, program_name=None):
  """The best recipe to initialize the JAX Distributed System has varied over time.

  This is the main initialization function that handles all backends (TPU/GPU/CPU).

  Args:
    raw_keys: Configuration dictionary.
    program_name: Optional name for the program (e.g., run_name). Used for tracking.
  """
  # Import here to avoid circular dependency
  from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
      initialization,
  )
  initialize_multi_tier_checkpointing = initialization.initialize_multi_tier_checkpointing

  if raw_keys.get("skip_jax_distributed_system"):
    max_logging.log("Skipping jax distributed system due to skip_jax_distributed_system=True flag.")
    return
  if raw_keys.get("enable_single_controller"):
    max_logging.log("Skipping jax distributed system since its not needed for single controller.")
    return
  if jax.distributed.is_initialized():
    max_logging.log("Jax distributed system is already initialized.")
    # Update metadata even if already initialized
    _update_metadata(backend=jax.default_backend(), program_name=program_name)
    return
  if raw_keys.get("inference_benchmark_test"):
    # Disable initialization for inference benchmark test.
    return
  if raw_keys.get("compile_topology"):
    # Don't initialize jax distributed with AOT compilation
    return

  # Store program name for logging
  prog_name = program_name or raw_keys.get("run_name", "unknown")
  max_logging.log(f"Initializing JAX distributed system for program: {prog_name}")

  if is_gpu_backend(raw_keys):
    max_logging.log("Attempting to initialize the jax distributed system for GPU backend...")
    initialize_jax_for_gpu(raw_keys, program_name)
    max_logging.log("Jax distributed system initialized on GPU!")
  elif is_cpu_backend(raw_keys):
    max_logging.log("Attempting to initialize the jax distributed system for CPU backend...")
    initialize_jax_for_cpu(raw_keys, program_name)
    max_logging.log("Jax distributed system initialized on CPUs!")
  elif raw_keys.get("enable_multi_tier_checkpointing"):
    max_logging.log("Attempting to initialize the jax distributed system for multi-tier checkpointing...")
    initialize_multi_tier_checkpointing(
        local_checkpoint_directory=raw_keys["local_checkpoint_directory"],
        backup_interval_minutes=raw_keys["multi_tier_checkpointing_backup_interval_minutes"],
        run_name=raw_keys["run_name"],
        jax_initialization_timeout_seconds=raw_keys["jax_distributed_initialization_timeout"],
        data_parallelism=raw_keys["mtc_data_parallelism"],
    )
    max_logging.log("Jax distributed system initialized for multi-tier checkpointing!")
    _update_metadata(backend=jax.default_backend(), program_name=program_name)
  elif (raw_keys.get("enable_checkpointing") and raw_keys.get("compile_topology_num_slices", -1) == -1) or raw_keys.get(
      "hardware"
  ) == "gpu_multiprocess":
    max_logging.log("Attempting to initialize the jax distributed system...")
    if not raw_keys.get("enable_emergency_checkpoint"):
      jax.distributed.initialize(initialization_timeout=raw_keys["jax_distributed_initialization_timeout"])
      _update_metadata(backend=jax.default_backend(), program_name=program_name)
    else:
      if raw_keys.get("hardware") == "gpu_multiprocess":
        max_logging.log("Initializing jax distributed to support local checkpointing with GPUs...")
        jax.distributed.initialize(initialization_timeout=raw_keys["jax_distributed_initialization_timeout"])
        ocp.multihost.initialize_runtime_to_distributed_ids()
        ocp.multihost.initialize_distributed_to_device_ids()
        _update_metadata(backend=jax.default_backend(), program_name=program_name)
      else:
        initialize_jax_for_tpu_with_emergency_checkpointing(raw_keys, program_name)
    max_logging.log("Jax distributed system initialized!")


def initialize_jax_for_backend(
    coordinator_address=None, num_processes=None, process_id=None, timeout=300, program_name=None
):
  """Initialize JAX distributed system for any accelerator (TPU/GPU/CPU).

  This function initializes the JAX distributed system if it's not already initialized.
  It automatically detects and handles TPU, GPU, and CPU backends appropriately.

  This function also acquires an inter-process lock to prevent multiple programs
  from simultaneously initializing JAX on the same resources.

  Args:
    coordinator_address: Optional coordinator address (e.g., "10.0.0.1:1234").
                        If None, JAX will auto-detect for TPUs.
    num_processes: Optional number of processes. If None, auto-detected.
    process_id: Optional process ID. If None, auto-detected.
    timeout: Initialization timeout in seconds. Default is 300.

  Returns:
    bool: True if initialization was successful, False if already initialized.

  Raises:
    RuntimeError: If resources are already in use by another process.

  Example:
    >>> initialize_jax_for_backend()
    >>> # Your training code here
    >>> shutdown_jax_distributed()
  """
  if jax.distributed.is_initialized():
    max_logging.log("JAX distributed system is already initialized.")
    return False

  # Check if resources are available
  if not is_resource_available():
    holder_info = get_resource_holder_info()
    if holder_info:
      raise RuntimeError(
          f"Resources are already in use by PID {holder_info.get('pid')} "
          f"on host {holder_info.get('hostname')}. "
          f"Use wait_for_resources() or ensure other processes release resources first."
      )

  try:
    backend = jax.default_backend()
    max_logging.log(f"Initializing JAX distributed system for backend: {backend}")

    if backend == "tpu":
      # For TPU, JAX can auto-detect most settings
      if coordinator_address:
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
            initialization_timeout=timeout,
        )
      else:
        # Let JAX auto-detect TPU configuration
        jax.distributed.initialize(initialization_timeout=timeout)

    elif backend == "gpu":
      # For GPU, we need explicit coordinator settings
      if coordinator_address is None:
        coordinator_ip = os.getenv("JAX_COORDINATOR_IP", "localhost")
        coordinator_port = os.getenv("JAX_COORDINATOR_PORT", "1234")
        coordinator_address = f"{coordinator_ip}:{coordinator_port}"

      if num_processes is None:
        num_processes = int(os.getenv("NNODES", "1"))

      if process_id is None:
        process_id = int(os.getenv("NODE_RANK", "0"))

      jax.distributed.initialize(
          coordinator_address=coordinator_address,
          num_processes=num_processes,
          process_id=process_id,
          initialization_timeout=timeout,
      )

    elif backend == "cpu":
      # For CPU, similar to GPU
      if coordinator_address is None:
        coordinator_ip = os.getenv("JAX_COORDINATOR_IP", "localhost")
        coordinator_address = f"{coordinator_ip}:1234"

      if process_id is None:
        job_index = int(os.getenv("JOB_INDEX", "0"))
        job_completion_index = int(os.getenv("JOB_COMPLETION_INDEX", "0"))
        processes_in_job = int(os.getenv("PROCESSES_IN_JOB", "1"))
        process_id = job_index * processes_in_job + job_completion_index

      if num_processes is None:
        num_processes = int(os.getenv("JAX_PROCESS_COUNT", "1"))

      jax.distributed.initialize(
          coordinator_address=coordinator_address,
          process_id=process_id,
          num_processes=num_processes,
          initialization_timeout=timeout,
      )

    max_logging.log("JAX distributed system initialized successfully!")
    max_logging.log(f"JAX process index: {jax.process_index()}")
    max_logging.log(f"JAX process count: {jax.process_count()}")
    max_logging.log(f"JAX local devices: {jax.local_devices()}")
    max_logging.log(f"JAX global device count: {jax.device_count()}")

    # Update metadata tracking
    _update_metadata(
        backend=backend,
        coordinator_address=coordinator_address,
        num_processes=num_processes or jax.process_count(),
        process_id=process_id if process_id is not None else jax.process_index(),
        program_name=program_name,
    )

    return True

  except Exception as e:
    max_logging.log(f"Error initializing JAX distributed system: {e}")
    raise


def shutdown_jax_distributed(clear_caches=True, force_gc=True):
  """Release accelerator resources (TPU/GPU/CPU) and shutdown JAX distributed system.

  This function attempts to cleanly shutdown the JAX distributed system and
  release resources. It also releases the inter-process lock so other programs
  can use the resources.

  Args:
    clear_caches: If True, clear JAX compilation caches. Default is True.
    force_gc: If True, force garbage collection. Default is True.

  Returns:
    bool: True if shutdown was successful, False if not initialized.

  Example:
    >>> initialize_jax_for_backend()
    >>> # Your training code here
    >>> shutdown_jax_distributed()
    >>> # Resources are now released and can be used by another process
  """
  if not jax.distributed.is_initialized():
    max_logging.log("JAX distributed system is not initialized, nothing to release.")
    return False

  try:
    max_logging.log("Releasing resources and shutting down JAX distributed system...")

    # Clear JAX compilation caches
    if clear_caches:
      max_logging.log("Clearing JAX compilation caches...")
      jax.clear_caches()

    # Shutdown the distributed system
    max_logging.log("Shutting down JAX distributed system...")
    jax.distributed.shutdown()

    # Force garbage collection to release memory
    if force_gc:
      max_logging.log("Running garbage collection...")
      gc.collect()

    # Clear metadata tracking and release inter-process lock
    _clear_metadata()

    max_logging.log("Resources released successfully!")
    return True

  except Exception as e:
    max_logging.log(f"Error releasing resources: {e}")
    # Even if there's an error, clear metadata
    _clear_metadata()
    return False


def reset_jax_system(timeout=300):
  """Reset JAX system by releasing and reinitializing.

  This is useful when you need to completely reset the JAX state between runs.

  Args:
    timeout: Initialization timeout in seconds. Default is 300.

  Returns:
    bool: True if reset was successful.

  Example:
    >>> reset_jax_system()
    >>> # JAX is now in a fresh state
  """
  max_logging.log("Resetting JAX system...")
  shutdown_jax_distributed()
  return initialize_jax_for_backend(timeout=timeout)


def maybe_shutdown_jax_distributed_system(clear_caches=True, force_gc=True):
  """Conditionally shutdown JAX distributed system if it's initialized.

  This is the counterpart to maybe_initialize_jax_distributed_system.
  It safely shuts down the system only if it's currently initialized.

  Args:
    clear_caches: If True, clear JAX compilation caches. Default is True.
    force_gc: If True, force garbage collection. Default is True.

  Returns:
    bool: True if shutdown was performed, False if system wasn't initialized.

  Example:
    >>> maybe_shutdown_jax_distributed_system()
    >>> # System is now cleanly shutdown if it was initialized
  """
  if not jax.distributed.is_initialized():
    max_logging.log("JAX distributed system is not initialized, nothing to shutdown.")
    return False

  max_logging.log("JAX distributed system is initialized, shutting down...")
  return shutdown_jax_distributed(clear_caches=clear_caches, force_gc=force_gc)


def get_resource_status():
  """Get current accelerator (TPU/GPU/CPU) and JAX distributed system status.

  This includes both live JAX status and tracked metadata, as well as
  inter-process resource sharing information.

  Returns:
    dict: Dictionary containing resource status information.

  Example:
    >>> status = get_resource_status()
    >>> print(f"Initialized: {status['is_initialized']}")
    >>> print(f"Device count: {status['device_count']}")
    >>> print(f"Resource available: {status['resource_available']}")
  """
  metadata = get_initialization_metadata()

  status = {
      "is_initialized": jax.distributed.is_initialized(),
      "backend": jax.default_backend(),
      "device_count": jax.device_count() if jax.distributed.is_initialized() else 0,
      "local_device_count": len(jax.local_devices()) if jax.distributed.is_initialized() else 0,
      "process_index": jax.process_index() if jax.distributed.is_initialized() else None,
      "process_count": jax.process_count() if jax.distributed.is_initialized() else None,
      "tracked_metadata": metadata,
      "resource_available": is_resource_available(),
      "current_pid": os.getpid(),
  }

  if jax.distributed.is_initialized():
    status["local_devices"] = [str(d) for d in jax.local_devices()]
    status["devices"] = [str(d) for d in jax.devices()]

  # Add resource holder info if resources are in use
  holder_info = get_resource_holder_info()
  if holder_info:
    status["resource_holder"] = holder_info

  return status


def print_resource_status():
  """Print current resource and JAX distributed system status.

  Example:
    >>> print_resource_status()
    JAX Distributed System Status:
    ================================
    Initialized: True
    Backend: tpu
    Device count: 8
    ...
  """
  status = get_resource_status()
  metadata = status.get("tracked_metadata", {})

  print("\n" + "=" * 60)
  print("JAX DISTRIBUTED SYSTEM & RESOURCE STATUS")
  print("=" * 60)
  print(f"Initialized: {status['is_initialized']}")
  print(f"Backend: {status['backend']}")
  print(f"Device count: {status['device_count']}")
  print(f"Local device count: {status['local_device_count']}")
  print(f"Current PID: {status['current_pid']}")
  print(f"Resource available: {status['resource_available']}")

  if not status["resource_available"] and "resource_holder" in status:
    holder = status["resource_holder"]
    print("\nResource held by:")
    print(f"  Program: {holder.get('program_name', 'unknown')}")
    print(f"  PID: {holder.get('pid')}")
    print(f"  Host: {holder.get('hostname')}")
    print(f"  Backend: {holder.get('backend')}")
    if holder.get("command_line"):
      print(f"  Command: {holder.get('command_line')}")
    if holder.get("initialization_time"):
      uptime = time.time() - holder["initialization_time"]
      print(f"  Uptime: {uptime:.2f} seconds")

  if status["is_initialized"]:
    print(f"\nProcess index: {status['process_index']}")
    print(f"Process count: {status['process_count']}")
    if metadata.get("program_name"):
      print(f"Program name: {metadata['program_name']}")
    print(f"\nLocal devices: {status['local_devices']}")

    # Print tracked metadata
    if metadata.get("initialization_time"):
      uptime = time.time() - metadata["initialization_time"]
      print("\nTracked Metadata:")
      print(f"  Coordinator: {metadata.get('coordinator_address', 'auto-detected')}")
      print(f"  Uptime: {uptime:.2f} seconds")

  print("=" * 60)


def ensure_jax_initialized(timeout=300):
  """Ensure JAX system is initialized, initialize if not.

  Args:
    timeout: Initialization timeout in seconds. Default is 300.

  Returns:
    bool: True if system is initialized (either was already or just initialized).

  Example:
    >>> ensure_jax_initialized()
    >>> # Now you can safely use JAX
  """
  if jax.distributed.is_initialized():
    max_logging.log("JAX distributed system already initialized.")
    return True

  max_logging.log("JAX distributed system not initialized, initializing now...")
  return initialize_jax_for_backend(timeout=timeout)


def cleanup_and_release_resources():
  """Comprehensive cleanup and resource release.

  This function performs a thorough cleanup:
  1. Clears all JAX caches
  2. Shuts down distributed system
  3. Forces garbage collection
  4. Clears Python caches
  5. Releases inter-process locks

  This is the most aggressive cleanup option.

  Example:
    >>> cleanup_and_release_resources()
    >>> # All resources are released
  """
  max_logging.log("Performing comprehensive resource cleanup...")

  try:
    # Clear JAX caches
    if jax.distributed.is_initialized():
      max_logging.log("Clearing JAX compilation caches...")
      jax.clear_caches()

      # Shutdown distributed system
      max_logging.log("Shutting down JAX distributed system...")
      jax.distributed.shutdown()

    # Force multiple garbage collection passes
    max_logging.log("Running garbage collection...")
    for i in range(3):
      collected = gc.collect()
      max_logging.log(f"GC pass {i+1}: collected {collected} objects")

    # Clear metadata
    _clear_metadata()

    max_logging.log("Resource cleanup completed successfully!")

  except Exception as e:
    max_logging.log(f"Warning: Error during resource cleanup: {e}")
    max_logging.log("Continuing anyway...")
    _clear_metadata()


def initialize_for_config(raw_keys, program_name=None):
  """Initialize JAX distributed system based on MaxText config.

  This is the preferred way to initialize JAX for MaxText applications.

  Args:
    raw_keys: MaxText configuration dictionary.
    program_name: Optional program name for tracking. Uses run_name from config if not provided.

  Returns:
    bool: True if initialization was performed.

  Example:
    >>> config = pyconfig.initialize(argv, auto_init_jax=False)
    >>> resource_manager.initialize_for_config(config.get_raw_keys(), program_name="my_training")
  """
  # Get program name from config if not provided
  if program_name is None:
    program_name = raw_keys.get("run_name", "unknown")

  # Use the main initialization function
  maybe_initialize_jax_distributed_system(raw_keys, program_name=program_name)

  return jax.distributed.is_initialized()


# Backward compatibility aliases (keeping old TPU-specific names)
def initialize_tpu_system(coordinator_address=None, num_processes=None, process_id=None, timeout=300):
  """Backward compatibility alias for initialize_jax_for_backend."""
  return initialize_jax_for_backend(coordinator_address, num_processes, process_id, timeout)


def release_tpu_system(clear_caches=True, force_gc=True):
  """Backward compatibility alias for shutdown_jax_distributed."""
  return shutdown_jax_distributed(clear_caches, force_gc)


def reset_tpu_system(timeout=300):
  """Backward compatibility alias for reset_jax_system."""
  return reset_jax_system(timeout)


def get_tpu_status():
  """Backward compatibility alias for get_resource_status."""
  return get_resource_status()


def print_tpu_status():
  """Backward compatibility alias for print_resource_status."""
  return print_resource_status()


def ensure_tpu_initialized(timeout=300):
  """Backward compatibility alias for ensure_jax_initialized."""
  return ensure_jax_initialized(timeout)


def cleanup_and_release_tpu():
  """Backward compatibility alias for cleanup_and_release_resources."""
  return cleanup_and_release_resources()


# New clearer aliases (recommended for new code)
def initialize_jax_distributed(coordinator_address=None, num_processes=None, process_id=None, timeout=300):
  """Alias for initialize_jax_for_backend with clearer name.

  Initialize JAX distributed system for any accelerator (TPU/GPU/CPU).
  """
  return initialize_jax_for_backend(coordinator_address, num_processes, process_id, timeout)


def get_jax_status():
  """Alias for get_resource_status with clearer name.

  Get current JAX distributed system status for any accelerator.
  """
  return get_resource_status()


def print_jax_status():
  """Alias for print_resource_status with clearer name.

  Print current JAX distributed system status for any accelerator.
  """
  return print_resource_status()

