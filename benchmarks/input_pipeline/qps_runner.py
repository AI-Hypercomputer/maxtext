# Copyright 2023–2026 Google LLC
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

"""Runner for the MaxText PyGrain + ArrayRecord benchmark (measured GCS QPS, bytes, step time, sample order).

Each simulated host is a separate spawned OS process. Hosts do not share C++
GCS clients, Grain worker pools, or trace logs. Each host builds the real
MaxText Grain pipeline (`get_datasets` + `pretrain_preprocessing_pipeline`:
ArrayRecord global shuffle, tokenization, `FirstFit` packing, `mp_prefetch`
workers) and records Unix-time phase boundaries: `init`, `first_batch`,
`warmup`, and `steady`. Every reported request count, byte count, and QPS is
measured:

- `--backend=gcs` (live `gs://`): `google-cloud-cpp` client tracing. See
  `gcs_trace.client_trace_env`. The trace goes to one file per process
  (host process and Grain workers), and parsing yields every transport-level
  read with its byte range, status (`200`/`206`/`429`/`503`/cancelled), and
  timestamp. Optionally cross-checked against Cloud Monitoring
  `storage.googleapis.com/api/request_count`.
- `--backend=emulator` (hermetic): one `InstrumentedGCSServer` per host records
  every request it serves. `--emulator_client_trace` also enables the client
  tracing above, so the trace parser can be checked against the server's
  ground truth.
- `--backend=posix` (GCSFuse mount or local disk): `/proc` I/O counters, NIC RX
  bytes, and, optionally, GCSFuse Prometheus `gcs_request_count` and Cloud
  Monitoring.

`--measure_mode=paired` (the default) runs an untraced timing pass and a traced
QPS pass with identical configs. Tracing stderr overhead therefore never
reaches step-time percentiles or `Examples/s`. Each host returns a SHA-256
digest over the exact batches it yielded, so every sweep point can prove
bitwise sample-order parity against its baseline.

Sweep flags take lists (numbers separated by `,`, strings by `|`). The Cartesian
product is run, and a JSON report plus Markdown tables are written.
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import hashlib
import itertools
import json
import multiprocessing
import multiprocessing.connection
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Sequence

from benchmarks.input_pipeline import gcs_emulator
from benchmarks.input_pipeline import gcs_trace
from benchmarks.input_pipeline import generate_synthetic_arrayrecords

EMULATOR_BUCKET = "qps-bench"
STARTUP_COUNTING_MODES = ("every_host", "sharded", "manifest")
DATA_SHARDING_MODES = ("per_host", "per_replica")


@dataclasses.dataclass(frozen=True)
class RunConfig:
  """One sweep point. All simulated hosts in a run share it."""

  modality: str = "text"
  backend: str = "emulator"  # emulator, gcs, posix.
  train_files: str = ""  # gs:// or local glob; emulator runs fill this in.
  host_count: int = 1
  hosts_per_replica: int = 8
  data_sharding: str = "per_host"
  startup_counting: str = "every_host"
  manifest_path: str = ""
  grain_worker_count: int = 0
  grain_per_worker_buffer_size: int = 1
  grain_prefetch_buffer_size: int = 500
  grain_num_threads: int = 16
  grain_reader_options: str = ""
  startup_jitter_max_s: float = 0.0
  per_host_batch_size: int = 4
  max_target_length: int = 2048
  warmup_steps: int = 2
  steps: int = 10
  target_step_time_s: float = 0.0
  lockstep: bool = True
  data_shuffle_seed: int = 0
  tokenizer_path: str = ""
  per_replica_full_batch: bool = True

  @property
  def host_batch_size(self) -> int:
    """Examples each host loads per step.

    With `per_replica` + `per_replica_full_batch`, every host loads its whole
    replica's batch (`hosts_per_replica` x `per_host_batch_size`), so the global
    batch of unique examples equals the `per_host` run's and only the duplicate
    reads differ (BP-8).
    """
    if self.data_sharding == "per_replica" and self.per_replica_full_batch:
      return self.per_host_batch_size * self.hosts_per_replica
    return self.per_host_batch_size

  def label(self) -> str:
    """Short human-readable run id for tables (excludes the storage path)."""
    parts = [
        f"H{self.host_count}",
        f"W{self.grain_worker_count}",
        f"pf{self.grain_prefetch_buffer_size}",
        self.startup_counting,
        self.data_sharding,
    ]
    if self.data_sharding == "per_replica":
      parts.append(f"R{self.hosts_per_replica}")
      parts.append(f"B{self.host_batch_size}")
    if self.startup_jitter_max_s:
      parts.append(f"jit{self.startup_jitter_max_s:g}s")
    if self.grain_reader_options:
      parts.append(self.grain_reader_options.replace(",", "+"))
    parts.append(self.backend)
    return "/".join(parts)

  def parity_key(self) -> tuple:
    """Runs with equal keys must yield bitwise-identical per-host sample streams.

    `grain_worker_count` is included because Grain packs within each
    `mp_prefetch` worker, so different worker counts legitimately yield different
    (still globally shuffled) packed batches. Prefetch buffer size, startup
    counting mode, reader options, jitter, and storage tier must not change order.

    Returns:
      Hashable parity group key.
    """
    return (
        self.modality,
        self.host_count,
        self.data_sharding,
        self.hosts_per_replica if self.data_sharding == "per_replica" else 0,
        self.grain_worker_count,
        self.host_batch_size,
        self.max_target_length,
        self.warmup_steps,
        self.steps,
        self.data_shuffle_seed,
    )


@dataclasses.dataclass(frozen=True)
class HostTask:
  run: RunConfig
  host_index: int
  traced: bool
  trace_dir: str = ""
  emulator_endpoint: str = ""


def dataloading_shard(run: RunConfig, host_index: int) -> tuple[int, int]:
  """Returns (dataloading_host_index, dataloading_host_count) for a host.

  `per_host` is MaxText's default: every host reads a distinct 1/H slice of the
  globally shuffled stream. `per_replica` emulates loading data once per model
  replica of `hosts_per_replica` hosts, where every host in a replica reads the
  same slice. That duplicates reads by `hosts_per_replica`x (BP-8 anti-pattern).

  Args:
    run: Sweep point.
    host_index: Simulated host index.

  Returns:
    (dataloading_host_index, dataloading_host_count).
  """
  if run.data_sharding == "per_host":
    return host_index, run.host_count
  if run.host_count % run.hosts_per_replica:
    raise ValueError(f"host_count={run.host_count} must be divisible by hosts_per_replica={run.hosts_per_replica}")
  return host_index // run.hosts_per_replica, run.host_count // run.hosts_per_replica


def unique_examples_per_step(run: RunConfig) -> int:
  """Distinct examples the cluster loads per step: distinct data slices x per-host batch."""
  return dataloading_shard(run, 0)[1] * run.host_batch_size


def startup_jitter_s(max_s: float, seed: int, host_index: int) -> float:
  """Deterministic per-host startup delay in [0, max_s) (BP-3).

  Same schedule as MaxText `grain_startup_jitter_max_seconds`, so harness waves
  match what a training job with that flag does.

  Args:
    max_s: Maximum delay in seconds. 0 disables jitter.
    seed: `data_shuffle_seed`.
    host_index: Simulated host index (`jax.process_index()` in training).

  Returns:
    Delay in seconds.
  """
  if max_s <= 0:
    return 0.0
  digest = hashlib.sha256(f"maxtext_grain_startup_jitter:{seed}:{host_index}".encode("utf-8")).digest()
  return max_s * (int.from_bytes(digest[:8], "big") / 2**64)


def parse_reader_options(spec: str) -> dict[str, str] | None:
  """Parses `key:value[,key:value]` into an ArrayRecord `reader_options` dict."""
  if not spec:
    return None
  out = {}
  for item in spec.split(","):
    key, sep, value = item.partition(":")
    if not sep or not key.strip():
      raise ValueError(f"Malformed reader option {item!r} in {spec!r}; expected key:value.")
    out[key.strip()] = value.strip()
  return out


def _glob_match(path: str, pattern: str) -> bool:
  """Glob match where `*`, `?`, and `[]` never cross `/` (like a GCS or filesystem glob)."""
  parts, pats = path.split("/"), pattern.split("/")
  return len(parts) == len(pats) and all(fnmatch.fnmatchcase(a, b) for a, b in zip(parts, pats))


def load_manifest_file_instructions(manifest_path: str, pattern: str):
  """Builds MaxText ArrayRecord `FileInstruction`s from a pre-indexed manifest (BP-1).

  The manifest is a JSON object of `{shard_path: num_records}`. Shards matching
  `pattern` are selected from its keys, so no GCS LIST and no shard open are
  needed.

  Args:
    manifest_path: Local or `gs://` manifest path.
    pattern: The `grain_train_files` glob.

  Returns:
    `{pattern: (FileInstruction, ...)}` in sorted shard order.
  """
  from maxtext.input_pipeline import grain_data_processing  # pylint: disable=import-outside-toplevel

  if manifest_path.startswith("gs://"):
    from maxtext.utils import gcs_utils  # pylint: disable=import-outside-toplevel

    raw = gcs_utils.read_bytes_from_gcs(manifest_path)
    if raw is None:
      raise FileNotFoundError(f"Could not read manifest {manifest_path}")
    manifest = json.loads(raw)
  else:
    with open(manifest_path, "r", encoding="utf-8") as f:
      manifest = json.load(f)
  files = sorted(k for k in manifest if _glob_match(k, pattern))
  if not files:
    raise ValueError(f"No manifest entries in {manifest_path} match {pattern}")
  fi_cls = grain_data_processing._FileInstruction  # pylint: disable=protected-access
  return {pattern: tuple(fi_cls(f, 0, int(manifest[f]), int(manifest[f])) for f in files)}


class _WorkerProcessSetup:
  """Identity map that configures each Grain `mp_prefetch` worker when it is unpickled.

  Grain pickles the pipeline into worker processes, and unpickling this object
  runs `_setup_worker_process` there before any record is read. The setup sends
  the worker's stderr (`std::clog`) to its own `host<H>.pid<P>.clog`, so traces
  from different processes never interleave. It changes neither element order
  nor content.
  """

  def __init__(self, trace_dir: str, host_index: int):
    self.trace_dir = trace_dir
    self.host_index = host_index

  def __reduce__(self):
    return (_setup_worker_process, (self.trace_dir, self.host_index))

  def __call__(self, element):
    return element


def _redirect_stderr(trace_dir: str, host_index: int) -> str:
  path = os.path.join(trace_dir, f"host{host_index}.pid{os.getpid()}.clog")
  fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
  sys.stderr.flush()
  os.dup2(fd, 2)
  os.close(fd)
  return path


def _setup_worker_process(trace_dir: str, host_index: int) -> _WorkerProcessSetup:
  if trace_dir and not os.path.exists(os.path.join(trace_dir, f"host{host_index}.pid{os.getpid()}.clog")):
    _redirect_stderr(trace_dir, host_index)
  return _WorkerProcessSetup(trace_dir, host_index)


def _default_tokenizer(name: str) -> str:
  """Resolves a bundled tokenizer relative to MaxText's assets directory."""
  from maxtext.utils import globals as mt_globals  # pylint: disable=import-outside-toplevel

  for root in (mt_globals.MAXTEXT_ASSETS_ROOT, os.path.join(mt_globals.MAXTEXT_PKG_DIR, "assets")):
    path = os.path.join(root, "tokenizers", name)
    if os.path.exists(path):
      return path
  raise FileNotFoundError(f"Tokenizer {name} not found; pass --tokenizer_path.")


def _build_config(run: RunConfig):
  """Builds a MaxText `pyconfig` for the benchmark's pre-training pipeline."""
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from maxtext.utils import globals as mt_globals  # pylint: disable=import-outside-toplevel

  if run.modality == "text":
    tok = run.tokenizer_path or _default_tokenizer("tokenizer.default")
    argv = [
        "",
        os.path.join(mt_globals.MAXTEXT_CONFIGS_DIR, "base.yml"),
        "run_name=qps_bench_text",
        'train_data_columns=["text"]',
        "packing=True",
        "grain_packing_type=first_fit",
    ]
  elif run.modality == "multimodal":
    tok = run.tokenizer_path or _default_tokenizer("tokenizer.gemma3")
    argv = [
        "",
        os.path.join(mt_globals.MAXTEXT_CONFIGS_DIR, "post_train", "sft-vision-chartqa.yml"),
        "run_name=qps_bench_multimodal",
        "model_name=gemma3-4b",
        'train_data_columns=["question","answer"]',
        "train_image_column=image",
        f"max_prefill_predict_length={run.max_target_length // 2}",
        "use_tunix_gradient_accumulation=False",
    ]
  else:
    raise ValueError(f"Unsupported modality: {run.modality}")
  argv += [
      f"per_device_batch_size={run.host_batch_size}",
      f"max_target_length={run.max_target_length}",
      "dataset_type=grain",
      "grain_file_type=arrayrecord",
      f"grain_train_files={run.train_files}",
      "tokenize_train_data=True",
      f"tokenizer_path={tok}",
      "num_epoch=1000",
      "enable_data_shuffling=True",
      f"data_shuffle_seed={run.data_shuffle_seed}",
      f"grain_worker_count={run.grain_worker_count}",
      f"grain_per_worker_buffer_size={run.grain_per_worker_buffer_size}",
      f"grain_num_threads={run.grain_num_threads}",
      f"grain_prefetch_buffer_size={run.grain_prefetch_buffer_size}",
      "enable_checkpointing=False",
      "skip_jax_distributed_system=True",
      "hardware=cpu",
  ]
  return pyconfig.initialize(argv)


def _build_host_dataset(run: RunConfig, config, host_index: int, conn):
  """Builds this host's ArrayRecord dataset with the requested startup counting mode."""
  import inspect  # pylint: disable=import-outside-toplevel
  from maxtext.input_pipeline import grain_data_processing as gdp  # pylint: disable=import-outside-toplevel

  shard_index, shard_count = dataloading_shard(run, host_index)
  kwargs = {
      "shuffle": config.enable_data_shuffling,
      "shuffle_seed": config.data_shuffle_seed,
      "shuffle_buffer_size": config.grain_shuffle_buffer_size,
      "num_epoch": config.num_epoch,
      "grain_worker_count": config.grain_worker_count,
      "grain_num_threads": config.grain_num_threads,
      "grain_prefetch_buffer_size": config.grain_prefetch_buffer_size,
      "grain_data_source_max_workers": config.grain_data_source_max_workers,
      "grain_index_storage_option": getattr(config, "grain_index_storage_option", None),
      "split": "train",
  }
  reader_options = parse_reader_options(run.grain_reader_options)
  if reader_options:
    params = inspect.signature(gdp.get_datasets).parameters
    if "grain_reader_options" in params:
      kwargs["grain_reader_options"] = run.grain_reader_options
    elif "reader_options" in params:
      kwargs["reader_options"] = reader_options
    else:
      raise ValueError("--grain_reader_options requires MaxText get_datasets reader-option support (BP-2).")

  def get_ds_fn(**kw):
    return gdp.get_datasets(run.train_files, "arrayrecord", **kwargs, **kw)

  if run.startup_counting == "every_host":
    return get_ds_fn(dataloading_host_index=shard_index, dataloading_host_count=shard_count)
  if run.startup_counting == "manifest":
    instructions = load_manifest_file_instructions(run.manifest_path, run.train_files)
    return get_ds_fn(
        dataloading_host_index=shard_index, dataloading_host_count=shard_count, file_instructions=instructions
    )
  if run.startup_counting == "sharded":
    loader_kwargs = {"max_workers": config.grain_data_source_max_workers}
    loader_cls = gdp._ColocatedArrayRecordDatasetLoader  # pylint: disable=protected-access
    if reader_options and "reader_options" in inspect.signature(loader_cls).parameters:
      loader_kwargs["reader_options"] = reader_options
    loader = loader_cls(get_ds_fn, run.train_files, **loader_kwargs)
    num_files, _, digest, local_counts = loader.count_local_shards(host_index, run.host_count)
    # The parent process stands in for the cross-host integer allgather.
    conn.send(("shard_counts", {"num_files": num_files, "digest": digest, "counts": list(local_counts)}))
    kind, payload = conn.recv()
    if kind != "all_shard_counts":
      raise RuntimeError(f"Unexpected exchange message: {kind}: {payload}")
    loader.set_shard_counts(payload)
    return loader(dataloading_host_index=shard_index, dataloading_host_count=shard_count)
  raise ValueError(f"Unknown startup_counting={run.startup_counting}")


def _packed_docs(batch, modality: str) -> int:
  import numpy as np  # pylint: disable=import-outside-toplevel

  if modality == "text" and "inputs_segmentation" in batch:
    seg = np.asarray(batch["inputs_segmentation"])
    return int(np.sum(np.max(seg, axis=-1)))
  return int(np.asarray(batch["inputs"]).shape[0])


def _bounded_close(it, timeout_s: float = 10.0) -> None:
  """Closes a Grain iterator, giving up after `timeout_s`.

  Closing an `mp_prefetch` iterator can block for minutes when workers sit on a
  full output queue that nobody drains anymore. Measurements are already sent by
  then, so after the timeout the host exits and `run_hosts` reaps its process
  group.

  Args:
    it: Iterator to close (ignored if it has no `close`).
    timeout_s: Maximum time to wait for `close()`.
  """
  if not hasattr(it, "close"):
    return
  closer = threading.Thread(target=it.close, daemon=True)
  closer.start()
  closer.join(timeout_s)
  if closer.is_alive():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # pylint: disable=protected-access


def host_main(task: HostTask, conn, barrier) -> None:
  """Entry point of one simulated host process."""
  run = task.run
  os.environ.setdefault("JAX_PLATFORMS", "cpu")
  for key in gcs_trace.CLIENT_TRACE_ENV_KEYS:
    os.environ.pop(key, None)
  if task.emulator_endpoint:
    os.environ.update(gcs_emulator.emulator_env(task.emulator_endpoint))
  if task.traced and task.trace_dir:
    os.environ.update(gcs_trace.client_trace_env())
    _redirect_stderr(task.trace_dir, task.host_index)
  try:
    from maxtext.input_pipeline import grain_data_processing as gdp  # pylint: disable=import-outside-toplevel

    config = _build_config(run)
    digest = gcs_trace.RecordOrderDigest()
    proc_before = gcs_trace.read_process_tree_stats()
    barrier.wait()  # Simultaneous start across hosts (BP-3 baseline).
    jitter = startup_jitter_s(run.startup_jitter_max_s, run.data_shuffle_seed, task.host_index)
    if jitter:
      time.sleep(jitter)

    init_start = time.time()
    ds = _build_host_dataset(run, config, task.host_index, conn)
    ds = ds.map(_WorkerProcessSetup(task.trace_dir if task.traced else "", task.host_index))
    pipeline_kwargs = {
        "config": config,
        "data_columns": config.train_data_columns,
        "tokenize": config.tokenize_train_data,
        "grain_worker_count": config.grain_worker_count,
        "grain_per_worker_buffer_size": config.grain_per_worker_buffer_size,
    }
    if config.use_sft and config.use_multimodal:
      pipeline_kwargs["image_column"] = config.train_image_column
    it = iter(gdp._get_pipeline_fn(config)(dataset=ds, **pipeline_kwargs))  # pylint: disable=protected-access
    init_end = time.time()

    batch = next(it)
    digest.update(batch)
    docs_first = _packed_docs(batch, run.modality)
    first_batch_end = time.time()

    def _step() -> tuple[float, int]:
      if run.lockstep:
        barrier.wait()
      t0 = time.perf_counter()
      b = next(it)
      dt = time.perf_counter() - t0
      digest.update(b)
      if run.target_step_time_s > dt:
        time.sleep(run.target_step_time_s - dt)
      return dt, _packed_docs(b, run.modality)

    for _ in range(run.warmup_steps):
      _step()
    warmup_end = time.time()
    step_starts, latencies, docs = [], [], 0
    for _ in range(run.steps):
      step_starts.append(time.time())
      dt, n = _step()
      latencies.append(dt)
      docs += n
    steady_end = time.time()
    proc_after = gcs_trace.read_process_tree_stats()
    conn.send(
        (
            "result",
            {
                "host": task.host_index,
                "pid": os.getpid(),
                "jitter_s": jitter,
                "marks": {
                    "init_start": init_start,
                    "init_end": init_end,
                    "first_batch_end": first_batch_end,
                    "warmup_end": warmup_end,
                    "steady_end": steady_end,
                    "step_starts": step_starts,
                },
                "step_latencies_s": latencies,
                "packed_docs_first_batch": docs_first,
                "packed_docs_steady": docs,
                "digest": digest.hexdigest(),
                "digest_batches": digest.batches,
                "process_tree_after": proc_after,
                "process_tree_io_delta": {k: proc_after[k] - proc_before[k] for k in ("rchar", "syscr", "read_bytes")},
            },
        )
    )
    conn.close()  # The result is delivered; the parent stops waiting on this host.
    _bounded_close(it)
  except BaseException as e:  # pylint: disable=broad-exception-caught
    try:
      barrier.abort()
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    if not conn.closed:
      conn.send(("error", f"host {task.host_index}: {type(e).__name__}: {e}\n{traceback.format_exc()}"))
  finally:
    sys.stderr.flush()
    conn.close()


def _host_entry(target, task, conn, barrier) -> None:
  """Runs `target` as the leader of a new session so `run_hosts` can reap its whole process tree."""
  try:
    os.setsid()
  except OSError:
    pass
  target(task, conn, barrier)


def kill_process_group(pgid: int, grace_s: float = 3.0) -> None:
  """Sends SIGTERM, then SIGKILL after `grace_s`, to every process in group `pgid`."""

  def _signal(sig) -> bool:
    try:
      os.killpg(pgid, sig)
      return True
    except (ProcessLookupError, PermissionError):
      return False

  if not _signal(signal.SIGTERM):
    return
  deadline = time.time() + grace_s
  while time.time() < deadline:
    try:
      os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
      return
    time.sleep(0.05)
  _signal(signal.SIGKILL)


def mp_context():
  """Returns the spawn context used for simulated host processes."""
  return multiprocessing.get_context("spawn")


def combine_shard_counts(messages: Sequence[dict[str, Any]]) -> list[int]:
  """Reassembles per-host `count_local_shards` slices (`files[h::H]`) into one list."""
  host_count = len(messages)
  num_files = {m["num_files"] for m in messages}
  digests = {m["digest"] for m in messages}
  if len(num_files) != 1 or len(digests) != 1:
    raise RuntimeError(f"Shard listing mismatch across hosts: num_files={num_files} digests={digests}")
  counts = [0] * num_files.pop()
  for h, m in enumerate(messages):
    expected = len(range(h, len(counts), host_count))
    if len(m["counts"]) != expected:
      raise RuntimeError(f"Host {h} returned {len(m['counts'])} counts, expected {expected}")
    counts[h::host_count] = m["counts"]
  return counts


def run_hosts(target, tasks: Sequence[Any], timeout_s: float = 3600.0) -> list[dict[str, Any]]:
  """Runs `target(task, conn, barrier)` in one spawned process per task.

  The parent services `shard_counts` exchange requests (an allgather stand-in
  for sharded startup counting) and collects one `result` per host. Any host
  error aborts the shared barrier, so no host hangs.

  Args:
    target: Picklable top-level function.
    tasks: One task per simulated host. The index is the host index.
    timeout_s: Overall deadline.

  Returns:
    Per-host result dicts in host order.

  Raises:
    RuntimeError: If any host fails or the deadline expires.
  """
  ctx = mp_context()
  n = len(tasks)
  barrier = ctx.Barrier(n)
  pipes = [ctx.Pipe(duplex=True) for _ in range(n)]
  procs = [ctx.Process(target=_host_entry, args=(target, tasks[h], pipes[h][1], barrier), daemon=False) for h in range(n)]
  for p in procs:
    p.start()
  for _, child in pipes:
    child.close()
  results: dict[int, dict[str, Any]] = {}
  exchange: dict[int, dict[str, Any]] = {}
  errors: list[str] = []
  deadline = time.time() + timeout_s
  open_conns = {h: pipes[h][0] for h in range(n)}
  drain_deadline = None  # After the first error, briefly collect the other hosts' errors to find the root cause.
  try:
    while open_conns and time.time() < (drain_deadline or deadline):
      ready = multiprocessing.connection.wait(list(open_conns.values()), timeout=1.0)
      for conn in ready:
        h = next(k for k, v in open_conns.items() if v is conn)
        try:
          kind, payload = conn.recv()
        except EOFError:
          open_conns.pop(h)
          if h not in results:
            errors.append(f"host {h}: exited without a result (exitcode={procs[h].exitcode})")
          continue
        if kind == "shard_counts":
          exchange[h] = payload
          if len(exchange) == n:
            try:
              combined = combine_shard_counts([exchange[i] for i in range(n)])
              for i in range(n):
                pipes[i][0].send(("all_shard_counts", combined))
            except RuntimeError as e:
              for i in range(n):
                pipes[i][0].send(("error", str(e)))
              errors.append(str(e))
        elif kind == "result":
          results[h] = payload
          open_conns.pop(h)  # Do not wait for the host's (possibly slow) iterator teardown.
        elif kind == "error":
          errors.append(payload)
          barrier.abort()
      if errors and drain_deadline is None:
        drain_deadline = min(deadline, time.time() + 15.0)
    if open_conns and not errors and time.time() >= deadline:
      errors.append(f"timed out after {timeout_s}s waiting for hosts {sorted(open_conns)}")
  finally:
    for p in procs:
      p.join(timeout=15 if not errors else 5)
      if p.is_alive():
        p.terminate()
        p.join(timeout=10)
    # Reap whatever is left in each host's process group (Grain workers orphaned by an OOM-killed host).
    for p in procs:
      if p.pid is not None:
        kill_process_group(p.pid)
  if errors:
    # Root causes first; BrokenBarrierError and missing results are knock-on effects of the abort.
    errors.sort(key=lambda e: "BrokenBarrierError" in e or "exited without a result" in e)
    raise RuntimeError("Simulated host failure(s):\n" + "\n".join(errors))
  return [results[h] for h in range(n)]


def _percentile(values: Sequence[float], q: float) -> float:
  if not values:
    return 0.0
  s = sorted(values)
  k = (len(s) - 1) * q / 100.0
  lo, hi = int(k), min(int(k) + 1, len(s) - 1)
  return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize_timing(run: RunConfig, host_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
  """Step-time and throughput summary from measured host results."""
  per_step_max = [max(r["step_latencies_s"][s] for r in host_results) for s in range(run.steps)]
  all_lat = [x for r in host_results for x in r["step_latencies_s"]]
  init = [r["marks"]["init_end"] - r["marks"]["init_start"] for r in host_results]
  first = [r["marks"]["first_batch_end"] - r["marks"]["init_end"] for r in host_results]
  steady_wall = [r["marks"]["steady_end"] - r["marks"]["warmup_end"] for r in host_results]
  data_wait = [sum(r["step_latencies_s"]) for r in host_results]
  return {
      "init_s_max": round(max(init), 4),
      "init_s_mean": round(sum(init) / len(init), 4),
      "first_batch_s_max": round(max(first), 4),
      "time_to_first_batch_s_max": round(max(i + f for i, f in zip(init, first)), 4),
      "step_p50_s": round(_percentile(all_lat, 50), 5),
      "step_p90_s": round(_percentile(all_lat, 90), 5),
      "step_p99_s": round(_percentile(all_lat, 99), 5),
      "step_max_s": round(max(all_lat, default=0.0), 5),
      "cluster_step_p50_s": round(_percentile(per_step_max, 50), 5),
      "cluster_step_p99_s": round(_percentile(per_step_max, 99), 5),
      # Delivered rate over the steady wall clock (includes --target_step_time_s pacing and lockstep waits).
      "examples_per_s_per_host": round(
          sum(run.steps * run.host_batch_size / max(1e-9, w) for w in steady_wall) / len(steady_wall), 2
      ),
      # Pipeline capacity: examples per second of data-wait time only (meaningful when unpaced or data-bound).
      "data_wait_examples_per_s_per_host": round(
          sum(run.steps * run.host_batch_size / max(1e-9, w) for w in data_wait) / len(data_wait), 2
      ),
      "steady_wall_s_max": round(max(steady_wall), 4),
      "packed_docs_per_step_per_host": round(
          sum(r["packed_docs_steady"] for r in host_results) / max(1, run.steps * len(host_results)), 3
      ),
  }


def _digest_parity_within_run(run: RunConfig, host_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
  digests = [r["digest"] for r in host_results]
  groups: dict[int, set[str]] = {}
  for h, d in enumerate(digests):
    groups.setdefault(dataloading_shard(run, h)[0], set()).add(d)
  return {
      "per_host": digests,
      "distinct_streams": len(set(digests)),
      "hosts_sharing_a_slice_identical": all(len(v) == 1 for v in groups.values()),
  }


def _scrape(url: str) -> str:
  from urllib import request  # pylint: disable=import-outside-toplevel

  with request.urlopen(url, timeout=10) as resp:  # pylint: disable=consider-using-with
    return resp.read().decode("utf-8", errors="replace")


def make_trace_dir(trace_root: str = "") -> str:
  """Creates a fresh per-pass trace directory under `trace_root` (created if missing) or the system temp dir."""
  if trace_root:
    os.makedirs(trace_root, exist_ok=True)
  return tempfile.mkdtemp(prefix="qps_trace_", dir=trace_root or None)


def run_pass(
    run: RunConfig, client_traced: bool, measure_qps: bool, args: argparse.Namespace, emulator_root: str
) -> dict[str, Any]:
  """Runs all hosts once and returns measured timing, request, and system metrics.

  Args:
    run: Sweep point.
    client_traced: Enable `google-cloud-cpp` client tracing in every host process.
    measure_qps: Collect request-level metrics (emulator records, GCSFuse
      counters, Cloud Monitoring). False for an untraced timing-only pass.
    args: Parsed flags.
    emulator_root: Local directory served by the emulators.

  Returns:
    Pass results.
  """
  servers: list[gcs_emulator.InstrumentedGCSServer] = []
  trace_dir = ""
  if client_traced:
    trace_dir = make_trace_dir(args.trace_root)
  if run.backend == "emulator":
    for h in range(run.host_count):
      srv = gcs_emulator.InstrumentedGCSServer(
          emulator_root,
          EMULATOR_BUCKET,
          host_index=h,
          fail_every_n=args.emulator_fail_every_n,
          fail_status=args.emulator_fail_status,
      )
      srv.start()
      servers.append(srv)
  tasks = [
      HostTask(run, h, client_traced, trace_dir, servers[h].endpoint if servers else "") for h in range(run.host_count)
  ]
  fuse_before = (
      _scrape(args.gcsfuse_metrics_url) if measure_qps and args.gcsfuse_metrics_url and run.backend == "posix" else ""
  )
  net_before = gcs_trace.read_net_dev()
  wall_start = time.time()
  try:
    host_results = run_hosts(host_main, tasks, timeout_s=args.host_timeout_s)
  finally:
    for srv in servers:
      srv.stop()
  wall_end = time.time()
  net_after = gcs_trace.read_net_dev()
  marks = {r["host"]: gcs_trace.PhaseMarks(host=r["host"], **r["marks"]) for r in host_results}
  out: dict[str, Any] = {
      "traced": client_traced,
      "wall_start": wall_start,
      "wall_end": wall_end,
      "timing": summarize_timing(run, host_results),
      "hosts": [
          {k: r[k] for k in ("host", "pid", "jitter_s", "digest", "process_tree_after", "process_tree_io_delta")}
          for r in host_results
      ],
      "parity": _digest_parity_within_run(run, host_results),
      "nic": gcs_trace.net_dev_delta(net_before, net_after),
      "loopback": gcs_trace.net_dev_delta(
          {"lo": net_before.get("lo", {})} if "lo" in net_before else {},
          {"lo": net_after.get("lo", {})} if "lo" in net_after else {},
          include_loopback=True,
      ),
  }
  if servers and measure_qps:
    reqs = [r for s in servers for r in s.snapshot()]
    out["requests_emulator"] = gcs_trace.summarize_requests(reqs, marks)
  if trace_dir:
    parsed = gcs_trace.parse_trace_dir(trace_dir, mp_context=mp_context())
    out["requests_client_trace"] = gcs_trace.summarize_requests(parsed.requests, marks, parsed.http_bytes_timeline)
    out["client_trace_stats"] = {
        "trace_dir": trace_dir,
        "files": len([f for f in os.listdir(trace_dir) if f.endswith(".clog")]),
        "records": parsed.records,
        "http_requests_seen": parsed.http_requests_seen,
        "grpc_requests_seen": parsed.grpc_requests_seen,
        "unmatched_http_requests": parsed.unmatched_http_requests,
    }
    if not args.keep_traces:
      shutil.rmtree(trace_dir, ignore_errors=True)
  if fuse_before:
    after = gcs_trace.parse_prometheus_counter(_scrape(args.gcsfuse_metrics_url), args.gcsfuse_metric)
    before = gcs_trace.parse_prometheus_counter(fuse_before, args.gcsfuse_metric)
    out["gcsfuse_metrics_delta"] = {k: v - before.get(k, 0.0) for k, v in after.items()}
  if measure_qps and args.cloud_monitoring_project and run.train_files.startswith("gs://"):
    if args.cloud_monitoring_delay_s > 0:
      time.sleep(args.cloud_monitoring_delay_s)
    out["cloud_monitoring"] = gcs_trace.query_cloud_monitoring_request_count(
        args.cloud_monitoring_project, run.train_files[len("gs://") :].split("/", 1)[0], wall_start, wall_end
    )
  return out


def _split(value: str, sep: str) -> list[str]:
  return [s.strip() for s in value.split(sep)] if value != "" else [""]


def expand_sweep(args: argparse.Namespace) -> list[RunConfig]:
  """Expands list-valued flags into the Cartesian product of `RunConfig`s."""
  axes = {
      "train_files": _split(args.train_files, "|"),
      "host_count": [int(v) for v in _split(args.host_counts, ",")],
      "grain_worker_count": [int(v) for v in _split(args.grain_worker_count, ",")],
      "grain_prefetch_buffer_size": [int(v) for v in _split(args.grain_prefetch_buffer_size, ",")],
      "startup_counting": _split(args.startup_counting, ","),
      "startup_jitter_max_s": [float(v) for v in _split(args.startup_jitter_max_s, ",")],
      "data_sharding": _split(args.data_sharding, ","),
      "grain_reader_options": _split(args.grain_reader_options, "|"),
  }
  for mode in axes["startup_counting"]:
    if mode not in STARTUP_COUNTING_MODES:
      raise ValueError(f"--startup_counting must be in {STARTUP_COUNTING_MODES}, got {mode}")
  for mode in axes["data_sharding"]:
    if mode not in DATA_SHARDING_MODES:
      raise ValueError(f"--data_sharding must be in {DATA_SHARDING_MODES}, got {mode}")
  runs = []
  for combo in itertools.product(*axes.values()):
    values = dict(zip(axes.keys(), combo))
    runs.append(
        RunConfig(
            modality=args.modality,
            backend=args.backend,
            hosts_per_replica=args.hosts_per_replica,
            manifest_path=args.manifest_path,
            grain_per_worker_buffer_size=args.grain_per_worker_buffer_size,
            grain_num_threads=args.grain_num_threads,
            per_host_batch_size=args.per_host_batch_size,
            per_replica_full_batch=args.per_replica_full_batch,
            max_target_length=args.max_target_length,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
            target_step_time_s=args.target_step_time_s,
            lockstep=args.lockstep,
            data_shuffle_seed=args.data_shuffle_seed,
            tokenizer_path=args.tokenizer_path,
            **values,
        )
    )
  return runs


def check_parity(results: list[dict[str, Any]]) -> None:
  """Annotates each run with sample-order parity against the first run in its parity group."""
  refs: dict[str, dict[str, Any]] = {}
  for res in results:
    if "error" in res:
      continue
    key = json.dumps(res["parity_key"])
    digests = res["digests"]
    if key not in refs:
      refs[key] = res
      res["parity_vs_baseline"] = {"baseline": res["label"], "status": "BASELINE"}
      continue
    ref = refs[key]
    ok = digests == ref["digests"]
    res["parity_vs_baseline"] = {
        "baseline": ref["label"],
        "status": "MATCH" if ok else "MISMATCH",
        "mismatched_hosts": [h for h, (a, b) in enumerate(zip(digests, ref["digests"])) if a != b],
    }


def _fmt(v: Any, nd: int = 3) -> str:
  if v is None:
    return "-"
  if isinstance(v, float):
    return f"{v:.{nd}f}"
  return str(v)


def markdown_report(results: list[dict[str, Any]]) -> str:
  """Renders timing, QPS, and parity tables for the guide and reports."""
  lines = ["## Step time and throughput (untraced timing pass unless marked `[traced]`)", ""]
  lines += [
      "| Run | Init s (max) | 1st batch s (max) | Step p50 s | Step p90 s | Step p99 s | Examples/s/host (wall) |",
      "|---|---|---|---|---|---|---|",
  ]
  for r in results:
    if "error" in r:
      lines.append(f"| {r['label']} | ERROR | | | | | |")
      continue
    t = r["timing"]
    traced = " [traced]" if r.get("timing_from_traced_pass") else ""
    lines.append(
        f"| {r['label']}{traced} | {_fmt(t['init_s_max'])} | {_fmt(t['first_batch_s_max'])} |"
        f" {_fmt(t['step_p50_s'], 4)} | {_fmt(t['step_p90_s'], 4)} | {_fmt(t['step_p99_s'], 4)} |"
        f" {_fmt(t['examples_per_s_per_host'], 1)} |"
    )
  lines += ["", "## Measured GCS requests (traced pass)", ""]
  lines += [
      "| Run | Source | GETs total | init | first_batch | warmup | steady | Steady GETs/step | Unique ex/step |"
      " Steady GETs/unique ex | Startup peak 1s QPS | Peak 1s QPS | 429 | 503 | Cancelled | Unique objects |"
      " Bytes recv MiB | NIC RX MiB |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
  ]
  for r in results:
    if "error" in r:
      continue
    for source, q in r.get("requests", {}).items():
      pp = q["per_phase"]
      oc = q["by_outcome"]
      lines.append(
          f"| {r['label']} | {source} | {q['total_gets']} | {pp['init']['gets']} | {pp['first_batch']['gets']} |"
          f" {pp['warmup']['gets']} | {pp['steady']['gets']} | {_fmt(q['steady_gets_per_step']['mean'], 2)} |"
          f" {r.get('unique_examples_per_step', '-')} | {_fmt(q.get('steady_gets_per_unique_example'), 3)} |"
          f" {q['startup_peak_1s_qps']} | {q['peak_1s_qps']} | {oc.get('throttled', 0)} | {oc.get('unavailable', 0)} |"
          f" {oc.get('cancelled', 0)} | {q['unique_objects']} | {q['bytes_received'] / 2**20:.2f} |"
          f" {r['nic_rx_bytes'] / 2**20:.2f} |"
      )
  lines += ["", "## Sample-order parity (SHA-256 over yielded batches)", ""]
  lines += [
      "| Run | Baseline | Status | Distinct host streams | Per-host digests (first 12 hex) |",
      "|---|---|---|---|---|",
  ]
  for r in results:
    if "error" in r:
      continue
    pv = r["parity_vs_baseline"]
    short = ", ".join(d[:12] for d in r["digests"])
    lines.append(f"| {r['label']} | {pv['baseline']} | {pv['status']} | {r['distinct_streams']} | {short} |")
  return "\n".join(lines) + "\n"


def passes_for(run: RunConfig, args: argparse.Namespace) -> tuple[str, ...]:
  """Returns the passes to run for one sweep point.

  Client tracing writes verbose stderr logs, so for live GCS (and the emulator
  with `--emulator_client_trace`) `paired` runs an untraced `timing` pass plus a
  traced `trace` pass. Emulator server-side recording, `/proc`, NIC, and GCSFuse
  counters do not perturb the client, so those backends need a single `measure`
  pass.

  Args:
    run: Sweep point.
    args: Parsed flags.

  Returns:
    Tuple of pass names.
  """
  client_tracing = run.backend == "gcs" or (run.backend == "emulator" and args.emulator_client_trace)
  if not client_tracing:
    return ("measure",)
  return {"paired": ("timing", "trace"), "timing": ("timing",), "trace": ("trace",)}[args.measure_mode]


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
  """Runs the full sweep and returns the JSON report."""
  runs = expand_sweep(args)
  emulator_root = ""
  datasets: dict[str, Any] = {}
  cleanup = []
  if args.backend == "emulator":
    emulator_root = args.data_dir or tempfile.mkdtemp(prefix="qps_bench_data_")
    if not args.data_dir:
      cleanup.append(emulator_root)
    gen = (
        generate_synthetic_arrayrecords.generate_text_dataset
        if args.modality == "text"
        else generate_synthetic_arrayrecords.generate_multimodal_dataset
    )
    info = gen(emulator_root, num_shards=args.num_shards, records_per_shard=args.records_per_shard)
    datasets[args.modality] = info
    pattern = f"gs://{EMULATOR_BUCKET}/{args.modality}/{generate_synthetic_arrayrecords.SHARD_PREFIX}-*"
    manifest = args.manifest_path
    if any(r.startup_counting == "manifest" for r in runs) and not manifest:
      with open(os.path.join(info["dir"], generate_synthetic_arrayrecords.MANIFEST), "r", encoding="utf-8") as f:
        local = json.load(f)
      manifest = os.path.join(emulator_root, "gs_manifest.json")
      with open(manifest, "w", encoding="utf-8") as f:
        json.dump({f"gs://{EMULATOR_BUCKET}/{os.path.relpath(k, emulator_root)}": v for k, v in local.items()}, f)
    runs = [dataclasses.replace(r, train_files=r.train_files or pattern, manifest_path=manifest) for r in runs]
  for r in runs:
    if not r.train_files:
      raise ValueError("--train_files is required for --backend=gcs/posix.")
    if r.startup_counting == "manifest" and not r.manifest_path:
      raise ValueError("--startup_counting=manifest requires --manifest_path (or --write_manifest first).")

  results = []
  try:
    for run in runs:
      passes = passes_for(run, args)
      print(f">>> {run.label()} passes={passes}", flush=True)
      res: dict[str, Any] = {"label": run.label(), "config": dataclasses.asdict(run), "parity_key": run.parity_key()}
      try:
        pass_results = {
            p: run_pass(
                run, client_traced=(p == "trace"), measure_qps=(p != "timing"), args=args, emulator_root=emulator_root
            )
            for p in passes
        }
      except RuntimeError as e:
        res["error"] = str(e)
        results.append(res)
        print(f"!!! {run.label()} failed: {e}", flush=True)
        continue
      timing_pass = pass_results.get("timing") or pass_results.get("measure") or pass_results["trace"]
      res["timing"] = timing_pass["timing"]
      res["timing_from_traced_pass"] = timing_pass["traced"]
      qps_pass = pass_results.get("trace") or pass_results.get("measure") or pass_results["timing"]
      res["requests"] = {
          k.replace("requests_", ""): v for k, v in qps_pass.items() if k.startswith("requests_") and v is not None
      }
      res["nic_rx_bytes"] = qps_pass["nic"]["rx_bytes"]
      res["unique_examples_per_step"] = unique_examples_per_step(run)
      for q in res["requests"].values():
        q["steady_gets_per_unique_example"] = round(
            q["steady_gets_per_step"]["mean"] / max(1, res["unique_examples_per_step"]), 4
        )
      res["digests"] = timing_pass["parity"]["per_host"]
      res["distinct_streams"] = timing_pass["parity"]["distinct_streams"]
      if len(pass_results) == 2:
        res["digests_identical_across_passes"] = (
            pass_results["timing"]["parity"]["per_host"] == pass_results["trace"]["parity"]["per_host"]
        )
      res["passes"] = pass_results
      results.append(res)
  finally:
    for d in cleanup:
      shutil.rmtree(d, ignore_errors=True)
  check_parity(results)
  return {"argv": sys.argv, "datasets": datasets, "runs": results}


def write_manifest(pattern: str, out_path: str, max_workers: int = 16) -> dict[str, int]:
  """Counts every shard once and writes a BP-1 manifest `{shard_path: num_records}`."""
  from concurrent import futures  # pylint: disable=import-outside-toplevel
  from maxtext.input_pipeline import grain_data_processing as gdp  # pylint: disable=import-outside-toplevel

  files = sorted(gdp.find_data_files(pattern))
  with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
    counts = dict(zip(files, ex.map(gdp._count_arrayrecord_file, files)))  # pylint: disable=protected-access
  if out_path.startswith("gs://"):
    from etils import epath  # pylint: disable=import-outside-toplevel

    epath.Path(out_path).write_text(json.dumps(counts, indent=2))
  else:
    with open(out_path, "w", encoding="utf-8") as f:
      json.dump(counts, f, indent=2)
  return counts


def create_parser() -> argparse.ArgumentParser:
  """CLI flags. List-valued sweep flags use `,` (numbers) or `|` (strings)."""
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  g = p.add_argument_group("data and backend")
  g.add_argument("--backend", choices=("emulator", "gcs", "posix"), default="emulator")
  g.add_argument("--train_files", default="", help="gs:// or local glob; '|' separates a BP-7 sweep across tiers.")
  g.add_argument("--modality", choices=("text", "multimodal"), default="text")
  g.add_argument("--data_dir", default="", help="Emulator: synthetic data root (generated if missing).")
  g.add_argument("--num_shards", type=int, default=16, help="Emulator synthetic shards.")
  g.add_argument("--records_per_shard", type=int, default=256, help="Emulator synthetic records per shard.")
  g.add_argument("--tokenizer_path", default="")
  g = p.add_argument_group("sweep axes (BP-1..BP-8)")
  g.add_argument("--host_counts", default="1", help="Simulated hosts per run, e.g. 1,4,8.")
  g.add_argument("--startup_counting", default="every_host", help="BP-1: every_host,sharded,manifest.")
  g.add_argument("--manifest_path", default="", help="BP-1 manifest JSON {shard_path: num_records}.")
  g.add_argument(
      "--grain_reader_options", default="", help="BP-2, e.g. 'footer_prefetch_bytes:1M|footer_prefetch_bytes:8M'."
  )
  g.add_argument("--startup_jitter_max_s", default="0", help="BP-3 deterministic per-host startup jitter, e.g. 0,10.")
  g.add_argument("--grain_worker_count", default="0", help="BP-4 mp_prefetch workers, e.g. 0,2,4,8,16.")
  g.add_argument("--grain_prefetch_buffer_size", default="500", help="BP-5 ReadOptions prefetch, e.g. 16,32,128,500.")
  g.add_argument("--data_sharding", default="per_host", help="BP-8: per_host,per_replica.")
  g.add_argument("--hosts_per_replica", type=int, default=8, help="BP-8 hosts reading the same slice in per_replica.")
  g.add_argument(
      "--per_replica_full_batch",
      type=lambda s: s.lower() in ("1", "true", "yes"),
      default=True,
      help="BP-8 per_replica: each host loads hosts_per_replica x per_host_batch_size (same global batch).",
  )
  g = p.add_argument_group("pipeline")
  g.add_argument("--grain_per_worker_buffer_size", type=int, default=1)
  g.add_argument("--grain_num_threads", type=int, default=16)
  g.add_argument("--per_host_batch_size", type=int, default=4)
  g.add_argument("--max_target_length", type=int, default=2048)
  g.add_argument("--warmup_steps", type=int, default=2)
  g.add_argument("--steps", type=int, default=10)
  g.add_argument("--target_step_time_s", type=float, default=0.0, help="Sleep to emulate accelerator step time.")
  g.add_argument("--lockstep", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
  g.add_argument("--data_shuffle_seed", type=int, default=0)
  g = p.add_argument_group("measurement")
  g.add_argument("--measure_mode", choices=("paired", "timing", "trace"), default="paired")
  g.add_argument("--emulator_client_trace", action="store_true", help="Also parse client traces in emulator runs.")
  g.add_argument("--emulator_fail_every_n", type=int, default=0, help="Inject a failure every Nth media GET.")
  g.add_argument("--emulator_fail_status", type=int, default=429)
  g.add_argument("--trace_root", default="", help="Directory for per-process client trace files.")
  g.add_argument("--keep_traces", action="store_true")
  g.add_argument("--cloud_monitoring_project", default="", help="Cross-check against Cloud Monitoring.")
  g.add_argument("--cloud_monitoring_delay_s", type=float, default=240.0)
  g.add_argument("--gcsfuse_metrics_url", default="", help="GCSFuse Prometheus endpoint for --backend=posix.")
  g.add_argument("--gcsfuse_metric", default="gcs_request_count")
  g.add_argument("--host_timeout_s", type=float, default=3600.0)
  g = p.add_argument_group("output")
  g.add_argument("--output_json", default="")
  g.add_argument("--output_markdown", default="")
  g.add_argument("--write_manifest", default="", help="Count --train_files shards once, write manifest, and exit.")
  return p


def main(argv: Sequence[str] | None = None) -> int:
  args = create_parser().parse_args(list(argv[1:]) if argv is not None else None)
  os.environ.setdefault("JAX_PLATFORMS", "cpu")
  if args.write_manifest:
    counts = write_manifest(args.train_files, args.write_manifest)
    print(f"Wrote {len(counts)} shard counts ({sum(counts.values())} records) to {args.write_manifest}")
    return 0
  report = run_benchmark(args)
  md = markdown_report(report["runs"])
  print(md)
  if args.output_json:
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2, default=str)
  if args.output_markdown:
    os.makedirs(os.path.dirname(os.path.abspath(args.output_markdown)), exist_ok=True)
    with open(args.output_markdown, "w", encoding="utf-8") as f:
      f.write(md)
  return 1 if any("error" in r for r in report["runs"]) else 0
