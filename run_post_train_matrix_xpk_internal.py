"""Drives the llama3.1-8b post-training matrix (SFT, DPO, RL, distillation) on XPK.

This is the XPK counterpart to run_e2e_matrix.py. That script runs the full reload round trip
locally from repo-relative paths; this one runs the four post-training entry points inside the
container, where the repo lives under /deps and only the installed `maxtext` package is on the
path. Each action reloads the same HuggingFace-converted base checkpoint and trains a handful of
steps, so a broken trainer shows up here rather than in a longer job.

Results land in a CSV printed at the end, one row per action.
"""

# pylint: disable=bad-indentation

import csv
import json
import os
import re
import subprocess
import time
import threading
import sys

from maxtext.utils.globals import HF_IDS

# The image checks the repo out at /deps (tracebacks report /deps/src/maxtext/...). globals.py
# derives MAXTEXT_REPO_ROOT from the installed package instead when the tree is not a git
# checkout, which yields paths that do not exist, so pin the root explicitly.
REPO_ROOT = os.environ.get("MAXTEXT_REPO_ROOT", "/deps")

# Selected per workload rather than edited in place, so one image drives the whole matrix:
#   MATRIX_MODEL=llama3-8b MATRIX_SCAN_MODE=unscanned MATRIX_ACTIONS=sft,dpo bash run_vllm_matrix_xpk.sh ...
# `or` rather than a get() default: the wrapper exports these unconditionally and they arrive
# as empty strings when unset, which a default argument would not catch.
MODEL = os.environ.get("MATRIX_MODEL") or "llama3.1-8b"
SCAN_MODE = os.environ.get("MATRIX_SCAN_MODE") or "scanned"

ALL_ACTIONS = ["sft", "dpo", "rl", "distill"]
_requested = [a.strip() for a in os.environ.get("MATRIX_ACTIONS", "").split(",") if a.strip()]
if _requested:
  unknown = [a for a in _requested if a not in ALL_ACTIONS]
  if unknown:
    raise SystemExit(f"Unknown MATRIX_ACTIONS {unknown}; choose from {ALL_ACTIONS}")
ACTIONS = _requested or ALL_ACTIONS

GCS_BASE = "gs://mesa-maxtext/validation_runs/post_train_xpk_0812"
HF_BASE = "gs://mesa-maxtext/huggingface_transformers"
LOCAL_LOGS = "/tmp/local_logs_post_train"
# Same prefix the conversion and vLLM harnesses use, so run_vllm_matrix_xpk.sh's report sync pulls
# post-training logs down beside them without needing its own configuration.
REPORTS_GCS = os.environ.get("MATRIX_REPORTS_GCS") or "gs://mesa-maxtext/hf_conversions_xpk/_reports"
CSV_REPORT = "/tmp/post_train_summary.csv"

# The trainers are invoked as modules, not file paths: the container has maxtext installed but no
# guaranteed working directory, so `python3 src/maxtext/...` would not resolve.
SCRIPTS = {
    "sft": "maxtext.trainers.post_train.sft.train_sft",
    "dpo": "maxtext.trainers.post_train.dpo.train_dpo",
    "rl": "maxtext.trainers.post_train.rl.train_rl",
    "distill": "maxtext.trainers.post_train.distillation.train_distill",
}

BASE_CONFIG = os.path.join(REPO_ROOT, "src/maxtext/configs/base.yml")
RL_CONFIG = os.path.join(REPO_ROOT, "src/maxtext/configs/post_train/rl.yml")
DPO_DATASET = os.path.join(REPO_ROOT, "tests/assets/local_datasets/dpo/dpo_3_column_dataset.json")
RL_CHAT_TEMPLATE = os.path.join(REPO_ROOT, "src/maxtext/examples/chat_templates/gsm8k_rl.json")

SCAN_BOOL = "True" if SCAN_MODE == "scanned" else "False"

# Per-model additions to every trainer invocation.
#
# qwen3-vl-2b carries a vision tower whose parameters the sharding rules do not cover, so 6.29% of
# them stay unsharded and assert_params_sufficiently_sharded refuses to start against its 2%
# default. tests/end_to_end/tpu/qwen3/vl_2b/test_qwen3_multimodal_sft.sh raises it to 0.05, but
# that test also sets use_multimodal=true and trains on ChartQA, where the vision tower is part of
# the computation; this matrix feeds text, so the vision parameters sit unsharded and the measured
# figure is above 0.05 as well. Raised past what was measured rather than matching the official
# number, because the number here describes a different configuration. The unsharded vision tower
# is worth reporting on its own, and is not something this setting fixes.
EXTRA_FLAGS = {
    "qwen3-vl-2b": ["sharding_tolerance=0.10"],
}

# Models whose SFT has to follow the multimodal route instead of the shared one.
#
# qwen3-vl-2b's forward pass reads batch["images"], so the text datasets the rest of the matrix
# uses fail with KeyError: 'images' no matter how the sharding tolerance is set. The supported
# path, from tests/end_to_end/tpu/qwen3/vl_2b/test_qwen3_multimodal_sft.sh, is a different trainer
# (train_sft_native, not the Tunix-backed train_sft) fed ChartQA. That makes this row a test of a
# different code path from every other SFT row here, which is worth knowing when reading the
# result: it says the multimodal trainer works, not that the Tunix one does.
MULTIMODAL_SFT = {
    "qwen3-vl-2b": {
        "module": "maxtext.trainers.post_train.sft.train_sft_native",
        # base.yml is not enough: MaxTextConfig rejects multimodal SFT unless
        # sft_train_on_completion_only is set, among other things. This config carries the whole
        # combination -- use_multimodal, the completion-only flag, frozen vision encoder, and the
        # ChartQA column mapping -- so it is used rather than reproduced flag by flag.
        "config": os.path.join(REPO_ROOT, "src/maxtext/configs/post_train/sft-vision-chartqa.yml"),
        "flags": [
            # The config defaults to pulling ChartQA from the Hub; these are the parquet copies the
            # official test reads instead, which do not depend on Hub availability mid-run.
            "hf_path=parquet",
            "hf_train_files=gs://maxtext-dataset/hf/chartqa/train-*",
            "sharding_tolerance=0.05",
            "checkpoint_storage_use_zarr3=False",
            "checkpoint_storage_use_ocdbt=False",
            "float32_qk_product=True",
            "float32_logits=True",
            "grain_worker_count=0",
        ],
    },
}

# Where this model's weights were converted to. The bulk of the matrix reads the shared
# huggingface_transformers layout, but models converted by run_hf_convert_matrix_xpk_internal.py
# land elsewhere, and llama3-8b and gemma4-31b exist only there.
CONVERSIONS = "gs://mesa-maxtext/hf_conversions_xpk"
CKPT_OVERRIDES = {
    "llama3-8b": f"{CONVERSIONS}/llama3-8b/{SCAN_MODE}/0/items",
    "gemma4-31b": f"{CONVERSIONS}/gemma4-31b/{SCAN_MODE}/0/items",
    # Multimodal, and only ever converted unscanned: MaxTextConfig rejects scan_layers=True for it
    # ("Deepstack visual embedding injection requires scan_layers=False").
    "qwen3-vl-2b": f"{CONVERSIONS}/qwen3-vl-2b/unscanned/0/items",
}
def _resolve_ckpt():
  """Prefers this repo's own conversion, falling back to the older HF_BASE layout.

  The fallback is not equivalent: HF_BASE holds conversions of unknown age and provenance, and a
  post-training run that quietly trains from one of those is not testing the checkpoints this repo
  produces. Which path was used is printed so a result can be read for what it is.
  """
  if os.environ.get("MATRIX_CKPT"):
    return os.environ["MATRIX_CKPT"]
  fresh = CKPT_OVERRIDES.get(MODEL, f"{CONVERSIONS}/{MODEL}/{SCAN_MODE}/0/items")
  probe = subprocess.run(["gcloud", "storage", "ls", f"{fresh}/"], capture_output=True, text=True, check=False)
  if probe.returncode == 0 and probe.stdout.strip():
    print(f"[INFO] Using this repo's conversion: {fresh}")
    return fresh
  legacy = f"{HF_BASE}/{MODEL}/to_maxtext/{SCAN_MODE}/0/items"
  print(f"[WARN] No conversion at {fresh}; falling back to {legacy}")
  return legacy


HF_CKPT = _resolve_ckpt()

# RL rolls out through vLLM and applies a chat template; MaxText's own tokenizer assets carry
# neither, so every RL job names the HuggingFace repo instead. Instruct-tuned repos are the ones
# that ship a chat template, so prefer one where it exists.
RL_REPOS = {
    "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma4-31b": "google/gemma-4-31b-it",
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
}
HF_REPO = RL_REPOS.get(MODEL) or HF_IDS.get(MODEL, "")



# The loader pulls one row per device per step, so the row count a run needs scales with the
# slice: 5 steps is 160 rows on a v6e-32 but 1280 on a v6e-256. Both bundled datasets are far
# smaller than that, so both get repeated up to a size that covers the slices we run on.
DATASET_ROWS_NEEDED = 4096


def prepare_c4_dataset():
  """Repeats the bundled C4 sample so it can feed the whole run.

  The checked-in parquet holds 200 rows, which is under even a v6e-32's 160-row requirement once
  eval is accounted for, and nowhere near a v6e-256's 1280. Repeating is fine here: this checks
  that the trainers load, step and save, not that they learn anything from the data.
  """
  import pyarrow as pa  # pylint: disable=import-outside-toplevel
  import pyarrow.parquet as pq  # pylint: disable=import-outside-toplevel

  src = os.path.join(REPO_ROOT, "tests/assets/local_datasets/c4_en_dataset_minimal/hf/c4/c4-train-00000-of-01637.parquet")
  table = pq.read_table(src)
  repeats = DATASET_ROWS_NEEDED // table.num_rows + 1
  expanded = pa.concat_tables([table] * repeats).slice(0, DATASET_ROWS_NEEDED)
  out_path = "/tmp/c4_expanded.parquet"
  pq.write_table(expanded, out_path)
  print(f"[INFO] Expanded C4 dataset from {table.num_rows} to {expanded.num_rows} rows at {out_path}")
  sys.stdout.flush()
  return out_path


def prepare_dpo_dataset():
  """Repeats the bundled preference dataset until it can feed the whole run.

  The bundled file holds 22 rows. The loader pulls `global_batch_size_to_load` rows per step,
  which is one per device (32 here) no matter how small per_device_batch_size is -- a fractional
  batch shrinks global_batch_size_to_train_on but not the load. Five steps therefore need 160
  rows, and the run dies on the first batch with "You may have run out of training data".
  Repeated rows are fine: this checks that DPO loads a checkpoint and steps, not that it learns.
  """
  with open(DPO_DATASET, encoding="utf-8") as f:
    rows = json.load(f)
  expanded = (rows * (DATASET_ROWS_NEEDED // len(rows) + 1))[:DATASET_ROWS_NEEDED]
  out_path = "/tmp/dpo_expanded_dataset.json"
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(expanded, f)
  print(f"[INFO] Expanded DPO dataset from {len(rows)} to {len(expanded)} rows at {out_path}")
  sys.stdout.flush()
  return out_path


def build_cmd(action, run_name):
  """Builds the trainer command for one post-training action."""
  multimodal = MULTIMODAL_SFT.get(MODEL) if action == "sft" else None
  config = multimodal["config"] if multimodal else (RL_CONFIG if action == "rl" else BASE_CONFIG)
  cmd = [
      "python3",
      "-m",
      multimodal["module"] if multimodal else SCRIPTS[action],
      config,
      f"run_name={run_name}",
      f"model_name={MODEL}",
      *(multimodal["flags"] if multimodal else EXTRA_FLAGS.get(MODEL, [])),
      f"scan_layers={SCAN_BOOL}",
      f"base_output_directory={GCS_BASE}/{SCAN_MODE}/{action}/{MODEL}",
      f"load_parameters_path={HF_CKPT}",
      # XPK submits these as Pathways workloads, where the client talks to a proxy rather than
      # joining a coordinator. Without this, max_utils calls jax.distributed.initialize() and
      # dies with "coordinator_address should be defined." before the trainer starts.
      "enable_single_controller=True",
      # Saving pulls params and opt_state back through the pathways-proxy sidecar, which XPK caps
      # at 100G. base.yml's weight_dtype=float32 makes that ~96GiB for 8B (params + Adam mu/nu,
      # mu_dtype inherits weight_dtype) and the proxy is OOMKilled mid-transfer, which surfaces on
      # the client as "Stream removed". bfloat16 halves the state to ~48GiB; it is also what the
      # repo's own SFT e2e tests use (tests/end_to_end/tpu/qwen3/30b/test_qwen3_sft.sh).
      "weight_dtype=bfloat16",
      "dtype=bfloat16",
      # Even at bfloat16 the save is ~45GiB, and Orbax's default device-to-host concurrency
      # (~89GiB) lets it materialise the whole tree at once, which OOMKills the 100G jax-tpu
      # container. Stream it in smaller pieces instead; the checkpoint written is identical.
      # This has a hard floor: the budget must exceed the largest single array, and
      # llama3.1-8b's mlp.wi_0.kernel is 3.75GiB, so anything below ~4 fails outright with
      # "Requested more bytes than we reserved space for". 8 got the first save through but the
      # host memory was not released before the end-of-training save. 4 is the smallest value
      # that clears the floor, so it gives the lowest peak the model allows.
      "checkpoint_storage_concurrent_gb=4",
      # Async saving is what OOMKilled the container: llama3-8b's 44.9 GiB save died with
      # exitCode 137 at step 1 while the device-to-host cap above was set and in force (the three
      # handlers logged save_device_host_concurrent_bytes=4000000000). The async engine's own
      # regulator reported "MemoryRegulated: Peak usage: 0.000000 GiB" for that save, i.e. it was
      # not measuring anything, so the cap it governs was not restraining the transfer either.
      # Synchronous saving takes that engine out of the path. Set MATRIX_ASYNC_CKPT=true to
      # restore the default and compare.
      f"async_checkpointing={os.environ.get('MATRIX_ASYNC_CKPT') or 'False'}",
      # Saving is off by default here because it is the one step these runs cannot survive: an 8B
      # SFT writes 44.9 GiB and the transfer to host OOMKills the container (exitCode 137) whether
      # the save is async or sync, with the device-to-host cap set and in force. What this matrix
      # answers is whether each trainer loads, steps and produces a loss for a given model and
      # layout; the save path is a separate defect, tracked separately, and letting it kill every
      # run answers nothing. MATRIX_SKIP_CKPT=False puts it back for testing the save itself.
      f"post_train_skip_checkpointing={os.environ.get('MATRIX_SKIP_CKPT') or 'True'}",
  ]

  if action == "rl":
    cmd.extend(["num_batches=2", "rl.num_generations=2", f"chat_template_path={RL_CHAT_TEMPLATE}"])
    # rl.yml splits the slice 50/50 between trainer and sampler (devices 0-15 vs 16-31 here).
    # On this 8-VM v6e-32 that split kills the Pathways workers with "ExecuteShard attempted to
    # execute on device id 27 which is not addressable by this client". Giving both roles the
    # whole slice is the unsplit single-slice behaviour the same code path already supports.
    cmd.extend(["trainer_devices_fraction=1.0", "sampler_devices_fraction=1.0"])
    # Sharing the slice means the vLLM sampler and the trainer compete for the same HBM. rl.yml's
    # 0.72 starves the trainer (RuntimeBufferAllocationFailure), while 0.3 starves the sampler,
    # which reports needing 508.59GiB of the slice's 1024GiB. 0.55 caps the sampler at ~563GiB,
    # just over what it asks for, and leaves the trainer the remaining ~460GiB.
    cmd.append("hbm_utilization_vllm=0.55")
  elif action == "dpo":
    # Batch stays at 1: the dataset is expanded instead, see prepare_dpo_dataset().
    cmd.extend(["steps=5", "per_device_batch_size=1"])
  else:
    # The minimal local C4 parquet holds 200 rows, so 5 steps at a global batch of 32 fit.
    cmd.extend(["steps=5", "per_device_batch_size=1"])

  # DPO needs real preference data (prompt/chosen/rejected); the others are fine on synthetic.
  if action == "dpo":
    cmd.extend(
        [
            "dataset_type=hf",
            "hf_path=json",
            f"hf_train_files={prepare_dpo_dataset()}",
            "train_data_columns=\"['prompt', 'chosen', 'rejected']\"",
            "tokenize_train_data=True",
            "use_dpo=True",
            "packing=False",
        ]
    )
  elif action != "rl" and not multimodal:
    # Skipped for multimodal SFT, which brings its own dataset: appending these afterwards replaced
    # ChartQA with the text-only C4 sample and the run died on
    # "Column name ['image', 'label', 'query'] not in the dataset. Columns in the dataset: ['text']".
    #
    # dataset_type=synthetic is unusable here: SyntheticDataIterator jits with static_argnums=0
    # over the config (synthetic_data_processing.py), and the pydantic MaxTextConfig is not
    # hashable, so the first batch dies with "unhashable type: 'MaxTextConfig'". Read the minimal
    # local C4 parquet instead. RL is left alone: rl.yml configures its own gsm8k dataset.
    cmd.extend(
        [
            "dataset_type=hf",
            "hf_path=parquet",
            f"hf_train_files={prepare_c4_dataset()}",
        ]
    )

  # Every action now reads a HuggingFace-format dataset, and that pipeline tokenizes through
  # AutoTokenizer, which cannot open MaxText's own tokenizer assets. Name the repo instead.
  # RL additionally applies a chat template, which only the Instruct repo carries.
  # HF_IDS resolves against whichever maxtext the interpreter imported, which is not always the
  # checkout being tested -- a model added to the repo's globals.py is absent from an older
  # installed copy, and indexing it directly turns that into a KeyError before the job even starts.
  hf_id = HF_REPO if action == "rl" else (HF_IDS.get(MODEL) or HF_REPO)
  if not hf_id:
    raise SystemExit(f"No HuggingFace repo known for {MODEL}; add it to RL_REPOS or HF_IDS.")
  cmd.extend([f"tokenizer_path={hf_id}", "tokenizer_type=huggingface"])
  if action == "rl":
    cmd.append(f"vllm_hf_config_path={HF_REPO}")

  # Distillation trains a student against a teacher, so the teacher needs its own weights.
  if action == "distill":
    cmd.extend(
        [
            f"teacher_overrides.load_parameters_path={HF_CKPT}",
            f"teacher_overrides.model_name={MODEL}",
            f"teacher_overrides.scan_layers={SCAN_BOOL}",
            "teacher_overrides.per_device_batch_size=1",
            # The teacher gets its own config, which does not inherit the student's flags. Without
            # this it defaults to enable_single_controller=False and calls jax.distributed
            # .initialize() after the student has already brought up the XLA backend, which fails
            # with "must be called before any JAX calls that might initialise the XLA backend".
            "teacher_overrides.enable_single_controller=True",
        ]
    )

  return cmd


# How long a trainer may produce nothing before it is treated as stuck.
#
# Three runs today sat at zero output for over an hour with the process alive on a few percent of
# one core and its log file untouched -- not compiling, which keeps a core busy and keeps writing,
# but waiting on something that never arrived. Each one held a node for the whole time and had to
# be noticed by hand. Compilation of an 8B step is minutes, not half an hour, so silence past this
# is a hang; the job is killed and reported as such rather than left to occupy the cluster.
STALL_TIMEOUT_SECONDS = 30 * 60


def ship_case_log(action, status, detail, log_path):
  """Appends the outcome to a case's log and copies it to the reports prefix.

  Shipped per case rather than at the end of the run, and for the same reason the vLLM harness does
  it: a job that takes the pod down carries every earlier case's log with it. The distillation
  matrix on 0815-0816 ran 44 cases whose per-case logs stayed inside the pod at
  /tmp/local_logs_post_train and were gone the moment it exited, leaving only the full pod dump to
  reconstruct results from.
  """
  if os.path.exists(log_path):
    with open(log_path, "a", encoding="utf-8") as f:
      f.write(f"\n{'=' * 100}\nRESULT: {status}\nDETAIL: {detail}\n")
  else:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
      f.write(f"RESULT: {status}\nDETAIL: {detail}\n")

  remote = f"{REPORTS_GCS}/{MODEL}_{SCAN_MODE}_{action}.log"
  if subprocess.run(["gcloud", "storage", "cp", log_path, remote], check=False).returncode == 0:
    print(f"[INFO] log -> {remote}")
  else:
    print(f"[WARN] could not copy {log_path} to {remote}")


def execute_command(cmd, log_path):
  """Executes a subprocess command, writing to log and streaming to stdout."""
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  env = os.environ.copy()

  cmd_str = " ".join(cmd)
  print(f"\n[EXECUTING]: {cmd_str}")
  print(f"[LOG PATH]: {log_path}")
  sys.stdout.flush()

  stalled = threading.Event()

  with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"Command: {cmd_str}\n\n")
    f.flush()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True) as process:
      last_output = [time.monotonic()]

      def watchdog():
        while process.poll() is None:
          if time.monotonic() - last_output[0] > STALL_TIMEOUT_SECONDS:
            stalled.set()
            print(
                f"\n[STALL] No output for {STALL_TIMEOUT_SECONDS // 60} minutes; killing.",
                flush=True,
            )
            process.kill()
            return
          time.sleep(30)

      threading.Thread(target=watchdog, daemon=True).start()

      for line in process.stdout:
        last_output[0] = time.monotonic()
        sys.stdout.write(line)
        sys.stdout.flush()
        f.write(line)
        f.flush()
      process.wait()

  if process.returncode != 0:
    print(f"[ERROR] Job failed with return code {process.returncode}. Check logs at: {log_path}")
  # The exit code carries the only account of a death that produced no traceback: a trainer killed
  # by the OOM killer exits on -9 and prints nothing, which is otherwise indistinguishable from a
  # clean failure when only PASS/FAIL is recorded.
  if stalled.is_set():
    return False, "stalled"
  return process.returncode == 0, process.returncode


# The Pathways proxy does not always survive a trainer that dies badly, and once it is gone every
# later action in the same pod fails identically on "Unable to establish connection to ifrt_proxy
# server" -- a report of four failures where there was one. Detected so the rest are marked SKIP
# for the reason they were actually skipped.
PROXY_GONE = "Unable to establish connection to ifrt_proxy server"


def proxy_died(log_path):
  try:
    with open(log_path, encoding="utf-8", errors="replace") as f:
      return PROXY_GONE in f.read()
  except OSError:
    return False


def last_error(log_path):
  """Best-effort one-line reason, for a summary that would otherwise say only FAIL."""
  wanted = ("Error", "Exception", "RESOURCE_EXHAUSTED", "Killed", "AssertionError")
  # Interpreter teardown emits its own errors after the real one -- "ImportError: sys.meta_path is
  # None, Python is likely shutting down" is a consequence of the exit, not a cause -- so the last
  # matching line is usually the least informative. These are dropped and the first real exception
  # kept: for qwen3-vl-2b the difference is reporting KeyError: 'images', which names the problem,
  # rather than a shutdown artefact that names nothing.
  noise = (
      "sys.meta_path is None",
      "Python is likely shutting down",
      # Printed during teardown after the process is already failing, and matched on "Exception":
      # it was reported as the cause of both gemma3-4b RL failures, whose actual error was a dtype
      # mismatch several frames earlier.
      "Exception ignored in",
      "GCSRecordWriter",
  )
  # Chained exceptions are printed oldest first, so the first one in the file is the one that was
  # caught and handled, not the one that stopped the run. Tunix's mappings.maybe_call probes
  # to_hf_mappings(backend) and falls back to to_hf_mappings() on TypeError, so every RL failure
  # below it led with "takes 1 positional argument but 2 were given" -- a successful probe -- while
  # the real cause ("gemma2-2b vLLM weight mapping not found") sat after the chaining marker.
  chain_markers = (
      "During handling of the above exception",
      "The above exception was the direct cause",
  )
  try:
    with open(log_path, encoding="utf-8", errors="replace") as f:
      raw = [ln.strip() for ln in f]
  except OSError:
    return ""

  last_marker = max(
      (i for i, ln in enumerate(raw) if any(m in ln for m in chain_markers)), default=-1
  )
  after_marker = [
      ln
      for ln in raw[last_marker + 1 :]
      if any(w in ln for w in wanted) and not any(n in ln for n in noise)
  ]
  lines = after_marker or [
      ln for ln in raw if any(w in ln for w in wanted) and not any(n in ln for n in noise)
  ]
  # "raise ValueError(" matches on "Error" but is the source line that threw, not the message.
  # An actual exception line reads "SomeError: what went wrong", so prefer one of those and fall
  # back to anything only when none is present.
  # The optional "[rankN]: " prefix is part of every line in a multi-host run, so anchoring the
  # exception pattern at the start of the line matched none of them and fell through to the raise
  # statement instead of the message it raises.
  formatted = [ln for ln in lines if re.match(r"^(\[rank\d+\]:\s*)?[\w.]*(Error|Exception)\b.*:", ln)]
  chosen = formatted or lines
  return chosen[0][:150] if chosen else ""


def run_matrix():
  """Runs each post-training action against the same base checkpoint."""
  os.makedirs(LOCAL_LOGS, exist_ok=True)
  with open(CSV_REPORT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["Model", "Scan Mode", "Action", "Run Name", "Status", "Detail"])

  all_passed = True
  proxy_lost = False

  for action in ACTIONS:
    run_name = f"ckpt_{action}_base"
    print(f"\n{'='*80}\nStarting {action} for Model: {MODEL} | Scan Mode: {SCAN_MODE}\n{'='*80}")
    sys.stdout.flush()

    log_path = f"{LOCAL_LOGS}/{SCAN_MODE}/{action}/{MODEL}_{run_name}.log"

    if proxy_lost:
      print(f"[SKIP] {MODEL} | {SCAN_MODE} | {action} | Pathways proxy gone, result would be meaningless")
      with open(CSV_REPORT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([MODEL, SCAN_MODE, action, run_name, "SKIP", "Pathways proxy gone"])
      ship_case_log(action, "SKIP", "Pathways proxy gone", log_path)
      continue

    success, returncode = execute_command(build_cmd(action, run_name), log_path)
    if not success and proxy_died(log_path):
      proxy_lost = True
    if not success:
      all_passed = False

    status = "PASS" if success else "FAIL"
    detail = ""
    if not success:
      if returncode == "stalled":
        detail = f"no output for {STALL_TIMEOUT_SECONDS // 60}m, killed as hung"
      else:
        # A negative code is a signal: -9 is SIGKILL, which on this host means the OOM killer.
        detail = f"exit={returncode}" + (" (SIGKILL, likely OOM)" if returncode == -9 else "")
      reason = last_error(log_path)
      if reason:
        detail += f" | {reason}"
    print(f"[{status}] {MODEL} | {SCAN_MODE} | {action} | {run_name}" + (f" | {detail}" if detail else ""))
    sys.stdout.flush()
    with open(CSV_REPORT, "a", newline="", encoding="utf-8") as f:
      csv.writer(f).writerow([MODEL, SCAN_MODE, action, run_name, status, detail])
    ship_case_log(action, status, detail, log_path)

  print("\n" + "=" * 80)
  print("POST-TRAIN VALIDATION SUMMARY")
  print("=" * 80)
  with open(CSV_REPORT, "r", encoding="utf-8") as f:
    print(f.read())
  print("=" * 80)

  if not all_passed:
    sys.exit(1)


if __name__ == "__main__":
  run_matrix()
