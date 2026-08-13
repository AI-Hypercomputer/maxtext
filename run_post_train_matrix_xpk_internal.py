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
import subprocess
import sys

from maxtext.utils.globals import HF_IDS

# The image checks the repo out at /deps (tracebacks report /deps/src/maxtext/...). globals.py
# derives MAXTEXT_REPO_ROOT from the installed package instead when the tree is not a git
# checkout, which yields paths that do not exist, so pin the root explicitly.
REPO_ROOT = os.environ.get("MAXTEXT_REPO_ROOT", "/deps")

MODEL = "llama3.1-8b"
SCAN_MODE = "scanned"
ACTIONS = [
    "sft",
    "dpo",
    "rl",
    "distill",
]

GCS_BASE = "gs://mesa-maxtext/validation_runs/post_train_xpk_0812"
HF_BASE = "gs://mesa-maxtext/huggingface_transformers"
LOCAL_LOGS = "/tmp/local_logs_post_train"
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
HF_CKPT = f"{HF_BASE}/{MODEL}/to_maxtext/{SCAN_MODE}/0/items"
# RL rolls out through vLLM and applies a chat template; MaxText's own tokenizer assets carry
# neither, so every RL job names the HuggingFace repo instead.
HF_REPO = "meta-llama/Meta-Llama-3.1-8B-Instruct"



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
  config = RL_CONFIG if action == "rl" else BASE_CONFIG
  cmd = [
      "python3",
      "-m",
      SCRIPTS[action],
      config,
      f"run_name={run_name}",
      f"model_name={MODEL}",
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
  elif action != "rl":
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
  hf_id = HF_REPO if action == "rl" else HF_IDS[MODEL]
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


def execute_command(cmd, log_path):
  """Executes a subprocess command, writing to log and streaming to stdout."""
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  env = os.environ.copy()

  cmd_str = " ".join(cmd)
  print(f"\n[EXECUTING]: {cmd_str}")
  print(f"[LOG PATH]: {log_path}")
  sys.stdout.flush()

  with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"Command: {cmd_str}\n\n")
    f.flush()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True) as process:
      for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        f.write(line)
      process.wait()

  if process.returncode != 0:
    print(f"[ERROR] Job failed with return code {process.returncode}. Check logs at: {log_path}")
  return process.returncode == 0


def run_matrix():
  """Runs each post-training action against the same base checkpoint."""
  os.makedirs(LOCAL_LOGS, exist_ok=True)
  with open(CSV_REPORT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["Model", "Scan Mode", "Action", "Run Name", "Status"])

  all_passed = True

  for action in ACTIONS:
    run_name = f"ckpt_{action}_base"
    print(f"\n{'='*80}\nStarting {action} for Model: {MODEL} | Scan Mode: {SCAN_MODE}\n{'='*80}")
    sys.stdout.flush()

    log_path = f"{LOCAL_LOGS}/{SCAN_MODE}/{action}/{MODEL}_{run_name}.log"
    success = execute_command(build_cmd(action, run_name), log_path)
    if not success:
      all_passed = False

    status = "PASS" if success else "FAIL"
    print(f"[{status}] {MODEL} | {SCAN_MODE} | {action} | {run_name}")
    sys.stdout.flush()
    with open(CSV_REPORT, "a", newline="", encoding="utf-8") as f:
      csv.writer(f).writerow([MODEL, SCAN_MODE, action, run_name, status])

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
