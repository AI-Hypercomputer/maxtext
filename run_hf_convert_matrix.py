"""Drives the HuggingFace -> MaxText checkpoint conversion matrix across models and scan modes.

The other matrices load base checkpoints that somebody converted by hand. This one regenerates
them, so a broken conversion shows up here instead of as a confusing failure downstream.

Run it when transformers is upgraded, when a model's HuggingFace repo changes, or to rebuild the
converted checkpoints a bucket is missing.

Usage:
  export HF_TOKEN=hf_...
  python run_hf_convert_matrix.py                       # every model in MODELS
  python run_hf_convert_matrix.py qwen3-0.6b gemma3-4b  # just these

  GCS_BASE=gs://my-bucket/conversions python run_hf_convert_matrix.py

Results land in hf_convert_summary.csv, and each run's output in local_logs_hf_convert/.
"""

# pylint: disable=bad-indentation

import csv
import json
import os
import subprocess
import sys

from etils import epath

MODELS = [
    "qwen3-0.6b",
    "qwen2.5-1.5b",
    "gemma2-2b",
    "gemma3-4b",
    "gemma4-e2b",
]
SCAN_MODES = ["scanned", "unscanned"]

GCS_BASE = os.environ.get("GCS_BASE", "gs://mesa-maxtext/hf_conversions")
LOCAL_LOGS = "local_logs_hf_convert"
CSV_REPORT = "hf_convert_summary.csv"


def get_loader_flags(model_name):
  """Returns the flags controlling how the HuggingFace weights are read.

  transformers renames a multimodal checkpoint's weights as it loads them, and MaxText's param
  map follows those names, so gemma3 has to be read through transformers rather than straight off
  the safetensors files. Reading it lazily fails with a missing key that looks like a corrupt
  checkpoint. Everything else is fine on the lazy loader, which uses far less RAM.
  """
  if model_name.startswith("gemma3"):
    return ["--lazy_load_tensors", "False", "--eager_load_method", "transformers"]
  return ["--lazy_load_tensors", "True"]


def execute_command(cmd, log_path):
  """Executes a subprocess command and writes the output to a log file."""
  os.makedirs(os.path.dirname(log_path), exist_ok=True)

  cmd_str = " ".join(cmd)
  print(f"\n[EXECUTING]: {cmd_str}")
  print(f"[LOG PATH]: {log_path}")

  with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"Command: {cmd_str}\n\n")
    f.flush()
    with subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy()) as process:
      try:
        process.wait(timeout=3600)
      except subprocess.TimeoutExpired:
        process.kill()
        print(f"[ERROR] Job timed out after 1 hour. Check logs at: {log_path}")
        return False
  return process.returncode == 0


def describe_checkpoint(output_dir):
  """Reports what the conversion actually wrote, or why the output is unusable.

  A conversion that exits zero can still leave nothing behind, so the exit code is not enough.
  Pre-training reads weights from the Linen `params.params.` collection, so that is what a usable
  checkpoint has to contain.
  """
  root = epath.Path(output_dir)
  if not root.exists():
    return False, "nothing written"

  steps = sorted((c.name for c in root.iterdir() if c.name.isdigit()), key=int)
  if not steps:
    return False, "no step directory"

  items = root / steps[-1] / "items"
  if not items.exists():
    return False, f"step {steps[-1]} has no items/"

  metadata = items / "array_metadatas"
  if not metadata.exists():
    return False, "no array metadata"

  entries = json.loads(next(iter(metadata.iterdir())).read_text())["array_metadatas"]
  names = [e["array_metadata"]["param_name"] for e in entries]
  in_params = [n for n in names if n.startswith("params.params.")]
  if not in_params:
    tops = sorted({n.split(".")[0] for n in names})
    return False, f"{len(names)} arrays, none under params.params. (top level: {tops})"
  return True, f"{len(in_params)}/{len(names)} arrays under params.params."


def first_error(log_path):
  """Returns the last exception line from a failed run, for the summary."""
  wanted = ("Error:", "Exception:")
  with open(log_path, encoding="utf-8", errors="replace") as f:
    hits = [line.strip() for line in f if any(w in line for w in wanted)]
  return hits[-1][:160] if hits else ""


def run_matrix(models):
  """Converts every model in both scan modes and records what landed on disk."""
  token = os.environ.get("HF_TOKEN") or os.environ.get("HF_AUTH_TOKEN")
  if not token:
    sys.exit("Set HF_TOKEN to a HuggingFace token that can read these repos.")

  with open(CSV_REPORT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["Model", "Scan Mode", "Status", "Detail"])

  for model in models:
    for scan_mode in SCAN_MODES:
      scan_bool = "True" if scan_mode == "scanned" else "False"
      output_dir = f"{GCS_BASE}/{model}/{scan_mode}"
      log_path = f"{LOCAL_LOGS}/{scan_mode}/{model}.log"

      cmd = [
          "python",
          "-m",
          "maxtext.checkpoint_conversion.to_maxtext",
          "src/maxtext/configs/base.yml",
          f"model_name={model}",
          f"base_output_directory={output_dir}",
          f"hf_access_token={token}",
          "hardware=cpu",
          "skip_jax_distributed_system=True",
          f"scan_layers={scan_bool}",
      ] + get_loader_flags(model)

      exited_clean = execute_command(cmd, log_path)
      usable, detail = describe_checkpoint(output_dir)
      if not exited_clean and not usable:
        detail = first_error(log_path) or detail

      status = "PASS" if exited_clean and usable else "FAIL"
      print(f"[{status}] {model} | {scan_mode} | {detail}")
      with open(CSV_REPORT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([model, scan_mode, status, detail])

  print(f"\nSummary written to {CSV_REPORT}")


if __name__ == "__main__":
  run_matrix(sys.argv[1:] or MODELS)
