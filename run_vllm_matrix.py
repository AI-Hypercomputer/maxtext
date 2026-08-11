"""Drives the vLLM decode validation matrix across models and scan modes.

Tests that pre-training checkpoints can be successfully loaded and evaluated
by the vLLM integration pipeline.
"""

# pylint: disable=bad-indentation

import os
import subprocess
import csv

from maxtext.utils.globals import HF_IDS

MODELS = [
    "llama3.1-8b",
    "gemma3-4b",
    # Add other models as needed
]
SCAN_MODES = ["scanned", "unscanned"]

GCS_BASE = "gs://mesa-maxtext/validation_runs/post_train_layout_0811_1"
LOCAL_LOGS = "local_logs_vllm"
CSV_REPORT = "vllm_validation_summary.csv"


def get_tokenizer_flags(model_name):
  """Returns tokenizer flags for vLLM decoding."""
  flags = []
  if "gemma4" in model_name:
    flags.extend(
        [
            "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer_gemma4.model",
            "tokenizer_type=sentencepiece",
        ]
    )
  elif "gemma3" in model_name:
    flags.extend(
        [
            "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.gemma3",
            "tokenizer_type=sentencepiece",
        ]
    )
  elif "gemma" in model_name:
    flags.extend(
        [
            "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.gemma",
            "tokenizer_type=sentencepiece",
        ]
    )
  elif "llama" in model_name:
    flags.extend(
        [
            "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken",
            "tokenizer_type=tiktoken",
        ]
    )
  elif "mistral" in model_name:
    flags.extend(
        [
            "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.mistral-v3",
            "tokenizer_type=sentencepiece",
        ]
    )
  elif "qwen" in model_name or "olmo" in model_name or "gpt-oss" in model_name:
    flags.extend([f"tokenizer_path={HF_IDS[model_name]}", "tokenizer_type=huggingface"])
  return flags


def execute_command(cmd, log_path):
  """Executes a subprocess command and writes the output to a log file."""
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  env = os.environ.copy()

  cmd_str = " ".join(cmd)
  print(f"\n[EXECUTING]: {cmd_str}")
  print(f"[LOG PATH]: {log_path}")

  with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"Command: {cmd_str}\n\n")
    f.flush()
    with subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env) as process:
      try:
        process.wait(timeout=3600)  # 1 hour timeout per decode run
      except subprocess.TimeoutExpired:
        process.kill()
        print(f"[ERROR] Job timed out after 1 hour. Check logs at: {log_path}")
        return False

  if process.returncode != 0:
    print(f"[ERROR] Job failed. Check logs at: {log_path}")
  return process.returncode == 0


def run_matrix():
  """Runs the vLLM decode validation matrix for all models and scan modes."""
  with open(CSV_REPORT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Scan Mode", "Phase", "Run Name", "Status"])

  for model in MODELS:
    for scan_mode in SCAN_MODES:
      print(f"\n{'='*80}\nStarting vLLM decode for Model: {model} | Scan Mode: {scan_mode}\n{'='*80}")
      scan_bool = "True" if scan_mode == "scanned" else "False"

      # 1. Run Pre-train to generate checkpoints
      run_name = "ckpt_pretrain_base"
      pretrain_cmd = [
          "python",
          "src/maxtext/trainers/pre_train/train.py",
          "src/maxtext/configs/base.yml",
          f"run_name={run_name}",
          f"model_name={model}",
          f"scan_layers={scan_bool}",
          f"base_output_directory={GCS_BASE}/{scan_mode}/pre_train/{model}",
          "steps=5",
          "per_device_batch_size=1",
          "checkpoint_period=5",
          "dataset_type=synthetic",
      ]
      pretrain_cmd.extend(get_tokenizer_flags(model))

      log_path_pretrain = f"{LOCAL_LOGS}/{scan_mode}/{model}_pre_train.log"
      success_pretrain = execute_command(pretrain_cmd, log_path_pretrain)
      status_pretrain = "PASS" if success_pretrain else "FAIL"
      print(f"[{status_pretrain}] {model} | {scan_mode} | pre_train")
      with open(CSV_REPORT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([model, scan_mode, "pre_train", run_name, status_pretrain])

      if not success_pretrain:
        continue

      # 2. Run vLLM Decode on the generated checkpoints
      load_path = f"{GCS_BASE}/{scan_mode}/pre_train/{model}/{run_name}/checkpoints/4/items"

      cmd = [
          "python",
          "-m",
          "maxtext.inference.vllm_decode",
          f"model_name={model}",
          f"load_parameters_path={load_path}",
          'vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]}',
          "hbm_utilization_vllm=0.9",
          "prompt=Suggest some famous landmarks in London.",
          "use_chat_template=True",
          f"scan_layers={scan_bool}",
      ]

      # Handle tokenizer config Overrides for vLLM
      cmd.extend(get_tokenizer_flags(model))

      if "llama3" in model:
        hf_model_name = f"meta-llama/Meta-Llama-3.1-{model.rsplit('-', maxsplit=1)[-1].upper()}-Instruct"
        cmd = [c for c in cmd if not c.startswith("tokenizer_path=") and not c.startswith("tokenizer_type=")]
        cmd.extend([f"tokenizer_path={hf_model_name}", "tokenizer_type=huggingface"])
        cmd.append(f"vllm_hf_config_path={hf_model_name}")
      elif "gemma3" in model:
        hf_model_name = f"google/gemma-3-{model.rsplit('-', maxsplit=1)[-1]}-it"
        cmd.append(f"vllm_hf_config_path={hf_model_name}")

      log_path = f"{LOCAL_LOGS}/{scan_mode}/{model}_vllm_decode.log"
      success = execute_command(cmd, log_path)

      status = "PASS" if success else "FAIL"
      print(f"[{status}] {model} | {scan_mode} | vllm_decode")
      with open(CSV_REPORT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([model, scan_mode, "vllm_decode", run_name, status])

      break

    break


if __name__ == "__main__":
  run_matrix()
