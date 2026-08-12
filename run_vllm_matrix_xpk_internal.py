"""Drives the vLLM decode validation matrix across models and scan modes.
Optimized for XPK execution.
"""

import os
import subprocess
import csv
import sys

# --- ABSOLUTE TOP: Register OmegaConf resolver for HF_TOKEN ---
# This fixes the "Unsupported interpolation type HF_TOKEN" error in MaxText
try:
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver("HF_TOKEN"):
        print("[INFO] Registering OmegaConf resolver for HF_TOKEN...")
        OmegaConf.register_new_resolver("HF_TOKEN", lambda: os.environ.get("HF_TOKEN", ""))
except ImportError:
    print("[WARNING] omegaconf not found. Resolver registration skipped.")
# -------------------------------------------------------------

# Try to import HF_IDS, fallback to empty dict if maxtext is not available locally
try:
    from maxtext.utils.globals import HF_IDS
except ImportError:
    HF_IDS = {}

MODELS = [
    "llama3.1-8b",
    # "gemma3-4b",
    # Add other models as needed
]
SCAN_MODES = [
   "scanned", 
  #  "unscanned"
  ]

GCS_BASE = "gs://mesa-maxtext/validation_runs/post_train_layout_0811_2"
LOCAL_LOGS = "/tmp/local_logs_vllm"
CSV_REPORT = "/tmp/vllm_validation_summary.csv"


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
    hf_id = HF_IDS.get(model_name, f"openai/{model_name}" if "gpt-oss" in model_name else model_name)
    flags.extend([f"tokenizer_path={hf_id}", "tokenizer_type=huggingface"])
  return flags


def execute_command(cmd, log_path):
  """Executes a subprocess command, writes to log and prints to stdout."""
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  env = os.environ.copy()

  cmd_str = " ".join(cmd)
  print(f"\n[EXECUTING]: {cmd_str}")
  print(f"[LOG PATH]: {log_path}")

  with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"Command: {cmd_str}\n\n")
    f.flush()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True) as process:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
        process.wait(timeout=3600)

  if process.returncode != 0:
    print(f"[ERROR] Job failed with return code {process.returncode}. Check logs at: {log_path}")
  return process.returncode == 0


def run_matrix():
  """Runs the vLLM decode validation matrix for all models and scan modes."""
  os.makedirs(LOCAL_LOGS, exist_ok=True)
  with open(CSV_REPORT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Scan Mode", "Phase", "Run Name", "Status"])

  all_passed = True

  for model in MODELS:
    for scan_mode in SCAN_MODES:
      print(f"\n{'='*80}\nStarting vLLM decode for Model: {model} | Scan Mode: {scan_mode}\n{'='*80}")
      scan_bool = "True" if scan_mode == "scanned" else "False"

      run_name = "ckpt_pretrain_base"
      load_path = f"{GCS_BASE}/{scan_mode}/pre_train/{model}/{run_name}/checkpoints/9/items"
      load_path = "gs://mesa-maxtext/huggingface_transformers/llama3.1-8b-Instruct/to_maxtext/scanned/0/items"
      cmd = [
          "python3",
          "-m",
          "maxtext.inference.vllm_decode",
          f"model_name={model}",
          f"load_parameters_path={load_path}",
          'vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]}',
          "hbm_utilization_vllm=0.35",
          "prompt=Suggest some famous landmarks in London.",
          "use_chat_template=True",
          "enable_single_controller=True",
          f"scan_layers={scan_bool}",
      ]

      # Note: hf_access_token is NOT passed here explicitly.
      # The global OmegaConf resolver will handle the ${HF_TOKEN} interpolation 
      # by reading it from the environment variable inside the process.

      cmd.extend(get_tokenizer_flags(model))

      if "llama3" in model:
        size = model.split("-")[1].upper() if "-" in model else "8B"
        hf_model_name = f"meta-llama/Meta-Llama-3.1-{size}-Instruct"
        # Filter out existing tokenizer flags
        cmd = [c for c in cmd if not c.startswith("tokenizer_path=") and not c.startswith("tokenizer_type=")]
        cmd.extend([f"tokenizer_path={hf_model_name}", "tokenizer_type=huggingface"])
        cmd.append(f"vllm_hf_config_path={hf_model_name}")
        cmd.append("ici_tensor_parallelism=8")
        cmd.append("max_prefill_predict_length=1024")
        cmd.append("max_target_length=1024")
        cmd.append("attention=dot_product")
      elif "gemma3" in model:
        size = model.split("-")[1] if "-" in model else "4b"
        hf_model_name = f"google/gemma-3-{size}-it"
        # Filter out existing local tokenizer flags
        cmd = [c for c in cmd if not c.startswith("tokenizer_path=") and not c.startswith("tokenizer_type=")]
        cmd.extend([f"tokenizer_path={hf_model_name}", "tokenizer_type=huggingface"])
        cmd.append(f"vllm_hf_config_path={hf_model_name}")
        cmd.append("use_standalone_converter=True")

      log_path = f"{LOCAL_LOGS}/{scan_mode}/{model}_vllm_decode.log"
      success = execute_command(cmd, log_path)
      if not success:
          all_passed = False

      status = "PASS" if success else "FAIL"
      print(f"[{status}] {model} | {scan_mode} | vllm_decode")
      with open(CSV_REPORT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([model, scan_mode, "vllm_decode", run_name, status])

  print("\n" + "="*80)
  print("VLLM VALIDATION SUMMARY")
  print("="*80)
  with open(CSV_REPORT, "r") as f:
      print(f.read())
  print("="*80)

  if not all_passed:
      sys.exit(1)

if __name__ == "__main__":
  run_matrix()
