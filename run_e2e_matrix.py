"""Drives the end-to-end checkpoint validation matrix across models and scan modes.

Each combination trains a base checkpoint and then reloads it from every trainer, so a
checkpoint written by one is proven readable by the others. Results land in a CSV.
"""

import os
import subprocess
import csv

from maxtext.utils.globals import HF_IDS

MODELS = [
    "llama3.1-8b",
    # "gemma3-4b",
    # "gemma2-2b",
    # "gemma4-e2b",
    # "qwen2.5-1.5b",
    # "qwen3-0.6b",
    # "olmo3-7b",
    # "gpt-oss-20b",
]
SCAN_MODES = ["scanned", "unscanned"]

GCS_BASE = "gs://mesa-maxtext/validation_runs/post_train_layout_v21"
HF_BASE = "gs://mesa-maxtext/huggingface_transformers"
LOCAL_LOGS = "local_logs"
CSV_REPORT = "validation_summary.csv"


def get_tokenizer_flags(model_name):
  """Returns the tokenizer config flags a model needs, so no job falls back to a missing default."""
  flags = []
  if "gemma4" in model_name:
    flags.extend(["tokenizer_path=src/maxtext/assets/tokenizers/tokenizer_gemma4.model", "tokenizer_type=sentencepiece"])
  elif "gemma3" in model_name:
    flags.extend(["tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.gemma3", "tokenizer_type=sentencepiece"])
  elif "gemma" in model_name:
    flags.extend(["tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.gemma", "tokenizer_type=sentencepiece"])
  elif "llama" in model_name:
    flags.extend(["tokenizer_path=src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken", "tokenizer_type=tiktoken"])
  elif "mistral" in model_name:
    flags.extend(["tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.mistral-v3", "tokenizer_type=sentencepiece"])
  elif "qwen" in model_name or "olmo" in model_name or "gpt-oss" in model_name:
    # These have no tokenizer asset of their own -- the path built from the model name does not
    # exist, and a huggingface tokenizer_type then reads it as a repo id and fails. Name the repo.
    flags.extend([f"tokenizer_path={HF_IDS[model_name]}", "tokenizer_type=huggingface"])
  return flags


def execute_command(cmd, log_path):
  """Runs one job to completion, tee-ing its output to `log_path`.

  Args:
    cmd: Argument list to run.
    log_path: File to write the command and its combined output to.

  Returns:
    Whether the job exited zero.
  """
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  env = os.environ.copy()

  cmd_str = " ".join(cmd)
  print(f"\n[EXECUTING]: {cmd_str}")
  print(f"[LOG PATH]: {log_path}")

  with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"Command: {cmd_str}\n\n")
    f.flush()
    with subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env) as process:
      process.wait()

  if process.returncode != 0:
    print(f"[ERROR] Job failed. Check logs at: {log_path}")
  return process.returncode == 0


def run_matrix():
  """Runs every model and scan mode through the matrix, recording each job's result."""
  with open(CSV_REPORT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Scan Mode", "Phase", "Action", "Run Name", "Status"])

  # The helpers below close over the loop variables, but each is only called within the
  # iteration that defines it, so the late-binding pylint warns about cannot happen.
  # pylint: disable=cell-var-from-loop
  for model in MODELS:
    for scan_mode in SCAN_MODES:
      print(f"\n{'='*80}\nStarting matrix for Model: {model} | Scan Mode: {scan_mode}\n{'='*80}")
      scan_bool = "True" if scan_mode == "scanned" else "False"
      hf_ckpt = f"{HF_BASE}/{model}/to_maxtext/{scan_mode}/0/items"

      def record_result(phase, action, run_name, success):
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {model} | {scan_mode} | {action} | {run_name}")
        with open(CSV_REPORT, "a", newline="", encoding="utf-8") as f:
          writer = csv.writer(f)
          writer.writerow([model, scan_mode, phase, action, run_name, status])

      def get_ckpt_path(action, run_name):
        if action == "rl":
          step = 1
        else:
          step = 9
        return f"{GCS_BASE}/{scan_mode}/{action}/{model}/{run_name}/checkpoints/{step}/items"

      def build_cmd(script, action, run_name, load_path, extra_flags=None):
        config = "src/maxtext/configs/post_train/rl.yml" if action == "rl" else "src/maxtext/configs/base.yml"
        cmd = [
            "python",
            script,
            config,
            f"run_name={run_name}",
            f"model_name={model}",
            f"scan_layers={scan_bool}",
            "async_checkpointing=False",
            "save_checkpoint_on_start=False",
            f"base_output_directory={GCS_BASE}/{scan_mode}/{action}/{model}",
        ]
        # Handle steps vs num_batches for RL
        if action == "rl":
          cmd.extend(["num_batches=2", "rl.num_generations=2"])
          if "gemma3" in model:
            cmd.append("chat_template_path=src/maxtext/examples/chat_templates/gemma-3-27b-chat_template.json")
            hf_model_name = f"google/gemma-3-{model.rsplit('-', maxsplit=1)[-1]}-it"
            cmd.append(f"vllm_hf_config_path={hf_model_name}")
          else:
            # RL needs a chat template, and the base checkpoints are not all instruction tuned,
            # so their tokenizers do not ship one. The round trip under test does not care which
            # template it is.
            cmd.append("chat_template_path=src/maxtext/examples/chat_templates/gsm8k_rl.json")
        else:
          cmd.extend(["steps=10", "per_device_batch_size=1"])

        # Handle datasets: DPO needs HF dataset, others can use synthetic to bypass tokenization
        if action == "dpo":
          cmd.extend(
              [
                  "dataset_type=hf",
                  "hf_path=json",
                  "hf_train_files=tests/assets/local_datasets/dpo/dpo_3_column_dataset.json",
                  "train_data_columns=\"['prompt', 'chosen', 'rejected']\"",
                  "tokenize_train_data=True",
                  "use_dpo=True",
                  "packing=False",
              ]
          )
        elif action != "rl":
          cmd.extend(["dataset_type=synthetic"])

        # Inject tokenizers for ALL jobs to prevent missing tokenizer config errors
        cmd.extend(get_tokenizer_flags(model))

        if action == "rl":
          # RL samples through vLLM, which wants the HuggingFace model, and applies a chat
          # template, which MaxText's own tokenizer assets do not carry. Name the repo for both.
          if "llama3" in model:
            hf_model_name = f"meta-llama/Meta-Llama-3.1-{model.rsplit('-', maxsplit=1)[-1].upper()}-Instruct"
          else:
            hf_model_name = HF_IDS[model]
          cmd = [c for c in cmd if not c.startswith("tokenizer_path=") and not c.startswith("tokenizer_type=")]
          cmd.extend([f"tokenizer_path={hf_model_name}", "tokenizer_type=huggingface"])
          if not any(c.startswith("vllm_hf_config_path=") for c in cmd):
            cmd.append(f"vllm_hf_config_path={hf_model_name}")
          if not any(c.startswith("vllm_hf_overrides=") for c in cmd):
            cmd.append('vllm_hf_overrides={architectures: ["MaxTextForCausalLM"]}')
          if not any(c.startswith("vllm_additional_config=") for c in cmd):
            cmd.append(f'vllm_additional_config={{"maxtext_config": {{"model_name": "{model}"}}}}')

        if action == "dpo":
          # DPO reads dataset_type=hf, whose pipeline tokenizes through AutoTokenizer. That cannot
          # open MaxText's own tokenizer assets, so name the HF repo for the model instead.
          cmd = [c for c in cmd if not c.startswith("tokenizer_path=") and not c.startswith("tokenizer_type=")]
          cmd.extend([f"tokenizer_path={HF_IDS[model]}", "tokenizer_type=huggingface"])

        if load_path:
          cmd.append(f"load_parameters_path={load_path}")
        if ("gemma" in model and scan_bool == "True") and (load_path and HF_BASE in load_path):
          cmd.append("use_standalone_converter=True")

        if extra_flags:
          cmd.extend(extra_flags)

        return cmd

      # Script paths
      pre_train_script = "src/maxtext/trainers/pre_train/train.py"
      sft_script = "src/maxtext/trainers/post_train/sft/train_sft.py"
      dpo_script = "src/maxtext/trainers/post_train/dpo/train_dpo.py"
      rl_script = "src/maxtext/trainers/post_train/rl/train_rl.py"
      distill_script = "src/maxtext/trainers/post_train/distillation/train_distill.py"

      # ---------------------------------------------------------
      # Phase A: Pre-train -> Post-train Reloading
      # ---------------------------------------------------------

      # 1. Generate Pre-train Base
      run_name = "ckpt_pretrain_base"
      cmd = build_cmd(pre_train_script, "pre_train", run_name, None)
      log_path = f"{LOCAL_LOGS}/{scan_mode}/pre_train/{model}/{run_name}.log"
      success = execute_command(cmd, log_path)
      record_result("Phase A", "pre_train", run_name, success)
      pt_ckpt = get_ckpt_path("pre_train", run_name)

      if success:
        # 2. Pre-train -> SFT Reload
        run_name = "sft_reload_pt"
        cmd = build_cmd(sft_script, "sft", run_name, pt_ckpt)
        log_path = f"{LOCAL_LOGS}/{scan_mode}/sft/{model}/{run_name}.log"
        success = execute_command(cmd, log_path)
        record_result("Phase A", "sft", run_name, success)

        # 3. Pre-train -> DPO Reload
        run_name = "dpo_reload_pt"
        cmd = build_cmd(dpo_script, "dpo", run_name, pt_ckpt)
        log_path = f"{LOCAL_LOGS}/{scan_mode}/dpo/{model}/{run_name}.log"
        success = execute_command(cmd, log_path)
        record_result("Phase A", "dpo", run_name, success)

        # 4. Pre-train -> RL Reload
        run_name = "rl_reload_pt"
        cmd = build_cmd(rl_script, "rl", run_name, pt_ckpt)
        log_path = f"{LOCAL_LOGS}/{scan_mode}/rl/{model}/{run_name}.log"
        success = execute_command(cmd, log_path)
        record_result("Phase A", "rl", run_name, success)

      # ---------------------------------------------------------
      # Phase B: Post-train -> Pre-train / Self Reloading
      # ---------------------------------------------------------

      # SFT Validations
      run_name = "ckpt_sft_base"
      cmd = build_cmd(sft_script, "sft", run_name, hf_ckpt)
      log_path = f"{LOCAL_LOGS}/{scan_mode}/sft/{model}/{run_name}.log"
      success = execute_command(cmd, log_path)
      record_result("Phase B", "sft", run_name, success)
      sft_ckpt = get_ckpt_path("sft", run_name)

      if success:
        rn = "sft_intra_reload"
        success2 = execute_command(
            build_cmd(sft_script, "sft", rn, sft_ckpt), f"{LOCAL_LOGS}/{scan_mode}/sft/{model}/{rn}.log"
        )
        record_result("Phase B", "sft", rn, success2)

        rn = "sft_cross_reload"
        success2 = execute_command(
            build_cmd(pre_train_script, "pre_train", rn, sft_ckpt), f"{LOCAL_LOGS}/{scan_mode}/pre_train/{model}/{rn}.log"
        )
        record_result("Phase B", "pre_train", rn, success2)

      # DPO Validations
      run_name = "ckpt_dpo_base"
      cmd = build_cmd(dpo_script, "dpo", run_name, hf_ckpt)
      log_path = f"{LOCAL_LOGS}/{scan_mode}/dpo/{model}/{run_name}.log"
      success = execute_command(cmd, log_path)
      record_result("Phase B", "dpo", run_name, success)
      dpo_ckpt = get_ckpt_path("dpo", run_name)

      if success:
        rn = "dpo_intra_reload"
        success2 = execute_command(
            build_cmd(dpo_script, "dpo", rn, dpo_ckpt), f"{LOCAL_LOGS}/{scan_mode}/dpo/{model}/{rn}.log"
        )
        record_result("Phase B", "dpo", rn, success2)

        rn = "dpo_cross_reload"
        success2 = execute_command(
            build_cmd(pre_train_script, "pre_train", rn, dpo_ckpt), f"{LOCAL_LOGS}/{scan_mode}/pre_train/{model}/{rn}.log"
        )
        record_result("Phase B", "pre_train", rn, success2)

      # RL Validations
      run_name = "ckpt_rl_base"
      cmd = build_cmd(rl_script, "rl", run_name, hf_ckpt)
      log_path = f"{LOCAL_LOGS}/{scan_mode}/rl/{model}/{run_name}.log"
      success = execute_command(cmd, log_path)
      record_result("Phase B", "rl", run_name, success)
      rl_ckpt = get_ckpt_path("rl", run_name)

      if success:
        rn = "rl_intra_reload"
        success2 = execute_command(
            build_cmd(rl_script, "rl", rn, rl_ckpt), f"{LOCAL_LOGS}/{scan_mode}/rl/{model}/{rn}.log"
        )
        record_result("Phase B", "rl", rn, success2)

        rn = "rl_cross_reload"
        success2 = execute_command(
            build_cmd(pre_train_script, "pre_train", rn, rl_ckpt), f"{LOCAL_LOGS}/{scan_mode}/pre_train/{model}/{rn}.log"
        )
        record_result("Phase B", "pre_train", rn, success2)

      # Distill Validations
      run_name = "ckpt_distill_base"
      distill_flags = [
          f"teacher_overrides.load_parameters_path={hf_ckpt}",
          "teacher_overrides.model_name=" + model,
          f"teacher_overrides.scan_layers={scan_bool}",
          "teacher_overrides.per_device_batch_size=1",
      ]
      cmd = build_cmd(distill_script, "distill", run_name, hf_ckpt, distill_flags)
      log_path = f"{LOCAL_LOGS}/{scan_mode}/distill/{model}/{run_name}.log"
      success = execute_command(cmd, log_path)
      record_result("Phase B", "distill", run_name, success)
      distill_ckpt = get_ckpt_path("distill", run_name)

      if success:
        rn = "distill_intra_reload"
        distill_flags2 = [
            f"teacher_overrides.load_parameters_path={hf_ckpt}",  # teacher loads base again
            "teacher_overrides.model_name=" + model,
            f"teacher_overrides.scan_layers={scan_bool}",
            "teacher_overrides.per_device_batch_size=1",
        ]
        success2 = execute_command(
            build_cmd(distill_script, "distill", rn, distill_ckpt, distill_flags2),
            f"{LOCAL_LOGS}/{scan_mode}/distill/{model}/{rn}.log",
        )
        record_result("Phase B", "distill", rn, success2)

        rn = "distill_cross_reload"
        success2 = execute_command(
            build_cmd(pre_train_script, "pre_train", rn, distill_ckpt),
            f"{LOCAL_LOGS}/{scan_mode}/pre_train/{model}/{rn}.log",
        )
        record_result("Phase B", "pre_train", rn, success2)


if __name__ == "__main__":
  run_matrix()
