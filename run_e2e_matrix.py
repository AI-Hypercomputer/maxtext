import os
import subprocess
import csv
import sys

MODELS = [
    "gemma3-4b", 
    # "gemma2-2b", 
    # "gemma4-e2b", 
    # "qwen2.5-1.5b", 
    # "qwen3-0.6b", 
    "llama3.1-8b", 
    # "olmo3-7b", 
    # "gpt-oss-20b"
]
SCAN_MODES = ["scanned", "unscanned"]

GCS_BASE = "gs://mesa-maxtext/validation_runs/post_train_layout_v11"
HF_BASE = "gs://mesa-maxtext/huggingface_transformers"
LOCAL_LOGS = "local_logs"
CSV_REPORT = "validation_summary.csv"

def get_tokenizer_flags(model_name):
    flags = []
    if "gemma4" in model_name:
        flags.extend(["tokenizer_path=src/maxtext/assets/tokenizers/tokenizer_gemma4.model", "tokenizer_type=sentencepiece"])
    elif "gemma3" in model_name:
        flags.extend(["tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.gemma3", "tokenizer_type=sentencepiece"])
    elif "gemma" in model_name:
        flags.extend(["tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.gemma", "tokenizer_type=sentencepiece"])
    elif "llama" in model_name:
        flags.extend(["tokenizer_path=src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken", "tokenizer_type=tiktoken"])
    elif "qwen" in model_name or "olmo" in model_name or "gpt-oss" in model_name:
        flags.extend([f"tokenizer_path=src/maxtext/assets/tokenizers/{model_name}", "tokenizer_type=huggingface"])
    return flags

def execute_command(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = os.environ.copy()
    
    cmd_str = " ".join(cmd)
    print(f"\n[EXECUTING]: {cmd_str}")
    print(f"[LOG PATH]: {log_path}")
    
    with open(log_path, "w") as f:
        f.write(f"Command: {cmd_str}\n\n")
        f.flush()
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        process.wait()
        
    if process.returncode != 0:
        print(f"[ERROR] Job failed. Check logs at: {log_path}")
    return process.returncode == 0

def run_matrix():
    with open(CSV_REPORT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Scan Mode", "Phase", "Action", "Run Name", "Status"])
        
    for model in MODELS:
        for scan_mode in SCAN_MODES:
            print(f"\n{'='*80}\nStarting matrix for Model: {model} | Scan Mode: {scan_mode}\n{'='*80}")
            scan_bool = "True" if scan_mode == "scanned" else "False"
            hf_ckpt = f"{HF_BASE}/{model}/to_maxtext/{scan_mode}/0/items"
            
            def record_result(phase, action, run_name, success):
                status = "PASS" if success else "FAIL"
                print(f"[{status}] {model} | {scan_mode} | {action} | {run_name}")
                with open(CSV_REPORT, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([model, scan_mode, phase, action, run_name, status])
                    
            def get_ckpt_path(action, run_name):
                step = "0" if action in ("sft", "distill") else "1"
                return f"{GCS_BASE}/{scan_mode}/{action}/{model}/{run_name}/checkpoints/{step}/items"

            def build_cmd(script, action, run_name, load_path, extra_flags=None):
                config = "src/maxtext/configs/post_train/rl.yml" if action == "rl" else "src/maxtext/configs/base.yml"
                cmd = [
                    "python", script, config,
                    f"run_name={run_name}",
                    f"model_name={model}",
                    f"scan_layers={scan_bool}",
                    f"base_output_directory={GCS_BASE}/{scan_mode}/{action}/{model}",
                    "checkpoint_period=1"
                ]
                # Handle steps vs num_batches for RL
                if action == "rl":
                    cmd.extend([
                        "num_batches=2", 
                        "rl.num_generations=2"
                    ])
                    if "gemma3" in model:
                        cmd.append("chat_template_path=src/maxtext/examples/chat_templates/gemma-3-27b-chat_template.json")
                        hf_model_name = f"google/gemma-3-{model.split('-')[-1]}-it"
                        cmd.append(f"vllm_hf_config_path={hf_model_name}")
                else:
                    cmd.extend(["steps=5", "per_device_batch_size=1"])

                # Handle datasets: DPO needs HF dataset, others can use synthetic to bypass tokenization
                if action == "dpo":
                    cmd.extend([
                        "dataset_type=hf",
                        "hf_path=json",
                        "hf_train_files=tests/assets/local_datasets/dpo/dpo_3_column_dataset.json",
                        "train_data_columns=\"['prompt', 'chosen', 'rejected']\"",
                        "tokenize_train_data=True",
                        "use_dpo=True",
                        "packing=False"
                    ])
                elif action != "rl":
                    cmd.extend(["dataset_type=synthetic"])

                # Inject tokenizers for ALL jobs to prevent missing tokenizer config errors
                cmd.extend(get_tokenizer_flags(model))

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
                success2 = execute_command(build_cmd(sft_script, "sft", rn, sft_ckpt), f"{LOCAL_LOGS}/{scan_mode}/sft/{model}/{rn}.log")
                record_result("Phase B", "sft", rn, success2)

                rn = "sft_cross_reload"
                success2 = execute_command(build_cmd(pre_train_script, "pre_train", rn, sft_ckpt), f"{LOCAL_LOGS}/{scan_mode}/pre_train/{model}/{rn}.log")
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
                success2 = execute_command(build_cmd(dpo_script, "dpo", rn, dpo_ckpt), f"{LOCAL_LOGS}/{scan_mode}/dpo/{model}/{rn}.log")
                record_result("Phase B", "dpo", rn, success2)

                rn = "dpo_cross_reload"
                success2 = execute_command(build_cmd(pre_train_script, "pre_train", rn, dpo_ckpt), f"{LOCAL_LOGS}/{scan_mode}/pre_train/{model}/{rn}.log")
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
                success2 = execute_command(build_cmd(rl_script, "rl", rn, rl_ckpt), f"{LOCAL_LOGS}/{scan_mode}/rl/{model}/{rn}.log")
                record_result("Phase B", "rl", rn, success2)

                rn = "rl_cross_reload"
                success2 = execute_command(build_cmd(pre_train_script, "pre_train", rn, rl_ckpt), f"{LOCAL_LOGS}/{scan_mode}/pre_train/{model}/{rn}.log")
                record_result("Phase B", "pre_train", rn, success2)

            # Distill Validations
            run_name = "ckpt_distill_base"
            distill_flags = [
                f"teacher_overrides.load_parameters_path={hf_ckpt}",
                "teacher_overrides.model_name=" + model,
                f"teacher_overrides.scan_layers={scan_bool}",
                "teacher_overrides.per_device_batch_size=1"
            ]
            cmd = build_cmd(distill_script, "distill", run_name, hf_ckpt, distill_flags)
            log_path = f"{LOCAL_LOGS}/{scan_mode}/distill/{model}/{run_name}.log"
            success = execute_command(cmd, log_path)
            record_result("Phase B", "distill", run_name, success)
            distill_ckpt = get_ckpt_path("distill", run_name)

            if success:
                rn = "distill_intra_reload"
                distill_flags2 = [
                    f"teacher_overrides.load_parameters_path={hf_ckpt}", # teacher loads base again
                    "teacher_overrides.model_name=" + model,
                    f"teacher_overrides.scan_layers={scan_bool}",
                    "teacher_overrides.per_device_batch_size=1"
                ]
                success2 = execute_command(build_cmd(distill_script, "distill", rn, distill_ckpt, distill_flags2), f"{LOCAL_LOGS}/{scan_mode}/distill/{model}/{rn}.log")
                record_result("Phase B", "distill", rn, success2)

                rn = "distill_cross_reload"
                success2 = execute_command(build_cmd(pre_train_script, "pre_train", rn, distill_ckpt), f"{LOCAL_LOGS}/{scan_mode}/pre_train/{model}/{rn}.log")
                record_result("Phase B", "pre_train", rn, success2)

if __name__ == "__main__":
    run_matrix()
