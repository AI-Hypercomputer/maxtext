import os
import sys
import runpy

sys.path.insert(0, os.getcwd())
os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")

sys.argv = [
    "forward_pass_logit_checker.py",
    "src/maxtext/configs/base.yml",
    "model_name=qwen3.5-35b-a3b-fp8",
    "load_parameters_path=/dev/shm/maxtext_qwen3.5_35b_fp8/0/items",
    "scan_layers=true",
    "per_device_batch_size=1",
    "max_prefill_predict_length=4",
    "max_target_length=4",
    "async_checkpointing=false",
    "sparse_matmul=false",
    "ici_fsdp_parallelism=1",
    "ici_expert_parallelism=-1",
    "matmul_precision=highest",
    "float32_logits=true",
    "float32_qk_product=true",
    "--golden_logits_path=/dev/shm/golden_qwen3.5_35b_fp8.pkl",
    "--max_kl_div=0.2",
]

print("Launching forward pass logit test against /dev/shm/golden_qwen3.5_35b_fp8.pkl with --max_kl_div=0.2...")
runpy.run_module("tests.utils.forward_pass_logit_checker", run_name="__main__")
