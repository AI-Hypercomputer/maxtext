import json
import os
import shutil
import subprocess

def create_vllm_config():
    # Based on deepseek3_tiny_dict in src/maxtext/checkpoint_conversion/utils/hf_model_configs.py
    # or src/maxtext/configs/models/deepseek3-tiny.yml
    config_dict = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "attention_type": "vllm_rpa",
        "attention": "vllm_rpa",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "ep_size": 1,
        "first_k_dense_replace": 3,
        "hidden_act": "silu",
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 64,
        "kv_lora_rank": 16,
        "max_position_embeddings": 163840,
        "model_type": "deepseek_v3",
        "moe_intermediate_size": 64,
        "moe_layer_freq": 1,
        "n_group": 8,
        "n_routed_experts": 16,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 4,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 61,
        "num_key_value_heads": 4,
        "num_nextn_predict_layers": 1,
        "q_lora_rank": 32,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "rope_theta": 10000,
            "type": "yarn",
        },
        "rope_theta": 10000,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "tie_word_embeddings": False,
        "topk_group": 4,
        "topk_method": "noaux_tc",
        "torch_dtype": "bfloat16",
        "transformers_version": "4.33.1",
        "use_cache": True,
        "v_head_dim": 128,
        "vocab_size": 129280,
    }
    
    vllm_dir = "/tmp/deepseek3-tiny-vllm"
    os.makedirs(vllm_dir, exist_ok=True)
    
    with open(os.path.join(vllm_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # We need a tokenizer. We can use a small one or copy from another model.
    # Since we are in a TPU environment, there should be some tokenizers around.
    # I'll try to find one.
    src_tokenizer_dir = "/mnt/workspace/maxtext/src/maxtext/assets/tokenizers/qwen3-tokenizer"
    if os.path.exists(src_tokenizer_dir):
        for item in os.listdir(src_tokenizer_dir):
            s = os.path.join(src_tokenizer_dir, item)
            d = os.path.join(vllm_dir, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
    
    print(f"vLLM config created at {vllm_dir}")
    return vllm_dir

def run_benchmark(vllm_dir):
    python_path = "/mnt/workspace/max_venv/bin/python3"
    cmd = [
        python_path, "src/maxtext/integration/tunix/weight_mapping/bench_weight_sync.py",
        "--model_name=deepseek3-tiny",
        f"--vllm_model_id={vllm_dir}",
        "--rand_init=True",
        "--ici_fsdp_parallelism=8",
        "--ici_tensor_parallelism=1",
        "--rollout_tensor_parallelism=1"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    vllm_dir = create_vllm_config()
    run_benchmark(vllm_dir)
