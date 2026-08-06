"""Standalone script to run Section 1 (vLLM Rollout & Routed Experts Extraction) in an isolated process."""
import os
import json
import numpy as np
from vllm import LLM, SamplingParams

def main():
    print("=" * 110)
    print("1. RUNNING REAL VLLM INFERENCE IN ISOLATED PROCESS TO EXTRACT ROUTED EXPERTS")
    print("=" * 110)

    prompt = "The capital of France is Paris and it is known for"
    unscanned_ckpt_path = "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items"

    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    vllm_engine = LLM(
        model="Qwen/Qwen3.5-35B-A3B",
        trust_remote_code=True,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=16,
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        enable_expert_parallel=False,
        enable_return_routed_experts=True,
        gpu_memory_utilization=0.9,
        hf_overrides={"architectures": ["MaxTextForCausalLM"]},
        additional_config={
            "maxtext_config": {
                "model_name": "qwen3.5-35b-a3b",
                "load_parameters_path": unscanned_ckpt_path,
                "scan_layers": False,
                "weight_dtype": "bfloat16",
                "attention": "vllm_rpa",
                "enable_nnx": True,
                "pure_nnx_decoder": True,
                "allow_split_physical_axes": True,
                "use_multimodal": False,
                "prefuse_moe_weights": True,
            }
        },
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    outputs = vllm_engine.generate([prompt], sampling_params)
    output = outputs[0].outputs[0]

    real_vllm_routed_experts = output.routed_experts
    prompt_token_ids = outputs[0].prompt_token_ids
    generated_token_ids = list(output.token_ids)

    if real_vllm_routed_experts is not None:
        np.save("/tmp/vllm_routed_experts.npy", np.array(real_vllm_routed_experts, dtype=np.int32))
        print(f"Saved /tmp/vllm_routed_experts.npy shape: {real_vllm_routed_experts.shape}")

    token_data = {
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
    }
    with open("/tmp/vllm_tokens.json", "w") as f:
        json.dump(token_data, f)
    print("Saved /tmp/vllm_tokens.json")

    vllm_engine.llm_engine.engine_core.shutdown()
    print("Section 1 process completed successfully!")

if __name__ == "__main__":
    main()
    os._exit(0)

