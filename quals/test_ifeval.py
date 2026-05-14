import argparse
import os
import re
import subprocess
import sys

try:
    from instruction_following_eval import get_examples, evaluate_instruction_following
except ImportError:
    print("Error: instruction_following_eval package is not installed.", file=sys.stderr)
    print("Please install it using: uv pip install git+https://github.com/josejg/instruction_following_eval.git", file=sys.stderr)
    sys.exit(1)


def decode_prompt(prompt: str, checkpoint_path: str, use_tunix: bool = False) -> str:
    """Runs maxtext.inference.vllm_decode as an isolated subprocess using environment variables to bypass YAML parsing."""
    cmd = [
        "python3", "-m", "maxtext.inference.vllm_decode",
        "model_name=qwen2.5-1.5b",
        "tokenizer_path=Qwen/Qwen2.5-1.5B-Instruct",
        f"load_parameters_path={checkpoint_path}",
        "vllm_hf_overrides={\"architectures\": [\"MaxTextForCausalLM\"]}",
        "ici_tensor_parallelism=1",
        "hbm_utilization_vllm=0.5",
        "decode_sampling_temperature=0.0",
        "decode_sampling_nucleus_p=1.0",
        "decode_sampling_top_k=-1",
        "use_chat_template=True",
        "scan_layers=False"
    ]

    if use_tunix:
        cmd.append("--use_tunix=True")

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    # Pass prompt cleanly via environment variable to eliminate all OmegaConf YAML dotlist parser collisions
    env["M_PROMPT"] = prompt

    print(f"Running vllm_decode (use_tunix={use_tunix}) for prompt: {prompt[:60]}...", flush=True)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    combined_output = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        print(f"Error running vllm_decode subprocess:\n{result.stderr}", file=sys.stderr)

    # Find the last logged output block to correctly capture single or multiline output strings
    last_idx = max(combined_output.rfind("Output: "), combined_output.rfind("Generated text: "))
    if last_idx == -1:
        print("Warning: Could not locate Generated text or Output markers in decoder output trace.", file=sys.stderr)
        return ""

    prefix_len = 8 if combined_output[last_idx:].startswith("Output: ") else 16
    return combined_output[last_idx + prefix_len:].strip()


def main():
    parser = argparse.ArgumentParser(description="Verify IFEval framework integration using robust isolated vLLM subprocess decoding.")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="gs://igorts_europe/converted_checkpoints/qwen2.5-1.5b-instruct_mcjax_v2/0/items",
        help="Path to the MaxText parameter checkpoint."
    )
    parser.add_argument(
        "--max_examples", 
        type=int, 
        default=10, 
        help="Maximum number of examples to evaluate for scoring."
    )
    parser.add_argument(
        "--use_tunix", 
        action="store_true", 
        help="Enable native Tunix weight loading adapter layer during decoding."
    )
    args = parser.parse_args()

    examples = get_examples()
    if args.max_examples > 0:
        examples = examples[:args.max_examples]

    print(f"Starting robust isolated evaluation on {len(examples)} IFEval examples using checkpoint:\n{args.checkpoint}\n", flush=True)

    responses = []
    for idx, example in enumerate(examples):
        print(f"--- Example {idx + 1}/{len(examples)} ---", flush=True)
        prompt = example["prompt"]
        
        response = decode_prompt(prompt, args.checkpoint, use_tunix=args.use_tunix)
        print(f"Response: {response}\n", flush=True)
        responses.append(response)

    print("Computing instruction following metrics...", flush=True)
    metrics = evaluate_instruction_following(examples, responses)

    print("\n# IFEval Verification Results")
    for metric_name, value in metrics.items():
        print(f"- **{metric_name}**: {value:.4f}")


if __name__ == "__main__":
    main()
