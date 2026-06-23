import subprocess
import json
import os
import sys
import argparse
from src.maxtext.experimental.agent.checkpoint_validation_agent.model_registry import get_model_config

def validate_checkpoint(model_name, override_length=None):
    # Fetch configuration dynamically
    config = get_model_config(model_name)
    target_length = override_length or config['max_target_length'] #override if provided, else use registry default
    print(f"Validating {model_name} with parameters at: {config['load_parameters_path']}")
    
    #run a smoke test using decode.py to check if model can load and initialize layers
    command = [
        "python3", "src/maxtext/inference/decode.py", "src/maxtext/configs/base.yml",
        f"load_parameters_path={config['load_parameters_path']}",
        f"model_name={config['maxtext_model_name']}",
        f"tokenizer_path={config['tokenizer_path']}",
        f"scan_layers={config['scan_layers']}",
        f"max_target_length={target_length}"
    ]
    
    #capture terminal printouts
    result = subprocess.run(command, capture_output=True, text=True)

    #create the report
    report = {
        "model": model_name,
        "success": result.returncode == 0, #if returncode is 0, command worked
        "stderr": result.stderr if result.returncode != 0 else "Success" #store error message if there's a failure
    }
    
    report_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(report_dir, exist_ok=True)
    output_path = os.path.join(report_dir, f"report_{model_name}.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Report saved to {output_path}")

#script runs only if called directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate MaxText Checkpoints")
    parser.add_argument("model_name", help="Model key in registry")
    parser.add_argument("--max_target_length", type=int, help="Override target length")
    args = parser.parse_args()

    try:
        validate_checkpoint(args.model_name, args.max_target_length)
    except Exception as e:
        print(f"FAILED: {e}")