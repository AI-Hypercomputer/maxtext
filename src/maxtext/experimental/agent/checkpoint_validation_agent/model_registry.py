import os

# Base path for checkpoints. should be strictly provided by the environment
BASE_PATH = os.getenv("MAXTEXT_CHECKPOINT_DIR")

#stop program if user didn't set the variable
if not BASE_PATH:
    raise EnvironmentError("MAXTEXT_CHECKPOINT_DIR not set.")

#store metadata for supported models
MODEL_REGISTRY = {
    "qwen3-4b-unscanned": {
        "maxtext_model_name": "qwen3-4b",
        "checkpoint_dir": "MaxText-Qwen3-4B-Unscanned", 
        "tokenizer_path": "Qwen/Qwen3-4B",
        "max_target_length": 2048,
        "scan_layers": False
    },
    #more will be added as progress is made
}

#used by main.py to get info
def get_model_config(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not registered.")
    
    #copy of registry to prevent accidental edits 
    configuration = MODEL_REGISTRY[model_name].copy()
    configuration["load_parameters_path"] = os.path.join(BASE_PATH, configuration["checkpoint_dir"], "0/items") #path to orbax weights
    return configuration