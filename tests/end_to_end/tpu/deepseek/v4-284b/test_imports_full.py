import sys
import os

# Ensure maxtext is in path
sys.path.insert(0, os.getcwd() + "/src")
sys.path.insert(0, os.getcwd())

print("1. Importing basic Python modules...")
sys.stdout.flush()
import argparse
from functools import partial
import json
import threading
import time
from typing import Any, Callable, List, Sequence
print("1. Done!")
sys.stdout.flush()

print("2. Importing absl...")
sys.stdout.flush()
import absl
print("2. Done!")
sys.stdout.flush()

print("3. Importing ml_dtypes...")
sys.stdout.flush()
import ml_dtypes
print("3. Done!")
sys.stdout.flush()

print("4. Importing torch...")
sys.stdout.flush()
import torch
print("4. Done!")
sys.stdout.flush()

print("5. Importing jax...")
sys.stdout.flush()
import jax
print("5. Done!")
sys.stdout.flush()

print("6. Importing flax...")
sys.stdout.flush()
import flax.linen as nn
print("6. Done!")
sys.stdout.flush()

print("7. Importing huggingface_hub...")
sys.stdout.flush()
from huggingface_hub import hf_hub_download, list_repo_files
print("7. Done!")
sys.stdout.flush()

print("8. Importing maxtext.configs...")
sys.stdout.flush()
from maxtext.configs import pyconfig
from maxtext.configs.types import DType
print("8. Done!")
sys.stdout.flush()

print("9. Importing maxtext.common.common_types...")
sys.stdout.flush()
from maxtext.common.common_types import MODEL_MODE_TRAIN
print("9. Done!")
sys.stdout.flush()

print("10. Importing maxtext.checkpoint_conversion.utils...")
sys.stdout.flush()
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from maxtext.checkpoint_conversion.utils.tensor_handling import apply_hook_fns
from maxtext.checkpoint_conversion.utils.utils import MemoryMonitorTqdm, load_hf_dict_from_transformers, load_hf_dict_from_safetensors, param_key_parts_from_path, print_peak_memory, print_ram_usage, save_weights_to_checkpoint, validate_and_filter_param_map_keys
print("10. Done!")
sys.stdout.flush()

print("11. Importing maxtext.inference.inference_utils...")
sys.stdout.flush()
from maxtext.inference.inference_utils import str2bool
print("11. Done!")
sys.stdout.flush()

print("12. Importing maxtext.layers...")
sys.stdout.flush()
from maxtext.layers import quantizations
print("12. Done!")
sys.stdout.flush()

print("13. Importing maxtext.models...")
sys.stdout.flush()
from maxtext.models import models
print("13. Done!")
sys.stdout.flush()

print("14. Importing maxtext.utils...")
sys.stdout.flush()
from maxtext.utils import max_logging, max_utils, maxtext_utils
from maxtext.utils.globals import HF_IDS
print("14. Done!")
sys.stdout.flush()

print("15. Importing numpy...")
sys.stdout.flush()
import numpy as np
print("15. Done!")
sys.stdout.flush()

print("16. Importing orbax...")
sys.stdout.flush()
from orbax.checkpoint import type_handlers
print("16. Done!")
sys.stdout.flush()

print("17. Importing safetensors...")
sys.stdout.flush()
from safetensors import safe_open
print("17. Done!")
sys.stdout.flush()

print("All imports completed successfully!")
sys.stdout.flush()
