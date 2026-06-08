import yaml
from types import SimpleNamespace

# Let's mock just enough to print the mapping
import sys
sys.path.append('src')

from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING
from maxtext.configs import types

class MockConfig:
    def __init__(self):
        self.model_name = "gemma3-4b"
        self.use_multimodal = False
        self.scan_layers = False
        self.num_experts = 0

config = MockConfig()
hf_config_dict = {"text_config": {}, "vision_config": {}}
param_map_mt_to_hf = PARAM_MAPPING[config.model_name](hf_config_dict, config, config.scan_layers)

print(param_map_mt_to_hf.get("params-decoder-decoder_norm-scale"))
