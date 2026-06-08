import sys
sys.path.append('src')

from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING
from maxtext.checkpoint_conversion.to_maxtext import convert_hf_lora_key_to_maxtext

import types

def mock_get_tensor(key):
    print("GETTER CALLED WITH:", key)

config = types.SimpleNamespace()
config.model_name = 'gemma3-4b'
config.scan_layers = False
config.use_multimodal = False

# ... Actually I can just mock out everything or just run the test directly?
