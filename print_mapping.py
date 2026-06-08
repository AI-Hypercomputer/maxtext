import sys
sys.path.append('src')
from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING
config = {'text_config': {'num_hidden_layers': 2}, 'vision_config': {'num_hidden_layers': 2}, 'use_multimodal': False}
import types
class DummyConfig:
    pass
maxtext_config = DummyConfig()
maxtext_config.inhomogeneous_layer_cycle_interval = 6
print(PARAM_MAPPING['gemma3-4b'](config, maxtext_config, False).get('params-decoder-decoder_norm-scale'))
