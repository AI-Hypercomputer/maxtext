import sys
from unittest.mock import MagicMock
sys.modules['jax'] = MagicMock()
sys.modules['jax.sharding'] = MagicMock()
sys.modules['jax.numpy'] = MagicMock()
sys.modules['common_types'] = MagicMock()
sys.modules['maxtext.configs.common_types'] = MagicMock()
sys.modules['omegaconf'] = MagicMock()

import os
sys.path.append('src')

from maxtext.configs import types
keys = list(types.MaxTextConfig.model_fields.keys())
print("max_position_embeddings in keys:", 'max_position_embeddings' in keys)
