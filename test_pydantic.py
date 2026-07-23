import sys
import types as py_types

def mock_module(name):
    mod = py_types.ModuleType(name)
    sys.modules[name] = mod
    return mod

jax = mock_module('jax')
jax.sharding = mock_module('jax.sharding')
jax.numpy = mock_module('jax.numpy')
jax.experimental = mock_module('jax.experimental')
jax.experimental.compilation_cache = mock_module('jax.experimental.compilation_cache')
jax.experimental.compilation_cache.compilation_cache = mock_module('jax.experimental.compilation_cache.compilation_cache')
jax.tree_util = mock_module('jax.tree_util')
jax.tree_util.register_pytree_node_class = lambda x: x
optax = mock_module('optax')
mock_module('omegaconf')
mock_module('yaml')
mock_module('pandas')

import os
sys.path.append('src')

from maxtext.configs import types
print('max_position_embeddings' in types.MaxTextConfig.model_fields.keys())
print('Total keys:', len(types.MaxTextConfig.model_fields.keys()))
