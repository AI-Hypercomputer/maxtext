
import jax
import common_types
Array = common_types.Array
from flax import struct
from typing import Any, List, Optional, Tuple, Callable, Union
import numpy as np
import math
import jax.numpy as jnp


# TODO: move this to Jetstream after all features are done for chunked prefill
# this is in decode to enable testing and debugging

@struct.dataclass
class ChunkMetadata:
  processed_chunks: Any
  next_pos: Union[jax.Array, np.ndarray]
  true_length: int
  chunk_size: int
  chunk_padded_tokens: Union[jax.Array, np.ndarray]
  processed: bool
  is_first_chunk: bool
  is_last_chunk:bool

def create_chunked_metadata(tokens, true_length, chunk_size):
  start = 0
  chunk_metadata_list = []
  num_chunks =int(math.ceil(len(tokens)/chunk_size))

  for chunk_num in range(num_chunks):
    true_length_of_chunk = chunk_size
    start = int(chunk_num*chunk_size)
    end = jnp.minimum((chunk_num+1)*chunk_size, true_length)
    true_length_of_chunk = end - start
    
    next_pos = jnp.full((1, 1), start + true_length_of_chunk, dtype=jnp.int32)
    is_first_chunk = chunk_num == 0
    is_last_chunk = chunk_num == num_chunks - 1

    chunk_padded_tokens=jax.lax.slice(tokens, (start,), (start+chunk_size,))
    
    chunk_metadata_list.append(ChunkMetadata(processed_chunks=None, 
                                             next_pos=next_pos, true_length=true_length_of_chunk,
                                              chunk_size=chunk_size, 
                                              chunk_padded_tokens=chunk_padded_tokens,
                                              processed=False,
                                              is_first_chunk=is_first_chunk,
                                              is_last_chunk=is_last_chunk, ))
    return chunk_metadata_list

