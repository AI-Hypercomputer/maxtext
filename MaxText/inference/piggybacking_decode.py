#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Piggybacking decode."""

from flax import struct
import jax


@struct.dataclass
class PiggyBackingDecodeParams:
  """Param to the model with piggybacking decode.

  Attributes:
    prefill_slot: A jax array of 1 int indicates the slot in decode state used for new prefill
    generate_slots: 
      A jax array of 1D ints indicates the slots in decode state used for the piggybacking decode.
      The length should match the number of tokens to generate in input tokens.
  """
  prefill_slot: jax.Array
  generate_slots: jax.Array


