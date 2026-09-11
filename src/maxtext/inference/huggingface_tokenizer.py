# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JetStream tokenizer compatibility across Transformers return formats."""

from collections.abc import Mapping

import numpy as np
from jetstream.engine import token_utils


class HuggingFaceTokenizer(token_utils.HuggingFaceTokenizer):
  """Normalize chat-template output before JetStream pads the token IDs."""

  def encode(self, s: str, **kwargs):
    if not getattr(self.metadata, "use_chat_template", False):
      return super().encode(s, **kwargs)

    encoded = self.tokenizer.apply_chat_template(
        [{"role": "user", "content": s}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        return_tensors="np",
    )
    # Some tokenizer implementations return BatchEncoding even when callers
    # expect an array. Extract IDs explicitly and preserve one-token sequences.
    if isinstance(encoded, Mapping):
      encoded = encoded["input_ids"]
    tokens = np.asarray(encoded).reshape(-1)
    return token_utils.pad_tokens(
        tokens,
        self.bos_id,
        self.pad_id,
        is_bos=False,
        prefill_lengths=kwargs.get("prefill_lengths"),
        max_prefill_length=kwargs.get("max_prefill_length"),
        jax_padding=kwargs.get("jax_padding", True),
    )
