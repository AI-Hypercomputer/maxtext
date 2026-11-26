# Copyright 2026 Google LLC
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

"""Qwen3.5's RMSNorms, as plain functions over parameter pytrees.

Both kinds compute their statistics and the normalised value in fp32; how they
round after that is `config.norm_numerics` (MaxText changed it on 2026-09-10;
see `Qwen35Config.norm_numerics`):

  * zero-centred (`Qwen3NextRMSNorm`, normalizations.py) -- the input /
    post-attention / final norms and the attention q/k norms.  The stored
    scale starts at 0 and the applied one is `scale + 1`.
      'fp32': `fp32(normed) * (fp32(scale) + 1)`, rounded once to `dtype`
              (`_ops.norm_fp32`; MaxText since 2026-09-10, and HF).
      'bf16': `bf16(normed) * (bf16(scale) + 1)`, the add and the multiply in
              `dtype` (`_ops.norm`, the DeepSeek port's RMSNorm; MaxText before
              2026-09-10).
  * plain (`Qwen3NextRMSNormGated`'s inner norm) -- the GDN output norm over
    the value head dim, ones-initialised, no offset.  `gdn.gated_rms_norm` adds
    the `silu(z)` gate and does the rounding: 'fp32' keeps the scaled value in
    fp32 until after the gate (MaxText since 2026-09-10); 'bf16' rounds the
    normalised value to `dtype` and scales it there (MaxText before it, and
    HF with its bf16 weights).

The GDN's q / k l2 norm is `gdn.l2norm`.
"""

from __future__ import annotations

import jax.numpy as jnp

from maxtext.experimental.mpmd.jax.deepseek.layers import precision_of  # noqa: F401  re-exported for the model modules
from maxtext.experimental.mpmd.jax.qwen35 import _ops


def rms_norm(params, x, config, zero_centered: bool = True):
  """`x / rms(x) * scale` (plain) or `* (scale + 1)` (zero-centred), in `dtype`.

  `zero_centered=False` is the plain norm under 'bf16' numerics only: the gated
  norm under 'fp32' scales in fp32 and rounds after its gate
  (`gdn.gated_rms_norm`).
  """
  if config.norm_numerics == "fp32" and zero_centered:
    scale = _ops.use_weight(params["scale"], jnp.float32, config) + 1.0
    return _ops.norm_fp32({"scale": scale}, x, config).astype(config.dtype)
  if zero_centered:
    params = {"scale": _ops.use_weight(params["scale"], config.dtype, config) + 1.0}
  return _ops.norm(params, x, config)
