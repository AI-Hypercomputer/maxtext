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

"""CPU-runnable HuggingFace reference for inclusionAI/Ming-3.0-audio-tiny.

The publisher's remote code (`modeling_bailing_moe_v3.py`, loaded unmodified via
`trust_remote_code`) calls GPU-only `flash-linear-attention` (fla) kernels in its Kimi Delta
Attention (KDA) layers. On CPU, fla's own fallbacks do not accept the in-kernel gate arguments
(`A_log`, `dt_bias`, `lower_bound`) and `FusedRMSNormGated` needs a GPU, so two symbols in the
remote module are rebound to fla's PyTorch reference implementations:

  * `chunk_kda` / `fused_recurrent_kda` -> `fla.ops.kda.gate.naive_kda_*gate` +
    `fla.ops.kda.naive.naive_recurrent_kda` (q/k L2-norm applied first, as the kernels do).
  * `FusedRMSNormGated` -> RMSNorm(x) * weight * act(g) in float32.

Everything else, including fla's `ShortConvolution` (which has a CPU path), is used as published.
`openai-whisper` and `torchaudio` must be importable (e.g. via PYTHONPATH).
"""

# pylint: disable=not-callable

from __future__ import annotations

import sys

import torch
from torch import nn

MING_HF_MODEL_ID = "inclusionAI/Ming-3.0-audio-tiny"


class TorchRMSNormGated(nn.Module):
  """`fla.modules.FusedRMSNormGated` reference: rmsnorm(x) * weight * act(g), float32 math."""

  def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", **kwargs):
    del kwargs
    super().__init__()
    self.eps = eps
    self.activation = activation
    self.weight = nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None

  def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
    """Returns rmsnorm(x) * weight * act(g)."""
    del residual, prenorm, residual_in_fp32
    xf, gf = x.float(), g.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
    if self.weight is not None:
      y = y * self.weight.float()
    y = y * torch.sigmoid(gf) if self.activation == "sigmoid" else y * gf * torch.sigmoid(gf)
    return y.to(x.dtype)


def torch_kda(
    q,
    k,
    v,
    g,
    beta,
    A_log=None,  # pylint: disable=invalid-name
    dt_bias=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    lower_bound=None,
    cu_seqlens=None,
    scale=None,
    **kwargs,
):
  """CPU replacement for `fla.ops.kda.chunk_kda` / `fused_recurrent_kda` built from fla's naive references."""
  from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate  # pylint: disable=import-outside-toplevel
  from fla.ops.kda.naive import naive_recurrent_kda  # pylint: disable=import-outside-toplevel

  del kwargs  # e.g. safe_gate: only selects a kernel variant, same math.
  assert cu_seqlens is None, "varlen is not supported by the CPU reference"
  if use_gate_in_kernel:
    if lower_bound is not None:
      g = naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound=lower_bound)
    else:
      g = naive_kda_gate(g, A_log, dt_bias)
  if use_qk_l2norm_in_kernel:  # fla l2norm: x / sqrt(sum(x^2) + 1e-6)
    q, k = (x.float() * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + 1e-6) for x in (q, k))
  return naive_recurrent_kda(
      q, k, v, g, beta, scale=scale, initial_state=initial_state, output_final_state=output_final_state
  )


def _patch_text_module(mod):
  mod.FusedRMSNormGated = TorchRMSNormGated
  mod.chunk_kda = torch_kda
  mod.fused_recurrent_kda = torch_kda


def get_remote_modules(model_path: str = MING_HF_MODEL_ID):
  """Returns `(text_module, audio_module)` remote-code modules with CPU patches applied."""
  from transformers.dynamic_module_utils import get_class_from_dynamic_module  # pylint: disable=import-outside-toplevel

  text_cls = get_class_from_dynamic_module("modeling_bailing_moe_v3.BailingMoeV3Model", model_path)
  text_mod = sys.modules[text_cls.__module__]
  _patch_text_module(text_mod)
  audio_cls = get_class_from_dynamic_module(
      "modeling_bailing_moe_v3_audio.BailingMoeV3AudioForConditionalGeneration", model_path
  )
  audio_mod = sys.modules[audio_cls.__module__]
  # The audio module imports BailingMoeV3Model via a relative import; make sure it
  # resolves to the patched module object.
  _patch_text_module(sys.modules[audio_mod.BailingMoeV3Model.__module__])
  return text_mod, audio_mod


def load_hf_model(model_path: str = MING_HF_MODEL_ID, dtype=torch.float32):
  """Loads the full audio+text HF model on CPU with the fla kernels replaced."""
  _, audio_mod = get_remote_modules(model_path)
  # Use the (patched) remote class directly so that from_pretrained cannot
  # re-import a fresh, unpatched copy of the module.
  model = audio_mod.BailingMoeV3AudioForConditionalGeneration.from_pretrained(
      model_path,
      dtype=dtype,
      trust_remote_code=True,
      attn_implementation="eager",
  ).eval()
  fix_rotary_inv_freq(model)
  return model


def fix_rotary_inv_freq(model):
  """Recomputes the non-persistent RoPE `inv_freq` buffers.

  With transformers>=5, remote-code models are materialized from the meta device and the
  non-persistent `BailingMoeV3RotaryEmbedding.inv_freq` buffer is left uninitialized (garbage
  memory, often ~0), which silently removes RoPE from the MLA layers. Recompute it from the config.
  """
  for module in model.modules():
    if type(module).__name__ == "BailingMoeV3RotaryEmbedding":
      inv_freq, scaling = module.rope_init_fn(module.config, None)
      module.inv_freq = inv_freq.to(module.inv_freq.device)
      module.original_inv_freq = module.inv_freq
      module.attention_scaling = scaling


def load_hf_processor(model_path: str = MING_HF_MODEL_ID):
  """Loads the publisher's BailingMoeV3AudioProcessor."""
  from transformers import AutoProcessor  # pylint: disable=import-outside-toplevel

  return AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
