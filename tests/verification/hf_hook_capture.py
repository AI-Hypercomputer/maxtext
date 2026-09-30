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

"""Zero-HF-code-modification single-layer reference capture via PyTorch forward/backward hooks."""

from __future__ import annotations

from typing import Any
import torch


def capture_hf_layer_fwd_bwd(
    layer_module: torch.nn.Module,
    h_in: torch.Tensor,
    cotangent_out: torch.Tensor,
    **forward_kwargs: Any,
) -> dict[str, Any]:
  """Runs one HF decoder layer in fp32 and captures boundary states, gradients, and discrete routing/indexer indices via hooks.

  Requires zero edits to HuggingFace transformers modeling files.
  """
  captured: dict[str, Any] = {
      "router_topk_idx": None,
      "router_scores": None,
      "indexer_topk_indices": None,
  }
  handles = []

  def _make_router_hook():
    def _hook(_module, _inputs, output):
      if isinstance(output, tuple) and len(output) >= 2:
        captured["router_topk_idx"] = output[0].detach().cpu().numpy()
        captured["router_scores"] = output[1].detach().cpu().numpy()

    return _hook

  def _make_indexer_hook():
    def _hook(_module, _inputs, output):
      if isinstance(output, tuple) and len(output) >= 1:
        captured["indexer_topk_indices"] = output[-1].detach().cpu().numpy()
      elif isinstance(output, torch.Tensor):
        captured["indexer_topk_indices"] = output.detach().cpu().numpy()

    return _hook

  for name, submod in layer_module.named_modules():
    cls_name = type(submod).__name__
    if name.endswith("gate") or cls_name in ("Gate", "DeepseekV4MoEGate", "MoEGate"):
      handles.append(submod.register_forward_hook(_make_router_hook()))
    elif "indexer" in name or "Indexer" in cls_name:
      handles.append(submod.register_forward_hook(_make_indexer_hook()))

  try:
    layer_module.zero_grad(set_to_none=True)
    h_in_var = h_in.detach().to(torch.float32).requires_grad_(True)
    out = layer_module(h_in_var, **forward_kwargs)
    h_out = out[0] if isinstance(out, tuple) else out
    loss = (h_out * cotangent_out.to(torch.float32)).sum()
    loss.backward()
    param_grads = {k: p.grad.detach().cpu().numpy() for k, p in layer_module.named_parameters() if p.grad is not None}
    return {
        "h_in": h_in_var.detach().cpu().numpy(),
        "h_out": h_out.detach().cpu().numpy(),
        "dh_in": h_in_var.grad.detach().cpu().numpy(),
        "dh_out": cotangent_out.detach().cpu().numpy(),
        "param_grads": param_grads,
        **captured,
    }
  finally:
    for h in handles:
      h.remove()
