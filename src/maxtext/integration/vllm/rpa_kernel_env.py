# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keeps tpu-inference's ragged-paged-attention kernel in sync with `attention`.

tpu-inference picks its RPA kernel from the USE_BATCHED_RPA_LONG_CTX_KERNEL /
USE_BATCHED_RPA_KERNEL environment variables. MaxText sets them from the
`attention` config value (vllm_batched_rpa_long_ctx / vllm_batched_rpa) so users
only configure one knob.

Older tpu-inference binds the kernel once, when
`tpu_inference.layers.common.attention_interface` is first imported. On a vLLM
worker that import happens during startup (flash_attn backend, model_loader),
before MaxText builds the model, so merely setting the variable then had no
effect and the default RPA v3 kernel ran silently. Newer tpu-inference resolves
the kernel on every call (`attention_interface.rpa_kernel_module`); for older
versions `select_rpa_kernel` rebinds the already-imported module's entry points.
"""

import importlib
import os
import sys

from maxtext.utils import max_logging

_ATTENTION_INTERFACE_MODULE = "tpu_inference.layers.common.attention_interface"

# attention config value -> tpu-inference env var it turns on.
ATTENTION_TO_RPA_ENV = {
    "vllm_batched_rpa_long_ctx": "USE_BATCHED_RPA_LONG_CTX_KERNEL",
    "vllm_batched_rpa": "USE_BATCHED_RPA_KERNEL",
}

# env var -> kernel module an import-time-bound tpu-inference would have picked.
_RPA_ENV_TO_KERNEL_MODULE = {
    "USE_BATCHED_RPA_LONG_CTX_KERNEL": "tpu_inference.kernels.experimental.batched_rpa_long_ctx.wrapper",
    "USE_BATCHED_RPA_KERNEL": "tpu_inference.kernels.experimental.batched_rpa.wrapper",
}


def select_rpa_kernel(attention: str | None, use_batched_rpa: bool = False) -> str | None:
  """Turns on the tpu-inference RPA kernel matching `attention`.

  Only ever enables a kernel: `vllm_rpa` leaves the environment alone so a
  kernel enabled through the environment (e.g. ROLLOUT_USE_BATCHED_RPA) keeps
  working as before.

  Args:
    attention: The MaxText `attention` config value.
    use_batched_rpa: Legacy `use_batched_rpa` override (same as
      attention="vllm_batched_rpa").

  Returns:
    The environment variable that was set, or None.
  """
  env_name = ATTENTION_TO_RPA_ENV.get(attention or "")
  if env_name is None and use_batched_rpa:
    env_name = "USE_BATCHED_RPA_KERNEL"
  if env_name is None:
    return None
  os.environ[env_name] = "1"
  _rebind_import_time_kernel(env_name)
  return env_name


def _rebind_import_time_kernel(env_name: str) -> None:
  """Points an already-imported, import-time-bound attention_interface at the kernel."""
  module = sys.modules.get(_ATTENTION_INTERFACE_MODULE)
  if module is None:
    # Not imported yet: tpu-inference reads the (now set) env var on import.
    return
  if hasattr(module, "rpa_kernel_module"):
    # Resolves the kernel on every call; nothing to do.
    return
  kernel = importlib.import_module(_RPA_ENV_TO_KERNEL_MODULE[env_name])
  if getattr(module, "ragged_paged_attention", None) is kernel.ragged_paged_attention:
    return
  module.ragged_paged_attention = kernel.ragged_paged_attention
  module.get_kv_cache_shape = kernel.get_kv_cache_shape
  max_logging.log(
      f"{_ATTENTION_INTERFACE_MODULE} was imported before {env_name} was set; rebound its RPA kernel to {kernel.__name__}."
  )
