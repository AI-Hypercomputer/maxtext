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
"""Thin MaxText adapter for the Tokamax Gated Delta Net (`gated_delta_net`) kernel."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from maxtext.common.common_types import Array

TOKAMAX_GDN_CHUNK_SIZE = 64
TOKAMAX_GDN_RESIDUAL_CHECKPOINT_NAME = "gdn_chunk_states"


def _tokamax_gdn_op() -> Callable[..., tuple[Array, Array | None]]:
  """Returns `tokamax.ops.gated_delta_net`, or raises a clear `ImportError`."""
  try:
    from tokamax import ops as tokamax_ops  # pylint: disable=import-outside-toplevel,g-import-not-at-top

    return tokamax_ops.gated_delta_net
  except (ImportError, AttributeError) as e:
    raise ImportError("use_tokamax_gdn=True requires a tokamax build providing `tokamax.ops.gated_delta_net`.") from e


def _resolve_implementation(
    query: Array,
    implementation: str | Sequence[str] | None,
) -> Any:
  """Resolves the Tokamax GDN implementation strictly without TPU->XLA fallback."""
  if implementation is not None:
    return implementation
  host_backend = jax.default_backend()
  if pltpu.is_tpu_device():
    if host_backend != "tpu":
      raise NotImplementedError(
          "use_tokamax_gdn=True does not yet support offline CPU->TPU cross-compilation; "
          "run on TPU or set use_tokamax_gdn=False."
      )
    return "mosaic"
  backend = host_backend
  if not isinstance(query, jax.core.Tracer) and hasattr(query, "devices"):
    try:
      devs = query.devices()
      if devs:
        backend = next(iter(devs)).platform
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  else:
    aval_sharding = getattr(getattr(query, "aval", None), "sharding", None)
    aval_mesh = getattr(aval_sharding, "mesh", None)
    abstract_device = getattr(aval_mesh, "abstract_device", None)
    if abstract_device is not None and getattr(abstract_device, "platform", None) is not None:
      backend = abstract_device.platform
  if backend == "tpu":
    if host_backend != "tpu":
      raise NotImplementedError(
          "use_tokamax_gdn=True does not yet support offline CPU->TPU cross-compilation; "
          "run on TPU or set use_tokamax_gdn=False."
      )
    return "mosaic"
  if backend == "cpu":
    return "xla"
  raise NotImplementedError(
      f"Tokamax GDN Mosaic kernel requires TPU (or CPU for reference tests), got backend={backend!r}."
  )


def tokamax_chunk_gated_delta_rule(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    b: Array,
    *,
    chunk_size: int = TOKAMAX_GDN_CHUNK_SIZE,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    use_qk_norm_in_gdn: bool = True,
    compute_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    cp_axis: str | tuple[str, ...] | None = None,
    segment_ids: Array | None = None,
    residual_checkpoint_name: str | None = TOKAMAX_GDN_RESIDUAL_CHECKPOINT_NAME,
    implementation: str | Sequence[str] | None = None,
) -> tuple[Array, Array | None]:
  """Evaluates chunked Gated Delta Net via `tokamax.ops.gated_delta_net` in `[B, T, H, D]` layout.

  `g` is passed pre-activated in float32 (`use_gate_in_kernel=False`) so GSPMD/explicit sharding
  reduces `A_log`/`dt_bias` gradients outside `shard_map` and preserves padding `g = 0.0` masking.
  `b` is passed pre-sigmoid in its input dtype (`use_beta_sigmoid_in_kernel=True`) so the op computes
  `sigmoid(b.astype(float32))` internally and skips `base.GatedDeltaNet`'s effectful post-sigmoid
  `checkify.debug_check` under `jax.checkpoint` rematerialization.

  Args:
    query: Query tensor of shape `[B, T, H_QK, K]`.
    key: Key tensor of shape `[B, T, H_QK, K]`.
    value: Value tensor of shape `[B, T, H_V, V]`.
    g: Pre-activated natural-log decay gate of shape `[B, T, H_V]`.
    b: Pre-sigmoid beta logits of shape `[B, T, H_V]`.
    chunk_size: Chunk length (`64`).
    initial_state: Optional initial recurrent state `[B, H_V, K, V]`.
    output_final_state: Whether to return the final recurrent state.
    use_qk_norm_in_gdn: Whether to L2-normalize `query` and `key` inside the kernel.
    compute_dtype: Compute dtype for `(query, key, value)` and returned `final_state`.
    cp_axis: Context-parallel mesh axis name(s); sequence-sharded CP is not yet enabled here.
    segment_ids: Packed document segment IDs; not yet enabled here.
    residual_checkpoint_name: `jax.ad_checkpoint.checkpoint_name` tag for custom remat policies.
    implementation: Optional explicit Tokamax backend override (`"mosaic"` or `"xla"`).

  Returns:
    `(output, final_state)` with `output` matching `query.dtype` and `final_state` in `compute_dtype` (or `None`).
  """
  if segment_ids is not None:
    raise NotImplementedError("tokamax_chunk_gated_delta_rule does not yet support packed segment_ids.")
  if cp_axis is not None:
    raise NotImplementedError(
        "tokamax_chunk_gated_delta_rule does not yet support sequence-sharded cp_axis; use head-sharded CP."
    )
  if chunk_size != TOKAMAX_GDN_CHUNK_SIZE:
    raise NotImplementedError(
        f"tokamax_chunk_gated_delta_rule currently requires chunk_size={TOKAMAX_GDN_CHUNK_SIZE}, got {chunk_size}."
    )
  seq_len = query.shape[1]
  if seq_len % chunk_size != 0:
    raise ValueError(
        f"tokamax_chunk_gated_delta_rule requires sequence length ({seq_len}) to be divisible by "
        f"chunk_size ({chunk_size})."
    )

  gated_delta_net = _tokamax_gdn_op()
  initial_dtype = query.dtype
  query_c = query.astype(compute_dtype)
  key_c = key.astype(compute_dtype)
  value_c = value.astype(compute_dtype)
  g_f32 = g.astype(jnp.float32)
  resolved_impl = _resolve_implementation(query_c, implementation)

  out, final_state = gated_delta_net(
      query=query_c,
      key=key_c,
      value=value_c,
      gate=g_f32,
      beta=b,
      initial_state=initial_state,
      output_final_state=output_final_state,
      use_qk_l2norm=use_qk_norm_in_gdn,
      use_gate_in_kernel=False,
      use_beta_sigmoid_in_kernel=True,
      chunk_size=chunk_size,
      segment_ids=None,
      residual_checkpoint_name=residual_checkpoint_name,
      implementation=resolved_impl,
  )
  if out.dtype != initial_dtype:
    out = out.astype(initial_dtype)
  if final_state is not None and final_state.dtype != compute_dtype:
    final_state = final_state.astype(compute_dtype)
  # `base.GatedDeltaNet._fwd` ("xla") ignores `residual_checkpoint_name`; tag `(out, final_state)` here to match
  # `PallasMosaicTpuGatedDeltaNet._tag_gdn_residuals` ("mosaic"), which also tags outputs alongside `GdnResiduals`.
  if residual_checkpoint_name is not None and resolved_impl == "xla":
    out = checkpoint_name(out, residual_checkpoint_name)
    if final_state is not None:
      final_state = checkpoint_name(final_state, residual_checkpoint_name)
  return out, final_state
