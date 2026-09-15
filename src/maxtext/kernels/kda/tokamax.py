# Copyright 2026 Ant Group. All Rights Reserved.
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

"""Tokamax KDA backend for maxtext.

Wraps ``tokamax._src.ops.experimental.kda.api.kimi_delta_attention`` with a
maxtext-compatible interface (batch-first ``[B, T, H, K]`` layout). Tokamax
natively supports ``[B, T]`` segment_ids, so no B*T flatten / per-batch
offset is needed.
"""

from __future__ import annotations

import importlib.metadata

import jax.numpy as jnp
from packaging.version import InvalidVersion, Version

# `kimi_delta_attention` lives on tokamax's experimental path
# `tokamax._src.ops.experimental.kda`, which first shipped in release 0.0.14
# (0.0.12 / 0.0.13 do not contain it, and the empty 0.1.0 upload was yanked).
# The import in `tokamax_chunk_kda` is deliberately lazy so an environment
# with an older tokamax can still import MaxText: `layers/nnx_decoders.py`
# imports this adapter's caller unconditionally, so a failure at module import
# time would break every NNX model instead of just KDA. Only an actual KDA
# call needs the API, and that call raises `kda_api_unavailable`, which names
# the version required.
#
# This is intentionally NOT expressed as a `tokamax>=0.0.14` requirement pin:
# the decoupled-mode environment pins tokamax to an older release on purpose
# (see scripts/generate_decoupled_requirements.py — newer tokamax imports
# xprof at module scope) and still has to import MaxText, and a version floor
# would also reject source installs of tokamax main, which work fine. The
# capability test is therefore the import itself; the version is diagnostic.
_MIN_TOKAMAX_VERSION = "0.0.14"


def _installed_tokamax_version() -> str | None:
  """Return the installed tokamax version, or None when it is not installed."""
  try:
    return importlib.metadata.version("tokamax")
  except importlib.metadata.PackageNotFoundError:
    return None


def _predates_kda(version: str) -> bool:
  """Whether *version* is a distribution known to predate the KDA module.

  An unparsable version is deliberately not treated as predating it: a source
  build (``pip install -e`` of a KDA branch) can report an older number while
  still providing the API, and claiming otherwise would send the user to
  ``pip install`` over a working checkout.
  """
  try:
    return Version(version) < Version(_MIN_TOKAMAX_VERSION)
  except InvalidVersion:
    return False


def kda_api_unavailable(cause: BaseException | None = None, *, detail: str = "") -> ImportError:
  """Build the ImportError raised when the tokamax KDA API cannot be imported.

  Names the first tokamax release that ships the API and the version actually
  installed, so the fix is obvious. The installed version is reported for
  diagnosis only — source installs may report a placeholder version while
  still providing the API, which is why this is raised on a failed import
  rather than on a version comparison.

  Args:
    cause: The original import failure, attached as ``__cause__``.
    detail: Optional sentence prefixed to the message (e.g. what the caller
      needed the API for and why degrading silently is not acceptable).

  Returns:
    An ``ImportError`` with an actionable message.
  """
  found = _installed_tokamax_version()
  if found is None:
    installed = "tokamax is not installed"
  elif _predates_kda(found):
    # Verified: the 0.0.12 and 0.0.13 distributions do not ship the KDA module.
    installed = f"the installed tokamax {found} predates it"
  else:
    installed = (
        f"tokamax {found} is installed but its KDA API could not be imported — a source build "
        "may be incomplete or a dependency of it may be missing"
    )
  message = (
      "KDA requires the tokamax KDA API (tokamax._src.ops.experimental.kda), first shipped in "
      f"tokamax {_MIN_TOKAMAX_VERSION}; {installed}. "
      f"Fix with: pip install -U 'tokamax>={_MIN_TOKAMAX_VERSION}'; for a source checkout, "
      "reinstall openxla/tokamax main."
  )
  if detail:
    message = f"{detail} {message}"
  # `raise X from Y` is statement-only syntax, so chain the cause by hand:
  # this mirrors what the interpreter does for that statement.
  error = ImportError(message)
  if cause is not None:
    error.__cause__ = cause
    error.__suppress_context__ = True
  return error


def _to_tokamax(q, k, v, g, beta):
  """[B, T, H, K] -> [H, B, T, K];  [B, T, H] -> [H, B, T]."""
  return (
      jnp.transpose(q, (2, 0, 1, 3)),
      jnp.transpose(k, (2, 0, 1, 3)),
      jnp.transpose(v, (2, 0, 1, 3)),
      jnp.transpose(g, (2, 0, 1, 3)),
      jnp.transpose(beta, (2, 0, 1)),
  )


def _to_maxtext(o_h):
  """[H, B, T, V] -> [B, T, H, V]."""
  return jnp.transpose(o_h, (1, 2, 0, 3))


def tokamax_chunk_kda(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    scale: float | None = None,
    initial_state: jnp.ndarray | None = None,
    output_final_state: bool = False,
    a_log: jnp.ndarray | None = None,
    delta_time_bias: jnp.ndarray | None = None,
    use_gate_in_kernel: bool = False,
    use_qk_l2norm: bool = False,
    lower_bound: float | None = None,
    segment_ids: jnp.ndarray | None = None,
    max_num_segments: int | None = None,
    context_parallel_metadata: object | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """KDA via tokamax, batch-first interface matching ``kernels.kda.chunk_kda``.

  Tokamax accepts ``[B, T]`` segment_ids natively so no B*T flatten
  or per-batch offset computation is needed.

  Args:
      q: [B, T, H, K] queries.
      k: [B, T, H, K] keys.
      v: [B, T, H, V] values.
      g: [B, T, H, K] raw gate values (activated in-kernel when
          ``use_gate_in_kernel=True``; otherwise already in log space).
      beta: [B, T, H] delta-rule mixing coefficient in [0, 1].
      scale: attention scale (default ``K ** -0.5``).
      initial_state: must be None (not yet supported in maxtext).
      output_final_state: must be False (not yet supported in maxtext).
      a_log: [H] per-head log decay-rate parameter. Required when
          ``use_gate_in_kernel=True``.
      delta_time_bias: [H*K] optional per-head, per-key-channel gate bias.
          Used only when ``use_gate_in_kernel=True``.
      use_gate_in_kernel: whether ``g`` is raw delta-time input that should
          be activated with ``a_log`` / ``delta_time_bias`` inside the kernel.
      use_qk_l2norm: whether to L2-normalize q/k on the last dim in-kernel.
      lower_bound: optional sigmoid-gate lower bound in ``[-5, 0)``. When
          None, the standard ``softplus`` gate path is used.
      segment_ids: [B, T] 1-based, 0=padding. Passed directly to tokamax.
      max_num_segments: static upper bound on varlen segments. Required when
          ``segment_ids`` is provided without ``initial_state``.
      context_parallel_metadata: optional ``tokamax ... ContextParallelMetadata``
          for context parallelism.

  Returns:
      (o, None) where o is [B, T, H, V].
  """
  # Input validation before the lazy import so the guards fire even on
  # installs without tokamax.
  if initial_state is not None:
    raise NotImplementedError("initial_state is not supported with tokamax backend")
  if output_final_state:
    raise NotImplementedError("output_final_state is not supported with tokamax backend")

  # Deliberately lazy: importing the KDA API must not fail at module import
  # time on installs without tokamax; only an actual KDA call requires it.
  try:
    from tokamax._src.ops.experimental.kda.api import (  # pylint: disable=import-outside-toplevel
        kimi_delta_attention,
    )
  except ImportError as exc:
    raise kda_api_unavailable(exc) from exc

  q_h, k_h, v_h, g_h, beta_h = _to_tokamax(q, k, v, g, beta)

  o_h, _ = kimi_delta_attention(
      query=q_h,
      key=k_h,
      value=v_h,
      gate=g_h,
      beta=beta_h,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      scale=scale,
      segment_ids=segment_ids,
      use_gate_in_kernel=use_gate_in_kernel,
      use_qk_l2norm=use_qk_l2norm,
      lower_bound=lower_bound,
      max_num_segments=max_num_segments,
      implementation="mosaic",  # tokamax's Pallas Mosaic TPU kernel ("mosaic" resolves to the mosaic_tpu impl)
      context_parallel_metadata=context_parallel_metadata,
  )

  o = _to_maxtext(o_h)
  return o, None
