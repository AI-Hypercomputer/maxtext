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

"""Tokamax Splash Attention kernel wrapper for DSv3."""

import functools

import jax
import jax.numpy as jnp
from maxtext.models.deepseek_lineage import base
from maxtext.models.deepseek_lineage import splash_attention_kernel as tokamax_splash
from maxtext.models.deepseek_lineage import splash_attention_mask as mask_lib
from maxtext.models.deepseek_lineage import splash_attention_mask_info as mask_info_lib
import numpy as np

# Re-export core types and constants from Tokamax.
SplashConfig = tokamax_splash.SplashConfig
QKVLayout = tokamax_splash.QKVLayout
SegmentIds = base.SegmentIds
DEFAULT_MASK_VALUE = base.DEFAULT_MASK_VALUE
SplashCustomReturnType = base.SplashCustomReturnType
SplashResidualsType = base.SplashResidualsType
MaskFunctionType = tokamax_splash.MaskFunctionType
LOG2E = tokamax_splash.LOG2E

# Backwards compatibility alias for BlockSizes
BlockSizes = SplashConfig


def _splash_attention_manual_fwd(
    fwd_mask_info: mask_info_lib.MaskInfo,
    dkv_mask_info: mask_info_lib.MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: base.SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool = False,
    config: SplashConfig | None = None,
    save_residuals: bool = True,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    mask_function: MaskFunctionType | None = None,
    fwd_mask_sparsity: float = 1.0,
    dkv_mask_sparsity: float = 1.0,
    max_logit_value: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Computes forward splash attention, returning output and logsumexp."""
  del dkv_mask_info, save_residuals, dkv_mask_sparsity
  if config is None:
    config = SplashConfig.get_default()

  # pylint: disable=protected-access
  out, stats = tokamax_splash._splash_attention_forward(
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks=sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      save_residuals=True,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=max_logit_value,
  )
  logsumexp = stats["logsumexp"]
  return out, logsumexp


@functools.partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "config",
        "save_residuals",
        "mask_value",
        "mask_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
    ],
)
def _splash_attention_manual_bwd(
    fwd_mask_info: mask_info_lib.MaskInfo,
    dkv_mask_info: mask_info_lib.MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    logsumexp: jax.Array,
    do: jax.Array,
    segment_ids: base.SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool = False,
    config: SplashConfig | None = None,
    save_residuals: bool = False,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    mask_function: MaskFunctionType | None = None,
    fwd_mask_sparsity: float = 1.0,
    dkv_mask_sparsity: float = 1.0,
    max_logit_value: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes backward splash attention using attention output and logsumexp."""
  del fwd_mask_info, sinks, save_residuals, fwd_mask_sparsity, max_logit_value
  if config is None:
    config = SplashConfig.get_default()
  if dkv_mask_info is None:
    raise ValueError("Need to specify backward blocks.")
  if not config.has_backward_blocks:
    raise ValueError("Need to specify backward blocks in config.")

  # Compute di: [num_heads, q_seq_len]
  di = jnp.einsum("hsd,hsd->hs", out.astype(jnp.float32), do.astype(jnp.float32))
  # pylint: disable=protected-access
  dq, dk, dv = tokamax_splash._splash_attention_bwd_dkv(
      q,
      k,
      v,
      segment_ids,
      logsumexp,
      do,
      di,
      bq=config.block_q_dkv,  # pyrefly: ignore[bad-argument-type]
      bkv=config.block_kv_dkv,  # pyrefly: ignore[bad-argument-type]
      bkv_compute=config.block_kv_dkv_compute,  # pyrefly: ignore[bad-argument-type]
      is_mqa=is_mqa,
      mask_info=dkv_mask_info,
      mask_value=mask_value,
      mask_function=mask_function,
      config=config,
      dkv_mask_sparsity=dkv_mask_sparsity,
  )
  return dq, dk, dv


@jax.tree_util.register_pytree_node_class
class SplashAttentionKernel:
  """SplashAttentionKernel wrapping Tokamax splash kernel with custom VJP hooks."""

  def __init__(
      self,
      fwd_mask_info: mask_info_lib.MaskInfo,
      dkv_mask_info: mask_info_lib.MaskInfo | None,
      **kwargs,
  ):
    self.kwargs = kwargs
    self.fwd_mask_info = fwd_mask_info
    self.dkv_mask_info = dkv_mask_info

  def __call__(self, *args, **kwargs) -> SplashCustomReturnType:
    return tokamax_splash._splash_attention(
        self.fwd_mask_info,
        self.dkv_mask_info,
        *args,
        **dict(self.kwargs, **kwargs),
    )

  def manual_fwd(self, *args, **kwargs) -> tuple[jax.Array, jax.Array]:
    return _splash_attention_manual_fwd(
        self.fwd_mask_info,
        self.dkv_mask_info,
        *args,
        **dict(self.kwargs, **kwargs),
    )

  def manual_bwd(self, *args, **kwargs) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _splash_attention_manual_bwd(
        self.fwd_mask_info,
        self.dkv_mask_info,
        *args,
        **dict(self.kwargs, **kwargs),
    )

  def manual_sharding_spec(self, sharding: jax.sharding.NamedSharding):
    """Returns a value that can be used as a shard_map partition spec for the kernel."""
    spec = sharding.spec
    if len(spec) == 2:
      _, q_seq_spec = spec[0], spec[1]
    elif len(spec) == 1:
      q_seq_spec = spec[0]
    else:
      raise ValueError(f"Unsupported sharding spec rank: {len(spec)}")

    seq_spec = jax.sharding.PartitionSpec(q_seq_spec) if q_seq_spec is not None else jax.sharding.PartitionSpec()
    replicated = jax.sharding.PartitionSpec()

    def _resolve_spec(arr):
      return seq_spec if arr is not None else None

    def _make_mask_info_spec(mask_info):
      if mask_info is None:
        return None
      return mask_info_lib.MaskInfo(
          mask_next=_resolve_spec(mask_info.mask_next),
          active_rows=_resolve_spec(mask_info.active_rows),
          active_cols=_resolve_spec(mask_info.active_cols),
          num_active_blocks=_resolve_spec(mask_info.num_active_blocks),
          block_mask=_resolve_spec(mask_info.block_mask),
          partial_mask_blocks=(
              replicated if mask_info.partial_mask_blocks is not None else None  # pyrefly: ignore[bad-argument-type]
          ),
          q_sequence=_resolve_spec(mask_info.q_sequence),
      )

    return SplashAttentionKernel(
        _make_mask_info_spec(self.fwd_mask_info),
        _make_mask_info_spec(self.dkv_mask_info),
        **self.kwargs,
    )

  def tree_flatten(self):
    return ((self.fwd_mask_info, self.dkv_mask_info), self.kwargs)

  @classmethod
  def tree_unflatten(cls, kwargs, values):
    fwd_mask_info, dkv_mask_info = values
    dkv_mask_info = mask_info_lib.MaskInfo(*dkv_mask_info) if dkv_mask_info is not None else None
    return SplashAttentionKernel(mask_info_lib.MaskInfo(*fwd_mask_info), dkv_mask_info, **kwargs)


def _make_splash_attention(
    mask: np.ndarray | mask_lib.Mask,
    *,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    q_seq_shards: int,
):
  """Creates a SplashAttentionKernel instance."""
  if len(mask.shape) != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")

  if isinstance(mask, np.ndarray):
    mask = mask_lib.NumpyMask(mask)

  if config is None:
    config = SplashConfig.get_default()

  if (config.qk_diag_skip or config.sv_diag_skip) and not isinstance(mask, mask_lib.CausalMask):
    param_name = "sv_diag_skip" if config.sv_diag_skip else "qk_diag_skip"
    raise ValueError(f"{param_name}=True requires a pure CausalMask; got " f"{type(mask).__name__}.")

  process_fn = functools.partial(
      mask_info_lib.process_mask,
      downcast_smem_data=downcast_smem_data,
      partial_mask_blocks_dtype=partial_mask_blocks_dtype,
      q_seq_shards=q_seq_shards,
  )

  fwd_mask_info, mask_function_fwd = process_fn(
      mask,
      (config.block_q, config.block_kv),
  )
  fwd_mask_sparsity = float(np.mean(fwd_mask_info.block_mask != 0))
  fwd_mask_info = jax.tree.map(jnp.array, fwd_mask_info)

  dkv_mask_info = None
  if config.has_backward_blocks:
    bq_dkv, bkv_dkv = config.block_q_dkv, config.block_kv_dkv
    dkv_mask_info, mask_function_dkv = process_fn(
        mask,
        (bq_dkv, bkv_dkv),
        is_dkv=True,
        return_dynamic_grid=config.dq_reduction_steps == 3,
    )
    assert (mask_function_fwd is None) == (mask_function_dkv is None)
    dkv_mask_sparsity = float(np.mean(dkv_mask_info.block_mask != 0))
    dkv_mask_info = jax.tree.map(jnp.array, dkv_mask_info)
  else:
    dkv_mask_sparsity = 1.0

  return SplashAttentionKernel(
      fwd_mask_info,
      dkv_mask_info,
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=mask_function_fwd,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
  )


make_splash_mha = functools.partial(_make_splash_attention, is_mqa=False)
make_splash_mqa = functools.partial(_make_splash_attention, is_mqa=True)
make_splash_mha_single_device = functools.partial(make_splash_mha, q_seq_shards=1)
make_splash_mqa_single_device = functools.partial(make_splash_mqa, q_seq_shards=1)
