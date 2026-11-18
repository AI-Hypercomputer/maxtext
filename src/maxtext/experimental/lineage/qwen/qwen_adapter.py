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

"""Adapter bridging MaxText configs and trainer hooks to Lineage Qwen 3.5 / Qwen MAX.

Defines the NNX model `QwenLineageTransformer` whose parameters are stored in
the pure-JAX Qwen Lineage layout (`params.ModelParams`), runs the batch-split /
pipelined-scan forward and chunked cross-entropy loss in `model.local_forward_loss`,
and integrates cleanly with MaxText's `train.py` via `lineage_adapter`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools
import os
from typing import Any, TYPE_CHECKING

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

if TYPE_CHECKING:
  from maxtext.experimental.lineage.qwen import config as qwen_config
  from maxtext.experimental.lineage.qwen import params as qwen_params

PARAMS_KEY = "params"


def is_qwen_lineage(cfg: Any) -> bool:
  """Whether `cfg` selects the Qwen 3.5 / Qwen MAX Lineage path."""
  if not bool(getattr(cfg, "use_lineage", False)):
    return False
  decoder_block = getattr(cfg, "decoder_block", "")
  block_val = getattr(decoder_block, "value", str(decoder_block)).lower()
  return block_val == "qwen3_5"


def _mesh_dims_from_config(
    ici_parallelism: Sequence[int],
    mesh_axes: Sequence[str],
) -> tuple[int, int, bool, bool]:
  """Extracts `(fsdp_size, ep_size, decouple_ep_d2d, ep_core_minor)` from mesh config."""
  axis_sizes = dict(zip(mesh_axes, (int(x) for x in ici_parallelism)))
  if "ep_d2d" in axis_sizes:
    fsdp_size = axis_sizes["fsdp"]
    ep_size = axis_sizes["expert"] * axis_sizes["ep_d2d"]
    ep_core_minor = list(mesh_axes).index("expert") < list(mesh_axes).index("ep_d2d")
    return fsdp_size, ep_size, True, ep_core_minor
  if "fsdp" in axis_sizes and "expert" in axis_sizes:
    return axis_sizes["fsdp"], axis_sizes["expert"], False, True
  raise ValueError(
      f"Qwen Lineage requires mesh_axes ('fsdp', 'expert', 'ep_d2d') or ('fsdp', 'expert'), got {tuple(mesh_axes)!r}."
  )


def create_device_mesh(
    ici_parallelism: Sequence[int],
    devices: Sequence[Any],
    *,
    config: Any = None,
) -> np.ndarray:
  """Builds the core-minor torus-aligned device mesh for Qwen Lineage."""
  from maxtext.experimental.lineage.qwen import config as qwen_config  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from maxtext.experimental.lineage.qwen import mesh as qwen_mesh  # pylint: disable=g-import-not-at-top,import-outside-toplevel

  mesh_axes = tuple(getattr(config, "mesh_axes", ("fsdp", "expert", "ep_d2d")))
  fsdp_size, ep_size, decouple_ep_d2d, ep_core_minor = _mesh_dims_from_config(ici_parallelism, mesh_axes)
  mesh_cfg = qwen_config.QwenConfig(
      fsdp_size=fsdp_size,
      ep_size=ep_size,
      decouple_ep_d2d=decouple_ep_d2d,
      ep_core_minor=ep_core_minor,
  )
  return qwen_mesh.create_mesh(mesh_cfg, devices=list(devices), aot_topology=None).devices


def validate_supported(cfg: Any) -> None:
  """Raises `ValueError` if `cfg` combines Qwen `use_lineage` with unsupported options."""
  problems: list[str] = []

  def require(ok: bool, message: str) -> None:
    if not ok:
      problems.append(message)

  get = functools.partial(getattr, cfg)
  require(get("scan_layers", True), "scan_layers must be true")
  require(int(get("num_decoder_layers", 0)) % 4 == 0, "num_decoder_layers must be a multiple of 4")
  require(
      int(get("inhomogeneous_layer_cycle_interval", 4)) == 4,
      "inhomogeneous_layer_cycle_interval must be 4",
  )
  require(get("shared_experts", 1) == 1, "shared_experts must be 1")
  require(not get("quantization", None), "quantization (AQT/Qwix) is unsupported; use lineage_quantization")
  require(
      get("gradient_accumulation_steps", 1) == 1 or not get("use_tunix_gradient_accumulation", False),
      "use_tunix_gradient_accumulation is unsupported",
  )
  require(get("training_objective", "causal_lm") == "causal_lm", "training_objective must be causal_lm")
  require(not get("use_multimodal", False), "use_multimodal is unsupported")
  require(not (get("enable_dropout", False) and get("dropout_rate", 0.0) > 0.0), "dropout is unsupported")
  require(not get("enable_diloco", False), "enable_diloco is unsupported")
  require(not getattr(get("lora", None), "enable_lora", False), "lora is unsupported")
  require(not get("parameter_memory_host_offload", False), "parameter_memory_host_offload is unsupported")

  if problems:
    raise ValueError("use_lineage=True (Qwen) does not support this config: " + "; ".join(problems) + ".")


def qwen_config_from_maxtext(
    cfg: Any,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> qwen_config.QwenConfig:
  """Translates a MaxText config + mesh into `qwen.config.QwenConfig`."""
  from maxtext.experimental.lineage.qwen import config as qwen_config  # pylint: disable=g-import-not-at-top,import-outside-toplevel

  axis_names = tuple(mesh.axis_names)
  axis_sizes = tuple(int(mesh.shape[a]) for a in axis_names)
  fsdp_size, ep_size, decouple_ep_d2d, ep_core_minor = _mesh_dims_from_config(axis_sizes, axis_names)

  num_decoder_layers = int(cfg.num_decoder_layers)
  num_superblocks = num_decoder_layers // 4
  emb_dim = int(cfg.emb_dim)
  moe_mlp_dim = int(cfg.moe_mlp_dim)
  model_name = str(getattr(cfg, "model_name", "")).lower()

  # Select the tuned TPU7x preset base (Qwen 3.5 vs Qwen MAX), then overlay MaxText fields.
  is_qwen3_5 = ("qwen3.5" in model_name) or (emb_dim == 4096 and moe_mlp_dim == 1024)
  if is_qwen3_5:
    base_cfg = qwen_config.qwen3_5_config()
  else:
    base_cfg = qwen_config.QwenConfig()

  hardware = str(getattr(cfg, "hardware", "tpu")).lower()
  is_tpu = hardware == "tpu" and all(
      getattr(d, "platform", "tpu") == "tpu" for d in getattr(mesh, "devices", np.array([])).flat
  )

  lineage_quant = str(getattr(cfg, "lineage_quantization", "none") or "none").lower()
  use_fp8 = is_tpu and lineage_quant != "none"

  ragged_bf = float(getattr(cfg, "ragged_buffer_factor", -1.0))
  if ragged_bf <= 0.0:
    ragged_bf = base_cfg.ragged_buffer_factor

  overrides: dict[str, Any] = dict(
      num_superblocks=num_superblocks,
      full_num_superblocks=num_superblocks,
      emb_dim=emb_dim,
      vocab_size=int(cfg.vocab_size),
      norm_eps=float(cfg.normalization_layer_epsilon),
      num_query_heads=int(cfg.num_query_heads),
      num_kv_heads=int(cfg.num_kv_heads),
      head_dim=int(cfg.head_dim),
      partial_rotary_factor=float(cfg.partial_rotary_factor),
      rope_max_timescale=float(cfg.rope_max_timescale),
      gdn_num_k_heads=int(cfg.gdn_num_key_heads),
      gdn_num_v_heads=int(cfg.gdn_num_value_heads),
      gdn_head_k_dim=int(cfg.gdn_key_head_dim),
      gdn_head_v_dim=int(cfg.gdn_value_head_dim),
      gdn_conv_kernel_size=int(cfg.gdn_conv_kernel_dim),
      gdn_chunk_size=int(cfg.gdn_chunk_size),
      num_experts=int(cfg.num_experts),
      num_experts_per_tok=int(cfg.num_experts_per_tok),
      moe_mlp_dim=moe_mlp_dim,
      shared_mlp_dim=int(cfg.mlp_dim),
      norm_topk_prob=bool(cfg.norm_topk_prob),
      float32_gate_logits=bool(cfg.float32_gate_logits),
      routing_mode=os.environ.get("MAXTEXT_QWEN_LINEAGE_ROUTING_MODE", "router"),
      ragged_buffer_factor=ragged_bf,
      per_device_batch_size=max(1, int(cfg.per_device_batch_size)),
      seq_len=int(cfg.max_target_length),
      fsdp_size=fsdp_size,
      ep_size=ep_size,
      decouple_ep_d2d=decouple_ep_d2d,
      ep_core_minor=ep_core_minor,
      sa_block_q=int(getattr(cfg, "sa_block_q", base_cfg.sa_block_q)),
      sa_block_kv=int(getattr(cfg, "sa_block_kv", base_cfg.sa_block_kv)),
      sa_block_kv_compute=int(getattr(cfg, "sa_block_kv_compute", base_cfg.sa_block_kv_compute)),
      sa_block_q_dkv=int(getattr(cfg, "sa_block_q_dkv", base_cfg.sa_block_q_dkv)),
      sa_block_kv_dkv=int(getattr(cfg, "sa_block_kv_dkv", base_cfg.sa_block_kv_dkv)),
      sa_block_kv_dkv_compute=int(getattr(cfg, "sa_block_kv_dkv_compute", base_cfg.sa_block_kv_dkv_compute)),
      num_vocab_tiling=max(1, int(getattr(cfg, "num_vocab_tiling", base_cfg.num_vocab_tiling))),
  )

  if not use_fp8:
    overrides.update(
        moe_fp8_weights=False,
        moe_fp8_gmm_compute=False,
        moe_fp8_token_dispatch=False,
        moe_fp8_bwd_dispatch=False,
        moe_use_ragged_silu_3d_fp8=False,
        moe_use_forked_3d_gmmv2=False,
        moe_bwd_dy_fp8_e5m2=False,
    )

  if not is_tpu:
    # Fall back to pure-JAX CPU-compatible kernels & unpinned collectives.
    overrides.update(
        use_pallas_kernels=False,
        moe_local_65key_sort=False,
        moe_use_tc_ragged=False,
        moe_3d_ep_tokens=False,
        ep_sc_pinning="unpinned",
        fsdp_ag_sc_pinning="unpinned",
        fsdp_rs_sc_pinning="unpinned",
        ep_combine_fwd_unpinned=True,
        ragged_single_sc=False,
        use_max_async_overlap_transform=False,
        use_lineage_pipelined_scan=False,
        fsdp_pipeline_mode="post_rs_tie",
        gdn_fwd_head_group_size=1,
        sa_qk_seq_minor=False,
        fp8_amax_no_combine=False,
        f2_early_mb0_dispatch=False,
        f2_light_router_ties=False,
        moe_bwd_tie_wgrad_before_gate=False,
        defer_meta_ag_to_dispatch=False,
        pipeline_bwd_remat_dispatch=False,
        lineage_defer_mb1_combine=False,
        gdn_rms_norm_barrier_stages=False,
        gdn_norm_layout_mode="token_minor",
        save_decoder_input_hbm=False,
        save_attn_context_hbm=False,
        save_query_proj_hbm=False,
        save_out_proj_hbm=False,
        save_attn_residuals_in_lineage=False,
        save_gdn_residuals_sublayers=(),
        save_moe_gate_up_sublayers=(),
        replicate_shared_gate=False,
        narrow_dense_stage2_only=False,
        bwd_early_mb1_dy_pmax=False,
        remat_bwd_tie_prev_dense_depth=0,
    )

  return qwen_config.QwenConfig(**{**base_cfg.__dict__, **overrides})


def _path_key(entry: Any) -> str:
  if isinstance(entry, jax.tree_util.GetAttrKey):
    return entry.name
  if isinstance(entry, jax.tree_util.SequenceKey):
    return str(entry.idx)
  if isinstance(entry, jax.tree_util.DictKey):
    return str(entry.key)
  raise TypeError(f"Unsupported key path entry: {entry!r}")


def _init_qwen_params_traced(
    rng_key: jax.Array,
    cfg: qwen_config.QwenConfig,
    mesh: jax.sharding.Mesh,
    *,
    param_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> tuple[qwen_params.ModelParams, qwen_params.ModelParams]:
  """Trace-safe (`nnx.eval_shape` / `jax.jit` compatible) sharded Qwen parameter init."""
  from maxtext.experimental.lineage.qwen import params as qwen_params  # pylint: disable=g-import-not-at-top,import-outside-toplevel

  abstract = qwen_params.get_abstract_params(cfg, mesh, param_dtype=param_dtype)
  leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(abstract)
  keys = jax.random.split(rng_key, len(leaves_with_path))
  out_leaves = []
  sharding_leaves = []

  for idx, (path, spec) in enumerate(leaves_with_path):
    path_str = jax.tree_util.keystr(path)
    sharding = spec.sharding
    pspec = sharding.spec
    local_shape, sharded_axes = qwen_params._local_shape_and_axes(spec.shape, pspec, mesh)  # pylint: disable=protected-access
    target_dtype = jnp.dtype(jnp.float32 if spec.dtype == jnp.float32 else param_dtype)

    if any(k in path_str for k in ("pre_attn_norm", "post_attn_norm", "decoder_norm", "q_norm", "k_norm")):
      kind, scale = "small_norm", 0.02
    elif "gdn_norm" in path_str:
      kind, scale = "ones", 1.0
    elif "A_log" in path_str:
      kind, scale = "a_log", 1.0
    elif "dt_bias" in path_str:
      kind, scale = "zeros", 0.0
    else:
      kind = "normal_scaled"
      if path_str.endswith(".wi"):
        fan_in = cfg.emb_dim
      elif path_str.endswith(".wo"):
        fan_in = cfg.moe_mlp_dim
      else:
        fan_in = spec.shape[-2] if len(spec.shape) >= 2 else spec.shape[-1]
      scale = float(fan_in**-0.5)

    def _local_init(subk, _lshape=local_shape, _axes=sharded_axes, _kind=kind, _scale=scale, _dt=target_dtype):
      if _axes:
        shard_id = jnp.int32(0)
        for ax in _axes:
          shard_id = shard_id * jnp.int32(mesh.shape[ax]) + jax.lax.axis_index(ax)
        subk = jax.random.fold_in(subk, shard_id)
      if _kind == "ones":
        arr = jnp.ones(_lshape, dtype=jnp.float32)
      elif _kind == "zeros":
        arr = jnp.zeros(_lshape, dtype=jnp.float32)
      elif _kind == "a_log":
        arr = jnp.log(jax.random.uniform(subk, _lshape, dtype=jnp.float32, minval=1.0, maxval=8.0))
      else:
        arr = jax.random.normal(subk, _lshape, dtype=jnp.float32) * _scale
      return arr.astype(_dt)

    arr = jax.shard_map(
        _local_init,
        mesh=mesh,
        in_specs=(P(),),
        out_specs=pspec,
        check_vma=False,
    )(keys[idx])
    out_leaves.append(arr)
    sharding_leaves.append(sharding)

  return (
      jax.tree_util.tree_unflatten(treedef, out_leaves),
      jax.tree_util.tree_unflatten(treedef, sharding_leaves),
  )


class QwenLineageTransformer(nnx.Module):
  """MaxText NNX model whose parameters and forward pass live in Qwen Lineage."""

  def __init__(
      self,
      config: Any,
      mesh: jax.sharding.Mesh,
      quant: Any = None,
      *,
      rngs: nnx.Rngs,
      model_mode: str = "train",
  ):
    if model_mode != "train":
      raise ValueError(f"use_lineage=True (Qwen) supports model_mode='train', got {model_mode!r}.")
    if quant is not None:
      raise ValueError("use_lineage=True (Qwen) does not support MaxText quantization.")
    validate_supported(config)

    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    cfg = qwen_config_from_maxtext(config, mesh)
    param_dtype = jnp.dtype(getattr(config, "weight_dtype", jnp.bfloat16))
    params, shardings = _init_qwen_params_traced(rngs.params(), cfg, mesh, param_dtype=param_dtype)

    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(params)
    sharding_leaves = jax.tree_util.tree_leaves(shardings)
    flat: dict[tuple[str, ...], nnx.Param] = {}
    for (path, leaf), sharding in zip(leaves_with_path, sharding_leaves):
      key = tuple(_path_key(entry) for entry in path)
      flat[key] = nnx.Param(leaf, out_sharding=sharding, eager_sharding=False)
    self._param_paths = tuple(flat)
    self.params = nnx.data(traverse_util.unflatten_dict(flat))

  def lineage_config(self) -> qwen_config.QwenConfig:
    """The Qwen Lineage config of this model."""
    return qwen_config_from_maxtext(self.config, self.mesh)

  def lineage_params(self) -> qwen_params.ModelParams:
    """The parameters as Qwen Lineage's `ModelParams` (array leaves)."""
    from maxtext.experimental.lineage.qwen import params as qwen_params  # pylint: disable=g-import-not-at-top,import-outside-toplevel

    pspecs = qwen_params.get_param_pspecs(self.lineage_config())
    treedef = jax.tree_util.tree_structure(pspecs)
    flat = traverse_util.flatten_dict(self.params)
    if len(flat) != treedef.num_leaves or len(self._param_paths) != treedef.num_leaves:
      raise ValueError(
          f"Parameter leaf count mismatch: state has {len(flat)} leaves, "
          f"paths has {len(self._param_paths)}, treedef expects {treedef.num_leaves}."
      )
    leaves = [flat[path].value for path in self._param_paths]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def loss_and_aux(
    model: QwenLineageTransformer,
    config: Any,
    data: Mapping[str, jax.Array],
    *,
    is_train: bool,
) -> tuple[jax.Array, dict[str, Any]]:
  """`train.loss_fn` of the Qwen Lineage model: returns `(loss, aux)`."""
  from maxtext.experimental.lineage.qwen import model as qwen_model  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from maxtext.experimental.lineage.qwen import params as qwen_params  # pylint: disable=g-import-not-at-top,import-outside-toplevel

  cfg = model.lineage_config()
  if cfg.use_max_async_overlap_transform:
    from maxtext.experimental.lineage.qwen import max_async_overlap_transform  # pylint: disable=g-import-not-at-top,import-outside-toplevel

    max_async_overlap_transform.register_max_async_overlap_transform(cfg)
  param_pspecs = qwen_params.get_param_pspecs(cfg)
  data_pspec = P(cfg.all_sharded_axes, None)

  inputs = data["inputs"]
  targets = data["targets"]
  positions = data["inputs_position"]
  segment_ids = data["inputs_segmentation"]
  target_mask = data["targets_segmentation"]

  def _local_loss_and_weights(
      params_local: qwen_params.ModelParams,
      tokens_local: jax.Array,
      targets_local: jax.Array,
      positions_local: jax.Array,
      segment_ids_local: jax.Array,
      target_mask_local: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    # Use target_mask_local as the active-token mask in chunked_lm_head_loss (0 = padding).
    seg_for_loss = jnp.where(target_mask_local != 0, segment_ids_local, jnp.zeros_like(segment_ids_local))
    mean_loss = qwen_model.local_forward_loss(
        params_local,
        tokens_local,
        targets_local,
        positions_local,
        seg_for_loss,
        cfg,
    )
    local_weights = jnp.sum((target_mask_local != 0).astype(jnp.float32))
    if cfg.num_devices > 1:
      reduce_axes = (
          ("fsdp", "expert", "ep_d2d")
          if (cfg.decouple_ep_d2d and cfg.ep_core_minor)
          else cfg.all_sharded_axes
      )
      total_weights = jax.lax.psum(local_weights, reduce_axes)
    else:
      total_weights = local_weights
    return mean_loss, total_weights

  # Run `value_and_grad` *inside* `shard_map` via `custom_vjp` so the Lineage
  # pipelined-scan custom_vjp sees shard-local tracers directly (matching
  # `model.make_value_and_grad_fn`) rather than differentiating through `shard_map`.
  @jax.custom_vjp
  def _sharded_loss_and_weights(params_tree, toks, tgts, pos, seg, tmask):
    return jax.shard_map(
        _local_loss_and_weights,
        mesh=model.mesh,
        in_specs=(param_pspecs, data_pspec, data_pspec, data_pspec, data_pspec, data_pspec),
        out_specs=(P(), P()),
        check_vma=False,
    )(params_tree, toks, tgts, pos, seg, tmask)

  def _sharded_loss_fwd(params_tree, toks, tgts, pos, seg, tmask):
    def _local_vag(p, tk, tg, ps, sg, tm):
      def _loss_only(p_in):
        return _local_loss_and_weights(p_in, tk, tg, ps, sg, tm)

      (m_loss, t_weights), grads = jax.value_and_grad(_loss_only, has_aux=True)(p)
      return m_loss, t_weights, grads

    m_loss, t_weights, grads = jax.shard_map(
        _local_vag,
        mesh=model.mesh,
        in_specs=(param_pspecs, data_pspec, data_pspec, data_pspec, data_pspec, data_pspec),
        out_specs=(P(), P(), param_pspecs),
        check_vma=False,
    )(params_tree, toks, tgts, pos, seg, tmask)
    return (m_loss, t_weights), (grads,)

  def _sharded_loss_bwd(res, g_out):
    (grads,) = res
    g_loss, _ = g_out
    scaled_grads = jax.tree.map(lambda g: g * g_loss.astype(g.dtype), grads)
    return (scaled_grads, None, None, None, None, None)

  _sharded_loss_and_weights.defvjp(_sharded_loss_fwd, _sharded_loss_bwd)

  mean_loss, total_weights = _sharded_loss_and_weights(
      model.lineage_params(),
      inputs,
      targets,
      positions,
      segment_ids,
      target_mask,
  )

  xent_sum = mean_loss * total_weights
  accumulate = (
      is_train
      and int(getattr(config, "gradient_accumulation_steps", 1)) > 1
      and not getattr(config, "use_tunix_gradient_accumulation", False)
  )
  loss = xent_sum if accumulate else mean_loss
  return loss, {
      "intermediate_outputs": {},
      "xent_sum": xent_sum,
      "z_loss": jnp.asarray(0.0, dtype=jnp.float32),
      "total_weights": total_weights,
      "moe_lb_loss": jnp.asarray(0.0, dtype=jnp.float32),
      "indexer_loss": 0.0,
      "moe_bias_updates": None,
      "mtp_moe_bias_updates": None,
      "mtp_loss": jnp.asarray(0.0, dtype=jnp.float32),
      "batch_stats": None,
      "has_moe_overflow": jnp.bool_(False),
      "diag_bias_values": [],
      "diag_bias_updates": [],
  }


def finalize_accumulated_step(model: QwenLineageTransformer, aux: dict[str, Any]) -> dict[str, Any]:
  """No-op for Qwen Lineage (no loss-free router bias or expert permutation state)."""
  del model
  return aux


def maxtext_restore_target(config: Any, mesh: jax.sharding.Mesh) -> dict[str, Any]:
  """Placeholder for MaxText->Qwen Lineage Orbax checkpoint restore target."""
  del config, mesh
  raise NotImplementedError(
      "Checkpoint restore from MaxText-layout params into QwenLineageTransformer is not yet wired; "
      "use params.convert_maxtext_scannable_block_params / stack_and_shard_model_params."
  )


def restore_params(
    model_state: nnx.State,
    maxtext_params: Mapping[str, Any],
    config: Any,
    mesh: jax.sharding.Mesh,
) -> None:
  """Placeholder for MaxText->Qwen Lineage parameter restore."""
  del model_state, maxtext_params, config, mesh
  raise NotImplementedError(
      "Checkpoint restore from MaxText-layout params into QwenLineageTransformer is not yet wired."
  )
