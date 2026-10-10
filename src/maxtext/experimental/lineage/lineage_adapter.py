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

"""Adapter bridging MaxText configs and trainer hooks to Lineage DeepSeek-V3.

Defines the NNX model `LineageTransformer` whose parameters are stored in
Lineage's layout (`dsv3_model.DSv3Params`), runs the full training/eval step in
`dsv3_model.dsv3_loss_and_aux`, and converts MaxText-layout
`load_parameters_path` checkpoints once at restore time via `maxtext_layout`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools
from typing import Any, TYPE_CHECKING

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
from maxtext.experimental.lineage import dsv3_config
from maxtext.experimental.lineage import dsv3_loss
from maxtext.experimental.lineage import dsv3_sharding
from maxtext.experimental.lineage import quantization
from maxtext.utils import max_logging
import numpy as np

if TYPE_CHECKING:
  from maxtext.experimental.lineage import dsv3_model

_RAGGED_BUFFER_FACTOR = 1.2  # Cross-layer ragged activation bank safety factor.
_DCN_AXIS = dsv3_sharding.DCN_AXIS

# `nnx.State` keys of `LineageTransformer`.
PARAMS_KEY = "params"
ROUTER_BIAS_KEY = "router_bias"
MTP_ROUTER_BIAS_KEY = "mtp_router_bias"
EXPERT_PERMUTATIONS_KEY = "expert_permutations"
MTP_EXPERT_PERMUTATIONS_KEY = "mtp_expert_permutations"

# Linen-style collection name used by MaxText checkpoints for routed gate biases.
MOE_BIAS_COLLECTION = "MoEBiasVar"


def is_native(cfg: Any) -> bool:
  """Whether `cfg` runs the Lineage-native DeepSeek-V3 model (`use_lineage`)."""
  return bool(getattr(cfg, "use_lineage", False))


class LineageRouterBiasVar(nnx.Variable):
  """Routed gate bias of a Lineage MoE stack.

  Loss-free load balancing advances it every training step
  (`dsv3_loss.routed_bias_updates`); it is not trained by the optimizer.
  """


RouterBiasVar = LineageRouterBiasVar


class LineageExpertPermutationVar(nnx.BatchStat):
  """Per-layer int32 expert permutation of a Lineage MoE stack.

  Subclasses `nnx.BatchStat` so params-only checkpoint helpers
  (`_abstract_params`) skip this runtime state while full-state checkpoints
  preserve it.
  """


ExpertPermutationVar = LineageExpertPermutationVar


def build_axis_mapping(
    cfg: Any, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None
) -> Mapping[str, str | tuple[str, ...]]:
  """Builds axis mapping from Lineage logical axes to MaxText physical mesh axes."""
  has_dcn = (_DCN_AXIS in mesh.axis_names) if mesh is not None else (_DCN_AXIS in (getattr(cfg, "mesh_axes", ()) or ()))
  return dsv3_sharding.build_axis_mapping(dict(cfg.logical_axis_rules), has_dcn=has_dcn)


create_physical_device_mesh = dsv3_sharding.create_physical_device_mesh


def create_device_mesh(
    ici_parallelism: Sequence[int],
    devices: Sequence[Any],
    dcn_parallelism: Sequence[int] | None = None,
    allow_split_physical_axes: bool | None = None,
) -> np.ndarray:
  """Maps logical mesh axis i one-to-one onto physical axis i, with mesh_utils fallback."""
  return dsv3_sharding.create_device_mesh(
      ici_parallelism,
      devices,
      dcn_parallelism=dcn_parallelism,
      allow_split_physical_axes=allow_split_physical_axes,
      log_fn=max_logging.log,
  )


def get_capacity_factor(cfg: Any) -> float:
  """Returns the positive capacity factor for Lineage layers from config."""
  if cfg.capacity_factor <= 0:
    raise ValueError(f"capacity_factor must be > 0, got: {cfg.capacity_factor}")
  return float(cfg.capacity_factor)


def get_quant_config(cfg: Any) -> quantization.QuantConfig | None:
  """Returns the Lineage `QuantConfig` selected by `cfg.lineage_quantization`."""
  if cfg.lineage_quantization not in dsv3_config.QUANT_CONFIGS:
    raise ValueError(
        "lineage_quantization must be one of"
        f" {sorted(dsv3_config.QUANT_CONFIGS)}, got:"
        f" {cfg.lineage_quantization!r}"
    )
  return dsv3_config.QUANT_CONFIGS[cfg.lineage_quantization]


def uses_megatron_seq_aux_loss(cfg: Any) -> bool:
  """Mirrors `moe.uses_megatron_seq_aux_loss`."""
  return bool(
      getattr(cfg, "moe_use_megatron_seq_aux_loss", False)
      and getattr(cfg, "routed_score_func", "") == "sigmoid"
      and not getattr(cfg, "te_moe_block", False)
  )


def _router_dtype(cfg: Any) -> jax.typing.DTypeLike | None:
  """Lineage expert-selection dtype: fp32 under `float32_gate_logits`."""
  return jnp.float32 if getattr(cfg, "float32_gate_logits", False) else None


def _router_bias_dtype(cfg: Any) -> jax.typing.DTypeLike | None:
  """Lineage routed-bias dtype: fp32 under `float32_gate_logits`, else `cfg.dtype`."""
  return _router_dtype(cfg) or getattr(cfg, "dtype", None)


def _mesh_scope(mesh: jax.sharding.Mesh):
  """Activates `mesh`'s abstract mesh for Lineage's explicit-sharding ops.

  Valid inside and outside `jax.jit` (unlike `jax.set_mesh`); MaxText traces
  model creation under `nnx.eval_shape` / `jax.jit` without a mesh context.

  Args:
    mesh: The device mesh.

  Returns:
    A context manager that sets `mesh.abstract_mesh`.
  """
  return jax.sharding.use_abstract_mesh(mesh.abstract_mesh)


def _unsupported_native_options(cfg: Any) -> list[str]:
  """Config options of the MaxText trainer that the Lineage model does not implement."""
  problems = []

  def require(ok: bool, message: str):
    if not ok:
      problems.append(message)

  get = functools.partial(getattr, cfg)
  require(get("mtp_num_layers", 0) >= 0, "mtp_num_layers must be >= 0")
  require(get("shared_experts", 1) == 1, "shared_experts must be 1")
  require(
      not get("quantization", None),
      "quantization (AQT/Qwix) is unsupported; use lineage_quantization",
  )
  require(
      get("gradient_accumulation_steps", 1) == 1,
      "gradient_accumulation_steps must be 1",
  )
  require(get("num_vocab_tiling", 1) <= 1, "num_vocab_tiling must be 1")
  require(
      not get("moe_dropless_fallback", None),
      "moe_dropless_fallback is unsupported",
  )
  require(
      not get("log_required_ragged_buffer_factor", False),
      "log_required_ragged_buffer_factor is unsupported",
  )
  require(
      not get("moe_log_max_load_ratio", False),
      "moe_log_max_load_ratio is unsupported",
  )
  require(not get("use_indexer", False), "use_indexer is unsupported")
  require(
      get("training_objective", "causal_lm") == "causal_lm",
      "training_objective must be causal_lm",
  )
  require(not get("use_multimodal", False), "use_multimodal is unsupported")
  require(
      not (get("enable_dropout", False) and get("dropout_rate", 0.0) > 0.0),
      "dropout is unsupported",
  )
  require(not get("te_moe_block", False), "te_moe_block is unsupported")
  require(not get("use_qk_clip", False), "use_qk_clip is unsupported")
  require(not get("enable_diloco", False), "enable_diloco is unsupported")
  require(
      not getattr(get("lora", None), "enable_lora", False),
      "lora is unsupported",
  )
  require(
      get("final_logits_soft_cap", None) is None,
      "final_logits_soft_cap is unsupported",
  )
  require(
      not (get("logits_via_embedding", False) and not get("normalize_embedding_logits", True)),
      "logits_via_embedding requires normalize_embedding_logits",
  )
  require(
      str(get("rope_type", "yarn")).lower().endswith("yarn"),
      "rope_type must be yarn",
  )
  require(get("rope_interleave", True), "rope_interleave must be true")
  require(
      get("routed_score_func", "sigmoid") == "sigmoid",
      "routed_score_func must be sigmoid",
  )
  require(get("scan_layers", True), "scan_layers must be true")
  require(
      not get("parameter_memory_host_offload", False),
      "parameter_memory_host_offload is unsupported",
  )
  return problems


def validate_supported(cfg: Any) -> None:
  """Raises `ValueError` if `cfg` combines `use_lineage` with unsupported options."""
  problems = _unsupported_native_options(cfg)
  if problems:
    raise ValueError("use_lineage=True does not support this config: " + "; ".join(problems) + ".")


def dsv3_config_from_maxtext(cfg: Any, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> dsv3_config.DSv3Config:
  """Builds the Lineage `DSv3Config` of a MaxText config.

  Args:
    cfg: MaxText config (`pyconfig.HyperParameters` or `types.MaxTextConfig`).
    mesh: The device mesh.

  Returns:
    The validated config.
  """
  num_decoder_layers = int(cfg.num_decoder_layers)
  num_dense_layers = int(cfg.first_num_dense_layers)
  router_dtype = _router_dtype(cfg)
  router_bias_dtype = _router_bias_dtype(cfg)
  model = dsv3_config.DSv3ModelConfig(
      vocab_size=int(cfg.vocab_size),
      emb_dim=int(cfg.emb_dim),
      num_dense_layers=num_dense_layers,
      num_sparse_layers=num_decoder_layers - num_dense_layers,
      num_mtp_layers=int(getattr(cfg, "mtp_num_layers", 0)),
      num_query_heads=int(cfg.num_query_heads),
      num_kv_heads=int(cfg.num_kv_heads),
      q_lora_rank=int(cfg.q_lora_rank),
      kv_lora_rank=int(cfg.kv_lora_rank),
      qk_head_dim=int(cfg.qk_nope_head_dim),
      rope_head_dim=int(cfg.qk_rope_head_dim),
      v_head_dim=int(cfg.v_head_dim),
      mscale=float(cfg.mscale),
      mlp_dim=int(cfg.mlp_dim),
      moe_mlp_dim=int(cfg.moe_mlp_dim),
      num_experts=int(cfg.num_experts),
      num_experts_per_tok=int(cfg.num_experts_per_tok),
      n_routing_groups=int(cfg.n_routing_groups),
      topk_routing_group=int(cfg.topk_routing_group),
      routed_scaling_factor=float(cfg.routed_scaling_factor),
      rope_theta=int(cfg.rope_max_timescale),
      max_position_embeddings=int(cfg.max_position_embeddings),
      original_max_position_embeddings=int(cfg.original_max_position_embeddings),
      rope_factor=int(cfg.rope_factor),
      beta_fast=int(cfg.beta_fast),
      beta_slow=int(cfg.beta_slow),
      norm_epsilon=float(cfg.normalization_layer_epsilon),
      tied_head=bool(getattr(cfg, "logits_via_embedding", False)),
      iota_embed=bool(getattr(cfg, "use_iota_embed", False)),
      dtype=jnp.dtype(cfg.dtype),
      weight_dtype=jnp.dtype(cfg.weight_dtype),
      router_dtype=None if router_dtype is None else jnp.dtype(router_dtype),
      router_bias_dtype=jnp.dtype(router_bias_dtype),
      head_dot_in_fp32=bool(getattr(cfg, "logits_dot_in_fp32", False)),
      cast_logits_to_fp32=bool(getattr(cfg, "cast_logits_to_fp32", True)),
  )
  axis_mapping = dsv3_sharding.build_axis_mapping(dict(cfg.logical_axis_rules), has_dcn=_DCN_AXIS in mesh.axis_names)
  sharding = dsv3_sharding.sharding_config(
      mesh,
      axis_mapping,
      emb_dim=model.emb_dim,
      qk_head_dim=model.qk_head_dim,
      rope_head_dim=model.rope_head_dim,
      v_head_dim=model.v_head_dim,
  )
  expert_permutation = getattr(cfg, "lineage_expert_permutation", None)
  kernels = dsv3_config.DSv3KernelConfig(
      max_target_length=int(cfg.max_target_length),
      sa_block_q=int(cfg.sa_block_q),
      sa_block_kv=int(cfg.sa_block_kv),
      sa_block_kv_compute=int(cfg.sa_block_kv_compute),
      sa_block_q_dkv=int(cfg.sa_block_q_dkv),
      sa_block_kv_dkv=int(cfg.sa_block_kv_dkv),
      sa_block_kv_dkv_compute=int(cfg.sa_block_kv_dkv_compute),
      sa_q_layout=cfg.sa_q_layout,
      sa_k_layout=cfg.sa_k_layout,
      sa_v_layout=cfg.sa_v_layout,
      capacity_factor=get_capacity_factor(cfg),
      ragged_buffer_factor=_RAGGED_BUFFER_FACTOR,
      quant=get_quant_config(cfg),
      max_async_overlap_transform=True,
      expert_permutation=("none" if expert_permutation is None else str(expert_permutation)),
  )
  routed_bias_update_rate = float(cfg.routed_bias_update_rate) if getattr(cfg, "routed_bias", False) else 0.0
  training = dsv3_config.DSv3TrainingConfig(
      load_balance_loss_weight=float(getattr(cfg, "load_balance_loss_weight", 0.0) or 0.0),
      megatron_seq_aux_loss=uses_megatron_seq_aux_loss(cfg),
      routed_bias_update_rate=routed_bias_update_rate,
      z_loss_multiplier=float(getattr(cfg, "z_loss_multiplier", 0.0) or 0.0),
      mtp_loss_scaling_factor=float(getattr(cfg, "mtp_loss_scaling_factor", 0.1)),
      mtp_eval_target_module=int(getattr(cfg, "mtp_eval_target_module", 0)),
      mtp_reuse_input_embedding=bool(getattr(cfg, "mtp_reuse_input_embedding", False)),
  )
  lineage_cfg = dsv3_config.DSv3Config(model=model, sharding=sharding, kernels=kernels, training=training)
  lineage_cfg.validate()
  return lineage_cfg


from_maxtext_config = dsv3_config_from_maxtext


def _path_key(entry: Any) -> str:
  """`nnx.State` key of one `jax.tree_util` key-path entry."""
  if isinstance(entry, jax.tree_util.GetAttrKey):
    return entry.name
  if isinstance(entry, jax.tree_util.SequenceKey):
    return str(entry.idx)
  if isinstance(entry, jax.tree_util.DictKey):
    return str(entry.key)
  raise TypeError(f"Unsupported key path entry: {entry!r}")


def _param_tree(
    params: dsv3_model.DSv3Params,
    shardings: dsv3_model.DSv3Params,
) -> tuple[dict[str, Any], tuple[tuple[str, ...], ...]]:
  """Wraps `params`' leaves in `nnx.Param`s, nested as dicts keyed by field names."""
  leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(params)
  sharding_leaves = jax.tree.leaves(shardings)
  if len(sharding_leaves) != len(leaves_with_path):
    raise ValueError(f"Got {len(sharding_leaves)} parameter shardings for" f" {len(leaves_with_path)} parameters.")
  flat = {}
  for (path, leaf), sharding in zip(leaves_with_path, sharding_leaves):
    key = tuple(_path_key(entry) for entry in path)
    flat[key] = nnx.Param(leaf, out_sharding=sharding, eager_sharding=False)
  return traverse_util.unflatten_dict(flat), tuple(flat)


def _state_var(
    var_type: type[nnx.Variable],
    value: jax.Array,
    sharding: jax.sharding.NamedSharding,
) -> nnx.Variable:
  return var_type(value, out_sharding=sharding, eager_sharding=False)


class LineageTransformer(nnx.Module):
  """MaxText NNX model whose parameters and forward pass live in Lineage.

  Parameters are `nnx.Param`s in Lineage's layout (`dsv3_model.DSv3Params`),
  nested under `params` by dataclass field name (MTP depths by index), so a
  training step does no layout translation. The routed gate biases and expert
  permutations (`dsv3_model.DSv3MutableState`) are non-trainable Variables
  (`router_bias`, `mtp_router_bias`, `expert_permutations`,
  `mtp_expert_permutations`) that `loss_and_aux` advances in training steps.
  Every Variable carries its physical `NamedSharding` as `out_sharding`, which
  MaxText's abstract-state machinery uses verbatim.

  The constructor matches `models.Transformer` as called by
  `model_creation_utils.get_transformer_model`.
  """

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
      raise ValueError(f"use_lineage=True supports model_mode='train', got {model_mode!r}.")
    if quant is not None:
      raise ValueError("use_lineage=True does not support MaxText quantization.")
    validate_supported(config)
    from maxtext.experimental.lineage import dsv3_model  # pylint: disable=g-import-not-at-top,import-outside-toplevel

    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    cfg = dsv3_config_from_maxtext(config, mesh)
    param_shardings, state_shardings = dsv3_model.dsv3_param_shardings(cfg, mesh)
    with _mesh_scope(mesh):
      params, state = dsv3_model.init_dsv3_params(rngs.params(), cfg, mesh)
    params_dict, self._param_paths = _param_tree(params, param_shardings)
    self.params = nnx.data(params_dict)
    assert state.router_bias is not None
    assert state_shardings.router_bias is not None
    self.router_bias = _state_var(RouterBiasVar, state.router_bias, state_shardings.router_bias)
    self.mtp_router_bias = nnx.data(
        [
            _state_var(RouterBiasVar, bias, sharding)
            for bias, sharding in zip(state.mtp_router_bias, state_shardings.mtp_router_bias)
        ]
    )
    if state.expert_permutations is not None:
      assert state_shardings.expert_permutations is not None
      self.expert_permutations = _state_var(
          ExpertPermutationVar,
          state.expert_permutations,
          state_shardings.expert_permutations,
      )
      self.mtp_expert_permutations = nnx.data(
          [
              _state_var(ExpertPermutationVar, permutation, sharding)
              for permutation, sharding in zip(
                  state.mtp_expert_permutations,
                  state_shardings.mtp_expert_permutations,
              )
          ]
      )
    else:
      self.expert_permutations = None
      self.mtp_expert_permutations = nnx.data([])

  def lineage_config(self) -> dsv3_config.DSv3Config:
    """The Lineage config of this model."""
    return dsv3_config_from_maxtext(self.config, self.mesh)

  def lineage_params(self) -> dsv3_model.DSv3Params:
    """The parameters as Lineage's `DSv3Params` (array leaves)."""
    from maxtext.experimental.lineage import dsv3_model  # pylint: disable=g-import-not-at-top,import-outside-toplevel

    param_shardings, _ = dsv3_model.dsv3_param_shardings(self.lineage_config(), self.mesh)
    treedef = jax.tree_util.tree_structure(param_shardings)
    flat = traverse_util.flatten_dict(self.params)
    if len(flat) != treedef.num_leaves or len(self._param_paths) != treedef.num_leaves:
      raise ValueError(
          f"Parameter leaf count mismatch: state has {len(flat)} leaves,"
          f" paths has {len(self._param_paths)}, treedef expects"
          f" {treedef.num_leaves}."
      )
    leaves = [flat[path].value for path in self._param_paths]
    return jax.tree_util.tree_unflatten(treedef, leaves)

  def lineage_state(self) -> dsv3_model.DSv3MutableState:
    """The mutable state as Lineage's `DSv3MutableState` (array leaves)."""
    from maxtext.experimental.lineage import dsv3_model  # pylint: disable=g-import-not-at-top,import-outside-toplevel

    return dsv3_model.DSv3MutableState(
        router_bias=self.router_bias.value,
        mtp_router_bias=tuple(bias.value for bias in self.mtp_router_bias),
        expert_permutations=(None if self.expert_permutations is None else self.expert_permutations.value),
        mtp_expert_permutations=tuple(permutation.value for permutation in self.mtp_expert_permutations),
    )

  def set_lineage_state(self, state: dsv3_model.DSv3MutableState) -> None:
    """Writes a Lineage state into the state Variables.

    Args:
      state: The state to write, e.g. `DSv3LossAux.new_state`.
    """
    assert state.router_bias is not None
    self.router_bias.value = state.router_bias
    for var, bias in zip(self.mtp_router_bias, state.mtp_router_bias, strict=True):
      var.value = bias
    if self.expert_permutations is not None:
      assert state.expert_permutations is not None
      self.expert_permutations.value = state.expert_permutations
      for var, permutation in zip(
          self.mtp_expert_permutations,
          state.mtp_expert_permutations,
          strict=True,
      ):
        var.value = permutation

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      **unused_kwargs: Any,
  ) -> jax.Array:
    """Returns the logits `[batch, seq, vocab]` (no MTP, no losses)."""
    from maxtext.experimental.lineage import dsv3_model  # pylint: disable=g-import-not-at-top,import-outside-toplevel

    cfg = self.lineage_config()
    batch = dsv3_model.DSv3Batch(
        inputs=decoder_input_tokens,
        targets=decoder_input_tokens,
        positions=decoder_positions,
        segment_ids=decoder_segment_ids,
        target_mask=jnp.ones_like(decoder_input_tokens),
    )
    with _mesh_scope(self.mesh):
      fwd = dsv3_model.dsv3_forward(
          self.lineage_params(),
          self.lineage_state(),
          batch,
          cfg,
          mesh=self.mesh,
      )
      return dsv3_model.dsv3_logits(self.lineage_params(), fwd.hidden, cfg, mesh=self.mesh)


def loss_and_aux(
    model: LineageTransformer,
    config: Any,
    data: Mapping[str, jax.Array],
    *,
    is_train: bool,
) -> tuple[jax.Array, dict[str, Any]]:
  """`train.loss_fn` of the Lineage model: the loss and MaxText's `aux` dict.

  Runs `dsv3_model.dsv3_loss_and_aux` on the micro-batch and, in training
  steps, advances the routed gate biases and expert permutations in `model`
  (they reach the train state through `train_step`'s non-parameter state).

  Args:
    model: The model.
    config: MaxText config.
    data: The batch (`inputs`, `targets`, `inputs_position`,
      `inputs_segmentation`, `targets_segmentation`).
    is_train: Training step (adds the MTP loss, updates the state).

  Returns:
    The loss and the `aux` dict with the keys `train_step` / `eval_step` read:
    `intermediate_outputs`, `xent_sum`, `z_loss`, `total_weights`,
    `moe_lb_loss`, `indexer_loss`, `moe_bias_updates`, `mtp_moe_bias_updates`,
    `mtp_loss`, `batch_stats`, `has_moe_overflow`, plus `diag_bias_values` /
    `diag_bias_updates` for `log_step_diagnostics`.
  """
  from maxtext.experimental.lineage import dsv3_model  # pylint: disable=g-import-not-at-top,import-outside-toplevel

  cfg = model.lineage_config()
  batch = dsv3_model.DSv3Batch(
      inputs=data["inputs"],
      targets=data["targets"],
      positions=data["inputs_position"],
      segment_ids=data["inputs_segmentation"],
      target_mask=data["targets_segmentation"],
  )
  with _mesh_scope(model.mesh):
    loss, aux = dsv3_model.dsv3_loss_and_aux(
        model.lineage_params(),
        model.lineage_state(),
        batch,
        cfg,
        mesh=model.mesh,
        train=is_train,
    )
  if is_train:
    model.set_lineage_state(aux.new_state)

  intermediate_outputs: dict[str, Any] = {}
  if not is_train and config.mtp_eval_target_module > 0:
    # `calculate_mtp_acceptance_rate` reads the MTP predictions of
    # `mtp_eval_target_module` at MaxText's sown paths.
    assert aux.mtp_preds is not None and aux.mtp_mask is not None
    intermediate_outputs["mtp_acceptance"] = {"mtp_block": {"mtp_preds": aux.mtp_preds, "mtp_mask": aux.mtp_mask}}
    intermediate_outputs["logits"] = aux.logits
  new_state = aux.new_state
  diag_bias_values = []
  diag_bias_updates = []
  if is_train and aux.router_bias_updates is not None:
    diag_bias_values.append(new_state.router_bias)
    diag_bias_updates.append(aux.router_bias_updates)
    diag_bias_values.extend(new_state.mtp_router_bias)
    diag_bias_updates.extend(aux.mtp_router_bias_updates)
  return loss, {
      "intermediate_outputs": intermediate_outputs,
      "xent_sum": aux.xent_sum,
      "z_loss": aux.z_loss_sum / (aux.total_weights + dsv3_loss.EPS),
      "total_weights": aux.total_weights,
      "moe_lb_loss": aux.moe_lb_loss,
      "indexer_loss": 0.0,
      "moe_bias_updates": None,
      "mtp_moe_bias_updates": None,
      "mtp_loss": aux.mtp_loss,
      "batch_stats": None,
      "has_moe_overflow": jnp.bool_(False),
      "diag_bias_values": diag_bias_values,
      "diag_bias_updates": diag_bias_updates,
  }


def maxtext_abstract_param_tree(config: Any, mesh: jax.sharding.Mesh) -> dict[str, Any]:
  """The MaxText-layout parameter tree (`jax.ShapeDtypeStruct`s) of `config`.

  Args:
    config: MaxText config.
    mesh: The device mesh.

  Returns:
    See `maxtext_layout.maxtext_abstract_param_tree`.
  """
  from maxtext.experimental.lineage.interop import maxtext_layout  # pylint: disable=g-import-not-at-top,import-outside-toplevel

  return maxtext_layout.maxtext_abstract_param_tree(
      dsv3_config_from_maxtext(config, mesh),
      mesh,
      param_scan_axis=config.param_scan_axis,
  )


def maxtext_restore_target(config: Any, mesh: jax.sharding.Mesh) -> dict[str, Any]:
  """The Linen-style Orbax restore target of a MaxText-layout checkpoint.

  Splits the MaxText-layout abstract parameter tree into `"params"` and (when
  `routed_bias` is enabled) `"MoEBiasVar"` for the `("gate", "bias")` leaves.
  `checkpointing.load_params_from_path` restores both legacy checkpoints that
  store routed gate biases under `params/params/...` (via
  `_alias_legacy_collections`) and checkpoints that store them under
  `params/MoEBiasVar/...`.

  Args:
    config: MaxText config.
    mesh: The device mesh.

  Returns:
    The restore target: `{"params": ..., "MoEBiasVar": ...}`.
  """
  tree = maxtext_abstract_param_tree(config, mesh)
  flat = traverse_util.flatten_dict(tree)
  params_flat = {}
  bias_flat = {}
  for path, val in flat.items():
    if path[-2:] == ("gate", "bias"):
      bias_flat[path] = val
    else:
      params_flat[path] = val
  target: dict[str, Any] = {"params": traverse_util.unflatten_dict(params_flat)}
  if bias_flat and getattr(config, "routed_bias", True):
    target[MOE_BIAS_COLLECTION] = traverse_util.unflatten_dict(bias_flat)
  return target


def _merge_restored_collections(restored: Mapping[str, Any]) -> dict[str, Any]:
  """Merges top-level collection dicts (`params`, `MoEBiasVar`) into one MaxText tree."""
  if hasattr(restored, "to_pure_dict"):
    restored = restored.to_pure_dict()
  if isinstance(restored, Mapping) and "model" in restored and len(restored) == 1:
    restored = restored["model"]
    if hasattr(restored, "to_pure_dict"):
      restored = restored.to_pure_dict()
  if isinstance(restored, Mapping) and "params" in restored:
    merged_flat = {}
    for collection in restored.values():
      if isinstance(collection, Mapping) and collection:
        merged_flat.update(traverse_util.flatten_dict(collection))
    return traverse_util.unflatten_dict(merged_flat)
  return dict(restored)


def lineage_state_pure_dict(params: dsv3_model.DSv3Params, state: dsv3_model.DSv3MutableState) -> dict[str, Any]:
  """`LineageTransformer`'s pure-dict state (arrays) of Lineage params and state.

  Expert permutations are runtime state and stay at the model's init.

  Args:
    params: Lineage parameters.
    state: Lineage mutable state (routed gate biases).

  Returns:
    A pure dict for `nnx.replace_by_pure_dict` on the model state.
  """
  flat = {tuple(_path_key(entry) for entry in path): leaf for path, leaf in jax.tree_util.tree_leaves_with_path(params)}
  pure: dict[str, Any] = {PARAMS_KEY: traverse_util.unflatten_dict(flat)}
  if state.router_bias is not None:
    pure[ROUTER_BIAS_KEY] = state.router_bias
  if state.mtp_router_bias:
    pure[MTP_ROUTER_BIAS_KEY] = {i: bias for i, bias in enumerate(state.mtp_router_bias)}
  return pure


def restore_params(
    model_state: nnx.State,
    maxtext_params: Mapping[str, Any],
    config: Any,
    mesh: jax.sharding.Mesh,
) -> None:
  """Overlays a restored MaxText-layout parameter tree onto a Lineage model state.

  Merges any top-level collection dicts (`params`, `MoEBiasVar`), converts the
  whole tree once (`maxtext_layout.convert_maxtext_params`) into Lineage's
  layout and shardings, and replaces the parameters and routed gate biases of
  `model_state` in place; everything else keeps its init value.

  Args:
    model_state: `nnx.State` of a `LineageTransformer`.
    maxtext_params: Restored MaxText parameters (either a two-collection dict
      from `maxtext_restore_target` or a bare MaxText parameter tree).
    config: MaxText config.
    mesh: The device mesh.
  """
  from maxtext.experimental.lineage.interop import maxtext_layout  # pylint: disable=g-import-not-at-top,import-outside-toplevel

  cfg = dsv3_config_from_maxtext(config, mesh)
  tree = _merge_restored_collections(maxtext_params)
  params, state = maxtext_layout.convert_maxtext_params(tree, cfg, mesh, param_scan_axis=config.param_scan_axis)
  nnx.replace_by_pure_dict(model_state, lineage_state_pure_dict(params, state))
