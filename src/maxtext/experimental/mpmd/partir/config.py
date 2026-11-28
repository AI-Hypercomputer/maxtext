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
"""Builds MaxText configs from :class:`hparams.Hparams`.

Every entry point in this package funnels through :func:`build_maxtext_config`
so that the trainer, the lowering check, the parity checker and the unit tests
all see byte-identical MaxText settings, and the MPMD and SPMD paths cannot
drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import types
from typing import Any

from maxtext.common.common_types import DecoderBlockType
from maxtext.common.common_types import get_weight_dtype
from maxtext.common.common_types import ShardMode
from maxtext.configs import pyconfig
from maxtext.layers import quantizations
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

from maxtext.experimental.mpmd.partir import hparams as hparams_lib

# Settings that make a run reproducible and self-contained. They are applied to
# every config this module produces, so an MPMD run and its SPMD reference can
# never disagree on, say, dropout.
_DETERMINISM_OVERRIDES = {
    # Pipelining splits the model into separately traced computations; scanned
    # layers would hide the per-layer boundaries PartIR needs to see.
    "scan_layers": "false",
    "enable_checkpointing": "false",
    "enable_dropout": "false",
    "dropout_rate": "0.0",
    # Synthetic data: this package benchmarks the pipeline, not the input
    # pipeline.
    "dataset_type": "synthetic",
    "enable_goodput_recording": "false",
    "monitor_goodput": "false",
    # `use_iota_embed` avoids a gather that would otherwise dominate the tiny
    # embedding stage and distort the profile.
    "use_iota_embed": "true",
    # Required so that the `base_*` overrides below are honoured on top of a
    # named model config.
    "override_model_config": "true",
}


# What MaxText's model or trainer (trainers/pre_train/train.py) adds that the MPMD step leaves out, by setting: when it
# is on, what it adds, and the value that turns it off. The MPMD model builds neither an MTP block nor a sinusoidal
# position embedding, and its loss collects none of the auxiliary terms MaxText's layers record; spmd_baseline's run of
# MaxText's trainer applies all of them. Each condition is the one MaxText's trainer applies the term under.
_LEFT_OUT = types.MappingProxyType(
    {
        "mtp_num_layers": (lambda cfg: cfg.mtp_num_layers > 0, "a multi-token-prediction block and its loss", "0"),
        "use_untrainable_positional_embedding": (
            lambda cfg: cfg.use_untrainable_positional_embedding,
            "a sinusoidal position embedding",
            "false",
        ),
        "load_balance_loss_weight": (
            lambda cfg: cfg.num_experts > 1 and cfg.load_balance_loss_weight > 0,
            "the MoE load-balance loss",
            "0",
        ),
        "z_loss_multiplier": (lambda cfg: cfg.z_loss_multiplier != 0, "the z-loss", "0"),
        "routed_bias_update_rate": (
            lambda cfg: cfg.routed_bias and cfg.routed_bias_update_rate > 0,
            "the routing-bias updates",
            "0",
        ),
        "indexer_loss_scaling_factor": (
            lambda cfg: cfg.use_indexer and cfg.indexer_loss_scaling_factor > 0,
            "the indexer loss",
            "0",
        ),
    }
)


def build_maxtext_config(
    hparams: hparams_lib.Hparams,
    *,
    overrides: Mapping[str, Any] | None = None,
    base_config_path: str | None = None,
) -> Any:
  """Materialises the MaxText config described by `hparams`.

  Args:
    hparams: Hyperparameters to translate.
    overrides: Extra ``key=value`` MaxText settings applied last, i.e. they win
      over everything this function derives. Used by tests.
    base_config_path: Base YAML to start from. Defaults to MaxText's
      ``base.yml``.

  Returns:
    The initialised MaxText config object (``pyconfig.HyperParameters``).

  Raises:
    ValueError: If the model is not one the MPMD model builds as MaxText
      describes it (see :func:`_check_buildable`).
  """
  base_yaml = base_config_path or os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
  # `ici_pipeline_parallelism` below is only MaxText's bookkeeping (the stage
  # count `models` reads back and `pipeline_parallel_layers`); MaxText's own
  # pipeline never runs. MaxText refuses `remat_policy=none` together with
  # pipeline parallelism, because its pipeline needs checkpointing, so it is
  # handed a policy it accepts. The MPMD model's own choice,
  # `hparams.mpmd_remat_policy`, goes to the model itself
  # (`models.PipelineGraphDef.remat_policy`), not through this config.
  maxtext_remat_policy = hparams.mpmd_remat_policy if hparams.mpmd_remat_policy != "none" else "full"

  # (`hparams` has checked that the model config exists: pyconfig would only
  # log a warning without it and measure base.yml's model under its name.)
  argv: list[str] = [
      # pyconfig treats argv[0] as the program name and ignores it.
      "maxtext_mpmd_partir",
      base_yaml,
      f"model_name={hparams.maxtext_model_name}",
      f"base_num_decoder_layers={hparams.num_layers}",
      f"per_device_batch_size={hparams.per_device_batch_size}",
      f"max_target_length={hparams.max_target_length}",
      f"ici_pipeline_parallelism={hparams.num_stages}",
      "dcn_pipeline_parallelism=1",
      # NB: `gradient_accumulation_steps` is deliberately NOT set here.
      # MaxText reads it as "replay this batch N times before applying the
      # optimizer", so it multiplies both the global batch and the analytic
      # FLOP count (`calculate_tflops_training_per_device`) by N. PartIR's
      # microbatches are the opposite: they *split* one global batch across
      # the pipeline. Setting it to `num_microbatches` therefore left
      # tokens/s correct but inflated TFLOP/s by exactly `num_microbatches`.
      f"remat_policy={maxtext_remat_policy}",
      f"learning_rate={hparams.learning_rate}",
      f"init_weights_seed={hparams.seed}",
  ]
  if hparams.emb_dim > 0:
    argv.append(f"base_emb_dim={hparams.emb_dim}")
  if hparams_lib.is_pure_moe_model(hparams.maxtext_model_name, hparams.first_num_dense_layers):
    moe_dim = hparams.moe_mlp_dim if hparams.moe_mlp_dim > 0 else hparams.mlp_dim
    if moe_dim > 0:
      argv.append(f"base_mlp_dim={moe_dim}")
      argv.append(f"base_moe_mlp_dim={moe_dim}")
  else:
    if hparams.mlp_dim > 0:
      argv.append(f"base_mlp_dim={hparams.mlp_dim}")
    if hparams.moe_mlp_dim > 0:
      argv.append(f"base_moe_mlp_dim={hparams.moe_mlp_dim}")
  if hparams.vocab_size > 0:
    argv.append(f"vocab_size={hparams.vocab_size}")

  argv += [f"{key}={value}" for key, value in _DETERMINISM_OVERRIDES.items()]

  if hparams.fsdp > 0:
    argv.append(f"ici_fsdp_parallelism={hparams.fsdp}")
  # Expert parallelism and token dispatch: one table shared with
  # `spmd_baseline` so both sides route tokens to experts identically.
  argv += [f"{key}={value}" for key, value in hparams_lib.moe_dispatch_overrides(hparams).items()]

  decoder_block = hparams.decoder_block
  if hparams.maxtext_model_name == "default":
    # `simple` and `default` build the simple_mlp block, not base.yml's. A
    # named model's config gives its block to pyconfig, and _check_buildable
    # checks what pyconfig resolved.
    argv.append(f"decoder_block={decoder_block}")
  pipeline_layers = (hparams.num_layers // hparams.num_stages) * hparams.num_stages
  if decoder_block not in ("simple", "simple_mlp") or pipeline_layers != hparams.num_layers:
    # Tell MaxText that *all* layers (or the largest stage-divisible prefix
    # when asymmetric stage_layer_counts has a sum not divisible by num_stages)
    # participate in the pipeline; the MPMD layer-to-mesh assignment is what
    # actually places all `hparams.num_layers` layers.
    argv.append(f"pipeline_parallel_layers={pipeline_layers}")

  if decoder_block == "deepseek" or hparams_lib.uses_yarn_rope(hparams.maxtext_model_name):
    # Several MaxText model configs (DeepSeek, Kimi-K2) ask for YaRN RoPE,
    # which MaxText implements with complex
    # arithmetic: it precomputes `freqs_cis = exp(i*theta)` as a `complex64`
    # table and rotates by complex multiplication (`layers/embeddings.py`,
    # `YarnRotaryEmbedding`). PartIR MPMD cannot partition that --
    # `MarkFragmentReservedMemoryPass::AddLiveValue` calls
    # `mlir::Type::getIntOrFloatBitWidth()` on every live value and a
    # `tensor<...xcomplex<f32>>` trips its assertion, crashing the compiler:
    #
    #   assert.h assertion failed at mlir/lib/IR/Types.cpp:125 in
    #   getIntOrFloatBitWidth(): isIntOrFloat() && "only integers and floats
    #   have a bitwidth"
    #
    # `rope_pairwise` does not help: it avoids complex *multiplication* but
    # still gathers from the complex table. So pin the standard real-valued
    # sin/cos RoPE. It changes the positional encoding, but not the parameter
    # count, the FLOP count, or anything else the pipeline benchmark measures
    # -- and `spmd_baseline` pins exactly the same value, so both sides stay
    # comparable. Revisit if PartIR learns to handle complex types.
    argv.append("rope_type=default")
  if decoder_block == "deepseek" and hparams.first_num_dense_layers >= 0:
    argv.append(f"first_num_dense_layers={hparams.first_num_dense_layers}")

  argv += [f"{key}={value}" for key, value in hparams_lib.float32_gmm_tiles(hparams).items()]
  argv += list(hparams.maxtext_overrides)
  if overrides:
    argv += [f"{key}={value}" for key, value in overrides.items()]

  cfg = pyconfig.initialize(argv)
  _check_buildable(cfg, hparams)
  return cfg


def _check_buildable(cfg: Any, hparams: hparams_lib.Hparams) -> None:
  """Refuses configs that ``models`` would not build as MaxText describes them.

  ``models.PipelinedTransformer`` builds MaxText's token embedding, decoder
  layers (of :data:`hparams.SUPPORTED_DECODER_BLOCKS`) and untied output head
  (final norm, ``logits_dense``) in MaxText's default form, and
  ``train_step.compute_loss`` trains them on the token cross-entropy alone. A
  model or objective that needs more would otherwise train as something else,
  silently, and ``spmd_baseline`` would compare it with the real one.

  Raises:
    ValueError: For an unsupported decoder block (or one other than the
      hyperparameters derived their settings for), a model MaxText builds with
      its m3 backend, tied embeddings, a learned position embedding,
      hyper-connections outside the deepseek block, quantization, explicit
      sharding, an output-head setting the MPMD model does not mirror, or a
      setting that adds to MaxText's model or loss what the MPMD step leaves
      out (:data:`_LEFT_OUT`).
  """
  name = hparams.model_name
  unsupported = hparams_lib.unsupported_decoder_block(name, cfg.decoder_block.value)
  if unsupported:
    raise ValueError(unsupported)
  if cfg.decoder_block.value != hparams.decoder_block:
    raise ValueError(
        f"{name}: MaxText resolved decoder_block={cfg.decoder_block.value}, but the settings config.py derives (and"
        f" spmd_baseline prints) are for the model config's decoder_block={hparams.decoder_block}."
    )
  if cfg.logits_via_embedding:
    raise ValueError(
        f"{name} ties its output projection to the embedding (logits_via_embedding), which the MPMD model does not"
        " build: pass --maxtext_overrides=logits_via_embedding=false to run its untied variant (spmd_baseline then"
        " runs the same one)."
    )
  refused = {
      # MaxText's create_model builds such a model with its separate m3
      # backend, not with NNXDecoder's layers.
      "use_m3_model": cfg.use_m3_model,
      # A learned position embedding (GPT-3).
      "trainable_position_size": cfg.trainable_position_size > 0,
      # Hyper-connection streams: `models.forward` expands and reduces them as
      # MaxText's decoder does for the deepseek block, whose layers take them;
      # the other supported layers do not.
      "mhc_expansion_rate": cfg.mhc_expansion_rate > 1 and cfg.decoder_block != DecoderBlockType.DEEPSEEK,
      # Every layer is built unquantized (`models._NNXDecoderLayerFactory`).
      "quantization": bool(cfg.quantization),
      # The stage meshes, the pre-gather and the deferred reduce-scatter assume
      # MaxText's auto sharding mode.
      "shard_mode": cfg.shard_mode != ShardMode.AUTO,
      # The MPMD step pre-gathers FSDP weights its own way (`models.pregather_params`).
      "prefetch_fsdp_weights": cfg.prefetch_fsdp_weights,
      # The output head (`models.OutputHead`): `logits_dense` with ("embed",
      # "vocab") kernel axes in the weight dtype, its logits cast to float32 by
      # the loss, as MaxText's default head.
      "lm_head_fsdp_as_vocab_tp": cfg.lm_head_fsdp_as_vocab_tp,
      "lm_head_weight_grad_in_kernel_order": bool(cfg.lm_head_weight_grad_in_kernel_order),
      "tie_lm_head_cotangents": cfg.tie_lm_head_cotangents,
      "cast_logits_to_fp32": not cfg.cast_logits_to_fp32,
      "weight_dtype": (
          get_weight_dtype(cfg, "logits_dense") != cfg.weight_dtype
          or quantizations.get_weight_quant_config(cfg, "logits_dense") is not None
      ),
  }
  settings = [key for key, is_refused in refused.items() if is_refused]
  if settings:
    raise ValueError(
        f"{name}: the MPMD model does not build {', '.join(f'{key}={cfg.get_keys()[key]}' for key in settings)};"
        " leave these settings at MaxText's defaults."
    )
  left_out = [key for key, (is_set, _, _) in _LEFT_OUT.items() if is_set(cfg)]
  if left_out:
    # A model config can set them (qwen3-235b-a22b's load_balance_loss_weight), so the hint is an override.
    *others, last = [f"{_LEFT_OUT[key][1]} ({key}={cfg.get_keys()[key]})" for key in left_out]
    added = f"{', '.join(others)} and {last}" if others else last
    hints = " ".join(f"--maxtext_overrides={key}={_LEFT_OUT[key][2]}" for key in left_out)
    raise ValueError(
        f"{name}: MaxText would add {added}, which the MPMD step leaves out: it trains the embedding, decoder layers and"
        f" output head on the token cross-entropy alone. Pass {hints} to train without {'them' if others else 'it'}"
        " (spmd_baseline then runs the same)."
    )


def describe(config: Any, hparams: hparams_lib.Hparams) -> str:
  """Returns a multi-line summary of the run, for the logs."""
  if hparams.stage_layer_counts:
    counts_str = ",".join(str(c) for c in hparams.stage_layer_counts)
    stage_desc = f"{hparams.num_stages} (stage_layer_counts={counts_str})"
  else:
    stage_desc = f"{hparams.num_stages} ({hparams.layers_per_stage} layers/stage)"
  lines = [
      (f"model_name            : {hparams.model_name}" f" (maxtext: {hparams.maxtext_model_name})"),
      f"decoder_block         : {config.decoder_block.value}",
      f"num_decoder_layers    : {config.num_decoder_layers}",
      f"emb_dim / mlp_dim     : {config.emb_dim} / {config.mlp_dim}",
      f"vocab_size            : {config.vocab_size}",
      f"max_target_length     : {config.max_target_length}",
      f"per_device_batch_size : {config.per_device_batch_size}",
      f"pipeline stages       : {stage_desc}",
      f"fsdp per stage        : {hparams.fsdp}",
      f"expert per stage      : {hparams.expert}",
      (
          f"moe_dispatch          : {hparams.moe_dispatch}"
          f" (ragged_buffer_factor={hparams.ragged_buffer_factor},"
          f" num_moe_token_chunks={hparams.num_moe_token_chunks})"
      ),
      f"num_microbatches      : {hparams.num_microbatches}",
      f"remat_policy (MPMD)   : {hparams.mpmd_remat_policy}",
      f"pipeline_schedule     : {hparams.pipeline_schedule}",
  ]
  if hparams.maxtext_overrides:
    lines.append(f"maxtext_overrides     : {' '.join(hparams.maxtext_overrides)}")
  return "\n".join(f"  {line}" for line in lines)
