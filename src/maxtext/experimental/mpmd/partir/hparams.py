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
"""Single source of truth for the MPMD pipeline's hyperparameters.

This module owns the ``absl`` flags that every entry point in this package shares (the trainer, the lowering check,
the parity checker, the SPMD baseline and the launcher). Keeping them here guarantees that:

  * a flag added once is understood by every entry point, and
  * the launcher (``launch.py``) forwards *exactly* the flags the trainer accepts (see :func:`to_job_args`), so
    launcher and trainer can never drift.

The flags are registered by :func:`define_flags`, which each entry point calls before it parses its command line, and
never at import: MaxText's test process imports many modules that define absl flags of their own (``vocab_size``,
``seed``, ...), and a second definition of a name raises ``DuplicateFlagError``.

The module depends only on ``absl``, ``omegaconf`` (for the model configs) and MaxText's path constants, so that the
launcher does not pull in JAX, Flax or MaxText's model code.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import os
import types
from typing import Any

from absl import flags
import omegaconf

from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

# Pipeline schedules, spelled as PartIR's proto enum (the vocabulary of every
# flag, log line and report in this package). `OSS_SCHEDULE_NAMES` maps each
# onto the string Shardy MPMD's `mpmd_pipeline_schedule` takes.
PIPELINE_SCHEDULES = (
    "NONE",
    "ONE_FWD_ONE_BWD",
    "GPIPE",
    "GPIPE_BUT_1F1B_FOR_LAST_MESH",
    "ZERO_BUBBLE_H1",
    "ZERO_BUBBLE_H2_ZERO_TX_LATENCY",
    "ZERO_BUBBLE_H2_HALF_TX_LATENCY",
    "ZERO_BUBBLE_H2_FULL_TX_LATENCY",
    "PARALLEL_PIPELINES_WITH_WRAP_AROUND",
    "CIRCULAR",
    "CIRCULAR_WITH_REVERSED_BACKWARD",
)

# `PIPELINE_SCHEDULES` -> Shardy MPMD's `mpmd_pipeline_schedule` value.
OSS_SCHEDULE_NAMES = types.MappingProxyType(
    {
        "NONE": "None",
        "ONE_FWD_ONE_BWD": "1F1B",
        "GPIPE": "GPipe",
        "GPIPE_BUT_1F1B_FOR_LAST_MESH": "GPipeBut1F1BForLastMesh",
        "ZERO_BUBBLE_H1": "ZeroBubbleH1",
        "ZERO_BUBBLE_H2_ZERO_TX_LATENCY": "ZeroBubbleH2ZeroTxLatency",
        "ZERO_BUBBLE_H2_HALF_TX_LATENCY": "ZeroBubbleH2HalfTxLatency",
        "ZERO_BUBBLE_H2_FULL_TX_LATENCY": "ZeroBubbleH2FullTxLatency",
        "PARALLEL_PIPELINES_WITH_WRAP_AROUND": "ParallelPipelinesWithWrapAround",
        "CIRCULAR": "Circular",
        "CIRCULAR_WITH_REVERSED_BACKWARD": "CircularWithReversedBackward",
    }
)

# Schedules a training step here cannot use, with the reason. Shardy's
# ParallelPipelinesWithWrapAround orders forward fragments only and aborts the
# process on a backward one (`SDY_CHECK(IsForwardFragment(...))` in
# pipeline_schedule.cc). The circular schedules order several logical stages
# per mesh, but the model places one contiguous chunk of layers on each stage
# (`models._unit_stages`), so they would run as GPipe under another name.
UNSUPPORTED_SCHEDULES = types.MappingProxyType(
    {
        "PARALLEL_PIPELINES_WITH_WRAP_AROUND": (
            "Shardy MPMD schedules only forward fragments with it, and aborts on a" " training step's backward fragments"
        ),
        "CIRCULAR": (
            "it needs several logical stages per mesh, and the model places one"
            " contiguous chunk of layers per stage: it would run as GPipe"
        ),
        "CIRCULAR_WITH_REVERSED_BACKWARD": (
            "it needs several logical stages per mesh, and the model places one"
            " contiguous chunk of layers per stage: it would run as GPipe"
        ),
    }
)

# The schedules a training step here can use.
USABLE_SCHEDULES = tuple(name for name in PIPELINE_SCHEDULES if name not in UNSUPPORTED_SCHEDULES)

# User-friendly aliases accepted on the command line.
_SCHEDULE_ALIASES = {
    "1F1B": "ONE_FWD_ONE_BWD",
    "1f1b": "ONE_FWD_ONE_BWD",
    "GPIPE": "GPIPE",
    "gpipe": "GPIPE",
    "ZB_H1": "ZERO_BUBBLE_H1",
    "ZB-H1": "ZERO_BUBBLE_H1",
    "ZB_H2": "ZERO_BUBBLE_H2_ZERO_TX_LATENCY",
    "ZB-H2": "ZERO_BUBBLE_H2_ZERO_TX_LATENCY",
    "ZERO_BUBBLE_H2": "ZERO_BUBBLE_H2_ZERO_TX_LATENCY",
    "ZB_H2_ZERO": "ZERO_BUBBLE_H2_ZERO_TX_LATENCY",
    "ZB-H2-ZERO": "ZERO_BUBBLE_H2_ZERO_TX_LATENCY",
    "ZB_H2_HALF": "ZERO_BUBBLE_H2_HALF_TX_LATENCY",
    "ZB-H2-HALF": "ZERO_BUBBLE_H2_HALF_TX_LATENCY",
    "ZB_H2_FULL": "ZERO_BUBBLE_H2_FULL_TX_LATENCY",
    "ZB-H2-FULL": "ZERO_BUBBLE_H2_FULL_TX_LATENCY",
    # Shardy MPMD's own spellings (`mpmd_pipeline_schedule` values), upper-cased.
    "GPIPEBUT1F1BFORLASTMESH": "GPIPE_BUT_1F1B_FOR_LAST_MESH",
    "ZEROBUBBLEH1": "ZERO_BUBBLE_H1",
    "ZEROBUBBLEH2ZEROTXLATENCY": "ZERO_BUBBLE_H2_ZERO_TX_LATENCY",
    "ZEROBUBBLEH2HALFTXLATENCY": "ZERO_BUBBLE_H2_HALF_TX_LATENCY",
    "ZEROBUBBLEH2FULLTXLATENCY": "ZERO_BUBBLE_H2_FULL_TX_LATENCY",
    "PARALLELPIPELINESWITHWRAPAROUND": "PARALLEL_PIPELINES_WITH_WRAP_AROUND",
    "CIRCULARWITHREVERSEDBACKWARD": "CIRCULAR_WITH_REVERSED_BACKWARD",
}


def normalize_schedule(schedule: str) -> str:
  """Maps a user-supplied schedule name onto a PartIR MPMD schedule.

  Args:
    schedule: Schedule name or alias, case insensitive (e.g. ``1F1B``).

  Returns:
    A schedule name guaranteed to be in :data:`PIPELINE_SCHEDULES`.

  Raises:
    ValueError: If `schedule` is not a known schedule or alias.
  """
  if schedule in _SCHEDULE_ALIASES:
    return _SCHEDULE_ALIASES[schedule]
  upper = schedule.upper()
  if upper in _SCHEDULE_ALIASES:
    return _SCHEDULE_ALIASES[upper]
  if upper in PIPELINE_SCHEDULES:
    return upper
  raise ValueError(
      f"Unknown pipeline schedule {schedule!r}. Expected one of"
      f" {sorted(PIPELINE_SCHEDULES)} or an alias in"
      f" {sorted(_SCHEDULE_ALIASES)}."
  )


@functools.lru_cache(maxsize=None)
def read_model_yaml_metadata(model_name: str) -> dict[str, Any]:
  """Returns the settings in MaxText's ``configs/models/<model_name>.yml``; ``{}`` for ``default`` and ``simple``.

  Read with ``omegaconf``, as pyconfig reads it, but without pyconfig, so that :mod:`hparams` and the launcher stay
  free of JAX: the hyperparameters are checked against the model's decoder block, MoE layout and RoPE type before any
  MaxText config is built. A key the model config leaves out takes base.yml's value in pyconfig, which is the default
  each caller reads it with (``num_experts`` 1, ``first_num_dense_layers`` 0, ``interleave_moe_layer_step`` 1).

  Args:
    model_name: MaxText model config name (e.g. ``llama3.1-8b``).

  Raises:
    FileNotFoundError: If the model config does not exist. pyconfig would only log a warning and go on with base.yml's
      defaults: the run would still *look* right (``model_name``, and the dimensions overridden explicitly), while
      everything taken from the model config (``num_experts``, the MLA head counts and LoRA ranks, ``routed_bias``,
      ``attention_type``) had quietly reverted; a ``deepseek3-test`` benchmark measured a generic 16-head dense model
      and reported about 12x too few FLOPs.
  """
  if model_name in ("simple", "default"):
    return {}
  yaml_path = os.path.join(MAXTEXT_CONFIGS_DIR, "models", f"{model_name}.yml")
  if not os.path.exists(yaml_path):
    raise FileNotFoundError(
        f"MaxText model config {yaml_path!r} is missing, so model_name={model_name!r} would silently fall back to"
        " base.yml defaults and measure a different model. Use a MaxText checkout that has this model, or a model name"
        f" from {os.path.join(MAXTEXT_CONFIGS_DIR, 'models')}."
    )
  return dict(omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(yaml_path)))


def infer_decoder_block(model_name: str) -> str:
  """Returns the ``decoder_block`` of MaxText's config of `model_name` (``simple_mlp`` for ``simple`` and ``default``).

  Args:
    model_name: MaxText model config name (or the ``simple`` alias).

  Raises:
    FileNotFoundError: If the model config does not exist (see :func:`read_model_yaml_metadata`).
    ValueError: If the model config names no ``decoder_block``, as those of the models MaxText builds with its
      separate m3 backend (``use_m3_model``) do.
  """
  if model_name in ("simple", "default"):
    return "simple_mlp"
  block = read_model_yaml_metadata(model_name).get("decoder_block")
  if not (isinstance(block, str) and block):
    raise ValueError(
        f"model_name={model_name}: its MaxText config names no decoder_block (those of the models MaxText builds"
        " with its m3 backend, use_m3_model, name none), so the MPMD pipeline refuses it rather than train another"
        f" architecture. Supported decoder blocks: {', '.join(SUPPORTED_DECODER_BLOCKS)}."
    )
  return block


# The decoder blocks `models.PipelinedTransformer` builds and has been run with.
# Every other `DecoderBlockType` is refused rather than built: some need inputs
# or per-layer wiring the per-layer named computations do not pass (DeepSeek4's
# token inputs and hyper-connection head, Gemma4-Small's shared KV caches, GLM5's
# dense/MoE pair, Kimi-Linear's layer specs), and the rest have not run here.
# (`simple` and `default` model names build `simple_mlp`; no model config selects
# MaxText's `default` or `simple` blocks.)
SUPPORTED_DECODER_BLOCKS = (
    "simple_mlp",
    "llama2",
    "mistral",
    "mixtral",
    "qwen3",
    "qwen3_moe",
    "deepseek",
)


def unsupported_decoder_block(model_name: str, decoder_block: str) -> str | None:
  """Returns why `decoder_block` cannot run here, or None if it is supported."""
  if decoder_block in SUPPORTED_DECODER_BLOCKS:
    return None
  return (
      f"model_name={model_name} uses decoder_block={decoder_block}, which the MPMD"
      " pipeline does not build (it would train a different architecture than"
      f" the one MaxText describes). Supported decoder blocks: {', '.join(SUPPORTED_DECODER_BLOCKS)}."
  )


def is_pure_moe_model(model_name: str, first_num_dense_layers: int = -1) -> bool:
  """Returns whether every decoder layer of `model_name` uses an MoE MLP block.

  MaxText validates that ``base_mlp_dim == base_moe_mlp_dim`` when a model has
  no dense MLP layers (or, for ``qwen3_moe``, overwrites ``base_mlp_dim`` with
  ``base_moe_mlp_dim``). Keeping the two in sync when ``mlp_dim`` is overridden
  without an explicit ``moe_mlp_dim`` ensures both MPMD and SPMD configure the
  requested expert width.

  Args:
    model_name: MaxText model config name, of a supported decoder block.
    first_num_dense_layers: Number of leading dense layers override (-1 =
      default from YAML).
  """
  resolved_name = "default" if model_name == "simple" else model_name
  if infer_decoder_block(resolved_name) in ("qwen3_moe", "mixtral"):
    return True
  meta = read_model_yaml_metadata(resolved_name)
  if int(meta.get("num_experts", 1)) <= 1:
    return False
  dense_layers = first_num_dense_layers if first_num_dense_layers >= 0 else int(meta.get("first_num_dense_layers", 0))
  return dense_layers == 0 and int(meta.get("interleave_moe_layer_step", 1)) == 1


# The `--remat_policy` values: the policies of MaxText's resolver
# (`NNXDecoder.get_remat_policy`, which `models` calls) that neither offload to
# host memory nor name quantization intermediates (every layer here is built
# unquantized). A literal, as this module stays free of JAX; models_test checks
# that MaxText's resolver accepts each. `none` is spelled by leaving --remat off.
REMAT_POLICIES = (
    "full",
    "minimal",
    "minimal_with_context",
    "minimal_flash",
    "save_dot_with_context_except_mlp",
    "save_dot_except_mlpwi",
    "save_dot_except_mlp",
    "save_qkv_proj",
    "save_out_proj",
)


# How routed tokens reach their expert shard when `expert > 1`:
#
#   ragged_a2a     : MaxText's default `jax.lax.ragged_all_to_all` over the
#                    `expert` axis (`sparse_matmul_with_all_to_all`), using
#                    megablox `gmm` on the received slice.
#   ring_of_experts: `use_ring_of_experts` -- the tokens of the EP group are
#                    all-gathered first, each device sorts and runs only the
#                    tokens routed to its local experts with MaxText's
#                    SparseCore ragged-sort kernels (`use_ragged_sort`), and
#                    the expert outputs are reduce-scattered back. Moves
#                    `expert` x more activation bytes than `ragged_a2a`,
#                    but uses only all-gather / reduce-scatter.
#
# `ragged_a2a` is the default because it is MaxText's default. (In runs of the
# stack this was ported from, it halted the TPU on the first forward step on a
# TPU v7x 4x4x4 slice with `--expert=32` while `ring_of_experts` ran; not
# reproduced with this port.)
MOE_DISPATCH_MODES = ("ragged_a2a", "ring_of_experts")


def moe_dispatch_overrides(hparams: "Hparams") -> dict[str, Any]:
  """MaxText config overrides implementing ``hparams.moe_dispatch``.

  Shared by ``config.build_maxtext_config`` (MPMD) and ``spmd_baseline`` so
  both sides dispatch tokens to experts identically. Empty when
  ``hparams.expert == 1``; ``ragged_a2a`` only forwards the expert degree and
  the optional ``ragged_buffer_factor`` cap so it stays byte-identical to
  MaxText's defaults.

  Args:
    hparams: The run's hyperparameters.

  Returns:
    ``{maxtext_key: value}`` overrides, with Python ``bool``/``int``/``float``
    values.
  """
  if hparams.expert <= 1:
    return {}
  overrides: dict[str, Any] = {"ici_expert_parallelism": hparams.expert}
  # `ragged_buffer_factor` sizes the routed-token buffer on every dispatch
  # path; MaxText's own default is the worst case (-1.0), so only forward a
  # cap.
  if hparams.ragged_buffer_factor > 0:
    overrides["ragged_buffer_factor"] = hparams.ragged_buffer_factor
  if hparams.moe_dispatch == "ragged_a2a":
    return overrides
  # `use_ragged_sort` is valid on both paths (it restricts the local permute
  # to the valid prefix of the buffer) but is only switched on with
  # ring-of-experts.
  overrides.update(
      {
          "use_ring_of_experts": True,
          "use_ragged_sort": True,
          "num_moe_token_chunks": hparams.num_moe_token_chunks,
      }
  )
  return overrides


# MaxText's MoE GMM tiles (base.yml: m x k x n = 512 x 1024 x 1024 for the
# forward, dlhs and drhs kernels alike) are sized for bfloat16. In float32 every
# buffer doubles and the kernel asks for 16.3 MiB of scoped VMEM, over v5p's
# 16 MiB default ("CompileTimeScopedVmemOom" when deepseek3-tiny's parity ran in
# float32). Halving the contracting and output tiles brings it back under the
# bfloat16 footprint; explicit `--maxtext_overrides` tiles still win.
_FLOAT32_GMM_TILES = {
    f"{weight}_tile_{kernel}_{dim}": 512
    for weight in ("wi", "wo")
    for kernel in ("fwd", "dlhs", "drhs")
    for dim in ("embed_dim", "mlp_dim")
}


def override_value(hparams: "Hparams", key: str) -> str | None:
  """Returns the last ``--maxtext_overrides`` value for `key`, or None."""
  value = None
  for item in hparams.maxtext_overrides:
    name, _, rhs = item.partition("=")
    if name == key:
      value = rhs
  return value


def float32_gmm_tiles(hparams: "Hparams") -> dict[str, int]:
  """Returns the GMM tile settings a float32 run needs (empty otherwise)."""
  if override_value(hparams, "dtype") in ("float32", "fp32"):
    return dict(_FLOAT32_GMM_TILES)
  return {}


# What an XProf trace covers: one TPU, one SparseCore and one SparseCore tile a
# traced host, small enough to open. `benchmark.profile` hands it to
# jax.profiler, the SPMD baseline to MaxText's profiler
# (enable_tpu_profiling_options).
TRACE_ADVANCED_CONFIGURATION = types.MappingProxyType(
    {
        "tpu_num_chips_to_profile_per_task": 1,
        "tpu_num_sparse_cores_to_trace": 1,
        "tpu_num_sparse_core_tiles_to_trace": 1,
    }
)


# MaxText settings that config.py derives from a hyperparameter (or must pin),
# with where to set them instead. A `--maxtext_overrides` of one of them would
# win in MaxText's config while the rest of the package (the optimizer's
# learning rate, the stage meshes, MPMD_RESULT, launch.py's checks) still used
# the hyperparameter.
OWNED_MAXTEXT_KEYS = types.MappingProxyType(
    {
        "model_name": "--model_name",
        "base_num_decoder_layers": "--num_layers",
        "per_device_batch_size": "--per_device_batch_size",
        "max_target_length": "--max_target_length",
        "ici_pipeline_parallelism": "--pp",
        "dcn_pipeline_parallelism": "--pp",
        "remat_policy": "--remat and --remat_policy",
        "learning_rate": "--learning_rate",
        "init_weights_seed": "--seed",
        "base_emb_dim": "--emb_dim",
        "base_mlp_dim": "--mlp_dim",
        "base_moe_mlp_dim": "--moe_mlp_dim",
        "vocab_size": "--vocab_size",
        "ici_fsdp_parallelism": "--fsdp",
        "ici_expert_parallelism": "--expert",
        "use_ring_of_experts": "--moe_dispatch",
        "ragged_buffer_factor": "--ragged_buffer_factor",
        "num_moe_token_chunks": "--num_moe_token_chunks",
        "first_num_dense_layers": "--first_num_dense_layers",
        "decoder_block": "--model_name (it follows the model)",
        "pipeline_parallel_layers": "--num_layers / --stage_layer_counts",
        "scan_layers": "nothing: pipelining needs one computation per layer",
        # MaxText replays the batch this many times and multiplies the analytic
        # FLOP count (and so TFLOP/s and MFU) by it; microbatches here split it.
        "gradient_accumulation_steps": "--num_microbatches",
        "enable_dropout": "nothing: the MPMD model runs every layer deterministic",
        "dropout_rate": "nothing: the MPMD model runs every layer deterministic",
    }
)


def _rope_is_pinned(model_name: str) -> bool:
  """Whether config.py pins `model_name`'s RoPE to ``rope_type=default``."""
  return infer_decoder_block(model_name) == "deepseek" or uses_yarn_rope(model_name)


def uses_yarn_rope(model_name: str) -> bool:
  """Returns whether `model_name` configures complex-valued YaRN RoPE."""
  resolved_name = "default" if model_name == "simple" else model_name
  return str(read_model_yaml_metadata(resolved_name).get("rope_type", "")).lower() == "yarn"


def parse_stage_layer_counts(
    raw: str | Sequence[int],
) -> tuple[int, ...]:
  """Parses a comma-separated string or integer sequence into stage counts.

  Args:
    raw: Either a comma-separated string (e.g. ``"7,10,11,12"``) or a sequence
      of integers. An empty string or empty sequence denotes uniform stage layer
      counts.

  Returns:
    A tuple of per-stage layer counts, or ``()`` when uniform.

  Raises:
    ValueError: If any entry is not a valid integer.
  """
  if isinstance(raw, str):
    stripped = raw.strip()
    if not stripped:
      return ()
    parts = [p.strip() for p in stripped.split(",")]
    if any(not p for p in parts):
      raise ValueError(f"Invalid stage_layer_counts {raw!r}: expected comma-separated" " integers.")
    try:
      return tuple(int(p) for p in parts)
    except ValueError as err:
      raise ValueError(f"Invalid stage_layer_counts {raw!r}: expected comma-separated" " integers.") from err
  return tuple(int(x) for x in raw)


@dataclasses.dataclass(frozen=True)
class Hparams:
  """Model, parallelism and benchmarking hyperparameters.

  Attributes:
    model_name: MaxText model config name (``deepseek3-tiny``,
      ``deepseek2-16b``, ``llama3.1-8b``, ``mixtral-8x7b``, ``qwen3-30b-a3b``,
      ``default``, ...) whose decoder block is in
      :data:`SUPPORTED_DECODER_BLOCKS`. The alias ``simple`` is mapped onto
      MaxText's ``default`` config.
    num_layers: Total number of decoder layers spread across the pipeline. Its
      default (8) applies to every model, real ones included.
    emb_dim: Model (embedding) dimension. -1 keeps the model YAML value
      (:func:`from_flags` passes -1 for a named model whose flag is not given).
    mlp_dim: Feed-forward hidden dimension. -1 keeps the model YAML value.
    moe_mlp_dim: MoE expert intermediate dimension. -1 keeps the model YAML
      value (or matches ``mlp_dim`` on pure-MoE models).
    per_device_batch_size: Batch size per device of a *single* stage mesh.
    max_target_length: Sequence length.
    vocab_size: Vocabulary size. -1 keeps the model YAML value.
    num_stages: Pipeline depth, i.e. number of stage meshes.
    fsdp: FSDP degree within each stage mesh; -1 means "use every device of the
      stage mesh not taken by ``expert``".
    expert: Expert-parallel degree within each stage mesh. MoE expert weights
      are partitioned across this axis (each device owns ``num_experts /
      expert`` whole experts) and tokens are routed to them with an all-to-all,
      instead of every device all-gathering the full expert weights for every
      microbatch. The batch is sharded over ``fsdp x expert`` devices either
      way. 1 disables expert parallelism.
    moe_dispatch: How routed tokens reach their expert shard when ``expert >
      1``; one of :data:`MOE_DISPATCH_MODES`. Ignored when ``expert == 1``.
    ragged_buffer_factor: Capacity of the per-shard routed-token buffer as a
      multiple of the perfectly balanced load. ``<= 0`` sizes it for the worst
      case (never drops tokens, at ``min(expert, num_experts / top_k)`` x the
      balanced size); ``> 0`` (e.g. ``2``) caps it and drops the overflow.
      Ignored when ``expert == 1``.
    num_moe_token_chunks: ``ring_of_experts`` only: split each microbatch's
      routed tokens into this many chunks so one chunk's all-gather /
      reduce-scatter overlaps the previous chunk's expert matmul. 1 disables.
    num_microbatches: Number of microbatches per optimizer step.
    pipeline_schedule: PartIR MPMD schedule name (already normalized).
    first_num_dense_layers: For DeepSeek models, how many leading layers are
      dense rather than MoE. -1 keeps the value from the model config.
    remat: Whether to rematerialize (recompute) activations during the
      backward pass, in both MPMD (a per-layer ``jax.checkpoint`` with
      ``remat_policy``) and the SPMD baseline.
    remat_policy: The MaxText rematerialisation policy each layer is
      checkpointed with under ``remat``, one of :data:`REMAT_POLICIES`; the
      SPMD baseline gets the same. Without ``remat`` no MPMD layer is
      checkpointed, and MaxText's config, which refuses ``none`` with pipeline
      parallelism, is handed ``full`` (``config.build_maxtext_config``).
    steps: Number of timed training steps.
    warmup_steps: Number of untimed warm-up steps run before timing.
    learning_rate: AdamW learning rate.
    seed: Seed for parameter initialisation and synthetic data.
    stage_layer_counts: Optional per-stage decoder layer counts (e.g. ``(7, 10,
      11, 12)``). Empty tuple distributes ``num_layers`` uniformly across
      ``num_stages``.
    maxtext_overrides: ``key=value`` MaxText settings applied after everything
      :mod:`config` derives, in order (e.g. ``("dtype=float32",)`` for a
      float32 parity check). Both arms of ``parity.py`` get them.
  """

  model_name: str = "deepseek3-tiny"
  num_layers: int = 8
  emb_dim: int = 512
  mlp_dim: int = 2048
  moe_mlp_dim: int = -1
  per_device_batch_size: float = 2.0
  max_target_length: int = 512
  vocab_size: int = 16384
  num_stages: int = 4
  fsdp: int = -1
  expert: int = 1
  moe_dispatch: str = "ragged_a2a"
  ragged_buffer_factor: float = -1.0
  num_moe_token_chunks: int = 1
  num_microbatches: int = 4
  pipeline_schedule: str = "ONE_FWD_ONE_BWD"
  first_num_dense_layers: int = -1
  remat: bool = False
  remat_policy: str = "full"
  steps: int = 5
  warmup_steps: int = 2
  learning_rate: float = 1e-3
  seed: int = 42
  stage_layer_counts: tuple[int, ...] | str = ()
  maxtext_overrides: tuple[str, ...] | str = ()

  def __post_init__(self):
    if self.num_stages <= 0:
      raise ValueError(f"num_stages must be positive, got {self.num_stages}.")
    if self.num_layers < self.num_stages:
      raise ValueError(
          f"num_layers ({self.num_layers}) must be >= num_stages"
          f" ({self.num_stages}) so that every stage owns at least one layer."
      )
    parsed_counts = parse_stage_layer_counts(self.stage_layer_counts)
    object.__setattr__(self, "stage_layer_counts", parsed_counts)
    if parsed_counts:
      if len(parsed_counts) != self.num_stages:
        raise ValueError(
            f"stage_layer_counts ({parsed_counts}) must have length"
            f" num_stages ({self.num_stages}), got {len(parsed_counts)}."
        )
      if any(count <= 0 for count in parsed_counts):
        raise ValueError(f"Every entry in stage_layer_counts ({parsed_counts}) must be" " positive.")
      if sum(parsed_counts) != self.num_layers:
        raise ValueError(f"sum(stage_layer_counts) ({sum(parsed_counts)}) must equal" f" num_layers ({self.num_layers}).")
    elif self.num_layers % self.num_stages:
      raise ValueError(
          f"num_layers ({self.num_layers}) must be divisible by num_stages"
          f" ({self.num_stages}) for a balanced pipeline."
      )
    if self.num_microbatches <= 0:
      raise ValueError(f"num_microbatches must be positive, got {self.num_microbatches}.")
    if self.expert <= 0:
      raise ValueError(f"expert must be positive, got {self.expert}.")
    if self.moe_dispatch not in MOE_DISPATCH_MODES:
      raise ValueError(f"moe_dispatch must be one of {MOE_DISPATCH_MODES}, got" f" {self.moe_dispatch!r}.")
    if self.num_moe_token_chunks <= 0:
      raise ValueError("num_moe_token_chunks must be positive, got" f" {self.num_moe_token_chunks}.")
    # First, as the override checks below read the model config: a missing model config raises FileNotFoundError.
    unsupported = unsupported_decoder_block(self.model_name, self.decoder_block)
    if unsupported:
      raise ValueError(unsupported)
    overrides = self.maxtext_overrides
    overrides = (overrides,) if isinstance(overrides, str) else tuple(overrides)
    for item in overrides:
      key, sep, _ = item.partition("=")
      if not sep or not key or key != key.strip():
        raise ValueError(f"maxtext_overrides entry {item!r}: expected key=value.")
      if key in OWNED_MAXTEXT_KEYS:
        raise ValueError(
            f"maxtext_overrides entry {item!r}: {key} is derived from the"
            f" hyperparameters; set it with {OWNED_MAXTEXT_KEYS[key]}."
        )
      if key == "rope_type" and _rope_is_pinned("default" if self.model_name == "simple" else self.model_name):
        raise ValueError(
            f"maxtext_overrides entry {item!r}: {self.model_name} asks for"
            " YaRN RoPE, which config.py pins to rope_type=default (PartIR"
            " MPMD asserts on its complex64 table)."
        )
    if self.remat_policy not in REMAT_POLICIES:
      raise ValueError(
          f"remat_policy={self.remat_policy} is not one the MPMD model checkpoints its layers with:"
          f" {', '.join(REMAT_POLICIES)} (MaxText's, except those that offload to host memory or save quantization"
          " intermediates). Without --remat no layer is rematerialised."
      )
    schedule = normalize_schedule(self.pipeline_schedule)
    if schedule in UNSUPPORTED_SCHEDULES:
      raise ValueError(f"pipeline_schedule {schedule} is not supported here:" f" {UNSUPPORTED_SCHEDULES[schedule]}.")
    # `object.__setattr__` because the dataclass is frozen.
    object.__setattr__(self, "maxtext_overrides", overrides)
    object.__setattr__(self, "pipeline_schedule", schedule)

  @property
  def layers_per_stage(self) -> int:
    """Returns the nominal uniform number of decoder layers per stage."""
    return self.num_layers // self.num_stages

  @property
  def mpmd_remat_policy(self) -> str:
    """Returns the policy the MPMD model checkpoints each layer with: ``remat_policy`` under ``remat``, else none."""
    return self.remat_policy if self.remat else "none"

  @property
  def maxtext_model_name(self) -> str:
    """Returns the MaxText config name, resolving the ``simple`` alias."""
    return "default" if self.model_name == "simple" else self.model_name

  @property
  def decoder_block(self) -> str:
    """Returns the MaxText ``decoder_block`` name for ``model_name``."""
    return infer_decoder_block(self.maxtext_model_name)


def devices_in_device_type(device_type: str) -> int:
  """Returns the JAX device count of an xpk device type, or 0 if unknown.

  ``v5p-N`` / ``v4-N`` count TensorCores (two per chip, one megacore JAX device
  per chip), ``v5litepod-N`` / ``v6e-N`` count chips, and ``tpu7x-AxBxC``
  names the chip box (two JAX devices per chip). ``<platform>=<AxB...>``
  strings fall back to :func:`devices_in_platform`.

  Args:
    device_type: For example ``v5p-32`` or ``tpu7x-4x4x4``.
  """
  if "=" in device_type:
    return devices_in_platform(device_type)
  gen, _, size = device_type.partition("-")
  if not size:
    return 0
  if "x" in size:
    chips = 1
    for dim in size.split("x"):
      if not dim.isdigit():
        return 0
      chips *= int(dim)
    return 2 * chips if gen.startswith("tpu7") else chips
  if not size.isdigit():
    return 0
  count = int(size)
  return count // 2 if gen in ("v5p", "v4") else count


def devices_in_platform(platform: str) -> int:
  """Returns the chip count implied by a ``<platform>=<AxB...>`` string.

  This is the product of the topology dimensions, i.e. the number of *chips*.
  On platforms that expose more than one core per chip it is not the number of
  JAX devices; callers who need that must pass it explicitly.

  Args:
    platform: A ``<platform>=<AxB...>`` string.

  Returns:
    The product of the topology dimensions, or 0 if it cannot be parsed.
  """
  if "=" not in platform:
    return 0
  _, _, topology = platform.partition("=")
  total = 1
  for dimension in topology.split("x"):
    if not dimension.isdigit():
      return 0
    total *= int(dimension)
  return total


def check_microbatch_split(hparams: Hparams, *, devices_per_stage: int, data_parallel_size: int) -> None:
  """Raises unless every microbatch splits evenly over a stage's batch-sharding devices.

  MaxText's TPU flash attention asserts that the batch divides over the
  ``fsdp`` x ``expert`` devices; a CPU lowering (dot-product attention) does
  not, so without this check the mistake surfaces only once the TPUs run. The
  deferred MoE expert-gradient reduce-scatter (``models.pregather_params``) also
  needs it.

  Args:
    hparams: The configuration to run.
    devices_per_stage: Devices of one stage mesh (the batch is data-parallel over
      one stage).
    data_parallel_size: Devices of a stage the batch is sharded over (``fsdp`` x
      ``expert``).

  Raises:
    ValueError: If the global batch does not split into the microbatches, or a
      microbatch not over the data-parallel devices.
  """
  global_batch = int(hparams.per_device_batch_size * devices_per_stage)
  micro = global_batch // hparams.num_microbatches
  if global_batch % hparams.num_microbatches or micro % max(data_parallel_size, 1):
    raise ValueError(
        f"Global batch {global_batch} (per_device_batch_size {hparams.per_device_batch_size} x"
        f" {devices_per_stage} devices per stage) in {hparams.num_microbatches} microbatches gives {micro} sequences"
        f" per microbatch, which must be a multiple of the {data_parallel_size} devices its batch is sharded over"
        " (MaxText's TPU attention asserts it)."
    )


def check_microbatch_split_on(hparams: Hparams, num_devices: int) -> None:
  """:func:`check_microbatch_split` for `num_devices` split into stages as ``mesh_utils.create_stage_meshes`` does.

  Each stage gets ``num_devices / num_stages`` devices, and its batch is sharded over ``fsdp x expert`` of them
  (``fsdp=-1``: every device not taken by ``expert``).

  Raises:
    ValueError: See :func:`check_microbatch_split`.
  """
  devices_per_stage = num_devices // hparams.num_stages
  fsdp = hparams.fsdp if hparams.fsdp > 0 else devices_per_stage // hparams.expert
  check_microbatch_split(hparams, devices_per_stage=devices_per_stage, data_parallel_size=fsdp * hparams.expert)


def ragged_buffer_note(hparams: Hparams) -> str | None:
  """Returns a note when MoE dispatch buffers are sized for the worst case, else None."""
  if hparams.expert <= 1 or hparams.ragged_buffer_factor > 0:
    return None
  return (
      "--expert > 1 with --ragged_buffer_factor <= 0 sizes every MoE dispatch buffer for the worst case"
      " (min(expert, experts / top_k) x the balanced load: 98304 rows per layer and microbatch for deepseek2-16b at"
      " --expert=8, which filled stage 0's HBM on a v5p-64). For a large MoE, consider --ragged_buffer_factor=2 (it"
      " drops tokens past twice the balanced load)."
  )


# Flags forwarded verbatim from the launcher (`launch.py`) to the trainer.
# `pp` is deliberately excluded: it is folded into `num_stages` below.
FORWARDED_FLAGS = (
    "model_name",
    "num_layers",
    "emb_dim",
    "mlp_dim",
    "moe_mlp_dim",
    "per_device_batch_size",
    "max_target_length",
    "vocab_size",
    "num_stages",
    "fsdp",
    "expert",
    "moe_dispatch",
    "ragged_buffer_factor",
    "num_moe_token_chunks",
    "num_microbatches",
    "pipeline_schedule",
    "first_num_dense_layers",
    "remat",
    "remat_policy",
    "steps",
    "warmup_steps",
    "learning_rate",
    "seed",
    "stage_layer_counts",
    "maxtext_overrides",
)


def define_flags(flag_values: flags.FlagValues = flags.FLAGS) -> None:
  """Registers the hyperparameter flags on `flag_values`; a second call is a no-op.

  Entry points call it before parsing their command line; tests pass a fresh
  ``flags.FlagValues()`` so that nothing is registered on the global
  ``flags.FLAGS`` of a shared test process.

  Args:
    flag_values: Where to register the flags.
  """
  if "num_microbatches" in flag_values:
    return
  # The dataclass defaults are the flag defaults.
  defaults = Hparams()
  fv = {"flag_values": flag_values}
  flags.DEFINE_string(
      "model_name",
      defaults.model_name,
      "MaxText model config name (e.g. simple, deepseek3-tiny, deepseek2-16b, llama3.1-8b, mixtral-8x7b,"
      f" qwen3-30b-a3b); its decoder block must be one of {', '.join(SUPPORTED_DECODER_BLOCKS)}.",
      **fv,
  )
  flags.DEFINE_integer(
      "num_layers",
      defaults.num_layers,
      "Total number of decoder layers, for every model: pass a model's own count (32 for llama3.1-8b) to train all"
      " of it.",
      **fv,
  )
  # See from_flags: only the toy models take these flags' defaults.
  toy = "The flag's default is for the toy models (simple, default, deepseek3-tiny); others keep their config's value"
  flags.DEFINE_integer(
      "emb_dim", defaults.emb_dim, f"Embedding (model) dimension. {toy} unless the flag is given (-1 keeps it).", **fv
  )
  flags.DEFINE_integer(
      "mlp_dim", defaults.mlp_dim, f"Feed-forward hidden dimension. {toy} unless the flag is given (-1 keeps it).", **fv
  )
  flags.DEFINE_integer(
      "moe_mlp_dim", defaults.moe_mlp_dim, "MoE expert hidden dimension (-1 keeps the model config value).", **fv
  )
  flags.DEFINE_float(
      "per_device_batch_size", defaults.per_device_batch_size, "Batch size per device of a single stage mesh.", **fv
  )
  flags.DEFINE_integer("max_target_length", defaults.max_target_length, "Sequence length.", **fv)
  flags.DEFINE_integer(
      "vocab_size", defaults.vocab_size, f"Vocabulary size. {toy} unless the flag is given (-1 keeps it).", **fv
  )
  flags.DEFINE_integer("num_stages", defaults.num_stages, "Pipeline depth (number of stages).", **fv)
  flags.DEFINE_integer("pp", -1, "Alias for --num_stages. Takes precedence when > 0.", **fv)
  flags.DEFINE_integer(
      "fsdp",
      defaults.fsdp,
      "FSDP degree within each stage mesh (-1 = all devices of the stage not taken by --expert).",
      **fv,
  )
  flags.DEFINE_integer(
      "expert",
      defaults.expert,
      "Expert-parallel degree within each stage mesh: MoE experts are partitioned across this many devices and tokens"
      " are all-to-all'd to them, instead of FSDP-gathering the expert weights every microbatch. Must divide"
      " num_experts; 1 disables expert parallelism.",
      **fv,
  )
  flags.DEFINE_enum(
      "moe_dispatch",
      defaults.moe_dispatch,
      MOE_DISPATCH_MODES,
      "How routed tokens reach their expert shard when --expert > 1. ragged_a2a: jax.lax.ragged_all_to_all (MaxText"
      " default). ring_of_experts: all-gather the EP group's tokens, SparseCore ragged-sort the local experts' share,"
      " reduce-scatter the outputs.",
      **fv,
  )
  flags.DEFINE_float(
      "ragged_buffer_factor",
      defaults.ragged_buffer_factor,
      "Routed-token buffer capacity per expert shard as a multiple of the balanced load. <= 0: worst case, never"
      " drops. > 0 (e.g. 2): capped, drops overflow.",
      **fv,
  )
  flags.DEFINE_integer(
      "num_moe_token_chunks",
      defaults.num_moe_token_chunks,
      "ring_of_experts only: chunks per microbatch so one chunk's all-gather / reduce-scatter overlaps the previous"
      " chunk's expert matmul. 1 disables.",
      **fv,
  )
  flags.DEFINE_integer("num_microbatches", defaults.num_microbatches, "Number of microbatches per optimizer step.", **fv)
  flags.DEFINE_string(
      "pipeline_schedule",
      defaults.pipeline_schedule,
      f"Pipeline schedule. Accepts aliases such as 1F1B. One of: {', '.join(USABLE_SCHEDULES)} (the other OSS"
      " schedules cannot run a training step here; see UNSUPPORTED_SCHEDULES).",
      **fv,
  )
  flags.DEFINE_integer(
      "first_num_dense_layers",
      defaults.first_num_dense_layers,
      "DeepSeek only: number of leading dense (non-MoE) layers. -1 keeps the model config value.",
      **fv,
  )
  flags.DEFINE_boolean(
      "remat",
      defaults.remat,
      "Rematerialize each layer's activations in the backward pass (jax.checkpoint with --remat_policy). Off by"
      " default; without it stage 0 ran out of HBM for mixtral-8x7b on a v5p-64 and llama3.1-8b on a v6e-16"
      " (per_device_batch_size 8, seq 2048).",
      **fv,
  )
  flags.DEFINE_string(
      "remat_policy",
      defaults.remat_policy,
      "MaxText rematerialisation policy each layer is checkpointed with under --remat: one of"
      f" {', '.join(REMAT_POLICIES)}.",
      **fv,
  )
  flags.DEFINE_integer("steps", defaults.steps, "Number of timed training steps.", **fv)
  flags.DEFINE_integer("warmup_steps", defaults.warmup_steps, "Number of untimed warm-up steps run before timing.", **fv)
  flags.DEFINE_float("learning_rate", defaults.learning_rate, "AdamW learning rate.", **fv)
  flags.DEFINE_integer("seed", defaults.seed, "Seed for initialisation and synthetic data.", **fv)
  flags.DEFINE_string(
      "stage_layer_counts",
      "",
      "Optional comma-separated per-stage decoder layer counts (e.g. '7,10,11,12'). Empty distributes num_layers"
      " uniformly across stages.",
      **fv,
  )
  flags.DEFINE_multi_string(
      "maxtext_overrides",
      [],
      "key=value MaxText setting applied after everything config.py derives; repeatable, e.g."
      " --maxtext_overrides=dtype=float32 for a float32 parity check (both arms get it).",
      **fv,
  )


def from_flags(flag_values: flags.FlagValues = flags.FLAGS) -> Hparams:
  """Builds an :class:`Hparams` from parsed flags registered by :func:`define_flags`."""
  fv = flag_values
  num_stages = fv.pp if fv.pp > 0 else fv.num_stages
  model_name = fv.model_name
  # The toy models take the flags' small dimensions; a real model keeps its
  # config's unless a dimension is passed explicitly.
  is_toy_model = model_name in ("deepseek3-tiny", "simple", "default")
  emb_dim = fv.emb_dim if (is_toy_model or fv["emb_dim"].present) else -1
  mlp_dim = fv.mlp_dim if (is_toy_model or fv["mlp_dim"].present) else -1
  vocab_size = fv.vocab_size if (is_toy_model or fv["vocab_size"].present) else -1
  return Hparams(
      model_name=model_name,
      num_layers=fv.num_layers,
      emb_dim=emb_dim,
      mlp_dim=mlp_dim,
      moe_mlp_dim=fv.moe_mlp_dim,
      per_device_batch_size=fv.per_device_batch_size,
      max_target_length=fv.max_target_length,
      vocab_size=vocab_size,
      num_stages=num_stages,
      fsdp=fv.fsdp,
      expert=fv.expert,
      moe_dispatch=fv.moe_dispatch,
      ragged_buffer_factor=fv.ragged_buffer_factor,
      num_moe_token_chunks=fv.num_moe_token_chunks,
      num_microbatches=fv.num_microbatches,
      pipeline_schedule=fv.pipeline_schedule,
      first_num_dense_layers=fv.first_num_dense_layers,
      remat=fv.remat,
      remat_policy=fv.remat_policy,
      steps=fv.steps,
      warmup_steps=fv.warmup_steps,
      learning_rate=fv.learning_rate,
      seed=fv.seed,
      stage_layer_counts=fv.stage_layer_counts,
      maxtext_overrides=tuple(fv.maxtext_overrides),
  )


def to_job_args(hparams: Hparams) -> dict[str, object]:
  """Returns the worker-binary command line for `hparams`.

  Used by the launcher (`launch.py`) so the flags sent to the Pathways job are
  derived from the same dataclass the trainer parses them back into.

  Args:
    hparams: Hyperparameters to serialise.

  Returns:
    A mapping of flag name to value (``--name=value`` on the command line; a
    list value is a repeated flag, one ``--name=item`` per item).
  """
  as_dict = dataclasses.asdict(hparams)
  as_dict["stage_layer_counts"] = ",".join(str(c) for c in hparams.stage_layer_counts)
  as_dict["maxtext_overrides"] = list(hparams.maxtext_overrides)
  return {name: as_dict[name] for name in FORWARDED_FLAGS}


def job_argv(args: dict[str, object]) -> list[str]:
  """Renders :func:`to_job_args`-style ``{name: value}`` as ``--flags``."""
  out = []
  for name, value in args.items():
    if isinstance(value, bool):
      out.append(f"--{name}" if value else f"--no{name}")
    elif isinstance(value, (list, tuple)):
      out += [f"--{name}={item}" for item in value]
    else:
      out.append(f"--{name}={value}")
  return out


def read_xla_flags_file(path: str) -> dict[str, str]:
  """Reads XLA / libtpu flags from `path`: one ``--key=value`` per line.

  Blank lines and ``#`` comment lines are skipped. No flag set is built in: a
  run passes the one its slice needs with ``--xla_flags_file``.

  Args:
    path: The flags file.

  Returns:
    ``{key: value}`` in file order, keys without the leading dashes.

  Raises:
    ValueError: If a line is not ``--key=value``.
  """
  flags_by_key: dict[str, str] = {}
  with open(path, encoding="utf-8") as handle:
    for number, raw in enumerate(handle, 1):
      line = raw.strip()
      if not line or line.startswith("#"):
        continue
      key, sep, value = line[2:].partition("=")
      if not line.startswith("--") or not sep or not key:
        raise ValueError(f"{path}:{number}: expected --key=value, got {line!r}")
      flags_by_key[key] = value
  return flags_by_key
