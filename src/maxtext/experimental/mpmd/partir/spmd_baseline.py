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
r"""Derives the equivalent MaxText **SPMD** pipeline configuration.

The point of comparison for this package is MaxText's own pipeline
(``maxtext/src/maxtext/layers/pipeline.py``), which implements pipelining
inside a single SPMD program: microbatches are circulated between stages with
``jax.lax.ppermute`` inside a ``shard_map`` over a ``stage`` mesh axis.

Translating one configuration into the other is not a rename, because the two
define ``per_device_batch_size`` against different device sets:

  * **MPMD** shards the batch over one *stage* mesh, since every microbatch
    traverses all stages. Global batch = ``pdbs x devices_per_stage``.
  * **MaxText SPMD** shards the batch over *all* devices, the ``stage`` axis
    included. Global batch = ``pdbs x total_devices``.

Comparing the two at the same nominal ``per_device_batch_size`` would
therefore silently give the SPMD run ``num_stages`` times more work per step.
:func:`spmd_flags` performs the conversion so both runs consume the same
number of tokens per step, which is the only basis on which their step times
can be compared.

Run this module to print the settings and the command that runs MaxText's own
trainer with them (multi-controller JAX, not Pathways); with ``--cluster`` it
wraps the command in a ``gcluster job submit`` (Cluster Toolkit; with
``--launcher=xpk``, the deprecated ``xpk workload create``)::

    python3 -m maxtext.experimental.mpmd.partir.spmd_baseline \
      --pp=4 --fsdp=4 --num_microbatches=4 --per_device_batch_size=4 --device_type=v5p-32
"""

from __future__ import annotations

from collections.abc import Sequence
import shlex
import time

from absl import app
from absl import flags

from maxtext.utils import max_logging

from maxtext.experimental.mpmd.partir import hparams as hparams_lib
from maxtext.experimental.mpmd.partir import launch

# MaxText's trainer, run as a module from an image that has MaxText installed.
MAXTEXT_TRAIN_MODULE = "maxtext.trainers.pre_train.train"

# Number of steady-state steps to capture in the xplane profile. Five steps is
# enough to average out per-step jitter while keeping the trace small enough to
# load in the XProf UI.
_PROFILER_STEPS = 5

# The flags a cluster command needs besides the device type.
_CLUSTER_FLAGS = ("cluster", "project", "zone", "docker_image")


def define_flags(flag_values: flags.FlagValues = flags.FLAGS) -> None:
  """Registers the baseline's flags, the hyperparameters' included, on `flag_values`."""
  hparams_lib.define_flags(flag_values)
  fv = {"flag_values": flag_values}
  flags.DEFINE_string(
      "device_type", "", "TPU type of the slice the baseline runs on, e.g. v5p-32, v6e-16, tpu7x-4x4x4.", **fv
  )
  flags.DEFINE_integer(
      "num_slices",
      1,
      "Number of slices. The pipeline's stages go over them as the MPMD run places them (whole stages a slice, or whole"
      " slices a stage); --pure_fsdp is data-parallel across them.",
      **fv,
  )
  flags.DEFINE_integer(
      "total_devices", -1, "JAX devices of all the slices. -1 derives them from --device_type and --num_slices.", **fv
  )
  flags.DEFINE_string(
      "run_name",
      "",
      "MaxText run_name, and by default the cluster workload's name; empty: spmd-baseline-<MMDDhhmmss>, so that"
      " repeated runs do not collide.",
      **fv,
  )
  flags.DEFINE_string(
      "output_dir",
      "",
      "MaxText base_output_directory (gs://...), where the xplane profile and TensorBoard files land; empty leaves"
      " MaxText's default.",
      **fv,
  )
  flags.DEFINE_boolean(
      "pure_fsdp",
      False,
      "Emit a Pure FSDP (ici_pipeline_parallelism=1) MaxText configuration with the same global batch size as the MPMD"
      " pipeline configuration.",
      **fv,
  )
  flags.DEFINE_string(
      "xla_flags_file",
      "",
      "File of --key=value XLA flags, one per line with # comments, put in LIBTPU_INIT_ARGS.",
      **fv,
  )
  flags.DEFINE_string("cluster", "", "GKE cluster: also print the command that runs the baseline on it.", **fv)
  flags.DEFINE_enum(
      "launcher",
      "gcluster",
      ["gcluster", "xpk"],
      "With --cluster: `gcluster job submit` (Cluster Toolkit), or `xpk workload create` (MaxText has deprecated XPK).",
      **fv,
  )
  flags.DEFINE_string("project", "", "GCP project of --cluster.", **fv)
  flags.DEFINE_string("zone", "", "Zone (or region) of --cluster.", **fv)
  flags.DEFINE_string("docker_image", "", "Image with MaxText installed, for --cluster.", **fv)
  flags.DEFINE_string("workload", "", "Workload name for --cluster (default: --run_name).", **fv)
  flags.DEFINE_string(
      "gcluster_compute_type", "", "--launcher=gcluster: --compute-type; empty derives it from --device_type.", **fv
  )
  flags.DEFINE_string(
      "gcluster_topology", "", "--launcher=gcluster: --topology; empty derives it from --device_type.", **fv
  )


devices_in_device_type = hparams_lib.devices_in_device_type
devices_in_platform = hparams_lib.devices_in_platform


def deepseek_dense_layers(hparams: hparams_lib.Hparams) -> int:
  """Returns a DeepSeek model's leading dense layers: ``--first_num_dense_layers``, else its model config's."""
  if hparams.first_num_dense_layers >= 0:
    return hparams.first_num_dense_layers
  return int(hparams_lib.read_model_yaml_metadata(hparams.maxtext_model_name).get("first_num_dense_layers", 0))


def pipelined_layers(hparams: hparams_lib.Hparams) -> int:
  """Returns how many decoder layers MaxText's pipeline holds in the baseline; the others run before it.

  The MPMD model pipelines all ``num_layers`` layers. MaxText's pipeline holds
  ``pipeline_parallel_layers`` of them, and for DeepSeek only MoE layers: its
  dense layers, and the MoE layers past the last multiple of the stage count, run
  before the pipeline, data-parallel over every device (``nnx_decoders``:
  ``_init_pipeline_deepseek``). Setting ``pipeline_parallel_layers`` to
  ``num_layers`` there would build the dense layers *and* ``num_layers`` MoE
  layers, a larger model than the MPMD side's.

  Raises:
    ValueError: If the layers cannot fill a balanced MaxText pipeline.
  """
  if hparams.decoder_block != "deepseek":
    if hparams.num_layers % hparams.num_stages:
      raise ValueError(
          f"MaxText's pipeline needs the {hparams.num_layers} layers to divide into the {hparams.num_stages} stages"
          f" (--stage_layer_counts={','.join(map(str, hparams.stage_layer_counts))} has no MaxText equivalent)."
      )
    return hparams.num_layers
  moe_layers = hparams.num_layers - deepseek_dense_layers(hparams)
  pipelined = moe_layers // hparams.num_stages * hparams.num_stages
  if pipelined <= 0:
    raise ValueError(
        f"{hparams.model_name} has {moe_layers} MoE layers in {hparams.num_layers}, fewer than the"
        f" {hparams.num_stages} stages: MaxText's pipeline holds only DeepSeek's MoE layers."
    )
  return pipelined


def unscanned_layers_outside_pipeline(hparams: hparams_lib.Hparams) -> bool:
  """Returns whether the baseline runs the layers outside MaxText's pipeline unscanned (``scan_layers=False``).

  MaxText wraps the *scanned* layers it runs outside its pipeline in
  ``jax.checkpoint`` (``remat_policy`` cannot be ``none`` with pipeline
  parallelism) and the unscanned ones not, so an MPMD run without ``--remat``
  is matched unscanned. But ``scan_layers=False`` changes MaxText's pipeline
  too: it then passes each stage's output on as the stage returns it, where it
  otherwise unwraps DeepSeek's ``(output, cache)`` tuple (``layers/pipeline.py``).
  A stage of several layers unwraps the tuple itself
  (``NNXSequentialPipelineStage``); a stage of one layer is the bare layer,
  whose tuple then fails the pipeline's shift (``zeros_like`` of a tuple). So
  the layers run unscanned only without ``--remat``, when there are any
  (DeepSeek's) and every stage holds more than one layer; otherwise MaxText's
  default (scanned) applies.
  """
  if hparams.decoder_block != "deepseek" or hparams.remat:
    return False
  pipelined = pipelined_layers(hparams)
  return hparams.num_layers > pipelined and pipelined // hparams.num_stages > 1


def _layers(count: int, kind: str) -> str:
  return f"{count} {kind} layer{'' if count == 1 else 's'}"


def baseline_notes(hparams: hparams_lib.Hparams, *, pure_fsdp: bool = False) -> list[str]:
  """Returns how the baseline's program differs from the MPMD run's beyond the pipeline itself."""
  if pure_fsdp:
    return []
  notes = []
  if len(set(hparams.stage_layer_counts)) > 1:
    notes.append(
        f"MaxText's pipeline gives every stage the same number of layers; the MPMD run's"
        f" --stage_layer_counts={','.join(map(str, hparams.stage_layer_counts))} has no MaxText equivalent."
    )
  if hparams.decoder_block != "deepseek":
    return notes
  pipelined = pipelined_layers(hparams)
  outside = hparams.num_layers - pipelined
  if not outside:  # MaxText's pipeline holds every layer, as the MPMD run does.
    return notes
  dense = deepseek_dense_layers(hparams)
  which = [f"DeepSeek's {_layers(dense, 'dense')}"] if dense else []
  if outside > dense:
    which.append(f"the {_layers(outside - dense, 'MoE')} past a multiple of the {hparams.num_stages} stages")
  notes.append(
      f"MaxText's pipeline holds {pipelined} of the {hparams.num_layers} layers, all MoE; {' and '.join(which)}"
      f" {'runs' if outside == 1 else 'run'} before it, data-parallel over all devices, where the MPMD run pipelines"
      " every layer."
  )
  if hparams.remat:
    return notes
  if unscanned_layers_outside_pipeline(hparams):
    how = (
        "Without --remat those layers run unscanned (scan_layers=False), the only MaxText path that does not"
        " rematerialise them under pipeline parallelism (remat_policy=none is refused there)"
    )
  else:
    how = (
        "Without --remat MaxText still rematerialises those layers (remat_policy=full, as MaxText refuses none under"
        " pipeline parallelism): it skips that only for unscanned layers, which its pipeline cannot run with one"
        " DeepSeek layer per stage"
    )
  notes.append(
      f"{how}; for a step-time comparison of DeepSeek, --remat on both sides checkpoints every layer with the same"
      " policy."
  )
  return notes


def spmd_flags(
    hparams: hparams_lib.Hparams,
    *,
    total_devices: int,
    num_slices: int = 1,
    pure_fsdp: bool = False,
) -> dict[str, str]:
  """Returns the MaxText settings reproducing `hparams` with SPMD.

  MaxText's ``ici_*`` degrees describe one slice and its ``dcn_*`` degrees the
  slices (``create_device_mesh`` checks their products). Over several slices
  the stages go where the MPMD run puts them (``mesh_utils``: no stage
  straddles a slice): each slice holds whole stages, so the stage axis is split
  over DCN (``dcn_pipeline_parallelism``) and ICI; or each stage holds whole
  slices, the outermost axis of its mesh, ``fsdp``, over DCN
  (``dcn_fsdp_parallelism``). Pure FSDP is data-parallel across the slices
  (``dcn_data_parallelism``), MaxText's recommended DCN axis.

  Args:
    hparams: The MPMD configuration to match.
    total_devices: Number of JAX devices of all the slices, needed to convert
      the per-device batch size between the two conventions.
    num_slices: Number of slices the devices are in.
    pure_fsdp: When True, emit a Pure FSDP (`ici_pipeline_parallelism=1`)
      configuration matching the MPMD run's global batch size instead of a
      pipelined SPMD configuration.

  Returns:
    An ordered mapping of MaxText config key to value.

  Raises:
    ValueError: If the configuration cannot be expressed as a balanced MaxText
      pipeline, if `total_devices` is not a multiple of the stage count or of
      the slice count, or if neither of the stage and slice counts divides the
      other.
  """
  if total_devices <= 0:
    raise ValueError(f"total_devices must be positive, got {total_devices}.")
  if num_slices <= 0 or total_devices % num_slices:
    raise ValueError(f"total_devices ({total_devices}) must split evenly over num_slices ({num_slices}).")
  if total_devices % hparams.num_stages:
    raise ValueError(f"total_devices ({total_devices}) must be divisible by num_stages" f" ({hparams.num_stages}).")
  if not pure_fsdp and hparams.num_stages % num_slices and num_slices % hparams.num_stages:
    raise ValueError(
        f"{num_slices} slices and {hparams.num_stages} stages must divide one another: the MPMD run puts whole stages"
        " on a slice, or whole slices in a stage."
    )
  if not pure_fsdp and hparams.num_microbatches % hparams.num_stages:
    raise ValueError(
        "MaxText requires num_pipeline_microbatches to be a multiple of the"
        f" stage count; got {hparams.num_microbatches} microbatches for"
        f" {hparams.num_stages} stages."
    )

  devices_per_slice = total_devices // num_slices
  devices_per_stage = total_devices // hparams.num_stages
  # The stage axis's share of the slices (DCN), and how many slices a stage holds.
  dcn_stages = min(num_slices, hparams.num_stages)
  stage_slices = num_slices // dcn_stages
  # Same tokens per step on both sides; see the module docstring.
  global_batch = hparams.per_device_batch_size * devices_per_stage
  spmd_per_device_batch_size = global_batch / total_devices

  if not pure_fsdp and global_batch % hparams.num_microbatches:
    raise ValueError(
        f"Global batch ({global_batch}) is not divisible by" f" num_microbatches ({hparams.num_microbatches})."
    )

  settings: dict[str, object] = {
      "model_name": hparams.maxtext_model_name,
      "base_num_decoder_layers": hparams.num_layers,
  }
  if hparams.emb_dim > 0:
    settings["base_emb_dim"] = hparams.emb_dim
  if hparams_lib.is_pure_moe_model(hparams.maxtext_model_name, hparams.first_num_dense_layers):
    moe_dim = hparams.moe_mlp_dim if hparams.moe_mlp_dim > 0 else hparams.mlp_dim
    if moe_dim > 0:
      settings["base_mlp_dim"] = moe_dim
      settings["base_moe_mlp_dim"] = moe_dim
  else:
    if hparams.mlp_dim > 0:
      settings["base_mlp_dim"] = hparams.mlp_dim
    if hparams.moe_mlp_dim > 0:
      settings["base_moe_mlp_dim"] = hparams.moe_mlp_dim
  if hparams.vocab_size > 0:
    settings["vocab_size"] = hparams.vocab_size
  settings.update(
      {
          "max_target_length": hparams.max_target_length,
          "per_device_batch_size": spmd_per_device_batch_size,
      }
  )
  if pure_fsdp:
    settings.update(
        {
            "ici_pipeline_parallelism": 1,
            "dcn_pipeline_parallelism": 1,
            "scan_layers": True,
            "allow_split_physical_axes": True,
        }
    )
  else:
    settings.update(
        {
            # --- pipeline ---
            "ici_pipeline_parallelism": hparams.num_stages // dcn_stages,
            "dcn_pipeline_parallelism": dcn_stages,
            **({"dcn_fsdp_parallelism": stage_slices} if stage_slices > 1 else {}),
            "num_pipeline_microbatches": hparams.num_microbatches,
            "num_layers_per_pipeline_stage": pipelined_layers(hparams) // hparams.num_stages,
            "num_pipeline_repeats": 1,
            "pipeline_parallel_layers": pipelined_layers(hparams),
            # MaxText's own recommendation: scan the pipeline iterations, not the
            # layers within a stage.
            "scan_layers_per_stage": False,
            "scan_pipeline_iterations": True,
            # 2D torus slices such as v6e-64 (physical [8, 8, 1]) and v6e-128
            # (physical [8, 16, 1]) cannot factor into (stage=4, fsdp=16) or
            # (stage=4, fsdp=32) without splitting a physical axis of size 8 into
            # 4 x 2. MPMD slices flat stage submeshes directly, so the SPMD baseline
            # must allow physical axis splitting to run on the same topologies.
            "allow_split_physical_axes": True,
        }
    )
  settings.update(
      {
          # --- match the MPMD determinism settings ---
          "dataset_type": "synthetic",
          "enable_checkpointing": False,
          "enable_goodput_recording": False,
          "monitor_goodput": False,
          "enable_dropout": False,
          "use_iota_embed": True,
          "override_model_config": True,
          # Rematerialization: the MPMD model checkpoints its layers only under
          # --remat (per layer, with --remat_policy). MaxText checkpoints its
          # pipeline's scan iterations when
          # `set_remat_policy_on_pipeline_iterations` is true, so that switch
          # follows --remat. `remat_policy` itself cannot be "none" with
          # pipeline parallelism (MaxText refuses it), so without --remat a
          # pipelined baseline names "full": the pipeline's own stage layers
          # are then not checkpointed (neither switch is on), but MaxText wraps
          # the *scanned* layers it runs outside the pipeline (DeepSeek's dense
          # and remaining MoE layers, `pipelined_layers`) in `jax.checkpoint`
          # whenever remat_policy is not "none"; see `scan_layers` below.
          "set_remat_policy_on_pipeline_iterations": hparams.remat,
          "set_remat_policy_on_layers_per_stage": False,
          "remat_policy": hparams.remat_policy if hparams.remat else ("none" if pure_fsdp else "full"),
          "learning_rate": hparams.learning_rate,
          "init_weights_seed": hparams.seed,
          # --- steps and profiling ---
          # Like train.py: warm-up steps, then the timed steps, then extra steps
          # under the profiler, so the profiler never slows a timed step (with the
          # profiler over the timed steps, its start made one 3.4 s and the next
          # 12 ms). MaxText writes the xplane profile under `tensorboard_dir`,
          # derived from `base_output_directory` (--output_dir) and `run_name`.
          "steps": hparams.warmup_steps + hparams.steps + _PROFILER_STEPS,
          "profiler": "xplane",
          "skip_first_n_steps_for_profiler": hparams.warmup_steps + hparams.steps,
          "profiler_steps": _PROFILER_STEPS,
          # The MPMD side's trace (benchmark.profile): one TPU, one SparseCore and
          # one SparseCore tile a traced host.
          "enable_tpu_profiling_options": True,
          **hparams_lib.TRACE_ADVANCED_CONFIGURATION,
      }
  )
  expert = max(hparams.expert, 1)
  # Mirrors `config.build_maxtext_config`: both sides must dispatch tokens to
  # experts the same way for the comparison to be about the pipeline.
  settings.update(hparams_lib.moe_dispatch_overrides(hparams))
  if pure_fsdp:
    if devices_per_slice % expert:
      raise ValueError(f"Devices per slice ({devices_per_slice}) is not divisible by the expert degree ({expert}).")
    fsdp_deg = hparams.fsdp * hparams.num_stages if hparams.fsdp > 0 else total_devices // expert
    # Each slice holds its share of the FSDP degree; the slices are data-parallel.
    if fsdp_deg % num_slices or devices_per_slice % (fsdp_deg // num_slices * expert):
      raise ValueError(
          f"total_devices ({total_devices}) in {num_slices} slice(s) does not split into Pure FSDP"
          f" degree x expert degree ({fsdp_deg} x {expert})."
      )
    settings["ici_fsdp_parallelism"] = fsdp_deg // num_slices
    tensor = devices_per_slice // (fsdp_deg // num_slices * expert)
    if tensor > 1:
      settings["ici_tensor_parallelism"] = tensor
    if num_slices > 1:
      settings["dcn_data_parallelism"] = num_slices
  elif hparams.fsdp > 0 or expert > 1:
    fsdp_deg = hparams.fsdp if hparams.fsdp > 0 else devices_per_stage // expert
    # A stage mesh is `fsdp x expert x tensor`; the tensor axis is whatever is
    # left over. It is normally 1, but not on platforms with more than one core
    # per chip (a TPU v7x 4x4x4 slice reports 128 devices for its 64 chips).
    # MaxText validates that the product of the ICI dimensions equals the
    # device count of a slice, so this axis has to be reproduced or the
    # baseline will not start.
    if devices_per_stage % (fsdp_deg * expert):
      raise ValueError(
          f"Devices per stage ({devices_per_stage}) is not divisible by the"
          f" FSDP x expert degree ({fsdp_deg} x {expert})."
      )
    if fsdp_deg % stage_slices:
      raise ValueError(
          f"Each stage holds {stage_slices} slices, over which its outermost axis, fsdp, runs: its FSDP degree"
          f" ({fsdp_deg}) must be a multiple of {stage_slices}."
      )
    settings["ici_fsdp_parallelism"] = fsdp_deg // stage_slices
    tensor = devices_per_stage // (fsdp_deg * expert)
    if tensor > 1:
      settings["ici_tensor_parallelism"] = tensor
  # As `config.build_maxtext_config`: `simple` and `default` build the
  # simple_mlp block, where MaxText's `default` config resolves to base.yml's
  # (llama2), so the baseline pins it; a named model's config gives its block
  # to both sides.
  decoder_block = hparams.decoder_block
  if hparams.maxtext_model_name == "default":
    settings["decoder_block"] = decoder_block
  if decoder_block == "deepseek" or hparams_lib.uses_yarn_rope(hparams.maxtext_model_name):
    # Mirrors `config.build_maxtext_config`. MPMD cannot run YaRN RoPE at all
    # (PartIR's MarkFragmentReservedMemoryPass asserts on the complex tensors
    # it produces -- see the comment there), so the MPMD side pins the standard
    # real-valued RoPE. The baseline has to pin it too, or the two sides stop
    # being the same model (nothing here tests the two sides' FLOPs for this
    # drift).
    settings["rope_type"] = "default"
  if decoder_block == "deepseek" and hparams.first_num_dense_layers >= 0:
    settings["first_num_dense_layers"] = hparams.first_num_dense_layers
  if not pure_fsdp and unscanned_layers_outside_pipeline(hparams):
    # MaxText runs the layers outside its pipeline unscanned without
    # `jax.checkpoint`, scanned with it; otherwise MaxText's default (scanned).
    settings["scan_layers"] = False
  # Mirrors `config.build_maxtext_config`: float32 GMM tiles, then the
  # overrides, applied last.
  settings.update(hparams_lib.float32_gmm_tiles(hparams))
  for item in hparams.maxtext_overrides:
    key, _, value = item.partition("=")
    settings[key] = value

  return {key: str(value) for key, value in settings.items()}


def trainer_command(
    hparams: hparams_lib.Hparams,
    *,
    total_devices: int,
    run_name: str,
    num_slices: int = 1,
    output_dir: str = "",
    pure_fsdp: bool = False,
    xla_flags: dict[str, str] | None = None,
) -> str:
  """Returns the shell command that runs MaxText's trainer with :func:`spmd_flags`.

  Args:
    hparams: The MPMD configuration to match.
    total_devices: Number of JAX devices in the run, of all its slices.
    run_name: MaxText ``run_name``; the profile lands under it.
    num_slices: Number of slices the devices are in.
    output_dir: MaxText ``base_output_directory`` (``gs://...``); empty omits it.
    pure_fsdp: Emit Pure FSDP (`ici_pipeline_parallelism=1`) flags.
    xla_flags: ``{key: value}`` XLA flags for ``LIBTPU_INIT_ARGS`` (see
      :func:`hparams.read_xla_flags_file`); None or empty sets none.

  Returns:
    A one-line shell command.
  """
  settings = spmd_flags(hparams, total_devices=total_devices, num_slices=num_slices, pure_fsdp=pure_fsdp)
  settings["run_name"] = run_name
  if output_dir:
    settings["base_output_directory"] = output_dir
  args = " ".join(shlex.quote(f"{key}={value}") for key, value in settings.items())
  env = ""
  if xla_flags:
    xla = " ".join(f"--{key}={value}" for key, value in xla_flags.items())
    env = f"export LIBTPU_INIT_ARGS={shlex.quote(xla)}; "
  return f"{env}python3 -u -m {MAXTEXT_TRAIN_MODULE} {args}"


def xpk_command(
    command: str,
    *,
    cluster: str,
    project: str,
    zone: str,
    device_type: str,
    num_slices: int,
    workload: str,
    docker_image: str,
) -> str:
  """Returns the ``xpk workload create`` command that runs `command` on every host of the slice(s)."""
  argv = [
      ("--cluster", cluster),
      ("--project", project),
      ("--zone", zone),
      ("--tpu-type", device_type),
      ("--num-slices", str(num_slices)),
      ("--workload", workload),
      ("--docker-image", docker_image),
      ("--command", command),
  ]
  return " \\\n".join(["xpk workload create", *(f"  {name} {shlex.quote(value)}" for name, value in argv)])


def gcluster_command(
    command: str,
    *,
    cluster: str,
    project: str,
    zone: str,
    compute_type: str,
    topology: str,
    num_slices: int,
    workload: str,
    docker_image: str,
) -> str:
  """Returns the ``gcluster job config set`` lines and the ``gcluster job submit`` that runs `command` on every host."""
  config = "\n".join(shlex.join(argv) for argv in launch.gcluster_config_commands(project, cluster, zone))
  argv = [
      ("--name", workload),
      ("--image", docker_image),
      ("--compute-type", compute_type),
      ("--topology", topology),
      ("--num-slices", str(num_slices)),
      ("--command", command),
  ]
  return (
      config + "\n" + " \\\n".join(["gcluster job submit", *(f"  {name}={shlex.quote(value)}" for name, value in argv)])
  )


def main(argv: Sequence[str], flag_values: flags.FlagValues = flags.FLAGS) -> None:
  if len(argv) > 1:
    raise app.UsageError(f"Unexpected command line arguments: {argv[1:]}")

  fv = flag_values
  try:
    hparams = hparams_lib.from_flags(fv)
  except (ValueError, FileNotFoundError) as error:
    raise app.UsageError(str(error)) from error
  total_devices = fv.total_devices
  if total_devices <= 0:
    total_devices = devices_in_device_type(fv.device_type) * fv.num_slices
  if total_devices <= 0:
    raise app.UsageError(
        f"Could not derive the device count from --device_type={fv.device_type!r}; pass --total_devices."
    )
  run_name = fv.run_name or f"spmd-baseline-{time.strftime('%m%d%H%M%S')}"
  if fv.cluster:
    missing = [f"--{name}" for name in _CLUSTER_FLAGS if not fv[name].value]
    if not fv.device_type:
      missing.append("--device_type")
    if missing:
      raise app.UsageError(f"--cluster needs {', '.join(missing)}.")

  # Before anything is printed: a configuration MaxText's pipeline cannot run is a usage error, not a traceback.
  try:
    settings = spmd_flags(hparams, total_devices=total_devices, num_slices=fv.num_slices, pure_fsdp=fv.pure_fsdp)
  except ValueError as error:
    raise app.UsageError(str(error)) from error
  mode_desc = "Pure FSDP" if fv.pure_fsdp else f"SPMD pipeline ({hparams.num_stages} stages)"
  for note in baseline_notes(hparams, pure_fsdp=fv.pure_fsdp):
    max_logging.warning(f"spmd_baseline: {note}")
  slices_desc = f" in {fv.num_slices} slices" if fv.num_slices > 1 else ""
  print(f"Equivalent MaxText {mode_desc} configuration ({total_devices} devices{slices_desc}):")
  for key, value in settings.items():
    print(f"  {key}={value}")
  command = trainer_command(
      hparams,
      total_devices=total_devices,
      run_name=run_name,
      num_slices=fv.num_slices,
      output_dir=fv.output_dir,
      pure_fsdp=fv.pure_fsdp,
      xla_flags=hparams_lib.read_xla_flags_file(fv.xla_flags_file) if fv.xla_flags_file else None,
  )
  print("\nRun MaxText's trainer with:\n")
  print(command)
  if not fv.cluster:
    return
  cluster = {"cluster": fv.cluster, "project": fv.project, "zone": fv.zone, "docker_image": fv.docker_image}
  workload = fv.workload or run_name
  print("\nOn the cluster:\n")
  if fv.launcher == "xpk":
    print(xpk_command(command, device_type=fv.device_type, num_slices=fv.num_slices, workload=workload, **cluster))
    return
  compute_type, topology = launch.gcluster_options(fv)
  print(
      gcluster_command(
          command,
          compute_type=compute_type,
          topology=topology,
          num_slices=fv.num_slices,
          workload=workload,
          **cluster,
      )
  )


if __name__ == "__main__":
  define_flags()
  app.run(main)
