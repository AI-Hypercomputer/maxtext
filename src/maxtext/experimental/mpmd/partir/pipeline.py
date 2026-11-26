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
"""Construction of the Shardy MPMD (OSS PartIR:MPMD) configuration.

This is the one place that decides how a :class:`hparams.Hparams` becomes an
``mpmd.Config``, so that the trainer and the lowering check build the same
program: "it lowers" then says something about "it runs".

Written for OSS Shardy MPMD as jaxlib 0.11.1 to 0.12.1 pin it (see
``sdy_mpmd_versions.SHARDY_COMMITS``), which shapes it:

  * options are a plain dict of ``mpmd_*`` keys, and the schedule is a string
    (:data:`hparams.OSS_SCHEDULE_NAMES`);
  * jaxlib's MPMD driver reads only five boolean options and the schedule
    (``ParsePartitioningOptions`` in mpmd_program.cc); the others are accepted
    by Python and then ignored, so they stay at their defaults;
  * nothing ties an output's layout to an input's (``make_config`` has no
    such argument); see :func:`build_plan`;
  * the lowered IFRT IR lives at ``lowered.partitioning_result.ifrt_ir_module``,
    a frozen dataclass, which :func:`pin_ifrt_transfers_to_producers` replaces.

Shardy MPMD is an optional dependency (``sdy_mpmd_compat``): it is imported
when an ``mpmd.Config`` is first needed, so building a :class:`PipelinePlan` for
the SPMD reference, and the tests that only do that, run without it.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import re
from typing import TYPE_CHECKING, Any

from maxtext.experimental.mpmd.partir import hparams as hparams_lib
from maxtext.experimental.mpmd.partir import mesh_utils

if TYPE_CHECKING:
  from maxtext.experimental.mpmd.partir.sdy_mpmd_compat import mpmd
  from maxtext.experimental.mpmd.partir.sdy_mpmd_compat import shardy_pipeline

# Partitioning options applied to every MPMD program built here.
#
# `infer_transfers` lets Shardy insert the cross-mesh transfers implied by the
# data dependencies between named computations, and
# `infer_cross_mesh_reductions` does the same for the gradient reductions that
# appear when a parameter is used on more than one mesh. Without them the user
# has to write every transfer by hand.
_BASE_PARTITIONING_OPTIONS = {
    "mpmd_infer_cross_mesh_reductions": True,
    "mpmd_infer_transfers": True,
    # Ignored by jaxlib's driver (always False there); kept so the program
    # asks for it once the driver reads it.
    "mpmd_absorb_inferred_fragments_on_entry_point_function": True,
    # Keep per-microbatch activation/gradient transfers independent: merging
    # adjacent `ifrt.CopyArrays` across microbatches (e.g. `dX_2k` and
    # `dX_{2k+1}` on stage 3 in 1F1B) creates a false cross-microbatch barrier
    # that stalls upstream stages until both microbatches complete. Also
    # ignored by jaxlib's driver, whose default is the same False.
    "mpmd_apply_merge_transfers_pass": False,
}


def shardy_mpmd() -> Any:
  """Returns Shardy MPMD's frontend (``mpmd``), importing it (``sdy_mpmd_compat``) on first use."""
  from maxtext.experimental.mpmd.partir import sdy_mpmd_compat  # pylint: disable=import-outside-toplevel

  return sdy_mpmd_compat.mpmd


def _shardy_pipeline() -> Any:
  """Returns Shardy MPMD's ``pipeline`` module (schedules and merge rules), importing it on first use."""
  from maxtext.experimental.mpmd.partir import sdy_mpmd_compat  # pylint: disable=import-outside-toplevel

  return sdy_mpmd_compat.shardy_pipeline


@dataclasses.dataclass(frozen=True)
class PipelinePlan:
  """Everything needed to lower and run one MPMD pipeline.

  Attributes:
    topology: The stage meshes backing the pipeline.
    name_to_mesh: Mapping from named computation to mesh name.
    name_to_stage: Mapping from named computation to stage id.
    schedule: The schedule in use (a :data:`hparams.PIPELINE_SCHEDULES` name).
    num_repeats: Pipeline traversals per forward pass (see :func:`build_plan`).
  """

  topology: mesh_utils.StageTopology
  name_to_mesh: dict[str, str]
  name_to_stage: dict[str, int]
  schedule: str
  num_repeats: int = 1

  @functools.cached_property
  def mpmd_config(self) -> mpmd.Config:
    """Returns the Shardy MPMD configuration, built (and Shardy imported) on first use.

    Raises:
      ValueError: If Shardy's sharding mesh is not :attr:`sharding_mesh`, which
        the shardings built from it assume.
    """
    config = _make_mpmd_config(self)
    if config.sharding_mesh != self.sharding_mesh:
      raise ValueError(
          f"Shardy MPMD's sharding mesh {config.sharding_mesh} is not stage 0's abstract mesh {self.sharding_mesh}."
      )
    return config

  @property
  def sharding_mesh(self) -> Any:
    """Returns the abstract mesh logical parameter shardings resolve against.

    It is what ``mpmd.Config.sharding_mesh`` returns, stage 0's abstract mesh:
    every stage mesh has the same shape and axis names, and Shardy re-shards each
    parameter onto its own stage's mesh when it partitions.
    """
    return self.topology.meshes[mesh_utils.mesh_name(0)].abstract_mesh

  @property
  def layer_mesh(self) -> Any:
    """Returns a concrete stage mesh, for *building* the model's layers.

    `sharding_mesh` is abstract. Under MPMD the parameters of different stages
    live on different device meshes, so no single concrete mesh spans them.
    That is fine for resolving logical shardings, but some MaxText layers
    inspect the mesh's devices while tracing -- DeepSeek's MLA attention picks
    its kernel with `self.mesh.devices[...].platform` (MaxText's
    `layers/attention_op.py`) -- and an abstract mesh raises `ValueError: AbstractMesh does not implement
    devices`. Stage 0's mesh, whose abstract mesh `sharding_mesh` is, stands in.
    """
    return self.topology.meshes[mesh_utils.mesh_name(0)]

  @property
  def num_devices(self) -> int:
    """Returns the total number of devices across all stages."""
    return self.topology.num_stages * self.topology.devices_per_stage

  def describe(self) -> str:
    """Returns a short human readable summary of the plan."""
    stage_counts: dict[str, int] = {}
    for name, mesh in self.name_to_mesh.items():
      if name.startswith(("pre_m", "post_m")):
        continue
      stage_counts[mesh] = stage_counts.get(mesh, 0) + 1
    placement = ", ".join(
        f"{mesh_utils.mesh_name(i)}:{stage_counts.get(mesh_utils.mesh_name(i), 0)}"
        for i in range(self.topology.num_stages)
    )
    return (
        f"  schedule       : {self.schedule}\n"
        f"  topology       : {self.topology.describe()}\n"
        f"  total devices  : {self.num_devices}\n"
        f"  computations   : {placement}"
    )


def _merge_last_stage_fwd_bwd(
    f1: shardy_pipeline.FragmentInfo,
    f2: shardy_pipeline.FragmentInfo,
    context: shardy_pipeline.PipelineContext,
) -> bool:
  """Merges forward and backward fragments of the same microbatch on the last stage.

  A split backward (the Zero Bubble schedules) merges its activation-gradient
  fragment B_a with the forward; the weight-gradient fragment B_w stays apart, for
  the schedule to place.
  """
  sdy_pipeline = _shardy_pipeline()
  info = sdy_pipeline.get_staged_scheduling_info(f1, f2, "last-stage fwd-bwd merge")
  if info is None:
    return False
  call_f1, transpose_f1, call_f2, transpose_f2 = info
  return (
      f1.stage_id == context.num_meshes - 1
      and f2.stage_id == context.num_meshes - 1
      and call_f1 == call_f2
      and transpose_f1 != transpose_f2
      and sdy_pipeline.SplitFragmentType.DROP_TRANSFERRED not in (f1.split_type, f2.split_type)
  )


def build_plan(
    hparams: hparams_lib.Hparams,
    *,
    devices: Sequence[Any] | None = None,
    num_repeats: int = 1,
) -> PipelinePlan:
  """Builds the MPMD pipeline described by `hparams`.

  Args:
    hparams: Pipeline hyperparameters.
    devices: Devices to build the stage meshes from. Defaults to
      ``jax.devices()``; AOT callers pass virtual topology devices instead.
    num_repeats: Number of pipeline traversals per forward pass. Values > 1
      produce a circular (interleaved) layer placement.

  Returns:
    The resulting :class:`PipelinePlan`; its ``mpmd_config`` is built when it is
    first read.
  """
  if devices is None:
    devices = mesh_utils.resolve_devices()

  topology = mesh_utils.create_stage_meshes(
      devices=devices,
      num_stages=hparams.num_stages,
      fsdp_size=hparams.fsdp,
      expert_size=hparams.expert,
  )
  name_to_mesh = mesh_utils.create_name_to_mesh_assignment(
      num_layers=hparams.num_layers,
      num_stages=hparams.num_stages,
      num_repeats=num_repeats,
      stage_layer_counts=hparams.stage_layer_counts,
  )
  name_to_stage = mesh_utils.create_name_to_stage_assignment(
      num_layers=hparams.num_layers,
      num_stages=hparams.num_stages,
      num_repeats=num_repeats,
      stage_layer_counts=hparams.stage_layer_counts,
  )
  return PipelinePlan(
      topology=topology,
      name_to_mesh=name_to_mesh,
      name_to_stage=name_to_stage,
      schedule=hparams.pipeline_schedule,
      num_repeats=num_repeats,
  )


def _make_mpmd_config(plan: PipelinePlan) -> mpmd.Config:
  """Builds `plan`'s ``mpmd.Config``: schedule, partitioning options and merge rule."""
  sdy_pipeline = _shardy_pipeline()
  partitioning_options: dict[str, bool | str] = dict(_BASE_PARTITIONING_OPTIONS)
  partitioning_options["mpmd_pipeline_schedule"] = hparams_lib.OSS_SCHEDULE_NAMES[plan.schedule]
  if plan.schedule.startswith("ZERO_BUBBLE_"):
    # The Zero Bubble schedules order split backward fragments. With this option
    # (one of the five options jaxlib's driver reads) Shardy's
    # SplitBwdFragmentsPass cuts each backward fragment that transfers a result
    # upstream into B_a, which computes the transferred activation gradient
    # (split_keep_transferred), and B_w, which computes the rest -- the weight
    # gradients -- from residuals B_a returns (split_drop_transferred); the
    # schedule runs B_w in what would be a bubble. The first stage transfers
    # nothing upstream, so its backward stays whole, and a weight gradient
    # computed by the op that also gives the activation gradient (the routed
    # experts' in MaxText's MoE shard_map) stays in B_a. pipeline_test checks
    # the split.
    partitioning_options["mpmd_split_bwd_fragments"] = True

  # Stage ids are passed explicitly: OSS does not fill them in from the mesh
  # index for the 1F1B / GPipeBut1F1B / ZB schedules, and the predicate rules
  # (`get_staged_scheduling_info`) raise on a fragment without one.
  make_config_kwargs: dict[str, Any] = {
      "partitioning_options": partitioning_options,
      "name_to_stage_assignment": plan.name_to_stage,
  }
  if plan.num_repeats == 1 and (plan.schedule == "ONE_FWD_ONE_BWD" or plan.schedule.startswith("ZERO_BUBBLE_")):
    # The last stage runs each microbatch's forward and backward -- under the
    # Zero Bubble schedules, its B_a -- back to back; merged, the forward's
    # residuals (the output head's logits among them) stay inside one program,
    # and B_w stays a fragment of its own (_merge_last_stage_fwd_bwd).
    # MergeForwardWithBackwardPass merges F_{S-1}(m) and B_{S-1}(m) at the
    # consumer position B_{S-1}(m) (after F_{S-2}(m+1) -> T_{S-2,S-1}(m+1)),
    # which places T_{S-2,S-1}(m) and T_{S-2,S-1}(m+1) adjacent before
    # F_{S-1}B_{S-1}(m) and causes IfrtMergeCopiesAndReshardsPass to fuse them.
    # RuleBasedMergePass instead merges F_{S-1}(m) and B_{S-1}(m) at the
    # producer position F_{S-1}(m) (immediately after T_{S-2,S-1}(m) and before
    # T_{S-2,S-1}(m+1)), keeping every forward and backward transfer unmerged.
    # (OSS never runs MergeForwardWithBackwardPass, and it honours merge
    # rules, so this rule is the only forward/backward merge.)
    make_config_kwargs["pipeline_schedule"] = sdy_pipeline.PipelineSchedule(
        schedule_merge_rule_builders=[
            functools.partial(
                sdy_pipeline.build_merge_rules_from_predicate,
                pred=_merge_last_stage_fwd_bwd,
            )
        ]
    )

  # The new state (output 0) must be laid out exactly like the state that
  # produced it (input 0), or the second call rejects its own predecessor's
  # output with "Input signature does not match the signature of the compiled
  # function". OSS `make_config` has no argument that ties the new state's
  # layout to the old one, so the callers pin the state's shardings on both
  # sides (`mpmd.jit(in_shardings=..., out_shardings=(state_shardings,
  # None))`), and each parameter's update is computed on the mesh that holds
  # it, so the meshes agree; `train.py` checks it by feeding step N's state to
  # step N+1.
  return shardy_mpmd().make_config(plan.topology.meshes, plan.name_to_mesh, **make_config_kwargs)


_SSA_TOKEN_RE = re.compile(r"(%[A-Za-z0-9_]+(?:#\d+)?)(?![A-Za-z0-9_#])")
_IFRT_CALL_RE = re.compile(r"^\s*(%[A-Za-z0-9_]+)(?::\d+)?,\s*(%[A-Za-z0-9_]+)\s*=\s*ifrt\.Call\b")
_COPY_HEADER_RE = re.compile(
    r"^(\s*)(%[A-Za-z0-9_]+)(?::(\d+))?,\s*(%[A-Za-z0-9_]+)\s*=\s*" r"ifrt\.CopyArrays\(([^)]*)\)(.*)$"
)


def _split_top_level_comma(text: str) -> list[str]:
  """Splits a comma-separated MLIR list at bracket depth 0."""
  parts: list[str] = []
  depth = 0
  start = 0
  for idx, ch in enumerate(text):
    if ch in "<([{":
      depth += 1
    elif ch in ">)]}":
      depth = max(0, depth - 1)
    elif ch == "," and depth == 0:
      part = text[start:idx].strip()
      if part:
        parts.append(part)
      start = idx + 1
  tail = text[start:].strip()
  if tail:
    parts.append(tail)
  return parts


def _split_arrow_top_level(text: str) -> tuple[str, str] | None:
  """Splits ``(in_types) -> out_types`` at bracket depth 0."""
  depth = 0
  for idx, ch in enumerate(text):
    if ch in "<([{":
      depth += 1
    elif ch in ">)]}":
      depth = max(0, depth - 1)
    elif depth == 0 and text.startswith(" -> ", idx):
      return text[:idx].strip(), text[idx + 4 :].strip()
  return None


def _replace_ssa_tokens(line: str, replacements: dict[str, str]) -> str:
  """Replaces exact SSA token occurrences in an MLIR statement line."""
  if not replacements:
    return line
  return _SSA_TOKEN_RE.sub(lambda m: replacements.get(m.group(1), m.group(1)), line)


def pin_ifrt_transfers_to_producers(lowered: Any) -> Any:
  """Un-merges cross-microbatch ``ifrt.CopyArrays`` and pins them to producers.

  Both MPMD's ``LowerToIfrt`` and IFRT's ``CompiledIfrtIrProgram::Create``
  (invoked inside ``lowered.compile()``) unconditionally run
  ``IfrtMergeCopiesAndReshardsPass``, which batches any two ``ifrt.CopyArrays``
  in ``@main`` that share ``(source_devices, dst_devices, donated)`` and have
  empty ``getControlInputs()``. In 1F1B steady state on the last stage, two
  ``fwd_bwd`` calls execute between upstream ``bwd`` calls, so their activation
  gradient transfers ``dX_{2k}`` and ``dX_{2k+1}`` get merged into a single
  ``ifrt.CopyArrays`` scheduled after ``fwd_bwd_{2k+1}``, stalling upstream
  ``bwd_{2k}`` for an entire extra ``fwd_bwd`` duration.

  This function:
    1. Splits any ``ifrt.CopyArrays`` whose inputs come from multiple distinct
       ``ifrt.Call`` producers into one ``ifrt.CopyArrays`` per producer Call,
       placed immediately after that producer Call.
    2. Attaches ``after %producer_control_output`` to every ``ifrt.CopyArrays``
       that depends on an ``ifrt.Call`` producer, which causes
       ``IfrtMergeCopiesAndReshardsPass`` inside ``lowered.compile()`` to keep
       each microbatch's transfer independent.

  The stall was measured with jax 0.11.1. With jax 0.12.0.1, Shardy 38ab99f4
  and Pathways 20261117 on TPU7x, deepseek3-tiny's 1F1B and GPipe programs (PP=4,
  8 microbatches) ran the same 51 transfers a step with and without the pins
  (one ``TransferManager::Recv`` per copy and destination device in the trace),
  at the same step time within noise: nothing was merged either way, and the
  pins are a guard for the stacks and schedules that do merge. On that stack
  ``LowerToIfrt`` does merge two producers' transfers for
  GPipeBut1F1BForLastMesh, ZB-H1 and the ZB-H2 schedules, which step 1 splits
  (``pipeline_test`` checks the rewrite on a synthetic program).

  Args:
    lowered: The :class:`MpmdLowered` object returned by
      ``mpmd.jit(...).lower``.

  Returns:
    ``lowered`` with its IFRT IR module replaced in-place. OSS keeps the module
    only in the frozen ``lowered.partitioning_result`` (there is no
    ``lowered.ifrt_ir_module``); ``compile()`` and ``as_text("ifrt")`` read it
    from there, so that is what gets replaced.
  """
  result = lowered.partitioning_result
  orig_module = result.ifrt_ir_module

  ir_text = str(orig_module)
  func_pos = ir_text.find("func.func public @main(")
  if func_pos < 0:
    return lowered
  sig_end = ir_text.find("\n", func_pos)
  if sig_end < 0:
    return lowered
  body_start = sig_end + 1
  body_end = ir_text.find("\n  }\n", body_start)
  if body_end < 0:
    return lowered

  body_text = ir_text[body_start:body_end]
  body_lines = body_text.splitlines()

  # Count occurrences of every SSA token in @main so we never split a CopyArrays
  # whose control output is explicitly referenced elsewhere.
  token_counts: dict[str, int] = {}
  for m in _SSA_TOKEN_RE.finditer(body_text):
    tok = m.group(1)
    token_counts[tok] = token_counts.get(tok, 0) + 1

  blocks: list[list[str]] = []
  call_info: dict[str, tuple[int, str]] = {}
  ssa_replacements: dict[str, str] = {}
  changed = False

  # A line-by-line rewrite of @main's text: one statement per line, with nested matching.
  # pylint: disable=too-many-nested-blocks
  for raw_line in body_lines:
    line = _replace_ssa_tokens(raw_line, ssa_replacements)
    if line != raw_line:
      changed = True

    m_call = _IFRT_CALL_RE.match(line)
    if m_call:
      out_base, ctrl_var = m_call.group(1), m_call.group(2)
      block_idx = len(blocks)
      blocks.append([line])
      call_info[out_base] = (block_idx, ctrl_var)
      continue

    if "ifrt.CopyArrays(" in line and " : (!ifrt.array" in line:
      header, type_rest = line.split(" : (!ifrt.array", 1)
      m_copy = _COPY_HEADER_RE.match(header)
      arrow_split = _split_arrow_top_level("(!ifrt.array" + type_rest)
      if m_copy and arrow_split is not None:
        indent, out_base, _, ctrl_out, inputs_str, tail = m_copy.groups()
        in_types_raw, out_types_raw = arrow_split
        if in_types_raw.startswith("(") and in_types_raw.endswith(")") and not re.search(r"\bafter\s+%", tail):
          in_types = _split_top_level_comma(in_types_raw[1:-1])
          if out_types_raw.startswith("(") and out_types_raw.endswith(")"):
            out_types = _split_top_level_comma(out_types_raw[1:-1])
          else:
            out_types = [out_types_raw.strip()]
          inputs = [x.strip() for x in inputs_str.split(",") if x.strip()]

          if len(inputs) == len(in_types) == len(out_types) and inputs:
            input_producers = [call_info.get(inp.split("#", 1)[0]) for inp in inputs]
            distinct_producers: list[tuple[int, str]] = []
            for prod in input_producers:
              if prod is not None and prod not in distinct_producers:
                distinct_producers.append(prod)

            if (
                len(distinct_producers) >= 2
                and all(p is not None for p in input_producers)
                and token_counts.get(ctrl_out, 0) <= 1
            ):
              groups: list[tuple[tuple[int, str], list[int]]] = []
              group_map: dict[tuple[int, str], list[int]] = {}
              for idx, prod in enumerate(input_producers):
                assert prod is not None
                if prod not in group_map:
                  idxs: list[int] = []
                  group_map[prod] = idxs
                  groups.append((prod, idxs))
                group_map[prod].append(idx)

              for g_idx, ((prod_block_idx, prod_ctrl), idxs) in enumerate(groups):
                sub_out_base = f"{out_base}_pin{g_idx}"
                sub_ctrl_out = f"{ctrl_out}_pin{g_idx}"
                if len(idxs) == 1:
                  lhs = f"{sub_out_base}, {sub_ctrl_out}"
                  rhs_out_type = out_types[idxs[0]]
                  ssa_replacements[f"{out_base}#{idxs[0]}"] = sub_out_base
                else:
                  lhs = f"{sub_out_base}:{len(idxs)}, {sub_ctrl_out}"
                  rhs_out_type = "(" + ", ".join(out_types[i] for i in idxs) + ")"
                  for j, orig_i in enumerate(idxs):
                    ssa_replacements[f"{out_base}#{orig_i}"] = f"{sub_out_base}#{j}"
                rhs_in_type = "(" + ", ".join(in_types[i] for i in idxs) + ")"
                sub_inputs = ", ".join(inputs[i] for i in idxs)
                sub_line = (
                    f"{indent}{lhs} = ifrt.CopyArrays({sub_inputs}) after"
                    f" {prod_ctrl}{tail} : {rhs_in_type} -> {rhs_out_type}"
                )
                blocks[prod_block_idx].append(sub_line)

              blocks.append([])
              changed = True
              continue

            if len(distinct_producers) == 1:
              _, prod_ctrl = distinct_producers[0]
              prefix = header[: len(header) - len(tail)]
              pinned_line = f"{prefix} after {prod_ctrl}{tail} : (!ifrt.array{type_rest}"
              blocks.append([pinned_line])
              changed = True
              continue

    blocks.append([line])

  if not changed:
    return lowered

  new_body_lines: list[str] = []
  for block in blocks:
    new_body_lines.extend(block)
  new_ir_text = ir_text[:body_start] + "\n".join(new_body_lines) + ir_text[body_end:]
  new_module = type(orig_module).parse(new_ir_text, context=orig_module.context)
  lowered.partitioning_result = dataclasses.replace(result, ifrt_ir_module=new_module)
  return lowered
