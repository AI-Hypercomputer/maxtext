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

"""Wires one harness run: real weights + pinned references -> the L1 / L2 / L3 / L4 gate, after the AOT check.

Targets per stage:
  L1: "forward" (gate L1) or "vjp" (gate L4 on the L1 path).
  L2: a group of consecutive layers, "0,1,2,3".
  L3: a scan geometry "2|3;4|5" (chain invariance, tier `lw_l3.L3_CHAIN_TIER`) or "moe:<layer>" (MoE invariance,
    tier `lw_l3.L3_MOE_TIER`).
  L4: one layer, "20".
Every program (`run_programs`) must appear in the AOT report with the run's tier, geometry, layers, activation quant
and harness code stamp, compiled and fitting. `finalize_report` turns the gate report into the run verdict: only PASS
passes, NOT_APPLICABLE_GEOMETRY only for a cell of the pinned expected-not-applicable file whose geometry it
recomputes, and every negative control must fire.
"""

import dataclasses
import pathlib
from collections.abc import Iterator, Mapping
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.layers import attention_deepseek41
from maxtext.models import deepseek41

from tests.utils.deepseek_v41_layerwise import (  # isort: skip
    lw_bundle,
    lw_checks,
    lw_config,
    lw_l1,
    lw_l2,
    lw_l3,
    lw_l4,
    lw_model,
    lw_programs,
)

# Leaf every leaf-perturbing negative control moves: present in every decoder layer, consumed by h_out.
NC_DECODER_LEAF = "mlp/shared_experts/wo/kernel"
NC_VISION_LEAF = "vision_encoder/DeepSeek41VisionEncoder_0/blocks_15/attn/wo/kernel"
NC_ALIGNER_LEAF = "vision_encoder/DeepSeek41VisionProjector_0/w2/kernel"
# Stage keys of measured_noise_floors.json `recommended_tolerances`.
FLOOR_STAGE = {"L1": "L1_vision_aligner", "L2": "L2_decoder_groups", "L4": "L4_backward"}
L1_TARGETS = ("forward", "vjp")
L3_MOE_TARGET_PREFIX = "moe:"
# Seed of the L3 output cotangent (both sides of every L3 comparison use the same one).
L3_COTANGENT_SEED = 0
EXPECTED_NOT_APPLICABLE_PATH = lw_checks.TOLERANCES_PATH.parent / "layerwise_expected_not_applicable.json"
_NC_CONTAINERS = ("negative_control", "negative_controls")
_NC_STATUSES = (lw_l2.FIRED, lw_l2.DID_NOT_FIRE)


@dataclasses.dataclass(frozen=True)
class ExpectedNotApplicable:
  """The pinned (stage, sample, layer, check) cells where NOT_APPLICABLE_GEOMETRY is expected, with their geometry."""

  sha256: str
  cells: Mapping[tuple[str, str, int, str], Mapping[str, int]]


def load_expected_not_applicable(path: pathlib.Path) -> ExpectedNotApplicable:
  """Parses the expected-not-applicable file (exact keys; a duplicate cell raises)."""
  pinned = lw_checks.load_pinned_json(path, expected_sha256=None)
  data = lw_checks.require_keys(pinned.data, ("schema_version", "entries"), str(path))
  if data["schema_version"] != 1:
    raise ValueError(f"{path} schema_version {data['schema_version']} != 1")
  cells = {}
  for entry in data["entries"]:
    lw_checks.require_keys(entry, ("stage", "sample", "layer", "check", "geometry", "reason"), f"{path} entry {entry}")
    cell = (entry["stage"], entry["sample"], int(entry["layer"]), entry["check"])
    if cell in cells:
      raise ValueError(f"{path} lists {cell} twice")
    cells[cell] = dict(entry["geometry"])
  return ExpectedNotApplicable(sha256=pinned.sha256, cells=cells)


@dataclasses.dataclass(frozen=True)
class StageRun:
  """One harness run and its pinned inputs; `memory` collects device memory records as the run goes."""

  stage: str
  tier: str
  sample: str
  target: str
  index: lw_bundle.BundleIndex
  hf_dir: pathlib.Path
  tokenizer_path: str
  tolerances: lw_checks.Tolerances
  floors: lw_bundle.NoiseFloors
  aot: lw_checks.PinnedJson
  expected_na: ExpectedNotApplicable
  memory: list[dict[str, Any]]
  perturb_mode: str = "none"
  run_defect2_multi_block_conditioning_gate: bool = False


@dataclasses.dataclass(frozen=True)
class EngramRows:
  """Rows of one layer's Engram table kept on device, and the input whose hashed rows the slice must cover."""

  active: np.ndarray
  controls: np.ndarray
  tokens: np.ndarray
  image_mask: np.ndarray
  include_packing_rows: bool = True


def run_programs(stage: str, target: str, num_devices: int) -> tuple[str, ...]:
  """The AOT program names one run executes (the one owner of the run -> program mapping)."""
  if stage == "L1":
    if target not in L1_TARGETS:
      raise ValueError(f"L1 target {target!r} is not one of {L1_TARGETS}")
    return (lw_programs.L1_VJP_PROGRAM if target == "vjp" else lw_programs.L1_FORWARD_PROGRAM,)
  if stage == "L2":
    return (lw_programs.group_program_name(parse_layers(target)),)
  if stage == "L4":
    (layer_id,) = parse_layers(target)
    return (lw_programs.layer_vjp_program_name(layer_id),)
  if stage == "L3":
    if target.startswith(L3_MOE_TARGET_PREFIX):
      return tuple(v.program_name() for v in lw_l3.moe_variants(num_devices).values())
    return tuple(lw_programs.chain_program_name(mode) for mode in lw_l3.L3_CHAIN_MODES)
  raise ValueError(f"Stage {stage!r} is not one of L1 / L2 / L3 / L4")


def aot_geometry(run: StageRun, sample: lw_bundle.SampleGeometry) -> dict[str, Any]:
  """The geometry fields every AOT record of this run must carry."""
  return {
      "sample": run.sample,
      "seq_len": sample.seq_len,
      "batch_size": sample.batch_size,
      "num_images": sample.num_images,
      "vision_grid": list(sample.vision_grid),
  }


def require_aot(run: StageRun, program: str, config: Any, match: Mapping[str, Any]) -> dict[str, Any]:
  """The AOT record of `program` for this tier, the activation quant of `config` and every field of `match`.

  Raises:
    ValueError: Unless exactly one record matches, it compiled and fits, and it was compiled from this harness code
      (`lw_checks.harness_code_sha256`).
  """
  full = {**match, "activation_quant": bool(config.ds41_activation_quant)}
  matches = [
      r
      for r in lw_checks.require(run.aot.data, "programs", str(run.aot.path))
      if "program" in r and r["program"] == program and r["tier"] == run.tier and all(r[k] == v for k, v in full.items())
  ]
  if len(matches) != 1:
    raise ValueError(f"AOT report {run.aot.path} has {len(matches)} records for {program} / {run.tier} / {full}")
  record = matches[0]
  code = lw_checks.harness_code_sha256()
  if record["harness_code_sha256"] != code:
    raise ValueError(
        f"AOT record for {program} was compiled from harness code {record['harness_code_sha256']}, not {code}"
    )
  model_code = lw_checks.model_code_sha256()
  if record["model_code_sha256"] != model_code:
    raise ValueError(
        f"AOT record for {program} was compiled from model code {record['model_code_sha256']}, not {model_code}"
    )
  if not (record["compiled"] and record["fits_device_hbm"]):
    raise ValueError(f"AOT record for {program} / {run.tier} did not compile or fit: {record}")
  return record


def preflight_aot(  # pylint: disable=too-many-positional-arguments,too-many-arguments
    stage: str,
    tier: str,
    sample: str,
    target: str,
    aot_json_data: Mapping[str, Any],
    aot_json_path: str = "aot.json",
    num_devices: int = 4,
    activation_quant: bool = False,
) -> list[dict[str, Any]]:
  """Validates require_aot upfront in milliseconds before loading weights, building mesh, or running cells.

  Raises:
    ValueError: If the AOT record is missing, failed compilation, or has a stale harness/model SHA256.
  """
  programs_list = lw_checks.require(aot_json_data, "programs", str(aot_json_path))
  sample_recs = [r for r in programs_list if r.get("sample") == sample]
  if not sample_recs:
    raise ValueError(f"AOT report {aot_json_path} contains no programs for sample {sample}")
  first_r = sample_recs[0]
  match_base = {
      "sample": sample,
      "seq_len": first_r["seq_len"],
      "batch_size": first_r["batch_size"],
      "num_images": first_r["num_images"],
      "vision_grid": list(first_r["vision_grid"]),
  }
  checks = []
  if stage == "L1":
    (prog,) = run_programs(stage, target, num_devices)
    checks.append((prog, match_base))
  elif stage == "L2":
    layer_ids = parse_layers(target)
    (prog,) = run_programs(stage, target, num_devices)
    checks.append((prog, {**match_base, "layers": list(layer_ids), "engram_sliced": True}))
  elif stage == "L3":
    if target.startswith("moe:"):
      layer_id = int(target.split(":", 1)[1])
      for _, variant in lw_l3.moe_variants(num_devices).items():
        checks.append((variant.program_name(), {**match_base, "layers": [layer_id]}))
    else:
      geom = parse_geometry(target)
      for prog in run_programs(stage, target, num_devices):
        checks.append((prog, {**match_base, "layers": list(geom.layer_ids)}))
  elif stage == "L4":
    (layer_id,) = parse_layers(target)
    (prog,) = run_programs(stage, target, num_devices)
    checks.append((prog, {**match_base, "layers": [layer_id], "engram_sliced": True}))
  else:
    raise ValueError(f"Unknown stage {stage}")

  verified_records = []
  code_sha = lw_checks.harness_code_sha256()
  model_sha = lw_checks.model_code_sha256()

  for program, match in checks:
    full = {**match, "activation_quant": activation_quant}
    matches = [
        r
        for r in programs_list
        if "program" in r
        and r["program"] == program
        and r.get("tier") == tier
        and all(r.get(k) == v for k, v in full.items())
    ]
    if len(matches) != 1:
      raise ValueError(f"AOT report {aot_json_path} has {len(matches)} records for {program} / {tier} / {full}")
    record = matches[0]
    if record["harness_code_sha256"] != code_sha:
      raise ValueError(
          f"AOT record for {program} was compiled from harness code {record['harness_code_sha256']}, not {code_sha}"
      )
    if record["model_code_sha256"] != model_sha:
      raise ValueError(
          f"AOT record for {program} was compiled from model code {record['model_code_sha256']}, not {model_sha}"
      )
    if not (record["compiled"] and record["fits_device_hbm"]):
      raise ValueError(f"AOT record for {program} / {tier} did not compile or fit: {record}")
    verified_records.append(record)

  return verified_records


def stage_config(
    run: StageRun, sample: lw_bundle.SampleGeometry, *, expert_parallelism: int, extra: Mapping[str, Any]
) -> Any:
  """The run's MaxText config; activation quant follows the reference variant the tier is compared to."""
  grid_extra = {} if sample.num_images == 0 else {"ds41_vision_grid": list(sample.vision_grid)}
  return lw_config.build_config(
      run.tier,
      reference_activation_quant=run.index.reference_activation_quant(run.sample, lw_bundle.TARGET_VARIANT[run.tier]),
      tokenizer_path=run.tokenizer_path,
      seq_len=sample.seq_len,
      batch_size=sample.batch_size,
      num_devices=len(jax.devices()),
      expert_parallelism=expert_parallelism,
      extra={**grid_extra, **extra},
  )


def bounds_for(run: StageRun, stage: str) -> lw_l2.GateBounds:
  """Gate bounds: tier a the tolerances file + code_parity floors; tier b the measured floors of `stage`. Agreement
  limits and the margin multiplier come from the floors file in both tiers."""
  production = run.tier == lw_config.TIER_PRODUCTION
  stage_key = FLOOR_STAGE[stage]
  has_cp = "code_parity" in run.floors.data.get("recommended_tolerances", {}).get(stage_key, {})
  return lw_l2.GateBounds(
      run.tier,
      run.tolerances,
      run.floors.production_bounds(stage_key) if production else None,
      run.floors.allowed_moe_failures(),
      run.floors.allowed_indexer_failures(),
      run.floors.bf16_gap_multiplier() if production else None,
      code_parity=run.floors.code_parity_bounds(stage_key) if not production and has_cp else None,
  )


def load_layer(  # pylint: disable=too-many-positional-arguments
    config: Any,
    mesh: Any,
    layer_id: int,
    hf_dir: pathlib.Path,
    engram: EngramRows | None,
    perturb_mode: str = "none",
) -> tuple[lw_model.ModuleParts, lw_model.EngramSlice | None]:
  """Real layer `layer_id` from the HF checkpoint; with `engram`, its Engram table is that row slice.

  Raises:
    ValueError: If the slice misses a row the layer's hasher reaches on `engram`'s input.
  """
  build = lw_programs.layer_build_fn(config, mesh, layer_id)
  parts = lw_model.abstract_parts(build, config, mesh)
  target = nnx.to_pure_dict(parts.weights)
  plan = table = full_table_abstract = None
  if engram is not None:
    table_abstract = target["engram"].pop("embed")
    active_rows = engram.active
    control_rows = engram.controls
    if engram.include_packing_rows and engram.tokens.shape[1] >= 16:
      temp_mod = lw_model.merge_fresh(parts.graphdef, parts.weights, parts.rest)
      pack_rows = lw_model.engram_packing_control_rows(temp_mod.engram_hasher.val, engram.tokens)
      active_rows = np.unique(np.concatenate([active_rows, pack_rows]))
      control_rows = np.setdiff1d(control_rows, active_rows)
    plan = lw_model.plan_engram_slice(
        active_rows, control_rows, num_table_rows=table_abstract.shape[0], row_multiple=mesh.devices.size
    )
    rows = lw_model.read_engram_rows(hf_dir, layer_id, plan.row_ids)
    table = lw_model.build_engram_slice_table(rows, plan, table_abstract.dtype, table_abstract.sharding)
  elif "engram" in target:
    full_table_abstract = target["engram"].pop("embed")
  loaded = lw_model.load_weights({"decoder": {f"layers_{layer_id}": target}}, config, hf_dir)["decoder"][
      f"layers_{layer_id}"
  ]
  if table is not None:
    loaded["engram"]["embed"] = table
    parts.weights["engram"]["embed"].set_value(jax.ShapeDtypeStruct(table.shape, table.dtype, sharding=table.sharding))
  elif full_table_abstract is not None:
    loaded["engram"]["embed"] = lw_model.load_full_engram_table(hf_dir, layer_id, full_table_abstract)
  if perturb_mode != "none" and layer_id == 9:
    gate_kernel = loaded["mlp"]["MoeBlock_0"]["gate"]["kernel"]
    if perturb_mode in ("scale_gate_1ulp_bf16_l9", "scale_gate_1ulp_bf16"):
      bf16_ulp_mult = 1.0 + (2.0**-7)
      scale_vec = jnp.where(
          (jnp.arange(gate_kernel.shape[1]) % 2) == 0,
          jnp.float32(bf16_ulp_mult),
          jnp.float32(1.0),
      )
      perturbed_kernel = (gate_kernel.astype(jnp.float32) * scale_vec[None, :]).astype(gate_kernel.dtype)
      loaded["mlp"]["MoeBlock_0"]["gate"]["kernel"] = jax.device_put(perturbed_kernel, gate_kernel.sharding)
    elif perturb_mode == "swap_experts_l9":
      col0 = gate_kernel[:, 0:1]
      col1 = gate_kernel[:, 1:2]
      rest_cols = gate_kernel[:, 2:]
      swapped_kernel = jnp.concatenate([col1, col0, rest_cols], axis=1)
      loaded["mlp"]["MoeBlock_0"]["gate"]["kernel"] = jax.device_put(swapped_kernel, gate_kernel.sharding)
    else:
      raise ValueError(f"Unsupported perturb_mode {perturb_mode!r}")
  filled = lw_model.fill_weights(parts, loaded)
  out = lw_model.ModuleParts(filled.graphdef, filled.weights, lw_model.materialize_rest(build, config, mesh))
  if plan is not None:
    hasher_input = {
        "input_ids": jnp.asarray(engram.tokens, jnp.int32),
        "image_mask": jnp.asarray(engram.image_mask, jnp.bool_),
    }
    lw_model.verify_slice_covers(out.merge().engram_hasher.val, plan, [hasher_input])
  return out, plan


def load_l1(config: Any, mesh: Any, hf_dir: pathlib.Path) -> lw_model.ModuleParts:
  """The real vision encoder + projector and token embedder from the HF checkpoint."""
  build = lw_programs.l1_build_fn(config, mesh)
  parts = lw_model.abstract_parts(build, config, mesh)
  filled = lw_model.fill_weights(parts, lw_model.load_weights(nnx.to_pure_dict(parts.weights), config, hf_dir))
  return lw_model.ModuleParts(filled.graphdef, filled.weights, lw_model.materialize_rest(build, config, mesh))


def run_stage(run: StageRun) -> dict[str, Any]:
  """Runs the gate of `run.stage` (see the module docstring); the result still needs `finalize_report`."""
  sample = run.index.sample(run.sample)
  stages = {"L1": run_l1, "L2": run_l2, "L3": run_l3, "L4": run_l4}
  if run.stage not in stages:
    raise ValueError(f"Stage {run.stage!r} is not one of {sorted(stages)}")
  return stages[run.stage](run, sample)


# pylint: disable=too-many-positional-arguments
def run_l1_embed_only(
    parts: lw_model.ModuleParts,
    config: Any,
    mesh: Any,
    tokens_np: np.ndarray,
    ref_h0: lw_l2.ReferenceTensor,
    bounds: lw_l2.GateBounds,
    vjp: bool,
    cotangent_h: np.ndarray | None = None,
) -> dict[str, Any]:
  """Runs L1.embed forward or VJP for 0-image samples (T0-T3) exercising token_embedder directly."""
  del config
  rep = lw_programs.replicated(mesh)
  tokens = jax.device_put(jnp.asarray(tokens_np, jnp.int32), rep)
  nc_leaf = "token_embedder/embedding"

  if not vjp:

    def embed_fwd(weights):
      mod = lw_model.merge_fresh(parts.graphdef, weights, parts.rest)
      return mod.token_embedder(tokens, model_mode=lw_programs.MODEL_MODE_TRAIN)

    cand_emb = np.asarray(jax.device_get(jax.jit(embed_fwd)(parts.weights)))
    checks = [lw_l2.compare_tensor("token_embedder.h0", cand_emb, ref_h0, None, bounds, "embed_tokens")]
    moved = lw_model.perturb_leaf(parts.weights, nc_leaf, bounds.tol.negative_control_rel_scale, seed=0)
    nc_emb = np.asarray(jax.device_get(jax.jit(embed_fwd)(moved)))
    nc_check = lw_l2.compare_tensor("token_embedder.h0", nc_emb, ref_h0, None, bounds, "embed_tokens")
    lw_model.release(moved, keep=parts.weights)
    fired = lw_checks.control_fired(checks, [nc_check])
    return {
        "status": lw_checks.worst_status([c["status"] for c in checks]),
        "checks": checks,
        "negative_control": {
            "NC_L1_embed": {
                "leaf": nc_leaf,
                "status": lw_l2.FIRED if fired else lw_l2.DID_NOT_FIRE,
            }
        },
    }
  else:
    assert cotangent_h is not None
    cot = jax.device_put(jnp.asarray(cotangent_h, jnp.float32), rep)

    def embed_vjp(weights):
      def loss(w):
        mod = lw_model.merge_fresh(parts.graphdef, w, parts.rest)
        emb = mod.token_embedder(tokens, model_mode=lw_programs.MODEL_MODE_TRAIN)
        return lw_programs.objective_term(emb, cot)

      return jax.grad(loss)(weights)

    def evaluate(weights):
      d_w = jax.jit(embed_vjp)(weights)
      pure = nnx.to_pure_dict(d_w)
      grad = pure["token_embedder"]["embedding"]
      rows = np.unique(tokens_np)
      derived = lw_l1.embedding_rows_reference(tokens_np, ref_h0, rows, grad.shape)
      return [lw_l2.compare_tensor("embedding.rows", grad[rows], derived, None, bounds, "dW_embedding_rows")]

    checks = evaluate(parts.weights)
    moved = lw_model.perturb_leaf(parts.weights, nc_leaf, bounds.tol.negative_control_rel_scale, seed=0)
    nc = evaluate(moved)
    lw_model.release(moved, keep=parts.weights)
    fired = lw_checks.control_fired(checks, nc)
    return {
        "status": lw_checks.worst_status([c["status"] for c in checks]),
        "checks": checks,
        "negative_control": {
            "NC_L1_embed": {
                "leaf": nc_leaf,
                "status": lw_l2.FIRED if fired else lw_l2.DID_NOT_FIRE,
            }
        },
    }


def run_l1(run: StageRun, sample: lw_bundle.SampleGeometry) -> dict[str, Any]:
  """Gate L1 (`target`="forward") or gate L4 on the L1 path (`target`="vjp")."""
  (program,) = run_programs(run.stage, run.target, len(jax.devices()))
  vjp = run.target == "vjp"
  extra = {"scan_layers": False, "remat_policy": "none", "freeze_vision_encoder_params": not vjp}
  config = stage_config(run, sample, expert_parallelism=1, extra=extra)
  mesh = lw_config.build_mesh(config, jax.devices())
  require_aot(run, program, config, aot_geometry(run, sample))
  parts = load_l1(config, mesh, run.hf_dir)
  run.memory.append(lw_model.device_memory("load_l1"))
  if sample.num_images == 0:
    ref_bundle = run.index.open(lw_bundle.l2_path(run.sample, 0))
    tokens = ref_bundle.get("tokens")
    ref_h0 = ref_bundle.tensor("layer_0_h_in", run.tier)
    cot = ref_bundle.get("cotangent_h_out") if vjp else None
    report = run_l1_embed_only(parts, config, mesh, tokens, ref_h0, bounds_for(run, "L4" if vjp else "L1"), vjp, cot)
  elif vjp:
    ref = lw_bundle.l1_vjp_reference(run.index, run.sample, run.tier)
    report = lw_l1.run_vjp(parts.graphdef, parts, config, mesh, ref, bounds_for(run, "L4"), nc_leaf=NC_ALIGNER_LEAF)
  else:
    ref = lw_bundle.l1_reference(run.index, run.sample, run.tier)
    report = lw_l1.run_forward(parts.graphdef, parts, config, mesh, ref, bounds_for(run, "L1"), nc_leaf=NC_VISION_LEAF)
  run.memory.append(lw_model.device_memory(program))
  return report


def run_l2(run: StageRun, sample: lw_bundle.SampleGeometry) -> dict[str, Any]:
  """Gate L2 on the group `target`; both tier a and tier b hold Engram tables as row slices."""
  layer_ids = parse_layers(run.target)
  (program_name,) = run_programs(run.stage, run.target, len(jax.devices()))
  sliced = True
  config = stage_config(run, sample, expert_parallelism=1, extra={"scan_layers": False, "remat_policy": "none"})
  mesh = lw_config.build_mesh(config, jax.devices())
  require_aot(
      run, program_name, config, {**aot_geometry(run, sample), "layers": list(layer_ids), "engram_sliced": sliced}
  )
  specs = deepseek41.build_layer_specs(config)
  ulp = run.floors.ulp_multiplier()
  references = [lw_bundle.l2_reference(run.index, run.sample, i, config, run.tier, ulp) for i in layer_ids]
  refs = [r for r, _ in references]
  inputs = references[0][1]
  parts, plans = [], []
  for i in layer_ids:
    engram = None
    if sliced and specs[i].has_engram:
      active, controls = lw_bundle.engram_rows(run.index, lw_bundle.l2_path(run.sample, i), f"layer_{i}_")
      engram = EngramRows(active, controls, inputs.tokens, inputs.image_mask)
    layer_parts, plan = load_layer(config, mesh, i, run.hf_dir, engram, perturb_mode=run.perturb_mode)
    run.memory.append(lw_model.device_memory(f"load_layer[{i}]"))
    parts.append(layer_parts)
    plans.append(plan)
  program = lw_programs.GroupProgram(config, mesh, tuple(layer_ids), tuple(p.graphdef for p in parts), tuple(plans))
  if run.run_defect2_multi_block_conditioning_gate:
    dim = config.head_dim
    base = float(config.rope_theta)
    defect2_inv_freq = 1.0 / (np.float32(base) ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)))
    with mock.patch.object(
        attention_deepseek41.DeepSeek41RotaryEmbedding, "inv_freq", jnp.asarray(defect2_inv_freq, dtype=jnp.float32)
    ):
      report = lw_l2.run_group(program, parts, refs, inputs, bounds_for(run, "L2"), nc_leaf=NC_DECODER_LEAF)
  else:
    report = lw_l2.run_group(program, parts, refs, inputs, bounds_for(run, "L2"), nc_leaf=NC_DECODER_LEAF)
  run.memory.append(lw_model.device_memory(program_name))
  return report


def run_l4(run: StageRun, sample: lw_bundle.SampleGeometry) -> dict[str, Any]:
  """Gate L4 on layer `target`; an Engram table is the slice of the reference's active + inactive (control) rows."""
  (layer_id,) = parse_layers(run.target)
  (program_name,) = run_programs(run.stage, run.target, len(jax.devices()))
  config = stage_config(run, sample, expert_parallelism=1, extra={"scan_layers": False, "remat_policy": "none"})
  mesh = lw_config.build_mesh(config, jax.devices())
  require_aot(run, program_name, config, {**aot_geometry(run, sample), "layers": [layer_id], "engram_sliced": True})
  ref, inputs = lw_bundle.l4_reference(run.index, run.sample, layer_id, config, run.tier)
  engram = None
  if deepseek41.build_layer_specs(config)[layer_id].has_engram:
    active, controls = lw_bundle.engram_rows(run.index, lw_bundle.l4_path(run.sample, layer_id), "")
    engram = EngramRows(active, controls, inputs.tokens, inputs.image_mask)
  parts, plan = load_layer(config, mesh, layer_id, run.hf_dir, engram)
  run.memory.append(lw_model.device_memory(f"load_layer[{layer_id}]"))
  program = lw_programs.GroupProgram(config, mesh, (layer_id,), (parts.graphdef,), (plan,))
  report = lw_l4.run_layer(program, parts, ref, inputs, bounds_for(run, "L4"), nc_leaf=NC_DECODER_LEAF)
  run.memory.append(lw_model.device_memory(program_name))
  return report


def run_l3(run: StageRun, sample: lw_bundle.SampleGeometry) -> dict[str, Any]:
  """L3 MoE invariance for a "moe:<layer>" target, else the scan / remat chain invariance."""
  if run.target.startswith(L3_MOE_TARGET_PREFIX):
    return run_l3_moe(run, sample, int(run.target[len(L3_MOE_TARGET_PREFIX) :]))
  return run_l3_chain(run, sample)


def run_l3_chain(run: StageRun, sample: lw_bundle.SampleGeometry) -> dict[str, Any]:
  """L3 scan / remat invariance on the real layers of the scan geometry `target`, from the sample's reference carry."""
  if run.tier != lw_l3.L3_CHAIN_TIER:
    raise ValueError(f"L3 chain runs in tier {lw_l3.L3_CHAIN_TIER}, not {run.tier}")
  lw_config.configure_runtime(lw_config.TIER_CODE_PARITY, platform=jax.default_backend())
  geometry = parse_geometry(run.target)
  unscanned = {"scan_layers": False, "remat_policy": "none", **lw_config.L3_CHAIN_PRECISION_OVERRIDES}
  base = stage_config(run, sample, expert_parallelism=1, extra=unscanned)
  mesh = lw_config.build_mesh(base, jax.devices())
  for name in run_programs(run.stage, run.target, len(jax.devices())):
    require_aot(run, name, base, {**aot_geometry(run, sample), "layers": list(geometry.layer_ids)})
  chain = lw_l3.load_chain(base, mesh, geometry.layer_ids, lambda c, m, i: load_layer(c, m, i, run.hf_dir, None)[0])
  run.memory.append(lw_model.device_memory("load_chain"))
  inputs = lw_bundle.chain_inputs(run.index, run.sample, geometry.layer_ids[0], base, L3_COTANGENT_SEED)
  report = lw_l3.run_chain_invariance(
      chain,
      stage_config(run, sample, expert_parallelism=1, extra={**unscanned, "remat_policy": "full"}),
      stage_config(run, sample, expert_parallelism=1, extra={**unscanned, "scan_layers": True}),
      geometry,
      inputs,
      run.tolerances,
      NC_DECODER_LEAF,
  )
  run.memory.append(lw_model.device_memory("L3.chain"))
  return {**report, "status": lw_checks.worst_status([report["remat"]["status"], report["scan_vs_unscan"]["status"]])}


def run_l3_moe(run: StageRun, sample: lw_bundle.SampleGeometry, layer_id: int) -> dict[str, Any]:
  """L3 MoE invariance on real layer `layer_id`: EP=1 vs EP=all (replayed routing), dense vs sparse (natural)."""
  if run.tier != lw_l3.L3_MOE_TIER:
    raise ValueError(f"L3 MoE runs in tier {lw_l3.L3_MOE_TIER}, not {run.tier}")
  configs = {}
  for name, variant in lw_l3.moe_variants(len(jax.devices())).items():
    config = stage_config(run, sample, expert_parallelism=variant.expert_parallelism, extra=variant.config_overrides())
    require_aot(run, variant.program_name(), config, {**aot_geometry(run, sample), "layers": [layer_id]})
    configs[name] = (config, lw_config.build_mesh(config, jax.devices()))
  config1, mesh1 = configs["dense_ep1"]
  if deepseek41.build_layer_specs(config1)[layer_id].has_engram:
    raise ValueError(f"L3 MoE runs on a layer without an Engram table; layer {layer_id} has one")
  base, _ = load_layer(config1, mesh1, layer_id, run.hf_dir, None)
  run.memory.append(lw_model.device_memory(f"load_layer[{layer_id}]"))
  inputs = lw_bundle.chain_inputs(run.index, run.sample, layer_id, config1, L3_COTANGENT_SEED)
  replay = lw_bundle.layer_replay(run.index, run.sample, layer_id, config1)
  report = lw_l3.run_moe_invariance(layer_id, base, configs, inputs, replay, run.tolerances, NC_DECODER_LEAF)
  run.memory.append(lw_model.device_memory("L3.moe"))
  families = [report["ep1_vs_epn"], *report["moe_implementations_vs_dense"].values()]
  return {**report, "status": lw_checks.worst_status([f["status"] for f in families])}


def parse_layers(text: str) -> tuple[int, ...]:
  """ "0,1,2,3" -> (0, 1, 2, 3)."""
  return tuple(int(x) for x in text.split(","))


def parse_geometry(text: str) -> lw_programs.ScanGeometry:
  """ "2|3;4|5" -> prefix (2,), repeats ((3,), (4,)), suffix (5,)."""
  prefix, repeats, suffix = text.split("|")
  return lw_programs.ScanGeometry(
      prefix=parse_layers(prefix), repeats=tuple(parse_layers(g) for g in repeats.split(";")), suffix=parse_layers(suffix)
  )


def _walk(node: Any, path: str) -> Iterator[tuple[str, str, Mapping[str, Any]]]:
  """Yields ("check" | "control", path, dict) for every status-bearing leaf dict under `node`. Under a
  `negative_control(s)` key, dicts are controls; elsewhere a status dict with no status dict below it is a check."""
  if isinstance(node, Mapping):
    for key, value in node.items():
      if key in _NC_CONTAINERS:
        controls = {key: value} if "status" in value else value
        for name, control in controls.items():
          yield "control", f"{path}/{name}", control
        continue
      yield from _walk(value, f"{path}/{key}")
    if "status" in node and not any(kind == "check" for kind, _, _ in _walk_children(node, path)):
      yield "check", path, node
  elif isinstance(node, list):
    for i, value in enumerate(node):
      yield from _walk(value, f"{path}[{i}]")


def _walk_children(node: Mapping[str, Any], path: str) -> Iterator[tuple[str, str, Mapping[str, Any]]]:
  for key, value in node.items():
    if key not in _NC_CONTAINERS:
      yield from _walk(value, f"{path}/{key}")


def finalize_report(report: Mapping[str, Any], run: StageRun) -> dict[str, Any]:
  """The run verdict: PASS iff every check is PASS or an expected NOT_APPLICABLE_GEOMETRY cell whose geometry matches
  the pinned file, and every negative control FIRED. Adds `status_counts`, the accepted / unaccepted entries and the
  controls that did not fire; an unknown status raises."""
  layers = report["layers"] if "layers" in report and isinstance(report["layers"], Mapping) else {}
  indexer_passed = any(
      isinstance(entry, Mapping)
      and "agreement" in entry
      and "indexer" in entry["agreement"]
      and entry["agreement"]["indexer"]["status"] == lw_checks.STATUS_PASS
      for entry in layers.values()
  )
  if indexer_passed and ("negative_controls" not in report or "NC_L2e" not in report["negative_controls"]):
    raise ValueError("An L2 layer passed indexer_agreement, so negative_controls must include NC_L2e")
  counts = {s: 0 for s in lw_checks.CHECK_STATUSES}
  accepted, unaccepted, not_fired = [], [], []
  for kind, path, node in _walk({k: v for k, v in report.items() if k != "status"}, ""):
    status = node["status"]
    if kind == "control":
      if status not in _NC_STATUSES + lw_checks.CHECK_STATUSES:
        raise ValueError(f"Negative control {path} has unknown status {status!r}")
      if status != lw_l2.FIRED:
        if status in (lw_checks.STATUS_NOT_APPLICABLE_GEOMETRY, "NOT_APPLICABLE"):
          continue
        not_fired.append({"control": path, "status": status})
      continue
    if status not in lw_checks.CHECK_STATUSES:
      raise ValueError(f"Check {path} has status {status!r} outside {lw_checks.CHECK_STATUSES}")
    counts[status] += 1
    if status == lw_checks.STATUS_PASS:
      continue
    if status == lw_checks.STATUS_NOT_APPLICABLE_GEOMETRY:
      cell = (run.stage, run.sample, int(node["layer"]), node["check"])
      listed = cell in run.expected_na.cells
      if listed and dict(run.expected_na.cells[cell]) == dict(node["geometry"]):
        accepted.append({"check": path, "cell": list(cell)})
        continue
      unaccepted.append(
          {"check": path, "status": status, "cell": list(cell), "geometry": node["geometry"], "cell_listed": listed}
      )
      continue
    unaccepted.append({"check": path, "status": status})
  passed = not unaccepted and not not_fired and counts[lw_checks.STATUS_PASS] > 0
  return {
      **report,
      "gate_status": report["status"],
      "status": lw_checks.STATUS_PASS if passed else lw_checks.STATUS_FAIL,
      "status_counts": counts,
      "accepted_not_applicable": accepted,
      "unaccepted_statuses": unaccepted,
      "negative_controls_not_fired": not_fired,
  }
