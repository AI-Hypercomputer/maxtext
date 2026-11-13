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

"""Ahead-of-time compiles every layerwise program for a TPU topology and records its per-device HBM.

The programs are the harness's own (`lw_programs`), lowered with abstract inputs on the topology mesh of
`train_compile.get_topology_mesh`; the device count comes from that topology. Each program is a real compile:
a program that does not fit is recorded with the compiler's memory analysis (or its error), never estimated.

Usage (host CPU, no TPU needed):
  python3 -m tests.utils.deepseek_v41_layerwise.aot_check_layerwise --compile-topology=v5p-8 \
      --tokenizer-path=<hf dir> --seq-len=2048 --batch-size=1 --output-json=<path>
"""

import argparse
import dataclasses
import functools
import json
import pathlib
import resource
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from jax.sharding import Mesh

from maxtext.models import deepseek41
from maxtext.trainers.pre_train import train_compile
from tests.utils.deepseek_v41_layerwise import lw_checks, lw_config, lw_l3, lw_model, lw_programs, lw_stages

_GIB = float(2**30)


def topology_config(args: argparse.Namespace, tier: str, *, expert_parallelism: int, extra: Mapping[str, Any]) -> Any:
  """A layerwise config targeting `args.compile_topology` (its device count sizes the mesh)."""
  grid_extra = {} if args.num_images == 0 else {"ds41_vision_grid": list(args.vision_grid)}
  return lw_config.build_config(
      tier,
      reference_activation_quant=args.reference_activation_quant,
      tokenizer_path=args.tokenizer_path,
      seq_len=args.seq_len,
      batch_size=args.batch_size,
      num_devices=args.num_devices,
      expert_parallelism=expert_parallelism,
      extra={
          "compile_topology": args.compile_topology,
          "compile_topology_num_slices": 1,
          **grid_extra,
          **extra,
      },
  )


def memory_stats(compiled: Any, device_hbm_gib: float) -> dict[str, Any]:
  """Per-device HBM of a compiled program from `compiled.memory_analysis()`."""
  mem = compiled.memory_analysis()
  if mem is None:
    raise RuntimeError("compiled.memory_analysis() returned None")
  io_bytes = mem.argument_size_in_bytes + mem.output_size_in_bytes - mem.alias_size_in_bytes
  total = mem.temp_size_in_bytes + io_bytes
  return {
      "temp_gib": mem.temp_size_in_bytes / _GIB,
      "argument_gib": mem.argument_size_in_bytes / _GIB,
      "output_gib": mem.output_size_in_bytes / _GIB,
      "alias_gib": mem.alias_size_in_bytes / _GIB,
      "temp_plus_io_gib": total / _GIB,
      "fits_device_hbm": total / _GIB <= device_hbm_gib,
  }


def compile_program(
    name: str, fn: Callable[..., Any], abstract_args: Sequence[Any], device_hbm_gib: float
) -> dict[str, Any]:
  """Lowers and compiles `fn(*abstract_args)`; records memory stats, or the compiler error if compilation fails."""
  t0 = time.time()
  try:
    compiled = jax.jit(fn).lower(*abstract_args).compile()
  except Exception as e:  # pylint: disable=broad-exception-caught
    # An out-of-memory or lowering error is the result of this program's check, recorded verbatim.
    return {
        "program": name,
        "compiled": False,
        "error": f"{type(e).__name__}: {e}"[:4000],
        "traceback_tail": traceback.format_exc()[-2000:],
        "compile_seconds": time.time() - t0,
    }
  return {
      "program": name,
      "compiled": True,
      **memory_stats(compiled, device_hbm_gib),
      "compile_seconds": time.time() - t0,
  }


def engram_slice_bound(config: Any, batch: int, seq: int, control_rows: int) -> int:
  """Most Engram slice rows a run can need: every token's hash columns distinct, plus the controls and the poison row."""
  cols = (config.ds41_engram_max_ngram_size - 1) * config.ds41_engram_n_heads
  return batch * seq * cols + control_rows + 1


def abstract_layer(config: Any, mesh: Mesh, layer_id: int, slice_rows: int | None) -> tuple[Any, Any, Any, Any]:
  """(graphdef, abstract weights, abstract rest, EngramSlice plan or None) of real layer `layer_id`.

  With `slice_rows`, the Engram table is the slice the harness builds (rows rounded up to the device count).
  """
  build = lw_programs.layer_build_fn(config, mesh, layer_id)
  parts = lw_model.abstract_parts(build, config, mesh)
  plan = None
  if slice_rows is not None:
    num_table_rows = parts.weights["engram"]["embed"].shape[0]
    plan = lw_model.plan_engram_slice(
        np.arange(slice_rows - 1), np.zeros((0,), np.int64), num_table_rows=num_table_rows, row_multiple=mesh.devices.size
    )
    embed = parts.weights["engram"]["embed"]
    embed.set_value(jax.ShapeDtypeStruct((plan.num_slice_rows, embed.shape[1]), embed.dtype, sharding=embed.sharding))
  rest = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=lw_programs.replicated(mesh)), parts.rest)
  return parts.graphdef, parts.weights, rest, plan


def abstract_io(config: Any, mesh: Mesh, layer_id: int, batch: int, seq: int) -> dict[str, Any]:
  """Abstract carry / positions / tokens / image mask / decoder_segment_ids for a program whose first layer is `layer_id`."""
  rep = lw_programs.replicated(mesh)
  return {
      "carry": lw_programs.abstract_carry(config, layer_id, batch, seq, rep),
      "positions": jax.ShapeDtypeStruct((batch, seq), jnp.int32, sharding=rep),
      "tokens": jax.ShapeDtypeStruct((batch, seq), jnp.int32, sharding=rep),
      "image_mask": jax.ShapeDtypeStruct((batch, seq), jnp.bool_, sharding=rep),
      "decoder_segment_ids": jax.ShapeDtypeStruct((batch, seq), jnp.int32, sharding=rep),
  }


def abstract_replay(config: Any, mesh: Mesh, layer_id: int, batch: int, seq: int) -> dict[str, Any]:
  """Abstract replay dict of layer `layer_id` ("indexer_mask" exactly for index sources)."""
  rep = lw_programs.replicated(mesh)
  spec = deepseek41.build_layer_specs(config)[layer_id]
  out = {"moe_topk": jax.ShapeDtypeStruct((batch, seq, config.num_experts_per_tok), jnp.int32, sharding=rep)}
  if spec.is_index_source:
    out["indexer_mask"] = jax.ShapeDtypeStruct((batch, seq, seq // spec.compress_ratio), jnp.bool_, sharding=rep)
  return out


class Suite:
  """Compiles programs for one topology and accumulates their records."""

  def __init__(self, args: argparse.Namespace):
    self.args = args
    self.records: list[dict[str, Any]] = []
    self.code_sha256 = lw_checks.harness_code_sha256()
    self.model_sha256 = lw_checks.model_code_sha256()

  def mesh_for(self, config: Any) -> Mesh:
    train_compile.validate_config(config)
    mesh = train_compile.get_topology_mesh(config)
    if mesh.devices.size != self.args.num_devices:
      raise ValueError(
          f"Topology {self.args.compile_topology} has {mesh.devices.size} devices, expected {self.args.num_devices}"
      )
    return mesh

  def add(  # pylint: disable=too-many-positional-arguments
      self,
      tier: str,
      name: str,
      fn: Callable[..., Any],
      abstract_args: Sequence[Any],
      meta: Mapping[str, Any],
      config: Any,
  ) -> None:
    """Compiles one program and records it with the run's geometry and the config's activation quant."""
    result = compile_program(name, fn, abstract_args, self.args.device_hbm_gib)
    self.record(tier, {**meta, "activation_quant": bool(config.ds41_activation_quant)}, result)

  def record(self, tier: str, meta: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    """Appends one program result with the run's geometry (and prints it, so a crash keeps the log)."""
    geometry = {
        "sample": self.args.sample,
        "seq_len": self.args.seq_len,
        "batch_size": self.args.batch_size,
        "num_images": self.args.num_images,
        "vision_grid": list(self.args.vision_grid),
    }
    self.records.append(
        {
            "tier": tier,
            **geometry,
            **meta,
            "harness_code_sha256": self.code_sha256,
            "model_code_sha256": self.model_sha256,
            **result,
        }
    )
    print(json.dumps(self.records[-1], default=str), flush=True)

  def group(self, tier: str, layer_ids: Sequence[int], *, sliced: bool, kind: str, capture: bool) -> None:
    """L2 group forward (`kind`="forward") or single-layer L4 VJP (`kind`="vjp")."""
    a = self.args
    config = topology_config(a, tier, expert_parallelism=1, extra={"scan_layers": False, "remat_policy": "none"})
    mesh = self.mesh_for(config)
    specs = deepseek41.build_layer_specs(config)
    bound = engram_slice_bound(config, a.batch_size, a.seq_len, a.engram_control_rows)
    layers = [abstract_layer(config, mesh, i, bound if (sliced and specs[i].has_engram) else None) for i in layer_ids]
    program = lw_programs.GroupProgram(
        config=config,
        mesh=mesh,
        layer_ids=tuple(layer_ids),
        graphdefs=tuple(l[0] for l in layers),
        engram_slices=tuple(l[3] for l in layers),
    )
    io = abstract_io(config, mesh, layer_ids[0], a.batch_size, a.seq_len)
    replays = [abstract_replay(config, mesh, i, a.batch_size, a.seq_len) for i in layer_ids]
    meta = {"layers": list(layer_ids), "engram_sliced": sliced, "kind": kind}
    weights, rest = [l[1] for l in layers], [l[2] for l in layers]
    if kind == "forward":
      fn = lw_programs.group_forward_entry(program, capture=capture)
      args = (
          weights,
          rest,
          io["carry"],
          io["positions"],
          io["tokens"],
          io["image_mask"],
          replays,
          io["decoder_segment_ids"],
      )
      self.add(tier, lw_programs.group_program_name(layer_ids), fn, args, meta, config)
      return
    rep = lw_programs.replicated(mesh)
    cotangents = {
        "h": jax.ShapeDtypeStruct(io["carry"]["h"].shape, jnp.float32, sharding=rep),
        "pre_mix": jax.ShapeDtypeStruct(io["carry"]["pre_mix"].shape, jnp.float32, sharding=rep),
    }
    if specs[layer_ids[0]].is_index_source:
      cotangents["indexer_scores"] = jax.ShapeDtypeStruct(replays[0]["indexer_mask"].shape, jnp.float32, sharding=rep)
    fn = lw_programs.layer_vjp_entry(program)
    args = (
        weights[0],
        rest[0],
        io["carry"],
        io["positions"],
        io["tokens"],
        io["image_mask"],
        replays[0],
        cotangents,
        io["decoder_segment_ids"],
    )
    self.add(tier, lw_programs.layer_vjp_program_name(layer_ids[0]), fn, args, meta, config)

  def l1(self, tier: str, *, vjp: bool) -> None:
    """The L1 forward (gate L1) or the L1-path VJP (gate L4 on L1)."""
    a = self.args
    config = topology_config(
        a,
        tier,
        expert_parallelism=1,
        extra={"scan_layers": False, "remat_policy": "none", "freeze_vision_encoder_params": not vjp},
    )
    mesh = self.mesh_for(config)
    parts = lw_model.abstract_parts(lw_programs.l1_build_fn(config, mesh), config, mesh)
    rep = lw_programs.replicated(mesh)
    rest = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=rep), parts.rest)
    n_h, n_w = config.ds41_vision_grid
    patch_dim = config.num_channels_for_vit * config.patch_size_for_vit**2
    tokens = jax.ShapeDtypeStruct((a.batch_size, a.seq_len), jnp.int32, sharding=rep)
    patches = jax.ShapeDtypeStruct((a.batch_size, a.num_images, n_h * n_w, patch_dim), jnp.float32, sharding=rep)
    if not vjp:
      # Captured like the harness runs it in both tiers (lw_l1.run_forward).
      fn = lw_programs.l1_forward_entry(parts.graphdef, config, capture=True)
      self.add(
          tier, lw_programs.L1_FORWARD_PROGRAM, fn, (parts.weights, rest, tokens, patches), {"kind": "forward"}, config
      )
      return
    cot = jax.ShapeDtypeStruct((a.batch_size, a.seq_len, config.emb_dim), jnp.float32, sharding=rep)
    fn = lw_programs.l1_vjp_entry(parts.graphdef, config)
    self.add(tier, lw_programs.L1_VJP_PROGRAM, fn, (parts.weights, rest, tokens, patches, cot), {"kind": "vjp"}, config)

  def l3_chain(self, tier: str, geometry: lw_programs.ScanGeometry, *, mode: str) -> None:
    """One L3 chain program (`mode` in `lw_l3.L3_CHAIN_MODES`): forward + grads."""
    a = self.args
    remat = mode.endswith("remat")
    lw_config.configure_runtime(lw_config.TIER_CODE_PARITY, platform="tpu")
    config = topology_config(
        a,
        tier,
        expert_parallelism=1,
        extra={
            "scan_layers": mode.startswith("scanned"),
            "remat_policy": "full" if remat else "none",
            **lw_config.L3_CHAIN_PRECISION_OVERRIDES,
        },
    )
    mesh = self.mesh_for(config)
    geometry.validate(config)
    io = abstract_io(config, mesh, geometry.layer_ids[0], a.batch_size, a.seq_len)
    cot = jax.ShapeDtypeStruct(io["carry"]["h"].shape, jnp.float32, sharding=lw_programs.replicated(mesh))
    meta = {"layers": list(geometry.layer_ids), "kind": "forward+vjp", "mode": mode}
    if mode.startswith("unscanned"):
      layers = [abstract_layer(config, mesh, i, None) for i in geometry.layer_ids]
      program = lw_programs.GroupProgram(
          config, mesh, geometry.layer_ids, tuple(l[0] for l in layers), (None,) * len(layers)
      )
      forward = functools.partial(lw_programs.chain_unscanned, program, remat=remat)
      weights, rest = [l[1] for l in layers], [l[2] for l in layers]
    else:
      prefix = [abstract_layer(config, mesh, i, None) for i in geometry.prefix]
      suffix = [abstract_layer(config, mesh, i, None) for i in geometry.suffix]
      stack_build = lw_programs.scanned_block_build_fn(config, mesh, geometry.repeats)
      stack = lw_model.abstract_parts(stack_build, config, mesh)
      rep = lw_programs.replicated(mesh)
      program = lw_programs.ScannedProgram(
          config=config,
          geometry=geometry,
          prefix=lw_programs.GroupProgram(
              config, mesh, geometry.prefix, tuple(l[0] for l in prefix), (None,) * len(prefix)
          ),
          stack_graphdef=stack.graphdef,
          suffix=lw_programs.GroupProgram(
              config, mesh, geometry.suffix, tuple(l[0] for l in suffix), (None,) * len(suffix)
          ),
      )
      forward = functools.partial(lw_programs.chain_scanned, program)
      weights = {"prefix": [l[1] for l in prefix], "stack": stack.weights, "suffix": [l[1] for l in suffix]}
      rest = {
          "prefix": [l[2] for l in prefix],
          "stack": jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=rep), stack.rest),
          "suffix": [l[2] for l in suffix],
      }
    fn = lw_programs.chain_entry(forward, config)
    self.add(
        tier,
        lw_programs.chain_program_name(mode),
        fn,
        (weights, rest, io["carry"], io["positions"], io["tokens"], io["image_mask"], cot),
        meta,
        config,
    )
    lw_config.configure_runtime(tier, platform="tpu")

  def l3_moe(self, tier: str, layer_id: int, variant: lw_l3.MoeVariant) -> None:
    """One L3 single-layer MoE forward + grads: replayed routing for the EP variants, natural for the implementations."""
    a = self.args
    extra = variant.config_overrides()
    meta = {"layers": [layer_id], "kind": "forward+vjp", **dataclasses.asdict(variant)}
    try:
      config = topology_config(a, tier, expert_parallelism=variant.expert_parallelism, extra=extra)
    except ValueError as e:
      # MaxText's config validation rejects this variant's flags: that is this program's result, recorded verbatim
      # (not compiled), so the harness refuses to run it.
      intended_aq = False if variant.activation_quant_off else a.reference_activation_quant
      self.record(
          tier,
          {**meta, "activation_quant": intended_aq},
          {"program": variant.program_name(), "compiled": False, "error": f"config: {e}"[:4000]},
      )
      return
    mesh = self.mesh_for(config)
    graphdef, weights, rest, _ = abstract_layer(config, mesh, layer_id, None)
    program = lw_programs.GroupProgram(config, mesh, (layer_id,), (graphdef,), (None,))
    io = abstract_io(config, mesh, layer_id, a.batch_size, a.seq_len)
    cot = jax.ShapeDtypeStruct(io["carry"]["h"].shape, jnp.float32, sharding=lw_programs.replicated(mesh))
    if variant.replayed_routing:
      replays = [abstract_replay(config, mesh, layer_id, a.batch_size, a.seq_len)]
      fn = lw_programs.replayed_chain_entry(program)
      args = ([weights], [rest], io["carry"], io["positions"], io["tokens"], io["image_mask"], replays, cot)
    else:
      fn = lw_programs.chain_entry(functools.partial(lw_programs.chain_unscanned, program, remat=False), config)
      args = ([weights], [rest], io["carry"], io["positions"], io["tokens"], io["image_mask"], cot)
    self.add(tier, variant.program_name(), fn, args, meta, config)


def parse_groups(text: str) -> list[tuple[int, ...]]:
  """ "0,1,2,3;14;20" -> [(0, 1, 2, 3), (14,), (20,)]."""
  return [tuple(int(x) for x in group.split(",")) for group in text.split(";")]


def parse_layer_list(text: str) -> list[int]:
  """ "2,20" -> [2, 20]; "none" -> []."""
  return [] if text == "none" else [int(x) for x in text.split(",")]


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
  """Compiles every program of both tiers for one sample geometry; returns the JSON report.

  Tier a runs the Engram tables as row slices (L2 and L4); tier b runs L2 on the full tables and L4 on slices.
  L3 (when requested) compiles the chain programs in `lw_l3.L3_CHAIN_TIER` and the MoE programs in
  `lw_l3.L3_MOE_TIER`.
  """
  suite = Suite(args)
  for tier in lw_config.PRECISION_TIERS:
    runtime = lw_config.configure_runtime(tier, platform="tpu")
    capture = tier == lw_config.TIER_PRODUCTION
    suite.l1(tier, vjp=False)
    suite.l1(tier, vjp=True)
    for group in parse_groups(args.l2_groups):
      suite.group(tier, group, sliced=True, kind="forward", capture=capture)
    for layer_id in parse_layer_list(args.l4_layers):
      suite.group(tier, (layer_id,), sliced=True, kind="vjp", capture=False)
    if args.l3 == "yes" and tier == lw_l3.L3_CHAIN_TIER:
      geometry = lw_stages.parse_geometry(args.scan_geometry)
      for mode in lw_l3.L3_CHAIN_MODES:
        suite.l3_chain(tier, geometry, mode=mode)
    if args.l3 == "yes" and tier == lw_l3.L3_MOE_TIER:
      for variant in lw_l3.moe_variants(args.num_devices).values():
        suite.l3_moe(tier, args.l3_layer, variant)
    suite.records.append({"tier": tier, "runtime": runtime})
  compiled = [r for r in suite.records if "program" in r]
  return {
      "compile_topology": args.compile_topology,
      "num_devices": args.num_devices,
      "device_hbm_gib": args.device_hbm_gib,
      "sample": args.sample,
      "seq_len": args.seq_len,
      "batch_size": args.batch_size,
      "num_images": args.num_images,
      "vision_grid": list(args.vision_grid),
      "reference_activation_quant": args.reference_activation_quant,
      "all_compiled": all(r["compiled"] for r in compiled),
      "all_fit": all(r["compiled"] and r["fits_device_hbm"] for r in compiled),
      "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
      "programs": suite.records,
  }


def flags_parser(argv: list[str]) -> argparse.Namespace:
  """Parses the CLI with argparse and marks absl flags parsed (absl app.run flags_parser contract)."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--compile-topology", required=True)
  parser.add_argument("--num-devices", type=int, required=True, help="JAX devices of the topology (v5p-8: 4).")
  parser.add_argument("--device-hbm-gib", type=float, required=True, help="HBM per device (v5p: 95.74).")
  parser.add_argument("--tokenizer-path", required=True)
  parser.add_argument("--seq-len", type=int, required=True)
  parser.add_argument("--batch-size", type=int, required=True)
  parser.add_argument("--num-images", type=int, required=True)
  parser.add_argument("--vision-grid", required=True, help='ViT patch grid "<h>,<w>" of the sample, e.g. "30,42"')
  parser.add_argument("--sample", required=True, help="Sample tag the geometry belongs to (recorded per program).")
  parser.add_argument("--reference-activation-quant", choices=("true", "false"), required=True)
  parser.add_argument("--engram-control-rows", type=int, required=True)
  parser.add_argument("--l2-groups", required=True, help='e.g. "0,1,2,3;14;20;24;39"')
  parser.add_argument("--l4-layers", required=True, help='e.g. "0,1,2,3,14,20,24,39", or "none"')
  parser.add_argument("--l3", choices=("yes", "no"), required=True, help="Compile the L3 programs for this sample.")
  parser.add_argument("--scan-geometry", required=True, help='prefix|repeats|suffix, e.g. "2|3;4|5" (L3 chain)')
  parser.add_argument("--l3-layer", type=int, required=True, help="Layer of the L3 MoE programs.")
  parser.add_argument("--output-json", required=True)
  args = parser.parse_args(argv[1:])
  args.reference_activation_quant = args.reference_activation_quant == "true"
  args.vision_grid = tuple(int(x) for x in args.vision_grid.split(","))
  flags.FLAGS.mark_as_parsed()
  return args


def main(args: argparse.Namespace) -> None:
  report = run_suite(args)
  out = pathlib.Path(args.output_json)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
  if not report["all_compiled"]:
    raise SystemExit(f"Some layerwise programs failed to compile; see {out}")


if __name__ == "__main__":
  app.run(main, flags_parser=flags_parser)
