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

"""Pinned reference bundles: `bundle_index.json` (pinned by sha256) and the .npz files it indexes.

Every bundle file is checked against the sha256 the index records before it is read, and every key read is checked
against the index's shape / dtype; a missing file or key raises. The adapters turn bundle keys into the gates'
reference structures in MaxText layouts (shared state as `compress_kv` [B, C, 512], `index_k` [B, C, 128], top-k
masks [B, S, C]), with the key names the reference capture documents (bundle_index.json
`shared_state_reference_citations`).
"""

import dataclasses
import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from maxtext.models import deepseek41
from tests.utils.deepseek_v41_layerwise import lw_checks, lw_config, lw_l1, lw_l2, lw_l3, lw_l4, lw_programs, lw_routing

DATA_PINS_PATH = lw_checks.TOLERANCES_PATH.parent / "layerwise_data_pins.json"
FP64_SUFFIX = "_fp64"
BF16_SUFFIX = "_bf16"
# The reference variant each tier is compared to (`ReferenceTensor.target`); its activation-quant flag sets the tier.
TARGET_VARIANT = {lw_config.TIER_CODE_PARITY: "fp64", lw_config.TIER_PRODUCTION: "fp32"}


def load_data_pins(path: pathlib.Path = DATA_PINS_PATH) -> lw_checks.PinnedJson:
  """Reads `layerwise_data_pins.json` (kept outside `HARNESS_DIR` so Pass-2 bundle refreshes do not change
  `harness_code_sha256`)."""
  pinned = lw_checks.load_pinned_json(path, expected_sha256=None)
  where = str(path)
  data = lw_checks.require_keys(pinned.data, ("schema_version", "bundle_index_sha256", "bundle_key_schema"), where)
  if data["schema_version"] != 1:
    raise ValueError(f"{where} schema_version {data['schema_version']} != 1")
  return pinned


@dataclasses.dataclass(frozen=True)
class SampleGeometry:
  """Geometry of one pinned input, taken from bundle shapes."""

  sample: str
  seq_len: int
  batch_size: int
  num_images: int
  vision_grid: tuple[int, int]


class BundleFile:
  """One indexed .npz; keys are validated against the index on every read."""

  def __init__(self, path: pathlib.Path, entry: Mapping[str, Any]):
    actual = lw_checks.sha256_file(path)
    expected = lw_checks.require(entry, "sha256", str(path))
    if actual != expected:
      raise ValueError(f"{path} has sha256 {actual}, bundle_index.json pins {expected}")
    self.path = path
    self.entry = entry
    self.keys = lw_checks.require(entry, "keys", str(path))
    self._npz = np.load(path, allow_pickle=False)

  def get(self, key: str) -> np.ndarray:
    meta = lw_checks.require(self.keys, key, str(self.path))
    value = self._npz[key]
    if list(value.shape) != list(meta["shape"]) or str(value.dtype) != meta["dtype"]:
      raise ValueError(f"{self.path}:{key} is {value.shape} {value.dtype}, index says {meta['shape']} {meta['dtype']}")
    return value

  def tensor(self, key: str, tier: str) -> lw_l2.ReferenceTensor:
    """The variants of a gated tensor a tier needs: tier a the fp64 target and the fp32 run (floor diagnostic);
    tier b the fp32 target and the bf16 run (per-op localization gap). Every needed variant is required."""
    if tier == lw_config.TIER_CODE_PARITY:
      return lw_l2.ReferenceTensor(fp64=self.get(key + FP64_SUFFIX), fp32=self.get(key), bf16=None)
    return lw_l2.ReferenceTensor(fp64=None, fp32=self.get(key), bf16=self.get(key + BF16_SUFFIX))


@dataclasses.dataclass
class BundleIndex:
  """The pinned index; `open` verifies each bundle file once per process (`opened` caches the verified files)."""

  root: pathlib.Path
  sha256: str
  data: Mapping[str, Any]
  opened: dict[str, BundleFile] = dataclasses.field(default_factory=dict)

  def _resolve_path(self, relative_path: str, expected_sha: str, bundles: Mapping[str, Any]) -> str:
    """Resolves `relative_path` or a deterministic sibling in `bundles` with the identical `sha256`."""
    if (self.root / relative_path).is_file():
      return relative_path
    for alt_rel in sorted(bundles):
      alt_entry = bundles[alt_rel]
      if (
          isinstance(alt_entry, Mapping)
          and "sha256" in alt_entry
          and str(alt_entry["sha256"]) == expected_sha
          and (self.root / alt_rel).is_file()
      ):
        return alt_rel
    return relative_path

  def _verify_sha256sums_pin(self, relative_path: str, expected_sha256: str) -> None:
    """Verifies that `SHA256SUMS.txt` exists and pins `relative_path` to `expected_sha256`."""
    sums_path = self.root / "SHA256SUMS.txt"
    if not sums_path.is_file():
      raise FileNotFoundError(f"Missing {sums_path}")
    pinned: dict[str, str] = {}
    for line_no, line in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), start=1):
      parts = line.split()
      if len(parts) != 2:
        raise ValueError(f"{sums_path}:{line_no} is not '<sha256>  <name>': {line!r}")
      digest, name = parts[0], parts[1].lstrip("*")
      if name in pinned:
        raise ValueError(f"{sums_path}:{line_no} repeats {name!r}")
      pinned[name] = digest
    sums_digest = lw_checks.require(pinned, relative_path, str(sums_path))
    if sums_digest != expected_sha256:
      raise ValueError(f"{sums_path} pins {relative_path} to {sums_digest}, bundle_index.json pins {expected_sha256}")

  def open(self, relative_path: str) -> BundleFile:
    """Opens and verifies `relative_path` (resolving an identical-SHA256 sibling in `bundles` when needed)."""
    if relative_path in self.opened:
      return self.opened[relative_path]
    bundles = lw_checks.require(self.data, "bundles", "bundle_index.json")
    orig_entry = lw_checks.require(bundles, relative_path, "bundle_index.json bundles")
    expected_sha = str(lw_checks.require(orig_entry, "sha256", relative_path))
    resolved = self._resolve_path(relative_path, expected_sha, bundles)
    self._verify_sha256sums_pin(relative_path, expected_sha)
    self._verify_sha256sums_pin(resolved, expected_sha)
    if resolved not in self.opened:
      self.opened[resolved] = BundleFile(self.root / resolved, orig_entry)
    self.opened[relative_path] = self.opened[resolved]
    return self.opened[relative_path]

  def reference_activation_quant(self, sample: str, variant: str) -> bool:
    """Whether the reference `variant` ("fp64" / "fp32") of `sample` ran the activation quant round trips."""
    if variant not in ("fp64", "fp32"):
      raise ValueError(f"Unsupported reference_activation_quant variant {variant!r}; expected 'fp64' or 'fp32'")
    bundles = lw_checks.require(self.data, "bundles", "bundle_index.json")
    target_file = self.open(l1_path(sample)) if l1_path(sample) in bundles else self.open(l2_path(sample, 0))
    flag = target_file.get(f"reference_activation_quant_{variant}")
    if flag.shape != () or flag.dtype != np.bool_:
      raise ValueError(
          f"{target_file.path}:reference_activation_quant_{variant} must be 0-D bool, got {flag.shape} {flag.dtype}"
      )
    return bool(flag.item())

  def sample(self, sample: str) -> SampleGeometry:
    """Returns the SampleGeometry for the named multimodal sample."""
    bundles = lw_checks.require(self.data, "bundles", "bundle_index.json")
    if l1_path(sample) in bundles:
      l1 = self.open(l1_path(sample))
      tokens = l1.get("tokens")
      pixels = l1.get("pixel_values")
      grid = tuple(int(x) for x in l1.get("image_grid_hw"))
      if grid[0] * grid[1] != pixels.shape[2]:
        raise ValueError(f"{l1.path}: grid {grid} does not give {pixels.shape[2]} patches")
      return SampleGeometry(sample, int(tokens.shape[1]), int(tokens.shape[0]), int(pixels.shape[1]), grid)
    l2_0 = self.open(l2_path(sample, 0))
    tokens = l2_0.get("tokens")
    return SampleGeometry(sample, int(tokens.shape[1]), int(tokens.shape[0]), 0, (0, 0))


def load_index(bundle_dir: pathlib.Path, *, pins_path: pathlib.Path = DATA_PINS_PATH) -> BundleIndex:
  pins = load_data_pins(pins_path)
  expected = str(lw_checks.require(pins.data, "bundle_index_sha256", str(pins_path)))
  pinned = lw_checks.load_pinned_json(bundle_dir / "bundle_index.json", expected_sha256=expected)
  return BundleIndex(root=bundle_dir, sha256=pinned.sha256, data=pinned.data)


def l1_path(sample: str) -> str:
  return f"ref_bundles/{sample}/l1_vision_aligner.npz"


def l2_path(sample: str, layer_id: int) -> str:
  return f"ref_bundles/{sample}/l2_layer_{layer_id:02d}.npz"


def l4_path(sample: str, layer_id: int) -> str:
  return f"ref_bundles/{sample}/l4_layer_{layer_id:02d}_vjp.npz"


@dataclasses.dataclass
class NoiseFloors:
  """The measured noise-floor file: tier-b bounds and the routing-agreement measurement."""

  sha256: str
  data: Mapping[str, Any]

  def code_parity_bounds(self, stage_key: str) -> Mapping[str, Mapping[str, float]]:
    return lw_checks.require_path(
        self.data, ("recommended_tolerances", stage_key, "code_parity"), "measured_noise_floors.json"
    )

  def production_bounds(self, stage_key: str) -> Mapping[str, Mapping[str, float]]:
    return lw_checks.require_path(
        self.data, ("recommended_tolerances", stage_key, "production"), "measured_noise_floors.json"
    )

  def _agreement(self) -> Mapping[str, Any]:
    return lw_checks.require_path(
        self.data, ("recommended_tolerances", "L2_decoder_groups", "router_agreement"), "measured_noise_floors.json"
    )

  def ulp_multiplier(self) -> float:
    return float(lw_checks.require(self._agreement(), "ulp_multiplier", "router_agreement"))

  def allowed_moe_failures(self) -> int:
    return int(lw_checks.require(self._agreement(), "max_high_margin_moe_failures", "router_agreement"))

  def allowed_indexer_failures(self) -> int:
    return int(lw_checks.require(self._agreement(), "max_high_margin_indexer_failures", "router_agreement"))

  def bf16_gap_multiplier(self) -> float:
    return float(
        lw_checks.require_path(
            self.data, ("recommended_tolerances", "safety_multiplier_bf16"), "measured_noise_floors.json"
        )
    )


def load_noise_floors(bundle_dir: pathlib.Path) -> NoiseFloors:
  """measured_noise_floors.json beside the bundles, verified here against the bundle dir's SHA256SUMS.txt; its
  sha256 is recorded in the report. A missing file, a missing SHA256SUMS.txt entry or a mismatch raises."""
  name = "measured_noise_floors.json"
  digest = lw_checks.verify_sha256sums(bundle_dir, [name])[name]
  pinned = lw_checks.load_pinned_json(bundle_dir / name, expected_sha256=digest)
  return NoiseFloors(sha256=pinned.sha256, data=pinned.data)


def sample_inputs(bundle: BundleFile) -> lw_l2.SampleInputs:
  decoder_segment_ids = bundle.get("decoder_segment_ids") if "decoder_segment_ids" in bundle.keys else None
  return lw_l2.SampleInputs(
      positions=bundle.get("positions"),
      tokens=bundle.get("tokens"),
      decoder_segment_ids=decoder_segment_ids,
      image_mask=bundle.get("image_token_mask"),
      text_pos_mask=bundle.get("text_pos_mask"),
      image_pos_mask=bundle.get("image_pos_mask"),
  )


def carry_in(bundle: BundleFile, config: Any, layer_id: int, *, h_key: str, pre_mix_key: str) -> dict[str, np.ndarray]:
  """The teacher-forced carry of `layer_id`: h, pre_mix and exactly the shared state the layer reads (keys per
  bundle_index.json `shared_state_reference_citations`)."""
  spec = deepseek41.build_layer_specs(config)[layer_id]
  seq = int(bundle.get("tokens").shape[1])
  p = f"layer_{layer_id}_"
  carry = {"h": bundle.get(h_key), "pre_mix": bundle.get(pre_mix_key)}
  for key in lw_programs.layer_reads(config, layer_id):
    if key == "compress_kv":
      carry[key] = bundle.get(f"{p}shared_compress_kv_in")
    elif key == "index_k":
      carry[key] = bundle.get(f"{p}shared_index_k_in")
    elif key == "topk_mask":
      carry[key] = lw_routing.topk_ids_to_mask(bundle.get(f"{p}topk_indices_in"), seq // spec.compress_ratio)
    elif key == "candidates":
      carry[key] = bundle.get(f"{p}candidates_mask_in")
    else:
      raise ValueError(f"Layer {layer_id} reads shared state {key!r} the bundle adapter has no key for")
  return carry


def _replay_from(bundle: BundleFile, config: Any, layer_id: int, *, prefix: str) -> dict[str, np.ndarray]:
  """The reference routing of `layer_id` as a replay dict ("indexer_mask" exactly for index sources)."""
  spec = deepseek41.build_layer_specs(config)[layer_id]
  replay = {"moe_topk": bundle.get(f"{prefix}ref_topk_experts")}
  if spec.is_index_source:
    seq = int(bundle.get("tokens").shape[1])
    replay["indexer_mask"] = lw_routing.topk_ids_to_mask(
        bundle.get(f"{prefix}ref_indexer_topk"), seq // spec.compress_ratio
    )
  return replay


def layer_replay(index: BundleIndex, sample: str, layer_id: int, config: Any) -> dict[str, np.ndarray]:
  """The reference routing of one layer (L3 EP comparison), from ref_bundles/<sample>/l2_layer_<id>.npz."""
  return _replay_from(index.open(l2_path(sample, layer_id)), config, layer_id, prefix=f"layer_{layer_id}_")


def engram_rows(index: BundleIndex, relative_path: str, key_prefix: str) -> tuple[np.ndarray, np.ndarray]:
  """(active, inactive) Engram table row ids the reference recorded in one bundle (inactive rows are controls)."""
  bundle = index.open(relative_path)
  return bundle.get(f"{key_prefix}engram_active_rows"), bundle.get(f"{key_prefix}engram_inactive_rows")


def l2_reference(  # pylint: disable=too-many-positional-arguments
    index: BundleIndex, sample: str, layer_id: int, config: Any, tier: str, ulp_multiplier: float
) -> tuple[lw_l2.ReferenceLayer, lw_l2.SampleInputs]:
  """The L2 reference of one layer from ref_bundles/<sample>/l2_layer_<id>.npz."""
  bundle = index.open(l2_path(sample, layer_id))
  p = f"layer_{layer_id}_"
  spec = deepseek41.build_layer_specs(config)[layer_id]
  replay = _replay_from(bundle, config, layer_id, prefix=p)
  moe_eligible = lw_routing.eligible_by_margin(
      bundle.get(f"{p}ref_gate_margins"),
      bundle.get(f"{p}ref_gate_boundary_ulp"),
      ulp_multiplier,
      tier=tier,
  )
  indexer_eligible = None
  if spec.is_index_source:
    indexer_eligible = lw_routing.eligible_by_margin(
        bundle.get(f"{p}ref_indexer_margins"),
        bundle.get(f"{p}ref_indexer_boundary_ulp"),
        ulp_multiplier,
        tier=tier,
        kind="indexer",
    )
  written = [k for k in lw_programs.layer_writes(config, layer_id) if k in lw_l2.GATED_SHARED_OUTPUTS]
  # Localization is a tier-b diagnostic, so only tier b reads the intermediates; a missing one is a localization
  # FAIL recorded by lw_l2.localize (never skipped).
  intermediates = {}
  if tier == lw_config.TIER_PRODUCTION:
    for _, _, name in lw_l2.localization_points(spec):
      key = lw_l2.intermediate_key(layer_id, name)
      if key in bundle.keys:
        intermediates[name] = bundle.tensor(key, tier)
  ref = lw_l2.ReferenceLayer(
      layer_id=layer_id,
      carry_in=carry_in(bundle, config, layer_id, h_key=f"{p}h_in", pre_mix_key=f"{p}pre_mix_in"),
      h_out=bundle.tensor(f"{p}h_out", tier),
      pre_mix_out=bundle.tensor(f"{p}pre_mix_out", tier),
      shared_out={k: bundle.tensor(f"{p}{k}_out", tier) for k in written},
      moe_topk=replay["moe_topk"],
      moe_eligible=moe_eligible,
      indexer_mask=replay["indexer_mask"] if spec.is_index_source else None,
      indexer_eligible=indexer_eligible,
      intermediates=intermediates,
  )
  return ref, sample_inputs(bundle)


def chain_inputs(
    index: BundleIndex, sample: str, first_layer: int, config: Any, cotangent_seed: int
) -> lw_l3.ChainInputs:
  """L3 chain inputs: the reference carry entering `first_layer` and a seeded output cotangent."""
  bundle = index.open(l2_path(sample, first_layer))
  p = f"layer_{first_layer}_"
  carry = carry_in(bundle, config, first_layer, h_key=f"{p}h_in", pre_mix_key=f"{p}pre_mix_in")
  cot = np.random.default_rng(cotangent_seed).standard_normal(carry["h"].shape).astype(np.float32)
  inputs = sample_inputs(bundle)
  return lw_l3.ChainInputs(
      carry=carry, positions=inputs.positions, tokens=inputs.tokens, image_mask=inputs.image_mask, cotangent_h=cot
  )


_EXPERT_HF = {"wi_0": "w1", "wi_1": "w3", "wo": "w2"}
_SKETCH_KEYS = {"wi_0": ("S_left", "S_right"), "wi_1": ("S_left", "S_right"), "wo": ("S_left_w2", "S_right_w2")}
_NOT_DENSE = ("dW_w1_", "dW_w2_", "dW_w3_", "dW_engram_active", "dW_engram_inactive")


def l4_reference(
    index: BundleIndex, sample: str, layer_id: int, config: Any, tier: str
) -> tuple[lw_l4.ReferenceVJP, lw_l2.SampleInputs]:
  """The L4 reference of one layer from ref_bundles/<sample>/l4_layer_<id>_vjp.npz; gradient names are the HF
  names relative to the layer (Block.named_parameters()), prefixed here with "layers.<id>."."""
  bundle = index.open(l4_path(sample, layer_id))
  spec = deepseek41.build_layer_specs(config)[layer_id]
  replay = _replay_from(bundle, config, layer_id, prefix="")
  cotangents = {"h": bundle.get("cotangent_h_out"), "pre_mix": bundle.get("cotangent_pre_mix_out")}
  if spec.is_index_source:
    cotangents["indexer_scores"] = bundle.get("cotangent_indexer_scores")
  dense = {
      f"layers.{layer_id}.{key[len('dW_'):]}": bundle.tensor(key, tier)
      for key in bundle.keys
      if key.startswith("dW_") and not key.startswith(_NOT_DENSE) and not key.endswith((FP64_SUFFIX, BF16_SUFFIX))
  }
  samples = {}
  sample_ids = bundle.get("expert_sample_ids")
  for name, w in _EXPERT_HF.items():
    stacked = bundle.tensor(f"dW_{w}_sample", tier)
    for i, eid in enumerate(sample_ids.tolist()):
      samples[(int(eid), name)] = lw_l2.ReferenceTensor(
          fp64=None if stacked.fp64 is None else stacked.fp64[i],
          fp32=stacked.fp32[i],
          bf16=None if stacked.bf16 is None else stacked.bf16[i],
      )
  ref = lw_l4.ReferenceVJP(
      layer_id=layer_id,
      carry_in=carry_in(bundle, config, layer_id, h_key="h_in", pre_mix_key="pre_mix_in"),
      replay=replay,
      cotangents=cotangents,
      d_h_in=bundle.tensor("d_hidden_in", tier),
      d_pre_mix_in=bundle.tensor("d_pre_mix_in", tier),
      dense=dense,
      expert_frob={n: bundle.tensor(f"dW_{w}_frob", tier) for n, w in _EXPERT_HF.items()},
      expert_max_abs={n: bundle.tensor(f"dW_{w}_max_abs", tier) for n, w in _EXPERT_HF.items()},
      expert_sketch={n: bundle.tensor(f"dW_{w}_sketch", tier) for n, w in _EXPERT_HF.items()},
      sketch_left={n: bundle.get(_SKETCH_KEYS[n][0]) for n in _EXPERT_HF},
      sketch_right={n: bundle.get(_SKETCH_KEYS[n][1]) for n in _EXPERT_HF},
      expert_samples=samples,
      engram_active_rows=bundle.get("engram_active_rows") if spec.has_engram else None,
      engram_active_grads=bundle.tensor("dW_engram_active", tier) if spec.has_engram else None,
  )
  image = bundle.get("image_token_mask")
  decoder_segment_ids = bundle.get("decoder_segment_ids") if "decoder_segment_ids" in bundle.keys else None
  inputs = lw_l2.SampleInputs(
      positions=bundle.get("positions"),
      tokens=bundle.get("tokens"),
      decoder_segment_ids=decoder_segment_ids,
      image_mask=image,
      text_pos_mask=~image,
      image_pos_mask=image,
  )
  return ref, inputs


def l1_reference(index: BundleIndex, sample: str, tier: str) -> lw_l1.ReferenceL1:
  """The L1 forward reference from ref_bundles/<sample>/l1_vision_aligner.npz."""
  bundle = index.open(l1_path(sample))
  blocks = bundle.tensor("vit_block_out", tier)
  return lw_l1.ReferenceL1(
      tokens=bundle.get("tokens"),
      patches=bundle.get("pixel_values"),
      image_mask=bundle.get("image_token_mask"),
      patch_embed=bundle.tensor("patch_embed_out", tier),
      vit_blocks=tuple(
          lw_l2.ReferenceTensor(
              fp64=None if blocks.fp64 is None else blocks.fp64[i],
              fp32=blocks.fp32[i],
              bf16=None if blocks.bf16 is None else blocks.bf16[i],
          )
          for i in range(blocks.fp32.shape[0])
      ),
      vit_norm=bundle.tensor("vit_norm_out", tier),
      aligner=bundle.tensor("aligner_out", tier),
      merged=bundle.tensor("post_merge_embeddings", tier),
      h0=bundle.tensor("h0", tier),
      pre_mix0=bundle.tensor("pre_mix0", tier),
  )


def l1_vjp_path(sample: str) -> str:
  return f"ref_bundles/L4_vjp_vision_aligner_{sample}.npz"


def l1_vjp_reference(index: BundleIndex, sample: str, tier: str) -> lw_l1.ReferenceL1VJP:
  """The L4 reference of the L1 path: every vision / aligner / delimiter gradient by HF name, d_images, d_embed."""
  bundle = index.open(l1_vjp_path(sample))
  leaves = {
      key[len("dW_") :]: bundle.tensor(key, tier)
      for key in bundle.keys
      if key.startswith("dW_") and not key.endswith((FP64_SUFFIX, BF16_SUFFIX))
  }
  forward = index.open(l1_path(sample))
  return lw_l1.ReferenceL1VJP(
      tokens=bundle.get("tokens"),
      patches=forward.get("pixel_values"),
      image_mask=forward.get("image_token_mask"),
      cotangent_merged=bundle.get("cotangent_span"),
      d_patches=bundle.tensor("d_images", tier),
      d_text_embeddings=bundle.tensor("d_embed_tokens", tier),
      leaves=leaves,
  )
