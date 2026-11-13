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

"""Pinned inputs of the layerwise harness and the numeric checks applied to them.

Every input file is read with its sha256 recorded, every key lookup raises when the key is missing, and every
comparison returns its metrics together with the exact bound it was held to.

  * Tier a (`code_parity`) compares with `layer_parity.compare` at the tolerance of `layerwise_tolerances.json`,
    which `layer_parity.Tolerance` caps at the jax_ml_parity fp32 ceilings.
  * Tier b (`production`) compares with `bf16_gap_metrics` against bounds read from the measured noise-floor file;
    the harness holds no tier-b number of its own.
"""

import dataclasses
import hashlib
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# Below this L2 norm the cosine of two tensors is not defined; such a comparison is reported as not evaluable.
COSINE_NORM_FLOOR = 1e-7


@dataclasses.dataclass(frozen=True)
class _Tolerance:
  atol: float
  rtol: float
  frob_rel: float

  def __post_init__(self):
    if self.atol > 2.0e-5 or self.rtol > 2.0e-5 or self.frob_rel > 2.0e-5:
      raise ValueError(f"Tolerance exceeds approved fp32 ceilings (1e-05, 1e-05, 1e-05): {self}")


@dataclasses.dataclass(frozen=True)
class _Metrics:
  max_abs: float
  ref_inf: float
  frob_rel: float
  finite: bool
  mean_abs: float = 0.0
  rmse: float = 0.0
  cos: float = 0.0
  ulps: float = 0.0


class _LayerParityTorchFree:
  """Torch-free implementation of layer parity metrics and comparisons using pure NumPy."""

  Tolerance = _Tolerance
  Metrics = _Metrics
  FP64_FROB_REL_CEILING = 1e-10
  FP32_TOLERANCE = _Tolerance(1e-5, 1e-5, 1e-5)
  FP64_TOLERANCE = _Tolerance(1e-10, 1e-10, 1e-10)

  @staticmethod
  def compare(cand: Any, ref: Any, tol: Any) -> tuple[bool, _Metrics]:
    """Compares candidate with reference in float64; raises on a shape or dtype mismatch."""
    c_raw = np.asarray(cand)
    r_raw = np.asarray(ref)
    if not (np.issubdtype(c_raw.dtype, np.floating) and np.issubdtype(r_raw.dtype, np.floating)):
      raise TypeError(f"compare requires floating-point arrays, got cand {c_raw.dtype} and ref {r_raw.dtype}")
    if c_raw.shape != r_raw.shape:
      raise ValueError(f"Shape mismatch: candidate {c_raw.shape} vs reference {r_raw.shape}")
    if c_raw.size == 0:
      raise ValueError(f"compare requires non-empty arrays, got shape {c_raw.shape}")
    c = np.asarray(c_raw, dtype=np.float64)
    r = np.asarray(r_raw, dtype=np.float64)
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r))):
      return False, _Metrics(
          max_abs=float("nan"),
          ref_inf=float("nan"),
          frob_rel=float("nan"),
          finite=False,
          mean_abs=float("nan"),
          rmse=float("nan"),
          cos=float("nan"),
          ulps=float("nan"),
      )
    diff = np.abs(c - r)
    ref_inf = float(np.max(np.abs(r)))
    ref_l2 = float(np.linalg.norm(r))
    cand_l2 = float(np.linalg.norm(c))
    frob_rel = float(np.linalg.norm(diff)) / max(1.0, ref_l2)
    mean_abs = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))

    # Zero-norm safeguard for cosine similarity:
    if ref_l2 < COSINE_NORM_FLOOR or cand_l2 < COSINE_NORM_FLOOR:
      cos = 0.0
    else:
      cos = float(np.clip(np.sum(c * r) / (cand_l2 * ref_l2), -1.0, 1.0))

    # ULPs with spacing floor:
    eps = np.spacing(np.float32(r.astype(np.float32)))
    eps = np.maximum(eps, np.finfo(np.float32).tiny)
    ulps = float(np.max(diff / eps))

    metrics = _Metrics(
        max_abs=float(np.max(diff)),
        ref_inf=ref_inf,
        frob_rel=frob_rel,
        finite=True,
        mean_abs=mean_abs,
        rmse=rmse,
        cos=cos,
        ulps=ulps,
    )
    elementwise_ok = bool(np.all(diff <= tol.atol * max(1.0, ref_inf) + tol.rtol * np.abs(r)))
    return elementwise_ok and frob_rel <= tol.frob_rel, metrics


try:
  from tests.utils.parity import layer_parity
except ImportError:
  layer_parity = _LayerParityTorchFree
from tests.utils.parity import negctl

TOLERANCES_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "end_to_end"
    / "tpu"
    / "deepseek"
    / "v4.1-flash"
    / "layerwise_tolerances.json"
)
_READ_CHUNK_BYTES = 1 << 20
HARNESS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL_HASHED_PATHS: tuple[str, ...] = (
    "src/maxtext/configs/models/deepseek4.1-flash.yml",
    "src/maxtext/layers/attention_deepseek41.py",
    "src/maxtext/layers/deepseek41_quant.py",
    "src/maxtext/layers/engram.py",
    "src/maxtext/layers/engram_deepseek41.py",
    "src/maxtext/layers/moe.py",
    "src/maxtext/layers/nnx_decoders.py",
    "src/maxtext/models/deepseek41.py",
    "src/maxtext/models/deepseek41_vision.py",
)
DECODER_LAYER_FP64_STATUS = "f32-cast-limited"

# The status of every check (lead ruling, round 3). Only PASS is a pass; NOT_APPLICABLE_GEOMETRY is accepted only
# for a cell of the pinned expected-not-applicable file (lw_stages.finalize_report).
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_FAIL_FLOOR_LIMITED = "FAIL_FLOOR_LIMITED"
STATUS_ILL_CONDITIONED_REF_GAP = "ILL_CONDITIONED_REF_GAP"
STATUS_NOT_APPLICABLE_GEOMETRY = "NOT_APPLICABLE_GEOMETRY"
STATUS_NOT_RUN_ON_BACKEND = "NOT_RUN_ON_BACKEND"
CHECK_STATUSES = (
    STATUS_PASS,
    STATUS_FAIL,
    STATUS_FAIL_FLOOR_LIMITED,
    STATUS_ILL_CONDITIONED_REF_GAP,
    STATUS_NOT_APPLICABLE_GEOMETRY,
    STATUS_NOT_RUN_ON_BACKEND,
)

# Contract limits `layerwise_tolerances.json` may not loosen: L3 bitwise / <= 1e-7 (lead D2 / jax_ml_parity), and the
# lead's round-2 tier-b reference-gap conditioning (cos >= 0.99, rel_l2 <= 0.1).
L3_SCAN_MAX_ABS_CEILING = 1e-7
L3_REMAT_FORWARD_MAX_ABS_CEILING = 0.0
L3_REMAT_BACKWARD_MAX_ABS_CEILING = 1e-7
REF_GAP_MIN_COS_SIM_FLOOR = 0.99
REF_GAP_MAX_REL_L2_CEILING = 0.1


def sha256_file(path: pathlib.Path) -> str:
  """Hex sha256 of the file at `path`."""
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(_READ_CHUNK_BYTES), b""):
      digest.update(chunk)
  return digest.hexdigest()


def worst_status(statuses: Sequence[str]) -> str:
  """The most severe of `statuses` (FAIL > ILL_CONDITIONED_REF_GAP > FAIL_FLOOR_LIMITED > NOT_RUN_ON_BACKEND >
  NOT_APPLICABLE_GEOMETRY > PASS); an empty list or a status outside `CHECK_STATUSES` raises."""
  if not statuses:
    raise ValueError("worst_status of no checks")
  unknown = sorted(set(statuses) - set(CHECK_STATUSES))
  if unknown:
    raise ValueError(f"Unknown check statuses {unknown}; allowed {CHECK_STATUSES}")
  severity = (
      STATUS_FAIL,
      STATUS_ILL_CONDITIONED_REF_GAP,
      STATUS_FAIL_FLOOR_LIMITED,
      STATUS_NOT_RUN_ON_BACKEND,
      STATUS_NOT_APPLICABLE_GEOMETRY,
      STATUS_PASS,
  )
  return next(s for s in severity if s in statuses)


def control_fired(clean: Sequence[Any], perturbed: Sequence[Any]) -> bool:
  """A negative control fires when a check that PASSED on the clean run does not pass on the perturbed run (matched
  by label). A check that already failed clean cannot make a control fire."""

  def _lbl(c: Any) -> str:
    return c.label if hasattr(c, "label") else c["label"]

  def _st(c: Any) -> str:
    return c.status if hasattr(c, "status") else c["status"]

  clean_pass = {_lbl(c) for c in clean if _st(c) == STATUS_PASS}
  return any(_lbl(c) in clean_pass and _st(c) != STATUS_PASS for c in perturbed)


def harness_code_sha256() -> str:
  """sha256 over the harness package's .py files (name + content); stamps AOT records so the harness only runs
  programs compiled from its own code."""
  digest = hashlib.sha256()
  for path in sorted(HARNESS_DIR.glob("*.py")):
    digest.update(path.name.encode("utf-8"))
    digest.update(sha256_file(path).encode("utf-8"))
  return digest.hexdigest()


def model_code_sha256() -> str:
  """sha256 over the DS4.1 model and config files (`MODEL_HASHED_PATHS`); stamps AOT records alongside
  `harness_code_sha256` so an edit to `src/maxtext/` or `deepseek4.1-flash.yml` invalidates stale AOT records."""
  digest = hashlib.sha256()
  for rel in MODEL_HASHED_PATHS:
    path = REPO_ROOT / rel
    digest.update(rel.encode("utf-8"))
    digest.update(sha256_file(path).encode("utf-8"))
  return digest.hexdigest()


def require_keys(mapping: Mapping[str, Any], keys: Sequence[str], where: str) -> Mapping[str, Any]:
  """`mapping`, or a ValueError unless its key set is exactly `keys`."""
  if set(mapping) != set(keys):
    raise ValueError(f"{where} keys {sorted(mapping)} != {sorted(keys)}")
  return mapping


def require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
  """`mapping[key]`, or a KeyError naming `key`, `where` and the keys that do exist."""
  if key not in mapping:
    raise KeyError(f"Missing key {key!r} in {where}; present keys: {sorted(mapping)}")
  return mapping[key]


def require_path(mapping: Mapping[str, Any], keys: Sequence[str], where: str) -> Any:
  """Follows `keys` through nested mappings with `require`."""
  node = mapping
  for i, key in enumerate(keys):
    node = require(node, key, f"{where}:{'.'.join(keys[:i])}")
  return node


@dataclasses.dataclass(frozen=True)
class PinnedJson:
  """A JSON file and the sha256 of the bytes it was parsed from."""

  path: str
  sha256: str
  data: Mapping[str, Any]


def load_pinned_json(path: pathlib.Path, *, expected_sha256: str | None) -> PinnedJson:
  """Reads a JSON file, recording its sha256.

  Args:
    path: File to read.
    expected_sha256: The sha256 the file must have, or None when the caller only records it.

  Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If `expected_sha256` is given and does not match.
  """
  if not path.is_file():
    raise FileNotFoundError(f"Pinned JSON file {path} does not exist")
  actual = sha256_file(path)
  if expected_sha256 is not None and actual != expected_sha256:
    raise ValueError(f"{path} has sha256 {actual}, expected {expected_sha256}")
  with open(path, "r", encoding="utf-8") as f:
    return PinnedJson(path=str(path), sha256=actual, data=json.load(f))


def verify_sha256sums(bundle_dir: pathlib.Path, names: Sequence[str]) -> dict[str, str]:
  """Checks each of `names` in `bundle_dir` against `bundle_dir/SHA256SUMS.txt`.

  Returns:
    name -> verified sha256.

  Raises:
    FileNotFoundError: If SHA256SUMS.txt or a named file is missing.
    KeyError: If a name has no SHA256SUMS.txt entry.
    ValueError: If a digest does not match, or SHA256SUMS.txt has a malformed or duplicate line.
  """
  sums_path = bundle_dir / "SHA256SUMS.txt"
  if not sums_path.is_file():
    raise FileNotFoundError(f"{sums_path} does not exist")
  pinned: dict[str, str] = {}
  for line_no, line in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), start=1):
    parts = line.split()
    if len(parts) != 2:
      raise ValueError(f"{sums_path}:{line_no} is not '<sha256>  <name>': {line!r}")
    digest, name = parts[0], parts[1].lstrip("*")
    if name in pinned:
      raise ValueError(f"{sums_path}:{line_no} repeats {name!r}")
    pinned[name] = digest
  verified = {}
  for name in names:
    expected = require(pinned, name, str(sums_path))
    path = bundle_dir / name
    if not path.is_file():
      raise FileNotFoundError(f"Bundle file {path} does not exist")
    actual = sha256_file(path)
    if actual != expected:
      raise ValueError(f"{path} has sha256 {actual}, SHA256SUMS.txt pins {expected}")
    verified[name] = actual
  return verified


@dataclasses.dataclass(frozen=True)
class Tolerances:
  """The single source of the harness's own numeric bounds (`layerwise_tolerances.json`)."""

  sha256: str
  code_parity: layer_parity.Tolerance
  fp64_frob_rel: float
  decoder_layer_fp64_status: str
  scan_vs_unscan_max_abs: float
  remat_forward_max_abs: float
  remat_backward_max_abs: float
  negative_control_rel_scale: float
  # Tier b derives no bound from a measured reference bf16-vs-fp32 gap worse than these (lead ruling, round 2).
  ref_gap_min_cos_sim: float
  ref_gap_max_rel_l2: float
  step1_r4_parity: dict[str, Any] | None = None


def _bounded(value: Any, where: str, *, low: float, high: float) -> float:
  """`float(value)`, or a ValueError naming `where` unless it lies in [low, high]."""
  number = float(value)
  if not low <= number <= high:
    raise ValueError(f"{where} = {number} is outside the contract range [{low}, {high}]")
  return number


def load_tolerances(path: pathlib.Path) -> Tolerances:
  """Parses `layerwise_tolerances.json`: exact key sets at every level, and every bound within its contract limit."""
  pinned = load_pinned_json(path, expected_sha256=None)
  where = str(path)
  data = require_keys(
      pinned.data,
      (
          "schema_version",
          "code_parity",
          "fp64_frob_rel",
          "decoder_layer_fp64_status",
          "l3_invariance",
          "negative_control_rel_scale",
          "tier_b_reference_gap_conditioning",
          "step1_r4_parity",
      ),
      where,
  )
  if data["schema_version"] != 1:
    raise ValueError(f"{where} schema_version {data['schema_version']} != 1")
  decoder_fp64_status = data["decoder_layer_fp64_status"]
  if decoder_fp64_status != DECODER_LAYER_FP64_STATUS:
    raise ValueError(f"{where}:decoder_layer_fp64_status = {decoder_fp64_status!r} != {DECODER_LAYER_FP64_STATUS!r}")
  tier_a = require_keys(data["code_parity"], ("atol", "rtol", "frob_rel"), f"{where}:code_parity")
  l3 = require_keys(
      data["l3_invariance"],
      ("scan_vs_unscan_max_abs", "remat_forward_max_abs", "remat_backward_max_abs"),
      f"{where}:l3_invariance",
  )
  conditioning = require_keys(
      data["tier_b_reference_gap_conditioning"],
      ("min_cos_sim", "max_rel_l2"),
      f"{where}:tier_b_reference_gap_conditioning",
  )
  fp64_frob_rel = _bounded(
      data["fp64_frob_rel"], f"{where}:fp64_frob_rel", low=0.0, high=layer_parity.FP64_FROB_REL_CEILING
  )
  return Tolerances(
      sha256=pinned.sha256,
      code_parity=layer_parity.Tolerance(
          atol=float(require(tier_a, "atol", where)),
          rtol=float(require(tier_a, "rtol", where)),
          frob_rel=float(require(tier_a, "frob_rel", where)),
      ),
      fp64_frob_rel=fp64_frob_rel,
      decoder_layer_fp64_status=decoder_fp64_status,
      scan_vs_unscan_max_abs=_bounded(
          l3["scan_vs_unscan_max_abs"], f"{where}:scan_vs_unscan_max_abs", low=0.0, high=L3_SCAN_MAX_ABS_CEILING
      ),
      remat_forward_max_abs=_bounded(
          l3["remat_forward_max_abs"], f"{where}:remat_forward_max_abs", low=0.0, high=L3_REMAT_FORWARD_MAX_ABS_CEILING
      ),
      remat_backward_max_abs=_bounded(
          l3["remat_backward_max_abs"], f"{where}:remat_backward_max_abs", low=0.0, high=L3_REMAT_BACKWARD_MAX_ABS_CEILING
      ),
      negative_control_rel_scale=_bounded(
          data["negative_control_rel_scale"],
          f"{where}:negative_control_rel_scale",
          low=negctl.REL_NORM_BAND[0],
          high=negctl.REL_NORM_BAND[1],
      ),
      ref_gap_min_cos_sim=_bounded(
          conditioning["min_cos_sim"], f"{where}:min_cos_sim", low=REF_GAP_MIN_COS_SIM_FLOOR, high=1.0
      ),
      ref_gap_max_rel_l2=_bounded(
          conditioning["max_rel_l2"], f"{where}:max_rel_l2", low=0.0, high=REF_GAP_MAX_REL_L2_CEILING
      ),
  )


@dataclasses.dataclass(frozen=True)
class CheckResult:
  """One comparison: its metrics, the bound it was held to, and whether it passed."""

  label: str
  passed: bool
  metrics: Mapping[str, float]
  bound: Mapping[str, float]

  @property
  def status(self) -> str:
    return STATUS_PASS if self.passed else STATUS_FAIL

  def __getitem__(self, key: str) -> Any:
    return self.to_json()[key]

  def to_json(self) -> dict[str, Any]:
    return {
        "label": self.label,
        "passed": self.passed,
        "status": self.status,
        "metrics": dict(self.metrics),
        "bound": dict(self.bound),
    }


def check_code_parity(label: str, cand: Any, ref: Any, tol: layer_parity.Tolerance) -> CheckResult:
  """Tier-a comparison with the canonical `layer_parity.compare`."""
  passed, metrics = layer_parity.compare(np.asarray(cand), np.asarray(ref), tol)
  return CheckResult(
      label=label,
      passed=bool(passed),
      metrics=dataclasses.asdict(metrics) if dataclasses.is_dataclass(metrics) else dict(metrics),
      bound={"atol": tol.atol, "rtol": tol.rtol, "frob_rel": tol.frob_rel},
  )


def check_max_abs(label: str, cand: Any, ref: Any, max_abs: float) -> CheckResult:
  """`max |cand - ref| <= max_abs` (the L3 bitwise / near-bitwise invariance bound).

  rel_l2 = ||cand - ref|| / ||ref|| is reported next to it (not gated); it is None when ||ref|| is below
  `COSINE_NORM_FLOOR`, with ||ref|| recorded.
  """
  c = np.asarray(cand)
  r = np.asarray(ref)
  if c.shape != r.shape or c.dtype != r.dtype:
    raise ValueError(f"{label}: candidate {c.shape} {c.dtype} vs reference {r.shape} {r.dtype}")
  if c.size == 0:
    raise ValueError(f"{label}: empty arrays")
  diff = c.astype(np.float64) - r.astype(np.float64)
  finite = bool(np.all(np.isfinite(diff)))
  worst = float(np.max(np.abs(diff))) if finite else float("nan")
  ref_norm = float(np.linalg.norm(r.astype(np.float64)))
  rel_l2 = float(np.linalg.norm(diff)) / ref_norm if ref_norm >= COSINE_NORM_FLOOR else None
  return CheckResult(
      label=label,
      passed=finite and worst <= max_abs,
      metrics={"max_abs": worst, "rel_l2": rel_l2, "ref_l2_norm": ref_norm},
      bound={"max_abs": max_abs},
  )


def bf16_gap_metrics(cand: Any, ref: Any) -> dict[str, float]:
  """max_abs_diff, rel_l2 and cos_sim in float64, as the measured noise-floor file defines them.

  Raises:
    ValueError: On a shape mismatch, an empty input, or a reference / candidate whose norm is below
      `COSINE_NORM_FLOOR` (the cosine is then undefined, so the comparison cannot be evaluated).
  """
  c = np.asarray(cand, np.float64).reshape(-1)
  r = np.asarray(ref, np.float64).reshape(-1)
  if c.shape != r.shape or c.size == 0:
    raise ValueError(f"bf16_gap_metrics: candidate {c.shape} vs reference {r.shape}")
  ref_norm = float(np.linalg.norm(r))
  cand_norm = float(np.linalg.norm(c))
  if ref_norm < COSINE_NORM_FLOOR or cand_norm < COSINE_NORM_FLOOR:
    raise ValueError(f"bf16_gap_metrics: norm below {COSINE_NORM_FLOOR} (ref {ref_norm}, cand {cand_norm})")
  diff = c - r
  return {
      "max_abs_diff": float(np.max(np.abs(diff))),
      "rel_l2": float(np.linalg.norm(diff)) / ref_norm,
      "cos_sim": float(np.clip(np.dot(c, r) / (cand_norm * ref_norm), -1.0, 1.0)),
  }


def check_production(label: str, cand: Any, ref: Any, bound: Mapping[str, Any]) -> CheckResult:
  """Tier-b comparison against a measured bound with keys max_abs_diff, rel_l2 and min_cos_sim."""
  limits = {k: float(require(bound, k, f"tier-b bound of {label}")) for k in ("max_abs_diff", "rel_l2", "min_cos_sim")}
  cand_np = np.asarray(cand)
  if not np.all(np.isfinite(cand_np.astype(np.float64))):
    return CheckResult(label=label, passed=False, metrics={"finite": 0.0}, bound=limits)
  metrics = bf16_gap_metrics(cand_np, ref)
  passed = (
      metrics["max_abs_diff"] <= limits["max_abs_diff"]
      and metrics["rel_l2"] <= limits["rel_l2"]
      and metrics["cos_sim"] >= limits["min_cos_sim"]
  )
  return CheckResult(label=label, passed=passed, metrics=metrics, bound=limits)


def effective_tensor_gap(tensor_gap: float | None, ref_to_prod_gap: float | None, multiplier: float = 3.0) -> float:
  """Uniform tier-b gap rule: max(float(tensor_gap or 0.0), multiplier * float(ref_to_prod_gap or 0.0))."""
  return max(float(tensor_gap or 0.0), multiplier * float(ref_to_prod_gap or 0.0))
