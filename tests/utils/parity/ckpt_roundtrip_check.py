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

"""Bidirectional mini-checkpoint round-trip checks (`HF -> MaxText -> HF`).

Verifies that converting a sliced real-weight Hugging Face mini-checkpoint into
MaxText format (`to_maxtext`, across `scan_layers=False` and `scan_layers=True`)
and exporting it back (`to_huggingface` with `--override_model_architecture`)
preserves the full parameter keyset, shapes, exact dtypes on sensitive leaves,
and numerical tensor values, paired with a negative control.
"""

from collections.abc import Collection, Mapping
import dataclasses
from typing import Any

import numpy as np

from tests.utils.parity import negctl


@dataclasses.dataclass(frozen=True)
class RoundTripTol:
  """Tolerance for comparing exported Hugging Face tensors against source tensors.

  Use `RoundTripTol.exact(...)` (`atol=0.0, rtol=0.0, frob_rel=0.0`) for lossless
  same-dtype mini-checkpoints, or pass explicit thresholds from the approved
  design tolerance table when comparing lossy quantized source weights against
  dequantized exports.
  """

  atol: float
  rtol: float
  frob_rel: float
  require_exact_dtypes: tuple[str, ...] = ("float32", "int32", "int64")

  @classmethod
  def exact(
      cls,
      require_exact_dtypes: tuple[str, ...] = ("float32", "int32", "int64"),
  ) -> "RoundTripTol":
    """Returns a bitwise-exact round-trip tolerance; listed dtypes must also match exactly."""
    return cls(
        atol=0.0,
        rtol=0.0,
        frob_rel=0.0,
        require_exact_dtypes=require_exact_dtypes,
    )

  @property
  def is_exact(self) -> bool:
    return self.atol == 0.0 and self.rtol == 0.0 and self.frob_rel == 0.0


def _is_integer_or_bool(arr: np.ndarray) -> bool:
  return bool(np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_))


def _validate_dtype_maps(
    source_tensors: Mapping[str, np.ndarray],
    exported_tensors: Mapping[str, np.ndarray],
    source_dtypes: Mapping[str, str] | None,
    exported_dtypes: Mapping[str, str] | None,
) -> None:
  """Ensures optional explicit dtype maps are provided together and match keys."""
  if (source_dtypes is None) != (exported_dtypes is None):
    raise ValueError(
        "Pass both `source_dtypes` and `exported_dtypes` or neither; got"
        f" source_dtypes={source_dtypes is not None},"
        f" exported_dtypes={exported_dtypes is not None}"
    )
  if source_dtypes is not None and set(source_dtypes) != set(source_tensors):
    raise KeyError(
        "source_dtypes keys must match source_tensors keys:"
        f" missing={sorted(set(source_tensors) - set(source_dtypes))},"
        f" extra={sorted(set(source_dtypes) - set(source_tensors))}"
    )
  if exported_dtypes is not None and set(exported_dtypes) != set(exported_tensors):
    raise KeyError(
        "exported_dtypes keys must match exported_tensors keys:"
        f" missing={sorted(set(exported_tensors) - set(exported_dtypes))},"
        f" extra={sorted(set(exported_dtypes) - set(exported_tensors))}"
    )


def check_hf_roundtrip(
    source_tensors: Mapping[str, np.ndarray],
    exported_tensors: Mapping[str, np.ndarray],
    *,
    tol: RoundTripTol,
    unexported_source_keys: Collection[str] = (),
    source_dtypes: Mapping[str, str] | None = None,
    exported_dtypes: Mapping[str, str] | None = None,
) -> tuple[bool, dict[str, dict[str, Any]], list[str]]:
  """Compares exported Hugging Face tensors against the source HF mini-checkpoint.

  Args:
    source_tensors: Original Hugging Face tensor dict before `HF -> MaxText`, keyed by
      HF parameter name.
    exported_tensors: Reverse-exported Hugging Face tensors from `to_huggingface`
      keyed by HF parameter name.
    tol: Explicit `RoundTripTol` (`RoundTripTol.exact()` for lossless exports).
    unexported_source_keys: Source keys intentionally omitted on export (for
      example quantization scale tensors consumed during dequantization). Every
      listed key must exist in `source_tensors`.
    source_dtypes: Optional framework dtype names (for example `"bfloat16"`) keyed by
      source tensor name for every key in `source_tensors`.
    exported_dtypes: Optional framework dtype names keyed by exported tensor name for
      every key in `exported_tensors`.

  Returns:
    `(passed, metrics, failures)` where `metrics[key]` summarizes per-tensor
    differences and `failures` lists every keyset, shape, dtype, finiteness, or
    numerical discrepancy.
  """
  if not source_tensors:
    raise ValueError("source_tensors is empty; pass every tensor in the mini-checkpoint")
  _validate_dtype_maps(source_tensors, exported_tensors, source_dtypes, exported_dtypes)

  unknown_skipped = sorted(set(unexported_source_keys) - set(source_tensors))
  if unknown_skipped:
    raise KeyError(f"unexported_source_keys contains keys absent from source_tensors: {unknown_skipped}")

  expected_keys = set(source_tensors) - set(unexported_source_keys)
  actual_keys = set(exported_tensors)
  failures: list[str] = []
  missing_keys = sorted(expected_keys - actual_keys)
  extra_keys = sorted(actual_keys - expected_keys)
  if missing_keys:
    failures.append(f"missing exported keys: {missing_keys}")
  if extra_keys:
    failures.append(f"unexpected exported keys: {extra_keys}")

  metrics: dict[str, dict[str, Any]] = {}
  for key in sorted(expected_keys & actual_keys):
    src = np.asarray(source_tensors[key])
    exp = np.asarray(exported_tensors[key])
    if src.size == 0 or exp.size == 0:
      raise ValueError(f"Tensor {key!r} has zero elements: source={src.shape}, exported={exp.shape}")
    src_dtype = source_dtypes[key] if source_dtypes is not None else src.dtype.name
    exp_dtype = exported_dtypes[key] if exported_dtypes is not None else exp.dtype.name

    if src.shape != exp.shape:
      metrics[key] = {
          "source_dtype": src_dtype,
          "exported_dtype": exp_dtype,
          "shape_match": False,
      }
      failures.append(f"{key}: shape mismatch exported {exp.shape} vs source {src.shape}")
      continue

    if tol.is_exact or src_dtype in tol.require_exact_dtypes:
      if exp_dtype != src_dtype:
        failures.append(f"{key}: dtype mismatch exported {exp_dtype!r} vs source {src_dtype!r}")

    if _is_integer_or_bool(src) or _is_integer_or_bool(exp):
      if _is_integer_or_bool(src) != _is_integer_or_bool(exp):
        metrics[key] = {
            "source_dtype": src_dtype,
            "exported_dtype": exp_dtype,
            "exact_equal": False,
        }
        failures.append(f"{key}: integer/float category mismatch exported {exp_dtype!r} vs source {src_dtype!r}")
        continue
      exact_ok = bool(np.array_equal(src, exp))
      metrics[key] = {
          "exact_equal": exact_ok,
          "source_dtype": src_dtype,
          "exported_dtype": exp_dtype,
      }
      if not exact_ok:
        failures.append(f"{key}: integer/bool tensor mismatch")
      continue

    s64 = np.asarray(src, dtype=np.float64)
    e64 = np.asarray(exp, dtype=np.float64)
    if not (np.all(np.isfinite(s64)) and np.all(np.isfinite(e64))):
      metrics[key] = {
          "max_abs": float("nan"),
          "frob_rel": float("nan"),
          "source_dtype": src_dtype,
          "exported_dtype": exp_dtype,
          "finite": False,
      }
      failures.append(f"{key}: non-finite values in source or exported tensor")
      continue

    diff = np.abs(e64 - s64)
    src_inf = float(np.max(np.abs(s64)))
    src_l2 = float(np.linalg.norm(s64))
    max_abs = float(np.max(diff))
    frob_rel = float(np.linalg.norm(diff)) / max(1.0, src_l2)
    metrics[key] = {
        "max_abs": max_abs,
        "frob_rel": frob_rel,
        "source_dtype": src_dtype,
        "exported_dtype": exp_dtype,
        "finite": True,
    }
    elementwise_ok = bool(np.all(diff <= tol.atol * max(1.0, src_inf) + tol.rtol * np.abs(s64)))
    if not (elementwise_ok and frob_rel <= tol.frob_rel):
      failures.append(
          f"{key}: max_abs={max_abs:.3e}, frob_rel={frob_rel:.3e} exceeds"
          f" atol={tol.atol}, rtol={tol.rtol}, frob_rel={tol.frob_rel}"
      )

  return not failures, metrics, failures


def perturb_one_tensor(
    tensors: Mapping[str, np.ndarray],
    *,
    target_key: str,
    rel_scale: float,
    seed: int,
) -> dict[str, np.ndarray]:
  """Returns a shallow copy of `tensors` with `target_key` moved by `negctl.scaled_perturbation`."""
  if target_key not in tensors:
    raise KeyError(f"target_key {target_key!r} absent from tensors")
  out = {k: np.asarray(v) for k, v in tensors.items()}
  out[target_key] = negctl.scaled_perturbation(out[target_key], name=target_key, rel_scale=rel_scale, seed=seed)
  return out


def negative_roundtrip_status(
    source_tensors: Mapping[str, np.ndarray],
    primary_exported: Mapping[str, np.ndarray],
    perturbed_exported: Mapping[str, np.ndarray],
    *,
    tol: RoundTripTol,
    unexported_source_keys: Collection[str] = (),
    source_dtypes: Mapping[str, str] | None = None,
    exported_dtypes: Mapping[str, str] | None = None,
) -> tuple[str, list[str]]:
  """Classifies a round-trip negative control as `FIRED`, `MISSED`, or `ERROR`.

  Perturb one parameter leaf in the MaxText checkpoint before running
  `to_huggingface` (or perturb one exported tensor in unit tests of the gate)
  and pass the resulting dict as `perturbed_exported`. Only `FIRED` is a pass.
  """
  primary_ok, _, primary_failures = check_hf_roundtrip(
      source_tensors,
      primary_exported,
      tol=tol,
      unexported_source_keys=unexported_source_keys,
      source_dtypes=source_dtypes,
      exported_dtypes=exported_dtypes,
  )
  if not primary_ok:
    return "ERROR", [f"unperturbed round-trip already fails: {primary_failures}"]
  if set(perturbed_exported) != set(primary_exported):
    return "ERROR", [
        "perturbed export changed keyset instead of tensor values:"
        f" missing={sorted(set(primary_exported) - set(perturbed_exported))},"
        f" extra={sorted(set(perturbed_exported) - set(primary_exported))}"
    ]

  non_finite = [
      f"perturbed tensor {k!r} is not finite"
      for k, v in perturbed_exported.items()
      if not _is_integer_or_bool(np.asarray(v)) and not np.all(np.isfinite(np.asarray(v, dtype=np.float64)))
  ]
  if all(np.array_equal(np.asarray(perturbed_exported[k]), np.asarray(primary_exported[k])) for k in primary_exported):
    non_finite.append(
        "perturbed exported tensors equal primary exported tensors: the"
        " perturbed MaxText leaf did not reach `to_huggingface` output"
    )
  if non_finite:
    return "ERROR", non_finite

  perturbed_ok, _, _ = check_hf_roundtrip(
      source_tensors,
      perturbed_exported,
      tol=tol,
      unexported_source_keys=unexported_source_keys,
      source_dtypes=source_dtypes,
      exported_dtypes=exported_dtypes,
  )
  if perturbed_ok:
    return "MISSED", ["perturbed exported tensors still pass round-trip check"]
  return "FIRED", []
