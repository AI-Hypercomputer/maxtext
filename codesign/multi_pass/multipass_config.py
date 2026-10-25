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

"""Configuration schema, dtype parser, and presets for Multi-Pass Emulation.

Supports dtype-agnostic multi-pass residual decomposition where any source tensor
of dtype `A` (`src_dtype`) is decomposed into `K` sequential passes of arbitrary
target dtypes `[B, C, D, ...]` (`pass_dtypes`):

  r_0 = cast(x, src_dtype)
  p_k = quantize(r_k / s_k, pass_dtypes[k]) * s_k,   r_{k+1} = r_k - p_k

and matrix multiplications (`dot_general`, `einsum`, `gmm`, `ragged_dot`) are
evaluated over configurable cross-term pass pairs `(i, j)` (`triangular` / `X3`,
`full` / `X4`, `asymmetric_lhs` / `X2`, `asymmetric_rhs` / `X2`, `karatsuba` / `X3`,
or explicit `custom` pass pairs) accumulated in `acc_dtype`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Custom Pass Slicer Registry
# ---------------------------------------------------------------------------

PassSlicerFn = Callable[..., list[tuple[Any, Any]]]

_CUSTOM_SLICER_REGISTRY: dict[str, PassSlicerFn] = {}


def register_pass_slicer(
    name: str, fn: PassSlicerFn | None = None
) -> Any:
  """Registers a custom multi-pass tensor decomposition function by name."""

  def _register(func: PassSlicerFn) -> PassSlicerFn:
    _CUSTOM_SLICER_REGISTRY[name.strip().lower()] = func
    return func

  if fn is not None:
    return _register(fn)
  return _register


def get_registered_pass_slicer(name: str) -> PassSlicerFn | None:
  """Looks up a custom pass slicer function by name."""
  return _CUSTOM_SLICER_REGISTRY.get(name.strip().lower())


# ---------------------------------------------------------------------------
# Module Path Target Presets
# ---------------------------------------------------------------------------

TARGET_PRESETS: dict[str, dict[str, Any]] = {
    "all_linears": {
        "module_path": ".*",
        "exclude_path": (
            r"(^|/)(token_embedder|embed|logits_dense|norm|scale)(/|$)"
        ),
        "op_names": ("dot_general", "gmm", "ragged_dot"),
    },
    "all_linear": {
        "module_path": ".*",
        "exclude_path": (
            r"(^|/)(token_embedder|embed|logits_dense|norm|scale)(/|$)"
        ),
        "op_names": ("dot_general", "gmm", "ragged_dot"),
    },
    "mlp": {
        "module_path": r"(^|/)(mlp|wi|wi_0|wi_1|wo|gate_proj|up_proj|down_proj|DenseGeneral)(/|$)",
        "exclude_path": (
            r"(^|/)(self_attention|attention|query|key|value|q_proj|k_proj|v_proj|"
            r"out|o_proj|token_embedder|logits_dense)(/|$)"
        ),
        "op_names": ("dot_general",),
    },
    "attn": {
        "module_path": (
            r"(^|/)(self_attention|attention|query|key|value|out|"
            r"q_proj|k_proj|v_proj|o_proj|qkv_proj)(/|$)"
        ),
        "exclude_path": r"(^|/)(qk_product|wv_product)(/|$)",
        "op_names": ("dot_general",),
    },
    "qkv": {
        "module_path": r"(^|/)(query|key|value|q_proj|k_proj|v_proj|qkv_proj)(/|$)",
        "exclude_path": None,
        "op_names": ("dot_general",),
    },
    "attn_out": {
        "module_path": r"(^|/)(self_attention/out|attention/out|o_proj|out_proj)(/|$)",
        "exclude_path": None,
        "op_names": ("dot_general",),
    },
    "moe": {
        "module_path": r"(^|/)(MoeBlock|routed_experts|shared_experts|moe)(/|$)",
        "exclude_path": r"(^|/)(gate)(/|$)",
        "op_names": ("dot_general", "gmm", "ragged_dot"),
    },
    "moe_router": {
        "module_path": r"(^|/)(MoeBlock_0/gate|routed_experts/gate|gate)(/|$)",
        "exclude_path": None,
        "op_names": ("dot_general",),
    },
    "all_including_attn_scores": {
        "module_path": ".*",
        "exclude_path": r"(^|/)(token_embedder|embed|logits_dense|norm|scale)(/|$)",
        "op_names": ("dot_general", "einsum", "gmm", "ragged_dot"),
    },
}


# ---------------------------------------------------------------------------
# Dtype Specification (`DtypeSpec`)
# ---------------------------------------------------------------------------

_STANDARD_FLOAT_SPECS: dict[str, tuple[int, int, bool, float | None]] = {
    # name -> (exp_bits, mantissa_bits, has_inf, max_normal_override)
    "fp32": (8, 23, True, None),
    "float32": (8, 23, True, None),
    "f32": (8, 23, True, None),
    "tf32": (8, 10, True, None),
    "bf16": (8, 7, True, None),
    "bfloat16": (8, 7, True, None),
    "fp16": (5, 10, True, 65504.0),
    "float16": (5, 10, True, 65504.0),
    "f16": (5, 10, True, 65504.0),
    "fp8_e4m3": (4, 3, False, 448.0),
    "float8_e4m3fn": (4, 3, False, 448.0),
    "f8e4m3": (4, 3, False, 448.0),
    "e4m3": (4, 3, False, 448.0),
    "fp8_e5m2": (5, 2, True, 57344.0),
    "float8_e5m2": (5, 2, True, 57344.0),
    "f8e5m2": (5, 2, True, 57344.0),
    "e5m2": (5, 2, True, 57344.0),
    "fp6_e3m2": (3, 2, False, 28.0),
    "f6e3m2": (3, 2, False, 28.0),
    "e3m2": (3, 2, False, 28.0),
    "fp6_e2m3": (2, 3, False, 7.5),
    "f6e2m3": (2, 3, False, 7.5),
    "e2m3": (2, 3, False, 7.5),
    "fp4_e2m1": (2, 1, False, 6.0),
    "f4e2m1": (2, 1, False, 6.0),
    "mxfp4": (2, 1, False, 6.0),
    "e2m1": (2, 1, False, 6.0),
    "fp4_e3m0": (3, 0, False, 16.0),
    "f4e3m0": (3, 0, False, 16.0),
    "e3m0": (3, 0, False, 16.0),
}

_EXMY_REGEX = re.compile(r"^f?(?:p\d+_)?e(\d+)m(\d+)(?:fn)?$", re.IGNORECASE)
_INT_REGEX = re.compile(r"^(u?)int(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class DtypeSpec:
  r"""Represents any standard or parametric floating-point / integer data type.

  Attributes:
    name: Canonical dtype name (e.g. `"bf16"`, `"fp8_e4m3"`, `"e5m3"`, `"int4"`).
    kind: `"float"`, `"int"`, `"uint"`, or `"auto"`.
    exp_bits: Number of exponent bits (for `"float"` kind).
    mantissa_bits: Number of explicit significand/mantissa bits (for `"float"` kind).
    int_bits: Total bitwidth including sign bit (for `"int"` / `"uint"` kind).
    signed: Whether the representation is signed.
    has_inf: Whether the max exponent is reserved for Inf/NaN (`True` for IEEE
      `fp32`/`bf16`/`fp16`/`e5m2`, `False` for finite-only `e4m3fn`/`e2m1`).
    max_val: Maximum representable finite positive value ($Q_{\max}$).
    min_subnormal: Smallest positive representable subnormal value.
  """

  name: str = "bf16"
  kind: str = "float"
  exp_bits: int = 8
  mantissa_bits: int = 7
  int_bits: int = 0
  signed: bool = True
  has_inf: bool = True
  max_val: float = 3.38953139e38
  min_subnormal: float = 2**-133

  @property
  def min_val(self) -> float:
    if self.kind == "uint":
      return 0.0
    if self.kind == "int":
      if self.int_bits <= 1:
        return -1.0
      return -float(1 << (self.int_bits - 1))
    return -self.max_val if self.signed else 0.0

  @property
  def total_bits(self) -> int:
    if self.kind in ("int", "uint"):
      return self.int_bits
    return (1 if self.signed else 0) + self.exp_bits + self.mantissa_bits

  @property
  def exp_bias(self) -> int:
    if self.exp_bits <= 0:
      return 0
    return (1 << (self.exp_bits - 1)) - 1

  @property
  def max_exponent(self) -> int:
    if self.exp_bits <= 0:
      return 0
    raw_max = (1 << self.exp_bits) - 1
    if self.has_inf and self.exp_bits >= 4:
      return (raw_max - 1) - self.exp_bias
    return raw_max - self.exp_bias

  @property
  def min_normal_exponent(self) -> int:
    if self.exp_bits <= 0:
      return 0
    return 1 - self.exp_bias

  @classmethod
  def parse(cls, spec: Any) -> DtypeSpec:
    """Parses a string, `DtypeSpec`, or dict into a normalized `DtypeSpec`.

    Supports:
      - Standard names: `"fp32"`, `"bf16"`, `"fp16"`, `"tf32"`, `"fp8_e4m3"`,
        `"fp8_e5m2"`, `"fp6_e3m2"`, `"fp6_e2m3"`, `"fp4_e2m1"`, `"fp4_e3m0"`,
        `"int8"`, `"int4"`, `"int2"`, `"int16"`, `"int32"`, `"uint8"`, `"uint4"`.
      - Arbitrary parametric float: `"eXmY"` (e.g., `"e5m3"`, `"e4m2"`, `"e6m1"`,
        `"e8m4"`).
      - Arbitrary parametric integer: `"intB"` or `"uintB"` (e.g., `"int6"`,
        `"int3"`, `"uint5"`).
      - `"auto"` / `"none"` / `"identity"`: Pass-through without dtype cast.
    """
    if isinstance(spec, DtypeSpec):
      return spec
    if isinstance(spec, Mapping):
      if "name" in spec and len(spec) == 1:
        return cls.parse(spec["name"])
      kind = str(spec.get("kind", "float")).lower()
      if kind in ("int", "uint"):
        bits = int(spec.get("int_bits", spec.get("bits", 8)))
        signed = kind == "int" and bool(spec.get("signed", True))
        return cls.from_int_bits(bits, signed=signed)
      exp_b = int(spec.get("exp_bits", 8))
      man_b = int(spec.get("mantissa_bits", 7))
      has_inf = bool(spec.get("has_inf", exp_b >= 5))
      return cls.from_exmy(exp_b, man_b, has_inf=has_inf, name=spec.get("name"))

    raw = str(spec).strip().lower()
    if not raw or raw in ("auto", "none", "identity", "native"):
      return cls(
          name="auto",
          kind="auto",
          exp_bits=8,
          mantissa_bits=23,
          max_val=3.4028235e38,
      )

    if raw in _STANDARD_FLOAT_SPECS:
      e_bits, m_bits, has_inf, max_override = _STANDARD_FLOAT_SPECS[raw]
      # Canonicalize aliases
      canon = raw
      if raw in ("float32", "f32"):
        canon = "fp32"
      elif raw == "bfloat16":
        canon = "bf16"
      elif raw in ("float16", "f16"):
        canon = "fp16"
      elif raw in ("float8_e4m3fn", "f8e4m3", "e4m3"):
        canon = "fp8_e4m3"
      elif raw in ("float8_e5m2", "f8e5m2", "e5m2"):
        canon = "fp8_e5m2"
      elif raw in ("f6e3m2", "e3m2"):
        canon = "fp6_e3m2"
      elif raw in ("f6e2m3", "e2m3"):
        canon = "fp6_e2m3"
      elif raw in ("f4e2m1", "mxfp4", "e2m1"):
        canon = "fp4_e2m1"
      elif raw in ("f4e3m0", "e3m0"):
        canon = "fp4_e3m0"
      return cls.from_exmy(
          e_bits, m_bits, has_inf=has_inf, max_override=max_override, name=canon
      )

    m_int = _INT_REGEX.match(raw)
    if m_int:
      unsigned_prefix, bit_str = m_int.groups()
      bits = int(bit_str)
      return cls.from_int_bits(bits, signed=(unsigned_prefix != "u"))

    m_exmy = _EXMY_REGEX.match(raw)
    if m_exmy:
      e_bits = int(m_exmy.group(1))
      m_bits = int(m_exmy.group(2))
      has_inf = not raw.endswith("fn") and e_bits >= 5
      return cls.from_exmy(e_bits, m_bits, has_inf=has_inf, name=f"e{e_bits}m{m_bits}")

    raise ValueError(
        f"Unrecognized dtype specification {spec!r}. Expected standard dtype "
        "('bf16', 'fp32', 'fp16', 'fp8_e4m3', 'fp8_e5m2', 'fp4_e2m1', 'int8', "
        "'int4', ...), custom float 'eXmY' (e.g. 'e5m3'), or custom int 'intB'."
    )

  @classmethod
  def from_exmy(
      cls,
      exp_bits: int,
      mantissa_bits: int,
      *,
      has_inf: bool = False,
      max_override: float | None = None,
      name: str | None = None,
  ) -> DtypeSpec:
    """Constructs a floating-point `DtypeSpec` with `exp_bits` and `mantissa_bits`."""
    if exp_bits < 1 or mantissa_bits < 0:
      raise ValueError(
          f"Invalid eXmY format: exp_bits={exp_bits}, mantissa_bits={mantissa_bits}"
      )
    bias = (1 << (exp_bits - 1)) - 1
    raw_max_exp = (1 << exp_bits) - 1
    if has_inf and exp_bits >= 4:
      max_exp = (raw_max_exp - 1) - bias
      max_sig = 2.0 - (2.0 ** (-mantissa_bits))
    elif exp_bits == 4 and mantissa_bits == 3 and not has_inf:
      # OFP8 E4M3FN reserves only S.1111.111 for NaN -> max normal is 448.0
      max_exp = raw_max_exp - bias
      max_sig = 1.75
    else:
      max_exp = raw_max_exp - bias
      max_sig = 2.0 - (2.0 ** (-mantissa_bits))

    computed_max = float(max_sig * (2.0**max_exp))
    max_val = float(max_override) if max_override is not None else computed_max
    min_normal_exp = 1 - bias
    min_sub = float(2.0 ** (min_normal_exp - mantissa_bits))
    canon_name = name or f"e{exp_bits}m{mantissa_bits}"
    return cls(
        name=canon_name,
        kind="float",
        exp_bits=exp_bits,
        mantissa_bits=mantissa_bits,
        int_bits=0,
        signed=True,
        has_inf=has_inf,
        max_val=max_val,
        min_subnormal=min_sub,
    )

  @classmethod
  def from_int_bits(cls, bits: int, *, signed: bool = True) -> DtypeSpec:
    """Constructs a signed or unsigned integer `DtypeSpec` of arbitrary bitwidth."""
    if bits < 1 or bits > 64:
      raise ValueError(f"Integer bitwidth must be in [1, 64], got {bits}")
    if signed:
      max_v = float((1 << (bits - 1)) - 1) if bits > 1 else 1.0
      name = f"int{bits}"
      kind = "int"
    else:
      max_v = float((1 << bits) - 1)
      name = f"uint{bits}"
      kind = "uint"
    return cls(
        name=name,
        kind=kind,
        exp_bits=0,
        mantissa_bits=0,
        int_bits=bits,
        signed=signed,
        has_inf=False,
        max_val=max_v,
        min_subnormal=1.0,
    )

  def to_dict(self) -> dict[str, Any]:
    return {
        "name": self.name,
        "kind": self.kind,
        "exp_bits": self.exp_bits,
        "mantissa_bits": self.mantissa_bits,
        "int_bits": self.int_bits,
        "signed": self.signed,
        "has_inf": self.has_inf,
        "max_val": self.max_val,
    }


def parse_dtype_list(
    dtypes_spec: Any,
    num_passes: int | None = None,
    default_dtype: str = "bf16",
) -> tuple[DtypeSpec, ...]:
  """Parses a single dtype or sequence of dtypes `[B, C, D, ...]` into `K` `DtypeSpec`s.

  Rules:
    - `dtypes_spec` can be a single string (`"fp8_e4m3"`), a comma-separated
      string (`"fp8_e4m3,fp8_e4m3"` or `"[fp8_e5m2, fp4_e2m1]"`), or a list/tuple
      of strings / `DtypeSpec`s.
    - If `num_passes` is `None`, the number of passes is `len(parsed_list)`.
    - If `num_passes` is specified and `len(parsed_list) == 1`, that dtype is
      repeated `num_passes` times.
    - If `len(parsed_list) < num_passes`, the last dtype is repeated to reach
      `num_passes`.
    - If `len(parsed_list) > num_passes`, it is truncated to `num_passes` ONLY
      when `num_passes` was explicitly forced smaller; otherwise callers that
      give an explicit list of length `M > 1` with default `num_passes` should
      infer `num_passes = len(parsed_list)`.
  """
  if dtypes_spec is None or dtypes_spec == "":
    base = DtypeSpec.parse(default_dtype)
    k = max(1, num_passes if num_passes is not None else 1)
    return tuple(base for _ in range(k))

  if isinstance(dtypes_spec, DtypeSpec):
    items = [dtypes_spec]
  elif isinstance(dtypes_spec, str):
    cleaned = dtypes_spec.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
      cleaned = cleaned[1:-1].strip()
    elif cleaned.startswith("(") and cleaned.endswith(")"):
      cleaned = cleaned[1:-1].strip()
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    items = [DtypeSpec.parse(p) for p in parts] if parts else [DtypeSpec.parse(default_dtype)]
  elif isinstance(dtypes_spec, Sequence):
    items = [DtypeSpec.parse(x) for x in dtypes_spec]
    if not items:
      items = [DtypeSpec.parse(default_dtype)]
  else:
    items = [DtypeSpec.parse(dtypes_spec)]

  if num_passes is None or num_passes <= 0:
    return tuple(items)

  if len(items) == num_passes:
    return tuple(items)
  if len(items) < num_passes:
    return tuple(items + [items[-1]] * (num_passes - len(items)))
  return tuple(items[:num_passes])


# ---------------------------------------------------------------------------
# Per-Operand Multi-Pass Specification (`OperandMultipassSpec`)
# ---------------------------------------------------------------------------

VALID_SCALE_MODES = frozenset({
    "none",
    "unit",
    "per_tensor_amax",
    "per_tensor_pow2",
    "per_channel_amax",
    "per_channel_pow2",
    "per_block_amax",
    "per_block_pow2",
    "bit_shift",
})

VALID_ROUNDING_MODES = frozenset({
    "rne",
    "stochastic",
    "truncate",
    "rtz",
})


@dataclass(frozen=True)
class OperandMultipassSpec:
  r"""Configures how a single tensor operand (`A`) is sliced into `K` passes `[B, C, D, ...]`.

  Attributes:
    enabled: Whether multi-pass decomposition is active for this operand.
    src_dtype: Source dtype `A` to cast/quantize the operand to before pass
      decomposition (e.g. `"bf16"`, `"fp32"`, `"int8"`, `"auto"`).
    num_passes: Number of residual passes `K` (`len(pass_dtypes)`).
    pass_dtypes: Tuple of `K` `DtypeSpec`s `[B, C, D, ...]` for Pass `0 .. K-1`.
    scale_mode: How each residual pass `r_k` is scaled into the representable
      range of `pass_dtypes[k]`:
      - `"none"`: No scaling factor ($s_k = 1$, direct rounding to `pass_dtypes[k]`).
      - `"per_tensor_amax"`: $s_k = \max(|r_k|) / Q_{\max}(\tau_k)$ across tensor.
      - `"per_tensor_pow2"`: Power-of-two scale $2^{\lceil \log_2(\max(|r_k|)/Q_{\max}(\tau_k)) \rceil}$
        across tensor.
      - `"per_channel_amax"`: Per-channel amax along the contracting dimension.
      - `"per_channel_pow2"`: Per-channel power-of-two scale along the contracting dimension.
      - `"per_block_amax"`: Microscaling (`block_size`) amax along the contracting dimension.
      - `"per_block_pow2"`: Microscaling (`block_size`) power-of-two scale along the
        contracting dimension.
      - `"bit_shift"`: Fixed integer bit-shift decomposition (e.g., `int8` into
        two `int4` passes with shift $2^4 = 16$).
    block_size: Block size along the contracting dimension when `scale_mode` is
      `"per_block_amax"` or `"per_block_pow2"` (default `128`).
    rounding_mode: `"rne"` (round-to-nearest-even), `"stochastic"`, or `"truncate"`.
    flush_subnormals: If `True`, values smaller than the minimum normal of
      `pass_dtypes[k]` are flushed to zero.
    custom_slicer: Optional registered custom slicer name.
    seed: PRNG seed for `"stochastic"` rounding.
  """

  enabled: bool = False
  src_dtype: DtypeSpec = field(default_factory=lambda: DtypeSpec.parse("bf16"))
  num_passes: int = 2
  pass_dtypes: tuple[DtypeSpec, ...] = field(
      default_factory=lambda: (DtypeSpec.parse("fp8_e4m3"), DtypeSpec.parse("fp8_e4m3"))
  )
  scale_mode: str = "per_tensor_pow2"
  block_size: int = 128
  rounding_mode: str = "rne"
  flush_subnormals: bool = False
  custom_slicer: str | None = None
  seed: int = 0

  def __post_init__(self) -> None:
    if isinstance(self.src_dtype, str):
      object.__setattr__(self, "src_dtype", DtypeSpec.parse(self.src_dtype))
    if not isinstance(self.pass_dtypes, tuple):
      object.__setattr__(
          self,
          "pass_dtypes",
          parse_dtype_list(self.pass_dtypes, num_passes=self.num_passes),
      )
    elif len(self.pass_dtypes) != self.num_passes and self.num_passes > 0:
      object.__setattr__(
          self,
          "pass_dtypes",
          parse_dtype_list(self.pass_dtypes, num_passes=self.num_passes),
      )
    if self.enabled:
      if self.num_passes < 1:
        raise ValueError(f"num_passes must be >= 1, got {self.num_passes}")
      if self.scale_mode not in VALID_SCALE_MODES:
        raise ValueError(
            f"Unsupported scale_mode={self.scale_mode!r}. "
            f"Valid modes: {sorted(VALID_SCALE_MODES)}"
        )
      if self.rounding_mode not in VALID_ROUNDING_MODES:
        raise ValueError(
            f"Unsupported rounding_mode={self.rounding_mode!r}. "
            f"Valid modes: {sorted(VALID_ROUNDING_MODES)}"
        )
      if self.block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {self.block_size}")

  @property
  def pass_dtype_names(self) -> list[str]:
    return [d.name for d in self.pass_dtypes]

  @property
  def subnormals(self) -> bool:
    return not self.flush_subnormals

  def summary(self) -> str:
    if not self.enabled:
      return "disabled"
    dtypes_str = ",".join(self.pass_dtype_names)
    blk = f",block={self.block_size}" if "block" in self.scale_mode else ""
    return (
        f"{self.src_dtype.name}->[{dtypes_str}]"
        f"(K={self.num_passes},scale={self.scale_mode}{blk},round={self.rounding_mode})"
    )

  def to_dict(self) -> dict[str, Any]:
    return {
        "enabled": self.enabled,
        "src_dtype": self.src_dtype.name,
        "num_passes": self.num_passes,
        "pass_dtypes": [d.name for d in self.pass_dtypes],
        "scale_mode": self.scale_mode,
        "block_size": self.block_size,
        "rounding_mode": self.rounding_mode,
        "flush_subnormals": self.flush_subnormals,
        "custom_slicer": self.custom_slicer,
        "seed": self.seed,
    }

  @classmethod
  def from_any(
      cls,
      value: Any,
      *,
      default_src_dtype: str = "bf16",
      default_num_passes: int = 2,
      default_pass_dtypes: Any = "fp8_e4m3",
      default_scale_mode: str = "per_tensor_pow2",
      default_block_size: int = 128,
      default_rounding_mode: str = "rne",
      default_flush_subnormals: bool = False,
  ) -> OperandMultipassSpec:
    """Parses an `OperandMultipassSpec` from a bool, DSL string, list, or dict.

    Supported inline DSL strings:
      - `"none"` / `"false"` / `"off"` / `"0"` -> `enabled=False`
      - `"true"` / `"on"` -> `enabled=True` with defaults
      - `"bf16->[fp8_e4m3,fp8_e4m3]"`
      - `"fp32->[bf16,bf16,bf16]@scale=none:round=rne"`
      - `"bf16->[fp8_e5m2,fp4_e2m1]@scale=per_block_pow2:block=128"`
      - `"[fp8_e4m3,fp8_e4m3]"` or `"fp8_e4m3,fp8_e4m3"`
      - `"fp8_e4m3*2"` or `"bf16*3"` (shorthand for `K` homogeneous passes)
    """
    if isinstance(value, OperandMultipassSpec):
      return value
    if value is None:
      return cls(enabled=False)
    if isinstance(value, bool):
      if not value:
        return cls(enabled=False)
      p_dtypes = parse_dtype_list(default_pass_dtypes, num_passes=default_num_passes)
      return cls(
          enabled=True,
          src_dtype=DtypeSpec.parse(default_src_dtype),
          num_passes=len(p_dtypes),
          pass_dtypes=p_dtypes,
          scale_mode=default_scale_mode,
          block_size=default_block_size,
          rounding_mode=default_rounding_mode,
          flush_subnormals=default_flush_subnormals,
      )
    if isinstance(value, Mapping):
      enabled = bool(value.get("enabled", True))
      if not enabled:
        return cls(enabled=False)
      src_dt = DtypeSpec.parse(value.get("src_dtype", default_src_dtype))
      raw_p_dtypes = value.get("pass_dtypes", default_pass_dtypes)
      explicit_k = value.get("num_passes", None)
      if explicit_k is not None:
        k = int(explicit_k)
        # If pass_dtypes is an explicit multi-element sequence of a different
        # length and num_passes wasn't meant to truncate it, respect explicit_k.
        p_dtypes = parse_dtype_list(raw_p_dtypes, num_passes=k)
      else:
        parsed_raw = parse_dtype_list(raw_p_dtypes, num_passes=None)
        if len(parsed_raw) > 1:
          k = len(parsed_raw)
          p_dtypes = parsed_raw
        else:
          k = default_num_passes
          p_dtypes = parse_dtype_list(raw_p_dtypes, num_passes=k)
      return cls(
          enabled=True,
          src_dtype=src_dt,
          num_passes=len(p_dtypes),
          pass_dtypes=p_dtypes,
          scale_mode=str(value.get("scale_mode", default_scale_mode)).lower(),
          block_size=int(value.get("block_size", default_block_size)),
          rounding_mode=str(value.get("rounding_mode", default_rounding_mode)).lower(),
          flush_subnormals=bool(
              value.get("flush_subnormals", default_flush_subnormals)
          ),
          custom_slicer=value.get("custom_slicer"),
          seed=int(value.get("seed", 0)),
      )

    if isinstance(value, Sequence) and not isinstance(value, str):
      p_dtypes = parse_dtype_list(value, num_passes=None)
      return cls(
          enabled=True,
          src_dtype=DtypeSpec.parse(default_src_dtype),
          num_passes=len(p_dtypes),
          pass_dtypes=p_dtypes,
          scale_mode=default_scale_mode,
          block_size=default_block_size,
          rounding_mode=default_rounding_mode,
          flush_subnormals=default_flush_subnormals,
      )

    s = str(value).strip()
    if not s or s.lower() in ("none", "false", "off", "0", "disabled", "dense"):
      return cls(enabled=False)
    if s.lower() in ("true", "on", "1", "enabled"):
      p_dtypes = parse_dtype_list(default_pass_dtypes, num_passes=default_num_passes)
      return cls(
          enabled=True,
          src_dtype=DtypeSpec.parse(default_src_dtype),
          num_passes=len(p_dtypes),
          pass_dtypes=p_dtypes,
          scale_mode=default_scale_mode,
          block_size=default_block_size,
          rounding_mode=default_rounding_mode,
          flush_subnormals=default_flush_subnormals,
      )

    # Parse optional `@key=val:key2=val2` modifiers
    scale_mode = default_scale_mode
    block_size = default_block_size
    rounding_mode = default_rounding_mode
    flush_subnormals = default_flush_subnormals
    custom_slicer = None

    if "@" in s:
      main_part, mod_part = s.split("@", 1)
      s = main_part.strip()
      for tok in mod_part.split(":"):
        tok = tok.strip()
        if not tok:
          continue
        if "=" in tok:
          k_mod, v_mod = [p.strip() for p in tok.split("=", 1)]
          k_low = k_mod.lower()
          if k_low in ("scale", "scale_mode"):
            scale_mode = v_mod.lower()
          elif k_low in ("block", "block_size"):
            block_size = int(v_mod)
          elif k_low in ("round", "rounding", "rounding_mode"):
            rounding_mode = v_mod.lower()
          elif k_low in ("flush", "flush_subnormals"):
            flush_subnormals = v_mod.lower() in ("true", "1", "yes")
          elif k_low == "subnormals":
            flush_subnormals = v_mod.lower() in ("false", "0", "no")
          elif k_low == "slicer":
            custom_slicer = v_mod
        elif tok.lower() in VALID_SCALE_MODES:
          scale_mode = tok.lower()

    src_dtype_str = default_src_dtype
    if "->" in s:
      lhs_part, rhs_part = s.split("->", 1)
      src_dtype_str = lhs_part.strip()
      s = rhs_part.strip()

    # Check shorthand `dtype*K` (e.g., `fp8_e4m3*2`)
    if "*" in s and not s.startswith("["):
      dt_part, k_part = s.split("*", 1)
      k_val = int(k_part.strip())
      p_dtypes = parse_dtype_list(dt_part.strip(), num_passes=k_val)
    else:
      parsed_list = parse_dtype_list(s, num_passes=None)
      if len(parsed_list) > 1 or s.startswith("["):
        p_dtypes = parsed_list
      else:
        p_dtypes = parse_dtype_list(s, num_passes=default_num_passes)

    return cls(
        enabled=True,
        src_dtype=DtypeSpec.parse(src_dtype_str),
        num_passes=len(p_dtypes),
        pass_dtypes=p_dtypes,
        scale_mode=scale_mode,
        block_size=block_size,
        rounding_mode=rounding_mode,
        flush_subnormals=flush_subnormals,
        custom_slicer=custom_slicer,
    )


# ---------------------------------------------------------------------------
# Named Multi-Pass Algorithm Presets (RFC & Literature)
# ---------------------------------------------------------------------------

MULTIPASS_PRESETS: dict[str, dict[str, Any]] = {
    "F8E4M3_F8E4M3_F32_X1": {
        "description": "1-pass FP8 E4M3 x E4M3 -> FP32 matmul (baseline single-pass FP8).",
        "src_dtype": "bf16",
        "num_passes": 1,
        "pass_dtypes": ["fp8_e4m3"],
        "scale_mode": "per_tensor_pow2",
        "cross_term_mode": "full",
        "acc_dtype": "fp32",
    },
    "F8E4M3_F8E4M3_F32_X2": {
        "description": (
            "2-pass asymmetric FP8 E4M3 matmul (hh + hl): LHS is 1 pass, "
            "RHS is 2 passes of FP8 E4M3, accumulated in FP32."
        ),
        "src_dtype": "bf16",
        "lhs_num_passes": 1,
        "lhs_pass_dtypes": ["fp8_e4m3"],
        "rhs_num_passes": 2,
        "rhs_pass_dtypes": ["fp8_e4m3", "fp8_e4m3"],
        "scale_mode": "per_tensor_pow2",
        "cross_term_mode": "asymmetric_rhs",
        "acc_dtype": "fp32",
    },
    "F8E4M3_F8E4M3_F32_X3": {
        "description": (
            "3-pass symmetric triangular FP8 E4M3 matmul (hh + hl + lh, dropping ll): "
            "both operands decomposed into 2 passes of FP8 E4M3, accumulated in FP32."
        ),
        "src_dtype": "bf16",
        "num_passes": 2,
        "pass_dtypes": ["fp8_e4m3", "fp8_e4m3"],
        "scale_mode": "per_tensor_pow2",
        "cross_term_mode": "triangular",
        "max_pass_degree": 1,
        "acc_dtype": "fp32",
    },
    "F8E4M3_F8E4M3_F32_X4": {
        "description": (
            "4-pass full cross-product FP8 E4M3 matmul (hh + hl + lh + ll): "
            "both operands decomposed into 2 passes of FP8 E4M3, accumulated in FP32."
        ),
        "src_dtype": "bf16",
        "num_passes": 2,
        "pass_dtypes": ["fp8_e4m3", "fp8_e4m3"],
        "scale_mode": "per_tensor_pow2",
        "cross_term_mode": "full",
        "acc_dtype": "fp32",
    },
    "F8E5M2_F8E5M2_F32_X3": {
        "description": (
            "3-pass symmetric triangular FP8 E5M2 matmul (hh + hl + lh, dropping ll) "
            "accumulated in FP32."
        ),
        "src_dtype": "bf16",
        "num_passes": 2,
        "pass_dtypes": ["fp8_e5m2", "fp8_e5m2"],
        "scale_mode": "per_tensor_pow2",
        "cross_term_mode": "triangular",
        "max_pass_degree": 1,
        "acc_dtype": "fp32",
    },
    "BF16_BF16_F32_X2": {
        "description": (
            "2-pass asymmetric BF16 matmul emulating FP32 via (hh + hl) BF16 passes."
        ),
        "src_dtype": "fp32",
        "lhs_num_passes": 1,
        "lhs_pass_dtypes": ["bf16"],
        "rhs_num_passes": 2,
        "rhs_pass_dtypes": ["bf16", "bf16"],
        "scale_mode": "none",
        "cross_term_mode": "asymmetric_rhs",
        "acc_dtype": "fp32",
    },
    "BF16_BF16_F32_X3": {
        "description": (
            "3-pass symmetric triangular BF16 matmul (Henry et al. ARITH 2019): "
            "FP32 operands decomposed into 2 BF16 passes (hh + hl + lh), accumulated in FP32."
        ),
        "src_dtype": "fp32",
        "num_passes": 2,
        "pass_dtypes": ["bf16", "bf16"],
        "scale_mode": "none",
        "cross_term_mode": "triangular",
        "max_pass_degree": 1,
        "acc_dtype": "fp32",
    },
    "BF16_BF16_F32_X6": {
        "description": (
            "6-pass symmetric triangular BF16 matmul (Henry et al. ARITH 2019): "
            "FP32 operands decomposed into 3 BF16 passes (i + j <= 2), accumulated in FP32."
        ),
        "src_dtype": "fp32",
        "num_passes": 3,
        "pass_dtypes": ["bf16", "bf16", "bf16"],
        "scale_mode": "none",
        "cross_term_mode": "triangular",
        "max_pass_degree": 2,
        "acc_dtype": "fp32",
    },
    "INT8_INT4_INT32_X2": {
        "description": (
            "2-pass asymmetric INT8 x INT4 matmul using two INT4 passes for LHS (INT8) "
            "and one INT4 pass for RHS, accumulated in INT32."
        ),
        "src_dtype": "int8",
        "lhs_num_passes": 2,
        "lhs_pass_dtypes": ["int4", "uint4"],
        "rhs_num_passes": 1,
        "rhs_pass_dtypes": ["int4"],
        "scale_mode": "bit_shift",
        "cross_term_mode": "asymmetric_lhs",
        "acc_dtype": "int32",
    },
    "INT8_INT8_INT32_X3": {
        "description": (
            "3-pass Karatsuba INT8 x INT8 matmul using INT4 passes accumulated in INT32."
        ),
        "src_dtype": "int8",
        "num_passes": 2,
        "pass_dtypes": ["int4", "uint4"],
        "scale_mode": "bit_shift",
        "cross_term_mode": "karatsuba",
        "acc_dtype": "int32",
    },
    "INT8_INT8_INT32_X4": {
        "description": (
            "4-pass exact low-high INT8 x INT8 matmul using INT4 passes (hh, hl, lh, ll) "
            "accumulated in INT32."
        ),
        "src_dtype": "int8",
        "num_passes": 2,
        "pass_dtypes": ["int4", "uint4"],
        "scale_mode": "bit_shift",
        "cross_term_mode": "full",
        "acc_dtype": "int32",
    },
}


# ---------------------------------------------------------------------------
# Cross-Term Pass Selection Utilities
# ---------------------------------------------------------------------------

VALID_CROSS_TERM_MODES = frozenset({
    "triangular",
    "full",
    "asymmetric_lhs",
    "asymmetric_rhs",
    "karatsuba",
    "custom",
})

VALID_BWD_MODES = frozenset({
    "multipass_bwd",
    "ste",
})


def parse_active_passes(spec: Any) -> tuple[tuple[int, int], ...] | None:
  """Parses explicit `(i, j)` pass pairs from string or sequence.

  Examples:
    - `"0:0,0:1,1:0"` -> `((0, 0), (0, 1), (1, 0))`
    - `[[0, 0], [0, 1], [1, 0]]` -> `((0, 0), (0, 1), (1, 0))`
  """
  if spec is None or spec == "" or spec == "auto":
    return None
  if isinstance(spec, str):
    cleaned = spec.strip()
    if not cleaned or cleaned.lower() in ("none", "auto"):
      return None
    pairs: list[tuple[int, int]] = []
    for item in cleaned.split(","):
      item = item.strip()
      if not item:
        continue
      if ":" in item:
        a_s, b_s = item.split(":", 1)
        pairs.append((int(a_s.strip()), int(b_s.strip())))
      elif "-" in item:
        a_s, b_s = item.split("-", 1)
        pairs.append((int(a_s.strip()), int(b_s.strip())))
    return tuple(pairs) if pairs else None
  if isinstance(spec, Sequence):
    pairs = []
    for item in spec:
      if isinstance(item, Sequence) and len(item) == 2:
        pairs.append((int(item[0]), int(item[1])))
      elif isinstance(item, str) and ":" in item:
        a_s, b_s = item.split(":", 1)
        pairs.append((int(a_s.strip()), int(b_s.strip())))
    return tuple(pairs) if pairs else None
  return None


def resolve_cross_term_pairs(
    lhs_passes: int,
    rhs_passes: int,
    cross_term_mode: str = "triangular",
    max_pass_degree: int | None = None,
    active_passes: tuple[tuple[int, int], ...] | str | None = None,
    *,
    max_total_pass_order: int | None = None,
) -> tuple[tuple[int, int], ...]:
  r"""Resolves the ordered tuple of `(i, j)` pass index pairs to execute.

  Args:
    lhs_passes: Number of passes $K_A$ on LHS.
    rhs_passes: Number of passes $K_B$ on RHS.
    cross_term_mode: One of `"triangular"`, `"full"`, `"asymmetric_lhs"`,
      `"asymmetric_rhs"`, `"karatsuba"`, `"custom"`.
    max_pass_degree: Maximum pass index sum $i + j$ for `"triangular"` mode.
      Defaults to $\max(K_A - 1, K_B - 1)$ (so for $K_A=K_B=2$, $i+j \le 1$,
      yielding the 3 passes `(0,0), (0,1), (1,0)`).
    active_passes: Explicit `(i, j)` pairs when `cross_term_mode == "custom"`
      or when explicitly provided.
    max_total_pass_order: Optional alias for `max_pass_degree` that also prunes
      pairs with $i + j > \text{max\_total\_pass\_order}$ in `"full"` mode.

  Returns:
    Tuple of `(i, j)` pairs with $0 \le i < K_A$ and $0 \le j < K_B$.
  """
  k_a = max(1, int(lhs_passes))
  k_b = max(1, int(rhs_passes))
  mode = cross_term_mode.strip().lower()
  eff_max_deg = (
      max_total_pass_order
      if max_total_pass_order is not None
      else max_pass_degree
  )

  parsed_active = (
      parse_active_passes(active_passes)
      if isinstance(active_passes, str)
      else active_passes
  )
  if parsed_active:
    filtered = tuple(
        (i, j) for (i, j) in parsed_active if 0 <= i < k_a and 0 <= j < k_b
    )
    if filtered:
      return filtered

  if mode == "full":
    pairs = tuple((i, j) for i in range(k_a) for j in range(k_b))
    if eff_max_deg is not None and int(eff_max_deg) >= 0:
      deg_lim = int(eff_max_deg)
      pairs = tuple((i, j) for (i, j) in pairs if (i + j) <= deg_lim)
    return pairs if pairs else ((0, 0),)

  if mode == "asymmetric_lhs":
    return tuple((i, 0) for i in range(k_a))

  if mode == "asymmetric_rhs":
    return tuple((0, j) for j in range(k_b))

  if mode == "karatsuba":
    # Karatsuba uses 3 special products for the top 2x2 passes plus any higher passes
    # For reporting/pair listing, return the 3 primary pass identifiers
    if k_a >= 2 and k_b >= 2:
      return ((0, 0), (1, 1), (0, 1))
    return tuple((i, j) for i in range(k_a) for j in range(k_b))

  # Default: "triangular" (i + j <= max_degree)
  deg = (
      int(eff_max_deg)
      if (eff_max_deg is not None and int(eff_max_deg) >= 0)
      else max(k_a - 1, k_b - 1)
  )
  pairs = tuple(
      (i, j)
      for i in range(k_a)
      for j in range(k_b)
      if (i + j) <= deg
  )
  return pairs if pairs else ((0, 0),)


def parse_layer_indices(layers_spec: Any) -> tuple[int, ...] | None:
  """Parses layer indices from int, list, comma-separated string, or range string `'0-7,12'`."""
  if layers_spec is None or layers_spec == "":
    return None
  if isinstance(layers_spec, int):
    return None if layers_spec < 0 else (layers_spec,)
  if isinstance(layers_spec, (list, tuple, set)):
    out: list[int] = []
    for item in layers_spec:
      sub = parse_layer_indices(item)
      if sub:
        out.extend(sub)
    return tuple(sorted(set(out))) if out else None

  s = str(layers_spec).strip()
  if not s or s.lower() in ("none", "all", "*", "-1"):
    return None
  indices: set[int] = set()
  for part in s.split(","):
    part = part.strip()
    if not part:
      continue
    if "-" in part and not part.startswith("-"):
      lo_s, hi_s = part.split("-", 1)
      lo, hi = int(lo_s.strip()), int(hi_s.strip())
      indices.update(range(min(lo, hi), max(lo, hi) + 1))
    else:
      val = int(part)
      if val >= 0:
        indices.add(val)
  return tuple(sorted(indices)) if indices else None


_LAYER_INDEX_PATTERNS = (
    re.compile(r"(?:^|/)layers_(\d+)(?:/|$)"),
    re.compile(r"(?:^|/)layers/(\d+)(?:/|$)"),
    re.compile(r"(?:^|/)layer_(\d+)(?:/|$)"),
)


def extract_layer_index(path_str: str) -> int | None:
  """Extracts integer decoder layer index from a module/parameter path if present."""
  for pat in _LAYER_INDEX_PATTERNS:
    m = pat.search(path_str)
    if m:
      return int(m.group(1))
  return None


# ---------------------------------------------------------------------------
# Rule-Level Configuration (`MultipassRuleConfig`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultipassRuleConfig:
  """Granular rule mapping module/layer/op selectors to multi-pass operand & GEMM specs.

  Attributes:
    name: Human-readable rule name for logging and diagnostics.
    module_path: Regex pattern matched against the normalized module path.
    exclude_path: Optional regex pattern for paths to exclude.
    layers: Optional tuple of allowed decoder layer indices (e.g. `(0, 1, 2)`).
    exclude_layers: Optional tuple of excluded decoder layer indices.
    op_names: Tuple of intercepted op types (`"dot_general"`, `"einsum"`, `"gmm"`,
      `"ragged_dot"`).
    lhs: Forward pass LHS (activation) multi-pass decomposition spec.
    rhs: Forward pass RHS (weight) multi-pass decomposition spec.
    cross_term_mode: Forward pass cross-term selection mode (`"triangular"`,
      `"full"`, `"asymmetric_lhs"`, `"asymmetric_rhs"`, `"karatsuba"`, `"custom"`).
    max_pass_degree: Max pass index sum $i + j$ in `"triangular"` mode.
    active_passes: Explicit `(i, j)` pass pairs when `cross_term_mode == "custom"`.
    acc_dtype: Accumulation dtype for summing pass dot products (`"fp32"`, `"bf16"`,
      `"int32"`, or custom `eXmY`).
    out_dtype: Output dtype of the multi-pass dot product (`"auto"` preserves original).
    bwd_mode: Backward gradient computation mode (`"multipass_bwd"` or `"ste"`).
    emulate_fwd: Whether to apply multi-pass emulation on the forward GEMM (`fprop`).
    emulate_dlhs: Whether to apply multi-pass emulation on the activation-gradient
      GEMM (`dgrad` / `dlhs = dy @ w^T`) when `bwd_mode == "multipass_bwd"`.
    emulate_drhs: Whether to apply multi-pass emulation on the weight-gradient
      GEMM (`wgrad` / `drhs = x^T @ dy`) when `bwd_mode == "multipass_bwd"`.
    dlhs_grad: Multi-pass spec for upstream gradient `dy` in `dlhs` GEMM.
    dlhs_rhs: Multi-pass spec for weight `w` in `dlhs` GEMM.
    dlhs_cross_term_mode: Cross-term mode for `dlhs` GEMM.
    drhs_lhs: Multi-pass spec for activation `x` in `drhs` GEMM.
    drhs_grad: Multi-pass spec for upstream gradient `dy` in `drhs` GEMM.
    drhs_cross_term_mode: Cross-term mode for `drhs` GEMM.
    input_activation: Optional standalone module input activation multi-pass spec.
    output_activation: Optional standalone module output activation multi-pass spec.
  """

  name: str = "default"
  module_path: str = ".*"
  exclude_path: str | None = (
      r"(^|/)(token_embedder|embed|logits_dense|norm|scale)(/|$)"
  )
  layers: tuple[int, ...] | None = None
  exclude_layers: tuple[int, ...] | None = None
  op_names: tuple[str, ...] = ("dot_general", "gmm", "ragged_dot")

  lhs: OperandMultipassSpec | None = None
  rhs: OperandMultipassSpec | None = None
  cross_term_mode: str = "triangular"
  max_pass_degree: int | None = None
  active_passes: tuple[tuple[int, int], ...] | None = None
  acc_dtype: DtypeSpec = field(default_factory=lambda: DtypeSpec.parse("fp32"))
  out_dtype: DtypeSpec = field(default_factory=lambda: DtypeSpec.parse("auto"))

  bwd_mode: str = "multipass_bwd"
  emulate_fwd: bool = True
  emulate_dlhs: bool = False
  emulate_drhs: bool = False
  dlhs_grad: OperandMultipassSpec | None = None
  dlhs_rhs: OperandMultipassSpec | None = None
  dlhs_cross_term_mode: str | None = None
  drhs_lhs: OperandMultipassSpec | None = None
  drhs_grad: OperandMultipassSpec | None = None
  drhs_cross_term_mode: str | None = None

  input_activation: OperandMultipassSpec | None = None
  output_activation: OperandMultipassSpec | None = None

  def __post_init__(self) -> None:
    if isinstance(self.acc_dtype, str):
      object.__setattr__(self, "acc_dtype", DtypeSpec.parse(self.acc_dtype))
    if isinstance(self.out_dtype, str):
      object.__setattr__(self, "out_dtype", DtypeSpec.parse(self.out_dtype))
    if self.cross_term_mode not in VALID_CROSS_TERM_MODES:
      raise ValueError(
          f"Unsupported cross_term_mode={self.cross_term_mode!r}. "
          f"Valid modes: {sorted(VALID_CROSS_TERM_MODES)}"
      )
    if self.bwd_mode not in VALID_BWD_MODES:
      raise ValueError(
          f"Unsupported bwd_mode={self.bwd_mode!r}. "
          f"Valid modes: {sorted(VALID_BWD_MODES)}"
      )

  @property
  def has_any_multipass(self) -> bool:
    for spec in (
        self.lhs,
        self.rhs,
        self.dlhs_grad,
        self.dlhs_rhs,
        self.drhs_lhs,
        self.drhs_grad,
        self.input_activation,
        self.output_activation,
    ):
      if spec is not None and spec.enabled:
        return True
    return False

  @property
  def effective_fwd_pairs(self) -> tuple[tuple[int, int], ...]:
    k_a = self.lhs.num_passes if (self.lhs and self.lhs.enabled) else 1
    k_b = self.rhs.num_passes if (self.rhs and self.rhs.enabled) else 1
    return resolve_cross_term_pairs(
        k_a,
        k_b,
        cross_term_mode=self.cross_term_mode,
        max_pass_degree=self.max_pass_degree,
        active_passes=self.active_passes,
    )

  def matches(
      self,
      path_str: str,
      op_name: str = "dot_general",
      *,
      allow_unindexed_scan_layer: bool = False,
  ) -> bool:
    """Returns True if `path_str` and `op_name` match this rule's selectors."""
    if self.op_names and op_name not in self.op_names and "*" not in self.op_names:
      return False

    norm_path = path_str.replace(".", "/")
    if self.exclude_path and re.search(self.exclude_path, norm_path):
      return False
    if not re.search(self.module_path, norm_path):
      return False

    if self.layers is not None or self.exclude_layers is not None:
      layer_idx = extract_layer_index(norm_path)
      if layer_idx is None:
        if not allow_unindexed_scan_layer:
          return False
      else:
        if self.layers is not None and layer_idx not in self.layers:
          return False
        if self.exclude_layers is not None and layer_idx in self.exclude_layers:
          return False

    return True

  def to_dict(self) -> dict[str, Any]:
    return {
        "name": self.name,
        "module_path": self.module_path,
        "exclude_path": self.exclude_path,
        "layers": list(self.layers) if self.layers is not None else None,
        "exclude_layers": (
            list(self.exclude_layers) if self.exclude_layers is not None else None
        ),
        "op_names": list(self.op_names),
        "lhs": self.lhs.to_dict() if self.lhs else None,
        "rhs": self.rhs.to_dict() if self.rhs else None,
        "cross_term_mode": self.cross_term_mode,
        "max_pass_degree": self.max_pass_degree,
        "active_passes": (
            [list(p) for p in self.active_passes] if self.active_passes else None
        ),
        "acc_dtype": self.acc_dtype.name,
        "out_dtype": self.out_dtype.name,
        "bwd_mode": self.bwd_mode,
        "emulate_fwd": self.emulate_fwd,
        "emulate_dlhs": self.emulate_dlhs,
        "emulate_drhs": self.emulate_drhs,
        "dlhs_grad": self.dlhs_grad.to_dict() if self.dlhs_grad else None,
        "dlhs_rhs": self.dlhs_rhs.to_dict() if self.dlhs_rhs else None,
        "dlhs_cross_term_mode": self.dlhs_cross_term_mode,
        "drhs_lhs": self.drhs_lhs.to_dict() if self.drhs_lhs else None,
        "drhs_grad": self.drhs_grad.to_dict() if self.drhs_grad else None,
        "drhs_cross_term_mode": self.drhs_cross_term_mode,
        "input_activation": (
            self.input_activation.to_dict() if self.input_activation else None
        ),
        "output_activation": (
            self.output_activation.to_dict() if self.output_activation else None
        ),
    }

  @classmethod
  def from_dict(
      cls,
      data: Mapping[str, Any],
      defaults: Mapping[str, Any] | None = None,
  ) -> MultipassRuleConfig:
    """Constructs a `MultipassRuleConfig` from a dictionary and optional defaults."""
    defs = dict(defaults or {})
    preset_name = str(data.get("target_preset", defs.get("target_preset", ""))).strip()
    preset_info = TARGET_PRESETS.get(preset_name, {})

    module_path = str(
        data.get(
            "module_path",
            preset_info.get("module_path", defs.get("module_path", ".*")),
        )
    )
    exclude_path = data.get(
        "exclude_path",
        preset_info.get(
            "exclude_path",
            defs.get(
                "exclude_path",
                r"(^|/)(token_embedder|embed|logits_dense|norm|scale)(/|$)",
            ),
        ),
    )
    if exclude_path in ("", "none", "None"):
      exclude_path = None

    layers = parse_layer_indices(data.get("layers", defs.get("layers")))
    exclude_layers = parse_layer_indices(
        data.get("exclude_layers", defs.get("exclude_layers"))
    )

    op_names_raw = data.get(
        "op_names",
        preset_info.get(
            "op_names", defs.get("op_names", ("dot_general", "gmm", "ragged_dot"))
        ),
    )
    if isinstance(op_names_raw, str):
      op_names = tuple(s.strip() for s in op_names_raw.split(",") if s.strip())
    else:
      op_names = tuple(str(s).strip() for s in op_names_raw)

    # Check if rule specifies an algorithm preset
    def _non_empty(val: Any) -> Any:
      if val is None:
        return None
      if isinstance(val, str) and val.strip() == "":
        return None
      if isinstance(val, (int, float)) and not isinstance(val, bool) and val < 0:
        return None
      if isinstance(val, (list, tuple)) and len(val) == 0:
        return None
      return val

    def _pick(*candidates: Any, fallback: Any = None) -> Any:
      for c in candidates:
        v = _non_empty(c)
        if v is not None:
          return v
      return fallback

    alg_preset_name = str(
        _pick(data.get("preset"), defs.get("preset"), fallback="")
    ).strip().upper()
    alg_preset = MULTIPASS_PRESETS.get(alg_preset_name, {})

    def_src = str(
        _pick(
            data.get("src_dtype"),
            defs.get("src_dtype"),
            alg_preset.get("src_dtype"),
            fallback="bf16",
        )
    )
    def_p_dtypes = _pick(
        data.get("pass_dtypes"),
        defs.get("pass_dtypes"),
        alg_preset.get("pass_dtypes"),
        fallback="fp8_e4m3",
    )
    explicit_def_k = _pick(
        data.get("num_passes"),
        defs.get("num_passes"),
        alg_preset.get("num_passes"),
        fallback=None,
    )
    if explicit_def_k is not None:
      def_k = int(explicit_def_k)
    else:
      parsed_def_dtypes = parse_dtype_list(def_p_dtypes, num_passes=None)
      def_k = len(parsed_def_dtypes) if len(parsed_def_dtypes) > 1 else 2

    def_scale = str(
        _pick(
            data.get("scale_mode"),
            defs.get("scale_mode"),
            alg_preset.get("scale_mode"),
            fallback="per_tensor_pow2",
        )
    ).lower()
    def_block = int(
        _pick(
            data.get("block_size"),
            defs.get("block_size"),
            alg_preset.get("block_size"),
            fallback=128,
        )
    )
    def_round = str(
        _pick(
            data.get("rounding_mode"),
            defs.get("rounding_mode"),
            alg_preset.get("rounding_mode"),
            fallback="rne",
        )
    ).lower()
    def_flush = bool(
        data.get(
            "flush_subnormals",
            alg_preset.get("flush_subnormals", defs.get("flush_subnormals", False)),
        )
    )

    def _build_op_spec(
        key: str,
        *,
        fallback_enabled: bool = False,
        role_k: int | None = None,
        role_dtypes: Any = None,
        role_src: str | None = None,
        role_scale: str | None = None,
    ) -> OperandMultipassSpec | None:
      raw_val = data.get(key, defs.get(key, None))
      if raw_val is None or (isinstance(raw_val, str) and raw_val.strip() == ""):
        raw_val = fallback_enabled
      eff_dtypes = role_dtypes if role_dtypes is not None else def_p_dtypes
      if role_k is not None:
        eff_k = role_k
      else:
        parsed_role_dt = parse_dtype_list(eff_dtypes, num_passes=None)
        eff_k = len(parsed_role_dt) if len(parsed_role_dt) > 1 else def_k
      spec = OperandMultipassSpec.from_any(
          raw_val,
          default_src_dtype=role_src or def_src,
          default_num_passes=eff_k,
          default_pass_dtypes=eff_dtypes,
          default_scale_mode=role_scale or def_scale,
          default_block_size=def_block,
          default_rounding_mode=def_round,
          default_flush_subnormals=def_flush,
      )
      return spec if spec.enabled else None

    lhs_dtypes = _pick(
        data.get("lhs_pass_dtypes"),
        defs.get("lhs_pass_dtypes"),
        alg_preset.get("lhs_pass_dtypes"),
        fallback=def_p_dtypes,
    )
    lhs_k_raw = _pick(
        data.get("lhs_num_passes"),
        defs.get("lhs_num_passes"),
        alg_preset.get("lhs_num_passes"),
        fallback=None,
    )
    lhs_k = int(lhs_k_raw) if lhs_k_raw is not None else None
    lhs_src = _pick(
        data.get("lhs_src_dtype"),
        defs.get("lhs_src_dtype"),
        alg_preset.get("lhs_src_dtype"),
        fallback=def_src,
    )
    lhs_scale = _pick(
        data.get("lhs_scale_mode"),
        defs.get("lhs_scale_mode"),
        alg_preset.get("lhs_scale_mode"),
        fallback=def_scale,
    )

    rhs_dtypes = _pick(
        data.get("rhs_pass_dtypes"),
        defs.get("rhs_pass_dtypes"),
        alg_preset.get("rhs_pass_dtypes"),
        fallback=def_p_dtypes,
    )
    rhs_k_raw = _pick(
        data.get("rhs_num_passes"),
        defs.get("rhs_num_passes"),
        alg_preset.get("rhs_num_passes"),
        fallback=None,
    )
    rhs_k = int(rhs_k_raw) if rhs_k_raw is not None else None
    rhs_src = _pick(
        data.get("rhs_src_dtype"),
        defs.get("rhs_src_dtype"),
        alg_preset.get("rhs_src_dtype"),
        fallback=def_src,
    )
    rhs_scale = _pick(
        data.get("rhs_scale_mode"),
        defs.get("rhs_scale_mode"),
        alg_preset.get("rhs_scale_mode"),
        fallback=def_scale,
    )

    emulate_lhs_flag = bool(
        data.get("emulate_lhs", defs.get("emulate_lhs", True))
    )
    emulate_rhs_flag = bool(
        data.get("emulate_rhs", defs.get("emulate_rhs", True))
    )

    lhs_spec = _build_op_spec(
        "lhs",
        fallback_enabled=emulate_lhs_flag,
        role_k=lhs_k,
        role_dtypes=lhs_dtypes,
        role_src=str(lhs_src) if lhs_src else None,
        role_scale=str(lhs_scale) if lhs_scale else None,
    )
    rhs_spec = _build_op_spec(
        "rhs",
        fallback_enabled=emulate_rhs_flag,
        role_k=rhs_k,
        role_dtypes=rhs_dtypes,
        role_src=str(rhs_src) if rhs_src else None,
        role_scale=str(rhs_scale) if rhs_scale else None,
    )

    cross_term_mode = str(
        _pick(
            data.get("cross_term_mode"),
            defs.get("cross_term_mode"),
            alg_preset.get("cross_term_mode"),
            fallback="triangular",
        )
    ).lower()
    raw_deg = _pick(
        data.get("max_pass_degree"),
        defs.get("max_pass_degree"),
        alg_preset.get("max_pass_degree"),
        fallback=None,
    )
    max_pass_degree = (
        int(raw_deg)
        if (raw_deg is not None and str(raw_deg).strip() not in ("", "none", "None", "-1"))
        else None
    )
    active_passes = parse_active_passes(
        _pick(data.get("active_passes"), defs.get("active_passes"), fallback=None)
    )
    if active_passes and cross_term_mode != "custom":
      cross_term_mode = "custom"

    acc_dtype = DtypeSpec.parse(
        _pick(
            data.get("acc_dtype"),
            defs.get("acc_dtype"),
            alg_preset.get("acc_dtype"),
            fallback="fp32",
        )
    )
    out_dtype = DtypeSpec.parse(
        _pick(
            data.get("out_dtype"),
            defs.get("out_dtype"),
            alg_preset.get("out_dtype"),
            fallback="auto",
        )
    )
    bwd_mode = str(
        _pick(data.get("bwd_mode"), defs.get("bwd_mode"), fallback="multipass_bwd")
    ).lower()
    emulate_fwd = bool(data.get("emulate_fwd", defs.get("emulate_fwd", True)))
    emulate_dlhs = bool(data.get("emulate_dlhs", defs.get("emulate_dlhs", False)))
    emulate_drhs = bool(data.get("emulate_drhs", defs.get("emulate_drhs", False)))

    dlhs_grad = _build_op_spec(
        "dlhs_grad",
        fallback_enabled=emulate_dlhs,
        role_k=lhs_k,
        role_dtypes=lhs_dtypes,
        role_src=str(lhs_src) if lhs_src else None,
        role_scale=str(lhs_scale) if lhs_scale else None,
    )
    dlhs_rhs = _build_op_spec(
        "dlhs_rhs",
        fallback_enabled=emulate_dlhs,
        role_k=rhs_k,
        role_dtypes=rhs_dtypes,
        role_src=str(rhs_src) if rhs_src else None,
        role_scale=str(rhs_scale) if rhs_scale else None,
    )
    drhs_lhs = _build_op_spec(
        "drhs_lhs",
        fallback_enabled=emulate_drhs,
        role_k=lhs_k,
        role_dtypes=lhs_dtypes,
        role_src=str(lhs_src) if lhs_src else None,
        role_scale=str(lhs_scale) if lhs_scale else None,
    )
    drhs_grad = _build_op_spec(
        "drhs_grad",
        fallback_enabled=emulate_drhs,
        role_k=rhs_k,
        role_dtypes=rhs_dtypes,
        role_src=str(rhs_src) if rhs_src else None,
        role_scale=str(rhs_scale) if rhs_scale else None,
    )

    dlhs_ct_raw = _pick(
        data.get("dlhs_cross_term_mode"), defs.get("dlhs_cross_term_mode"), fallback=None
    )
    dlhs_cross_term_mode = (
        str(dlhs_ct_raw).lower()
        if dlhs_ct_raw and str(dlhs_ct_raw).lower() not in ("none", "auto")
        else None
    )
    drhs_ct_raw = _pick(
        data.get("drhs_cross_term_mode"), defs.get("drhs_cross_term_mode"), fallback=None
    )
    drhs_cross_term_mode = (
        str(drhs_ct_raw).lower()
        if drhs_ct_raw and str(drhs_ct_raw).lower() not in ("none", "auto")
        else None
    )

    in_act = _build_op_spec("input_activation", fallback_enabled=False)
    out_act = _build_op_spec("output_activation", fallback_enabled=False)

    return cls(
        name=str(data.get("name", preset_name or alg_preset_name or "default")),
        module_path=module_path,
        exclude_path=exclude_path,
        layers=layers,
        exclude_layers=exclude_layers,
        op_names=op_names,
        lhs=lhs_spec,
        rhs=rhs_spec,
        cross_term_mode=cross_term_mode,
        max_pass_degree=max_pass_degree,
        active_passes=active_passes,
        acc_dtype=acc_dtype,
        out_dtype=out_dtype,
        bwd_mode=bwd_mode,
        emulate_fwd=emulate_fwd,
        emulate_dlhs=emulate_dlhs,
        emulate_drhs=emulate_drhs,
        dlhs_grad=dlhs_grad,
        dlhs_rhs=dlhs_rhs,
        dlhs_cross_term_mode=dlhs_cross_term_mode,
        drhs_lhs=drhs_lhs,
        drhs_grad=drhs_grad,
        drhs_cross_term_mode=drhs_cross_term_mode,
        input_activation=in_act,
        output_activation=out_act,
    )


# ---------------------------------------------------------------------------
# Top-Level Multi-Pass Emulation Config (`MultipassEmulationConfig`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultipassEmulationConfig:
  """Top-level configuration for Multi-Pass Emulation across MaxText workflows.

  Attributes:
    enabled: Master switch enabling multi-pass emulation.
    rules: Ordered tuple of `MultipassRuleConfig` rules. The first matching rule
      for a `(module_path, op_name)` pair is applied.
    start_step: Training step at which multi-pass emulation activates (`0` = from start).
    end_step: Optional training step after which multi-pass emulation deactivates
      (`None` or `-1` = active until end of training).
    emulate_reference_model: Whether to apply multi-pass emulation to the Reference
      model in RL (GRPO/GSPO) and DPO workflows (default `False`).
    emulate_teacher_model: Whether to apply multi-pass emulation to the Teacher
      model in Knowledge Distillation workflows (default `False`).
    bake_weight_multipass_on_save: Whether to materialize static weight multi-pass
      reconstruction (`sum_k p_k`) into saved checkpoints (default `False`).
    apply_weight_multipass_on_load: Whether to materialize static weight multi-pass
      reconstruction onto loaded model parameters (default `False`).
    log_diagnostics: Whether to log matched module paths and pass summaries.
    auxiliary_features: Optional tuple of additional `EmulationFeature` plugins.
  """

  enabled: bool = False
  rules: tuple[MultipassRuleConfig, ...] = ()
  start_step: int = 0
  end_step: int | None = None
  emulate_reference_model: bool = False
  emulate_teacher_model: bool = False
  bake_weight_multipass_on_save: bool = False
  apply_weight_multipass_on_load: bool = False
  log_diagnostics: bool = True
  auxiliary_features: tuple[Any, ...] = ()

  @property
  def multipass_enabled(self) -> bool:
    return self.enabled and any(r.has_any_multipass for r in self.rules)

  @property
  def active_auxiliary_features(self) -> tuple[Any, ...]:
    if not self.enabled:
      return ()
    return tuple(
        f for f in self.auxiliary_features if getattr(f, "enabled", False)
    )

  @property
  def extra_features(self) -> dict[str, Any]:
    return {f.feature_name: f for f in self.auxiliary_features}

  # Backward-compatible aliases so shared hooks/checkpoint utilities work seamlessly
  @property
  def sparsity_enabled(self) -> bool:
    return self.multipass_enabled

  @property
  def sparsify_reference_model(self) -> bool:
    return self.emulate_reference_model

  @property
  def sparsify_teacher_model(self) -> bool:
    return self.emulate_teacher_model

  @property
  def bake_weight_sparsity_on_save(self) -> bool:
    return self.bake_weight_multipass_on_save

  @property
  def apply_weight_sparsity_on_load(self) -> bool:
    return self.apply_weight_multipass_on_load

  def is_active_at_step(self, step: int | None) -> bool:
    """Returns True if multi-pass emulation is active at training step `step`."""
    if not self.enabled:
      return False
    if not self.multipass_enabled and not self.active_auxiliary_features:
      return False
    if step is None:
      return True
    if step < self.start_step:
      return False
    if self.end_step is not None and self.end_step >= 0 and step > self.end_step:
      return False
    return True

  def match_rule(
      self,
      module_path: str,
      op_name: str = "dot_general",
      *,
      step: int | None = None,
      allow_unindexed_scan_layer: bool = False,
  ) -> MultipassRuleConfig | None:
    """Returns the first matching `MultipassRuleConfig` for `(module_path, op_name)`."""
    if not self.is_active_at_step(step):
      return None
    for rule in self.rules:
      if rule.matches(
          module_path,
          op_name,
          allow_unindexed_scan_layer=allow_unindexed_scan_layer,
      ):
        return rule
    return None

  def summary(self) -> str:
    if not self.enabled:
      return "MultipassEmulationConfig(enabled=False)"
    parts = []
    for r in self.rules:
      lhs_s = r.lhs.summary() if (r.lhs and r.lhs.enabled) else "off"
      rhs_s = r.rhs.summary() if (r.rhs and r.rhs.enabled) else "off"
      pairs_s = "+".join(f"({i},{j})" for i, j in r.effective_fwd_pairs)
      bwd_flags = []
      if r.emulate_dlhs:
        bwd_flags.append("dlhs")
      if r.emulate_drhs:
        bwd_flags.append("drhs")
      bwd_s = f"{r.bwd_mode}[{','.join(bwd_flags)}]" if bwd_flags else r.bwd_mode
      layer_s = f", layers={list(r.layers)}" if r.layers is not None else ""
      parts.append(
          f"Rule[{r.name}](path={r.module_path!r}{layer_s}, ops={list(r.op_names)}, "
          f"lhs={lhs_s}, rhs={rhs_s}, cross_terms={r.cross_term_mode}[{pairs_s}], "
          f"acc={r.acc_dtype.name}, out={r.out_dtype.name}, bwd={bwd_s})"
      )
    for aux in self.active_auxiliary_features:
      parts.append(aux.summary())
    sched = ""
    if self.start_step > 0 or (self.end_step is not None and self.end_step >= 0):
      sched = f", schedule=[{self.start_step}, {self.end_step}]"
    return f"MultipassEmulationConfig(enabled=True{sched}, rules=[{'; '.join(parts)}])"

  def to_dict(self) -> dict[str, Any]:
    return {
        "enabled": self.enabled,
        "start_step": self.start_step,
        "end_step": self.end_step,
        "emulate_reference_model": self.emulate_reference_model,
        "emulate_teacher_model": self.emulate_teacher_model,
        "bake_weight_multipass_on_save": self.bake_weight_multipass_on_save,
        "apply_weight_multipass_on_load": self.apply_weight_multipass_on_load,
        "log_diagnostics": self.log_diagnostics,
        "rules": [r.to_dict() for r in self.rules],
        "auxiliary_features": {
            f.feature_name: f.to_dict() for f in self.auxiliary_features
        },
    }

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), indent=2)

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> MultipassEmulationConfig:
    """Builds `MultipassEmulationConfig` from a dictionary or flat MaxText config keys."""
    if not data:
      return cls(enabled=False)

    if (
        "multipass_emulation" in data
        and isinstance(data["multipass_emulation"], Mapping)
        and data["multipass_emulation"]
    ):
      data = data["multipass_emulation"]

    def _get(key: str, default: Any = None) -> Any:
      for candidate in (
          key,
          f"emulate_multipass_{key}",
          f"multipass_{key}",
          f"emulate_{key}",
      ):
        if candidate in data and data[candidate] is not None:
          val = data[candidate]
          if isinstance(val, str) and val.strip() == "":
            continue
          if isinstance(val, (int, float)) and not isinstance(val, bool) and val < 0:
            continue
          if isinstance(val, (list, tuple)) and len(val) == 0:
            continue
          return val
      return default

    preset_raw = _get("preset", "")
    preset_str = str(preset_raw).strip().upper() if preset_raw else ""

    explicit_enabled = data.get(
        "enabled",
        data.get(
            "emulate_multipass_enabled",
            data.get("multipass_enabled", None),
        ),
    )

    rules_raw = _get("rules", None)
    if isinstance(rules_raw, str) and rules_raw.strip():
      if rules_raw.strip().startswith("[") or rules_raw.strip().startswith("{"):
        rules_raw = json.loads(rules_raw)
      else:
        rules_raw = None

    has_shortcut = bool(
        preset_str
        or rules_raw
        or _get("pass_dtypes")
        or _get("lhs_pass_dtypes")
        or _get("rhs_pass_dtypes")
    )
    enabled = (
        bool(explicit_enabled)
        if explicit_enabled is not None
        else has_shortcut
    )

    if not enabled:
      return cls(enabled=False)

    # Resolve module_path and exclude_path from target_modules / target_submodules if specified
    raw_mod_path = _get("module_path", ".*")
    raw_exc_path = _get(
        "exclude_path",
        r"(^|/)(token_embedder|embed|logits_dense|norm|scale)(/|$)",
    )
    target_modules_str = str(_get("target_modules", "")).strip()
    target_submodules_str = str(_get("target_submodules", "")).strip()

    if raw_mod_path == ".*" and (target_modules_str or target_submodules_str):
      mod_group_map = {
          "mlp": r"(?:mlp|wi|wi_0|wi_1|wo|gate_proj|up_proj|down_proj|DenseGeneral)",
          "attn": r"(?:self_attention|attention|query|key|value|out|q_proj|k_proj|v_proj|o_proj|qkv_proj)",
          "attention": r"(?:self_attention|attention|query|key|value|out|q_proj|k_proj|v_proj|o_proj|qkv_proj)",
          "moe": r"(?:MoeBlock|routed_experts|shared_experts|moe)",
          "logits": r"(?:logits_dense)",
      }
      mod_tokens = [
          m.strip().lower()
          for m in target_modules_str.split(",")
          if m.strip()
      ]
      sub_tokens = [
          re.escape(s.strip())
          for s in target_submodules_str.split(",")
          if s.strip()
      ]
      if "logits" in mod_tokens or "all" in mod_tokens:
        raw_exc_path = r"(^|/)(token_embedder|embed|norm|scale)(/|$)"
      group_pats = [
          mod_group_map[m] for m in mod_tokens if m in mod_group_map
      ]
      if group_pats and sub_tokens:
        g_union = "|".join(group_pats)
        s_union = "|".join(sub_tokens)
        raw_mod_path = rf"(^|/)({g_union})(/.*)?/({s_union})(/|$)|(^|/)({s_union})(/|$)"
        # More strictly: require group in path and submodule as leaf/component
        raw_mod_path = rf"(?=.*(^|/)({g_union})(/|$)).*(^|/)({s_union})(/|$)"
      elif sub_tokens:
        s_union = "|".join(sub_tokens)
        raw_mod_path = rf"(^|/)({s_union})(/|$)"
      elif group_pats and "all" not in mod_tokens:
        g_union = "|".join(group_pats)
        raw_mod_path = rf"(^|/)({g_union})(/|$)"

    defaults = {
        "preset": preset_str,
        "target_preset": _get("target_preset", "all_linears"),
        "module_path": raw_mod_path,
        "exclude_path": raw_exc_path,
        "layers": _get("layers", _get("target_layers", None)),
        "exclude_layers": _get("exclude_layers", None),
        "op_names": _get("op_names", ("dot_general", "gmm", "ragged_dot")),
        "src_dtype": _get("src_dtype", None),
        "num_passes": _get("num_passes", _get("passes", None)),
        "pass_dtypes": _get("pass_dtypes", None),
        "lhs_src_dtype": _get("lhs_src_dtype", None),
        "lhs_num_passes": _get("lhs_num_passes", None),
        "lhs_pass_dtypes": _get("lhs_pass_dtypes", None),
        "lhs_scale_mode": _get("lhs_scale_mode", None),
        "rhs_src_dtype": _get("rhs_src_dtype", None),
        "rhs_num_passes": _get("rhs_num_passes", None),
        "rhs_pass_dtypes": _get("rhs_pass_dtypes", None),
        "rhs_scale_mode": _get("rhs_scale_mode", None),
        "emulate_lhs": _get("lhs", True),
        "emulate_rhs": _get("rhs", True),
        "scale_mode": _get("scale_mode", None),
        "block_size": _get("block_size", _get("scale_block_size", 128)),
        "rounding_mode": _get("rounding_mode", "rne"),
        "flush_subnormals": _get("flush_subnormals", False),
        "cross_term_mode": _get("cross_term_mode", None),
        "max_pass_degree": _get(
            "max_pass_degree", _get("max_total_pass_order", None)
        ),
        "active_passes": _get(
            "active_passes", _get("custom_active_pairs", None)
        ),
        "acc_dtype": _get("acc_dtype", None),
        "out_dtype": _get("out_dtype", "auto"),
        "bwd_mode": _get("bwd_mode", "multipass_bwd"),
        "emulate_fwd": _get("fwd", _get("on_fwd", True)),
        "emulate_dlhs": _get("dlhs", _get("on_dlhs", False)),
        "emulate_drhs": _get("drhs", _get("on_drhs", False)),
        "dlhs_grad": _get("dlhs_grad", None),
        "dlhs_rhs": _get("dlhs_rhs", None),
        "dlhs_cross_term_mode": _get("dlhs_cross_term_mode", None),
        "drhs_lhs": _get("drhs_lhs", None),
        "drhs_grad": _get("drhs_grad", None),
        "drhs_cross_term_mode": _get("drhs_cross_term_mode", None),
        "input_activation": _get("input_activation", None),
        "output_activation": _get("output_activation", None),
    }

    rules: list[MultipassRuleConfig] = []
    if isinstance(rules_raw, Sequence) and not isinstance(rules_raw, str):
      for idx, item in enumerate(rules_raw):
        if isinstance(item, MultipassRuleConfig):
          rules.append(item)
        elif isinstance(item, Mapping):
          rule_dict = dict(item)
          rule_dict.setdefault("name", f"rule_{idx}")
          rules.append(MultipassRuleConfig.from_dict(rule_dict, defaults=defaults))
    elif isinstance(rules_raw, Mapping):
      for rule_name, rule_body in rules_raw.items():
        if isinstance(rule_body, Mapping):
          rule_dict = dict(rule_body)
          rule_dict.setdefault("name", str(rule_name))
          rules.append(MultipassRuleConfig.from_dict(rule_dict, defaults=defaults))

    if not rules:
      default_rule = MultipassRuleConfig.from_dict(defaults, defaults=defaults)
      if default_rule.has_any_multipass:
        rules.append(default_rule)

    # Parse any registered EmulationFeature plugins
    aux_features: list[Any] = []
    try:
      try:
        from maxtext.codesign.emulation_base import get_registered_emulation_features
      except ImportError:
        from emulation_base import get_registered_emulation_features  # type: ignore
      reg_features = get_registered_emulation_features()
      aux_map = data.get("auxiliary_features", {})
      for feat_name, feat_cls in reg_features.items():
        if feat_name in ("multipass_precision", "multipass"):
          continue
        feat_data = data.get(feat_name, aux_map.get(feat_name) if isinstance(aux_map, Mapping) else None)
        if isinstance(feat_data, Mapping):
          aux_features.append(feat_cls.from_dict(feat_data))
    except Exception:
      pass

    start_step = int(_get("start_step", 0))
    end_step_raw = _get("end_step", None)
    end_step = (
        int(end_step_raw)
        if end_step_raw is not None and int(end_step_raw) >= 0
        else None
    )

    return cls(
        enabled=bool(rules or aux_features),
        rules=tuple(rules),
        start_step=start_step,
        end_step=end_step,
        emulate_reference_model=bool(
            data.get(
                "emulate_reference_model",
                data.get("emulate_multipass_reference_model", False),
            )
        ),
        emulate_teacher_model=bool(
            data.get(
                "emulate_teacher_model",
                data.get("emulate_multipass_teacher", False),
            )
        ),
        bake_weight_multipass_on_save=bool(
            _get("bake_weight_multipass_on_save", False)
        ),
        apply_weight_multipass_on_load=bool(
            _get("apply_weight_multipass_on_load", False)
        ),
        log_diagnostics=bool(_get("log_diagnostics", True)),
        auxiliary_features=tuple(aux_features),
    )

  @classmethod
  def from_maxtext_config(cls, config: Any) -> MultipassEmulationConfig:
    """Extracts `MultipassEmulationConfig` from a MaxText `HyperParameters` object or dict."""
    if isinstance(config, MultipassEmulationConfig):
      return config
    if isinstance(config, Mapping):
      return cls.from_dict(config)
    if hasattr(config, "get_keys"):
      return cls.from_dict(config.get_keys())
    if hasattr(config, "__dict__"):
      return cls.from_dict(vars(config))
    return cls(enabled=False)


# ---------------------------------------------------------------------------
# Default Config Keys Registered into MaxText `pyconfig` (With Documentation)
# ---------------------------------------------------------------------------

# Documented reference dictionary describing every newly introduced CLI / YAML flag.
MULTIPASS_FLAG_DOCUMENTATION: dict[str, str] = {
    "emulate_multipass_enabled": (
        "Master boolean switch enabling Multi-Pass Precision Emulation across "
        "training or evaluation workflows."
    ),
    "emulate_multipass_preset": (
        "Optional named multi-pass algorithm preset from the XLA:TPU RFC and literature: "
        "'F8E4M3_F8E4M3_F32_X1', 'F8E4M3_F8E4M3_F32_X2', 'F8E4M3_F8E4M3_F32_X3', "
        "'F8E4M3_F8E4M3_F32_X4', 'F8E5M2_F8E5M2_F32_X3', 'BF16_BF16_F32_X2', "
        "'BF16_BF16_F32_X3', 'BF16_BF16_F32_X6', 'INT8_INT4_INT32_X2', "
        "'INT8_INT8_INT32_X3', or 'INT8_INT8_INT32_X4'. Any explicit CLI/YAML flag "
        "overrides the corresponding preset field."
    ),
    "emulate_src_dtype": (
        "Source tensor dtype 'A' into which each operand is cast/quantized prior to "
        "residual pass slicing (e.g., 'bf16', 'fp32', 'fp16', 'int8', 'auto', or custom 'eXmY')."
    ),
    "emulate_num_passes": (
        "Number of sequential residual passes 'K' into which each operand is decomposed "
        "(e.g., 1, 2, 3, 4). If 'emulate_pass_dtypes' provides an explicit list of length M > 1, "
        "K defaults to M unless explicitly overridden."
    ),
    "emulate_pass_dtypes": (
        "Target dtype(s) '[B, C, D, ...]' for each residual pass. Can be a single dtype "
        "(e.g. 'fp8_e4m3', 'bf16', 'int4', 'e5m3') repeated K times, or a comma-separated / "
        "YAML list of heterogeneous dtypes per pass (e.g. 'fp8_e4m3,fp8_e4m3' or "
        "'[fp8_e5m2, fp8_e4m3, fp4_e2m1]'). Supports standard floats, ints, and arbitrary 'eXmY'/'intB'."
    ),
    "emulate_lhs_src_dtype": (
        "Optional override for the LHS (activation) operand source dtype 'A'."
    ),
    "emulate_lhs_num_passes": (
        "Optional override for the number of residual passes K_A on the LHS (activation) operand."
    ),
    "emulate_lhs_pass_dtypes": (
        "Optional override for the LHS (activation) per-pass target dtypes '[B_0, B_1, ...]'."
    ),
    "emulate_lhs_scale_mode": (
        "Optional override for the LHS (activation) residual scaling mode."
    ),
    "emulate_rhs_src_dtype": (
        "Optional override for the RHS (weight) operand source dtype 'A'."
    ),
    "emulate_rhs_num_passes": (
        "Optional override for the number of residual passes K_B on the RHS (weight) operand."
    ),
    "emulate_rhs_pass_dtypes": (
        "Optional override for the RHS (weight) per-pass target dtypes '[C_0, C_1, ...]'."
    ),
    "emulate_rhs_scale_mode": (
        "Optional override for the RHS (weight) residual scaling mode."
    ),
    "emulate_multipass_lhs": (
        "Boolean switch controlling whether the LHS (activation) operand is decomposed into passes."
    ),
    "emulate_multipass_rhs": (
        "Boolean switch controlling whether the RHS (weight) operand is decomposed into passes."
    ),
    "emulate_scale_mode": (
        "Per-pass residual scaling strategy before quantizing to each pass dtype: "
        "'per_tensor_pow2' (power-of-two tensor scale), 'per_tensor_amax' (FP32 tensor amax scale), "
        "'per_channel_pow2' (contracting-axis channel power-of-two scale), 'per_channel_amax', "
        "'per_block_pow2' (MX-style block power-of-two scale), 'per_block_amax', "
        "'bit_shift' (integer bit-shift slicing), or 'none' (unscaled direct rounding)."
    ),
    "emulate_block_size": (
        "Block size along the contracting dimension when 'emulate_scale_mode' is "
        "'per_block_pow2' or 'per_block_amax' (default: 128)."
    ),
    "emulate_rounding_mode": (
        "Rounding mode applied when quantizing each pass: 'rne' (round-to-nearest-even), "
        "'stochastic' (unbiased stochastic rounding), or 'truncate' (round toward zero)."
    ),
    "emulate_flush_subnormals": (
        "If True, flushes values smaller than the minimum normal of the pass dtype to zero."
    ),
    "emulate_cross_term_mode": (
        "Cross-term pass pair selection policy for GEMM accumulation: "
        "'triangular' (all (i,j) with i + j <= max_pass_degree, e.g. X3 'hh+hl+lh' for K=2), "
        "'full' (all K_A x K_B cross-terms, e.g. X4 'hh+hl+lh+ll' for K=2), "
        "'asymmetric_lhs' (X2 where LHS has K_A passes and RHS has 1 pass), "
        "'asymmetric_rhs' (X2 where LHS has 1 pass and RHS has K_B passes), "
        "'karatsuba' (3-matmul Karatsuba recombination for 2-pass decomposition), or "
        "'custom' (explicit pass pairs via 'emulate_active_passes')."
    ),
    "emulate_max_pass_degree": (
        "Maximum pass index sum (i + j) included when 'emulate_cross_term_mode=triangular'. "
        "Set to -1 to default to max(K_A - 1, K_B - 1) (which gives i+j <= 1 -> X3 for K=2, "
        "and i+j <= 2 -> X6 for K=3)."
    ),
    "emulate_active_passes": (
        "Explicit comma-separated '(i:j)' pass index pairs to execute when "
        "'emulate_cross_term_mode=custom' (e.g., '0:0,0:1,1:0')."
    ),
    "emulate_acc_dtype": (
        "Accumulator dtype used to accumulate each pass's dot product and sum across passes "
        "(e.g., 'fp32', 'bf16', 'int32', or custom 'eXmY')."
    ),
    "emulate_out_dtype": (
        "Output dtype to cast the accumulated multi-pass result into ('auto' preserves "
        "the original matmul output dtype, or specify 'bf16', 'fp32', etc.)."
    ),
    "emulate_bwd_mode": (
        "Backward pass differentiation mode: 'multipass_bwd' (custom VJP supporting "
        "independent multi-pass emulation on forward, dlhs, and drhs GEMMs) or "
        "'ste' (Straight-Through Estimator in backward)."
    ),
    "emulate_fwd": (
        "Boolean switch enabling multi-pass emulation on the forward pass GEMM (fprop)."
    ),
    "emulate_dlhs": (
        "Boolean switch enabling multi-pass emulation on the activation-gradient GEMM "
        "(dgrad: dlhs = dy @ w^T) when 'emulate_bwd_mode=multipass_bwd'."
    ),
    "emulate_drhs": (
        "Boolean switch enabling multi-pass emulation on the weight-gradient GEMM "
        "(wgrad: drhs = x^T @ dy) when 'emulate_bwd_mode=multipass_bwd'."
    ),
    "emulate_dlhs_cross_term_mode": (
        "Optional cross-term mode override for the backward 'dlhs' (activation gradient) GEMM."
    ),
    "emulate_drhs_cross_term_mode": (
        "Optional cross-term mode override for the backward 'drhs' (weight gradient) GEMM."
    ),
    "emulate_target_preset": (
        "Module selector preset: 'all_linears' (all dense/MoE projections except embeddings "
        "and norms), 'mlp', 'attn', 'qkv', 'attn_out', 'moe', 'moe_router', or "
        "'all_including_attn_scores'."
    ),
    "emulate_module_path": (
        "Regex pattern matched against normalized module paths to select target layers."
    ),
    "emulate_exclude_path": (
        "Regex pattern of module paths to exclude from multi-pass emulation."
    ),
    "emulate_layers": (
        "Optional decoder layer index filter (e.g., '14', '0,1,2', or '8-15'). "
        "Works under both scan_layers=True and scan_layers=False."
    ),
    "emulate_exclude_layers": (
        "Optional decoder layer indices to exclude from multi-pass emulation."
    ),
    "emulate_op_names": (
        "Comma-separated list or sequence of intercepted operations: "
        "'dot_general', 'einsum', 'gmm', 'ragged_dot'."
    ),
    "emulate_start_step": (
        "Training step (inclusive) at which multi-pass emulation turns on (default: 0). "
        "Steps before start_step execute standard dense/full-precision matmuls without recompilation."
    ),
    "emulate_end_step": (
        "Training step (inclusive) after which multi-pass emulation turns off (-1 = never)."
    ),
    "emulate_reference_model": (
        "If True, also applies multi-pass emulation to the Reference model in RL/DPO "
        "(default: False, keeping Reference model un-emulated)."
    ),
    "emulate_teacher_model": (
        "If True, also applies multi-pass emulation to the Teacher model in Distillation "
        "(default: False, keeping Teacher model un-emulated)."
    ),
    "emulate_bake_weight_multipass_on_save": (
        "If True, bakes static multi-pass weight reconstruction into saved checkpoints."
    ),
    "emulate_apply_weight_multipass_on_load": (
        "If True, applies static multi-pass weight reconstruction when loading checkpoints."
    ),
    "emulate_log_diagnostics": (
        "If True, logs matched module paths and active multi-pass configuration summaries."
    ),
    "emulate_rules": (
        "Optional YAML list of granular per-module/per-layer rule dictionaries."
    ),
}

DEFAULT_MULTIPASS_CONFIG_KEYS: dict[str, Any] = {
    "multipass_emulation": {},
    "emulate_multipass_enabled": False,
    "emulate_multipass_preset": "",
    "emulate_src_dtype": "",
    "emulate_num_passes": -1,
    "emulate_multipass_passes": -1,
    "emulate_pass_dtypes": "",
    "emulate_lhs_src_dtype": "",
    "emulate_lhs_num_passes": -1,
    "emulate_lhs_pass_dtypes": "",
    "emulate_lhs_scale_mode": "",
    "emulate_rhs_src_dtype": "",
    "emulate_rhs_num_passes": -1,
    "emulate_rhs_pass_dtypes": "",
    "emulate_rhs_scale_mode": "",
    "emulate_multipass_lhs": True,
    "emulate_multipass_rhs": True,
    "emulate_scale_mode": "",
    "emulate_block_size": -1,
    "emulate_scale_block_size": -1,
    "emulate_rounding_mode": "rne",
    "emulate_flush_subnormals": False,
    "emulate_subnormals": True,
    "emulate_cross_term_mode": "",
    "emulate_max_pass_degree": -1,
    "emulate_max_total_pass_order": -1,
    "emulate_active_passes": "",
    "emulate_custom_active_pairs": "",
    "emulate_acc_dtype": "",
    "emulate_out_dtype": "auto",
    "emulate_use_reconstructed_fast_path": False,
    "emulate_bwd_mode": "multipass_bwd",
    "emulate_fwd": True,
    "emulate_dlhs": False,
    "emulate_drhs": False,
    "emulate_multipass_on_fwd": True,
    "emulate_multipass_on_dlhs": False,
    "emulate_multipass_on_drhs": False,
    "emulate_dlhs_cross_term_mode": "",
    "emulate_drhs_cross_term_mode": "",
    "emulate_target_preset": "all_linears",
    "emulate_target_modules": "",
    "emulate_target_submodules": "",
    "emulate_target_layers": "",
    "emulate_module_path": ".*",
    "emulate_exclude_path": (
        r"(^|/)(token_embedder|embed|logits_dense|norm|scale)(/|$)"
    ),
    "emulate_layers": "",
    "emulate_exclude_layers": "",
    "emulate_op_names": ["dot_general", "gmm", "ragged_dot"],
    "emulate_start_step": 0,
    "emulate_end_step": -1,
    "emulate_multipass_start_step": 0,
    "emulate_multipass_end_step": -1,
    "emulate_reference_model": False,
    "emulate_teacher_model": False,
    "emulate_target_models": "actor",
    "emulate_sync_reconstructed_weights_to_rollout": True,
    "emulate_bake_weight_multipass_on_save": False,
    "emulate_apply_weight_multipass_on_load": False,
    "emulate_log_diagnostics": True,
    "emulate_rules": [],
    "emulate_multipass_rules": [],
}

# Aliases for compatibility with shared hooks / checkpoint utilities
EmulationConfig = MultipassEmulationConfig
SparsityEmulationConfig = MultipassEmulationConfig
SparsityRuleConfig = MultipassRuleConfig
TensorSparsitySpec = OperandMultipassSpec
DEFAULT_EMULATION_CONFIG_KEYS = DEFAULT_MULTIPASS_CONFIG_KEYS
