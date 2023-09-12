"""MaxText AQT sweeps configurations."""

from typing import Optional

from aqt.jax.v2.config import DotGeneral
from aqt.jax.v2.config import DotGeneralRaw
from aqt.jax.v2.config import set_static_bound
from aqt.jax.v2.config import set_stochastic_rounding


def fully_quantized(
    *,
    fwd_bits: int | None = 8,
    dlhs_bits: int | None = 8,
    drhs_bits: int | None = 8,
    use_fwd_quant: bool = True,
    use_stochastic_rounding: Optional[bool] = True,
    # Typically we have (but it's a caller's responsibility to check):
    # - vjp_lhs_stochastic_rounding is referring to the gradient and
    # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
    vjp_lhs_stochastic_rounding: Optional[bool] = None,
    vjp_rhs_stochastic_rounding: Optional[bool] = None,
    # The dummy static bound flag is temporary, for performance benchmarking.
    use_dummy_static_bound: bool = False,
    rng_type: str = 'jax.uniform',
) -> DotGeneral:
  """Fully Quantized Training."""
  fwd = DotGeneralRaw.make(fwd_bits, fwd_bits)
  dlhs = DotGeneralRaw.make(dlhs_bits, dlhs_bits)
  drhs = DotGeneralRaw.make(drhs_bits, drhs_bits)
  cfg = DotGeneral(fwd=fwd, dlhs=dlhs, drhs=drhs)

  # Surprising: lhs quantization determines what drhs can do.
  if fwd_bits is not None:
    # Only rhs is accepting MultiTensor.
    cfg.dlhs.rhs.use_fwd_quant = use_fwd_quant
    cfg.drhs.rhs.use_fwd_quant = use_fwd_quant

  # Stochastic Rounding
  # These 3 variables are used to ensure we don't mix
  # old and new style of SR configuration.
  old_style_sr_config = use_stochastic_rounding is not None
  new_style_sr_config_lhs = vjp_lhs_stochastic_rounding is not None
  new_style_sr_config_rhs = vjp_rhs_stochastic_rounding is not None
  assert new_style_sr_config_lhs == new_style_sr_config_rhs, (
      'if you use new style SR config (vjp_xhs_stochastic_rounding), do pass'
      ' both lhs and rhs explicitely.'
  )
  assert new_style_sr_config_lhs != old_style_sr_config

  true = True  # A crude way to get around g-explicit-bool-comparison warning

  # By default use jax.uniform for stochastic rounding
  if use_stochastic_rounding == true:
    set_stochastic_rounding(cfg, True, True, rng_type)

  if vjp_lhs_stochastic_rounding == true:
    set_stochastic_rounding(cfg, True, False, rng_type)

  if vjp_rhs_stochastic_rounding == true:
    set_stochastic_rounding(cfg, False, True, rng_type)

  if use_dummy_static_bound:
    set_static_bound(cfg, 1.0)

  return cfg


def sweep1(
    fwd_int8: bool,
    dlhs_int8: bool,
    drhs_int8: bool,
    use_dummy_static_bound: bool=False,
    rng_type: str = 'jax.uniform',
    use_fwd_quant=False,
) -> DotGeneral:
  fqt_config = fully_quantized(
      fwd_bits=8 if fwd_int8 else None,
      dlhs_bits=8 if dlhs_int8 else None,
      drhs_bits=8 if drhs_int8 else None,
      use_fwd_quant=use_fwd_quant,
      use_stochastic_rounding=None,
      vjp_lhs_stochastic_rounding=True,
      vjp_rhs_stochastic_rounding=False,
      use_dummy_static_bound=use_dummy_static_bound,
      rng_type=rng_type,
  )
  return fqt_config
