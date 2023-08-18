"""MaxText AQT sweeps configurations."""

from typing import Optional

import aqt.jax.v2.config as aqt_config
from aqt.jax.v2.config import DotGeneral
from aqt.jax.v2.config import set_static_bound
from aqt.jax.v2.config import set_stochastic_rounding


def fully_quantized(
    *,
    fwd_bits: int | None = 8,
    bwd_bits: int | None = 8,
    use_fwd_quant: bool = True,
    use_stochastic_rounding: Optional[bool] = True,
    # Typically we have (but it's a caller's responsibility to check):
    # - vjp_lhs_stochastic_rounding is referring to the gradient and
    # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
    vjp_lhs_stochastic_rounding: Optional[bool] = None,
    vjp_rhs_stochastic_rounding: Optional[bool] = None,
    # The dummy static bound flag is temporary, for performance benchmarking.
    use_dummy_static_bound: bool = False,
) -> DotGeneral:
  """Fully Quantized Training."""
  cfg = DotGeneral.make(
      lhs_bits=fwd_bits,
      rhs_bits=fwd_bits,
      bwd_bits=bwd_bits,
      use_fwd_quant=use_fwd_quant,
  )

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
    set_stochastic_rounding(cfg, True, True, 'jax.uniform')

  if vjp_lhs_stochastic_rounding == true:
    set_stochastic_rounding(cfg, True, False, 'jax.uniform')

  if vjp_rhs_stochastic_rounding == true:
    set_stochastic_rounding(cfg, False, True, 'jax.uniform')

  if use_dummy_static_bound:
    set_static_bound(cfg, 1.0)

  return cfg


def sweep1(fwd_int8: bool, bwd_int8: bool) -> aqt_config.DotGeneral:
  fqt_config = aqt_config.fully_quantized(
      fwd_bits=8 if fwd_int8 else None,
      bwd_bits=8 if bwd_int8 else None,
      use_fwd_quant=False,
      use_stochastic_rounding=None,
      vjp_lhs_stochastic_rounding=True,
      vjp_rhs_stochastic_rounding=False,
  )
  return fqt_config

