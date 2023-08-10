"""MaxText AQT sweeps configurations."""

# pylint: skip-file

import aqt.jax.v2.config as aqt_config


def sweep1(fwd_int8: bool, bwd_int8: bool) -> aqt_config.DotGeneral:
  fqt_config = aqt_config.fully_quantized(
      fwd_bits=8 if fwd_int8 else None,
      bwd_bits=8 if bwd_int8 else None,
      use_fwd_quant=False,
      use_stochastic_rounding=None,
      vjp_lhs_stochastic_rounding=True,
      vjp_rhs_stochastic_rounding=False,
      fwd_save_accumulator_memory=False,
      bwd_save_accumulator_memory=False,
  )
  return fqt_config