"""MaxText AQT sweeps configurations."""

import aqt.jax.v2.config as aqt_config


def set_bwd_cfg(cfg: aqt_config.DotGeneral, dlhs_int8: bool, drhs_int8: bool):
  """Configure bwd matmuls separately."""
  dlhs_bits = 8 if dlhs_int8 else None
  drhs_bits = 8 if drhs_int8 else None
  # Save use_fwd_quant values
  dlhs_fwd_quant = cfg.dlhs.rhs.use_fwd_quant
  drhs_fwd_quant = cfg.drhs.rhs.use_fwd_quant
  # Configure bwd matmuls
  dlhs_cfg = aqt_config.DotGeneralRaw.make(dlhs_bits, dlhs_bits)
  drhs_cfg = aqt_config.DotGeneralRaw.make(drhs_bits, drhs_bits)
  dlhs_cfg.rhs.use_fwd_quant = dlhs_fwd_quant
  drhs_cfg.rhs.use_fwd_quant = drhs_fwd_quant
  cfg.dlhs = dlhs_cfg
  cfg.drhs = drhs_cfg


def sweep1(fwd_int8: bool, dlhs_int8: bool, drhs_int8: bool) -> aqt_config.DotGeneral:
  fqt_config = aqt_config.fully_quantized(
      fwd_bits=8 if fwd_int8 else None,
      bwd_bits=None,
      use_fwd_quant=False,
      use_stochastic_rounding=None,
      vjp_lhs_stochastic_rounding=True,
      vjp_rhs_stochastic_rounding=False,
  )
  set_bwd_cfg(fqt_config, dlhs_int8=dlhs_int8, drhs_int8=drhs_int8)
  return fqt_config

