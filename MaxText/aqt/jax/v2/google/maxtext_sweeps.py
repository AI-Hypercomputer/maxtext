"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """


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
