#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Quantization library."""

import functools
from typing import Optional

from aqt.jax.v2 import aqt_dot_general as aqt
from aqt.jax.v2 import config
import jax.numpy as jnp
import common_types


def int8_dot_general(aqt_rng: Optional[common_types.PRNGKey], maxtext_config: Optional[common_types.Config]):
  """Rewrite dot_general to aqt int8 quantized dot_general."""
  if aqt_rng is None:
    raise ValueError('aqt_rng cannot be None.')

  if maxtext_config is None:
    raise ValueError('config cannot be None.')

  if maxtext_config.local_aqt_shards == 0:
    aqt_cfg = config.config_v3(
        fwd_bits=8,
        dlhs_bits=8,
        drhs_bits=None,
        rng_type='jax.uniform',
        dlhs_local_aqt=None,
        drhs_local_aqt=None,
        fwd_accumulator_dtype=jnp.int32,
        dlhs_accumulator_dtype=jnp.int32,
        drhs_accumulator_dtype=jnp.int32,
    )
  else:
    aqt_cfg = config.config_v3(
        fwd_bits=8,
        dlhs_bits=8,
        drhs_bits=8,
        rng_type='jax.uniform',
        dlhs_local_aqt=None,
        drhs_local_aqt=None,
        fwd_accumulator_dtype=jnp.int32,
        dlhs_accumulator_dtype=jnp.int32,
        drhs_accumulator_dtype=jnp.int32,
    )
  aqt_dot_general = aqt.make_dot_general(aqt_cfg)
  context = aqt.Context(key=aqt_rng, train_step=None)
  aqt_dot_general = functools.partial(aqt_dot_general, context=context)
  return aqt_dot_general
