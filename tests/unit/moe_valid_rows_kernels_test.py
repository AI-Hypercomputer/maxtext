# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The valid-rows MoE kernels (`unsort_bwd_rows`, `glu_rows`) against the XLA code they replace.

On CPU the kernels run in Pallas interpret mode with the TPU7x chip description, over a buffer whose last
1024-row block is partial. On a TPU the SiLU-GLU kernels must match XLA's fusions bit for bit for every bf16 gate.
"""

import jax
from jax._src.pallas.mosaic import tpu_info
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels.ragged import glu_rows
from maxtext.kernels.ragged import unsort_bwd_rows as unsort
from maxtext.layers import moe

_ROWS, _HIDDEN, _MLP = 2560, 256, 128


@pytest.fixture(autouse=True)
def _tpu7x_interpret():
  if jax.default_backend() == "cpu":
    tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
    tpu_info.get_tpu_info.cache_clear()


def _bf16(key, shape, scale=1.0):
  return (jax.random.normal(key, shape, jnp.float32) * scale).astype(jnp.bfloat16)


def _bits(x, rows):
  return np.asarray(x)[:rows].view(np.uint16)


def _silu_glu(gate, up):
  return (jax.nn.silu(gate) * up).astype(jnp.bfloat16)


@pytest.mark.parametrize("limit", [0, 1000, 1700, _ROWS])
def test_unsort_bwd_rows(limit):
  """Weighted gradient bit-identical and row dot to f32 summation order on rows < limit, zeros up to the block end."""
  keys = jax.random.split(jax.random.PRNGKey(0), 3)
  gathered, tokens = _bf16(keys[0], (_ROWS, _HIDDEN)), _bf16(keys[1], (_ROWS, _HIDDEN))
  weights = jax.random.uniform(keys[2], (_ROWS,), jnp.float32, -1.0, 1.0)
  interpret = True if jax.default_backend() == "cpu" else None
  grad, dot = unsort.unsort_bwd_rows(gathered, tokens, weights, jnp.int32(limit), interpret=interpret)
  ref_grad, ref_dot = unsort.unsort_bwd_rows_reference(gathered, tokens, weights, jnp.int32(limit))
  block_end = min(_ROWS, max(-(-limit // 1024), 1) * 1024)
  np.testing.assert_array_equal(_bits(grad, block_end), _bits(ref_grad, block_end))
  np.testing.assert_allclose(np.asarray(dot)[:limit], np.asarray(ref_dot)[:limit], rtol=0, atol=1e-4)
  np.testing.assert_array_equal(np.asarray(dot)[limit:block_end], 0.0)


def _glu_inputs(rows, mlp):
  """Every bf16 bit pattern as a gate value, against random up and output gradients."""
  gate = np.resize(np.arange(1 << 16, dtype=np.uint16).view(jnp.bfloat16), (rows, mlp))
  keys = jax.random.split(jax.random.PRNGKey(1), 2)
  fused = jnp.concatenate([jnp.asarray(gate), _bf16(keys[0], (rows, mlp), 4.0)], axis=-1)
  return fused, _bf16(keys[1], (rows, mlp), 4.0)


def test_glu_rows_interpret_rows_and_blocks():
  """CPU: the visited rows (partial last block) hold the per-tile formula; XLA:CPU may contract it differently."""
  if jax.default_backend() != "cpu":
    pytest.skip("interpret-mode check")
  fused, d_out = _glu_inputs(_ROWS, _MLP)
  gate, up = fused[:, :_MLP], fused[:, _MLP:]
  act = glu_rows.glu_fwd_rows(fused, jnp.int32(2100), interpret=True)
  d_fused = glu_rows.glu_bwd_rows(fused, d_out, jnp.int32(2100), interpret=True)
  ref_act = glu_rows._glu_fwd_tile(gate, up)  # pylint: disable=protected-access
  ref_d_fused = jnp.concatenate(glu_rows._glu_bwd_tile(gate, up, d_out), axis=-1)  # pylint: disable=protected-access
  for got, want in ((act, ref_act), (d_fused, ref_d_fused)):
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=2**-6, atol=1e-37)


@pytest.mark.tpu_only
def test_glu_rows_match_xla_fusions():
  """TPU: activation and gradient bit-identical to `fused_glu`'s XLA path, for every bf16 gate."""
  rows = 8192
  fused, d_out = _glu_inputs(rows, 1024)
  limit = 6000
  block_end = 6144
  act = jax.jit(glu_rows.glu_fwd_rows)(fused, jnp.int32(limit))
  d_fused = jax.jit(glu_rows.glu_bwd_rows)(fused, d_out, jnp.int32(limit))
  # Jitted, so XLA fuses the reference as it does in the model (f32 inside the fusion, bf16 outputs).
  ref_act = jax.jit(lambda f: moe.fused_glu(f, _silu_glu))(fused)
  ref_d_fused = jax.jit(lambda f, d: jax.vjp(lambda x: moe.fused_glu(x, _silu_glu), f)[1](d)[0])(fused, d_out)
  np.testing.assert_array_equal(_bits(act, block_end), _bits(ref_act, block_end))
  np.testing.assert_array_equal(_bits(d_fused, block_end), _bits(ref_d_fused, block_end))
