# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gdn_conv1d_input_in_qkv_dtype: the GDN Conv1D + SiLU with its halo-extended input in qkv's dtype.

The option computes the same float32 values in the same order (it only changes which tensors are materialized), so
op by op every output is bit-identical. Compiled, the forward outputs and the input / conv-state gradients stay
bit-identical, but the conv weight gradient is a float32 reduction over batch x sequence whose summation order XLA
may change when it fuses the tap's bf16 -> float32 cast into it (measured on CPU: a few ulp, in the configuration
that recomputes the pre-activation without packing or a conv state). The option must also actually change the
traced program; a test that only compared values would pass if the switch were ignored, so the value checks are
paired with checks that the two programs differ.
"""

# pylint: disable=protected-access

import types

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.extend import core as jax_core
import numpy as np

from maxtext.configs import types as config_types
from maxtext.kernels.gdn import gdn_bwd_pallas
from maxtext.kernels.gdn.gdn_bwd import compute_conv1d_bwd

_KERNEL_SIZE = 4


def _bits(x):
  """The array's bytes, so equality means bit-identical (NaN-safe, -0.0 != 0.0)."""
  return np.asarray(jax.device_get(x)).tobytes()


def _conv_inputs(seed, *, batch=2, seq_len=24, dim=16, with_state, with_bias, with_segments, dtype=jnp.bfloat16):
  """Seeded Conv1D + SiLU inputs: (qkv, conv_weight, conv_bias, conv_state, dy, segment_ids, conv_halo_seg)."""
  keys = jax.random.split(jax.random.PRNGKey(seed), 6)
  qkv = jax.random.normal(keys[0], (batch, seq_len, dim), jnp.float32).astype(dtype)
  conv_weight = jax.random.normal(keys[1], (_KERNEL_SIZE, 1, dim), jnp.float32)
  conv_bias = jax.random.normal(keys[2], (dim,), jnp.float32) if with_bias else None
  conv_state = (
      jax.random.normal(keys[3], (batch, _KERNEL_SIZE - 1, dim), jnp.float32).astype(dtype) if with_state else None
  )
  dy = jax.random.normal(keys[4], (batch, seq_len, dim), jnp.float32)
  segment_ids = conv_halo_seg = None
  if with_segments:
    # Two packed documents and trailing padding (segment 0) per row.
    row = np.concatenate([np.full(9, 1), np.full(11, 2), np.zeros(seq_len - 20)]).astype(np.int32)
    segment_ids = jnp.asarray(np.stack([row] * batch))
    if with_state:
      conv_halo_seg = jnp.ones((batch, _KERNEL_SIZE - 1), jnp.int32)
  return qkv, conv_weight, conv_bias, conv_state, dy, segment_ids, conv_halo_seg


def _pad_or_concat_out_dtypes(jaxpr, shape):
  """Dtypes of every pad / concatenate in `jaxpr` (sub-jaxprs included) whose output has `shape`."""
  found = []

  def visit(jx):
    for eqn in jx.eqns:
      if eqn.primitive.name in ("pad", "concatenate"):
        aval = eqn.outvars[0].aval
        if tuple(aval.shape) == tuple(shape):
          found.append(aval.dtype)
      for sub in jax_core.jaxprs_in_params(eqn.params):
        visit(sub)

  visit(jaxpr.jaxpr)
  return found


_CASES = [
    {"testcase_name": f"state{s}_bias{b}_seg{g}", "with_state": s, "with_bias": b, "with_segments": g}
    for s in (False, True)
    for b in (False, True)
    for g in (False, True)
]


class GdnConv1dInputInQkvDtypeTest(parameterized.TestCase):

  @parameterized.named_parameters(*_CASES)
  def test_forward_bit_identical(self, with_state, with_bias, with_segments):
    qkv, w, bias, state, _, seg, halo = _conv_inputs(
        1, with_state=with_state, with_bias=with_bias, with_segments=with_segments
    )

    def fwd(narrow):
      return compute_conv1d_bwd.conv1d_silu_fwd(
          qkv,
          w,
          bias,
          _KERNEL_SIZE,
          conv_state=state,
          segment_ids=seg,
          conv_halo_seg=halo,
          conv_input_in_qkv_dtype=narrow,
      )

    ref_out, ref_act = jax.jit(lambda: fwd(False))()
    out, act = jax.jit(lambda: fwd(True))()
    self.assertEqual(ref_out.dtype, out.dtype)
    self.assertEqual(ref_act.dtype, act.dtype)
    self.assertEqual(_bits(ref_out), _bits(out))
    self.assertEqual(_bits(ref_act), _bits(act))

  @parameterized.named_parameters(*_CASES)
  def test_backward_bit_identical(self, with_state, with_bias, with_segments):
    qkv, w, bias, state, dy, seg, halo = _conv_inputs(
        2, with_state=with_state, with_bias=with_bias, with_segments=with_segments
    )
    conv_out, _ = compute_conv1d_bwd.conv1d_silu_fwd(
        qkv, w, bias, _KERNEL_SIZE, conv_state=state, segment_ids=seg, conv_halo_seg=halo
    )
    for cached_conv_out in (None, conv_out):
      for return_d_conv_state in (False, True):

        def bwd(narrow, cached_conv_out=cached_conv_out, return_d_conv_state=return_d_conv_state):
          return compute_conv1d_bwd.conv1d_silu_bwd(
              qkv,
              w,
              bias,
              dy,
              _KERNEL_SIZE,
              conv_out=cached_conv_out,
              conv_state=state,
              return_d_conv_state=return_d_conv_state,
              segment_ids=seg,
              conv_halo_seg=halo,
              conv_input_in_qkv_dtype=narrow,
          )

        # Op by op: the same values in the same order, so every output is bit-identical.
        ref, got = bwd(False), bwd(True)
        self.assertEqual(len(ref), len(got))
        for r, g in zip(ref, got):
          if r is None:
            self.assertIsNone(g)
            continue
          self.assertEqual(r.dtype, g.dtype)
          self.assertEqual(_bits(r), _bits(g))
        # Compiled: dx, db and d_conv_state stay bit-identical; dw (index 1) is a float32 reduction whose order XLA
        # may change with the fusion, so it is compared to float32 summation-order tolerance.
        ref = jax.jit(lambda bwd=bwd: bwd(False))()
        got = jax.jit(lambda bwd=bwd: bwd(True))()
        for i, (r, g) in enumerate(zip(ref, got)):
          if r is None:
            self.assertIsNone(g)
            continue
          self.assertEqual(r.dtype, g.dtype)
          if i == 1:
            np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=2e-5, atol=1e-6)
          else:
            self.assertEqual(_bits(r), _bits(g))

  @parameterized.named_parameters(
      {"testcase_name": "no_state", "with_state": False},
      {"testcase_name": "state", "with_state": True},
  )
  def test_conv_input_is_narrow_only_when_enabled(self, with_state):
    """Positive control: the default path builds the [B, S + K - 1, D] input in float32, the option in bf16."""
    qkv, w, bias, state, dy, _, _ = _conv_inputs(3, with_state=with_state, with_bias=True, with_segments=False)
    shape = (qkv.shape[0], qkv.shape[1] + _KERNEL_SIZE - 1, qkv.shape[2])
    for fn in ("fwd", "bwd"):

      def run(narrow, fn=fn):
        if fn == "fwd":
          return compute_conv1d_bwd.conv1d_silu_fwd(
              qkv, w, bias, _KERNEL_SIZE, conv_state=state, conv_input_in_qkv_dtype=narrow
          )
        return compute_conv1d_bwd.conv1d_silu_bwd(
            qkv, w, bias, dy, _KERNEL_SIZE, conv_state=state, conv_input_in_qkv_dtype=narrow
        )

      off = _pad_or_concat_out_dtypes(jax.make_jaxpr(lambda run=run: run(False))(), shape)
      on = _pad_or_concat_out_dtypes(jax.make_jaxpr(lambda run=run: run(True))(), shape)
      # The backward also pads dz (float32) to the same shape; only the conv input's buffer changes dtype.
      self.assertNotIn(jnp.dtype(jnp.bfloat16), off, fn)
      self.assertEqual(on.count(jnp.dtype(jnp.bfloat16)), 1, fn)
      self.assertEqual(off.count(jnp.dtype(jnp.float32)), on.count(jnp.dtype(jnp.float32)) + 1, fn)

  def test_scope_sets_the_default(self):
    """conv_input_in_qkv_dtype=None follows conv_input_in_qkv_dtype_scope(); the default is off."""
    qkv, w, bias, state, _, _, _ = _conv_inputs(4, with_state=True, with_bias=True, with_segments=False)

    def jaxpr_text(narrow):
      return str(
          jax.make_jaxpr(
              lambda: compute_conv1d_bwd.conv1d_silu_fwd(
                  qkv, w, bias, _KERNEL_SIZE, conv_state=state, conv_input_in_qkv_dtype=narrow
              )
          )()
      )

    off, on = jaxpr_text(False), jaxpr_text(True)
    self.assertNotEqual(off, on)
    self.assertEqual(jaxpr_text(None), off)
    with compute_conv1d_bwd.conv_input_in_qkv_dtype_scope(True):
      self.assertEqual(jaxpr_text(None), on)
      self.assertEqual(jaxpr_text(False), off)  # an explicit argument wins over the scope
      with compute_conv1d_bwd.conv_input_in_qkv_dtype_scope(False):
        self.assertEqual(jaxpr_text(None), off)
      self.assertEqual(jaxpr_text(None), on)
    self.assertEqual(jaxpr_text(None), off)

  def test_gdn_decoupled_conv1d_grads_bit_identical(self):
    """Through the custom VJP: the same loss and gradients, and the option reaches the backward rule's trace."""
    gdn_bwd_pallas.ensure_cpu_interpret_registered()
    is_tpu = jax.default_backend() == "tpu"
    batch, chunk_size, num_chunks = 1, (64 if is_tpu else 32), 2
    seq_len = chunk_size * num_chunks
    num_k_heads, num_v_heads = 1, 2
    head_k_dim = head_v_dim = 128 if is_tpu else 64
    dim = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim
    keys = jax.random.split(jax.random.PRNGKey(5), 8)
    qkv = jax.random.normal(keys[0], (batch, seq_len, dim), jnp.float32).astype(jnp.bfloat16)
    b = jax.random.normal(keys[1], (batch, seq_len, num_v_heads), jnp.float32)
    a = jax.random.normal(keys[2], (batch, seq_len, num_v_heads), jnp.float32)
    conv_weight = jax.random.normal(keys[3], (_KERNEL_SIZE, 1, dim), jnp.float32)
    conv_bias = jax.random.normal(keys[4], (dim,), jnp.float32)
    a_log = jax.random.normal(keys[5], (num_v_heads,), jnp.float32)
    dt_bias = jax.random.normal(keys[6], (num_v_heads,), jnp.float32)
    do = jax.random.normal(keys[7], (batch, seq_len, num_v_heads, head_v_dim), jnp.float32)
    args = (qkv, b, a, conv_weight, conv_bias, a_log, dt_bias)

    def loss(narrow, qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
      out, _ = gdn_bwd_pallas.gdn_decoupled_conv1d(
          qkv=qkv_in,
          b=b_in,
          a=a_in,
          conv_weight=cw_in,
          conv_bias=cb_in,
          a_log=al_in,
          dt_bias=dt_in,
          conv_state=None,
          recurrent_state=None,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_k_dim,
          head_v_dim=head_v_dim,
          conv_kernel_size=_KERNEL_SIZE,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=True,
          compute_dtype=jnp.float32,
          conv_input_in_qkv_dtype=narrow,
      )
      return jnp.sum(out.astype(jnp.float32) * do)

    ref_val, ref_grads = jax.value_and_grad(lambda *x: loss(False, *x), argnums=tuple(range(7)))(*args)
    val, grads = jax.value_and_grad(lambda *x: loss(True, *x), argnums=tuple(range(7)))(*args)
    self.assertEqual(_bits(ref_val), _bits(val))
    for i, (r, g) in enumerate(zip(ref_grads, grads)):
      self.assertEqual(r.dtype, g.dtype)
      if i == 3:  # conv_weight: a float32 reduction over batch x sequence (see the module docstring)
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=2e-5, atol=1e-6)
      else:
        self.assertEqual(_bits(r), _bits(g))

    # The local forward convolves a float32 copy of qkv either way, so a bf16 [B, S + K - 1, D] conv input can only
    # come from the backward rule: present with the option, absent without it.
    shape = (batch, seq_len + _KERNEL_SIZE - 1, dim)
    grad_off = jax.make_jaxpr(jax.grad(lambda *x: loss(False, *x), argnums=tuple(range(7))))(*args)
    grad_on = jax.make_jaxpr(jax.grad(lambda *x: loss(True, *x), argnums=tuple(range(7))))(*args)
    self.assertNotIn(jnp.dtype(jnp.bfloat16), _pad_or_concat_out_dtypes(grad_off, shape))
    self.assertIn(jnp.dtype(jnp.bfloat16), _pad_or_concat_out_dtypes(grad_on, shape))

  def test_requires_gdn_kernel(self):
    def cfg(**overrides):
      fields = {
          "decoder_block": config_types.DecoderBlockType.QWEN3_5,
          "ici_context_parallelism": 1,
          "dcn_context_parallelism": 1,
          "ici_context_usp_ulysses_parallelism": 1,
          "dcn_context_usp_ulysses_parallelism": 1,
          "use_gdn_kernel": False,
          "gdn_conv1d_input_in_qkv_dtype": False,
          "gdn_gated_norm_layout": "auto",
          "gdn_in_proj_kernel_order": False,
      }
      fields.update(overrides)
      return types.SimpleNamespace(**fields)

    config_types.MaxTextConfig.validate_gdn_config(cfg(), is_rl_config=False)
    with self.assertRaisesRegex(ValueError, "gdn_conv1d_input_in_qkv_dtype=True requires use_gdn_kernel=True"):
      config_types.MaxTextConfig.validate_gdn_config(cfg(gdn_conv1d_input_in_qkv_dtype=True), is_rl_config=False)

  def test_config_default_is_off(self):
    self.assertIs(config_types.Qwen3Next.model_fields["gdn_conv1d_input_in_qkv_dtype"].default, False)


if __name__ == "__main__":
  absltest.main()
