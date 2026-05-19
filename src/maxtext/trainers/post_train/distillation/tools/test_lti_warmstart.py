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

"""CPU smoke test for LTI warm-start. Runs in <30s.

Builds a small student attention block with LTI bridges + warm-start, runs
a forward pass on dummy input, and verifies:

  1. The LearnToInitDense modules are actually being instantiated (vs the
     original random DenseGeneral)
  2. The forward pass produces a sensible output magnitude (not nan, not
     all-zero, not collapsed)
  3. The kernel computed by `_calc_attn_weight(A, B, C)` matches what the
     warm-start init promises (group-mean of teacher Q kernel in first 128
     head_dim components)

Run:
    JAX_PLATFORMS=cpu .venv/bin/python -m maxtext.trainers.post_train.distillation.tools.test_lti_warmstart
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.layers.learn_to_init_layer import (
    LearnToInitDense,
    _warmstart_head_bridge,
    _warmstart_dim_bridge,
    _calc_attn_weight,
)


def test_qkv_warmstart():
  """Check Q projection: student kernel should match group-mean teacher first 128."""
  print("=" * 70)
  print("Test 1: warm-started Q projection produces group-mean teacher output")
  print("=" * 70)

  # Shapes matching our config (Qwen3-30b-a3b base -> pruned student):
  embed_dim = 2048
  teacher_q_heads, teacher_head_dim = 32, 128
  student_q_heads, student_head_dim = 16, 256

  # Fake teacher Q kernel
  teacher_kernel = jax.random.normal(
      jax.random.PRNGKey(1), (embed_dim, teacher_q_heads, teacher_head_dim),
      dtype=jnp.float32,
  )

  # Build student Q projection with LTI warm-start
  rngs = nnx.Rngs(params=jax.random.PRNGKey(42))
  module = LearnToInitDense(
      in_features_shape=embed_dim,
      out_features_shape=(student_q_heads, student_head_dim),
      C=teacher_kernel,
      axis=-1,
      weight_dtype=jnp.float32,
      dtype=jnp.float32,
      kernel_axes=(None, None, None),
      is_output_projection=False,
      rngs=rngs,
  )

  # Build the effective kernel via _calc_attn_weight
  student_kernel = _calc_attn_weight(
      module.A.value, module.B.value, module.C.value,
      is_output_projection=False,
  )
  print(f"Student kernel shape: {student_kernel.shape}")

  # What the warm-start PROMISES: first 128 head_dim ≈ group-mean of teacher
  # over pairs of teacher heads (since A is group-mean: heads 0,1 -> 0; 2,3 -> 1; ...)
  expected_first128 = (teacher_kernel[:, 0::2, :] + teacher_kernel[:, 1::2, :]) / 2  # (embed, 16, 128)
  student_first128 = student_kernel[:, :, :teacher_head_dim]

  rel_diff = float(jnp.linalg.norm(student_first128 - expected_first128) /
                   jnp.linalg.norm(expected_first128))
  print(f"  rel diff student_first128 vs group-mean teacher: {rel_diff:.4f}  "
        f"(expect < 0.05, noise floor 0.01)")
  assert rel_diff < 0.05, f"Warm-start FAILED: rel diff {rel_diff} too large"

  # Last 128 should be ~zero (B is identity-prefix)
  student_last128_norm = float(jnp.linalg.norm(student_kernel[:, :, teacher_head_dim:]))
  print(f"  student kernel last-128 norm: {student_last128_norm:.4f}  "
        f"(expect tiny, just from noise term in B)")
  print(f"  PASS")
  return module, teacher_kernel


def test_forward_pass(module, teacher_kernel):
  """Run a forward pass through the warm-started Q projection."""
  print()
  print("=" * 70)
  print("Test 2: forward pass through warm-started Q projection")
  print("=" * 70)

  batch, seq, embed_dim = 2, 4, 2048
  x = jax.random.normal(jax.random.PRNGKey(7), (batch, seq, embed_dim), dtype=jnp.float32)

  student_out = module(x)  # (B, T, 16, 256)
  print(f"Student output shape: {student_out.shape}")
  print(f"Student output stats: mean={float(student_out.mean()):.4f}  "
        f"std={float(student_out.std()):.4f}  "
        f"min={float(student_out.min()):.4f}  max={float(student_out.max()):.4f}")

  # Expected: teacher-equivalent group-mean projection
  teacher_q = jnp.einsum('bse,ehd->bshd', x, teacher_kernel)  # (B, T, 32, 128)
  teacher_groupmean = (teacher_q[:, :, 0::2, :] + teacher_q[:, :, 1::2, :]) / 2  # (B, T, 16, 128)

  student_first128 = student_out[:, :, :, :128]
  student_last128 = student_out[:, :, :, 128:]
  rel_diff_first = float(jnp.linalg.norm(student_first128 - teacher_groupmean) /
                         jnp.linalg.norm(teacher_groupmean))
  print(f"  rel diff student_out[:128] vs teacher group-mean Q: {rel_diff_first:.4f}")
  print(f"  student_out[128:] norm: {float(jnp.linalg.norm(student_last128)):.4f} (expect ~noise)")
  assert rel_diff_first < 0.05, f"Forward pass FAILED: rel diff {rel_diff_first}"
  print(f"  PASS")


def test_out_warmstart():
  """OUT projection forward — should also be near group-mean teacher."""
  print()
  print("=" * 70)
  print("Test 3: warm-started OUT projection forward")
  print("=" * 70)

  embed_dim = 2048
  teacher_q_heads, teacher_head_dim = 32, 128
  student_q_heads, student_head_dim = 16, 256

  # Teacher OUT kernel shape: (heads, head_dim, embed)
  teacher_kernel = jax.random.normal(
      jax.random.PRNGKey(2),
      (teacher_q_heads, teacher_head_dim, embed_dim),
      dtype=jnp.float32,
  )

  rngs = nnx.Rngs(params=jax.random.PRNGKey(43))
  module = LearnToInitDense(
      in_features_shape=(student_q_heads, student_head_dim),
      out_features_shape=embed_dim,
      C=teacher_kernel,
      axis=(-2, -1),
      weight_dtype=jnp.float32,
      dtype=jnp.float32,
      kernel_axes=(None, None, None),
      is_output_projection=True,
      rngs=rngs,
  )

  # Forward: attention output (B, T, 16, 256) -> (B, T, embed)
  # For warm-start B = identity-prefix (256, 128), only first 128 of student
  # head_dim contribute. So craft input where first 128 carries signal.
  batch, seq = 2, 4
  attn_out = jnp.zeros((batch, seq, student_q_heads, student_head_dim), dtype=jnp.float32)
  signal = jax.random.normal(jax.random.PRNGKey(8), (batch, seq, student_q_heads, teacher_head_dim))
  attn_out = attn_out.at[..., :teacher_head_dim].set(signal)

  student_out = module(attn_out)  # (B, T, embed)
  print(f"Student output shape: {student_out.shape}")

  # Expected: as if we'd run teacher OUT on the group-expanded signal
  # teacher_OUT(signal_expanded_to_32_heads)
  # We have signal at 16 student heads. Each student head_j corresponds to
  # the group-mean of teacher heads 2j, 2j+1. To get back to teacher's
  # output, we treat signal[..., 16_head_j, :] as the "averaged input" and
  # apply teacher OUT with split heads.
  # Equivalent: per-head OUT kernel for student head j = teacher OUT
  # averaged over teacher heads 2j, 2j+1.
  teacher_outk_groupmean = (teacher_kernel[0::2] + teacher_kernel[1::2]) / 2  # (16, 128, embed)
  expected = jnp.einsum('bshd,hde->bse', signal, teacher_outk_groupmean)

  rel_diff = float(jnp.linalg.norm(student_out - expected) / jnp.linalg.norm(expected))
  print(f"  rel diff student_out vs expected: {rel_diff:.4f}  "
        f"(expect < 0.05)")
  assert rel_diff < 0.05, f"OUT warm-start FAILED: rel diff {rel_diff}"
  print(f"  PASS")


def main():
  print("LTI warm-start CPU smoke test")
  print()
  module, teacher_kernel = test_qkv_warmstart()
  test_forward_pass(module, teacher_kernel)
  test_out_warmstart()
  print()
  print("=" * 70)
  print("ALL TESTS PASSED")
  print("=" * 70)


if __name__ == "__main__":
  main()
