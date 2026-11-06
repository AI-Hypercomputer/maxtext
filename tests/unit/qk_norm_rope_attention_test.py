# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end check of `fused_qk_norm_rope` in the `Attention` layer: flag on == flag off (output and grads)."""

# pylint: disable=too-many-positional-arguments

import sys

from flax import nnx
import jax
from jax.sharding import Mesh
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.common.common_types import MODEL_MODE_TRAIN, AttentionType
from maxtext.configs import pyconfig
from maxtext.layers.attentions import Attention
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_ON_TPU = jax.default_backend() == "tpu"
_SEQ = 512


def _config(fused, dtype, attention, matmul_precision):
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      run_name="test",
      enable_checkpointing=False,
      model_name="gemma4-26b",
      per_device_batch_size=1.0,
      max_target_length=_SEQ,
      dtype=dtype,
      attention=attention,
      sa_block_q=128,
      sa_block_kv=128,
      sa_block_kv_compute=128,
      sa_block_q_dkv=128,
      sa_block_kv_dkv=128,
      sa_block_kv_dkv_compute=128,
      sa_block_q_dq=128,
      sa_block_kv_dq=128,
      matmul_precision=matmul_precision,
      fused_qk_norm_rope=fused,
  )


def _attention(cfg, mesh, layer):
  """Mirrors the Gemma 4 global / local attention construction in models/gemma4.py."""
  batch = cfg.global_batch_size_to_train_on
  if layer == "global":
    kwargs = {
        "num_kv_heads": cfg.global_num_kv_heads,
        "head_dim": cfg.global_head_dim,
        "attention_type": AttentionType.GLOBAL,
        "partial_rotary_factor": cfg.global_rope_proportion,
        "rope_max_timescale": cfg.global_rope_max_timescale,
        "share_kv_projections": True,
    }
  else:
    kwargs = {
        "num_kv_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "attention_type": AttentionType.LOCAL_SLIDING,
        "partial_rotary_factor": cfg.local_rope_proportion,
        "rope_max_timescale": cfg.local_rope_max_timescale,
        "sliding_window_size": cfg.sliding_window_size,
    }
  return Attention(
      config=cfg,
      num_query_heads=cfg.num_query_heads,
      max_target_length=cfg.max_target_length,
      max_prefill_predict_length=cfg.max_prefill_predict_length,
      attention_kernel=cfg.attention,
      inputs_q_shape=(batch, _SEQ, cfg.emb_dim),
      inputs_kv_shape=(batch, _SEQ, cfg.emb_dim),
      mesh=mesh,
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      dropout_rate=0.0,
      use_qk_norm=True,
      use_v_norm=True,
      query_pre_attn_scalar=1.0,
      model_mode=MODEL_MODE_TRAIN,
      rngs=nnx.Rngs(params=0, dropout=1),
      **kwargs,
  )


def _rel_l2(actual, expected):
  a = np.asarray(actual, np.float32)
  e = np.asarray(expected, np.float32)
  return float(np.linalg.norm(a - e) / max(np.linalg.norm(e), 1e-6))


def _run_one(cfg, mesh, layer, lnx, positions, segment_ids, dy):
  """Output, dL/dx and dL/dparams of one `Attention` layer under `cfg`."""
  module = _attention(cfg, mesh, layer)
  graphdef, params, rest = nnx.split(module, nnx.Param, ...)

  def loss(params, lnx):
    model = nnx.merge(graphdef, params, rest)
    out, _ = model(lnx, lnx, inputs_positions=positions, decoder_segment_ids=segment_ids, deterministic=True)
    return jnp.sum(out.astype(jnp.float32) * dy), out

  with mesh:
    (_, out), (dstate, dlnx) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(params, lnx)
  return [out, dlnx] + jax.tree.leaves(dstate)


def _run(layer, dtype, attention, matmul_precision, rel_l2, bf16_noise_ratio=None):
  """Builds the layer with the flag off/on (same init) and compares output, dL/dx and dL/dparams.

  With `bf16_noise_ratio`, a third f32/highest-precision run is the truth: the fused run's distance
  to it must be at most `bf16_noise_ratio` times the unfused bf16 run's distance (bf16 noise floor).
  """
  jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)
  cfg_ref = _config(False, dtype, attention, matmul_precision)
  cfg_fused = _config(True, dtype, attention, matmul_precision)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)
  batch = cfg_ref.global_batch_size_to_train_on
  k1, k2 = jax.random.split(jax.random.PRNGKey(0))
  lnx = jax.random.normal(k1, (batch, _SEQ, cfg_ref.emb_dim), jnp.float32).astype(cfg_ref.dtype)
  positions = jnp.broadcast_to(jnp.arange(_SEQ, dtype=jnp.int32)[None], (batch, _SEQ))
  segment_ids = jnp.ones((batch, _SEQ), jnp.int32)
  dy = jax.random.normal(k2, (batch, _SEQ, cfg_ref.emb_dim), jnp.float32)
  args = (mesh, layer, lnx, positions, segment_ids, dy)

  ref = _run_one(cfg_ref, *args)
  fused = _run_one(cfg_fused, *args)
  assert len(ref) == len(fused) > 2
  errs = [_rel_l2(f, r) for f, r in zip(fused, ref)]
  if bf16_noise_ratio is None:
    assert max(errs) <= rel_l2, f"rel L2 errors (out, dx, params...): {errs}"
    return
  truth = _run_one(_config(False, "float32", "dot_product", "highest"), mesh, layer, lnx, positions, segment_ids, dy)
  err_fused = [_rel_l2(f, t) for f, t in zip(fused, truth)]
  err_ref = [_rel_l2(r, t) for r, t in zip(ref, truth)]
  print(f"\n[{layer}] rel L2 fused-vs-unfused {errs}\n  fused-vs-f32 {err_fused}\n  unfused-vs-f32 {err_ref}")
  assert max(errs) <= rel_l2, f"rel L2 fused vs unfused: {errs}"
  for e_f, e_r in zip(err_fused, err_ref):
    assert e_f <= bf16_noise_ratio * e_r + 1e-4, f"fused farther from f32 than XLA bf16: {err_fused} vs {err_ref}"


@pytest.mark.tpu_only
@pytest.mark.skipif(not _ON_TPU, reason="TPU only")
@pytest.mark.parametrize("layer", ["global", "local"])
def test_attention_fused_qk_norm_rope_f32_dot_product(layer):
  """f32 activations, full-precision matmuls, generic attention op: fused == unfused to ~1e-4."""
  _run(layer, "float32", "dot_product", "highest", rel_l2=1e-4)


@pytest.mark.tpu_only
@pytest.mark.skipif(not _ON_TPU, reason="TPU only")
@pytest.mark.parametrize("layer", ["global", "local"])
def test_attention_fused_qk_norm_rope_bf16_flash(layer):
  """bf16 + splash attention (the training configuration): fused == unfused up to bf16 rounding noise."""
  _run(layer, "bfloat16", "flash", "default", rel_l2=1e-1, bf16_noise_ratio=1.2)
