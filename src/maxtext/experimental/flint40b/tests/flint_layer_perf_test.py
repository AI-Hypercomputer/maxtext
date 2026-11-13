# Copyright 2026 Google LLC
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

"""Real 40B-width 4-layer MoE superblock latency & HBM audit on 4 TPU v7x chips."""

from __future__ import annotations

import json
import time
from typing import Any

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_layers_scan
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import flint_types

P = PartitionSpec
_ALL_AXES = flint_config.ALL_AXES


def _superblock_tflops_per_tc(cfg: flint_config.FlintConfig) -> float:
  """Computes exact fwd+bwd TFLOPs/TC for one 4-layer `(3 slide + 1 global)` SB."""
  m = cfg.model
  pdb = cfg.train.global_batch_size / cfg.sharding.num_devices
  seq_len = cfg.train.seq_len
  d = m.hidden_size
  q_dim = m.num_attention_heads * m.head_dim
  kv_dim = m.num_key_value_heads * m.head_dim

  attn_matmul = 2 * d * q_dim + 2 * d * kv_dim
  moe_active_matmul = (
      d * m.num_experts
      + 3 * d * m.shared_expert_intermediate_size
      + m.num_experts_per_tok * (3 * d * m.moe_intermediate_size)
  )
  sb_active_matmul = 4 * (attn_matmul + moe_active_matmul)
  weight_tflops = 6.0 * pdb * seq_len * sb_active_matmul / 1e12

  eff_w = min(m.sliding_window, seq_len)
  sliding_pairs = seq_len * eff_w - 0.5 * eff_w * (eff_w - 1)
  attn_flops_per_seq = (1 * 6.0 * (seq_len**2) + 3 * 12.0 * sliding_pairs) * (m.num_attention_heads * m.head_dim)
  attn_tflops = pdb * attn_flops_per_seq / 1e12
  return weight_tflops + attn_tflops


class FlintLayerPerfTest(absltest.TestCase):

  def _run_superblock_perf(self, *, label: str, **extra_overrides: Any) -> dict[str, Any]:
    # 9-layer (`2 dense + 1x4 superblock + 3 tail`) 40B-width config so
    # `scanned_blocks[0]` yields an unstacked 40B-width 4-layer superblock.
    cfg = flint_config.flint_40b_config(
        num_hidden_layers=9,
        num_dense_prefix_layers=2,
        fsdp_size=8,
        cp_size=1,
        ep_size=1,
        global_batch_size=8,
        seq_len=2048,
        use_splash=True,
        compute_dtype="bfloat16",
        param_dtype="float32",
        remat_policy="layer",
        scan_layers=True,
        **extra_overrides,
    )
    mesh = flint_sharding.create_mesh(cfg.sharding)
    full_params = flint_sharding.init_random_params(jax.random.key(0), mesh, cfg)
    assert full_params.decoder.scanned_blocks is not None
    sb_weights: flint_types.FlintSuperblockWeights[jax.Array] = jax.tree.map(
        lambda a: a[0], full_params.decoder.scanned_blocks
    )
    del full_params

    single_sparse_spec = flint_sharding.param_pspecs(cfg).decoder.tail_moe[0]
    sb_specs = flint_types.FlintSuperblockWeights(
        sliding_0=single_sparse_spec,
        sliding_1=single_sparse_spec,
        sliding_2=single_sparse_spec,
        global_3=single_sparse_spec,
    )
    sb_shardings = jax.tree.map(lambda s: NamedSharding(mesh, s), sb_specs)
    sb_weights = jax.device_put(sb_weights, sb_shardings)

    act_spec = flint_sharding.activation_pspec()
    b_spec = flint_sharding.batch_pspec()
    act_sh = NamedSharding(mesh, act_spec)
    tok_sh = NamedSharding(mesh, b_spec)

    bsz, seq_len, hidden_size = (
        cfg.train.global_batch_size,
        cfg.train.seq_len,
        cfg.model.hidden_size,
    )
    x_in = jax.jit(
        lambda k: jax.random.normal(k, (bsz, seq_len, hidden_size), dtype=jnp.bfloat16),
        out_shardings=act_sh,
    )(jax.random.key(1))
    up_grad = jax.jit(
        lambda k: jax.random.normal(k, (bsz, seq_len, hidden_size), dtype=jnp.bfloat16),
        out_shardings=act_sh,
    )(jax.random.key(2))

    pos_np = np.broadcast_to(np.arange(seq_len, dtype=np.int32)[None, :], (bsz, seq_len)).copy()
    seg_np = np.ones((bsz, seq_len), dtype=np.int32)
    msk_np = np.ones((bsz, seq_len), dtype=np.int32)
    pos_sh = jax.device_put(pos_np, tok_sh)
    seg_sh = jax.device_put(seg_np, tok_sh)
    msk_sh = jax.device_put(msk_np, tok_sh)

    empty_spec = P()

    def _local_sb_fwd_bwd(w_l, x_l, pos_l, seg_l, msk_l, up_l):
      def _loss(w_in, x_in):
        y_l, _, _ = flint_layers_scan.run_superblock(x_in, w_in, pos_l, seg_l, msk_l, cfg)
        return jnp.sum(y_l.astype(jnp.float32) * up_l.astype(jnp.float32)) / float(bsz * seq_len)

      loc_loss, (dw_l, dx_l) = jax.value_and_grad(_loss, argnums=(0, 1))(w_l, x_l)
      dw_l = jax.tree.map(
          lambda g, sp: (jax.lax.psum(g.astype(jnp.float32), _ALL_AXES).astype(g.dtype) if sp == empty_spec else g),
          dw_l,
          sb_specs,
      )
      return jax.lax.psum(loc_loss, _ALL_AXES), dw_l, dx_l

    jitted_step = jax.jit(
        jax.shard_map(
            _local_sb_fwd_bwd,
            mesh=mesh,
            in_specs=(sb_specs, act_spec, b_spec, b_spec, b_spec, act_spec),
            out_specs=(P(), sb_specs, act_spec),
            check_vma=False,
        ),
        in_shardings=(sb_shardings, act_sh, tok_sh, tok_sh, tok_sh, act_sh),
        out_shardings=(NamedSharding(mesh, P()), sb_shardings, act_sh),
    )

    compiled = jitted_step.lower(sb_weights, x_in, pos_sh, seg_sh, msk_sh, up_grad).compile()
    mem = compiled.memory_analysis()
    self.assertIsNotNone(mem)
    assert mem is not None
    hbm_bytes = (
        int(mem.temp_size_in_bytes)
        + int(mem.argument_size_in_bytes)
        + int(mem.output_size_in_bytes)
        - int(mem.alias_size_in_bytes)
    )
    hbm_gib = hbm_bytes / float(1024**3)
    self.assertGreater(hbm_gib, 0.0)
    self.assertLess(hbm_gib, 24.0)

    for _ in range(2):
      out_warmup = compiled(sb_weights, x_in, pos_sh, seg_sh, msk_sh, up_grad)
      jax.block_until_ready(out_warmup)

    step_ms_samples: list[float] = []
    last_loss = 0.0
    for _ in range(5):
      t0 = time.perf_counter()
      out_timed = compiled(sb_weights, x_in, pos_sh, seg_sh, msk_sh, up_grad)
      jax.block_until_ready(out_timed)
      elapsed_ms = (time.perf_counter() - t0) * 1000.0
      step_ms_samples.append(elapsed_ms)
      last_loss = float(out_timed[0])

    self.assertTrue(np.isfinite(last_loss))
    median_step_ms = float(np.median(step_ms_samples))
    self.assertGreater(median_step_ms, 0.0)
    self.assertLess(median_step_ms, 5000.0)

    sb_tflops = _superblock_tflops_per_tc(cfg)
    tflops_per_sec_per_tc = sb_tflops / (median_step_ms / 1000.0)
    self.assertGreater(tflops_per_sec_per_tc, 0.0)

    summary = {
        "benchmark": label,
        "use_fp8": cfg.kernels.use_fp8,
        "num_devices": 8,
        "global_batch_size": bsz,
        "seq_len": seq_len,
        "median_step_ms": round(median_step_ms, 4),
        "hbm_gib": round(hbm_gib, 4),
        "superblock_tflops_per_tc": round(sb_tflops, 6),
        "tflops_per_sec_per_tc": round(tflops_per_sec_per_tc, 4),
        "step_ms_samples": [round(v, 4) for v in step_ms_samples],
    }
    print("FLINT_LAYER_PERF_RECEIPT=" + json.dumps(summary), flush=True)
    return summary

  def test_real_40b_width_superblock_memory_and_step_latency(self):
    self.assertEqual(jax.default_backend(), "tpu")
    self.assertLen(jax.devices(), 8)
    self._run_superblock_perf(label="flint_40b_superblock_4layer_tpu_v7x_4_bf16")
    self._run_superblock_perf(
        label="flint_40b_superblock_4layer_tpu_v7x_4_fp8_opt",
        moe_impl="pallas_gmm_v2",
        sc_collective_mode="dual_sc",
        routed_w_tiled=True,
        moe_bijective_permute=True,
        moe_slot_major_pack=True,
        fuse_layer_projections=True,
        replicate_router_gate=True,
        prefetch_routed_weights=True,
        transpose_routed_down_ag=True,
        transpose_dense_rhs_ag=True,
        moe_zero_scatter_router=True,
        use_fp8=True,
    )


if __name__ == "__main__":
  absltest.main()
