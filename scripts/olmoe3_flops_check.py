"""olmoe3-3p5b training FLOPs per token from the real parameter shapes, against MaxText's formula and the reference report.

  JAX_PLATFORMS=cpu python scripts/olmoe3_flops_check.py
"""
import os, sys, re
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests", "unit"))
import jax, numpy as np
import olmoe3_test as T
from maxtext.utils import maxtext_utils
for seq in (4096, 8192):
  cfg = T._config("olmoe3-3p5b", max_target_length=seq, extra=("dtype=bfloat16", "override_model_config=True", "gdn_chunk_size=128"))
  _, model = T._build(cfg)
  params = T._abstract_params(cfg, model, seq)
  flat = jax.tree_util.tree_flatten_with_path(params)[0]
  E, K = cfg.num_experts, cfg.num_experts_per_tok
  tot = act_mm = emb_tab = expert = 0
  rows = {}
  for path, leaf in flat:
    name = "/".join(str(getattr(p, "key", getattr(p, "idx", p))) for p in path)
    n = int(np.prod(leaf.shape)); tot += n
    if leaf.ndim < 2: continue  # norms, biases, A_log, dt_bias: no matmul
    if "token_embedder" in name: emb_tab += n; continue  # gather, not matmul
    if "conv" in name: continue  # depthwise conv: counted separately below
    if ("MoeBlock" in name or "moe" in name.lower()) and leaf.ndim == 3 and leaf.shape[0] == E:
      expert += n; act_mm += n * K / E; key = "routed experts (x16/512)"
    else:
      act_mm += n; key = re.sub(r"layers_\d+", "layers_N", name).split("/")[1] if "/" in name else name
    rows[key] = rows.get(key, 0) + (n * K / E if "routed" in key else n)
  B = seq
  # Non-weight math per token, forward:
  L_attn = cfg.num_decoder_layers // cfg.inhomogeneous_layer_cycle_interval
  L_kda = cfg.num_decoder_layers - L_attn
  attn_noncausal = 4 * seq * cfg.num_query_heads * cfg.head_dim * L_attn
  h, dk, dv, C = cfg.gdn_num_value_heads, cfg.gdn_key_head_dim, cfg.gdn_value_head_dim, cfg.gdn_chunk_size
  kda_state = 2 * 3 * h * dk * dv * L_kda  # delta = u - w S, S += k^T delta, out = q S
  kda_intra = 2 * h * C * (dk + dk + dv + dv + dk) * L_kda  # q k^T, k k^T, inverse x [v|k], scores x delta (C/2 avg would halve)
  conv = 2 * cfg.gdn_conv_kernel_dim * (2 * h * dk + h * dv) * L_kda
  weight = 2 * act_mm
  mt_attn, mt_learn = maxtext_utils.calculate_olmoe3_tflops_training_per_device(cfg, 2 * B * cfg.emb_dim * cfg.vocab_size)
  g = lambda x: x / 1e9
  print(f"== seq {seq}: total params {tot/1e9:.3f} B, routed expert params {expert/1e9:.2f} B, embedding table {emb_tab/1e6:.1f} M")
  print(f"   active matmul params {act_mm/1e9:.4f} B  -> 6N = {g(3*weight):.3f} GFLOP/token")
  for k, v in sorted(rows.items(), key=lambda kv: -kv[1])[:8]: print(f"      {v/1e6:9.1f} M  {k}")
  print(f"   attention core ({L_attn} layers): causal {g(3*attn_noncausal/2):.3f}, non-causal {g(3*attn_noncausal):.3f} GFLOP/token")
  print(f"   KDA core ({L_kda} layers): state {g(3*kda_state):.3f}, intra-chunk {g(3*kda_intra):.3f}, conv {g(3*conv):.4f} GFLOP/token")
  ours_c = 3 * (weight + attn_noncausal / 2 + kda_state + kda_intra + conv)
  ours_n = 3 * (weight + attn_noncausal + kda_state + kda_intra + conv)
  print(f"   first principles: causal attn {g(ours_c):.3f}, non-causal attn {g(ours_n):.3f} GFLOP/token")
  print(f"   MaxText formula: learnable {mt_learn*1e12/B/1e9:.3f} + attention {mt_attn*1e12/B/1e9:.3f} = {(mt_learn+mt_attn)*1e12/B/1e9:.3f} GFLOP/token ({(mt_learn+mt_attn):.2f} TF/step)")
  print(f"   reference report: {(91.14 if seq == 4096 else 184.75)*1e12/B/1e9:.3f} GFLOP/token")
