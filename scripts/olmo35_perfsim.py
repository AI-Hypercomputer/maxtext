"""perfsim estimates for the OLMo 3.5 partner family on Ironwood.

The four rungs from allenai/OLMo-core@codex/partner-model-family-20260914,
src/scripts/standalone/standalone_configs.py. perfsim has no preset for them, so
each is built from the shipped olmoe3_3p5b preset with the family's geometry:
head_dim 128, latent = d_model/2 (moe_input_bottleneck 2.0), expert hidden =
d_model, 512 experts top-16, and a full-attention layer every 8th block.

Ratios only. perfsim assumes perfect overlap and prices the KDA cumsum as an MXU
node; on our measured OLMoE3 it predicts 1339 ms against 1752 ms measured, and
puts KDA at 27% of step against 6% measured. Treat step time as optimistic and
compare rungs against each other, not against a target.

  PYTHONPATH=/home/agagik_google_com/olmo35/perfsim:/home/agagik_google_com/olmo35/perfsim-deps \
    python3 scripts/olmo35_perfsim.py
"""

import argparse
import dataclasses

from perfsim.core.engine.inference import driver as idriver
from perfsim.core.engine.inference.config import InferenceParConfig, ServingConfig
from perfsim.core.engine.training import driver as tdriver
from perfsim.core.engine.training.config import ParallelismConfig, TrainingConfig
from perfsim.plugins.hardware import HW_PRESETS
from perfsim.plugins.models.presets import MODEL_PRESETS

# name -> (d_model, n_layers, n_heads, n_kv_heads, total_params, active_params)
FAMILY = {
    "tiny": (1024, 16, 8, 4, 12_496_341_632, 794_233_472),
    "small": (1536, 40, 16, 8, 72_237_847_936, 3_780_515_200),
    "medium": (2560, 64, 24, 12, 322_601_566_720, 15_421_227_520),
    "large": (4608, 80, 48, 24, 1_310_163_554_560, 62_133_719_296),
}

# Training slice per rung. Weights plus fp32 master and Adam moments are 14
# bytes per parameter, sharded over the FSDP mesh, so the slice has to grow with
# total params, not with active params.
TRAIN_HW = {"tiny": "v7x_4x4x4", "small": "v7x_4x4x4", "medium": "v7x_4x8x8", "large": "v7x_8x8x16"}

# Serving plan per rung: (tp, ep). tp always divides num_kv_heads, otherwise the
# KV cache silently replicates (see kda-headdim-answer.md). ep spreads the
# routed experts so the resident weights fit.
SERVE_PLAN = {"tiny": (4, 1), "small": (8, 1), "medium": (4, 8), "large": (8, 8)}

# Expert-parallel degrees to try for training. Pure FSDP (ep=1) all-gathers every
# expert weight every step, which dominates the step at these total-parameter
# counts, so ep is the first thing to sweep rather than an afterthought.
EP_SWEEP = (1, 2, 4, 8, 16, 32)


def build(name):
  d, layers, heads, kv_heads, _, _ = FAMILY[name]
  return dataclasses.replace(
      MODEL_PRESETS["olmoe3_3p5b"],
      name=f"olmo35_{name}",
      hidden_dim=d,
      num_layers=layers,
      num_q_heads=heads,
      num_kv_heads=kv_heads,
      head_dim=128,
      expert_hidden_dim=d,
      shared_expert_hidden_dim=d,
      dense_ffn_hidden_dim=8 * d,
      num_dense_layers=1,
      full_attention_interval=8,
      linear_num_key_heads=heads,
      linear_num_value_heads=heads,
      linear_key_head_dim=128,
      linear_value_head_dim=256,
      moe_input_bottleneck=2.0,  # latent = d_model / 2
  )


def chips(hw):
  n = 1
  for dim in hw.slice_shape:
    n *= dim
  return n


def categorize(name):
  if name.startswith(("LinAttn", "GDN")):
    return "kda"
  if name.startswith(("FSDP", "AllReduce", "AllGather", "ReduceScatter", "A2A", "TP_", "EP_")):
    return "comm"
  if "Attn" in name or name.startswith(("Flash", "KV", "Q_proj", "O_proj", "MLA")):
    return "attn"
  if name.startswith(("Expert", "MoE", "Router", "Shared")):
    return "moe"
  return "other"


def call(x):
  return x() if callable(x) else x


def shares(node):
  out = {}
  for leaf in call(node.flat_leaves):
    out[categorize(leaf.name)] = out.get(categorize(leaf.name), 0.0) + call(leaf.total_time_us) / 1e3
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--seq", type=int, default=8192)
  # The spec pins only Tiny's global batch (16,777,216 tokens = 2048 sequences);
  # the other rungs are TODO. Holding it across rungs keeps them comparable and
  # matches the one recipe number that is actually validated.
  ap.add_argument("--batch-tokens", type=int, default=16_777_216)
  ap.add_argument("--pdb", type=int, default=1, help="unused; batch is set by --batch-tokens")
  ap.add_argument("--decode-ctx", type=int, default=8192)
  ap.add_argument("--decode-batch", type=int, default=64)
  args = ap.parse_args()

  def train(name, hw, devices, ep):
    model = build(name)
    seqs = max(1, args.batch_tokens // args.seq)
    tr = TrainingConfig(global_batch_size=seqs, seq_len=args.seq, remat_policy="full")
    par = ParallelismConfig(
        tp=1,
        ep_mesh=(ep,),
        cp_mesh=(1,),
        fsdp_mesh=(devices // ep),
        fsdp_transpose_mesh=(1,),
        dp_mesh=(1,),
        pp_mesh=(1,),
        pp_microbatches=1,
    )
    if not isinstance(par.fsdp_mesh, tuple):
      par = dataclasses.replace(par, fsdp_mesh=(devices // ep,))
    r = tdriver.run(hw, model, tr, par)
    ms = call(r.total_time_us) / 1e3
    # perfsim reports MACs per device, so conventional FLOPs are 2x, and the
    # relevant peak is per device (a chip carries two).
    tflops = 2 * call(r.total_flops) / (ms / 1e3) / 1e12
    return ms, tflops, tflops / (hw.bf16_tflops_per_chip / hw.d2d_num_chips), shares(r)

  print(
      f"TRAINING  seq={args.seq} global batch {args.batch_tokens:,} tokens "
      f"({args.batch_tokens // args.seq} sequences) remat=full. Expert parallelism swept."
  )
  print(
      f"  {'rung':<8}{'active':>9}{'total':>9}{'slice':>14}{'dev':>6}{'ep':>4}{'step s':>9}"
      f"{'TF/s/dev':>9}{'MFU':>7}{'tok/s/dev':>11}{'kda%':>7}{'moe%':>7}{'comm%':>7}"
  )
  for name, (_, _, _, _, total, active) in FAMILY.items():
    hw = HW_PRESETS[TRAIN_HW[name]]
    devices = chips(hw) * hw.d2d_num_chips
    results = {}
    for ep in EP_SWEEP:
      if devices % ep:
        continue
      try:
        results[ep] = train(name, hw, devices, ep)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"  {name:<8}{'':>38}{ep:>4}   {' '.join(str(exc).split())[:60]}")
    if not results:
      continue
    best_ep = min(results, key=lambda e, r=results: r[e][0])
    for ep in sorted({1, best_ep}):
      ms, tflops, mfu, sh = results[ep]
      tokens = args.batch_tokens
      tag = "  <- best" if ep == best_ep and best_ep != 1 else ""
      head = (
          f"  {name:<8}{active/1e9:8.2f}B{total/1e9:8.1f}B{TRAIN_HW[name]:>14}{devices:>6}"
          if ep == 1
          else f"  {'':<8}{'':>9}{'':>9}{'':>14}{'':>6}"
      )
      print(
          f"{head}{ep:>4}{ms/1e3:9.3f}{tflops:9.0f}{mfu*100:6.1f}%"
          f"{tokens/(ms/1e3)/devices:11.0f}{sh.get('kda',0)/ms*100:6.1f}%"
          f"{sh.get('moe',0)/ms*100:6.1f}%{sh.get('comm',0)/ms*100:6.1f}%{tag}"
      )

  print(f"\nDECODE  ctx={args.decode_ctx} batch={args.decode_batch}")
  print(
      f"  {'rung':<8}{'tp':>4}{'ep':>4}{'dev':>6}{'step ms':>10}{'tok/s':>10}"
      f"{'tok/s/dev':>11}{'kda ms':>9}{'attn ms':>9}{'moe ms':>9}"
  )
  for name in FAMILY:
    tp, ep = SERVE_PLAN[name]
    model = build(name)
    sv = ServingConfig(
        batch_size=args.decode_batch, context_len=args.decode_ctx, prompt_len=min(512, args.decode_ctx // 2)
    )
    try:
      r = idriver.run_decode(HW_PRESETS["v7x_4x4x4"], model, sv, InferenceParConfig(tp=tp, ep=ep))
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"  {name:<8}{tp:>4}{ep:>4}   {' '.join(str(exc).split())[:80]}")
      continue
    ms = call(r.total_time_us) / 1e3
    sh = shares(r)
    devices = tp * ep
    tps = args.decode_batch / (ms / 1e3)
    print(
        f"  {name:<8}{tp:>4}{ep:>4}{devices:>6}{ms:10.2f}{tps:10.0f}{tps/devices:11.1f}"
        f"{sh.get('kda',0):9.2f}{sh.get('attn',0):9.2f}{sh.get('moe',0):9.2f}"
    )


if __name__ == "__main__":
  main()
