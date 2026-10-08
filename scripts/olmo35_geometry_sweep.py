"""What geometry would clear the MFU target for OLMo 3.5? (perfsim)

Phase 6.5 of olmo35-ironwood-plan.md. The measured numbers say the OLMo 3.5 MoE
sits far below its dense siblings (olmo3-7b 24-27.6%, olmo3-32b 30.3%, all on the
same stack and slice). This prices the architectural component of that gap instead
of asserting it.

The organising identity is

    rows per expert per device = seq * pdb * top_k / num_experts

which at the shipped geometry (512 experts, top-16) is exactly one MXU tile of 256
rows at pdb=1, so the M dimension of every expert GEMM never amortises. Active
expert FLOPs per token are `top_k * 3 * latent * expert_hidden`, so granularity can
be traded at **iso active FLOPs**: halving top_k and doubling expert hidden leaves
the compute per token unchanged while making each expert GEMM twice as large.

Arms are iso-active by construction except where marked. Ratios only; perfsim's
absolute optimism is regime-dependent (measured 0.62x at 8 devices, ~1.7x at 128).

  PYTHONPATH=/home/agagik_google_com/olmo35/perfsim:/home/agagik_google_com/olmo35/perfsim-deps \
    python3 scripts/olmo35_geometry_sweep.py --rung small
"""

import argparse
import dataclasses

from perfsim.core.engine.training import driver as tdriver
from perfsim.core.engine.training.config import ParallelismConfig, TrainingConfig
from perfsim.plugins.hardware import HW_PRESETS

from olmo35_perfsim import FAMILY, build, call, chips


def arms(d):
  """(label, overrides, note). Active expert FLOPs/token = top_k*3*latent*eh."""
  return [
      ("base 512e top16 eh=d", {}, "shipped"),
      ("256e top8 eh=2d", {"num_experts": 256, "top_k": 8, "expert_hidden_dim": 2 * d}, "iso-active"),
      ("128e top8 eh=2d", {"num_experts": 128, "top_k": 8, "expert_hidden_dim": 2 * d}, "iso-active, fewer total"),
      ("128e top4 eh=4d", {"num_experts": 128, "top_k": 4, "expert_hidden_dim": 4 * d}, "iso-active"),
      ("64e top4 eh=4d", {"num_experts": 64, "top_k": 4, "expert_hidden_dim": 4 * d}, "iso-active"),
      ("head_dim 256", {"head_dim": 256}, "Principle 1; halves head count"),
      (
          "128e top8 eh=2d + hd256",
          {"num_experts": 128, "top_k": 8, "expert_hidden_dim": 2 * d, "head_dim": 256},
          "combined",
      ),
      ("latent d/3", {"moe_input_bottleneck": 3.0}, "NOT iso-active: less expert compute"),
  ]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--rung", default="small", choices=list(FAMILY))
  ap.add_argument("--hw", default="v7x_4x4x4")
  ap.add_argument("--seq", type=int, default=8192)
  ap.add_argument("--pdb", type=int, default=4)
  args = ap.parse_args()

  d, layers, _, _, total, active = FAMILY[args.rung]
  hw = HW_PRESETS[args.hw]
  devices = chips(hw) * hw.d2d_num_chips
  peak = hw.bf16_tflops_per_chip / hw.d2d_num_chips
  base = build(args.rung)

  print(f"{args.rung}: d_model {d}, {layers} layers, {active/1e9:.2f}B active / {total/1e9:.1f}B total")
  print(f"{args.hw}, {devices} devices, seq {args.seq}, pdb {args.pdb}, remat full\n")
  print(f"  {'arm':<26}{'rows/exp':>9}{'M tiles':>8}{'step s':>9}{'TF/s/dev':>9}" f"{'MFU':>7}{'vs base':>9}  note")
  ref = None
  for label, over, note in arms(d):
    model = dataclasses.replace(base, **over) if over else base
    experts = over.get("num_experts", 512)
    top_k = over.get("top_k", 16)
    rows = args.seq * args.pdb * top_k // experts
    tr = TrainingConfig(global_batch_size=devices * args.pdb, seq_len=args.seq, remat_policy="full")
    par = ParallelismConfig(
        tp=1,
        ep_mesh=(1,),
        cp_mesh=(1,),
        fsdp_mesh=(devices,),
        fsdp_transpose_mesh=(1,),
        dp_mesh=(1,),
        pp_mesh=(1,),
        pp_microbatches=1,
    )
    try:
      r = tdriver.run(hw, model, tr, par)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"  {label:<26}{rows:>9}{'':>8}{'n/a':>9}  {' '.join(str(exc).split())[:44]}")
      continue
    ms = call(r.total_time_us) / 1e3
    tf = 2 * call(r.total_flops) / (ms / 1e3) / 1e12
    ref = ref or ms
    print(f"  {label:<26}{rows:>9}{rows // 256:>8}{ms/1e3:9.3f}{tf:9.0f}" f"{tf/peak*100:6.1f}%{ref/ms:8.2f}x  {note}")


if __name__ == "__main__":
  main()
