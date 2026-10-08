"""perfsim projection for olmoe3-3p5b training on TPU v4 slices.

perfsim ships no v4 preset, so the chip is derived from v5p (both expose two
TensorCores as one megacore device) with v4's published numbers: 275 TF/s bf16,
1.2 TB/s and 32 GiB HBM, 16 MiB VMEM, 128 MiB CMEM, and half v5p's per-link ICI.
The model is perfsim's own OLMOE3_3P5B preset.

  PYTHONPATH=<perfsim>:<perfsim-deps> python3 scripts/olmoe3_v4_perfsim.py \
      --chips 64 --pdb 1 --seq 4096 --ep 4

Prints step time, tokens/s/chip, and the top of the breakdown tree. perfsim
assumes roofline GEMMs and perfect overlap, so read ratios, not absolutes.
"""

import argparse
import dataclasses

from perfsim.core.engine.training import driver
from perfsim.core.engine.training.config import ParallelismConfig, TrainingConfig
from perfsim.plugins.hardware import HW_PRESETS
from perfsim.plugins.models.presets import OLMOE3_3P5B

V4_PEAK_TFLOPS = 275.0


def v4_spec(chips: int):
  shapes = {64: (4, 4, 4), 128: (4, 4, 8), 256: (4, 8, 8), 512: (8, 8, 8)}
  return dataclasses.replace(
      HW_PRESETS["v5p_128"],
      name=f"v4-{2 * chips}",
      bf16_tflops_per_chip=V4_PEAK_TFLOPS,
      hbm_bw_gbs_per_chip=1200.0,
      hbm_capacity_gib_per_chip=32.0,
      sram_mib_per_chip=128.0,
      vmem_usable_mib_per_core=16.0,
      ici_bw_gbs_per_link=45.0,
      slice_shape=shapes[chips],
  )


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--chips", type=int, default=64)
  p.add_argument("--pdb", type=int, default=1)
  p.add_argument("--seq", type=int, default=4096)
  p.add_argument("--ep", type=int, default=4)
  p.add_argument("--remat", default="full")
  p.add_argument("--depth", type=int, default=2, help="breakdown tree depth to print")
  a = p.parse_args()

  hw = v4_spec(a.chips)
  train = TrainingConfig(
      global_batch_size=a.pdb * a.chips,
      seq_len=a.seq,
      remat_policy=a.remat,
      weights_dtype="bf16",
      tp_ag_dtype="bf16",
  )
  par = ParallelismConfig(ep_mesh=(a.ep,), fsdp_mesh=(a.chips // a.ep,))
  root = driver.run(hw, OLMOE3_3P5B, train, par)

  step_s = root.time_us * 1e-6
  tokens = a.pdb * a.chips * a.seq
  print(f"# {hw.name} chips={a.chips} pdb={a.pdb} seq={a.seq} ep={a.ep} remat={a.remat}")
  print(f"step {step_s * 1e3:.1f} ms, {tokens / step_s / a.chips:.0f} tok/s/chip")

  def walk(node, depth, indent=0):
    if indent > depth:
      return
    share = 100 * node.time_us * 1e-6 / step_s if step_s else 0
    print(f"{'  ' * indent}{node.name:48s} {node.time_us * 1e-3:9.1f} ms {share:5.1f}%")
    for child in sorted(node.children, key=lambda c: -c.time_us)[:8]:
      walk(child, depth, indent + 1)

  walk(root, a.depth)


if __name__ == "__main__":
  main()
