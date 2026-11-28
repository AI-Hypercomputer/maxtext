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

"""Lowering tests of the Shardy MPMD pipeline on 4 CPU devices, and tests of the last-stage merge rule on made-up
fragment infos and of the IFRT IR rewrite on a synthetic program.

Shardy MPMD's IFRT IR program compiles only on Pathways, so these stop at the lowered program. The lowering and
merge-rule tests need the shardy-mpmd frontend (see sdy_mpmd_compat.py) and skip without it.
"""

import dataclasses
import re
from typing import Any
import unittest

import jax

from maxtext.experimental.mpmd.partir import hparams as hp
from maxtext.experimental.mpmd.partir import pipeline as pl

try:
  from maxtext.experimental.mpmd.partir import runner
  from maxtext.experimental.mpmd.partir import train_compile
  from maxtext.experimental.mpmd.partir.sdy_mpmd_compat import mpmd
  from maxtext.experimental.mpmd.partir.sdy_mpmd_compat import shardy_pipeline

  _SKIP = ""
except ModuleNotFoundError as error:  # shardy-mpmd is an optional dependency
  if not (error.name or "").startswith("shardy"):
    raise
  _SKIP = f"needs shardy-mpmd: {error}"


def cpu_devices(count):
  """The first `count` CPU devices, or a skip when there are fewer."""
  devices = jax.devices("cpu")
  if len(devices) < count:
    raise unittest.SkipTest(f"needs {count} CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count={count})")
  return devices[:count]


def lower(split_bwd_fragments=None, **kw):
  """Lowers the simple model's training step, at PP=2 x FSDP=2 unless `kw` says otherwise; returns (lowered, runnable).

  `split_bwd_fragments`, when not None, overrides the plan's `mpmd_split_bwd_fragments` partitioning option.
  """
  hparams = hp.Hparams(
      **{
          "model_name": "simple",
          "num_layers": 4,
          "num_stages": 2,
          "num_microbatches": 4,
          "per_device_batch_size": 2.0,
          "max_target_length": 64,
          **kw,
      }
  )
  run = runner.build(hparams, devices=cpu_devices(4), abstract=True)
  config = run.plan.mpmd_config
  if split_bwd_fragments is not None:
    options = {**config.partitioning_options, "mpmd_split_bwd_fragments": split_bwd_fragments}
    config = dataclasses.replace(config, partitioning_options=options)
  lowered = mpmd.jit(
      run.train_step,
      mpmd_config=config,
      in_shardings=run.in_shardings,
      out_shardings=run.out_shardings,
      donate_argnums=(0,),
  ).lower(run.state, run.batch)
  return lowered, run


_FRAGMENT_CALL_RE = re.compile(
    r'^\s*(%\w+)(?::\d+)? = mpmd\.fragment_call<mesh="(\w+)", origin=\[([^\]]*)\]> @([\w.$]+)\('
)
_TRANSFER_RE = re.compile(r"mpmd\.transfer (%\w+)")
_CONTRACTING_DIMS_RE = re.compile(r"stablehlo\.dot_general .*contracting_dims = \[([\d, ]*)\] x")


@dataclasses.dataclass(frozen=True, order=True)
class Fragment:
  """A fragment call of the partitioned MPMD module's @main.

  Attributes:
    kind: "F" (every origin a forward), "B" (every origin a backward, transpose count 1) or "FB" (merged).
    transferred: Whether one of its results goes to another mesh.
    weight_dots: dot_generals that contract two dimensions of each operand: the rows and positions of the simple
      model's [rows, positions, features] activations, dW = X^T dY.
    other_dots: The other dot_generals (forward, dX = dY W^T), which contract one dimension or none.
  """

  kind: str
  transferred: bool
  weight_dots: int
  other_dots: int


def fragments(lowered):
  """Each mesh's fragment calls (no pre_m / post_m), in program order, as `Fragment`s."""
  text = lowered.as_text(dialect="mpmd")
  main = text[text.index("func.func public @main(") :]
  main = main[: main.index("\n  }\n")]
  transferred = set(_TRANSFER_RE.findall(main))
  calls = {}
  for line in main.splitlines():
    match = _FRAGMENT_CALL_RE.match(line)
    if not match or re.search(r'"(pre|post)_m\d+"', match.group(3)):
      continue
    result, mesh, origins, callee = match.groups()
    backward = re.findall(r'"\w+"(\(1\))?', origins)
    kind = "B" if all(backward) else "FB" if any(backward) else "F"
    start = text.index(f"func.func @{callee}(")
    body = text[start : text.index("\n  }\n", start)]
    contracted = [len([d for d in dims.split(",") if d.strip()]) for dims in _CONTRACTING_DIMS_RE.findall(body)]
    weight = contracted.count(2)
    calls.setdefault(mesh, []).append(Fragment(kind, result in transferred, weight, len(contracted) - weight))
  return calls


# PP=4 on the 4 CPU devices, one a stage: 4 microbatches of 2 sequences.
_PP4 = {"num_stages": 4, "per_device_batch_size": 8.0}


def dot_totals(mesh_fragments):
  """(weight-gradient dots, other dots) over a mesh's fragments."""
  return sum(f.weight_dots for f in mesh_fragments), sum(f.other_dots for f in mesh_fragments)


def leading_forwards(mesh_fragments):
  """The forwards a mesh runs before its first fragment with a backward in it."""
  return next((i for i, f in enumerate(mesh_fragments) if f.kind != "F"), len(mesh_fragments))


# A synthetic @main: call @a's outputs 0 and 2 and call @b's output go to the other mesh in one merged CopyArrays,
# whose outputs @c and the return use; @c's output comes back in a copy of its own.
_A = "!ifrt.array<tensor<2xf32>, #sp, [0, 1]>"
_B = "!ifrt.array<tensor<2xf32>, #sp, [2, 3]>"
_HEADER = f"""module @jit_f attributes {{ifrt.num_devices = 4 : i32}} {{
  func.func public @main(%arg0: {_A}) -> ({_B}, {_B}, {_B}, {_A}) attributes {{ifrt.function}} {{
"""
_FOOTER = """  }
}
"""
_MERGED = f"""    %a:3, %ca = ifrt.Call @a::@main(%arg0) on devices [0, 1] : ({_A}) -> ({_A}, {_A}, {_A})
    %b, %cb = ifrt.Call @b::@main(%arg0) after %ca on devices [0, 1] : ({_A}) -> {_A}
    %t:3, %ct = ifrt.CopyArrays(%a#0, %b, %a#2) : ({_A}, {_A}, {_A}) -> ({_B}, {_B}, {_B})
    %c, %cc = ifrt.Call @c::@main(%t#0, %t#1) on devices [2, 3] : ({_B}, {_B}) -> {_B}
    %u, %cu = ifrt.CopyArrays(%c) : ({_B}) -> {_A}
    return %c, %t#2, %t#1, %u : {_B}, {_B}, {_B}, {_A}
"""
# Each producer's inputs in a copy of their own, right after it and pinned to it, and every use of the merged copy's
# outputs renamed to the split copies' (#0 and #2 to the first one's #0 and #1, #1 to the second one).
_PINNED = f"""    %a:3, %ca = ifrt.Call @a::@main(%arg0) on devices [0, 1] : ({_A}) -> ({_A}, {_A}, {_A})
    %t_pin0:2, %ct_pin0 = ifrt.CopyArrays(%a#0, %a#2) after %ca : ({_A}, {_A}) -> ({_B}, {_B})
    %b, %cb = ifrt.Call @b::@main(%arg0) after %ca on devices [0, 1] : ({_A}) -> {_A}
    %t_pin1, %ct_pin1 = ifrt.CopyArrays(%b) after %cb : ({_A}) -> {_B}
    %c, %cc = ifrt.Call @c::@main(%t_pin0#0, %t_pin1) on devices [2, 3] : ({_B}, {_B}) -> {_B}
    %u, %cu = ifrt.CopyArrays(%c) after %cc : ({_B}) -> {_A}
    return %c, %t_pin0#1, %t_pin1, %u : {_B}, {_B}, {_B}, {_A}
"""


class _TextModule:
  """Stands in for the IFRT IR module: its text, parsed back as it is."""

  context = None

  def __init__(self, text):
    self.text = text

  def __str__(self):
    return self.text

  @classmethod
  def parse(cls, text, context=None):
    del context
    return cls(text)


@dataclasses.dataclass(frozen=True)
class _PartitioningResult:
  ifrt_ir_module: Any


class PinTransfersTest(unittest.TestCase):
  """pin_ifrt_transfers_to_producers on a synthetic program: no Shardy needed.

  With jax 0.12.0.1 and Shardy 38ab99f4, LowerToIfrt merges two producers' transfers into one CopyArrays for ZB-H1,
  the ZB-H2 schedules and GPipeBut1F1BForLastMesh (not for 1F1B or GPipe), and the rewrite splits it and renames every
  use of its outputs: a use renamed to another value of the same type would still parse, and compute something else.
  """

  def test_a_merged_copy_is_split_per_producer(self):
    lowered = types_namespace(partitioning_result=_PartitioningResult(_TextModule(_HEADER + _MERGED + _FOOTER)))
    self.assertIs(pl.pin_ifrt_transfers_to_producers(lowered), lowered)
    self.assertEqual(str(lowered.partitioning_result.ifrt_ir_module), _HEADER + _PINNED + _FOOTER)

  def test_a_program_already_pinned_is_left_alone(self):
    module = _TextModule(_HEADER + _PINNED + _FOOTER)
    lowered = types_namespace(partitioning_result=_PartitioningResult(module))
    pl.pin_ifrt_transfers_to_producers(lowered)
    self.assertIs(lowered.partitioning_result.ifrt_ir_module, module)


def types_namespace(**kwargs):
  """A mutable stand-in for MpmdLowered."""
  return type("Lowered", (), kwargs)()


def fragment_info(transpose_count, split_type=None, call_counter=0, stage=3):
  """Microbatch `call_counter`'s fragment of `stage`, as Shardy describes it to the merge rule at PP=4 with 4 layers.

  `transpose_count` is 0 for the forward and 1 for the backward; `split_type` marks B_a (KEEP_TRANSFERRED) and B_w
  (DROP_TRANSFERRED) of a split backward. The last stage's fragments also hold the output head.
  """
  names = (f"layer{stage}", "unembed") if stage == 3 else (f"layer{stage}",)
  return shardy_pipeline.FragmentInfo(
      origins=tuple(shardy_pipeline.FragmentOrigin(name, transpose_count) for name in names),
      stage_id=stage,
      call_counter=call_counter,
      split_type=split_type,
      mesh_name=f"m{stage}",
  )


@unittest.skipIf(bool(_SKIP), _SKIP)
class LastStageMergeRuleTest(unittest.TestCase):
  """pipeline._merge_last_stage_fwd_bwd on made-up fragment infos: nothing is lowered.

  A rule that also matched B_w would put the forward in two merge rules, on which Shardy aborts the whole process (a
  CHECK in rule_based_merge.cc) instead of failing a lowering test.
  """

  def test_the_forward_merges_with_b_a_not_b_w(self):
    merge = pl._merge_last_stage_fwd_bwd  # pylint: disable=protected-access
    keep = shardy_pipeline.SplitFragmentType.KEEP_TRANSFERRED
    drop = shardy_pipeline.SplitFragmentType.DROP_TRANSFERRED
    forward = fragment_info(0)
    cases = {
        "F + B_a": (forward, fragment_info(1, keep), True),
        "F + B_w": (forward, fragment_info(1, drop), False),
        "B_a + B_w": (fragment_info(1, keep), fragment_info(1, drop), False),
        "F + the whole backward (1F1B)": (forward, fragment_info(1), True),
        "F + the next microbatch's B_a": (forward, fragment_info(1, keep, call_counter=1), False),
        "F + B_a on stage 2": (fragment_info(0, stage=2), fragment_info(1, keep, stage=2), False),
    }
    context = shardy_pipeline.PipelineContext(num_meshes=4)
    for case, (f1, f2, merged) in cases.items():
      for swapped in (False, True):
        with self.subTest(case, swapped=swapped):
          self.assertEqual(merge(*((f2, f1) if swapped else (f1, f2)), context), merged)


@unittest.skipIf(bool(_SKIP), _SKIP)
class LoweringTest(unittest.TestCase):

  def test_schedule_option_and_stage_ids(self):
    for schedule, option in (
        ("ZB-H1", "ZeroBubbleH1"),
        ("ZB-H2-ZERO", "ZeroBubbleH2ZeroTxLatency"),
        ("ZB-H2-HALF", "ZeroBubbleH2HalfTxLatency"),
        ("ZB-H2-FULL", "ZeroBubbleH2FullTxLatency"),
    ):
      with self.subTest(schedule=schedule):
        plan = pl.build_plan(
            hp.Hparams(model_name="simple", num_layers=8, num_stages=4, pipeline_schedule=schedule),
            devices=cpu_devices(4),
        )
        opts = plan.mpmd_config.partitioning_options
        self.assertEqual(opts["mpmd_pipeline_schedule"], option)
        self.assertTrue(opts["mpmd_split_bwd_fragments"])
        self.assertEqual(plan.mpmd_config.name_to_stage_assignment["layer7"], 3)

  def test_1f1b_merges_the_last_stage(self):
    """The last-stage forward/backward merge rule removes one fragment per microbatch."""
    gpipe, _ = lower(pipeline_schedule="GPIPE")
    one_f_one_b, _ = lower(pipeline_schedule="1F1B")
    n_gpipe = train_compile.summarize_ifrt(gpipe.as_text(dialect="ifrt")).calls
    n_1f1b = train_compile.summarize_ifrt(one_f_one_b.as_text(dialect="ifrt")).calls
    self.assertEqual(n_gpipe - n_1f1b, 4)

  def test_pin_transfers_to_producers(self):
    lowered, run = lower(pipeline_schedule="1F1B")
    before = train_compile.summarize_ifrt(lowered.as_text(dialect="ifrt"))
    self.assertEqual(before.pinned_copies, 0)
    pl.pin_ifrt_transfers_to_producers(lowered)
    after = train_compile.summarize_ifrt(lowered.as_text(dialect="ifrt"))
    self.assertGreater(after.pinned_copies, 0)
    self.assertEqual(after.calls, before.calls)
    # The pinned module is what compile() would read.
    self.assertIn("after %", str(lowered.partitioning_result.ifrt_ir_module))
    # Every stage holds its own layers (2 per stage), embed on m0, unembed on m1.
    self.assertEqual(
        [run.plan.name_to_mesh[name] for name in ("embed", "layer1", "layer2", "unembed")], ["m0", "m0", "m1", "m1"]
    )

  def test_split_and_pinned_transfers_with_zero_bubble(self):
    """ZB-H1: LowerToIfrt merges two microbatches' stage-0 transfers; each goes after its own producer again."""
    lowered, _ = lower(pipeline_schedule="ZB-H1")
    before = train_compile.summarize_ifrt(lowered.as_text(dialect="ifrt"))
    pl.pin_ifrt_transfers_to_producers(lowered)
    text = lowered.as_text(dialect="ifrt")
    after = train_compile.summarize_ifrt(text)
    self.assertGreater(after.copies, before.copies)  # a merged copy was split
    self.assertEqual(after.pinned_copies, after.copies)
    self.assertEqual(after.calls, before.calls)

  def test_zero_bubble_splits_the_backward_fragments(self):
    """ZB-H1 / ZB-H2 at PP=4: each backward that sends an activation gradient upstream becomes two fragments.

    B_a (Shardy's split_keep_transferred) computes the transferred activation gradient; B_w (split_drop_transferred)
    computes the weight gradients, which stay on the mesh, so the schedule can run it in a bubble. The first stage
    sends nothing upstream: its backward stays whole. On the last stage the forward merges with B_a, as 1F1B's does
    with the whole backward.

    Each schedule keeps its own order: before stage s's first backward, ZB-H1 runs 1F1B's P - s forwards, and ZB-H2
    runs 2(P - s) - 1, plus 2(P - s - 1) times a transfer's share of a stage's compute: 0, 1/2 or 1 for ZERO, HALF and
    FULL (Shardy's pipeline_schedule.cc); at most the 4 microbatches.
    """
    gpipe = fragments(lower(pipeline_schedule="GPIPE", **_PP4)[0])
    self.assertEqual([len(gpipe[mesh]) for mesh in ("m0", "m1", "m2", "m3")], [8, 8, 8, 8])
    # The forwards before the first backward on m1 and m2.
    expected_leading_forwards = {"ZB-H1": [3, 2], "ZB-H2-ZERO": [4, 3], "ZB-H2-HALF": [4, 4], "ZB-H2-FULL": [4, 4]}
    for schedule, expected in expected_leading_forwards.items():
      with self.subTest(schedule=schedule):
        zero_bubble = fragments(lower(pipeline_schedule=schedule, **_PP4)[0])
        self.assertEqual(sorted(zero_bubble["m0"]), sorted(gpipe["m0"]))
        for mesh in ("m1", "m2", "m3"):
          b_a = [f for f in zero_bubble[mesh] if f.kind != "F" and f.transferred]
          b_w = [f for f in zero_bubble[mesh] if f.kind == "B" and not f.transferred]
          self.assertEqual([len(b_a), len(b_w)], [4, 4], zero_bubble[mesh])
          self.assertTrue(all(f.weight_dots == 0 for f in b_a), b_a)
          self.assertTrue(all(f.weight_dots > 0 and f.other_dots == 0 for f in b_w), b_w)
          # Every dot_general of GPipe's forward and backward, in exactly one fragment: nothing is recomputed.
          self.assertEqual(dot_totals(zero_bubble[mesh]), dot_totals(gpipe[mesh]))
        self.assertEqual({f.kind for f in zero_bubble["m3"]}, {"FB", "B"})
        order = {mesh: " ".join(f.kind for f in zero_bubble[mesh]) for mesh in ("m1", "m2")}
        self.assertEqual([leading_forwards(zero_bubble[mesh]) for mesh in ("m1", "m2")], expected, order)

  def test_zero_bubble_without_the_split_option_runs_the_1f1b_fragments(self):
    """The split is pipeline._make_mpmd_config's mpmd_split_bwd_fragments: without it ZB-H1 runs 1F1B's fragments."""
    one_f_one_b = fragments(lower(pipeline_schedule="1F1B", **_PP4)[0])
    unsplit = fragments(lower(pipeline_schedule="ZB-H1", split_bwd_fragments=False, **_PP4)[0])
    self.assertEqual(
        {mesh: sorted(f) for mesh, f in unsplit.items()}, {mesh: sorted(f) for mesh, f in one_f_one_b.items()}
    )

  def test_fsdp_pregather_fragments(self):
    """FSDP=2 stages pre-gather their weights in their own pre_m{s} computation."""
    lowered, _ = lower(pipeline_schedule="1F1B")
    self.assertIn("pre_m", lowered.as_text(dialect="mpmd"))

  def test_deepseek3_tiny_lowers_to_ifrt_ir(self):
    """train_compile's CPU path on an MoE model with the deferred expert-gradient reduce-scatter."""
    hparams = hp.Hparams(
        model_name="deepseek3-tiny",
        num_layers=4,
        num_stages=2,
        fsdp=2,
        num_microbatches=2,
        per_device_batch_size=2.0,
        max_target_length=32,
        emb_dim=128,
        mlp_dim=256,
        vocab_size=512,
    )
    cpu_devices(4)
    lowered, compiled, summary = train_compile.run_compile(hparams, compile_topology="cpu")
    self.assertIsNone(compiled)  # compiles only on Pathways
    self.assertEqual(set(summary.calls_per_mesh), {"m0", "m1"})
    self.assertGreater(summary.copies, 0)
    self.assertGreater(summary.pinned_copies, 0)
    self.assertIn("post_m1", lowered.as_text(dialect="mpmd"))

  def test_deepseek_hyper_connections_cross_the_stage_boundary(self):
    """With mHC the activations between the stages carry the 4 residual streams: [rows, seq, streams, emb]."""
    hparams = hp.Hparams(
        model_name="deepseek3-tiny",
        num_layers=4,
        num_stages=2,
        fsdp=2,
        num_microbatches=2,
        per_device_batch_size=2.0,
        max_target_length=32,
        emb_dim=128,
        mlp_dim=256,
        vocab_size=512,
        maxtext_overrides="mhc_expansion_rate=4",
    )
    cpu_devices(4)
    lowered, _, summary = train_compile.run_compile(hparams, compile_topology="cpu")
    self.assertEqual(set(summary.calls_per_mesh), {"m0", "m1"})
    copies = [line for line in lowered.as_text(dialect="ifrt").splitlines() if "ifrt.CopyArrays(" in line]
    # [microbatch rows (2 on 4 devices, 4 on 8), 32 positions, 4 streams, 128]
    self.assertTrue([line for line in copies if re.search(r"tensor<\d+x32x4x128xbf16>", line)], copies)

  def test_train_compile_refuses_a_microbatch_the_stage_cannot_split(self):
    cpu_devices(4)
    hparams = hp.Hparams(model_name="simple", num_layers=4, num_stages=2, num_microbatches=4, per_device_batch_size=2.0)
    with self.assertRaisesRegex(ValueError, r"sequences per microbatch, which must be a multiple of the \d+ devices"):
      train_compile.run_compile(hparams, compile_topology="cpu")


if __name__ == "__main__":
  unittest.main()
