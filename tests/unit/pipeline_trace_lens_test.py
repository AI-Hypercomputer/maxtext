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

"""Unit tests for in-place XSpace pipeline trace decoding, alignment, registry, and fallback."""

import json
import os
import tempfile
import types as pytypes
import unittest

from maxtext.utils import pipeline_trace_lens as lens


def _emit_synthetic_phased_ops(pb, ops_ln, orc, start_ps: int, iter_dur_ps: int) -> int:
  """Populates synthetic phased-schedule while-loops and subtick compute ops."""
  cur = start_ps
  for ph_idx, ph in enumerate(orc.phases):
    ph_start = cur
    ph_dur = ph["length"] * iter_dur_ps
    pb.event(ops_ln, pb.event_md(f"while.{ph_idx + 1}"), ph_start, ph_dur)
    sc = ph["scope"]
    for m_i in range(ph["length"]):
      ms = ph_start + m_i * iter_dur_ps
      for subtick in (0, 1):
        t_idx = 2 * (ph["start_macro"] + m_i) + subtick
        sub_base = ms + subtick * (iter_dur_ps // 2)
        offset_in_sub = 50_000_000
        for kind, letter in (("F", "f"), ("B", "b"), ("W", "w")):
          if not orc.scheduled(t_idx, kind):
            continue
          pb.event(
              ops_ln,
              pb.event_md(
                  f"fusion.{sc}_{subtick}_{letter}",
                  stats={
                      "tf_op": f"jit(train_step)/pipeline_module/{sc}/custom_subtick_{subtick}_{letter}/matmul",
                      "hlo_category": "convolution fusion",
                  },
              ),
              sub_base + offset_in_sub,
              250_000_000,
          )
          offset_in_sub += 280_000_000
    cur = ph_start + ph_dur + 200_000_000
  return cur


def _emit_synthetic_fwd_bwd_ops(
    pb,
    ops_ln,
    n_iter: int,
    start_ps: int,
    iter_dur_ps: int,
    *,
    unrolled: bool = False,
    split_fwd_loops: int = 1,
) -> int:
  """Populates synthetic forward/backward pipeline loops (scanned, unrolled, or split)."""
  cur = start_ps
  if unrolled:
    for it in range(n_iter):
      it_s = cur + it * iter_dur_ps
      pb.event(
          ops_ln,
          pb.event_md(
              f"fusion.fwd_bubble.{it}",
              stats={
                  "tf_op": "jit(train_step)/pipeline_module/pp_bubble_select/select",
                  "hlo_category": "loop fusion",
              },
          ),
          it_s + 50_000_000,
          50_000_000,
      )
      pb.event(
          ops_ln,
          pb.event_md(
              f"fusion.fwd_compute.{it}",
              stats={
                  "tf_op": "jit(train_step)/pipeline_module/pp_stage_compute/matmul",
                  "hlo_category": "convolution fusion",
              },
          ),
          it_s + 120_000_000,
          1_300_000_000,
      )
      pb.event(
          ops_ln,
          pb.event_md(
              f"collective-permute-done.{it}",
              stats={
                  "tf_op": "jit(train_step)/pipeline_module/pp_shift/ppermute",
                  "hlo_category": "collective-permute-done",
              },
          ),
          it_s + 1_500_000_000,
          250_000_000,
      )
    cur += n_iter * iter_dur_ps + 500_000_000
    for it in range(n_iter):
      it_s = cur + it * iter_dur_ps
      pb.event(
          ops_ln,
          pb.event_md(
              f"collective-permute-done.bwd.{it}",
              stats={
                  "tf_op": "jit(train_step)/transpose(pipeline_module)/pp_shift/ppermute",
                  "hlo_category": "collective-permute-done",
              },
          ),
          it_s + 50_000_000,
          150_000_000,
      )
      pb.event(
          ops_ln,
          pb.event_md(
              f"fusion.bwd_compute.{it}",
              stats={
                  "tf_op": "jit(train_step)/transpose(pipeline_module)/pp_stage_compute/matmul",
                  "hlo_category": "convolution fusion",
              },
          ),
          it_s + 220_000_000,
          1_500_000_000,
      )
    return cur + n_iter * iter_dur_ps + 500_000_000

  # Forward while loop(s)
  loop_id = 1
  for split_idx in range(split_fwd_loops):
    seg_iters = max(1, n_iter // split_fwd_loops) if split_fwd_loops > 1 else n_iter
    fwd_start = cur
    fwd_dur = seg_iters * iter_dur_ps
    pb.event(ops_ln, pb.event_md(f"while.{loop_id}"), fwd_start, fwd_dur)
    loop_id += 1
    for it in range(seg_iters):
      it_s = fwd_start + it * iter_dur_ps
      pb.event(
          ops_ln,
          pb.event_md(
              f"fusion.fwd_compute_{split_idx}",
              stats={
                  "tf_op": "jit(train_step)/pipeline_module/while/body/pp_stage_compute/matmul",
                  "hlo_category": "convolution fusion",
              },
          ),
          it_s + 100_000_000,
          1_400_000_000,
      )
      pb.event(
          ops_ln,
          pb.event_md(
              f"collective-permute-start.{split_idx + 1}",
              stats={
                  "tf_op": "jit(train_step)/pipeline_module/while/body/pp_shift/ppermute",
                  "hlo_category": "collective-permute-start",
              },
          ),
          it_s + 1_500_000_000,
          100_000_000,
      )
      pb.event(
          ops_ln,
          pb.event_md(
              f"collective-permute-done.{split_idx + 1}",
              stats={
                  "tf_op": "jit(train_step)/pipeline_module/while/body/pp_shift/ppermute",
                  "hlo_category": "collective-permute-done",
              },
          ),
          it_s + 1_650_000_000,
          250_000_000,
      )
    cur = fwd_start + fwd_dur + 300_000_000

  # Backward while loop
  bwd_start = cur
  bwd_dur = n_iter * iter_dur_ps
  pb.event(ops_ln, pb.event_md(f"while.{loop_id}"), bwd_start, bwd_dur)
  for it in range(n_iter):
    it_s = bwd_start + it * iter_dur_ps
    pb.event(
        ops_ln,
        pb.event_md(
            "fusion.bwd_compute",
            stats={
                "tf_op": "jit(train_step)/transpose(pipeline_module/while/body)/pp_stage_compute/matmul",
                "hlo_category": "convolution fusion",
            },
        ),
        it_s + 100_000_000,
        1_400_000_000,
    )
    pb.event(
        ops_ln,
        pb.event_md(
            "all-gather-done.1",
            stats={
                "tf_op": "jit(train_step)/transpose(pipeline_module/while/body)/fsdp_ag",
                "hlo_category": "all-gather",
            },
        ),
        it_s + 1_550_000_000,
        200_000_000,
    )
  return bwd_start + bwd_dur + 500_000_000


def _make_synthetic_xspace(
    hostname: str,
    profile_start_ns: int,
    device_ordinals: list[int],
    schedule: str,
    num_stages: int,
    num_microbatches: int,
    num_repeats: int = 1,
    forwarding_delay: int = 1,
    clock_skew_ps: int = 0,
    include_hlo_metadata: bool = False,
    include_megascale_custom: bool = False,
    unrolled: bool = False,
    split_fwd_loops: int = 1,
    fallback_n_iter: int = 5,
):
  """Builds a realistic synthetic XSpace with Task Environment, /host:metadata, /host:CPU, and /device:TPU:k."""
  space = lens.XSpace()
  space.hostnames.append(hostname)

  if include_hlo_metadata:
    meta_plane = space.planes.add()
    meta_plane.id = 0
    meta_plane.name = "/host:metadata"
    pb = lens._PlaneBuilder(meta_plane)  # pylint: disable=protected-access
    pb.add_stats(meta_plane, {"hlo_proto_blob": b"\x08\x01\x12\x0eSYNTHETIC_HLO!"})

  if include_megascale_custom:
    ms_plane = space.planes.add()
    ms_plane.id = 0
    ms_plane.name = "/device:CUSTOM:Megascale Trace"
    pb = lens._PlaneBuilder(ms_plane)  # pylint: disable=protected-access
    pb.add_stats(ms_plane, {"process_id": 0})

  env = space.planes.add()
  env.id = 0
  env.name = "Task Environment"
  env_b = lens._PlaneBuilder(env)  # pylint: disable=protected-access
  env_b.add_stats(
      env,
      {
          "profile_start_time": lens._u64(profile_start_ns),  # pylint: disable=protected-access
          "profile_stop_time": lens._u64(profile_start_ns + 200_000_000),  # pylint: disable=protected-access
      },
  )

  # /host:CPU with step_num=5
  cpu = space.planes.add()
  cpu.id = 0
  cpu.name = "/host:CPU"
  cpu_b = lens._PlaneBuilder(cpu)  # pylint: disable=protected-access
  cpu_ln = cpu_b.line(1, "main")
  cpu_b.event(cpu_ln, cpu_b.event_md("train"), 50_000_000_000, 100_000_000_000, {"step_num": 5})

  orc = lens.PipelineOracle(schedule, num_stages, num_microbatches, num_repeats, forwarding_delay=forwarding_delay)

  # Step window starts at 50 ms relative to profile_start_ns (+ skew).
  step_start_ps = 50_000_000_000 + clock_skew_ps
  iter_dur_ps = 2_000_000_000  # 2 ms per iter/macro

  for ord_idx in device_ordinals:
    tp = space.planes.add()
    tp.id = 2 + ord_idx
    tp.name = f"/device:TPU:{ord_idx}"
    pb = lens._PlaneBuilder(tp)  # pylint: disable=protected-access
    pb.add_stats(tp, {"device_id": ord_idx, "device_type_string": "TPU v6e"})
    mod_ln = pb.line(2, "XLA Modules")
    ops_ln = pb.line(3, "XLA Ops")

    cur = step_start_ps + 1_000_000_000  # 1 ms pre-step
    if orc.is_phased:
      cur = _emit_synthetic_phased_ops(pb, ops_ln, orc, cur, iter_dur_ps)
    else:
      n_iter = fallback_n_iter if orc.is_fallback else orc.n_iter
      cur = _emit_synthetic_fwd_bwd_ops(
          pb,
          ops_ln,
          n_iter,
          cur,
          iter_dur_ps,
          unrolled=unrolled,
          split_fwd_loops=split_fwd_loops,
      )

    # Global all-reduce anchor at end of step (spans all devices -> clock alignment anchor).
    ar_name = "all-reduce-done.global replica_groups=[1,4]<=[4]"
    pb.event(
        ops_ln,
        pb.event_md(ar_name, stats={"hlo_category": "all-reduce", "tf_op": "jit(train_step)/grad_norm"}),
        cur + 100_000_000,
        400_000_000,
    )
    step_end_ps = cur + 1_000_000_000
    pb.event(mod_ln, pb.event_md("jit_train_step.1"), step_start_ps, step_end_ps - step_start_ps)

  return space, step_start_ps, step_end_ps


def _write_sidecar(
    path: str,
    hostname: str,
    process_index: int,
    process_count: int,
    mesh_device_ids,
    local_devices,
    schedule: str,
    num_stages: int,
    num_microbatches: int,
    num_repeats: int = 1,
    forwarding_delay: int = 1,
):
  """Writes a minimal pp_trace_metadata/v0 sidecar JSON file."""
  meta = {
      "schema": "pp_trace_metadata/v0",
      "hostname": hostname,
      "process_index": process_index,
      "process_count": process_count,
      "mesh_axes": ["data", "stage", "fsdp"],
      "mesh_shape": [1, num_stages, 1],
      "mesh_device_ids": mesh_device_ids,
      "stage_axis": 1,
      "local_devices": local_devices,
      "pipeline": {
          "schedule": schedule,
          "num_stages": num_stages,
          "num_microbatches": num_microbatches,
          "num_repeats": num_repeats,
          "forwarding_delay": forwarding_delay,
      },
  }
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(meta, f)


def _write_four_process_fixture(
    tmp: str,
    *,
    schedule: str,
    num_microbatches: int,
    num_repeats: int = 1,
    forwarding_delay: int = 1,
    unrolled: bool = False,
    split_fwd_loops: int = 1,
    fallback_n_iter: int = 5,
):
  """Writes a 4-process staged trace directory with multi-ms cross-host clock skews."""
  skews_ps = [0, 2_500_000_000, -1_800_000_000, 900_000_000]  # +2.5ms, -1.8ms, +0.9ms
  mesh_ids = [[[0], [1], [2], [3]]]
  host0_path = None
  orig_host0_plane_bytes = {}
  for proc in range(4):
    if proc == 0:
      base_dir = tmp
      meta_name = "pipeline_trace_metadata_process_00000.json"
    else:
      base_dir = os.path.join(tmp, lens.STAGE_TRACES_SUBDIR, f"process_{proc:05d}")
      meta_name = "pipeline_trace_metadata.json"
    prof_dir = os.path.join(base_dir, "plugins", "profile", "2026_11_19_12_00_00")
    os.makedirs(prof_dir)
    # Intentionally reuse the same hostname across all 4 processes (single-VM multi-process scenario).
    xp_path = os.path.join(prof_dir, "shared-vm.xplane.pb")
    if proc == 0:
      host0_path = xp_path
    sp, _, _ = _make_synthetic_xspace(
        hostname="shared-vm",
        profile_start_ns=1_700_000_000_000_000_000,
        device_ordinals=[0],
        schedule=schedule,
        num_stages=4,
        num_microbatches=num_microbatches,
        num_repeats=num_repeats,
        forwarding_delay=forwarding_delay,
        clock_skew_ps=skews_ps[proc],
        include_hlo_metadata=(proc == 0),
        unrolled=unrolled,
        split_fwd_loops=split_fwd_loops,
        fallback_n_iter=fallback_n_iter,
    )
    if proc == 0:
      orig_host0_plane_bytes = {p.name: p.SerializeToString() for p in sp.planes}
    with open(xp_path, "wb") as f:
      f.write(sp.SerializeToString())
    _write_sidecar(
        os.path.join(base_dir, meta_name),
        hostname="shared-vm",
        process_index=proc,
        process_count=4,
        mesh_device_ids=mesh_ids,
        local_devices=[{"id": proc, "process_index": proc, "coords": [proc, 0, 0], "core_on_chip": 0, "stage": proc}],
        schedule=schedule,
        num_stages=4,
        num_microbatches=num_microbatches,
        num_repeats=num_repeats,
        forwarding_delay=forwarding_delay,
    )
  assert host0_path is not None
  return host0_path, orig_host0_plane_bytes


class PipelineTraceLensTest(unittest.TestCase):
  """Tests in-place XSpace augmentation, plane preservation, alignment, registry, and fallback."""

  def test_single_process_ring_mesh_preserves_existing_planes_and_exact_clock(self):
    with tempfile.TemporaryDirectory() as tmp:
      prof_dir = os.path.join(tmp, "plugins", "profile", "2026_11_19_12_00_00")
      os.makedirs(prof_dir)
      host0_path = os.path.join(prof_dir, "vm-host.xplane.pb")
      orig_space, step_start_ps, step_end_ps = _make_synthetic_xspace(
          hostname="vm-host",
          profile_start_ns=1_700_000_000_000_000_000,
          device_ordinals=[0, 1, 2, 3],
          schedule="gpipe",
          num_stages=4,
          num_microbatches=4,
          include_hlo_metadata=True,
          include_megascale_custom=True,
      )
      orig_plane_bytes = {p.name: p.SerializeToString() for p in orig_space.planes}
      with open(host0_path, "wb") as f:
        f.write(orig_space.SerializeToString())

      # Ring order [0, 1, 3, 2]: local device 2 is Stage 3, local device 3 is Stage 2.
      local_devs = [
          {"id": 0, "process_index": 0, "coords": [0, 0, 0], "core_on_chip": 0, "stage": 0},
          {"id": 1, "process_index": 0, "coords": [1, 0, 0], "core_on_chip": 0, "stage": 1},
          {"id": 2, "process_index": 0, "coords": [0, 1, 0], "core_on_chip": 0, "stage": 3},
          {"id": 3, "process_index": 0, "coords": [1, 1, 0], "core_on_chip": 0, "stage": 2},
      ]
      _write_sidecar(
          os.path.join(tmp, "pipeline_trace_metadata_process_00000.json"),
          hostname="vm-host",
          process_index=0,
          process_count=1,
          mesh_device_ids=[[[0], [1], [3], [2]]],
          local_devices=local_devs,
          schedule="gpipe",
          num_stages=4,
          num_microbatches=4,
      )

      stats = lens.augment_host0_xplane_in_place(host0_path, search_dir=tmp)
      self.assertEqual(stats["schedule"], "gpipe")
      self.assertEqual(stats["stages_measured"], [0, 1, 2, 3])
      self.assertFalse(stats["fallback_mode"])
      # Only 2 custom planes are added (Stages + Bubbles & Comm Waits).
      self.assertEqual(stats["planes_added"], 2)

      with open(host0_path, "rb") as f:
        aug = lens.XSpace.FromString(f.read())
      aug_by_name = {p.name: p for p in aug.planes}

      # 100% byte-identical preservation of every original plane (/device:TPU:*, /host:metadata, Task Environment).
      for pname, expected_bytes in orig_plane_bytes.items():
        self.assertIn(pname, aug_by_name)
        self.assertEqual(aug_by_name[pname].SerializeToString(), expected_bytes, f"Plane {pname} was modified!")

      custom_added = [
          p for p in aug.planes if p.name.startswith("/device:CUSTOM:") and p.name != "/device:CUSTOM:Megascale Trace"
      ]
      self.assertEqual(len(custom_added), 2)
      # Plane IDs must start above Megascale's id=0 and remain < 200.
      self.assertEqual([p.id for p in custom_added], [1, 2])
      self.assertTrue(custom_added[0].name.startswith("/device:CUSTOM:1. Pipeline Stages"))
      self.assertEqual(custom_added[1].name, "/device:CUSTOM:2. Bubbles & Comm Waits (By Stage)")
      self.assertFalse(any("XLA Ops" in p.name for p in custom_added))

      stage_plane = custom_added[0]
      self.assertIn("All 4 Cores / 4 Chips", stage_plane.name)
      lines_by_name = {ln.name: ln for ln in stage_plane.lines}
      self.assertIn("1 Step & Hardware Map", lines_by_name)
      self.assertIn("2 Phase \u00b7 Tick", lines_by_name)

      # Verify exact picosecond alignment of Step span with Process 0's native jit_train_step window.
      step_ev = lines_by_name["1 Step & Hardware Map"].events[0]
      self.assertEqual(step_ev.offset_ps, step_start_ps)
      self.assertEqual(step_ev.duration_ps, step_end_ps - step_start_ps)

      # Verify ring mapping: Stage 2 maps to Core 3, Stage 3 maps to Core 2.
      row_names = [ln.name for ln in stage_plane.lines if ln.name.startswith("3.")]
      self.assertEqual(len(row_names), 4)
      self.assertIn("Stage 2 \u00b7 Core 3", row_names[2])
      self.assertIn("Stage 3 \u00b7 Core 2", row_names[3])

      # Verify every event across all new planes has offset_ps > 0 and flows exist.
      flow_count = 0
      dummy_count = 0
      for p in custom_added:
        for ln in p.lines:
          for ev in ln.events:
            self.assertGreater(ev.offset_ps, 0)
            st = lens._extract_stats(ev, p.stat_metadata)  # pylint: disable=protected-access
            if "flow" in st:
              flow_count += 1
            if st.get("kind") == "dummy":
              dummy_count += 1
      self.assertGreater(flow_count, 0)
      self.assertGreater(dummy_count, 0)

      # Idempotency: calling augment_host0_xplane_in_place a second time keeps exact same plane count.
      lens.augment_host0_xplane_in_place(host0_path, search_dir=tmp)
      with open(host0_path, "rb") as f:
        aug2 = lens.XSpace.FromString(f.read())
      self.assertEqual(len(aug2.planes), len(aug.planes))

  def test_four_process_staged_discovery_alignment_and_builtin_schedules(self):
    for schedule, num_microbatches, num_repeats, forwarding_delay, unrolled in (
        ("gpipe", 4, 1, 1, False),
        ("circular", 4, 2, 1, False),
        ("circular", 8, 2, 2, False),
        ("gpipe", 4, 1, 1, True),
    ):
      with (
          self.subTest(schedule=schedule, delay=forwarding_delay, unrolled=unrolled),
          tempfile.TemporaryDirectory() as tmp,
      ):
        host0_path, orig_host0_plane_bytes = _write_four_process_fixture(
            tmp,
            schedule=schedule,
            num_microbatches=num_microbatches,
            num_repeats=num_repeats,
            forwarding_delay=forwarding_delay,
            unrolled=unrolled,
        )
        stats = lens.augment_host0_xplane_in_place(host0_path, search_dir=tmp)
        self.assertEqual(stats["schedule"], schedule)
        self.assertEqual(stats["stages_measured"], [0, 1, 2, 3])
        self.assertFalse(stats["fallback_mode"])
        self.assertEqual(stats["planes_added"], 2)
        # Global all-reduce anchors recover the injected multi-ms clock skews to 0.0 us residual.
        self.assertAlmostEqual(stats["align_p95_us"], 0.0, places=2)

        with open(host0_path, "rb") as f:
          aug = lens.XSpace.FromString(f.read())
        aug_by_name = {p.name: p for p in aug.planes}
        for pname, expected_bytes in orig_host0_plane_bytes.items():
          self.assertIn(pname, aug_by_name)
          self.assertEqual(aug_by_name[pname].SerializeToString(), expected_bytes)
        custom_added = [p for p in aug.planes if p.name.startswith("/device:CUSTOM:")]
        self.assertEqual(len(custom_added), 2)
        self.assertTrue(custom_added[0].name.startswith("/device:CUSTOM:1. Pipeline Stages"))
        self.assertEqual(custom_added[1].name, "/device:CUSTOM:2. Bubbles & Comm Waits (By Stage)")
        self.assertFalse(any("XLA Ops" in p.name for p in custom_added))
        stage_rows = [ln.name for ln in custom_added[0].lines if ln.name.startswith("3.")]
        self.assertEqual(len(stage_rows), 4)
        self.assertEqual(len(custom_added[1].lines), 4)

  def test_register_custom_phased_schedule_via_public_registry(self):
    """Tests registering a custom phased schedule with register_pipeline_schedule."""
    sched_name = "test_custom_phased"

    def _build_custom_phased(num_stages, num_microbatches, num_repeats=2, forwarding_delay=1):
      del num_repeats
      tick_f = pytypes.SimpleNamespace(has_f=True, b_mode="none", w_pop=False, has_tail=False)
      tick_fbw = pytypes.SimpleNamespace(has_f=True, b_mode="split", w_pop=True, has_tail=False)
      phases = [
          {"scope": "custom_phase_warmup", "start_macro": 0, "length": 2, "a": tick_f, "b": tick_f, "label": "warmup"},
          {"scope": "custom_phase_core", "start_macro": 2, "length": 2, "a": tick_fbw, "b": tick_fbw, "label": "core"},
      ]

      def _pass_slot(tick, stage, kind):
        mb = (tick + stage) % num_microbatches
        chunk = tick % 2
        valid = not (kind == "W" and tick < 4)
        return mb, chunk, valid

      return lens.PipelineSchedule(
          name=sched_name,
          mode="phased",
          num_stages=num_stages,
          num_microbatches=num_microbatches,
          num_repeats=2,
          forwarding_delay=forwarding_delay,
          num_ticks=8,
          num_macros=4,
          phases=phases,
          pass_slot=_pass_slot,
      )

    lens.register_pipeline_schedule(sched_name, _build_custom_phased, aliases=("tcp_alias",))
    self.addCleanup(lens.unregister_pipeline_schedule, sched_name)
    self.assertIn(sched_name, lens.get_registered_pipeline_schedules())
    orc = lens.PipelineOracle("tcp_alias", 4, 4)
    self.assertEqual(orc.tail_scope(), "")

    with tempfile.TemporaryDirectory() as tmp:
      host0_path, _ = _write_four_process_fixture(tmp, schedule="tcp_alias", num_microbatches=4, num_repeats=2)
      stats = lens.augment_host0_xplane_in_place(host0_path, search_dir=tmp)
      self.assertEqual(stats["schedule"], sched_name)
      self.assertFalse(stats["fallback_mode"])
      self.assertEqual(stats["planes_added"], 2)
      self.assertAlmostEqual(stats["align_p95_us"], 0.0, places=2)

      with open(host0_path, "rb") as f:
        aug = lens.XSpace.FromString(f.read())
      stage_plane = next(p for p in aug.planes if p.name.startswith("/device:CUSTOM:1. Pipeline Stages"))
      md_names = {md.name for md in stage_plane.event_metadata.values()}
      self.assertIn("Forward", md_names)
      self.assertIn("Backward", md_names)
      self.assertIn("WeightGrad", md_names)

  def test_unregistered_schedule_falls_back_gracefully_without_mislabeling(self):
    """Unregistered schedules align clocks and emit raw iter spans + waits without crashing."""
    with tempfile.TemporaryDirectory() as tmp:
      host0_path, orig_host0_plane_bytes = _write_four_process_fixture(
          tmp,
          schedule="future_unregistered_algo",
          num_microbatches=4,
          num_repeats=1,
          fallback_n_iter=5,
      )
      stats = lens.augment_host0_xplane_in_place(host0_path, search_dir=tmp)
      self.assertEqual(stats["schedule"], "future_unregistered_algo")
      self.assertTrue(stats["fallback_mode"])
      self.assertIn("unregistered", stats["fallback_reason"])
      self.assertEqual(stats["stages_measured"], [0, 1, 2, 3])
      self.assertEqual(stats["planes_added"], 2)
      self.assertAlmostEqual(stats["align_p95_us"], 0.0, places=2)

      with open(host0_path, "rb") as f:
        aug = lens.XSpace.FromString(f.read())
      aug_by_name = {p.name: p for p in aug.planes}
      for pname, expected_bytes in orig_host0_plane_bytes.items():
        self.assertEqual(aug_by_name[pname].SerializeToString(), expected_bytes)

      stage_plane = next(p for p in aug.planes if p.name.startswith("/device:CUSTOM:1. Pipeline Stages"))
      bubble_plane = next(p for p in aug.planes if p.name == "/device:CUSTOM:2. Bubbles & Comm Waits (By Stage)")

      # Step & Hardware Map must clearly point to developer docs.
      map_line = next(ln for ln in stage_plane.lines if ln.name == "1 Step & Hardware Map")
      step_ev = map_line.events[0]
      step_md = stage_plane.event_metadata[step_ev.metadata_id]
      step_stats = lens._extract_stats(step_ev, stage_plane.stat_metadata)  # pylint: disable=protected-access
      self.assertEqual(step_stats.get("fallback_mode"), 1)
      self.assertEqual(step_stats.get("developer_docs"), lens.DEVELOPER_DOCS_PATH)
      self.assertIn(lens.DEVELOPER_DOCS_PATH, step_md.display_name)

      # Stacked stage rows emit raw `iter 0 .. T-1` spans and NEVER guess Forward/Backward microbatch labels.
      stage_lines = [ln for ln in stage_plane.lines if ln.name.startswith("3.")]
      self.assertEqual(len(stage_lines), 4)
      for ln in stage_lines:
        self.assertEqual(len(ln.events), 10)  # 5 fwd-loop + 5 bwd-loop raw iters
        for idx, ev in enumerate(ln.events):
          md = stage_plane.event_metadata[ev.metadata_id]
          self.assertEqual(md.name, "Pipeline Iter (Fallback)")
          self.assertTrue(md.display_name.startswith(f"iter {idx}"))
          ev_stats = lens._extract_stats(ev, stage_plane.stat_metadata)  # pylint: disable=protected-access
          self.assertNotIn("mb", ev_stats)
          self.assertNotIn("flow", ev_stats)

      # Plane 2 still captures PP ppermute Wait and MoE/FSDP Wait spans on every stage.
      self.assertEqual(len(bubble_plane.lines), 4)
      b_md_names = {md.name for md in bubble_plane.event_metadata.values()}
      self.assertIn("PP ppermute Wait", b_md_names)
      self.assertIn("MoE/FSDP Wait", b_md_names)
      pp_wait_mds = [md for md in bubble_plane.event_metadata.values() if md.name == "PP ppermute Wait"]
      self.assertTrue(all(md.display_name.startswith("pp wait ") for md in pp_wait_mds))

  def test_unexpected_split_loop_structure_triggers_graceful_fallback(self):
    """If a registered schedule has unexpected split loops, fall back instead of mislabeling."""
    with tempfile.TemporaryDirectory() as tmp:
      host0_path, _ = _write_four_process_fixture(
          tmp,
          schedule="gpipe",
          num_microbatches=4,
          num_repeats=1,
          split_fwd_loops=2,
      )
      stats = lens.augment_host0_xplane_in_place(host0_path, search_dir=tmp)
      self.assertTrue(stats["fallback_mode"])
      self.assertIn("expected 1 fwd + 1 bwd", stats["fallback_reason"])
      self.assertEqual(stats["planes_added"], 2)
      self.assertAlmostEqual(stats["align_p95_us"], 0.0, places=2)


if __name__ == "__main__":
  unittest.main()
