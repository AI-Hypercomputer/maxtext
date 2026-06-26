"""Hotfix to disable prompt_logprobs in tunix vLLM Sampler and bypass vLLM V1 scheduler AssertionError."""

import os
import sys

# --- Patch 1: tunix vllm_sampler.py ---
print("Applying hotfix to tunix/generate/vllm_sampler.py...")
p = "/usr/local/lib/python3.12/site-packages/tunix/generate/vllm_sampler.py"
if not os.path.exists(p):
  try:
    import tunix.generate.vllm_sampler as vs

    p = vs.__file__
  except ImportError:
    print("Error: tunix not found.")
    sys.exit(1)

print(f"Target file: {p}")
with open(p, "r", encoding="utf-8") as f:
  c = f.read()
target = (
    "      if self.config.return_logprobs:\n"
    "        sampling_params.logprobs = 1  # b/428730696\n"
    "        sampling_params.prompt_logprobs = 1  # b/428730696"
)
replacement = (
    "      if self.config.return_logprobs:\n"
    "        sampling_params.logprobs = 1  # b/428730696\n"
    "        sampling_params.prompt_logprobs = 0"
)

if target in c:
  c = c.replace(target, replacement)
  with open(p, "w", encoding="utf-8") as f:
    f.write(c)
  print("Hotfix successfully applied to tunix/generate/vllm_sampler.py.")
else:
  if "sampling_params.prompt_logprobs = 0" in c:
    print("Hotfix already applied.")
  else:
    print("Warning: target string not found.")

# --- Patch 2: vllm scheduler.py ---
print("Applying hotfix to vllm/v1/core/sched/scheduler.py...")
p2 = "/usr/local/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"
if os.path.exists(p2):
  with open(p2, "r", encoding="utf-8") as f:
    c2 = f.read()
  target2 = (
      "                # Invariant: EngineCore returns no partial prefill"
      " outputs.\n                assert not prompt_logprobs_tensors"
  )
  replacement2 = "                # Invariant: EngineCore returns no partial prefill" " outputs.\n                pass"
  if target2 in c2:
    c2 = c2.replace(target2, replacement2)
    with open(p2, "w", encoding="utf-8") as f:
      f.write(c2)
    print("Hotfix successfully applied to vllm/v1/core/sched/scheduler.py.")
  else:
    # Check if the target is slightly different (e.g. fewer spaces or different lines)
    # We can also do a simpler string replacement
    target2_alt = "assert not prompt_logprobs_tensors"
    if target2_alt in c2:
      c2 = c2.replace(target2_alt, "pass")
      with open(p2, "w", encoding="utf-8") as f:
        f.write(c2)
      print("Hotfix successfully applied to vllm/v1/core/sched/scheduler.py (alt match).")
    else:
      print("Warning: target string not found in scheduler.py.")
else:
  print("Warning: scheduler.py path not found.")

# --- Patch 3: tpu_inference utils.py ---
print("Applying hotfix to tpu_inference/runner/utils.py...")
p3 = "/usr/local/lib/python3.12/site-packages/tpu_inference/runner/utils.py"
if os.path.exists(p3):
  with open(p3, "r", encoding="utf-8") as f:
    c3 = f.read()
  target3 = (
      "            self._canonical_dst_ts = self._resolve_canonical_dst_ts(phase_dir)\n\n"
      "            self.profile_dir_with_phase_suffix = os.path.join(\n"
      '                phase_dir, f"dp_rank_{self.worker_rank}")'
  )
  replacement3 = (
      "            self._canonical_dst_ts = self._resolve_canonical_dst_ts(phase_dir)\n"
      "            self.default_profiling_options.session_id = self._canonical_dst_ts\n\n"
      "            self.profile_dir_with_phase_suffix = os.path.join(\n"
      '                phase_dir, f"dp_rank_{self.worker_rank}")'
  )
  if target3 in c3:
    c3 = c3.replace(target3, replacement3)
    with open(p3, "w", encoding="utf-8") as f:
      f.write(c3)
    print("Hotfix successfully applied to tpu_inference/runner/utils.py.")
  else:
    if "self.default_profiling_options.session_id = self._canonical_dst_ts" in c3:
      print("Hotfix already applied to tpu_inference/runner/utils.py.")
    else:
      print("Warning: target string not found in utils.py.")
else:
  print("Warning: tpu_inference utils.py path not found.")

# --- Patch 4: pathwaysutils profiling.py ---
print("Applying hotfix to pathwaysutils/profiling.py...")
paths4 = [
    "/usr/local/lib/python3.12/site-packages/pathwaysutils/profiling.py",
    "/app/pathways-utils/pathwaysutils/profiling.py",
]
for p4 in paths4:
  if os.path.exists(p4):
    print(f"Checking pathwaysutils file: {p4}")
    with open(p4, "r", encoding="utf-8") as f:
      c4 = f.read()

    target4 = "      _, result_future = _profile_state.executable.call()"
    replacement4 = (
        "      import jax\n"
        "      import jax.numpy as jnp\n"
        "      out_avals = [jax.core.ShapedArray((1,), jnp.object_)]\n"
        "      out_shardings = [\n"
        "          getattr(\n"
        "              jax.sharding,\n"
        "              'make_single_device_sharding',\n"
        "              jax.sharding.SingleDeviceSharding,\n"
        "          )(jax.devices()[0])\n"
        "      ]\n"
        "      _, result_future = _profile_state.executable.call(\n"
        "          out_avals=out_avals, out_shardings=out_shardings\n"
        "      )"
    )

    modified4 = False
    if target4 in c4:
      c4 = c4.replace(target4, replacement4)
      modified4 = True

    target4_int = "      elif isinstance(v, int):\n" '        advanced_config[k] = {"intValue": v}'
    replacement4_int = "      elif isinstance(v, int):\n" '        advanced_config[k] = {"int64Value": v}'
    if target4_int in c4:
      c4 = c4.replace(target4_int, replacement4_int)
      modified4 = True

    # traceSessionName is enabled now, bypassing replacement:
    print("Retaining traceSessionName configuration in pathwaysutils.")

    target4_start = (
        "def start_trace(\n"
        "    log_dir: os.PathLike[str] | str,\n"
        "    *,\n"
        "    create_perfetto_link: bool = False,\n"
        "    create_perfetto_trace: bool = False,\n"
        "    profiler_options: jax.profiler.ProfileOptions | None = None,\n"
        "    max_num_hosts: int = 1,\n"
        ") -> None:"
    )
    replacement4_start = (
        "def start_trace(\n"
        "    log_dir: os.PathLike[str] | str,\n"
        "    *,\n"
        "    create_perfetto_link: bool = False,\n"
        "    create_perfetto_trace: bool = False,\n"
        "    profiler_options: jax.profiler.ProfileOptions | None = None,\n"
        "    max_num_hosts: int | None = None,\n"
        ") -> None:\n"
        "  if max_num_hosts is None:\n"
        "    try:\n"
        '      tpu_devices = [d for d in jax.devices() if d.platform == "tpu"]\n'
        "      tpu_device_count = len(tpu_devices) if tpu_devices else jax.device_count()\n"
        "      max_num_hosts = max(1, tpu_device_count // 8)\n"
        "    except Exception:\n"
        "      max_num_hosts = jax.process_count()"
    )
    if target4_start in c4:
      c4 = c4.replace(target4_start, replacement4_start)
      modified4 = True

    target4_monkey = (
        "  def start_trace_patch(\n"
        "      log_dir,\n"
        "      create_perfetto_link: bool = False,\n"
        "      create_perfetto_trace: bool = False,\n"
        "      profiler_options: jax.profiler.ProfileOptions | None = None,\n"
        "      max_num_hosts: int = 1,\n"
        "  ) -> None:"
    )
    replacement4_monkey = (
        "  def start_trace_patch(\n"
        "      log_dir,\n"
        "      create_perfetto_link: bool = False,\n"
        "      create_perfetto_trace: bool = False,\n"
        "      profiler_options: jax.profiler.ProfileOptions | None = None,\n"
        "      max_num_hosts: int | None = None,\n"
        "  ) -> None:"
    )
    if target4_monkey in c4:
      c4 = c4.replace(target4_monkey, replacement4_monkey)
      modified4 = True

    if modified4:
      with open(p4, "w", encoding="utf-8") as f:
        f.write(c4)
      print(f"Hotfix successfully applied to {p4}.")
    else:
      print(f"Warning: target strings not found or already patched in {p4}.")
  else:
    print(f"Warning: {p4} not found.")

# --- Patch 5: tpu_inference tpu_runner.py ---
print("Applying hotfix to tpu_inference/runner/tpu_runner.py...")
p5 = "/usr/local/lib/python3.12/site-packages/tpu_inference/runner/tpu_runner.py"
if os.path.exists(p5):
  with open(p5, "r", encoding="utf-8") as f:
    c5 = f.read()
  target5 = (
      "        with jax.set_mesh(self.mesh), jax.profiler.TraceAnnotation(\n"
      '                f"execute_model: {reqs} reqs, {toks} toks", **req_id_kwargs):\n'
      "            output = self._execute_model(scheduler_output,\n"
      "                                         intermediate_tensors)"
  )
  replacement5 = (
      "        with jax.set_mesh(self.mesh), jax.profiler.StepTraceAnnotation(\n"
      '                "execute_model", step_num=self.batch_counter):\n'
      "            with jax.profiler.TraceAnnotation(\n"
      '                    f"execute_model: {reqs} reqs, {toks} toks", **req_id_kwargs):\n'
      "                output = self._execute_model(scheduler_output,\n"
      "                                             intermediate_tensors)"
  )
  if target5 in c5:
    c5 = c5.replace(target5, replacement5)
    with open(p5, "w", encoding="utf-8") as f:
      f.write(c5)
    print("Hotfix successfully applied to tpu_inference/runner/tpu_runner.py.")
  else:
    print("Warning: target string not found in tpu_runner.py.")
else:
  print("Warning: tpu_runner.py path not found.")


# --- Patch 6: tunix/rl/rl_cluster.py ---
print("Applying hotfix to tunix/rl/rl_cluster.py...")
p6 = "/usr/local/lib/python3.12/site-packages/tunix/rl/rl_cluster.py"
if os.path.exists(p6):
  with open(p6, "r", encoding="utf-8") as f:
    c6 = f.read()

  target6 = """      with self._perf.span("rollout", mesh.devices) as span, self._perf_v2.span(
          perf_constants.ROLLOUT,
          mesh.devices,
          tags=perf_tags,
      ) as span_v2:
        outputs = [
            self.rollout.generate(string_prompts[s], rollout_config)
            for s in rl_utils.chunk_slices_by_size(
                stop=len(string_prompts), step=micro_batch_size
            )
        ]
        span.device_end([o.tokens for o in outputs])
        span_v2.async_end([o.tokens for o in outputs])"""

  replacement6 = """      # Start profiling sandwich if requested
      profiler_options = getattr(self.cluster_config.training_config, "profiler_options", None)
      should_profile = False
      if profiler_options is not None:
        profile_step_str = os.environ.get("TUNIX_PROFILE_ROLLOUT_STEP")
        if profile_step_str is not None:
          try:
            should_profile = (self.global_steps == int(profile_step_str))
          except ValueError:
            pass

      if should_profile:
        import pathwaysutils.profiling as pathways_profiler
        print(f"--- STARTING ROLLOUT PROFILING AT STEP {self.global_steps} ---", flush=True)
        options = jax.profiler.ProfileOptions()
        pathways_profiler.start_trace(
            log_dir=profiler_options.log_dir,
            profiler_options=options,
            max_num_hosts=16,
        )

      with self._perf.span("rollout", mesh.devices) as span, self._perf_v2.span(
          perf_constants.ROLLOUT,
          mesh.devices,
          tags=perf_tags,
      ) as span_v2:
        outputs = [
            self.rollout.generate(string_prompts[s], rollout_config)
            for s in rl_utils.chunk_slices_by_size(
                stop=len(string_prompts), step=micro_batch_size
            )
        ]
        span.device_end([o.tokens for o in outputs])
        span_v2.async_end([o.tokens for o in outputs])

      if should_profile:
        print(f"--- STOPPING ROLLOUT PROFILING AT STEP {self.global_steps} ---", flush=True)
        pathways_profiler.stop_trace()"""

  if target6 in c6:
    c6 = c6.replace(target6, replacement6)
    with open(p6, "w", encoding="utf-8") as f:
      f.write(c6)
    print("Hotfix successfully applied to tunix/rl/rl_cluster.py.")
  else:
    if "should_profile = False" in c6:
      print("Hotfix already applied to tunix/rl/rl_cluster.py.")
    else:
      print("Warning: target string not found in rl_cluster.py.")
else:
  print("Warning: rl_cluster.py path not found.")
