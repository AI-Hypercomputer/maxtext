# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import importlib
from typing import Sequence
import jax
import time
import os
try:
  from maxtext.inference import aot_cache
except ImportError:
  from maxtext.models.wan import aot_cache
from maxtext.configs import pyconfig
from maxtext.utils import max_logging, max_utils
from absl import app
from diffusers.utils import export_to_video
import flax


jax.config.update("jax_use_shardy_partitioner", True)


@contextlib.contextmanager
def transformer_engine_context():
  """If TransformerEngine is available, this context manager provides physical mesh details."""
  try:
    from transformer_engine.jax.sharding import global_shard_guard, MeshResource
    mesh_resource = MeshResource(
        dp_resource=None,
        tp_resource="tensor",
        fsdp_resource="fsdp",
        pp_resource=None,
        cp_resource="context",
    )
    with global_shard_guard(mesh_resource):
      yield
  except ImportError:
    yield


def call_pipeline(config, pipeline, prompt, negative_prompt, num_inference_steps=None):
  if num_inference_steps is None:
    num_inference_steps = config.num_inference_steps
  return pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      height=config.height,
      width=config.width,
      num_frames=config.num_frames,
      num_inference_steps=num_inference_steps,
      guidance_scale=config.guidance_scale,
      use_cfg_cache=config.use_cfg_cache,
      use_magcache=config.use_magcache,
      magcache_thresh=config.magcache_thresh,
      magcache_K=config.magcache_K,
      retention_ratio=config.retention_ratio,
      use_kv_cache=config.use_kv_cache,
  )


def _get_output_filename(config, filename_prefix: str, idx: int) -> str:
  """Generates a model-aware output video filename."""
  output_pattern = getattr(config, "output_filename_pattern", None)
  if output_pattern:
    return output_pattern.format(prefix=filename_prefix, seed=config.seed, index=idx)
  m_name = (getattr(config, "model_name", "") or getattr(config, "pretrained_model_name_or_path", "output")).lower()
  model_tag = "wan" if "wan" in m_name else m_name.split("/")[-1].split("-")[0].split("_")[0]
  return f"{filename_prefix}{model_tag}_output_{config.seed}_{idx}.mp4"


def get_pipeline_class(config):
  """Resolves the diffusion pipeline class dynamically based on model config."""
  pipeline_name = getattr(config, "pipeline_name", None)
  if pipeline_name:
    import maxtext.inference.pipelines as pipelines
    if hasattr(pipelines, pipeline_name):
      return getattr(pipelines, pipeline_name)

  # Discover architecture type from model_type, model_family, or model_name
  model_type = getattr(config, "model_type", None) or getattr(config, "model_family", None)
  if not model_type or str(model_type).lower() in ("t2v", "i2v", "t2i", "i2i"):
    name = (
        getattr(config, "model_name", "")
        or getattr(config, "pretrained_model_name_or_path", "")
    ).lower()
    if "wan" in name:
      model_type = "wan"
    elif "flux" in name:
      model_type = "flux"
    elif "sd3" in name or "stable-diffusion-3" in name:
      model_type = "sd3"
    elif "cogvideo" in name:
      model_type = "cogvideox"
    else:
      model_type = name.split("/")[-1].split("-")[0]

  if model_type == "wan":
    from maxtext.inference.pipelines.wan_pipeline import WanPipeline2_1
    return WanPipeline2_1

  try:
    module = importlib.import_module(f"maxtext.inference.pipelines.{model_type}_pipeline")
    for attr in dir(module):
      if attr.lower() == f"{model_type}pipeline" or attr.endswith("Pipeline2_1") or attr.endswith("Pipeline"):
        return getattr(module, attr)
  except ModuleNotFoundError:
    pass

  raise ValueError(
      f"Could not determine pipeline class for model '{getattr(config, 'model_name', '')}'. "
      f"Please specify 'pipeline_name' or configure a supported diffusion model family."
  )


def inference_generate_video(config, pipeline, filename_prefix=""):
  s0 = time.perf_counter()
  prompt_file = getattr(config, "prompt_file", "")
  prompts = max_utils.load_prompts(prompt_file, default_prompt=config.prompt)
  batch_size = config.global_batch_size_to_train_on
  is_multi_prompt = len(prompts) > 1 or bool(prompt_file)

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width},"
      f" frames: {config.num_frames}, total prompts: {len(prompts)}, video prefix: {filename_prefix}"
  )

  gcs_output_path = max_utils.get_gcs_output_path(config)
  saved_video_paths = []

  if not is_multi_prompt:
    prompt = [prompts[0]] * batch_size
    negative_prompt = [config.negative_prompt] * batch_size
    videos = call_pipeline(config, pipeline, prompt, negative_prompt)
    max_logging.log(f"video {filename_prefix}, generation time: {(time.perf_counter() - s0):.2f}s")
    for i in range(len(videos)):
      video_path = _get_output_filename(config, filename_prefix, i)
      export_to_video(videos[i], video_path, fps=config.fps)
      saved_video_paths.append(video_path)
      if gcs_output_path:
        max_utils.upload_file_to_gcs(gcs_output_path, video_path, subdir="videos")
        max_utils.delete_file(f"./{video_path}")
  else:
    for i, padded_chunk, actual_chunk_len in max_utils.chunk_and_pad(prompts, batch_size):
      negative_prompt = [config.negative_prompt] * batch_size

      videos = call_pipeline(config, pipeline, padded_chunk, negative_prompt)
      for j in range(actual_chunk_len):
        prompt_idx = i + j
        video_path = _get_output_filename(config, filename_prefix, prompt_idx)
        export_to_video(videos[j], video_path, fps=config.fps)
        saved_video_paths.append(video_path)
        if gcs_output_path:
          max_utils.upload_file_to_gcs(gcs_output_path, video_path, subdir="videos")
          max_utils.delete_file(f"./{video_path}")
    max_logging.log(f"all videos {filename_prefix}, total generation time: {(time.perf_counter() - s0):.2f}s")

  return saved_video_paths


def run(config, pipeline=None, filename_prefix="", commit_hash=None):
  writer = max_utils.initialize_summary_writer(config)
  if jax.process_index() == 0 and writer:
    max_logging.log(f"TensorBoard logs will be written to: {config.tensorboard_dir}")

    if commit_hash:
      writer.add_text("inference/git_commit_hash", commit_hash, global_step=0)
      max_logging.log(f"Git Commit Hash: {commit_hash}")
    else:
      max_logging.log("Could not retrieve Git commit hash.")

  if pipeline is None:
    load_start = time.perf_counter()
    checkpoint_dir = getattr(config, "checkpoint_directory", None) or getattr(config, "checkpoint_dir", None)
    pipeline_cls = get_pipeline_class(config)
    if checkpoint_dir:
      pipeline = pipeline_cls.from_checkpoint(config, checkpoint_dir=checkpoint_dir)
    else:
      pipeline = pipeline_cls.from_pretrained(config)
    load_time = time.perf_counter() - load_start
    max_logging.log(f"load_time: {load_time:.1f}s")
  else:
    load_time = 0.0

  # Per-shape AOT executable cache: deserialization starts on background
  # threads now and overlaps the remaining setup; unknown shapes silently
  # fall back to jit and are serialized by save_pending() after warmup.
  aot_cache.install(
      getattr(config, "aot_cache_dir", ""),
      meta={
          "model": config.pretrained_model_name_or_path,
          "attention": config.attention,
          # Kernel block sizes change the lowered graph, not the input
          # shapes — they must key the executable or a re-tuned config
          # would silently hit stale binaries.
          "flash_block_sizes": str(config.flash_block_sizes),
          "mesh_shape": str(pipeline.mesh.shape),
          "vae_spatial": str(config.vae_spatial),
          "vae_decode_chunk": str(config.vae_decode_chunk),
          "weights_dtype": str(config.weights_dtype),
          "activations_dtype": str(config.activations_dtype),
          "scan_layers": str(config.scan_layers),
          "jax": jax.__version__,
      },
      mesh=pipeline.mesh,
  )
  # Deserialization is seconds and warmup must see the loaded executables
  # to hit them; without this the first call races the loader threads.
  aot_cache.wait_for_loads()

  s0 = time.perf_counter()

  # Disable profiler for the first two runs to avoid duplicate uploads
  original_enable_profiler = config.enable_profiler if "enable_profiler" in config.get_keys() else False
  config.get_keys()["enable_profiler"] = False

  prompt_file = getattr(config, "prompt_file", "")
  prompts = max_utils.load_prompts(prompt_file, default_prompt=config.prompt)
  batch_size = config.global_batch_size_to_train_on
  is_multi_prompt = len(prompts) > 1 or bool(prompt_file)

  # Using global_batch_size_to_train_on so not to create more config variables
  warmup_prompt = [prompts[0]] * batch_size
  warmup_negative_prompt = [config.negative_prompt] * batch_size

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width},"
      f" frames: {config.num_frames}, total prompts: {len(prompts)}"
  )
  # Warmup with 2 denoising steps instead of a full run: step 0 runs the
  # high-noise transformer and step 1 crosses the boundary to the low-noise
  # one (e.g. multi-stage denoisers), so every executable of the full run (transformers,
  # text encoder, VAE decode) gets compiled at a fraction of the cost. The
  # step count only changes the Python loop trip count, not traced shapes.
  warmup_steps = min(2, config.num_inference_steps)
  max_logging.log(f"Compile warmup: {warmup_steps} denoising steps")
  # Zero-execution warmup: wrapped transformer passes lower+compile (or
  # reuse the deserialized AOT executable) and return sharded zeros, so
  # the warmup pays compile time only, never real denoise compute. The
  # returned videos are garbage by design and are discarded below.
  with aot_cache.warmup_mode():
    videos = call_pipeline(config, pipeline, warmup_prompt, warmup_negative_prompt, num_inference_steps=warmup_steps)
  if isinstance(videos, tuple):
    videos, warmup_trace = videos
    warmup_str = ", ".join(f"{stage}={seconds:.1f}s" for stage, seconds in warmup_trace.items())
    max_logging.log(f"Warmup breakdown: {warmup_str}")

  # Serialize any newly-compiled shapes synchronously while still inside
  # warmup-accounted time; a background save would compete with the first
  # real generation (DiffusionServing PR#39 first-generation-stall lesson).
  aot_cache.save_pending()

  max_logging.log("===================== Model details =======================")
  max_logging.log(f"model name: {config.model_name}")
  max_logging.log(f"model path: {config.pretrained_model_name_or_path}")
  max_logging.log(f"model type: {config.model_type}")
  max_logging.log(f"hardware: {jax.devices()[0].platform}")
  max_logging.log(f"number of devices: {jax.device_count()}")
  max_logging.log(f"per_device_batch_size: {config.per_device_batch_size}")
  max_logging.log(f"total prompts to generate: {len(prompts)}")
  max_logging.log("============================================================")

  compile_time = time.perf_counter() - s0
  max_logging.log(f"compile_time: {compile_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/compile_time", compile_time, global_step=0)

  s0 = time.perf_counter()
  saved_video_path = []
  gcs_output_path = max_utils.get_gcs_output_path(config)

  if not is_multi_prompt:
    prompt = [prompts[0]] * batch_size
    negative_prompt = [config.negative_prompt] * batch_size
    outputs = call_pipeline(config, pipeline, prompt, negative_prompt)
    if isinstance(outputs, tuple):
      videos, trace = outputs
    else:
      videos = outputs
      trace = {}
    for i in range(len(videos)):
      video_path = _get_output_filename(config, filename_prefix, i)
      export_to_video(videos[i], video_path, fps=config.fps)
      saved_video_path.append(video_path)
      if gcs_output_path:
        max_utils.upload_file_to_gcs(gcs_output_path, video_path, subdir="videos")
  else:
    trace = {}
    for i, padded_chunk, actual_chunk_len in max_utils.chunk_and_pad(prompts, batch_size):
      negative_prompt = [config.negative_prompt] * batch_size

      outputs = call_pipeline(config, pipeline, padded_chunk, negative_prompt)
      if isinstance(outputs, tuple):
        videos, trace = outputs
      else:
        videos = outputs
      for j in range(actual_chunk_len):
        prompt_idx = i + j
        video_path = _get_output_filename(config, filename_prefix, prompt_idx)
        export_to_video(videos[j], video_path, fps=config.fps)
        saved_video_path.append(video_path)
        if gcs_output_path:
          max_utils.upload_file_to_gcs(gcs_output_path, video_path, subdir="videos")

  generation_time = time.perf_counter() - s0
  max_logging.log(f"generation_time: {generation_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/generation_time", generation_time, global_step=0)
    num_videos = len(saved_video_path)
    if num_videos > 0:
      generation_time_per_video = generation_time / num_videos
      writer.add_scalar(
          "inference/generation_time_per_video",
          generation_time_per_video,
          global_step=0,
      )
      max_logging.log(f"generation time per video: {generation_time_per_video}")
    else:
      max_logging.log("Warning: Number of videos is zero, cannot calculate generation_time_per_video.")
  summary = [
      f"\n{'=' * 50}",
      "  TIMING SUMMARY",
      f"{'=' * 50}",
      f"  Load (checkpoint):   {load_time:>7.1f}s",
      f"  Compile:             {compile_time:>7.1f}s",
      f"  Inference:           {generation_time:>7.1f}s",
  ]
  if trace:
    vae_decode_total = trace.get("vae_decode", 0.0)
    vae_decode_tpu = trace.get("vae_decode_tpu", 0.0)
    vae_decode_post = vae_decode_total - vae_decode_tpu
    summary.extend([
        f"  {'─' * 40}",
        f"  Conditioning:        {trace.get('conditioning', 0.0):>7.1f}s",
        f"    - VAE Encode:      {trace.get('vae_encode', 0.0):>7.1f}s",
        f"  Denoise Total:       {trace.get('denoise_total', 0.0):>7.1f}s",
        f"  VAE Decode:          {vae_decode_total:>7.1f}s",
        f"    - TPU Compute:     {vae_decode_tpu:>7.1f}s",
        f"    - Host Formatting: {vae_decode_post:>7.1f}s",
    ])
  summary.append(f"{'=' * 50}")
  max_logging.log("\n".join(summary))

  s0 = time.perf_counter()
  # Restore original profiler setting for the profiling run
  config.get_keys()["enable_profiler"] = original_enable_profiler
  if original_enable_profiler:
    # Injecting user requested XLA tracing flags
    xla_flags = os.environ.get("XLA_FLAGS", "")
    new_flags = "--xla_enable_mxu_trace=true --xla_jf_dump_llo_html=true --xla_tpu_enable_llo_profiling=true"
    os.environ["XLA_FLAGS"] = f"{xla_flags} {new_flags}"
    max_logging.log(f"Injected XLA_FLAGS for profiling: {new_flags}")

    profiler_prompt = [prompts[0]] * batch_size
    profiler_negative_prompt = [config.negative_prompt] * batch_size
    videos = call_pipeline(config, pipeline, profiler_prompt, profiler_negative_prompt)
    if isinstance(videos, tuple):
      videos = videos[0]
    generation_time_with_profiler = time.perf_counter() - s0
    max_logging.log(f"generation_time_with_profiler: {generation_time_with_profiler}")
    if writer and jax.process_index() == 0:
      writer.add_scalar(
          "inference/generation_time_with_profiler",
          generation_time_with_profiler,
          global_step=0,
      )

  return saved_video_path


def main(argv: Sequence[str]) -> None:
  commit_hash = max_utils.get_git_commit_hash()
  pyconfig.initialize(argv)
  try:
    flax.config.update("flax_always_shard_variable", False)
  except LookupError:
    pass
  run(pyconfig.config, commit_hash=commit_hash)


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
