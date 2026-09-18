# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Autoregressive Video Decoding for Stitched Qwen3-VL-4B + Qwen3-14B Model.

Evaluates stitched Omni checkpoints on video input (.mp4) using native
spatio-temporal patch merging and fast KV-cached autoregressive decoding.

Usage:
  python3 src/maxtext/experimental/omni_qwen_14b_3stage/decode_video_omni_qwen_14b.py \
    --config_path="src/maxtext/experimental/omni_qwen_14b_3stage/maxtext-omni-qwen3-vl-14b.yml" \
    --checkpoint_path="gs://your-bucket/omni_checkpoints/omni_qwen3_vl_14b_3stage/0/items" \
    --video_path="tests/assets/test_video.mp4" \
    --prompt="<|video|> Describe what you can see in this video."
"""

import maxtext
# Eagerly initialize core MaxText C++ and model dependencies
_ = (maxtext.Mesh, maxtext.pyconfig, maxtext.models, maxtext.model_creation_utils)

import functools
import os
import sys
from absl import app, flags
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import omegaconf
from transformers import AutoTokenizer

from maxtext.common import checkpointing
from maxtext.configs import pyconfig
from maxtext.multimodal import processor as mm_processor
from maxtext.multimodal import processor_qwen3_omni
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR

FLAGS = flags.FLAGS


def _define_flag(fn, name, default, help_str):
  if name not in FLAGS:
    fn(name, default, help_str)


_REPO_ROOT = os.environ.get("MAXTEXT_REPO_ROOT", os.path.abspath(os.path.join(MAXTEXT_PKG_DIR, "..", "..")))
_DEFAULT_TEST_VIDEO = os.path.join(_REPO_ROOT, "tests", "assets", "test_video.mp4")
_DEFAULT_CHECKPOINT = (
    "gs://your-bucket/experimental/omni_qwen3_vl_14b_4stage/"
    "stage1_coco_narratives/omni_qwen3_vl_14b_4stage_stage1_coco/checkpoints/1999/items"
)

_define_flag(
    flags.DEFINE_string,
    "config_path",
    os.path.join(
        MAXTEXT_PKG_DIR,
        "experimental",
        "omni_qwen_14b_3stage",
        "maxtext-omni-qwen3-vl-14b.yml",
    ),
    "Path to model YAML configuration.",
)
_define_flag(flags.DEFINE_string, "checkpoint_path", _DEFAULT_CHECKPOINT, "Path to checkpoint directory.")
_define_flag(flags.DEFINE_boolean, "text_only", False, "Enable pure text-only decoding without visual input.")
_define_flag(flags.DEFINE_string, "prompt", "", "Custom prompt for video or text decoding.")
_define_flag(flags.DEFINE_string, "video_path", _DEFAULT_TEST_VIDEO, "Path to .mp4 video file.")
_define_flag(flags.DEFINE_integer, "num_frames", 4, "Number of frames to extract from video.")
_define_flag(flags.DEFINE_integer, "max_new_tokens", 128, "Maximum autoregressive tokens to generate.")
_define_flag(flags.DEFINE_string, "description", "Omni Qwen3-VL 14B Video Decode", "Description.")


def normalize_checkpoint_path(path: str) -> str:
  path = path.rstrip("/")
  if not path.endswith("/items"):
    path = f"{path}/items"
  return path


def initialize_model_and_weights(config, checkpoint_path, mesh):
  """Initializes MaxText model and restores weights into sharded mesh."""
  with jax.set_mesh(mesh):
    model = maxtext.model_creation_utils.from_config(config, mesh=mesh)

    abstract_vars = maxtext_utils.get_abstract_param(model, config)
    target_params_abstract = max_utils.unbox_logicallypartioned(abstract_vars["params"])

    max_logging.log(f">>> Restoring checkpoint from: {checkpoint_path}")
    restored = checkpointing.load_params_from_path(
        checkpoint_path,
        {"params": target_params_abstract},
        config.checkpoint_storage_concurrent_gb,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
    )
    return model, restored.get("params", restored)


def load_omni_config(yaml_path, checkpoint_path):
  """Loads YAML and initializes MaxText config for inference."""
  custom_cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(yaml_path), resolve=True)
  num_devs = len(jax.devices())
  per_dev_bs = 1.0 / num_devs
  base_yml = os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")

  argv = [
      sys.argv[0],
      base_yml,
      "override_model_config=True",
      "skip_jax_distributed_system=True",
      "ici_fsdp_parallelism=1",
      "ici_data_parallelism=1",
      "ici_autoregressive_parallelism=1",
      "ici_tensor_parallelism=-1",
      "attention=autoselected",
      "scan_layers=false",
      "dtype=bfloat16",
      "weight_dtype=bfloat16",
      f"per_device_batch_size={per_dev_bs}",
      "async_checkpointing=False",
      f"load_parameters_path={checkpoint_path}",
      "video_max_grid_t=32",
      "video_max_grid_h=32",
      "video_max_grid_w=32",
      "use_audio_in_video=False",
      "max_prefill_predict_length=512",
      "max_target_length=1024",
  ]

  omni_skip_keys = {
      "llm_model_name",
      "base_config",
      "model_name",
      "per_device_batch_size",
      "eval_per_device_batch_size",
      "ici_fsdp_parallelism",
      "ici_data_parallelism",
      "ici_tensor_parallelism",
      "ici_autoregressive_parallelism",
      "ici_expert_parallelism",
  }
  for k, v in custom_cfg.items():
    if k in omni_skip_keys:
      continue
    if isinstance(v, str):
      argv.append(f"{k}='{v}'")
    else:
      argv.append(f"{k}={v}")

  config = pyconfig.initialize(
      argv,
      override_model_config=True,
      skip_jax_distributed_system=True,
      log_config=False,
  )
  object.__setattr__(config, "model_name", "maxtext-omni-qwen3-vl-14b")
  object.__setattr__(config, "micro_batch_size_to_train_on", 1)
  object.__setattr__(config, "global_batch_size_to_train_on", 1)
  object.__setattr__(config, "per_device_batch_size", per_dev_bs)
  return config


def load_or_generate_video(video_path: str, num_frames: int = 4, height: int = 384, width: int = 384) -> np.ndarray:
  """Loads video frames from .mp4 or synthesizes animated frames if not found."""
  resolved_path = None
  candidates = [
      video_path,
      os.path.expanduser(video_path),
      os.path.join(_REPO_ROOT, video_path),
      _DEFAULT_TEST_VIDEO,
  ]
  for candidate in candidates:
    if os.path.exists(candidate):
      resolved_path = candidate
      break

  if resolved_path:
    try:
      import decord # pylint: disable=import-outside-toplevel
      vr = decord.VideoReader(resolved_path)
      total_frames = len(vr)
      frame_idx = np.linspace(0, total_frames - 1, num_frames, dtype=int)
      frames = vr.get_batch(frame_idx).asnumpy()  # (T, H, W, C)
      video_array = np.transpose(frames, (0, 3, 1, 2))  # (T, C, H, W)
      max_logging.log(f"Loaded {len(video_array)} frames ({video_array.shape}) from {resolved_path} via decord")
      return video_array
    except Exception as e:
      max_logging.log(f"Decord video reader failed: {e}. Generating synthetic video.")

  # Generate synthetic moving gradient frames (T, C, H, W)
  max_logging.log(f"Generating synthetic animated video with {num_frames} frames ({height}x{width})")
  actual_frames = num_frames if num_frames % 2 == 0 else num_frames + 1
  frames = []
  for t in range(actual_frames):
    phase = 2 * np.pi * t / actual_frames
    y_grad = np.linspace(0, 255, height)[:, None]
    x_grad = np.linspace(0, 255, width)[None, :]
    r = ((y_grad + 128 * np.sin(phase)) % 256).astype(np.uint8)
    g = ((x_grad + 128 * np.cos(phase)) % 256).astype(np.uint8)
    b = (np.full((height, width), int(128 + 127 * np.sin(phase)))).astype(np.uint8)
    frame = np.stack([r, g, b], axis=0)
    frames.append(frame)
  return np.array(frames, dtype=np.uint8)


@functools.partial(jax.jit, static_argnums=(0,))
def _prefill_step_text(model, params, tokens, positions, segment_ids):
  """Prefill step for pure text."""
  logits, mutated_vars = model.apply(
      {"params": params},
      decoder_input_tokens=tokens,
      decoder_positions=positions,
      decoder_segment_ids=segment_ids,
      enable_dropout=False,
      model_mode="prefill",
      mutable=["cache"],
  )
  return logits, mutated_vars["cache"]


@functools.partial(jax.jit, static_argnums=(0,))
def _prefill_step_video(model, params, tokens, positions, segment_ids, videos, video_masks, video_grid_thw):
  """Prefill step with video embeddings."""
  logits, mutated_vars = model.apply(
      {"params": params},
      decoder_input_tokens=tokens,
      decoder_positions=positions,
      decoder_segment_ids=segment_ids,
      encoder_videos=videos,
      encoder_video_masks=video_masks,
      encoder_video_grid_thw=video_grid_thw,
      enable_dropout=False,
      model_mode="prefill",
      mutable=["cache"],
  )
  return logits, mutated_vars["cache"]


@functools.partial(jax.jit, static_argnums=(0,))
def _ar_step(model, params, cache, token, position):
  """Executes a single-token autoregressive step reusing the KV cache."""
  logits, mutated_vars = model.apply(
      {"params": params, "cache": cache},
      decoder_input_tokens=token,
      decoder_positions=position,
      decoder_segment_ids=None,
      enable_dropout=False,
      model_mode="autoregressive",
      mutable=["cache"],
  )
  return logits, mutated_vars["cache"]


@functools.partial(jax.jit, static_argnums=(0, 5))
def _generate_loop(model, params, cache, first_token, start_pos, num_steps: int):
  """Executes remaining autoregressive steps on TPU using jax.lax.scan without host syncs."""

  def step_fn(carry, _):
    current_cache, current_token, current_pos = carry
    logits, mutated_vars = model.apply(
        {"params": params, "cache": current_cache},
        decoder_input_tokens=current_token,
        decoder_positions=current_pos,
        decoder_segment_ids=None,
        enable_dropout=False,
        model_mode="autoregressive",
        mutable=["cache"],
    )
    next_token = jnp.argmax(logits[:, 0:1, :], axis=-1).astype(jnp.int32)
    next_pos = current_pos + 1
    new_cache = mutated_vars["cache"]
    return (new_cache, next_token, next_pos), next_token

  init_carry = (cache, first_token, start_pos)
  _, generated_tokens_seq = jax.lax.scan(step_fn, init_carry, None, length=num_steps)
  return generated_tokens_seq


def decode_text_sample(model, params, config, mesh, tokenizer, prompt_str, max_new_tokens=128):
  """Decodes pure text sample."""
  formatted_prompt = f"<|im_start|>user\n{prompt_str}<|im_end|>\n<|im_start|>assistant\n"
  initial_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
  true_length = len(initial_tokens)
  prefill_len = config.max_prefill_predict_length
  pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
  padded_tokens = initial_tokens + [pad_token] * (prefill_len - true_length)
  tokens = np.array([padded_tokens[:prefill_len]], dtype=np.int32)
  positions = np.tile(np.arange(prefill_len, dtype=np.int32), (1, 1))
  segment_ids = np.zeros((1, prefill_len), dtype=np.int32)
  segment_ids[:, :true_length] = 1

  eos_token_id = tokenizer.eos_token_id
  im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
  tokens_to_decode = []

  with jax.set_mesh(mesh):
    logits, cache = _prefill_step_text(model, params, tokens, positions, segment_ids)
    first_token_id = int(jnp.argmax(logits[0, true_length - 1, :]))
    tokens_to_decode.append(first_token_id)

    if max_new_tokens > 1 and first_token_id not in (eos_token_id, im_end_id):
      first_token_arr = jnp.array([[first_token_id]], dtype=np.int32)
      first_pos_arr = jnp.array([[true_length]], dtype=np.int32)
      logits_1, ar_cache = _ar_step(model, params, cache, first_token_arr, first_pos_arr)
      second_token_id = int(jnp.argmax(logits_1[0, 0, :]))
      tokens_to_decode.append(second_token_id)

      num_scan_steps = min(max_new_tokens - 2, config.max_target_length - config.max_prefill_predict_length - 2)
      num_scan_steps = max(0, num_scan_steps)
      if num_scan_steps > 0 and second_token_id not in (eos_token_id, im_end_id):
        second_token_arr = jnp.array([[second_token_id]], dtype=np.int32)
        second_pos_arr = jnp.array([[true_length + 1]], dtype=np.int32)
        ar_tokens_seq = _generate_loop(model, params, ar_cache, second_token_arr, second_pos_arr, num_scan_steps)
        for tok in np.array(ar_tokens_seq).reshape(-1).tolist():
          if tok in (eos_token_id, im_end_id):
            break
          tokens_to_decode.append(tok)

  decoded_text = tokenizer.decode(tokens_to_decode, skip_special_tokens=True).strip()
  return decoded_text, tokens_to_decode


def decode_video_omni_sample(model, params, config, mesh, tokenizer, prompt_str, video_array, max_new_tokens=128):
  """Prefills video frames through Qwen3 ViT + MLP Projector and decodes response."""
  video_processed, video_grid_thw = processor_qwen3_omni.preprocess_video(video_array, config)
  video_values = np.reshape(
      video_processed,
      (
          1,
          config.num_channels_for_vit,
          config.temporal_patch_size_for_vit * int(video_grid_thw[0, 0]),
          config.patch_size_for_vit * int(video_grid_thw[0, 1]),
          config.patch_size_for_vit * int(video_grid_thw[0, 2]),
      ),
  )
  video_mask = None
  processor_outputs = processor_qwen3_omni.Qwen3OmniPreprocessorOutput(
      video_values=video_values,
      video_grid_thw=video_grid_thw,
      video_mask=None,
      video_second_per_grid=np.asarray([config.temporal_patch_size_for_vit], dtype=np.float32),
      num_videos=1,
  )
  dtype = jnp.bfloat16 if config.dtype == "bfloat16" else jnp.float32
  video_values_jax = jnp.array(video_values, dtype=dtype)
  video_mask_jax = None
  video_grid_thw_jax = jnp.array(video_grid_thw, dtype=jnp.int32)

  # 2. Format prompt and expand video placeholder tokens
  formatted_prompt = mm_processor.reformat_prompt(
      prompt=prompt_str,
      image_placeholder=config.image_placeholder,
      video_placeholder=config.video_placeholder,
      model_name=config,
      num_images=0,
      num_videos=1,
  )

  # Ensure special tokens are recognized
  initial_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
  combined_tokens = mm_processor.prepare_text_for_image_fusion(
      tokens=initial_tokens,
      config=config,
      processor_output=processor_outputs,
  ).tolist()

  # 3. Construct padded sequences
  true_length = len(combined_tokens)
  prefill_len = config.max_prefill_predict_length
  if true_length > prefill_len:
    raise ValueError(
        f"Video + prompt tokens ({true_length}) exceeds max_prefill_predict_length ({prefill_len}). "
        "Increase max_prefill_predict_length."
    )

  pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
  padded_tokens = combined_tokens + [pad_token] * (prefill_len - true_length)
  tokens = np.array([padded_tokens[:prefill_len]], dtype=np.int32)
  positions = np.tile(np.arange(prefill_len, dtype=np.int32), (1, 1))
  segment_ids = np.zeros((1, prefill_len), dtype=np.int32)
  segment_ids[:, :true_length] = 1

  # 4. Prefill and AR decoding with KV Cache
  eos_token_id = tokenizer.eos_token_id
  im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
  tokens_to_decode = []

  with jax.set_mesh(mesh):
    logits, cache = _prefill_step_video(
        model,
        params,
        tokens,
        positions,
        segment_ids,
        video_values_jax,
        video_mask_jax,
        video_grid_thw_jax,
    )
    first_token_id = int(jnp.argmax(logits[0, true_length - 1, :]))
    tokens_to_decode.append(first_token_id)

    if max_new_tokens > 1 and first_token_id not in (eos_token_id, im_end_id):
      first_token_arr = jnp.array([[first_token_id]], dtype=np.int32)
      first_pos_arr = jnp.array([[true_length]], dtype=np.int32)
      logits_1, ar_cache = _ar_step(model, params, cache, first_token_arr, first_pos_arr)
      second_token_id = int(jnp.argmax(logits_1[0, 0, :]))
      tokens_to_decode.append(second_token_id)

      num_scan_steps = min(max_new_tokens - 2, config.max_target_length - config.max_prefill_predict_length - 2)
      num_scan_steps = max(0, num_scan_steps)
      if num_scan_steps > 0 and second_token_id not in (eos_token_id, im_end_id):
        second_token_arr = jnp.array([[second_token_id]], dtype=np.int32)
        second_pos_arr = jnp.array([[true_length + 1]], dtype=np.int32)
        ar_tokens_seq = _generate_loop(model, params, ar_cache, second_token_arr, second_pos_arr, num_scan_steps)
        for tok in np.array(ar_tokens_seq).reshape(-1).tolist():
          if tok in (eos_token_id, im_end_id):
            break
          tokens_to_decode.append(tok)

  decoded_text = tokenizer.decode(tokens_to_decode, skip_special_tokens=True).strip()
  return decoded_text, tokens_to_decode


def run_evaluation(checkpoint_path, config, description="Omni Qwen3-VL 14B Video Decode", max_new_tokens=128):
  """Runs evaluation on video input or text prompt."""
  max_logging.log("=" * 70)
  max_logging.log(f">>> [VIDEO DECODE EVALUATION] {description}")
  max_logging.log(f"    Checkpoint: {checkpoint_path}")
  max_logging.log(f"    Mode:       {'Text-Only' if FLAGS.text_only else 'Video Multimodal'}")
  max_logging.log("=" * 70)

  if jax.local_devices()[0].platform == "cpu":
    mesh = Mesh(np.array([jax.devices("cpu")[0]]), axis_names=("data",))
  else:
    mesh = maxtext_utils.get_mesh_from_config(config)

  model, restored_params = initialize_model_and_weights(config, checkpoint_path, mesh)
  tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

  # Text-only evaluation
  if FLAGS.text_only:
    prompt_str = FLAGS.prompt or "Explain how video transformer architectures process multi-frame sequences."
    max_logging.log("\n--- [Text-Only Sample] ---")
    response, token_ids = decode_text_sample(
        model=model,
        params=restored_params,
        config=config,
        mesh=mesh,
        tokenizer=tokenizer,
        prompt_str=prompt_str,
        max_new_tokens=max_new_tokens,
    )
    max_logging.log(
        f"""
----------------------------------------------------------------------
  Question:       {prompt_str}
  Generated IDs:  {token_ids}
  Model Response: {response}
----------------------------------------------------------------------"""
    )
    return

  # Video evaluation
  video_array = load_or_generate_video(
      video_path=FLAGS.video_path,
      num_frames=FLAGS.num_frames,
      height=config.image_size_for_vit // 2,
      width=config.image_size_for_vit // 2,
  )
  prompt_str = FLAGS.prompt or "<|video|> Describe the sequence of actions in this video clip."
  max_logging.log(f"Video input frames shape: {video_array.shape}")
  max_logging.log(f"Video prompt: {prompt_str}")

  response, token_ids = decode_video_omni_sample(
      model=model,
      params=restored_params,
      config=config,
      mesh=mesh,
      tokenizer=tokenizer,
      prompt_str=prompt_str,
      video_array=video_array,
      max_new_tokens=max_new_tokens,
  )

  max_logging.log(
      f"""
======================================================================
[Video Evaluation Result]
  Prompt:         {prompt_str}
  Video Frames:   {video_array.shape[0]} frames ({video_array.shape})
  Generated IDs:  {token_ids}
  Model Response: {response}
======================================================================"""
  )


def main(argv):
  config_path = FLAGS.config_path
  checkpoint_path = FLAGS.checkpoint_path
  assert checkpoint_path, "Must specify --checkpoint_path"

  config = load_omni_config(config_path, checkpoint_path)
  run_evaluation(
      checkpoint_path=checkpoint_path,
      config=config,
      description=FLAGS.description,
      max_new_tokens=FLAGS.max_new_tokens,
  )


if __name__ == "__main__":
  app.run(main)
