# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LingBot-World-Fast image + camera-trajectory -> video pipeline.

Reuses the Wan diffusion pipeline (`m3/diffuser_pipelines/wan_pipeline.py`) for everything but the denoising loop:
mesh creation, the UMT5 text encoder and its prompt encoding, the Wan 2.1 VAE (encode / decode passes), checkpoint
restore and sharded weight placement. LingBot-specific parts (mirroring `wan/image2video_fast.py`):

* Inputs: a prompt, a first frame and an action sequence (camera trajectory `poses.npy` / `intrinsics.npy`).
* I2V conditioning by channel concatenation: [first-frame mask (4), VAE latents of first frame + zero frames (16)].
* Few-step causal autoregressive sampling: latent frames are generated `chunk_size` at a time. Each chunk is denoised
  with the distilled 4-step schedule (predict x0, re-noise to the next timestep) while attending to the KV cache of the
  previous chunks; its clean x0 is then run once at t=0 to commit its keys / values to the cache. No CFG.
"""

import time
from functools import partial
from typing import List, Optional, Union

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
import PIL
from transformers import AutoTokenizer

from maxtext.m3.configs.diffusers_config import DiffusionConfig
from maxtext.m3.diffuser_pipelines.pipeline_utils import _restored_component
from maxtext.m3.diffuser_pipelines.wan_pipeline import (
    WanPipeline,
    cast_with_exclusion,
    put_params_into_state,
    state_shardings,
    vae_encode_pass,
)
from maxtext.m3.models.lingbot import camera_lingbot
from maxtext.m3.models.lingbot.transformer_lingbot import LingBotWorldModel, rope_cos_sin
from maxtext.m3.models.wan import WAN_MESH_AXES, WanSharding
from maxtext.m3.core.sharding import TensorType
from maxtext.m3.models.wan.utils_wan import SELF_ATTN_HEAD, get_flash_block_sizes, get_precision
from maxtext.utils import max_logging

# =============================================================================
# 1. Distilled few-step schedule
# =============================================================================


class LingBotFastScheduler:
  """The fixed few-step flow-matching schedule of `WanI2VFast` (host-side numpy).

  The reference builds `FlowUniPCMultistepScheduler(shift=1)`, calls `set_timesteps(1000, shift=sample_shift)` and keeps
  `timesteps[timesteps_index]`. Two sigma lookups are reproduced exactly: `_convert_flow_pred_to_x0` takes the first
  timestep closest to t, `add_noise` (`index_for_timestep`) the second exact match when the int64 timesteps repeat.
  """

  def __init__(self, num_train_timesteps: int = 1000, shift: float = 10.0, timesteps_index=(0, 179, 358, 679)):
    n = num_train_timesteps
    alphas = np.linspace(1, 1 / n, n)[::-1].copy()
    base_sigmas = (1.0 - alphas).astype(np.float32)
    sigma_max, sigma_min = float(base_sigmas[0]), float(base_sigmas[-1])
    sigmas = np.linspace(sigma_max, sigma_min, n + 1).copy()[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    self.all_timesteps = (sigmas * n).astype(np.int64)
    self.all_sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
    self.timesteps = self.all_timesteps[list(timesteps_index)]

  def x0_sigma(self, timestep) -> float:
    """sigma used by `_convert_flow_pred_to_x0` (float64)."""
    idx = int(np.argmin(np.abs(self.all_timesteps.astype(np.float64) - float(timestep))))
    return float(self.all_sigmas.astype(np.float64)[idx])

  def noise_sigma(self, timestep) -> np.float32:
    """sigma used by `add_noise` (float32)."""
    indices = np.nonzero(self.all_timesteps == timestep)[0]
    return self.all_sigmas[indices[1 if len(indices) > 1 else 0]]


# =============================================================================
# 2. Host-side input preprocessing
# =============================================================================


def _cubic_coefficients(t: np.ndarray, a: float = -0.75) -> np.ndarray:
  """PyTorch's bicubic convolution weights (A = -0.75) for fractional offsets t -> [..., 4]."""

  def conv1(x):
    return ((a + 2) * x - (a + 3)) * x * x + 1

  def conv2(x):
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a

  return np.stack([conv2(t + 1), conv1(t), conv1(1 - t), conv2(2 - t)], axis=-1)


def _resize_axis_bicubic(x: np.ndarray, out_size: int, axis: int) -> np.ndarray:
  """1D bicubic resize of `axis` to `out_size` (PyTorch semantics)."""
  in_size = x.shape[axis]
  if in_size == out_size:
    return x
  scale = in_size / out_size
  real = scale * (np.arange(out_size, dtype=np.float64) + 0.5) - 0.5  # align_corners=False
  index = np.floor(real).astype(np.int64)
  weights = _cubic_coefficients((real - index).astype(np.float32))  # [out, 4]
  taps = np.clip(index[:, None] + np.arange(-1, 3)[None, :], 0, in_size - 1)  # [out, 4]
  x = np.moveaxis(x, axis, -1)
  out = np.einsum("...ok,ok->...o", x[..., taps], weights.astype(x.dtype))
  return np.moveaxis(out, -1, axis)


def resize_bicubic(image: np.ndarray, height: int, width: int) -> np.ndarray:
  """`F.interpolate(mode='bicubic', align_corners=False)` (no antialias) of a [..., H, W] float32 array."""
  return _resize_axis_bicubic(_resize_axis_bicubic(image, height, -2), width, -1).astype(np.float32)


def load_image(image) -> np.ndarray:
  """PIL image / path -> [3, H, W] float32 in [-1, 1] (`TF.to_tensor(img).sub_(0.5).div_(0.5)`)."""
  if isinstance(image, str):
    image = PIL.Image.open(image)
  image = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
  return (np.transpose(image, (2, 0, 1)) - 0.5) / 0.5


def first_frame_mask(num_frames: int, latent_frames: int, latent_height: int, latent_width: int) -> np.ndarray:
  """[4, latent_frames, h, w] I2V mask: 1 on the (4x repeated) first frame, 0 elsewhere, folded like the VAE stride."""
  msk = np.zeros((num_frames + 3, latent_height, latent_width), np.float32)
  msk[:4] = 1.0
  msk = msk.reshape((latent_frames, 4, latent_height, latent_width))
  return np.transpose(msk, (1, 0, 2, 3))


def latent_geometry(image_hw, max_area: int, num_frames: int, chunk_size: int, vae_stride=(4, 8, 8), patch=(1, 2, 2)):
  """(height, width, lat_h, lat_w, lat_f, num_frames) exactly as `WanI2VFast.generate` derives them."""
  h0, w0 = image_hw
  aspect_ratio = h0 / w0
  lat_h = round(np.sqrt(max_area * aspect_ratio) // vae_stride[1] // patch[1] * patch[1])
  lat_w = round(np.sqrt(max_area / aspect_ratio) // vae_stride[2] // patch[2] * patch[2])
  lat_f = (num_frames - 1) // vae_stride[0] + 1
  lat_f = int(lat_f - (lat_f % chunk_size))
  return lat_h * vae_stride[1], lat_w * vae_stride[2], lat_h, lat_w, lat_f, (lat_f - 1) * vae_stride[0] + 1


# =============================================================================
# 3. Jitted steps
# =============================================================================


@partial(jax.jit, static_argnums=(0,))
def transformer_chunk_pass(
    graphdef,
    state,
    rest_of_state,
    latents,
    timestep,
    condition,
    cam,
    cross_kv,
    rope_cos,
    rope_sin,
    kv_cache,
    current_start,
):
  """One causal DiT forward over a chunk -> (flow prediction, (chunk keys, chunk values))."""
  model = nnx.merge(graphdef, state, rest_of_state)
  return model(latents, timestep, condition, cam, cross_kv, rope_cos, rope_sin, kv_cache, current_start)


@partial(jax.jit, static_argnums=(0,))
def cross_kv_pass(graphdef, state, rest_of_state, context):
  return nnx.merge(graphdef, state, rest_of_state).compute_cross_kv(context)


@partial(jax.jit, donate_argnums=(0,))
def commit_kv_cache(kv_cache, chunk_kv, current_start):
  """Writes a chunk's keys / values into the cache (in place) at token offset `current_start`."""
  start = jnp.asarray(current_start, jnp.int32)
  zero = jnp.zeros((), jnp.int32)
  return tuple(
      jax.lax.dynamic_update_slice(cache, new.astype(cache.dtype), (zero, zero, start, zero, zero))
      for cache, new in zip(kv_cache, chunk_kv)
  )


@jax.jit
def flow_to_x0(flow, latents, sigma):
  """x0 = x_t - sigma_t * flow (`_convert_flow_pred_to_x0`)."""
  return (latents - sigma * flow).astype(latents.dtype)


@jax.jit
def add_noise(x0, noise, sigma):
  """x_t = (1 - sigma) * x0 + sigma * noise (`FlowUniPCMultistepScheduler.add_noise`)."""
  return (1 - sigma) * x0 + sigma * noise


# =============================================================================
# 4. Pipeline
# =============================================================================


def _lingbot_model_kwargs(config: DiffusionConfig, model_config: dict, ckpt_scan_layers=None) -> dict:
  """Adds the runtime (sharding / dtype / attention) settings from `config` to a LingBotWorldModel config."""
  model_config = dict(model_config)
  model_config["sharding"] = WanSharding.from_rules(config.logical_axis_rules, WAN_MESH_AXES)
  model_config["dtype"] = config.activations_dtype
  model_config["weights_dtype"] = config.weights_dtype
  model_config["attention"] = config.attention
  model_config["precision"] = get_precision(config)
  model_config["flash_block_sizes"] = get_flash_block_sizes(config)
  model_config["flash_min_seq_length"] = config.flash_min_seq_length
  model_config["scan_layers"] = config.scan_layers if ckpt_scan_layers is None else ckpt_scan_layers
  return model_config


class LingBotWorldPipeline(WanPipeline):
  """LingBot-World-Fast pipeline: Wan text encoder / VAE / loading + the causal few-step sampler."""

  def __init__(self, config: DiffusionConfig, transformer: Optional[LingBotWorldModel], **kwargs):
    super().__init__(config=config, **kwargs)
    self.transformer = transformer

  # --------------------------------------------------------------------------- loading

  @classmethod
  def load_tokenizer(cls, config: DiffusionConfig):
    """umt5-xxl tokenizer of `base_model_name_or_path` (sub-folder `tokenizer_subfolder`)."""
    return AutoTokenizer.from_pretrained(
        getattr(config, "tokenizer_path", "") or config.base_model_name_or_path,
        subfolder=getattr(config, "tokenizer_subfolder", "google/umt5-xxl"),
    )

  @classmethod
  def load_scheduler(cls, config, restored_checkpoint):
    """The fixed distilled schedule (the checkpoint has no scheduler config)."""
    scheduler = LingBotFastScheduler(
        num_train_timesteps=getattr(config, "num_train_timesteps", 1000),
        shift=config.flow_shift,
        timesteps_index=tuple(config.timesteps_index),
    )
    return scheduler, None

  @classmethod
  def load_vae(cls, devices_array, mesh, rngs, config, restored_checkpoint=None):
    """Wan VAE, with the checkpoint weights cast to `vae_weights_dtype` (float32 like the reference VAE).

    (`WanPipeline.load_vae` casts them to the transformer's `weights_dtype`.)
    """
    vae_dtype = config.vae_weights_dtype
    original = config.weights_dtype
    config.weights_dtype = vae_dtype
    try:
      return super().load_vae(devices_array, mesh, rngs, config, restored_checkpoint)
    finally:
      config.weights_dtype = original

  @classmethod
  def load_transformer(cls, devices_array, mesh, rngs, config, restored_checkpoint=None, subfolder="transformer"):
    """Builds a sharded LingBotWorldModel (eval_shape) and places the checkpoint params on `mesh`."""
    with jax.set_mesh(mesh):
      checkpoint_state, model_config = _restored_component(restored_checkpoint, subfolder, LingBotWorldModel)
      ckpt_scan_layers = model_config.pop("scan_layers", None)
      model_config = _lingbot_model_kwargs(config, model_config, ckpt_scan_layers)
      model = nnx.eval_shape(lambda rngs: LingBotWorldModel(**model_config, rngs=rngs), rngs=rngs)
      graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)
      shardings = dict(nnx.to_flat_state(state_shardings(state, mesh)))
      state = dict(nnx.to_flat_state(state))
      params = checkpoint_state.get("params", checkpoint_state)
      params = jax.tree_util.tree_map_with_path(
          lambda path, x: cast_with_exclusion(path, x, dtype_to_cast=config.weights_dtype), params
      )
      state = put_params_into_state(state, params, shardings, mesh, restored_checkpoint=True, subfolder=subfolder)
      return nnx.merge(graphdef, nnx.from_flat_state(state), rest_of_state)

  @classmethod
  def _load_and_init(
      cls,
      config: DiffusionConfig,
      restored_checkpoint=None,
      load_vae=True,
      load_text_encoder=True,
      load_transformer=True,
      load_scheduler=True,
  ):
    components = cls._create_common_components(
        config,
        restored_checkpoint=restored_checkpoint,
        load_vae=load_vae,
        load_text_encoder=load_text_encoder,
        load_scheduler=load_scheduler,
    )
    transformer = None
    if load_transformer:
      transformer = cls.load_transformer(
          devices_array=components["devices_array"],
          mesh=components["mesh"],
          rngs=components["rngs"],
          config=config,
          restored_checkpoint=restored_checkpoint,
      )
    pipeline = cls(
        tokenizer=components["tokenizer"],
        text_encoder=components["text_encoder"],
        transformer=transformer,
        vae=components["vae"],
        vae_cache=components["vae_cache"],
        scheduler=components["scheduler"],
        scheduler_state=components["scheduler_state"],
        devices_array=components["devices_array"],
        mesh=components["mesh"],
        vae_mesh=components["vae_mesh"],
        config=config,
    )
    return pipeline, transformer

  def _get_num_channel_latents(self) -> int:
    return self.transformer.out_dim

  # --------------------------------------------------------------------------- stages

  def encode_first_frame(self, image: np.ndarray, num_frames: int) -> jax.Array:
    """[3, h, w] first frame (+ num_frames - 1 zero frames) -> normalized VAE latents [1, 16, lat_f, lat_h, lat_w]."""
    video = np.concatenate([image[:, None], np.zeros((3, num_frames - 1, *image.shape[1:]), np.float32)], axis=1)
    video = jnp.asarray(video[None], dtype=getattr(self.vae, "dtype", jnp.float32))
    with jax.set_mesh(self.vae_mesh):
      graphdef, state, rest_of_state = nnx.split(self.vae, nnx.Param, ...)
      mu = vae_encode_pass(graphdef, state, rest_of_state, video)  # [1, T, H, W, C]
    mu = jnp.transpose(mu.astype(jnp.float32), (0, 4, 1, 2, 3))
    mean = jnp.asarray(self.vae.latents_mean, jnp.float32).reshape(1, -1, 1, 1, 1)
    inv_std = 1.0 / jnp.asarray(self.vae.latents_std, jnp.float32).reshape(1, -1, 1, 1, 1)
    return (mu - mean) * inv_std

  def decode_latents(self, latents: jax.Array) -> np.ndarray:
    """Normalized latents [1, 16, T, H, W] -> uint8 video [1, F, H, W, 3] (Wan VAE decode, reference scaling)."""
    mean = jnp.asarray(self.vae.latents_mean, jnp.float32).reshape(1, -1, 1, 1, 1)
    inv_std = 1.0 / jnp.asarray(self.vae.latents_std, jnp.float32).reshape(1, -1, 1, 1, 1)
    latents = latents.astype(jnp.float32) / inv_std + mean
    return self._decode_latents_to_video(latents.astype(getattr(self.vae, "dtype", jnp.float32)))

  def _noise_source(self, noise, seed: int):
    """Initial / per-step Gaussian noise: from `noise` (a dict with `noise` [16, T, H, W] and `step_noise`
    [chunks, steps - 1, 16, chunk, H, W], e.g. dumped from the reference) or from `jax.random` with `seed`."""
    key = jax.random.key(seed)

    def initial(shape):
      if noise is not None:
        return jnp.asarray(noise["noise"], jnp.float32)[None]
      return jax.random.normal(jax.random.fold_in(key, 0), shape, jnp.float32)

    def step(shape, chunk_id, step_id):
      if noise is not None:
        return jnp.asarray(noise["step_noise"][chunk_id, step_id], jnp.float32)[None]
      return jax.random.normal(jax.random.fold_in(key, 1 + chunk_id * 1000 + step_id), shape, jnp.float32)

    return initial, step

  def __call__(
      self,
      prompt: Union[str, List[str]] = None,
      negative_prompt: Union[str, List[str]] = None,  # pylint: disable=unused-argument  # no CFG (distilled)
      height: int = 480,
      width: int = 832,
      num_frames: int = 81,
      num_inference_steps: Optional[int] = None,
      image=None,
      action_path: Optional[str] = None,
      noise: Optional[dict] = None,
      output_type: str = "np",
      intermediates: Optional[dict] = None,
      **unused_kwargs,
  ):
    """Generates a video from `prompt`, the first frame `image` and the camera trajectory in `action_path`.

    Args:
      prompt: text prompt (batch size 1).
      height / width: output area `height * width` (the aspect ratio follows the image, as the reference).
      num_frames: requested frames (4n + 1); clipped to the trajectory and to a whole number of chunks.
      num_inference_steps: first N steps of the distilled schedule (default: all of `timesteps_index`).
      image: PIL image / path (default `config.image_path`).
      action_path: directory with `poses.npy` / `intrinsics.npy` (default `config.action_path`).
      noise: optional fixed noise (see `_noise_source`), for parity runs against the reference.
      output_type: "np" (uint8 video) or "latent".
      intermediates: optional dict filled with intermediate arrays (debugging / parity).

    Returns:
      (video [1, F, H, W, 3] uint8 or latents, timing trace).
    """
    config = self.config
    trace = {}
    t_start = time.perf_counter()
    if isinstance(prompt, (list, tuple)):
      if len(prompt) != 1:
        raise NotImplementedError("LingBot-World-Fast generation supports batch size 1.")
      prompt = prompt[0]
    image = image if image is not None else config.image_path
    action_path = action_path if action_path is not None else getattr(config, "action_path", "")
    if not action_path:
      raise ValueError("LingBot-World needs an action sequence: set action_path (poses.npy / intrinsics.npy).")
    chunk_size = config.chunk_size
    timesteps = self.scheduler.timesteps
    if num_inference_steps is not None and 0 < num_inference_steps < len(timesteps):
      timesteps = timesteps[:num_inference_steps]

    # 1. Inputs: action sequence, first frame, geometry.
    c2ws, intrinsics, num_frames = camera_lingbot.load_action(action_path, num_frames)
    image = load_image(image)
    height, width, lat_h, lat_w, lat_f, num_frames = latent_geometry(
        image.shape[1:], height * width, num_frames, chunk_size
    )
    image = resize_bicubic(image, height, width)
    max_logging.log(
        f"LingBot: {num_frames} frames {height}x{width}, latents {lat_f}x{lat_h}x{lat_w}, chunks of {chunk_size},"
        f" timesteps {list(timesteps)}"
    )

    # 2. Conditioning: text (UMT5), first frame (VAE + mask), camera (Plücker).
    prompt_embeds = self._get_t5_prompt_embeds(prompt=[prompt]).astype(config.activations_dtype)
    image_latents = self.encode_first_frame(image, num_frames)
    mask = jnp.asarray(first_frame_mask(num_frames, lat_f, lat_h, lat_w))[None]
    condition = jnp.concatenate([mask, image_latents], axis=1)
    cam = camera_lingbot.camera_condition(
        c2ws,
        intrinsics,
        height,
        width,
        lat_h,
        lat_w,
        lat_f,
        chunk_size,
        motion_scale=getattr(config, "camera_motion_scale", 1.0),
    )
    cam = jnp.asarray(cam, config.activations_dtype)
    trace["conditioning"] = time.perf_counter() - t_start
    if intermediates is not None:
      intermediates.update(prompt_embeds=prompt_embeds, condition=condition, cam=cam)

    # 3. Causal few-step sampling.
    t_denoise = time.perf_counter()
    transformer = self.transformer
    graphdef, state, rest_of_state = nnx.split(transformer, nnx.Param, ...)
    replicated = NamedSharding(self.mesh, P())
    frame_seqlen = (lat_h // transformer.patch_size[1]) * (lat_w // transformer.patch_size[2])
    chunk_tokens = chunk_size * frame_seqlen
    num_chunks = lat_f // chunk_size
    initial_noise, step_noise = self._noise_source(noise, config.seed)
    with jax.set_mesh(self.mesh):
      noise_latents = jax.device_put(initial_noise((1, transformer.out_dim, lat_f, lat_h, lat_w)), replicated)
      cross_kv = cross_kv_pass(graphdef, state, rest_of_state, prompt_embeds)
      cache_shape = transformer.kv_cache_shape(1, lat_f * frame_seqlen)
      cache_sharding = NamedSharding(
          self.mesh, transformer.sharding("kv_cache", (None, None, None, SELF_ATTN_HEAD, None), TensorType.ACTIVATION)
      )
      kv_cache = tuple(
          jax.jit(partial(jnp.zeros, cache_shape, config.activations_dtype), out_shardings=cache_sharding)()
          for _ in range(2)
      )
      chunks = []
      for chunk_id in range(num_chunks):
        frames = slice(chunk_id * chunk_size, (chunk_id + 1) * chunk_size)
        current_start = jnp.int32(chunk_id * chunk_tokens)
        rope_cos, rope_sin = rope_cos_sin(transformer.head_dim, chunk_id * chunk_size, chunk_size, lat_h // 2, lat_w // 2)
        args = (condition[:, :, frames], cam[:, :, frames], cross_kv, rope_cos, rope_sin, kv_cache, current_start)
        latents = noise_latents[:, :, frames]
        for step_id, t in enumerate(timesteps):
          flow, _ = transformer_chunk_pass(graphdef, state, rest_of_state, latents, jnp.full((1,), t, jnp.float32), *args)
          x0 = flow_to_x0(flow, latents, jnp.float32(self.scheduler.x0_sigma(t)))
          if step_id < len(timesteps) - 1:
            sigma = jnp.float32(self.scheduler.noise_sigma(timesteps[step_id + 1]))
            latents = add_noise(x0, step_noise(x0.shape, chunk_id, step_id), sigma)
          if intermediates is not None:
            intermediates.setdefault("flow", []).append(np.asarray(flow))
        chunks.append(x0)
        # Commit the clean chunk to the KV cache (t = 0).
        _, chunk_kv = transformer_chunk_pass(graphdef, state, rest_of_state, x0, jnp.zeros((1,), jnp.float32), *args)
        kv_cache = commit_kv_cache(kv_cache, chunk_kv, current_start)
      latents = jnp.concatenate(chunks, axis=2)
      latents.block_until_ready()
    trace["denoise_total"] = time.perf_counter() - t_denoise
    if intermediates is not None:
      intermediates["latents"] = np.asarray(latents)
    if output_type == "latent":
      return latents, trace

    # 4. VAE decode.
    t_decode = time.perf_counter()
    video = self.decode_latents(latents)
    trace["vae_decode"] = time.perf_counter() - t_decode
    return video, trace
