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

"""Ming-Image (inclusionAI/Ming-Image-0.1-Design) text-to-image inference pipeline.

Port of the reference `ImageGenerator.sample` / `ImageGenerationPipeline.__call__`
(github.com/inclusionAI/Ming-Image, `diffusion/{generator,pipeline}.py`) and of
`BailingMM2NativeForConditionalGeneration.get_condition_embeds_for_image_gen`:

1. Prompt -> chat template -> token ids, followed by `<image_start>` + 256 generation query tokens + `<image_end>`.
2. The MLLM (BailingMoeV2) runs once over the sequence, the query-token slots holding the learnable query embeddings.
3. `cap_feats_2` (directvlm): hidden states of layers [5, 12, 20] on the prompt tokens -> `proj_directvlm`.
   `cap_feats`: last hidden state on the query tokens -> `proj_in` -> Qwen2 connector -> `proj_out`.
4. Flow-matching Euler sampling (FlowMatchEulerDiscreteScheduler with dynamic shifting, `sigma_min = 0`) of the
   diffusion transformer; the model predicts `-velocity`. CFG only when `guidance_scale > 1` (default 1.0: off), with
   all-zero negative conditioning as the reference.
5. VAE decode (`AutoencoderKLQwenImage` == Wan VAE with 4 RGBA channels) of `latents / scaling + shift`.
"""

# The jitted stage functions take graphdefs / states / arrays as flat positional args (like wan_pipeline.py).
# pylint: disable=too-many-positional-arguments

from functools import partial
import math
import os
import time
from typing import List, Optional, Union

import flax
from flax import nnx
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.inference import aot_cache
from maxtext.m3.diffuser_pipelines.pipeline_utils import DiffusionPipeline, _restored_component, _to_int_path
from maxtext.m3.diffuser_pipelines.wan_pipeline import (
    WanPipeline,
    cast_with_exclusion,
    state_shardings,
    vae_decode_pass,
)
from maxtext.m3.models.ming_image import connector_ming, text_encoder_ming
from maxtext.m3.models.ming_image.connector_ming import MingConditioner, MingConnector
from maxtext.m3.models.ming_image.text_encoder_ming import MingTextEncoder
from maxtext.m3.models.ming_image.transformer_ming import MingImageTransformer
from maxtext.m3.models.wan import VAE_MESH_AXES, WAN_MESH_AXES, WanSharding
from maxtext.m3.models.wan.utils_wan import create_wan_mesh, get_flash_block_sizes, get_precision
from maxtext.utils import max_logging

# Reference `ImageGenerationPipeline.__call__` dynamic shift (diffusers FlowMatchEulerDiscreteScheduler defaults).
_BASE_IMAGE_SEQ_LEN = 256
_MAX_IMAGE_SEQ_LEN = 4096
_BASE_SHIFT = 0.5

# Constructor arguments set by the pipeline at runtime. A stored component config can also carry them (e.g. the
# connector's HF config.json has `"dtype": "float32"`, which `model_init_config` keeps), so they are dropped from it.
_RUNTIME_KWARGS = ("dtype", "weights_dtype", "precision", "sharding", "mesh", "scan_layers", "attention")


def _architecture_config(model_config: dict) -> dict:
  return {k: v for k, v in model_config.items() if k not in _RUNTIME_KWARGS}


def calculate_shift(image_seq_len: int) -> float:
  """`mu` of the reference pipeline: max_shift 1.35 (and max_seq = seq_len) from 4096 image tokens on, else 1.15."""
  if image_seq_len >= _MAX_IMAGE_SEQ_LEN:
    max_seq_len, max_shift = image_seq_len, 1.35
  else:
    max_seq_len, max_shift = _MAX_IMAGE_SEQ_LEN, 1.15
  m = (max_shift - _BASE_SHIFT) / (max_seq_len - _BASE_IMAGE_SEQ_LEN)
  return image_seq_len * m + (_BASE_SHIFT - m * _BASE_IMAGE_SEQ_LEN)


def flow_match_sigmas(num_inference_steps: int, image_seq_len: int) -> np.ndarray:
  """Sigmas (length `num_inference_steps + 1`, terminal 0) of the reference sampler.

  The reference sets `scheduler.sigma_min = 0` and `use_dynamic_shifting = True`, so diffusers'
  `FlowMatchEulerDiscreteScheduler.set_timesteps(mu=...)` gives `time_shift(mu, linspace(1, 0, N))` (exponential
  time shift) followed by a terminal 0.
  """
  mu = calculate_shift(image_seq_len)
  sigmas = np.linspace(1.0, 0.0, num_inference_steps, dtype=np.float64)
  with np.errstate(divide="ignore"):
    sigmas = np.where(sigmas > 0, math.exp(mu) / (math.exp(mu) + (1.0 / np.maximum(sigmas, 1e-30) - 1.0)), 0.0)
  return np.append(sigmas, 0.0).astype(np.float32)


# Plain jit (like Wan's UMT5 `text_encoder_forward_pass`), NOT `aot_cache.cached_jit`: in `aot_cache.warmup_mode()` a
# cached_jit returns all-zero outputs, which `encode_prompt` would then cache as the prompt's conditioning.
@partial(jax.jit, static_argnums=(0, 2, 4))
def text_encoder_forward_pass(
    llm_graphdef,
    llm_state,
    cond_graphdef,
    cond_state,
    conn_graphdef,
    conn_state,
    input_ids,
    position_ids,
    attention_mask,
    image_mask,
    gen_mask,
    prompt_mask,
):
  """MLLM + conditioning head -> (cap_feats [B, Q, 2560], cap_feats_2 [B, L, 3840]) in float32."""
  llm = nnx.merge(llm_graphdef, llm_state)
  conditioner = nnx.merge(cond_graphdef, cond_state)
  connector = nnx.merge(conn_graphdef, conn_state)
  selected, last_hidden = llm(
      input_ids,
      position_ids,
      attention_mask,
      image_mask,
      query_embeds=conditioner.query_tokens[...],
      output_layers=tuple(conditioner.selected_hidden_states_layers),
  )
  cap_feats_2 = conditioner.directvlm(jnp.concatenate(selected, axis=-1), prompt_mask)
  num_query = conditioner.query_tokens.shape[0]
  query_hidden = connector_ming.gather_gen_tokens(last_hidden, gen_mask, num_query)
  cap_feats = conditioner.learnable(query_hidden, connector)
  return cap_feats.astype(jnp.float32), cap_feats_2.astype(jnp.float32)


@partial(aot_cache.cached_jit, static_argnames=("do_cfg",))
def transformer_euler_step(
    graphdef,
    state,
    rest_of_state,
    latents,
    sigma,
    sigma_next,
    cap_feats,
    cap_mask,
    cap_feats_2,
    cap2_mask,
    guidance_scale,
    do_cfg: bool = False,
):
  """One Euler step `x <- x + (sigma_next - sigma) * v` with `v = -transformer(x, 1 - sigma)` (float32 latents)."""
  transformer = nnx.merge(graphdef, state, rest_of_state)
  batch = latents.shape[0]
  timestep = jnp.full((batch,), 1.0 - sigma, dtype=jnp.float32)  # reference: (1000 - t) / 1000, t = 1000 * sigma
  model_latents = latents.astype(transformer.dtype)
  if do_cfg:
    # Negative conditioning is all zeros (same lengths) as in the reference `ImageGenerator.sample`.
    out = transformer(
        jnp.concatenate([model_latents, model_latents], axis=0),
        jnp.concatenate([timestep, timestep], axis=0),
        jnp.concatenate([cap_feats, jnp.zeros_like(cap_feats)], axis=0),
        jnp.concatenate([cap_mask, cap_mask], axis=0),
        jnp.concatenate([cap_feats_2, jnp.zeros_like(cap_feats_2)], axis=0),
        jnp.concatenate([cap2_mask, cap2_mask], axis=0),
    ).astype(jnp.float32)
    pos, neg = out[:batch], out[batch:]
    out = pos + guidance_scale * (pos - neg)  # reference formula (not neg + s * (pos - neg))
  else:
    out = transformer(model_latents, timestep, cap_feats, cap_mask, cap_feats_2, cap2_mask).astype(jnp.float32)
  return latents + (sigma_next - sigma) * (-out)


def _place_params(model_factory, params, mesh: Mesh, dtype, rngs: nnx.Rngs, name: str):
  """eval_shape `model_factory`, then device_put the (string-keyed) checkpoint `params` with the model's shardings."""
  model = nnx.eval_shape(model_factory, rngs=rngs)
  graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)
  shardings = dict(nnx.to_flat_state(state_shardings(state, mesh)))
  state = dict(nnx.to_flat_state(state))
  t_start = time.perf_counter()
  paths, values = [], []
  for path, val in flax.traverse_util.flatten_dict(params).items():
    path = _to_int_path(path[:-1] if path[-1] == "value" else path)
    if path not in state:
      raise ValueError(f"Checkpoint {name} param {path} is not a model param.")
    paths.append(path)
    values.append(np.asarray(val) if dtype is None else np.asarray(val).astype(dtype, copy=False))
  if len(paths) != len(state):
    missing = sorted(set(state) - set(paths))
    raise ValueError(f"Checkpoint {name}_state misses {len(missing)} params, e.g. {missing[:3]}.")
  arrays = jax.device_put(values, [shardings[path].value for path in paths])
  jax.block_until_ready(arrays)
  for path, array in zip(paths, arrays):
    state[path].value = array
  num_bytes = sum(v.nbytes for v in values)
  max_logging.log(f"{name} ({num_bytes / 2**30:.2f} GiB) on device in {time.perf_counter() - t_start:.1f}s")
  return nnx.merge(graphdef, nnx.from_flat_state(state), rest_of_state)


class MingImagePipeline(DiffusionPipeline):
  """Ming-Image text-to-image pipeline (see the module docstring)."""

  _transformer_keys = ["transformer"]

  def __init__(
      self,
      config,
      transformer: Optional[MingImageTransformer],
      *,
      mesh: Mesh,
      vae_mesh: Mesh,
      vae=None,
      text_encoder: Optional[MingTextEncoder] = None,
      connector: Optional[MingConnector] = None,
      conditioner: Optional[MingConditioner] = None,
      tokenizer=None,
      llm_config: Optional[dict] = None,
  ):
    self.config = config
    self.transformer = transformer
    self.mesh = mesh
    self.vae_mesh = vae_mesh
    self.vae = vae
    self.text_encoder = text_encoder
    self.connector = connector
    self.conditioner = conditioner
    self.tokenizer = tokenizer
    self.llm_config = llm_config or {}
    self.vae_scaling_factor = float(getattr(config, "vae_scaling_factor", 8.0064))
    self.vae_shift_factor = float(getattr(config, "vae_shift_factor", 0.0))
    self.vae_scale_factor_spatial = 8  # 2 ** len(temperal_downsample)
    self._prompt_embeds_cache = {}

  # --------------------------------------------------------------------------------------------------------------------
  # Loading
  # --------------------------------------------------------------------------------------------------------------------

  @classmethod
  def _snapshot_dir(cls, config) -> str:
    """Local HF snapshot (tokenizer files live in `mllm/`)."""
    path = config.pretrained_model_name_or_path
    if os.path.isdir(path):
      return path
    cache_dir = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", "/dev/shm/hf_cache"), "maxdiffusion"
    )
    return snapshot_download(path, cache_dir=cache_dir, allow_patterns=["mllm/*.json", "mllm/*.py", "mllm/*.txt"])

  @classmethod
  def load_text_components(cls, config, mesh: Mesh, rngs: nnx.Rngs, restored_checkpoint):
    """MLLM, connector and conditioning head on `mesh`, tensor parallel (`text_encoder_logical_axis_rules`).

    They run at `text_encoder_precision` (default HIGHEST) in `text_encoder_dtype` (default float32): the connector has
    activations in the thousands, and TPU DEFAULT precision (one bf16 pass, even for float32 inputs) scrambles the
    generation-query features (`cap_feats`) enough for the DiT to ignore the prompt. Only ~1.4B params are active, so
    this costs little next to the denoising loop.
    """
    dtype = jnp.dtype(getattr(config, "text_encoder_dtype", "float32"))
    precision = jax.lax.Precision[str(getattr(config, "text_encoder_precision", "HIGHEST")).upper()]
    llm_sharding = WanSharding.from_rules(config.text_encoder_logical_axis_rules, WAN_MESH_AXES)
    connector_sharding = WanSharding.from_rules(config.connector_logical_axis_rules, WAN_MESH_AXES)
    components = {}
    for name, model_cls, sharding in (
        ("mllm", MingTextEncoder, llm_sharding),
        ("connector", MingConnector, connector_sharding),
        ("mlp", MingConditioner, connector_sharding),
    ):
      params, model_config = _restored_component(restored_checkpoint, name, model_cls)
      model_config = _architecture_config(model_config)

      def factory(rngs, model_cls=model_cls, model_config=model_config, sharding=sharding):
        return model_cls(
            **model_config, dtype=dtype, weights_dtype=dtype, precision=precision, sharding=sharding, rngs=rngs
        )

      components[name] = _place_params(factory, params, mesh, dtype, rngs, name)
      if name == "mllm":
        components["llm_config"] = dict(model_config.get("llm_config", {}))
    return components

  @classmethod
  def load_transformer(cls, config, mesh: Mesh, rngs: nnx.Rngs, restored_checkpoint):
    """The diffusion transformer, sharded by `logical_axis_rules`."""
    params, model_config = _restored_component(restored_checkpoint, "transformer", MingImageTransformer)
    model_config = _architecture_config(model_config)
    sharding = WanSharding.from_rules(config.logical_axis_rules, WAN_MESH_AXES)
    weights_dtype = jnp.dtype(config.weights_dtype)
    params = jax.tree_util.tree_map_with_path(lambda p, x: cast_with_exclusion(p, x, weights_dtype), params)

    def factory(rngs):
      return MingImageTransformer(
          **model_config,
          sharding=sharding,
          dtype=jnp.dtype(config.activations_dtype),
          weights_dtype=weights_dtype,
          attention=config.attention,
          precision=get_precision(config),
          flash_block_sizes=get_flash_block_sizes(config),
          flash_min_seq_length=getattr(config, "flash_min_seq_length", 4096),
          rngs=rngs,
      )

    with jax.set_mesh(mesh):
      return _place_params(factory, params, mesh, None, rngs, "transformer")

  @classmethod
  def _load_and_init(
      cls,
      config,
      restored_checkpoint=None,
      load_vae=True,
      load_text_encoder=True,
      load_transformer=True,
      load_scheduler=True,
  ):
    del load_scheduler  # The flow-matching sigmas are computed in closed form (see `flow_match_sigmas`).
    mesh = create_wan_mesh(config)
    total_devices = math.prod(mesh.devices.shape)
    vae_spatial = getattr(config, "vae_spatial", -1)
    vae_spatial = total_devices if vae_spatial == -1 else vae_spatial
    vae_mesh = Mesh(mesh.devices.flatten().reshape(total_devices // vae_spatial, vae_spatial), VAE_MESH_AXES)
    rngs = nnx.Rngs(jax.random.key(config.seed))

    vae = None
    if load_vae:
      max_logging.log("Loading VAE")
      vae, _ = WanPipeline.load_vae(mesh.devices, vae_mesh, rngs, config, restored_checkpoint=restored_checkpoint)
    text = {}
    tokenizer = None
    if load_text_encoder:
      max_logging.log("Loading tokenizer, MLLM text encoder, connector and conditioning head")
      tokenizer = text_encoder_ming.load_tokenizer(os.path.join(cls._snapshot_dir(config), "mllm"))
      text = cls.load_text_components(config, mesh, rngs, restored_checkpoint)
    transformer = None
    if load_transformer:
      max_logging.log("Loading diffusion transformer")
      transformer = cls.load_transformer(config, mesh, rngs, restored_checkpoint)

    pipeline = cls(
        config,
        transformer,
        mesh=mesh,
        vae_mesh=vae_mesh,
        vae=vae,
        text_encoder=text.get("mllm"),
        connector=text.get("connector"),
        conditioner=text.get("mlp"),
        tokenizer=tokenizer,
        llm_config=text.get("llm_config"),
    )
    return pipeline, transformer

  # --------------------------------------------------------------------------------------------------------------------
  # Conditioning
  # --------------------------------------------------------------------------------------------------------------------

  def _text_length(self, num_tokens: int) -> int:
    """Smallest configured static length bucket that holds `num_tokens` (+ the query block)."""
    buckets = sorted(getattr(self.config, "text_length_buckets", None) or [self.config.max_sequence_length])
    for bucket in buckets:
      if num_tokens <= bucket:
        return bucket
    return buckets[-1]

  def encode_prompt(self, prompt: Union[str, List[str]]):
    """Returns `(cap_feats, cap_mask, cap_feats_2, cap2_mask)` for a batch of prompts (cached per prompt tuple)."""
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    cache_key = tuple(prompts)
    if cache_key in self._prompt_embeds_cache:
      return self._prompt_embeds_cache[cache_key]

    token_ids = [text_encoder_ming.format_t2i_prompt(p, self.tokenizer) for p in prompts]
    num_query = self.conditioner.query_tokens.shape[0]
    length = self._text_length(max(len(t) for t in token_ids) + num_query + 2)
    samples = [text_encoder_ming.build_t2i_inputs(t, length, self.llm_config) for t in token_ids]
    batch = {k: np.stack([s[k] for s in samples], axis=0) for k in samples[0]}
    batch["position_ids"] = np.moveaxis(batch["position_ids"], 0, 1)  # [B, 3, L] -> [3, B, L]

    llm_graphdef, llm_state = nnx.split(self.text_encoder)
    cond_graphdef, cond_state = nnx.split(self.conditioner)
    conn_graphdef, conn_state = nnx.split(self.connector)
    with jax.set_mesh(self.mesh):
      cap_feats, cap_feats_2 = text_encoder_forward_pass(
          llm_graphdef,
          llm_state,
          cond_graphdef,
          cond_state,
          conn_graphdef,
          conn_state,
          jnp.asarray(batch["input_ids"]),
          jnp.asarray(batch["position_ids"]),
          jnp.asarray(batch["attention_mask"]),
          jnp.asarray(batch["image_mask"]),
          jnp.asarray(batch["gen_mask"]),
          jnp.asarray(batch["prompt_mask"]),
      )
    cap_mask = jnp.ones(cap_feats.shape[:2], dtype=bool)
    # Reference: directvlm states are cut at the batch's last prompt token (`directvlm_max_valid_ind`); every position
    # before it is a DiT caption token (non-prompt positions zeroed by the mask multiply).
    prompt_mask = batch["prompt_mask"].astype(bool)
    valid_len = int(np.max(np.where(prompt_mask.any(axis=0))[0])) + 1
    cap2_mask = jnp.asarray(np.broadcast_to(np.arange(length)[None, :] < valid_len, prompt_mask.shape))
    out = (cap_feats, cap_mask, cap_feats_2, cap2_mask)
    if not aot_cache.in_warmup():  # Defensive: never cache conditioning computed during a zero-exec warmup.
      if len(self._prompt_embeds_cache) >= 16:
        self._prompt_embeds_cache.pop(next(iter(self._prompt_embeds_cache)))
      self._prompt_embeds_cache[cache_key] = out
    return out

  # --------------------------------------------------------------------------------------------------------------------
  # Sampling
  # --------------------------------------------------------------------------------------------------------------------

  def prepare_latents(self, batch_size: int, height: int, width: int) -> jax.Array:
    """Initial float32 Gaussian latents, channel-last (B, H/8, W/8, C)."""
    num_channels = self.transformer.in_channels
    shape = (batch_size, height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial, num_channels)
    return jax.random.normal(jax.random.key(self.config.seed), shape, dtype=jnp.float32)

  def _decode(self, latents: jax.Array, trace: dict) -> np.ndarray:
    """latents (B, h, w, C) -> uint8 RGBA images (B, H, W, 4)."""
    t_start = time.perf_counter()
    latents = latents / self.vae_scaling_factor + self.vae_shift_factor
    latents = latents[:, None].astype(jnp.dtype(self.config.vae_dtype))  # (B, T=1, h, w, C)
    with jax.set_mesh(self.vae_mesh):
      graphdef, state, rest_of_state = nnx.split(self.vae, nnx.Param, ...)
      images = vae_decode_pass(graphdef, state, rest_of_state, latents)
      images.block_until_ready()
    trace["vae_decode_tpu"] = time.perf_counter() - t_start
    images = np.asarray(jax.device_get(images))
    trace["vae_decode"] = time.perf_counter() - t_start
    return images[:, 0]  # drop the frame axis

  def __call__(
      self,
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      height: Optional[int] = None,
      width: Optional[int] = None,
      num_inference_steps: Optional[int] = None,
      guidance_scale: Optional[float] = None,
      **unused_kwargs,
  ):
    """Generates images; returns `(uint8 images (B, H, W, 4), trace)`.

    `negative_prompt` is ignored: the reference negative conditioning is all zeros.
    """
    del negative_prompt, unused_kwargs
    height = height or self.config.height
    width = width or self.config.width
    num_inference_steps = num_inference_steps or self.config.num_inference_steps
    guidance_scale = self.config.guidance_scale if guidance_scale is None else guidance_scale
    multiple = 2 * self.vae_scale_factor_spatial
    if height % multiple or width % multiple:
      raise ValueError(f"height and width must be multiples of {multiple}, got {height}x{width}.")
    trace = {}

    t_start = time.perf_counter()
    cap_feats, cap_mask, cap_feats_2, cap2_mask = self.encode_prompt(prompt)
    jax.block_until_ready(cap_feats)
    trace["conditioning"] = time.perf_counter() - t_start

    batch_size = cap_feats.shape[0]
    latents = self.prepare_latents(batch_size, height, width)
    image_seq_len = (latents.shape[1] // 2) * (latents.shape[2] // 2)
    sigmas = flow_match_sigmas(num_inference_steps, image_seq_len)
    do_cfg = guidance_scale > 1.0

    t_start = time.perf_counter()
    graphdef, state, rest_of_state = nnx.split(self.transformer, nnx.Param, ...)
    with jax.set_mesh(self.mesh):
      for i in range(num_inference_steps):
        latents = transformer_euler_step(
            graphdef,
            state,
            rest_of_state,
            latents,
            jnp.float32(sigmas[i]),
            jnp.float32(sigmas[i + 1]),
            cap_feats.astype(self.transformer.dtype),
            cap_mask,
            cap_feats_2.astype(self.transformer.dtype),
            cap2_mask,
            jnp.float32(guidance_scale),
            do_cfg=do_cfg,
        )
      latents.block_until_ready()
    trace["denoise_total"] = time.perf_counter() - t_start

    images = self._decode(latents, trace)
    return images, trace
