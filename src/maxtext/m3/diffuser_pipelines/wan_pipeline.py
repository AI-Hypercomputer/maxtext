# Copyright 2023–2025 Google LLC
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

"""Wan text-to-video and image-to-video inference pipeline."""

from abc import abstractmethod
import concurrent.futures
from functools import partial
import math
import os
import threading
import time
from typing import List, Optional, Tuple, Union
from diffusers.utils import is_ftfy_available
import flax
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from maxtext.m3.configs.diffusers_config import DiffusionConfig
from maxtext.m3.diffuser_pipelines.pipeline_utils import (
    DiffusionPipeline,
    Profiler,
    _restored_component,
    _to_int_path,
    profiler_enabled,
)
from maxtext.m3.models.diffuser_schedulers.scheduling_unipc_multistep import FlaxUniPCMultistepScheduler, UniPCMultistepSchedulerState
from maxtext.inference import aot_cache
from maxtext.m3.models.wan import (
    VAE_MESH_AXES,
    WAN_MESH_AXES,
    AutoencoderKLWan,
    AutoencoderKLWanCache,
    UMT5EncoderModel,
    WanModel,
    WanSharding,
)
from maxtext.m3.models.wan.text_encoder_wan import check_tensor_parallel_divides
from maxtext.m3.models.wan.utils_wan import create_wan_mesh, device_put_replicated, get_flash_block_sizes, get_precision
from maxtext.multimodal.processor_vae import PipelineImageInput, VaeVideoProcessor
from maxtext.utils import max_logging
import numpy as np
from transformers import AutoTokenizer


import html
import re
from transformers import CLIPImageProcessor

try:
  from transformers.models.clip.modeling_flax_clip import FlaxCLIPVisionModel
except ModuleNotFoundError:
  try:
    from transformers import FlaxCLIPVisionModel
  except ImportError:
    FlaxCLIPVisionModel = None
import PIL

# Params whose path matches any of these keywords are kept in float32 by
# cast_with_exclusion regardless of weights_dtype (mirrors
# WAN2_1_HF_TO_MAXTEXT.FP32_PARAM_KEYWORDS used at conversion time).
_CAST_EXCLUSION_KEYWORDS = (
    "norm",  # For all LayerNorm/GroupNorm layers
    "condition_embedder",  # The entire time/text conditioning module
    "scale_shift_table",  # Catches both the final and the AdaLN tables
)


def _is_cast_excluded(path_str: str) -> bool:
  return any(keyword in path_str.lower() for keyword in _CAST_EXCLUSION_KEYWORDS)


def cast_with_exclusion(path, x, dtype_to_cast):
  """Casts arrays to dtype_to_cast, but keeps params from any 'norm' layer in float32."""
  path_str = ".".join(str(k.key) if isinstance(k, jax.tree_util.DictKey) else str(k) for k in path)

  if _is_cast_excluded(path_str):
    # Keep LayerNorm/GroupNorm weights and biases in full precision
    target_dtype = jnp.float32
  else:
    # Cast everything else to dtype_to_cast
    target_dtype = dtype_to_cast
  if x.dtype == np.dtype(target_dtype):
    # Already final (e.g. cast during weight loading) - avoid a full copy.
    return x
  return x.astype(target_dtype)


def basic_clean(text):
  if is_ftfy_available():
    import ftfy  # pylint: disable=import-outside-toplevel  # optional dependency

    text = ftfy.fix_text(text)
  text = html.unescape(html.unescape(text))
  return text.strip()


def whitespace_clean(text):
  text = re.sub(r"\s+", " ", text)
  text = text.strip()
  return text


def prompt_clean(text):
  text = whitespace_clean(basic_clean(text))
  return text


def state_shardings(state, mesh: Mesh):
  """`NamedSharding`s on `mesh` for an nnx state, from the physical `out_sharding` specs the model's `Sharding` set.

  Leaves without one (no metadata, or not a `Param`) are replicated.
  """
  return jax.tree.map(
      lambda spec: NamedSharding(mesh, spec if isinstance(spec, P) else P()),
      nnx.get_partition_spec(state),
      is_leaf=lambda x: x is None or isinstance(x, P),
  )


# Concurrent transformer loads (WAN 2.2's two experts) must not interleave
# their device transfers: shared PCIe lanes degrade ~50% under contention.
_DEVICE_PUT_LOCK = threading.Lock()


def put_params_into_state(
    state: dict,
    params: dict,
    logical_state_sharding: dict,
    mesh: Mesh,
    restored_checkpoint=None,
    subfolder: str = "",
) -> dict:
  """Moves host params into the flat nnx state on device.

  Shared by the WAN 2.x and VACE pipelines. Single-process: replicated
  params are the bulk of the bytes; a direct device_put broadcasts the same
  bytes over every device's PCIe stream (~2GB/s each). Instead, stage them
  sharded along dim0 (each device receives only 1/n of the bytes over PCIe)
  and replicate on-device through ICI, which is an order of magnitude
  faster than host links. Multi-process: per-param device_put_replicated
  with a process_allgather fallback.

  Args:
    state: Flat nnx state dict (path tuple -> VariableState) to fill.
    params: Host-side param tree with final dtypes.
    logical_state_sharding: Flat dict of target NamedShardings per path.
    mesh: Device mesh the shardings refer to.
    restored_checkpoint: If set, params came from an orbax restore and paths
      need 'value' suffix / block-index normalization.
    subfolder: Label used only for logging.

  Returns:
    The same `state` dict with `.value` set to on-device arrays.
  """
  t_put_start = time.perf_counter()
  put_specs = []
  for path, val in flax.traverse_util.flatten_dict(params).items():
    if restored_checkpoint:
      if path[-1] == "value":
        path = path[:-1]  # remove 'value'
      path = _to_int_path(path)

    put_specs.append((path, val, logical_state_sharding[path].value))

  if jax.process_count() == 1:
    n_devices = mesh.devices.size
    dim0_sharding = NamedSharding(mesh, P(mesh.axis_names))

    def stages_via_ici(val, target_sharding) -> bool:
      return (
          target_sharding.is_fully_replicated
          and val.ndim > 0
          and val.shape[0] % n_devices == 0
          and val.nbytes >= 1 << 26  # 64MB: below this, staging overhead wins
      )

    staged_ids = [i for i, (_, val, target_sharding) in enumerate(put_specs) if stages_via_ici(val, target_sharding)]
    direct_ids = [i for i in range(len(put_specs)) if i not in set(staged_ids)]

    put_arrays = [None] * len(put_specs)
    # Lock through block_until_ready: puts are async, and concurrent expert
    # transfers on shared PCIe lanes degrade ~50%.
    with _DEVICE_PUT_LOCK:
      if staged_ids:
        # Per-device slice puts run each device's PCIe lane in parallel
        # (a sharded device_put of the whole list serializes near single-
        # lane speed). Gathers go in ~6GB chunks: chunk N replicates over
        # ICI while chunk N+1's host transfers stream, and the transient
        # HBM reservation stays bounded.
        chunk_limit_bytes = 6 << 30
        chunks, current, current_bytes = [], [], 0
        for i in staged_ids:
          current.append(i)
          current_bytes += put_specs[i][1].nbytes
          if current_bytes >= chunk_limit_bytes:
            chunks.append(current)
            current, current_bytes = [], 0
        if current:
          chunks.append(current)
        for chunk in chunks:
          # One batched put per device: per-tensor-per-device calls pay
          # dispatch overhead 8x per tensor and defeat lane pipelining.
          slices_by_device = {}
          index_maps = []
          for i in chunk:
            val = put_specs[i][1]
            indices_map = dim0_sharding.addressable_devices_indices_map(val.shape)
            index_maps.append(list(indices_map.items()))
            for d, index in indices_map.items():
              slices_by_device.setdefault(d, []).append(val[index])
          shards_by_device = {d: iter(jax.device_put(slices, d)) for d, slices in slices_by_device.items()}
          sharded_arrays = []
          for i, device_indices in zip(chunk, index_maps):
            val = put_specs[i][1]
            shards = [next(shards_by_device[d]) for d, _ in device_indices]
            sharded_arrays.append(jax.make_array_from_single_device_arrays(val.shape, dim0_sharding, shards))
          # out_shardings must be the exact target sharding objects (not an
          # equivalent P()): downstream jit cache keys include arg shardings,
          # so a different-but-equivalent spec would force a full recompile.
          replicate_fn = jax.jit(lambda xs: xs, out_shardings=[put_specs[i][2] for i in chunk])
          for i, replicated in zip(chunk, replicate_fn(sharded_arrays)):
            put_arrays[i] = replicated
      if direct_ids:
        for i, put_array in zip(
            direct_ids,
            jax.device_put(
                [put_specs[i][1] for i in direct_ids],
                [put_specs[i][2] for i in direct_ids],
            ),
        ):
          put_arrays[i] = put_array
      jax.block_until_ready([a for a in put_arrays if a is not None])
    for (path, _, _), put_array in zip(put_specs, put_arrays):
      state[path].value = put_array
  else:
    for path, val, target_sharding in put_specs:
      try:
        state[path].value = device_put_replicated(val, target_sharding)
      except Exception as e:  # pylint: disable=broad-exception-caught  # fall back to process_allgather
        max_logging.log(f"Failed to device_put_replicated {path}: {e}")
        max_logging.log(f"Trying to use process_allgather for {path}")
        val_on_host = jax.experimental.multihost_utils.process_allgather(val, tiled=True)
        state[path].value = device_put_replicated(val_on_host, target_sharding)
        del val_on_host
  jax.block_until_ready([state[path].value for path, _, _ in put_specs])
  max_logging.log(
      f"Transformer {subfolder or 'transformer'} weights on device in" f" {time.perf_counter() - t_put_start:.1f}s"
  )
  return state


# For some reason, jitting this function increases the memory significantly, so instead manually move weights to device.
def _wan_model_kwargs(config: DiffusionConfig, wan_config: dict, ckpt_scan_layers=None) -> dict:
  """Adds the runtime (sharding / dtype / attention) settings from `config` to a WanModel config."""
  wan_config = dict(wan_config)
  if config.model_type == "I2V":
    # WAN 2.1 I2V uses image embeddings via CLIP encoder (image_dim and added_kv_proj_dim are set)
    # WAN 2.2 I2V uses VAE-encoded latent conditioning (image_dim and added_kv_proj_dim are None in the transformer config)
    if config.model_name == "wan2.1":
      if wan_config.get("image_seq_len") is None:
        wan_config["image_seq_len"] = 257

  wan_config["sharding"] = WanSharding.from_rules(config.logical_axis_rules, WAN_MESH_AXES)
  wan_config["dtype"] = config.activations_dtype
  wan_config["weights_dtype"] = config.weights_dtype
  wan_config["attention"] = config.attention
  wan_config["precision"] = get_precision(config)
  wan_config["flash_block_sizes"] = get_flash_block_sizes(config)
  wan_config["flash_min_seq_length"] = config.flash_min_seq_length
  wan_config["dropout"] = config.dropout
  wan_config["mask_padding_tokens"] = config.mask_padding_tokens
  # The param layout (stacked or per-layer blocks) is fixed by the checkpoint.
  wan_config["scan_layers"] = config.scan_layers if ckpt_scan_layers is None else ckpt_scan_layers
  wan_config["enable_jax_named_scopes"] = config.enable_jax_named_scopes
  wan_config["attention_config"] = {
      "use_base2_exp": config.use_base2_exp,
      "use_experimental_scheduler": config.use_experimental_scheduler,
      "ulysses_shards": getattr(config, "ulysses_shards", -1),
      "ulysses_attention_chunks": getattr(config, "ulysses_attention_chunks", 1),
  }
  return wan_config


def model_init_config_for_wan(model_config: dict) -> dict:
  """Filters a (possibly raw HF) transformer config to WanModel.__init__ arguments."""
  from maxtext.checkpoint_conversion.to_maxtext_diffusers import model_init_config  # pylint: disable=import-outside-toplevel

  model_config = dict(model_config)
  model_config.pop("scan_layers", None)
  return model_init_config(WanModel, model_config)


def create_random_sharded_transformer(mesh: Mesh, config: DiffusionConfig, wan_config: dict, seed: int = 0):
  """Builds a randomly initialized WanModel directly on `mesh` shardings (no checkpoint read).

  Used for performance benchmarking (the GPU baseline also trains from a random init on mock data).
  """
  wan_config = _wan_model_kwargs(config, model_init_config_for_wan(wan_config))
  abstract = nnx.eval_shape(lambda: WanModel(**wan_config, rngs=nnx.Rngs(seed)))
  graphdef, abstract_state = nnx.split(abstract)
  shardings = state_shardings(abstract_state, mesh)
  # Flat leaves: Variable metadata (e.g. init fns) differs between traces, so whole-state out_shardings mismatch.
  state_leaves, state_treedef = jax.tree.flatten(abstract_state)
  sharding_leaves = jax.tree.leaves(shardings, is_leaf=lambda x: isinstance(x, NamedSharding))
  if len(sharding_leaves) != len(state_leaves):
    raise ValueError(f"{len(sharding_leaves)} shardings for {len(state_leaves)} WanModel state leaves.")

  @partial(jax.jit, out_shardings=sharding_leaves)
  def init_leaves():
    return jax.tree.leaves(nnx.state(WanModel(**wan_config, rngs=nnx.Rngs(seed))))

  return nnx.merge(graphdef, jax.tree.unflatten(state_treedef, init_leaves()))


def create_sharded_logical_transformer(
    devices_array: np.array,
    mesh: Mesh,
    rngs: nnx.Rngs,
    config: DiffusionConfig,
    restored_checkpoint=None,
    subfolder: str = "",
):
  """Build a WanModel via eval_shape and place checkpoint params directly onto `mesh` shardings."""

  def create_model(rngs: nnx.Rngs, wan_config: dict):
    wan_transformer = WanModel(**wan_config, rngs=rngs)
    return wan_transformer

  # 1. Load config and params from the restored (or in-memory converted) checkpoint.
  checkpoint_state, wan_config = _restored_component(restored_checkpoint, subfolder, WanModel)
  ckpt_scan_layers = wan_config.pop("scan_layers", None)
  wan_config = _wan_model_kwargs(config, wan_config, ckpt_scan_layers)

  # 2. eval_shape - will not use flops or create weights on device
  # thus not using HBM memory.
  p_model_factory = partial(create_model, wan_config=wan_config)
  wan_transformer = nnx.eval_shape(p_model_factory, rngs=rngs)
  graphdef, state, rest_of_state = nnx.split(wan_transformer, nnx.Param, ...)

  # 3. retrieve the state shardings (the physical specs `WanSharding` set on each param).
  logical_state_sharding = dict(nnx.to_flat_state(state_shardings(state, mesh)))
  state = dict(nnx.to_flat_state(state))

  # 4. Move the checkpoint params to device using the state shardings from (3) above.
  # This helps with loading sharded weights directly into the accelerators without first copying them
  # all to one device and then distributing them, thus using low HBM memory.
  if "params" in checkpoint_state:  # if checkpointed with optimizer
    params = checkpoint_state["params"]
  else:  # if not checkpointed with optimizer
    params = checkpoint_state

  # No-op (returns leaves unchanged) when the converter already cast to the
  # final dtypes.
  params = jax.tree_util.tree_map_with_path(
      lambda path, x: cast_with_exclusion(path, x, dtype_to_cast=config.weights_dtype),
      params,
  )
  state = put_params_into_state(
      state,
      params,
      logical_state_sharding,
      mesh,
      restored_checkpoint=True,
      subfolder=subfolder,
  )
  state = nnx.from_flat_state(state)

  wan_transformer = nnx.merge(graphdef, state, rest_of_state)
  return wan_transformer


def check_text_encoder_sharding(mesh: Mesh, text_encoder: UMT5EncoderModel) -> None:
  """Validates the text encoder's sharding (`text_encoder_logical_axis_rules`) against `mesh`.

  The table shards the UMT5 text encoder Megatron-style (tensor parallel): the "heads" / "mlp" / "vocab" dims
  (q/k/v/wi columns, o/wo rows, embedding rows) are split over the mesh axes `text_encoder_logical_axis_rules` gives
  them (every mesh axis by default), while "embed" dims and
  the activations' batch / length stay replicated, so each block needs one all-reduce after `proj_attn` and one after
  `wo`. On a v6e-8 split 8-way, umt5-xxl fp32 (22.7 GB) is 2.8 GB per chip.

  Limit: those mesh axes must multiply to a divisor of gcd(num_heads, d_ff, vocab_size), which is 64 for umt5-xxl (64
  heads). Splitting over all mesh axes therefore works up to 64 chips; a bigger mesh needs fewer axes in
  `text_encoder_logical_axis_rules` (the encoder is then replicated over the dropped axes).
  """
  check_tensor_parallel_divides(
      text_encoder.sharding, mesh, math.gcd(text_encoder.num_heads, text_encoder.d_ff, text_encoder.vocab_size)
  )


@partial(jax.jit, static_argnums=(0,))
def text_encoder_forward_pass(graphdef, state, input_ids, attention_mask):
  """UMT5 last hidden state in float32, zero past each prompt's length (as diffusers' `_get_t5_prompt_embeds`)."""
  prompt_embeds = nnx.merge(graphdef, state)(input_ids, attention_mask)
  seq_lens = jnp.sum(attention_mask > 0, axis=1)
  valid = jnp.arange(input_ids.shape[1])[None, :] < seq_lens[:, None]
  return jnp.where(valid[..., None], prompt_embeds, 0).astype(jnp.float32)


class WanPipeline(DiffusionPipeline):
  r"""Pipeline for text-to-video generation using Wan.

  Every model component is built from the converted MaxText checkpoint (`to_maxtext_diffusers.py`):

  - text_encoder (`UMT5EncoderModel`, `maxtext.m3.models.wan.text_encoder_wan`): UMT5-XXL prompt encoder.
  - transformer (`WanModel`): conditional transformer that denoises the latents.
  - vae (`AutoencoderKLWan`): encodes / decodes videos to / from latents.
  - scheduler (`FlaxUniPCMultistepScheduler`): built from the checkpoint's `scheduler_config`.

  The tokenizer is the only component loaded from the Hugging Face model repo (`pretrained_model_name_or_path`,
  subfolder `tokenizer`): it has no weights to convert (a SentencePiece vocabulary plus config files), and
  `AutoTokenizer` reads it directly, as MaxText's LLM input pipelines do with `tokenizer_path`. The converter
  downloads it into the HF cache together with the weights.
  """

  def __init__(
      self,
      tokenizer: AutoTokenizer,
      text_encoder: UMT5EncoderModel,
      vae: AutoencoderKLWan,
      vae_cache: AutoencoderKLWanCache,
      scheduler: FlaxUniPCMultistepScheduler,
      scheduler_state: UniPCMultistepSchedulerState,
      devices_array: np.array,
      mesh: Mesh,
      config: DiffusionConfig,
      image_processor: Optional[CLIPImageProcessor] = None,
      image_encoder: Optional[FlaxCLIPVisionModel] = None,
      **kwargs,
  ):
    self.tokenizer = tokenizer
    self.text_encoder = text_encoder
    self.vae = vae
    self.vae_cache = vae_cache
    self.scheduler = scheduler
    self.scheduler_state = scheduler_state
    self.devices_array = devices_array
    self.mesh = mesh
    self.config = config
    self.model_name = config.model_name
    self.image_processor = image_processor
    self.image_encoder = image_encoder

    self.vae_mesh = kwargs.get("vae_mesh", mesh)

    self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
    self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
    self.video_processor = VaeVideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)  # For i2v tasks to be added.

    self.p_run_inference = None
    # encode_prompt result cache: same-prompt calls (warmup + real run,
    # repeated serving requests) skip the text encoder.
    self._prompt_embeds_cache = {}

  def check_inputs(
      self,
      prompt: Union[str, List[str]] = None,
      negative_prompt: Optional[Union[str, List[str]]] = None,
      height: int = 480,
      width: int = 832,
      prompt_embeds: Optional[jax.Array] = None,
      negative_prompt_embeds: Optional[jax.Array] = None,
      **kwargs,
  ):
    """Validate user-facing pipeline inputs and shape contracts."""
    if prompt is not None and prompt_embeds is not None:
      raise ValueError(
          f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`:"
          f" {prompt_embeds}. Please make sure to only forward one of the two."
      )
    elif negative_prompt is not None and negative_prompt_embeds is not None:
      raise ValueError(
          f"Cannot forward both `negative_prompt`: {negative_prompt} and"
          f" `negative_prompt_embeds`: {negative_prompt_embeds}. Please make"
          " sure to only forward one of the two."
      )

    mesh = getattr(self, "vae_mesh", getattr(self, "mesh", None))
    if mesh is not None and hasattr(mesh, "shape"):
      vae_spatial = mesh.shape.get("vae_spatial", 1)
      if vae_spatial > 1 and (width // 8) % vae_spatial != 0:
        max_logging.log(
            "Warning: Latent width is not divisible by vae_spatial mesh axis"
            f" ({vae_spatial}). VAE spatial sharding will be partially"
            " bypassed."
        )

  @classmethod
  def load_text_encoder(
      cls,
      mesh: Mesh,
      rngs: nnx.Rngs,
      config: DiffusionConfig,
      restored_checkpoint=None,
  ) -> UMT5EncoderModel:
    """Load the NNX UMT5 text encoder from the checkpoint onto `mesh`, in `text_encoder_dtype`.

    Params are sharded by `text_encoder_logical_axis_rules` (tensor parallel over the mesh, see
    `check_text_encoder_sharding`). The
    forward runs at `Precision.HIGHEST`, so a float32 encoder matches the torch reference instead of using one bf16
    pass.
    """
    te_params, te_config = _restored_component(restored_checkpoint, "text_encoder", UMT5EncoderModel)
    dtype = jnp.dtype(getattr(config, "text_encoder_dtype", "float32"))

    def create_model(rngs: nnx.Rngs):
      return UMT5EncoderModel(
          **te_config,
          dtype=dtype,
          weights_dtype=dtype,
          precision=jax.lax.Precision.HIGHEST,
          sharding=WanSharding.from_rules(config.text_encoder_logical_axis_rules, WAN_MESH_AXES),
          rngs=rngs,
      )

    # 1. eval shape (no weights allocated), 2. the params' shardings on `mesh`, 3. place the checkpoint params.
    text_encoder = nnx.eval_shape(create_model, rngs=rngs)
    graphdef, state, rest_of_state = nnx.split(text_encoder, nnx.Param, ...)
    check_text_encoder_sharding(mesh, text_encoder)
    shardings = dict(nnx.to_flat_state(state_shardings(state, mesh)))
    state = dict(nnx.to_flat_state(state))

    t_start = time.perf_counter()
    paths, values = [], []
    for path, val in flax.traverse_util.flatten_dict(te_params).items():
      path = _to_int_path(path[:-1] if path[-1] == "value" else path)
      if path not in state:
        raise ValueError(f"Checkpoint text_encoder param {path} is not a UMT5EncoderModel param.")
      paths.append(path)
      values.append(np.asarray(val).astype(dtype, copy=False))
    if len(paths) != len(state):
      missing = sorted(set(state) - set(paths))
      raise ValueError(f"Checkpoint text_encoder_state misses {len(missing)} params, e.g. {missing[:3]}.")
    with _DEVICE_PUT_LOCK:  # Do not interleave with a concurrent transformer transfer.
      arrays = jax.device_put(values, [shardings[path].value for path in paths])
      jax.block_until_ready(arrays)
    for path, array in zip(paths, arrays):
      state[path].value = array
    num_bytes = sum(v.nbytes for v in values)
    max_logging.log(
        f"Text encoder ({num_bytes / 2**30:.1f} GiB {dtype.name}, heads sharded as"
        f" {text_encoder.sharding('heads', ('heads',))}) on device in {time.perf_counter() - t_start:.1f}s"
    )
    return nnx.merge(graphdef, nnx.from_flat_state(state), rest_of_state)

  @classmethod
  def load_tokenizer(cls, config: DiffusionConfig):
    """Load the text tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    return tokenizer

  @classmethod
  def load_image_encoder(cls, config: DiffusionConfig):
    """Load the CLIP image processor and Flax CLIP vision encoder (WAN 2.1 I2V)."""
    image_processor = CLIPImageProcessor.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="image_processor"
    )
    try:
      image_encoder = FlaxCLIPVisionModel.from_pretrained(
          config.pretrained_model_name_or_path,
          subfolder="image_encoder",
          dtype=jnp.float32,
      )
    except Exception as e:
      max_logging.error(f"Failed to load FlaxCLIPVisionModel: {e}")
      raise
    return image_processor, image_encoder

  @classmethod
  def load_vae(
      cls,
      devices_array: np.array,
      mesh: Mesh,
      rngs: nnx.Rngs,
      config: DiffusionConfig,
      restored_checkpoint=None,
  ):
    """Load the Wan VAE from the checkpoint onto the VAE mesh `mesh`; returns (vae, vae_cache)."""
    vae_params, vae_config = _restored_component(restored_checkpoint, "vae", AutoencoderKLWan)

    def create_model(rngs: nnx.Rngs, config: DiffusionConfig):
      wan_vae = AutoencoderKLWan(
          **{
              **vae_config,
              "sharding": WanSharding.from_rules(config.vae_logical_axis_rules, VAE_MESH_AXES),
              "dtype": config.vae_dtype,
              "weights_dtype": config.vae_weights_dtype,
              "vae_decode_chunk": config.vae_decode_chunk,
              "vae_encode_chunk": config.vae_encode_chunk,
          },
          rngs=rngs,
      )
      return wan_vae

    # 1. eval shape
    p_model_factory = partial(create_model, config=config)
    wan_vae = nnx.eval_shape(p_model_factory, rngs=rngs)
    graphdef, state = nnx.split(wan_vae, nnx.Param)

    # 2. retrieve the state shardings (the physical specs the VAE's `WanSharding` set on each param).
    logical_state_sharding = dict(nnx.to_flat_state(state_shardings(state, mesh)))
    state = dict(nnx.to_flat_state(state))

    # 3. Move the checkpoint weights to device using the state shardings from (2) above.
    # This helps with loading sharded weights directly into the accelerators without fist copying them
    # all to one device and then distributing them, thus using low HBM memory.
    weights_dtype = jnp.dtype(config.weights_dtype)
    for path, val in flax.traverse_util.flatten_dict(vae_params).items():
      path = _to_int_path(path[:-1] if path[-1] == "value" else path)
      val = val.astype(weights_dtype)
      target_sharding = logical_state_sharding[path].value
      if config.replicate_vae:
        target_sharding = NamedSharding(mesh, P())
      state[path].value = device_put_replicated(val, target_sharding)
    state = nnx.from_flat_state(state)

    wan_vae = nnx.merge(graphdef, state)
    vae_cache = AutoencoderKLWanCache(wan_vae)
    return wan_vae, vae_cache

  @classmethod
  def load_transformer(
      cls,
      devices_array: np.array,
      mesh: Mesh,
      rngs: nnx.Rngs,
      config: DiffusionConfig,
      restored_checkpoint=None,
      subfolder="transformer",
  ):
    """Load a sharded WanModel transformer from the restored checkpoint."""
    with jax.set_mesh(mesh):
      wan_transformer = create_sharded_logical_transformer(
          devices_array=devices_array,
          mesh=mesh,
          rngs=rngs,
          config=config,
          restored_checkpoint=restored_checkpoint,
          subfolder=subfolder,
      )
    return wan_transformer

  @classmethod
  def load_scheduler(cls, config, restored_checkpoint):
    """Build the UniPC scheduler and its state from the checkpoint's scheduler_config."""
    if "scheduler_config" not in set(restored_checkpoint.keys()):
      raise ValueError(
          "Checkpoint has no `scheduler_config`. Re-run"
          " checkpoint_conversion/to_maxtext_diffusers.py to convert all components."
      )
    scheduler, scheduler_state = FlaxUniPCMultistepScheduler.from_config(
        dict(restored_checkpoint["scheduler_config"]),
        flow_shift=config.flow_shift,  # 5.0 for 720p, 3.0 for 480p
        dtype=config.scheduler_dtype,
    )
    return scheduler, scheduler_state

  def encode_image(self, image: PipelineImageInput, num_videos_per_prompt: int = 1):
    """Return CLIP penultimate-layer image embeddings, repeated per video."""
    if not isinstance(image, list):
      image = [image]
    image_inputs = self.image_processor(images=image, return_tensors="np")
    pixel_values = jnp.array(image_inputs.pixel_values)

    image_encoder_output = self.image_encoder(pixel_values, output_hidden_states=True)
    image_embeds = image_encoder_output.hidden_states[-2]

    image_embeds = jnp.repeat(image_embeds, num_videos_per_prompt, axis=0)
    return image_embeds

  def _get_t5_prompt_embeds(
      self,
      prompt: Union[str, List[str]] = None,
      num_videos_per_prompt: int = 1,
      max_sequence_length: Optional[int] = None,
  ):
    """Encode prompts with the UMT5 text encoder, zero-padded to max_sequence_length."""
    if max_sequence_length is None:
      max_sequence_length = getattr(self.config, "max_sequence_length", 512)

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]

    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="np",
    )
    graphdef, state = nnx.split(self.text_encoder)
    with jax.set_mesh(self.mesh):
      prompt_embeds = text_encoder_forward_pass(
          graphdef, state, jnp.asarray(text_inputs.input_ids), jnp.asarray(text_inputs.attention_mask)
      )

    # duplicate text embeddings for each generation per prompt (prompt-major, like diffusers' repeat + view)
    prompt_embeds = jnp.repeat(prompt_embeds, num_videos_per_prompt, axis=0)
    return prompt_embeds

  def encode_prompt(
      self,
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      num_videos_per_prompt: int = 1,
      max_sequence_length: Optional[int] = None,
      prompt_embeds: jax.Array = None,
      negative_prompt_embeds: jax.Array = None,
  ):
    """Return (prompt_embeds, negative_prompt_embeds), encoding with UMT5 and caching by prompt."""
    if max_sequence_length is None:
      max_sequence_length = getattr(self.config, "max_sequence_length", 512)

    if prompt is not None:
      prompt = [prompt] if isinstance(prompt, str) else prompt
      batch_size = len(prompt)
    else:
      batch_size = prompt_embeds.shape[0] // num_videos_per_prompt

    if negative_prompt is None:
      negative_prompt = [""] * batch_size
    elif isinstance(negative_prompt, str):
      negative_prompt = [negative_prompt] * batch_size

    # Same-prompt calls (warmup then the real generation, or repeated
    # serving requests) should not re-run the text encoder.
    cache_key = None
    if prompt is not None and prompt_embeds is None and negative_prompt_embeds is None:
      cache_key = (
          tuple(prompt),
          tuple(negative_prompt),
          num_videos_per_prompt,
          max_sequence_length,
      )
      cached = self._prompt_embeds_cache.get(cache_key)
      if cached is not None:
        return cached

    use_batched_text_encoder = self.config.use_batched_text_encoder
    if use_batched_text_encoder and prompt_embeds is None and negative_prompt_embeds is None:
      # Batch both together
      combined_prompts = prompt + negative_prompt
      combined_embeds = self._get_t5_prompt_embeds(
          prompt=combined_prompts,
          num_videos_per_prompt=num_videos_per_prompt,
          max_sequence_length=max_sequence_length,
      )

      # Split back
      prompt_embeds = combined_embeds[: batch_size * num_videos_per_prompt]
      negative_prompt_embeds = combined_embeds[batch_size * num_videos_per_prompt :]

    else:
      # Fallback to separate encoding if one of them is already provided
      if prompt_embeds is None:
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
        )

      if negative_prompt_embeds is None:
        negative_prompt_embeds = self._get_t5_prompt_embeds(
            prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    if cache_key is not None:
      if len(self._prompt_embeds_cache) >= 16:  # bound long-serving growth
        self._prompt_embeds_cache.pop(next(iter(self._prompt_embeds_cache)))
      self._prompt_embeds_cache[cache_key] = (
          prompt_embeds,
          negative_prompt_embeds,
      )

    return prompt_embeds, negative_prompt_embeds

  def prepare_latents(
      self,
      batch_size: int,
      vae_scale_factor_temporal: int,
      vae_scale_factor_spatial: int,
      height: int = 480,
      width: int = 832,
      num_frames: int = 81,
      num_channels_latents: int = 16,
  ):
    """Sample initial Gaussian latents (B, C, T', H', W') from config.seed."""
    rng = jax.random.key(self.config.seed)
    num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
    shape = (
        batch_size,
        num_channels_latents,
        num_latent_frames,
        int(height) // vae_scale_factor_spatial,
        int(width) // vae_scale_factor_spatial,
    )
    latents = jax.random.normal(rng, shape=shape, dtype=jnp.float32)

    return latents

  def prepare_latents_i2v_base(
      self,
      image: jax.Array,
      num_frames: int,
      dtype: jnp.dtype,
      last_image: Optional[jax.Array] = None,
      trace: Optional[dict] = None,
  ) -> Tuple[jax.Array, jax.Array]:
    """Encodes the initial image(s) into latents to be used as conditioning.

    Returns:
        latent_condition: The VAE encoded latents of the image(s).
        video_condition: The input to the VAE.
    """
    height, width = image.shape[-2:]
    image = image[:, :, jnp.newaxis, :, :]  # [B, C, 1, H, W]

    if last_image is None:
      video_condition = jnp.concatenate(
          [
              image,
              jnp.zeros(
                  (
                      image.shape[0],
                      image.shape[1],
                      num_frames - 1,
                      height,
                      width,
                  ),
                  dtype=image.dtype,
              ),
          ],
          axis=2,
      )
    else:
      last_image = last_image[:, :, jnp.newaxis, :, :]
      video_condition = jnp.concatenate(
          [
              image,
              jnp.zeros(
                  (
                      image.shape[0],
                      image.shape[1],
                      num_frames - 2,
                      height,
                      width,
                  ),
                  dtype=image.dtype,
              ),
              last_image,
          ],
          axis=2,
      )

    vae_dtype = getattr(self.vae, "dtype", jnp.float32)
    video_condition = video_condition.astype(vae_dtype)
    t_vae_encode_start = time.perf_counter()
    with jax.set_mesh(self.vae_mesh):
      graphdef, state, rest_of_state = nnx.split(self.vae, nnx.Param, ...)
      encoded_output = vae_encode_pass(graphdef, state, rest_of_state, video_condition)
      if hasattr(encoded_output, "block_until_ready"):
        encoded_output.block_until_ready()

    if trace is not None:
      trace["vae_encode"] = time.perf_counter() - t_vae_encode_start

    # Normalize latents
    latents_mean = jnp.array(self.vae.latents_mean).reshape(1, 1, 1, 1, self.vae.z_dim)
    latents_std = jnp.array(self.vae.latents_std).reshape(1, 1, 1, 1, self.vae.z_dim)
    latent_condition = encoded_output
    latent_condition = latent_condition.astype(dtype)
    latent_condition = (latent_condition - latents_mean) / latents_std

    return latent_condition, video_condition

  def _denormalize_latents(self, latents: jax.Array) -> jax.Array:
    """Denormalizes latents using VAE statistics."""
    dtype = self.config.activations_dtype
    latents_mean = jnp.array(self.vae.latents_mean, dtype=dtype).reshape(1, self.vae.z_dim, 1, 1, 1)
    latents_std = 1.0 / jnp.array(self.vae.latents_std, dtype=dtype).reshape(1, self.vae.z_dim, 1, 1, 1)
    latents = latents / latents_std + latents_mean
    return latents

  def _decode_latents_to_video(self, latents: jax.Array, trace: Optional[dict] = None) -> np.ndarray:
    """Decodes latents to video frames and postprocesses."""
    t_vae_tpu_start = time.perf_counter()
    with jax.set_mesh(self.vae_mesh):
      graphdef, state, rest_of_state = nnx.split(self.vae, nnx.Param, ...)
      video = vae_decode_pass(graphdef, state, rest_of_state, latents)
      video.block_until_ready()
    if trace is not None:
      trace["vae_decode_tpu"] = time.perf_counter() - t_vae_tpu_start

    if hasattr(video, "addressable_shards") and len(video.addressable_shards) > 0:
      if video.addressable_shards[0].data.shape[0] < video.shape[0]:
        video = np.concatenate(
            [np.asarray(shard.data) for shard in video.addressable_shards],
            axis=0,
        )
      else:
        video = np.asarray(video.addressable_shards[0].data)
    else:
      video = np.asarray(video)
    return video

  @classmethod
  def _create_common_components(
      cls,
      config,
      restored_checkpoint=None,
      load_vae=True,
      load_text_encoder=True,
      load_scheduler=True,
      i2v=False,
  ):
    """Create the mesh and load the shared VAE, text encoder, tokenizer and scheduler components."""
    mesh = create_wan_mesh(config)
    devices_array = mesh.devices

    vae_spatial = getattr(config, "vae_spatial", -1)
    total_devices = math.prod(devices_array.shape)

    if vae_spatial == -1:
      vae_spatial = total_devices

    assert total_devices % vae_spatial == 0, (
        f"total devices ({total_devices}) must be a multiple of vae_spatial" f" ({vae_spatial})"
    )

    flat_devices = devices_array.flatten()
    vae_devices_array = flat_devices.reshape(total_devices // vae_spatial, vae_spatial)

    vae_mesh = Mesh(vae_devices_array, VAE_MESH_AXES)
    max_logging.log(f"Created VAE specific mesh with axes {VAE_MESH_AXES} to support spatial sharding of {vae_spatial}.")

    rng = jax.random.key(config.seed)
    rngs = nnx.Rngs(rng)

    components = {
        "vae": None,
        "vae_cache": None,
        "devices_array": devices_array,
        "rngs": rngs,
        "mesh": mesh,
        "vae_mesh": vae_mesh,
        "tokenizer": None,
        "text_encoder": None,
        "scheduler": None,
        "scheduler_state": None,
        "image_processor": None,
        "image_encoder": None,
    }

    if load_vae:
      max_logging.log("Loading VAE")
      components["vae"], components["vae_cache"] = cls.load_vae(
          devices_array=devices_array,
          mesh=vae_mesh,
          rngs=rngs,
          config=config,
          restored_checkpoint=restored_checkpoint,
      )

    if load_text_encoder:
      max_logging.log("Loading Tokenizer and Text Encoder")
      components["tokenizer"] = cls.load_tokenizer(config=config)
      components["text_encoder"] = cls.load_text_encoder(
          mesh=mesh, rngs=rngs, config=config, restored_checkpoint=restored_checkpoint
      )
      if cls._needs_image_encoder(config, i2v=i2v):
        (
            components["image_processor"],
            components["image_encoder"],
        ) = cls.load_image_encoder(config)

    if load_scheduler:
      components["scheduler"], components["scheduler_state"] = cls.load_scheduler(
          config=config, restored_checkpoint=restored_checkpoint
      )

    return components

  @classmethod
  def _needs_image_encoder(cls, config: DiffusionConfig, i2v: bool = False) -> bool:
    return i2v and config.model_name == "wan2.1"

  @abstractmethod
  def _get_num_channel_latents(self) -> int:
    """Returns the number of input channels for the transformer."""

  def _prepare_model_inputs_i2v(
      self,
      prompt: Union[str, List[str]],
      image: Union[PIL.Image.Image, List[PIL.Image.Image]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      num_videos_per_prompt: int = 1,
      max_sequence_length: Optional[int] = None,
      prompt_embeds: Optional[jax.Array] = None,
      negative_prompt_embeds: Optional[jax.Array] = None,
      image_embeds: Optional[jax.Array] = None,
      last_image: Optional[PIL.Image.Image] = None,
  ):
    """Encode prompts (and CLIP image embeddings for WAN 2.1) for I2V."""
    if max_sequence_length is None:
      max_sequence_length = getattr(self.config, "max_sequence_length", 512)
    if prompt is not None and isinstance(prompt, str):
      prompt = [prompt]
    batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0] // num_videos_per_prompt
    effective_batch_size = batch_size * num_videos_per_prompt

    # 1. Encode Prompts
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # 2. Encode Image (only for WAN 2.1 I2V which uses CLIP image embeddings)
    # WAN 2.2 I2V does not use CLIP image embeddings, it uses VAE latent conditioning instead
    transformer_dtype = self.config.activations_dtype

    if self.config.model_name == "wan2.1":
      # WAN 2.1 I2V: Use CLIP image encoder
      if image_embeds is None:
        if last_image is not None:
          # TODO: first-last-frame (FLF2V) conditioning needs both images' CLIP tokens in one sequence (diffusers
          # folds them in `WanImageEmbedding`); not ported, so stacking them along the batch would mismatch the latents.
          raise NotImplementedError("last_image (Wan 2.1 first-last-frame conditioning) is not supported yet.")
        image_embeds = self.encode_image(image, num_videos_per_prompt=num_videos_per_prompt)
        self.image_seq_len = image_embeds.shape[1]

      if batch_size > 1 and image_embeds.shape[0] == num_videos_per_prompt:
        # One conditioning image shared by every prompt (a list of `batch_size` images is already per prompt).
        image_embeds = jnp.tile(image_embeds, (batch_size, 1, 1))

      image_embeds = image_embeds.astype(transformer_dtype)
    else:
      # WAN 2.2 I2V: No CLIP image embeddings, set to None or empty tensor
      # The actual image conditioning happens via VAE latents in prepare_latents
      image_embeds = None
    prompt_embeds = prompt_embeds.astype(transformer_dtype)
    if negative_prompt_embeds is not None:
      negative_prompt_embeds = negative_prompt_embeds.astype(transformer_dtype)

    # Use same sharding logic as T2V pipeline for consistent behavior
    data_sharding = NamedSharding(self.mesh, P())
    if self.config.global_batch_size_to_train_on // self.config.per_device_batch_size == 0:
      data_sharding = jax.sharding.NamedSharding(self.mesh, P(*self.config.data_sharding))

    prompt_embeds = jax.device_put(prompt_embeds, data_sharding)
    negative_prompt_embeds = jax.device_put(negative_prompt_embeds, data_sharding)
    if image_embeds is not None:
      image_embeds = jax.device_put(image_embeds, data_sharding)

    return (
        prompt_embeds,
        negative_prompt_embeds,
        image_embeds,
        effective_batch_size,
    )

  def _prepare_model_inputs(
      self,
      prompt: Union[str, List[str]] = None,
      negative_prompt: Union[str, List[str]] = None,
      height: int = 480,
      width: int = 832,
      num_frames: int = 81,
      num_inference_steps: int = 50,
      num_videos_per_prompt: Optional[int] = 1,
      max_sequence_length: Optional[int] = None,
      latents: jax.Array = None,
      prompt_embeds: jax.Array = None,
      negative_prompt_embeds: jax.Array = None,
  ):
    """Encode prompts, prepare initial latents and scheduler state for T2V."""
    self.check_inputs(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )
    if max_sequence_length is None:
      max_sequence_length = getattr(self.config, "max_sequence_length", 512)

    if num_frames % self.vae_scale_factor_temporal != 1:
      max_logging.log(
          "`num_frames -1` has to be divisible by" f" {self.vae_scale_factor_temporal}. Rounding to the nearest number."
      )
      num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
    num_frames = max(num_frames, 1)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
      prompt = [prompt]

    batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0] // num_videos_per_prompt

    debug_timers = bool(os.environ.get("WAN_DEBUG_COND_TIMERS"))
    t_probe = time.perf_counter()
    with jax.named_scope("Encode-Prompt"):
      prompt_embeds, negative_prompt_embeds = self.encode_prompt(
          prompt=prompt,
          negative_prompt=negative_prompt,
          max_sequence_length=max_sequence_length,
          prompt_embeds=prompt_embeds,
          negative_prompt_embeds=negative_prompt_embeds,
      )
    if debug_timers:
      max_logging.log(f"[cond] encode_prompt {time.perf_counter() - t_probe:.1f}s")
      t_probe = time.perf_counter()

    num_channel_latents = self._get_num_channel_latents()
    if latents is None:
      latents = self.prepare_latents(
          batch_size=batch_size,
          vae_scale_factor_temporal=self.vae_scale_factor_temporal,
          vae_scale_factor_spatial=self.vae_scale_factor_spatial,
          height=height,
          width=width,
          num_frames=num_frames,
          num_channels_latents=num_channel_latents,
      )
    if debug_timers:
      max_logging.log(f"[cond] prepare_latents {time.perf_counter() - t_probe:.1f}s")
      t_probe = time.perf_counter()

    data_sharding = NamedSharding(self.mesh, P())
    # Using global_batch_size_to_train_on so not to create more config variables
    if self.config.global_batch_size_to_train_on // self.config.per_device_batch_size == 0:
      data_sharding = jax.sharding.NamedSharding(self.mesh, P(*self.config.data_sharding))

    latents = jax.device_put(latents, data_sharding)
    prompt_embeds = jax.device_put(prompt_embeds, data_sharding)
    negative_prompt_embeds = jax.device_put(negative_prompt_embeds, data_sharding)
    if debug_timers:
      jax.block_until_ready([latents, prompt_embeds, negative_prompt_embeds])
      max_logging.log(f"[cond] device_put {time.perf_counter() - t_probe:.1f}s")
      t_probe = time.perf_counter()

    scheduler_state = self.scheduler.set_timesteps(
        self.scheduler_state,
        num_inference_steps=num_inference_steps,
        shape=latents.shape,
    )
    if debug_timers:
      max_logging.log(f"[cond] set_timesteps {time.perf_counter() - t_probe:.1f}s")

    return (
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        scheduler_state,
        num_frames,
    )

  @abstractmethod
  def __call__(self, **kwargs):
    """Runs the inference pipeline."""


@partial(
    aot_cache.cached_jit,
    static_argnames=(
        "do_classifier_free_guidance",
        "return_residual",
        "skip_blocks",
    ),
)
def transformer_forward_pass(
    graphdef,
    sharded_state,
    rest_of_state,
    latents,
    timestep,
    prompt_embeds,
    do_classifier_free_guidance,
    guidance_scale,
    encoder_hidden_states_image=None,
    skip_blocks=None,
    cached_residual=None,
    return_residual=False,
    kv_cache=None,
    rotary_emb=None,
    encoder_attention_mask=None,
):
  """One jitted transformer forward (optionally CFG-batched) from a split nnx model."""
  if do_classifier_free_guidance and latents.shape[0] != prompt_embeds.shape[0]:
    latents = jnp.concatenate([latents, latents], axis=0)
  wan_transformer = nnx.merge(graphdef, sharded_state, rest_of_state)
  outputs = wan_transformer(
      hidden_states=latents,
      timestep=timestep,
      encoder_hidden_states=prompt_embeds,
      encoder_hidden_states_image=encoder_hidden_states_image,
      skip_blocks=skip_blocks,
      cached_residual=cached_residual,
      return_residual=return_residual,
      kv_cache=kv_cache,
      rotary_emb=rotary_emb,
      encoder_attention_mask=encoder_attention_mask,
  )

  if return_residual:
    noise_pred, residual_x = outputs
  else:
    noise_pred = outputs

  if do_classifier_free_guidance:
    bsz = latents.shape[0] // 2
    noise_cond = noise_pred[:bsz]  # First half = conditional
    noise_uncond = noise_pred[bsz:]  # Second half = unconditional
    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

  if return_residual:
    return noise_pred, residual_x
  return noise_pred


@aot_cache.cached_jit
def vae_encode_pass(graphdef, state, rest_of_state, video):
  """Encodes conditioning video to its deterministic latent (I2V path)."""
  wan_vae = nnx.merge(graphdef, state, rest_of_state)
  return wan_vae.encode(video, AutoencoderKLWanCache(wan_vae), return_dict=False)[0].mode()


@aot_cache.cached_jit
def vae_decode_pass(graphdef, state, rest_of_state, latents):
  """Decodes latents and postprocesses to uint8 frames as ONE executable.

  Wrapped in the AOT cache so warm processes deserialize instead of paying
  the deep conv-stack trace + compile-cache lookup, and zero-exec warmup
  skips the decode compute entirely. The feat cache is rebuilt from the
  merged module — it is pure per-call temporary state.
  """
  wan_vae = nnx.merge(graphdef, state, rest_of_state)
  video = wan_vae.decode(latents, AutoencoderKLWanCache(wan_vae), return_dict=False)[0]
  video = (video / 2.0) + 0.5
  video = jnp.clip(video, 0.0, 1.0)
  video = (video * 255.0).astype(jnp.uint8)
  if not jax.sharding.get_abstract_mesh().empty:
    video = jax.lax.with_sharding_constraint(video, P())  # Replicated on the VAE mesh in scope.
  return video


@aot_cache.cached_jit
def transformer_forward_pass_full_cfg(
    graphdef,
    sharded_state,
    rest_of_state,
    latents_doubled: jnp.array,
    timestep: jnp.array,
    prompt_embeds_combined: jnp.array,
    guidance_scale: float,
    encoder_hidden_states_image=None,
    kv_cache=None,
    rotary_emb=None,
    encoder_attention_mask=None,
):
  """Full CFG forward pass.

  Accepts pre-doubled latents and pre-concatenated [cond, uncond] prompt embeds.
  Returns the merged noise_pred plus raw noise_cond and noise_uncond for
  CFG cache storage.  Keeping cond/uncond separate avoids a second forward
  pass on cache steps.
  """
  wan_transformer = nnx.merge(graphdef, sharded_state, rest_of_state)
  bsz = latents_doubled.shape[0] // 2
  noise_pred = wan_transformer(
      hidden_states=latents_doubled,
      timestep=timestep,
      encoder_hidden_states=prompt_embeds_combined,
      encoder_hidden_states_image=encoder_hidden_states_image,
      skip_blocks=False,
      cached_residual=None,
      return_residual=False,
      kv_cache=kv_cache,
      rotary_emb=rotary_emb,
      encoder_attention_mask=encoder_attention_mask,
  )
  noise_cond = noise_pred[:bsz]
  noise_uncond = noise_pred[bsz:]
  noise_pred_merged = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
  return noise_pred_merged, noise_cond, noise_uncond


@aot_cache.cached_jit
def transformer_forward_pass_cfg_cache(
    graphdef,
    sharded_state,
    rest_of_state,
    latents_cond: jnp.array,
    timestep_cond: jnp.array,
    prompt_cond_embeds: jnp.array,
    cached_noise_cond: jnp.array,
    cached_noise_uncond: jnp.array,
    guidance_scale: float,
    w1: float = 1.0,
    w2: float = 1.0,
    encoder_hidden_states_image=None,
    kv_cache=None,
    rotary_emb=None,
    encoder_attention_mask=None,
):
  """CFG-Cache forward pass with FFT frequency-domain compensation.

  FasterCache (Lv et al., ICLR 2025) CFG-Cache:
    1. Compute frequency-domain bias:  ΔF = FFT(uncond) - FFT(cond)
    2. Split into low-freq (ΔLF) and high-freq (ΔHF) via spectral mask
    3. Apply phase-dependent weights:
         F_low  = FFT(new_cond)_low  + w1 * ΔLF
         F_high = FFT(new_cond)_high + w2 * ΔHF
    4. Reconstruct:  uncond_approx = IFFT(F_low + F_high)

  w1/w2 encode the denoising phase:
    Early (high noise): w1=1+α, w2=1   → boost low-freq correction
    Late  (low noise):  w1=1,   w2=1+α → boost high-freq correction
  where α=0.2 (FasterCache default).

  On TPU this compiles to a single static XLA graph with half the batch size
  of a full CFG pass.
  """
  wan_transformer = nnx.merge(graphdef, sharded_state, rest_of_state)
  noise_cond = wan_transformer(
      hidden_states=latents_cond,
      timestep=timestep_cond,
      encoder_hidden_states=prompt_cond_embeds,
      encoder_hidden_states_image=encoder_hidden_states_image,
      kv_cache=kv_cache,
      rotary_emb=rotary_emb,
      encoder_attention_mask=encoder_attention_mask,
  )

  # FFT over spatial dims (H, W) — last 2 dims of [B, C, F, H, W]
  fft_cond_cached = jnp.fft.rfft2(cached_noise_cond.astype(jnp.float32))
  fft_uncond_cached = jnp.fft.rfft2(cached_noise_uncond.astype(jnp.float32))
  fft_bias = fft_uncond_cached - fft_cond_cached

  # Build low/high frequency mask (25% cutoff)
  h = fft_bias.shape[-2]
  w_rfft = fft_bias.shape[-1]
  ch = jnp.maximum(1, h // 4)
  cw = jnp.maximum(1, w_rfft // 4)
  freq_h = jnp.arange(h)
  freq_w = jnp.arange(w_rfft)
  # Low-freq: indices near DC (0) in both dims; account for wrap-around in dim H
  low_h = (freq_h < ch) | (freq_h >= h - ch + 1)
  low_w = freq_w < cw
  low_mask = (low_h[:, None] & low_w[None, :]).astype(jnp.float32)
  high_mask = 1.0 - low_mask

  # Apply phase-dependent weights to frequency bias
  fft_bias_weighted = fft_bias * (low_mask * w1 + high_mask * w2)

  # Reconstruct unconditional output
  fft_cond_new = jnp.fft.rfft2(noise_cond.astype(jnp.float32))
  fft_uncond_approx = fft_cond_new + fft_bias_weighted
  noise_uncond_approx = jnp.fft.irfft2(fft_uncond_approx, s=noise_cond.shape[-2:]).astype(noise_cond.dtype)

  noise_pred_merged = noise_uncond_approx + guidance_scale * (noise_cond - noise_uncond_approx)
  return noise_pred_merged, noise_cond


def nearest_interp(src, target_len):
  """Nearest neighbor interpolation for ratio scaling layout."""
  src_len = len(src)
  if target_len == 1:
    return np.array([src[-1]])
  indices = np.round(np.linspace(0, src_len - 1, target_len)).astype(np.int32)
  return src[indices]


def init_magcache(num_inference_steps, retention_ratio, mag_ratios_base):
  """Initialize MagCache variables and interpolate ratios.

  Args:
      num_inference_steps: Number of inference steps.
      retention_ratio: Retention ratio of unchanged steps.
      mag_ratios_base: Base magnitude ratios array or list.
  """
  accumulated_ratio_cond = 1.0
  accumulated_ratio_uncond = 1.0
  accumulated_err_cond = 0.0
  accumulated_err_uncond = 0.0
  accumulated_steps_cond = 0
  accumulated_steps_uncond = 0
  cached_residual = None

  skip_warmup = int(num_inference_steps * retention_ratio)

  mag_ratios_base = np.array(mag_ratios_base)

  if len(mag_ratios_base) != num_inference_steps * 2:
    mag_cond = nearest_interp(mag_ratios_base[0::2], num_inference_steps)
    mag_uncond = nearest_interp(mag_ratios_base[1::2], num_inference_steps)
    mag_ratios = np.concatenate([mag_cond.reshape(-1, 1), mag_uncond.reshape(-1, 1)], axis=1).reshape(-1)
  else:
    mag_ratios = mag_ratios_base

  return (
      accumulated_ratio_cond,
      accumulated_ratio_uncond,
      accumulated_err_cond,
      accumulated_err_uncond,
      accumulated_steps_cond,
      accumulated_steps_uncond,
      cached_residual,
      skip_warmup,
      mag_ratios,
  )


def magcache_step(
    step,
    mag_ratios,
    accumulated_state,
    magcache_thresh,
    magcache_K,
    skip_warmup=0,
    use_magcache=None,
):
  """Update MagCache accumulated state and decide if to skip.

  Args:
      step: Current inference step.
      mag_ratios: Interpolated magnitude ratios array.
      accumulated_state: Tuple containing accumulated variables.
      magcache_thresh: Error threshold.
      magcache_K: Max skip steps.
      skip_warmup: Warmup steps threshold.
      use_magcache: Optional manual override boolean to enable/disable cache for
        this step.
  """

  (
      accumulated_ratio_cond,
      accumulated_ratio_uncond,
      accumulated_err_cond,
      accumulated_err_uncond,
      accumulated_steps_cond,
      accumulated_steps_uncond,
  ) = accumulated_state

  cur_mag_ratio_cond = mag_ratios[step * 2]
  cur_mag_ratio_uncond = mag_ratios[step * 2 + 1]

  if use_magcache is None:
    use_magcache = True
    if step < skip_warmup:
      use_magcache = False

  skip_blocks = False
  if use_magcache:
    new_ratio_cond = accumulated_ratio_cond * cur_mag_ratio_cond
    new_ratio_uncond = accumulated_ratio_uncond * cur_mag_ratio_uncond

    err_cond = np.abs(1.0 - new_ratio_cond)
    err_uncond = np.abs(1.0 - new_ratio_uncond)

    if (
        accumulated_err_cond + err_cond < magcache_thresh
        and accumulated_steps_cond < magcache_K
        and accumulated_err_uncond + err_uncond < magcache_thresh
        and accumulated_steps_uncond < magcache_K
    ):
      skip_blocks = True
      accumulated_ratio_cond = new_ratio_cond
      accumulated_ratio_uncond = new_ratio_uncond
      accumulated_err_cond += err_cond
      accumulated_err_uncond += err_uncond
      accumulated_steps_cond += 1
      accumulated_steps_uncond += 1
    else:
      accumulated_ratio_cond = 1.0
      accumulated_ratio_uncond = 1.0
      accumulated_err_cond = 0.0
      accumulated_err_uncond = 0.0
      accumulated_steps_cond = 0
      accumulated_steps_uncond = 0

  new_state = (
      accumulated_ratio_cond,
      accumulated_ratio_uncond,
      accumulated_err_cond,
      accumulated_err_uncond,
      accumulated_steps_cond,
      accumulated_steps_uncond,
  )
  return skip_blocks, new_state


class WanPipeline2_1(WanPipeline):
  """Pipeline for WAN 2.1 with a single transformer."""

  def __init__(self, config: DiffusionConfig, transformer: Optional[WanModel], **kwargs):
    super().__init__(config=config, **kwargs)
    self.transformer = transformer

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
    # Load VAE/tokenizer/text-encoder/scheduler in a background thread while
    # the main thread converts the 14B transformer: the small components are
    # hidden behind the transformer conversion time. The mesh/rngs built here are
    # deterministic duplicates of the ones _create_common_components builds
    # (same devices, same seed).
    common_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    common_future = common_executor.submit(
        cls._create_common_components,
        config,
        restored_checkpoint=restored_checkpoint,
        load_vae=load_vae,
        load_text_encoder=load_text_encoder,
        load_scheduler=load_scheduler,
    )
    transformer = None
    try:
      if load_transformer:
        mesh = create_wan_mesh(config)
        devices_array = mesh.devices
        rngs = nnx.Rngs(jax.random.key(config.seed))
        transformer = super().load_transformer(
            devices_array=devices_array,
            mesh=mesh,
            rngs=rngs,
            config=config,
            restored_checkpoint=restored_checkpoint,
            subfolder="transformer",
        )
      common_components = common_future.result()
    finally:
      common_executor.shutdown(wait=True)

    pipeline = cls(
        tokenizer=common_components["tokenizer"],
        text_encoder=common_components["text_encoder"],
        transformer=transformer,
        vae=common_components["vae"],
        vae_cache=common_components["vae_cache"],
        scheduler=common_components["scheduler"],
        scheduler_state=common_components["scheduler_state"],
        devices_array=common_components["devices_array"],
        mesh=common_components["mesh"],
        vae_mesh=common_components["vae_mesh"],
        config=config,
    )

    return pipeline, transformer

  def _get_num_channel_latents(self) -> int:
    return self.transformer.in_channels

  def __call__(
      self,
      prompt: Union[str, List[str]] = None,
      negative_prompt: Union[str, List[str]] = None,
      height: int = 480,
      width: int = 832,
      num_frames: int = 81,
      num_inference_steps: int = 50,
      guidance_scale: float = 5.0,
      num_videos_per_prompt: Optional[int] = 1,
      max_sequence_length: Optional[int] = None,
      latents: Optional[jax.Array] = None,
      prompt_embeds: Optional[jax.Array] = None,
      negative_prompt_embeds: Optional[jax.Array] = None,
      use_cfg_cache: bool = False,
      use_magcache: bool = False,
      magcache_thresh: Optional[float] = None,
      magcache_K: Optional[int] = None,
      retention_ratio: Optional[float] = None,
      use_kv_cache: bool = False,
      output_type: str = "np",
  ):
    config = getattr(self, "config", None)
    if output_type not in ["np", "latent"]:
      raise ValueError(f"output_type must be one of ['np', 'latent'], got {output_type}")

    if max_sequence_length is None:
      max_sequence_length = getattr(config, "max_sequence_length", 512)
    if magcache_thresh is None:
      magcache_thresh = getattr(config, "magcache_thresh", 0.12)
    if magcache_K is None:
      magcache_K = getattr(config, "magcache_K", 2)
    if retention_ratio is None:
      retention_ratio = getattr(config, "retention_ratio", 0.2)

    if use_cfg_cache and guidance_scale <= 1.0:
      raise ValueError(
          "use_cfg_cache=True requires guidance_scale > 1.0 (got"
          f" {guidance_scale}). CFG cache accelerates classifier-free guidance,"
          " which is disabled when guidance_scale <= 1.0."
      )
    trace = {}
    t_cond_start = time.perf_counter()

    (
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        scheduler_state,
        num_frames,
    ) = self._prepare_model_inputs(
        prompt,
        negative_prompt,
        height,
        width,
        num_frames,
        num_inference_steps,
        num_videos_per_prompt,
        max_sequence_length,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
    )
    latents.block_until_ready()
    prompt_embeds.block_until_ready()
    trace["conditioning"] = time.perf_counter() - t_cond_start

    graphdef, state, rest_of_state = nnx.split(self.transformer, nnx.Param, ...)

    p_run_inference = partial(
        run_inference_2_1,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        scheduler=self.scheduler,
        scheduler_state=scheduler_state,
        use_cfg_cache=use_cfg_cache,
        use_magcache=use_magcache,
        magcache_thresh=magcache_thresh,
        magcache_K=magcache_K,
        retention_ratio=retention_ratio,
        height=height,
        mag_ratios_base=getattr(config, "mag_ratios_base", None),
        config=self.config,
        use_kv_cache=use_kv_cache,
    )

    t_denoise_start = time.perf_counter()
    with jax.set_mesh(self.mesh):
      latents = p_run_inference(
          graphdef=graphdef,
          sharded_state=state,
          rest_of_state=rest_of_state,
          latents=latents,
          prompt_embeds=prompt_embeds,
          negative_prompt_embeds=negative_prompt_embeds,
      )
      latents = self._denormalize_latents(latents)
      latents.block_until_ready()
    trace["denoise_total"] = time.perf_counter() - t_denoise_start

    if output_type == "latent":
      return latents, trace

    t_decode_start = time.perf_counter()
    video = self._decode_latents_to_video(latents, trace=trace)
    if hasattr(video, "block_until_ready"):
      video.block_until_ready()
    trace["vae_decode"] = time.perf_counter() - t_decode_start

    return video, trace


def run_inference_2_1(
    graphdef,
    sharded_state,
    rest_of_state,
    latents: jnp.array,
    prompt_embeds: jnp.array,
    negative_prompt_embeds: jnp.array,
    guidance_scale: float,
    num_inference_steps: int,
    scheduler: FlaxUniPCMultistepScheduler,
    scheduler_state,
    use_cfg_cache: bool = False,
    use_magcache: bool = False,
    magcache_thresh: float = 0.12,
    magcache_K: int = 2,
    retention_ratio: float = 0.2,
    height: int = 480,
    mag_ratios_base: Optional[List[float]] = None,
    config=None,
    use_kv_cache: bool = False,
):
  """Denoising loop for WAN 2.1 T2V with FasterCache CFG-Cache.

  CFG-Cache strategy (Lv et al., ICLR 2025, enabled via use_cfg_cache=True):
  - Full CFG steps  : run transformer on [cond, uncond] batch (batch×2).
                      Cache raw noise_cond and noise_uncond for FFT bias.
  - Cache steps     : run transformer on cond batch only (batch×1).
                      Estimate uncond via FFT frequency-domain compensation:
                        ΔF = FFT(cached_uncond) - FFT(cached_cond)
                        Split ΔF into low-freq (ΔLF) and high-freq (ΔHF).
                        uncond_approx = IFFT(FFT(new_cond) + w1*ΔLF + w2*ΔHF)
                      Phase-dependent weights (α=0.2):
                        Early (high noise): w1=1.2, w2=1.0 (boost low-freq)
                        Late  (low noise):  w1=1.0, w2=1.2 (boost high-freq)
  - Schedule        : full CFG for the first 1/3 of steps, then
                      full CFG every 5 steps, cache the rest.

  Two separately-compiled JAX-jitted functions handle full and cache steps so
  XLA sees static shapes throughout — the key requirement for TPU efficiency.
  """
  do_cfg = guidance_scale > 1.0
  bsz = latents.shape[0]

  data_shards = 1
  try:
    if hasattr(latents, "sharding") and hasattr(latents.sharding, "mesh"):
      data_shards = latents.sharding.mesh.shape["data"] * latents.sharding.mesh.shape.get("fsdp", 1)
  except Exception:  # pylint: disable=broad-exception-caught
    # Unknown sharding layout: assume a single data shard.
    pass

  if use_cfg_cache and do_cfg and bsz % data_shards != 0:
    max_logging.log(
        f"Warning: Disabling CFG cache because batch size {bsz} is not"
        f" divisible by data shards {data_shards}. This often happens with"
        " data_parallelism > 1 and per_device_batch_size = 1."
    )
    use_cfg_cache = False
  # Resolution-dependent CFG cache config (FasterCache / MixCache guidance)
  if height >= 720:
    # 720p: conservative — protect last 40%, interval=5
    cfg_cache_interval = 5
    cfg_cache_start_step = int(num_inference_steps / 3)
    cfg_cache_end_step = int(num_inference_steps * 0.9)
    cfg_cache_alpha = 0.2
  else:
    # 480p: moderate — protect last 2 steps, interval=5
    cfg_cache_interval = 5
    cfg_cache_start_step = int(num_inference_steps / 3)
    cfg_cache_end_step = num_inference_steps - 2
    cfg_cache_alpha = 0.2

  # Pre-split embeds once, outside the loop.
  prompt_cond_embeds = prompt_embeds
  prompt_embeds_combined = None
  if do_cfg:
    prompt_embeds_combined = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)

  # Pre-compute cache schedule and phase-dependent weights.
  # t₀ = midpoint step; before t₀ boost low-freq, after boost high-freq.
  t0_step = num_inference_steps // 2
  first_full_step_seen = False
  step_is_cache = []
  step_w1w2 = []
  for s in range(num_inference_steps):
    is_cache = (
        use_cfg_cache
        and do_cfg
        and first_full_step_seen
        and cfg_cache_start_step <= s < cfg_cache_end_step
        and (s - cfg_cache_start_step) % cfg_cache_interval != 0
    )
    step_is_cache.append(is_cache)
    if not is_cache:
      first_full_step_seen = True
    # Phase-dependent weights: w = 1 + α·I(condition)
    if s < t0_step:
      step_w1w2.append((1.0 + cfg_cache_alpha, 1.0))  # early: boost low-freq
    else:
      step_w1w2.append((1.0, 1.0 + cfg_cache_alpha))  # late: boost high-freq

  # Cache tensors (on-device JAX arrays, initialised to None).
  cached_noise_cond = None
  cached_noise_uncond = None

  transformer_obj = nnx.merge(graphdef, sharded_state, rest_of_state)

  # Compute RoPE once as it only depends on shape
  dummy_hidden_states = jnp.zeros(
      (
          latents.shape[0],
          latents.shape[2],
          latents.shape[3],
          latents.shape[4],
          latents.shape[1],
      )
  )
  rotary_emb = transformer_obj.rope(dummy_hidden_states)

  kv_cache = None
  encoder_attention_mask = None

  if use_kv_cache:
    kv_cache, encoder_attention_mask = transformer_obj.compute_kv_cache(
        prompt_embeds_combined if do_cfg else prompt_cond_embeds
    )

  if use_magcache and do_cfg:
    magcache_init = init_magcache(num_inference_steps, retention_ratio, mag_ratios_base)
    accumulated_state = magcache_init[:6]
    cached_residual = magcache_init[6]
    skip_warmup = magcache_init[7]
    mag_ratios = magcache_init[8]

  first_profiling_step = config.skip_first_n_steps_for_profiler if config else 0
  profiler_steps = config.profiler_steps if config else 0
  last_profiling_step = np.clip(
      first_profiling_step + profiler_steps - 1,
      first_profiling_step,
      num_inference_steps - 1,
  )

  scan_diffusion_loop = getattr(config, "scan_diffusion_loop", False) if config else False
  timesteps = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)

  if scan_diffusion_loop and not use_magcache and not use_cfg_cache:
    scheduler_state = scheduler_state.replace(
        last_sample=jnp.zeros_like(latents),
        step_index=jnp.array(0, dtype=jnp.int32),
    )

    def scan_body(carry, t):
      current_latents, current_scheduler_state = carry

      if do_cfg:
        timestep = jnp.broadcast_to(t, bsz * 2)
        noise_pred = transformer_forward_pass(
            graphdef,
            sharded_state,
            rest_of_state,
            current_latents,
            timestep,
            prompt_embeds_combined,
            do_classifier_free_guidance=True,
            guidance_scale=guidance_scale,
            kv_cache=kv_cache,
            rotary_emb=rotary_emb,
            encoder_attention_mask=encoder_attention_mask,
        )
      else:
        timestep = jnp.broadcast_to(t, bsz)
        noise_pred = transformer_forward_pass(
            graphdef,
            sharded_state,
            rest_of_state,
            current_latents,
            timestep,
            prompt_cond_embeds,
            do_classifier_free_guidance=False,
            guidance_scale=guidance_scale,
            kv_cache=kv_cache,
            rotary_emb=rotary_emb,
            encoder_attention_mask=encoder_attention_mask,
        )

      new_latents, new_scheduler_state = scheduler.step(
          current_scheduler_state,
          noise_pred,
          t,
          current_latents,
          return_dict=False,
      )

      return (new_latents, new_scheduler_state), None

    initial_carry = (latents, scheduler_state)

    final_carry, _ = jax.lax.scan(scan_body, initial_carry, timesteps)

    final_latents, _ = final_carry
    return final_latents

  profiler = None
  for step in range(num_inference_steps):
    if config and profiler_enabled(config) and step == first_profiling_step:
      profiler = Profiler(config)
      profiler.start()

    t = timesteps[step]

    if use_magcache and do_cfg:
      timestep = jnp.broadcast_to(t, bsz * 2 if do_cfg else bsz)

      skip_blocks, accumulated_state = magcache_step(
          step,
          mag_ratios,
          accumulated_state,
          magcache_thresh,
          magcache_K,
          skip_warmup,
      )

      noise_pred, residual_x_cur = transformer_forward_pass(
          graphdef,
          sharded_state,
          rest_of_state,
          latents,
          timestep,
          prompt_embeds_combined if do_cfg else prompt_cond_embeds,
          do_classifier_free_guidance=do_cfg,
          guidance_scale=guidance_scale,
          skip_blocks=bool(skip_blocks),
          cached_residual=cached_residual,
          return_residual=True,
          kv_cache=kv_cache,
          rotary_emb=rotary_emb,
          encoder_attention_mask=encoder_attention_mask,
      )

      if not skip_blocks:
        cached_residual = residual_x_cur

    else:
      is_cache_step = step_is_cache[step]

      if is_cache_step:
        w1, w2 = step_w1w2[step]
        timestep = jnp.broadcast_to(t, bsz)
        kv_cache_cond = jax.tree.map(lambda x: x[:, :bsz], kv_cache) if kv_cache is not None else None
        encoder_attention_mask_cond = encoder_attention_mask[:bsz] if encoder_attention_mask is not None else None
        noise_pred, cached_noise_cond = transformer_forward_pass_cfg_cache(
            graphdef,
            sharded_state,
            rest_of_state,
            latents,
            timestep,
            prompt_cond_embeds,
            cached_noise_cond,
            cached_noise_uncond,
            guidance_scale=guidance_scale,
            w1=jnp.float32(w1),
            w2=jnp.float32(w2),
            kv_cache=kv_cache_cond,
            rotary_emb=rotary_emb,
            encoder_attention_mask=encoder_attention_mask_cond,
        )

      elif do_cfg:
        latents_doubled = jnp.concatenate([latents] * 2)
        timestep = jnp.broadcast_to(t, bsz * 2)
        (
            noise_pred,
            cached_noise_cond,
            cached_noise_uncond,
        ) = transformer_forward_pass_full_cfg(
            graphdef,
            sharded_state,
            rest_of_state,
            latents_doubled,
            timestep,
            prompt_embeds_combined,
            guidance_scale=guidance_scale,
            kv_cache=kv_cache,
            rotary_emb=rotary_emb,
            encoder_attention_mask=encoder_attention_mask,
        )

      else:
        timestep = jnp.broadcast_to(t, bsz)
        noise_pred = transformer_forward_pass(
            graphdef,
            sharded_state,
            rest_of_state,
            latents,
            timestep,
            prompt_cond_embeds,
            do_classifier_free_guidance=False,
            guidance_scale=guidance_scale,
            kv_cache=kv_cache,
            rotary_emb=rotary_emb,
            encoder_attention_mask=encoder_attention_mask,
        )

    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()

    if config and profiler_enabled(config) and step == last_profiling_step:
      if profiler:
        latents.block_until_ready()
        profiler.stop()

  return latents
