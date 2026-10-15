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

"""Model-agnostic diffusion pipeline utilities and base classes."""

from abc import abstractmethod
from typing import List, Tuple, Union

from diffusers.utils import BaseOutput
import flax
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from maxtext.multimodal.processor_vae import VaeImageProcessor

INDEX_FILE = "diffusion_flax_model.bin"


@flax.struct.dataclass
class ImagePipelineOutput(BaseOutput):
  """
  Output class for image pipelines.

  Args:
      images (`List[PIL.Image.Image]` or `np.ndarray`)
          List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
          num_channels)`.
  """

  images: Union[List[Image.Image], np.ndarray]


def _restored_component(restored_checkpoint, component: str, model_cls):
  """Returns (state, model_config) of a diffusers component from a restored checkpoint.

  Checkpoints from to_maxtext_diffusers.py store `<component>_state` / `<component>_config` items
  (e.g. `transformer_state`, `vae_state`). Older checkpoints only have a transformer, stored as
  `wan_state` / `model_state` (WAN 2.1) or `high_noise_transformer_state` (diffusers `transformer`) and
  `low_noise_transformer_state` (diffusers `transformer_2`) for WAN 2.2. The config is filtered to
  `model_cls.__init__` arguments, since older checkpoints stored the raw HF config.
  """
  from maxtext.checkpoint_conversion.to_maxtext_diffusers import model_init_config  # pylint: disable=import-outside-toplevel

  keys = set(restored_checkpoint.keys())
  legacy_states = {
      "transformer": ("wan_state", "model_state", "high_noise_transformer_state"),
      "transformer_2": ("low_noise_transformer_state",),
  }
  state_key = next(
      (k for k in (f"{component}_state", *legacy_states.get(component, ())) if k in keys),
      None,
  )
  if state_key is None:
    raise ValueError(
        f"Checkpoint has no `{component}_state` (items: {sorted(keys)}). Re-run"
        " checkpoint_conversion/to_maxtext_diffusers.py to convert all components."
    )
  config_key = next(
      (k for k in (f"{component}_config", "wan_config", "model_config") if k in keys),
      None,
  )
  model_config = dict(restored_checkpoint[config_key]) if config_key else {}
  return restored_checkpoint[state_key], model_init_config(model_cls, model_config)


def _to_int_path(path: tuple) -> tuple:
  """Restored checkpoints key NNX list indices as strings ("0"); NNX state uses ints."""
  return tuple(int(p) if isinstance(p, str) and p.isdigit() else p for p in path)


class DiffusionPipeline:
  r"""Base class for diffusion pipelines: checkpoint loading and shared helpers."""

  # Pipeline attributes holding the transformer(s) that `_load_and_init` returns after the pipeline object.
  _transformer_keys = ["transformer"]

  @staticmethod
  def numpy_to_pil(images):
    """
    Convert a NumPy image or a batch of images to a PIL image.
    """
    return VaeImageProcessor.numpy_to_pil(images)

  # TODO: make it compatible with jax.lax
  def progress_bar(self, iterable):
    if not hasattr(self, "_progress_bar_config"):
      self._progress_bar_config = {}
    elif not isinstance(self._progress_bar_config, dict):
      raise ValueError(f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}.")

    return tqdm(iterable, **self._progress_bar_config)

  def set_progress_bar_config(self, **kwargs):
    self._progress_bar_config = kwargs

  @classmethod
  @abstractmethod
  def _load_and_init(
      cls,
      config,
      restored_checkpoint=None,
      load_vae=True,
      load_text_encoder=True,
      load_transformer=True,
      load_scheduler=True,
  ):
    """Loads and initializes the pipeline components; returns `(pipeline, *transformers)`."""
    raise NotImplementedError

  @classmethod
  def quantize_transformer(cls, config, model, pipeline, mesh):  # pylint: disable=unused-argument
    """Hook for `from_checkpoint`: returns the (optionally quantized) transformer. No quantization by default."""
    return model

  @classmethod
  def _resolve_and_validate_load_flags(
      cls,
      vae_only=False,
      load_vae=None,
      load_text_encoder=None,
      load_transformer=None,
      load_scheduler=None,
  ) -> Tuple[bool, bool, bool, bool]:
    """Resolve (load_vae, load_text_encoder, load_transformer, load_scheduler), rejecting conflicts."""
    if vae_only:
      if load_vae is False:
        raise ValueError("Conflict: vae_only=True but load_vae=False")
      if load_text_encoder is True:
        raise ValueError("Conflict: vae_only=True but load_text_encoder=True")
      if load_transformer is True:
        raise ValueError("Conflict: vae_only=True but load_transformer=True")
      if load_scheduler is True:
        raise ValueError("Conflict: vae_only=True but load_scheduler=True")
      return True, False, False, False

    return (
        True if load_vae is None else load_vae,
        True if load_text_encoder is None else load_text_encoder,
        True if load_transformer is None else load_transformer,
        True if load_scheduler is None else load_scheduler,
    )

  @classmethod
  def from_checkpoint(
      cls,
      config,
      checkpoint_dir=None,
      restored_checkpoint=None,
      vae_only=False,
      load_vae=None,
      load_text_encoder=None,
      load_transformer=None,
      load_scheduler=None,
  ):
    """Build the pipeline from a converted Orbax checkpoint (restored here if not given).

    The checkpoint comes from `checkpoint_conversion/to_maxtext_diffusers.py`. The subclass's `_load_and_init`
    builds the components; each returned transformer then goes through `quantize_transformer`.
    """
    if restored_checkpoint is None:
      if checkpoint_dir is None:
        checkpoint_dir = getattr(config, "checkpoint_directory", None) or getattr(config, "checkpoint_dir", None)
      if not checkpoint_dir:
        raise ValueError(
            "checkpoint_directory is required: convert the model first with"
            " checkpoint_conversion/to_maxtext_diffusers.py."
        )
      from maxtext.checkpoint_conversion.to_maxtext_diffusers import restore_diffusers_checkpoint  # pylint: disable=import-outside-toplevel

      restored_checkpoint = restore_diffusers_checkpoint(checkpoint_dir)
      if restored_checkpoint is None:
        raise FileNotFoundError(f"Could not restore Orbax checkpoint from {checkpoint_dir}")

    (
        load_vae,
        load_text_encoder,
        load_transformer,
        load_scheduler,
    ) = cls._resolve_and_validate_load_flags(
        vae_only=vae_only,
        load_vae=load_vae,
        load_text_encoder=load_text_encoder,
        load_transformer=load_transformer,
        load_scheduler=load_scheduler,
    )
    outputs = cls._load_and_init(
        config,
        restored_checkpoint=restored_checkpoint,
        load_vae=load_vae,
        load_text_encoder=load_text_encoder,
        load_transformer=load_transformer,
        load_scheduler=load_scheduler,
    )
    pipeline = outputs[0]
    loaded_transformers = outputs[1:]

    for key, transformer in zip(cls._transformer_keys, loaded_transformers):
      quantized = cls.quantize_transformer(config, transformer, pipeline, pipeline.mesh)
      setattr(pipeline, key, quantized)

    return pipeline
