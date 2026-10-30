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

"""VAE image and video preprocessors and postprocessors for diffusion pipelines."""

import warnings
from typing import List, Optional, Tuple, Union

import jax
import numpy as np

import PIL.Image

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import CONFIG_NAME, PIL_INTERPOLATION, deprecate
from diffusers.utils import export_to_video as _diffusers_export_to_video

from maxtext.multimodal import utils as mm_utils


PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    jax.Array,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[jax.Array],
]

_ARRAY_TYPES = (np.ndarray, jax.Array)


def export_to_video(video_frames, output_video_path=None, fps=10):
  """dtype-aware wrapper around diffusers' `export_to_video`.

  diffusers always computes `(frame * 255).astype(np.uint8)` for numpy frames,
  which overflows (and color-inverts) frames that are already uint8, e.g. the
  output of the fused `vae_decode_pass`. uint8 frames are passed as PIL images,
  which diffusers writes without rescaling.
  """
  frames = np.asarray(video_frames)
  if frames.dtype == np.uint8:
    video_frames = [PIL.Image.fromarray(frame) for frame in frames]
  return _diffusers_export_to_video(video_frames, output_video_path, fps=fps)


def is_valid_image(image) -> bool:
  r"""
  Checks if the input is a valid image.

  A valid image can be:
  - A `PIL.Image.Image`.
  - A 2D or 3D `np.ndarray` or `jax.Array` (grayscale or color image).

  Args:
      image (`Union[PIL.Image.Image, np.ndarray, jax.Array]`):
          The image to validate. It can be a PIL image, a NumPy array, or a JAX array.

  Returns:
      `bool`:
          `True` if the input is a valid image, `False` otherwise.
  """
  return isinstance(image, PIL.Image.Image) or (isinstance(image, _ARRAY_TYPES) and image.ndim in (2, 3))


def is_valid_image_imagelist(images):
  r"""
  Checks if the input is a valid image or list of images.

  The input can be one of the following formats:
  - A 4D array (batch of images).
  - A valid single image: `PIL.Image.Image`, 2D array (grayscale image), 3D array.
  - A list of valid images.

  Args:
      images (`Union[np.ndarray, jax.Array, PIL.Image.Image, List]`):
          The image(s) to check. Can be a batch of images (4D array), a single image, or a list of valid
          images.

  Returns:
      `bool`:
          `True` if the input is valid, `False` otherwise.
  """
  if isinstance(images, _ARRAY_TYPES) and images.ndim == 4:
    return True
  elif is_valid_image(images):
    return True
  elif isinstance(images, list):
    return all(is_valid_image(image) for image in images)
  return False


class VaeImageProcessor(ConfigMixin):
  """
  Image processor for VAE.

  numpy port of diffusers' `VaeImageProcessor` (no torch), keeping diffusers' layouts: array inputs are channels-last
  (`[B, H, W, C]`) and `preprocess` returns channels-first `[B, C, H, W]` arrays in [-1, 1], the input layout of the
  Wan VAE encoder.

  Args:
      do_resize (`bool`, *optional*, defaults to `True`):
          Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`. Can accept
          `height` and `width` arguments from [`image_processor.VaeImageProcessor.preprocess`] method.
      vae_scale_factor (`int`, *optional*, defaults to `8`):
          VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
      spatial_patch_size (`Tuple[int, int]`, *optional*, defaults to `(1, 1)`):
          Additional height/width patch multiple used by models that require image sizes aligned to
          `vae_scale_factor * spatial_patch_size`.
      resample (`str`, *optional*, defaults to `lanczos`):
          Resampling filter to use when resizing the image.
      resize_mode (`str`, *optional*, defaults to `"default"`):
          Resize strategy for PIL images. `"default"` resizes directly to the target size. `"fill"` preserves aspect
          ratio and letterboxes the image to the target size.
      do_normalize (`bool`, *optional*, defaults to `True`):
          Whether to normalize the image to [-1,1].
      do_binarize (`bool`, *optional*, defaults to `False`):
          Whether to binarize the image to 0/1.
      do_convert_rgb (`bool`, *optional*, defaults to be `False`):
          Whether to convert the images to RGB format.
      do_convert_grayscale (`bool`, *optional*, defaults to be `False`):
          Whether to convert the images to grayscale format.
      fill_color (`Union[int, float, Tuple, str]`, *optional*, defaults to `0`):
          Fill color used when `resize_mode="fill"`.
  """

  config_name = CONFIG_NAME

  @register_to_config
  def __init__(
      self,
      do_resize: bool = True,
      vae_scale_factor: int = 8,
      spatial_patch_size: Tuple[int, int] = (1, 1),
      resample: str = "lanczos",
      resize_mode: str = "default",
      do_normalize: bool = True,
      do_binarize: bool = False,
      do_convert_rgb: bool = False,
      do_convert_grayscale: bool = False,
      fill_color: Union[int, float, Tuple, str] = 0,
  ):
    super().__init__()
    if do_convert_rgb and do_convert_grayscale:
      raise ValueError(
          "`do_convert_rgb` and `do_convert_grayscale` can not both be set to `True`,"
          " if you intended to convert the image into RGB format, please set `do_convert_grayscale = False`.",
          " if you intended to convert the image into grayscale format, please set `do_convert_rgb = False`",
      )
    if resize_mode not in {"default", "fill"}:
      raise ValueError(f"Unsupported resize_mode '{resize_mode}'. Expected one of: 'default', 'fill'.")

  @staticmethod
  def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
      images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
      # special case for grayscale (single channel) images
      pil_images = [PIL.Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
      pil_images = [PIL.Image.fromarray(image) for image in images]

    return pil_images

  @staticmethod
  def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
    """
    Convert a PIL image or a list of PIL images to NumPy arrays.
    """
    if not isinstance(images, list):
      images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    return images

  @staticmethod
  def nhwc_to_nchw(images: np.ndarray) -> np.ndarray:
    """
    Convert a channels-last image batch `[B, H, W, C]` (or `[B, H, W]`) to channels-first `[B, C, H, W]`.
    """
    if images.ndim == 3:
      images = images[..., None]

    return np.transpose(images, (0, 3, 1, 2))

  @staticmethod
  def nchw_to_nhwc(images: np.ndarray) -> np.ndarray:
    """
    Convert a channels-first image batch `[B, C, H, W]` to a channels-last float32 NumPy image `[B, H, W, C]`.
    """
    return np.transpose(np.asarray(images), (0, 2, 3, 1)).astype(np.float32)

  @staticmethod
  def normalize(images):
    """
    Normalize an image array to [-1,1].
    """
    return 2.0 * images - 1.0

  @staticmethod
  def denormalize(images):
    """
    Denormalize an image array to [0,1].
    """
    return np.clip(images / 2 + 0.5, 0, 1)

  @staticmethod
  def convert_to_grayscale(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Converts a PIL image to grayscale format.
    """
    image = image.convert("L")

    return image

  def get_default_height_width(
      self,
      image: Union[PIL.Image.Image, np.ndarray],
      height: Optional[int] = None,
      width: Optional[int] = None,
  ):
    """
    This function return the height and width that are downscaled to the next integer multiple of
    `vae_scale_factor`.

    Args:
        image(`PIL.Image.Image` or `np.ndarray`):
            The image input, can be a PIL image or a channels-first array of shape `[batch, channel, height, width]`
            (the layout `preprocess` converts arrays to before calling this).
        height (`int`, *optional*, defaults to `None`):
            The height in preprocessed image. If `None`, will use the height of `image` input.
        width (`int`, *optional*`, defaults to `None`):
            The width in preprocessed. If `None`, will use the width of the `image` input.
    """

    if height is None:
      height = image.height if isinstance(image, PIL.Image.Image) else image.shape[2]

    if width is None:
      width = image.width if isinstance(image, PIL.Image.Image) else image.shape[3]

    mod_h = self.config.vae_scale_factor * self.config.spatial_patch_size[0]
    mod_w = self.config.vae_scale_factor * self.config.spatial_patch_size[1]
    width, height = (x - x % mod for x, mod in ((width, mod_w), (height, mod_h)))

    return height, width

  def resize_and_fill(
      self,
      image: PIL.Image.Image,
      width: int,
      height: int,
  ) -> PIL.Image.Image:
    """
    Resize a PIL image to fit within the target size while preserving aspect ratio, then letterbox it.
    """
    target_ratio = width / height
    source_ratio = image.width / image.height

    resized_width = width if target_ratio < source_ratio else image.width * height // image.height
    resized_height = height if target_ratio >= source_ratio else image.height * width // image.width

    resized = image.resize((resized_width, resized_height), resample=PIL_INTERPOLATION[self.config.resample])
    canvas = PIL.Image.new("RGB", (width, height), color=self.config.fill_color)
    canvas.paste(
        resized,
        box=(width // 2 - resized_width // 2, height // 2 - resized_height // 2),
    )
    return canvas

  @staticmethod
  def _nearest_indices(in_size: int, out_size: int) -> np.ndarray:
    """Source indices of `torch.nn.functional.interpolate(mode="nearest")`: `min(floor(dst * scale), in - 1)`.

    Computed in float32 like torch (`scale = float(in) / out`) so the indices match bit for bit.
    """
    scale = np.float32(in_size) / np.float32(out_size)
    idx = np.floor(np.arange(out_size, dtype=np.float32) * scale).astype(np.int64)
    return np.minimum(idx, in_size - 1)

  @classmethod
  def _resize_nearest_nchw(cls, images: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize of `[B, C, H, W]`, same index rule as `torch.nn.functional.interpolate` (nearest)."""
    in_h, in_w = images.shape[-2:]
    rows = cls._nearest_indices(in_h, height)
    cols = cls._nearest_indices(in_w, width)
    return images[..., rows[:, None], cols[None, :]]

  def resize(
      self,
      image: Union[PIL.Image.Image, np.ndarray],
      height: Optional[int] = None,
      width: Optional[int] = None,
  ) -> Union[PIL.Image.Image, np.ndarray]:
    """
    Resize image. Arrays are channels-first `[B, C, H, W]` and resized with nearest-neighbor interpolation.
    """
    if isinstance(image, PIL.Image.Image):
      if self.config.resize_mode == "fill":
        image = self.resize_and_fill(image, width, height)
      else:
        image = image.resize((width, height), resample=PIL_INTERPOLATION[self.config.resample])
    elif isinstance(image, _ARRAY_TYPES):
      image = self._resize_nearest_nchw(np.asarray(image), height, width)
    return image

  def binarize(self, image: np.ndarray) -> np.ndarray:
    """
    create a mask
    """
    image[image < 0.5] = 0
    image[image >= 0.5] = 1
    return image

  def preprocess(
      self,
      image: PipelineImageInput,
      height: Optional[int] = None,
      width: Optional[int] = None,
  ) -> np.ndarray:
    """
    Preprocess the image input into a channels-first `[B, C, H, W]` array in [-1, 1].

    Accepted formats are PIL images and channels-last NumPy / JAX arrays (`[H, W, C]` or `[B, H, W, C]`), or lists
    of them.
    """
    supported_formats = (PIL.Image.Image,) + _ARRAY_TYPES

    # Expand the missing dimension for a 3-dimensional array that represents a grayscale image
    if self.config.do_convert_grayscale and isinstance(image, _ARRAY_TYPES) and image.ndim == 3:
      image = np.asarray(image)
      # the array could have 2 possible shapes:
      #   1. batch x height x width: insert channel dimension on last position
      #   2. height x width x channel: insert batch dimension on first position
      if image.shape[-1] == 1:
        image = np.expand_dims(image, axis=0)
      else:
        image = np.expand_dims(image, axis=-1)

    if isinstance(image, supported_formats):
      image = [image]
    elif not (isinstance(image, list) and all(isinstance(i, supported_formats) for i in image)):
      raise ValueError(
          f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support"
          f" {', '.join(f.__name__ for f in supported_formats)}"
      )

    if isinstance(image[0], PIL.Image.Image):
      if self.config.do_convert_rgb:
        image = [mm_utils.convert_to_RGB(i) for i in image]
      elif self.config.do_convert_grayscale:
        image = [self.convert_to_grayscale(i) for i in image]
      if self.config.do_resize:
        height, width = self.get_default_height_width(image[0], height, width)
        image = [self.resize(i, height, width) for i in image]
      image = self.pil_to_numpy(image)  # to np, channels-last
      image = self.nhwc_to_nchw(image)  # to channels-first

    else:
      image = [np.asarray(i) for i in image]
      image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)

      image = self.nhwc_to_nchw(image)

      height, width = self.get_default_height_width(image, height, width)
      if self.config.do_resize:
        image = self.resize(image, height, width)

    # expected range [0,1], normalize to [-1,1]
    do_normalize = self.config.do_normalize
    if image.min() < 0 and do_normalize:
      warnings.warn(
          "Passing `image` as an array with value range in [-1,1] is deprecated. The expected value range for"
          f" image arrays is [0,1]. You passed `image` with value range [{image.min()},{image.max()}]",
          FutureWarning,
      )
      do_normalize = False

    if do_normalize:
      image = self.normalize(image)

    if self.config.do_binarize:
      image = self.binarize(image)

    return image

  def postprocess(
      self,
      image: np.ndarray,
      output_type: str = "pil",
      do_denormalize: Optional[List[bool]] = None,
  ):
    """Convert a channels-first `[B, C, H, W]` image array in [-1, 1] to latent / np / PIL output."""
    if not isinstance(image, _ARRAY_TYPES):
      raise ValueError(
          f"Input for postprocessing is in incorrect format: {type(image)}. We only support NumPy / JAX arrays"
      )
    if output_type not in ["latent", "np", "pil"]:
      deprecation_message = (
          f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of"
          " these instead: `pil`, `np`, `latent`"
      )
      deprecate("Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False)
      output_type = "np"

    if output_type == "latent":
      return image

    image = np.asarray(image)
    if do_denormalize is None:
      do_denormalize = [self.config.do_normalize] * image.shape[0]

    image = np.stack([self.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])])

    image = self.nchw_to_nhwc(image)

    if output_type == "np":
      return image

    return self.numpy_to_pil(image)


class VaeVideoProcessor(VaeImageProcessor):
  r"""Simple video processor (numpy port of diffusers' `VideoProcessor`; videos are `[B, F, H, W, C]` in,
  `[B, C, F, H, W]` out)."""

  def preprocess_video(self, video, height: Optional[int] = None, width: Optional[int] = None) -> np.ndarray:
    r"""
    Preprocesses input video(s) into a channels-first `[B, C, F, H, W]` array in [-1, 1].

    Args:
        video (`List[PIL.Image]`, `List[List[PIL.Image]]`, `np.ndarray`, `List[np.ndarray]`):
            The input video. It can be one of the following:
            * List of the PIL images.
            * List of list of PIL images.
            * 4D NumPy / JAX arrays (expected shape for each array `(num_frames, height, width, num_channels)`).
            * List of 4D NumPy / JAX arrays (expected shape for each array `(num_frames, height, width,
              num_channels)`).
            * 5D NumPy / JAX arrays: expected shape for each array `(batch_size, num_frames, height, width,
              num_channels)`.
        height (`int`, *optional*, defaults to `None`):
            The height in preprocessed frames of the video. If `None`, will use the `get_default_height_width()` to
            get default height.
        width (`int`, *optional*`, defaults to `None`):
            The width in preprocessed frames of the video. If `None`, will use get_default_height_width()` to get
            the default width.
    """
    if isinstance(video, list) and isinstance(video[0], _ARRAY_TYPES) and video[0].ndim == 5:
      warnings.warn(
          "Passing `video` as a list of 5d arrays is deprecated."
          "Please concatenate the list along the batch dimension and pass it as a single 5d array",
          FutureWarning,
      )
      video = np.concatenate([np.asarray(v) for v in video], axis=0)

    # ensure the input is a list of videos:
    # - if it is a batch of videos (5d array), it is converted to a list of videos (a list of 4d arrays)
    # - if it is a single video, it is converted to a list of one video.
    if isinstance(video, _ARRAY_TYPES) and video.ndim == 5:
      video = list(np.asarray(video))
    elif (isinstance(video, list) and is_valid_image(video[0])) or is_valid_image_imagelist(video):
      video = [video]
    elif isinstance(video, list) and is_valid_image_imagelist(video[0]):
      pass  # Already a list of videos.
    else:
      raise ValueError(
          "Input is in incorrect format. Currently, we only support numpy.ndarray, jax.Array, PIL.Image.Image"
      )

    video = np.stack([self.preprocess(img, height=height, width=width) for img in video], axis=0)

    # move the number of channels before the number of frames.
    video = np.transpose(video, (0, 2, 1, 3, 4))

    return video

  def postprocess_video(self, video: np.ndarray, output_type: str = "np") -> Union[np.ndarray, List[PIL.Image.Image]]:
    r"""
    Converts a `[B, C, F, H, W]` video array to frames for export.

    Args:
        video (`np.ndarray`): The video as a channels-first array.
        output_type (`str`, defaults to `"np"`): Output type of the postprocessed `video` array.
    """
    video = np.asarray(video)
    batch_size = video.shape[0]
    outputs = []
    for batch_idx in range(batch_size):
      batch_vid = np.transpose(video[batch_idx], (1, 0, 2, 3))
      batch_output = self.postprocess(batch_vid, output_type)
      outputs.append(batch_output)

    if output_type == "np":
      outputs = np.stack(outputs)
    elif output_type != "pil":
      raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pil']")

    return outputs
