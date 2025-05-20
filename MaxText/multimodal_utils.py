"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Utils needed by multimodal pipelines for image processing."""

from typing import List, Union, Optional, Set, Tuple
import os

import numpy as np

from PIL import Image

import jax
import jax.numpy as jnp
import torch

from collections import defaultdict
import math

from dataclasses import dataclass


NUM_IMAGE_CHANNELS = 3

# Constants for Gemma3-specific processing
GEMMA_DEFAULT_IMAGE_SIZE = 896
GEMMA_IMAGE_MEAN = (127.5,) * 3
GEMMA_IMAGE_STD = (127.5,) * 3
GEMMA_BEGIN_IMAGE_TOKEN = 255999
GEMMA_END_IMAGE_TOKEN = 262144
GEMMA_NEW_LINE_TOKEN = 108
GEMMA_TOKEN_PLACEHOLDER = -2
# The number of GEMMA_TOKEN_PLACEHOLDER tokens per image in Gemma3
GEMMA_NUM_PLACEHOLDER_TOKENS_PER_IMAGE = 256
# +4 means 4 extra tokens to pad around image: \n\n, <start_of_image>, <end_of_image>, \n\n
# One MEDIA means one image or multiple images in one video, but now we only support one image
GEMMA_NUM_TOKENS_PER_MEDIA = GEMMA_NUM_PLACEHOLDER_TOKENS_PER_IMAGE + 4


def load_image_from_path(image_path):
  """Loads an image from a given file path and returns a jnp.array."""
  if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image not found at path {image_path}. Please specify a valid image path")
  try:
    image = Image.open(image_path).convert("RGB")
    image.load()  # Load image data to catch errors early
    return jnp.asarray(np.array(image))
  except (IOError, OSError) as e:
    raise IOError(f"Error loading image from {image_path}")


def _normalize_images(images, mean, std):
  """Normalize the image to zero mean and unit variance.
  Change the image mean and std based on parameters mean and std.
  Args:
    images: The images to normalize.
    mean: tuple[float, float, float].
    std: tuple[float, float, float].
  Returns:
    The normalized images.
  """
  images -= jnp.asarray(mean)
  images /= jnp.asarray(std)
  return images


def pre_process_gemma3_image(image):
  """Performs a bi-linear resize (with anti-aliasing) and normalizes the image."""
  image_shape = (GEMMA_DEFAULT_IMAGE_SIZE, GEMMA_DEFAULT_IMAGE_SIZE, NUM_IMAGE_CHANNELS)
  image = jax.image.resize(
      image,
      shape=image_shape,
      method="bilinear",
      antialias=True,
  )
  image = _normalize_images(image, mean=GEMMA_IMAGE_MEAN, std=GEMMA_IMAGE_STD)
  image = jnp.clip(image, -1, 1)
  return image


@dataclass
class SizeDict:
  """
  Hashable dictionary to store image size information.
  """

  height: Optional[int] = None
  width: Optional[int] = None
  longest_edge: Optional[int] = None
  shortest_edge: Optional[int] = None
  max_height: Optional[int] = None
  max_width: Optional[int] = None

  def __getitem__(self, key):
    if hasattr(self, key):
      return getattr(self, key)
    raise KeyError(f"Key {key} not found in SizeDict.")


class TensorType:
  """
  Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
  tab-completion in an IDE.
  """

  PYTORCH = "pt"
  TENSORFLOW = "tf"
  NUMPY = "np"
  JAX = "jax"
  MLX = "mlx"


def get_factors(dividend: int) -> Set[int]:
  """
  Calculate all factors of a given number, i.e. a divisor that leaves
  no remainder. For example, if dividend=12, it will return {1, 2, 3, 4, 6, 12}.

  Args:
      dividend (int): The number to find factors for.

  Returns:
      set: A set containing all factors of the number.
  """
  factors_set = set()

  for i in range(1, int(dividend**0.5) + 1):
    if dividend % i == 0:
      factors_set.add(i)
      factors_set.add(dividend // i)
  return factors_set


def find_supported_resolutions(max_num_chunks: int, patch_size: SizeDict) -> torch.Tensor:
  """
  Computes all of the allowed resolutions for a fixed number of chunks
  and patch_size. Useful for when dividing an image into chunks.

  Args:
      max_num_chunks (int): Maximum number of chunks for processing.
      patch_size (int): Size of the side of the patch.

  Returns:
      torch.Tensor: List of possible resolutions as tuples (height, width).

  Example:
      >>> max_num_chunks = 5
      >>> patch_size = 224
      >>> find_supported_resolutions(max_num_chunks, patch_size)
      tensor([(224, 896), (448, 448), (224, 224), (896, 224), (224, 672),
      (672, 224), (224, 448), (448, 224)])

      Given max_num_chunks=4, patch_size=224, it will create a dictionary:
      {
      0.25: [(1, 4)],
      1.0: [(2, 2), (1, 1)],
      4.0: [(4, 1)],
      0.33: [(1, 3)],
      3.0: [(3, 1)],
      0.5: [(1, 2)],
      2.0: [(2, 1)]
      }

      and return the resolutions multiplied by the patch_size:
      [(1*224, 4*224), (2*224, 2*224), ..., (2*224, 1*224)]
  """
  height, width = patch_size.height, patch_size.width
  if height != width:
    raise ValueError("`size` must be square.")

  patch_size = height

  asp_dict = defaultdict(list)
  for chunk_size in range(max_num_chunks, 0, -1):
    _factors = sorted(get_factors(chunk_size))
    _asp_ratios = [(factor, chunk_size // factor) for factor in _factors]
    for height, width in _asp_ratios:
      ratio_float = height / width
      asp_dict[ratio_float].append((height, width))

  # get the resolutions multiplied by the patch_size
  possible_resolutions = []
  for key, value in asp_dict.items():
    for height, depth in value:
      possible_resolutions.append((height * patch_size, depth * patch_size))

  return possible_resolutions


def group_images_by_shape(
    images: list["torch.Tensor"],
) -> tuple[dict[tuple[int, int], list["torch.Tensor"]], dict[int, tuple[tuple[int, int], int]]]:
  """
  Groups images by shape.
  Returns a dictionary with the shape as key and a list of images with that shape as value,
  and a dictionary with the index of the image in the original list as key and the shape and index in the grouped list as value.
  """
  grouped_images = {}
  grouped_images_index = {}
  for i, image in enumerate(images):
    shape = image.shape[1:]
    if shape not in grouped_images:
      grouped_images[shape] = []
    grouped_images[shape].append(image)
    grouped_images_index[i] = (shape, len(grouped_images[shape]) - 1)
  # stack images with the same shape
  grouped_images = {shape: torch.stack(images, dim=0) for shape, images in grouped_images.items()}
  return grouped_images, grouped_images_index


def get_best_fit(
    image_size: Tuple[int, int],
    possible_resolutions: torch.Tensor,
    resize_to_max_canvas: bool = False,
) -> Tuple[int, int]:
  """
  Determines the best canvas possible from a list of possible resolutions to, without distortion,
  resize an image to.

  For each possible resolution, calculates the scaling factors for
  width and height, and selects the smallest one, which is the limiting side.
  E.g. to match the canvas you can upscale height by 2x, and width by 1.5x,
  therefore, the maximum upscaling you can do is min(2, 1.5) = 1.5.

  If upscaling is possible (any of the scaling factors is greater than 1),
  then picks the smallest upscaling factor > 1, unless resize_to_max_canvas is True.

  If upscaling is not possible, then picks the largest scaling factor <= 1, i.e.
  reduce downscaling as much as possible.

  If there are multiple resolutions with the same max scale, we pick the one with the lowest area,
  to minimize padding. E.g., the same image can be upscaled to 224x224 and 224x448, but the latter
  has more padding.

  Args:
      image_size (Tuple[int, int]): A tuple containing the height and width of the image.
      possible_resolutions (torch.Tensor): A tensor of shape (N, 2) where each
          row represents a possible resolution (height, width).
      resize_to_max_canvas (bool): If True, will return the largest upscaling resolution.

  Returns:
      List[int]: The best resolution [height, width] for the given image.

  Example:
      >>> image_size = (200, 300)
      >>> possible_resolutions = torch.tensor([[224, 672],
      ...                                     [672, 224],
      ...                                     [224, 448],
      ...                                     [448, 224],
      ...                                     [224, 224]])
      >>> get_best_fit(image_size, possible_resolutions)
      [224, 448]

      We have:
          scale_w = tensor([2.2400, 0.7467, 1.4933, 0.7467, 0.7467])
          scale_h = tensor([1.1200, 3.3600, 1.1200, 2.2400, 1.1200])
          scales = tensor([1.1200, 0.7467, 1.1200, 0.7467, 0.7467])
      Only one of the scales > 1:
          upscaling_possible = tensor([1.1200, 1.1200])
          smallest_rescale = tensor(1.1200)
      So we pick the resolution with the smallest smallest area:
          areas = tensor([150528, 100352]) # [672, 224], [224, 448]
          optimal_canvas = tensor([224, 448])
  """

  original_height, original_width = image_size

  # get all possible resolutions heights/widths
  target_heights, target_widths = (
      possible_resolutions[:, 0],
      possible_resolutions[:, 1],
  )

  # get scaling factors to resize the image without distortion
  scale_w = target_widths / original_width
  scale_h = target_heights / original_height

  # get the min scale between width and height (limiting side -> no distortion)
  scales = torch.where(scale_h > scale_w, scale_w, scale_h)

  # filter only scales that allow upscaling
  upscaling_options = scales[scales >= 1]
  if len(upscaling_options) > 0:
    if resize_to_max_canvas:
      selected_scale = torch.max(upscaling_options)
    else:
      selected_scale = torch.min(upscaling_options)
  else:
    # no upscaling possible,
    # get the minimum downscaling (max scale for scales<1)
    downscaling_options = scales[scales < 1]
    selected_scale = torch.max(downscaling_options)

  # get all resolutions that support this scaling factor,
  # e.g. you can upscale to 224x224, 224x448, 224x672 without distortion
  chosen_canvas = possible_resolutions[scales == selected_scale]

  # if there are multiple resolutions,
  # get the one with minimum area to reduce padding
  if len(chosen_canvas) > 1:
    areas = chosen_canvas[:, 0] * chosen_canvas[:, 1]
    optimal_idx = torch.argmin(areas)
    optimal_canvas = chosen_canvas[optimal_idx]
  else:
    optimal_canvas = chosen_canvas[0]

  return tuple(optimal_canvas.tolist())


def get_max_res_without_distortion(
    image_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Determines the maximum resolution to which an image can be resized to without distorting its
    aspect ratio, based on the target resolution.

    Args:
        image_size (Tuple[int, int]): The original resolution of the image (height, width).
        target_resolution (Tuple[int, int]): The desired resolution to fit the image into (height, width).
    Returns:
        Tuple[int, int]: The optimal dimensions (height, width) to which the image should be resized.
    Example:
        >>> _get_max_res_without_distortion([200, 300], target_size = [450, 200])
        (134, 200)
        >>> _get_max_res_without_distortion([800, 600], target_size = [450, 1300])
        (450, 338)
    """

    original_height, original_width = image_size
    target_height, target_width = target_size

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.floor(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.floor(original_width * scale_h), target_width)

    return new_height, new_width


def pad_to_best_fit(
    images: "torch.Tensor",
    target_size: Tuple[int, int],
    background_color: Union[int, Tuple[int, int, int]] = 0,
) -> "torch.Tensor":
    """
    Pads an image to fit the target size.

    Args:
        images (`np.ndarray`):
            The images to pad.
        background_color (`int` or `Tuple[int, int, int]`, *optional*, defaults to 0):
            The color to use for the padding. Can be an integer for single channel or a
            tuple of integers representing for multi-channel images. If passed as integer
            in mutli-channel mode, it will default to `0` in subsequent channels.
    Returns:
        `torch.Tensor`: The padded images.
    """

    num_channels = images.shape[1] if len(images.shape) == 4 else images.shape[0]
    if isinstance(background_color, int):
        background_color = [background_color] + [0] * (num_channels - 1)
    elif len(background_color) != num_channels:
        raise ValueError(
            f"background_color must have no more than {num_channels} elements to match the number of channels"
        )

    height, width = images.shape[-2:]
    target_height, target_width = target_size
    paste_x_right = target_width - width
    paste_y_right = target_height - height
    padded_images = F.pad(images, padding=[0, 0, paste_x_right, paste_y_right], fill=background_color)

    return padded_images


def split_to_tiles(images: torch.Tensor, num_tiles_height: int, num_tiles_width: int) -> torch.Tensor:
    # Split image into number of required tiles (width x height)
    batch_size, num_channels, height, width = images.size()
    images = images.view(
        batch_size,
        num_channels,
        num_tiles_height,
        height // num_tiles_height,
        num_tiles_width,
        width // num_tiles_width,
    )
    # Permute dimensions to reorder the axes
    image = images.permute(0, 2, 4, 1, 3, 5).contiguous()
    # Reshape into the desired output shape (batch_size * 4, num_channels, width/2, height/2)
    image = image.view(
        batch_size,
        num_tiles_width * num_tiles_height,
        num_channels,
        height // num_tiles_height,
        width // num_tiles_width,
    )
    return image


def reorder_images(
    processed_images: dict[tuple[int, int], "torch.Tensor"], grouped_images_index: dict[int, tuple[int, int]]
) -> list["torch.Tensor"]:
    """
    Reconstructs a list of images in the original order.
    """
    return [
        processed_images[grouped_images_index[i][0]][grouped_images_index[i][1]]
        for i in range(len(grouped_images_index))
    ]


def get_size_with_aspect_ratio(image_size, size, max_size=None) -> tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size.

    Args:
        image_size (`Tuple[int, int]`):
            The input image size.
        size (`int`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
    """
    height, width = image_size
    raw_size = None
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            raw_size = max_size * min_original_size / max_original_size
            size = int(round(raw_size))

    if (height <= width and height == size) or (width <= height and width == size):
        oh, ow = height, width
    elif width < height:
        ow = size
        if max_size is not None and raw_size is not None:
            oh = int(raw_size * height / width)
        else:
            oh = int(size * height / width)
    else:
        oh = size
        if max_size is not None and raw_size is not None:
            ow = int(raw_size * width / height)
        else:
            ow = int(size * width / height)

    return (oh, ow)


# Logic adapted from torchvision resizing logic: https://github.com/pytorch/vision/blob/511924c1ced4ce0461197e5caa64ce5b9e558aab/torchvision/transforms/functional.py#L366
def get_resize_output_image_size(
    input_image: np.ndarray,
    size: Union[int, tuple[int, int], list[int], tuple[int]],
    default_to_square: bool = True,
    max_size: Optional[int] = None,
    input_data_format: Optional[Union[str, str]] = None,
) -> tuple:
    """
    Find the target (height, width) dimension of the output image after resizing given the input image and the desired
    size.

    Args:
        input_image (`np.ndarray`):
            The image to resize.
        size (`int` or `Tuple[int, int]` or List[int] or `Tuple[int]`):
            The size to use for resizing the image. If `size` is a sequence like (h, w), output size will be matched to
            this.

            If `size` is an int and `default_to_square` is `True`, then image will be resized to (size, size). If
            `size` is an int and `default_to_square` is `False`, then smaller edge of the image will be matched to this
            number. i.e, if height > width, then image will be rescaled to (size * height / width, size).
        default_to_square (`bool`, *optional*, defaults to `True`):
            How to convert `size` when it is a single int. If set to `True`, the `size` will be converted to a square
            (`size`,`size`). If set to `False`, will replicate
            [`torchvision.transforms.Resize`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)
            with support for resizing only the smallest edge and providing an optional `max_size`.
        max_size (`int`, *optional*):
            The maximum allowed for the longer edge of the resized image: if the longer edge of the image is greater
            than `max_size` after being resized according to `size`, then the image is resized again so that the longer
            edge is equal to `max_size`. As a result, `size` might be overruled, i.e the smaller edge may be shorter
            than `size`. Only used if `default_to_square` is `False`.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `tuple`: The target (height, width) dimension of the output image after resizing.
    """
    if isinstance(size, (tuple, list)):
        if len(size) == 2:
            return tuple(size)
        elif len(size) == 1:
            # Perform same logic as if size was an int
            size = size[0]
        else:
            raise ValueError("size must have 1 or 2 elements if it is a list or tuple")

    if default_to_square:
        return (size, size)

    height, width = get_image_size(input_image, input_data_format)
    short, long = (width, height) if width <= height else (height, width)
    requested_new_short = size

    new_short, new_long = requested_new_short, int(requested_new_short * long / short)

    if max_size is not None:
        if max_size <= requested_new_short:
            raise ValueError(
                f"max_size = {max_size} must be strictly greater than the requested "
                f"size for the smaller edge size = {size}"
            )
        if new_long > max_size:
            new_short, new_long = int(max_size * new_short / new_long), max_size

    return (new_long, new_short) if width <= height else (new_short, new_long)


def get_image_size_for_max_height_width(
    image_size: tuple[int, int],
    max_height: int,
    max_width: int,
) -> tuple[int, int]:
    """
    Computes the output image size given the input image and the maximum allowed height and width. Keep aspect ratio.
    Important, even if image_height < max_height and image_width < max_width, the image will be resized
    to at least one of the edges be equal to max_height or max_width.

    For example:
        - input_size: (100, 200), max_height: 50, max_width: 50 -> output_size: (25, 50)
        - input_size: (100, 200), max_height: 200, max_width: 500 -> output_size: (200, 400)

    Args:
        image_size (`Tuple[int, int]`):
            The image to resize.
        max_height (`int`):
            The maximum allowed height.
        max_width (`int`):
            The maximum allowed width.
    """
    height, width = image_size
    height_scale = max_height / height
    width_scale = max_width / width
    min_scale = min(height_scale, width_scale)
    new_height = int(height * min_scale)
    new_width = int(width * min_scale)
    return new_height, new_width


def f_resize(
    img: Tensor,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[bool] = True,
) -> Tensor:
    r"""Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        img (PIL Image or Tensor): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.NEAREST_EXACT``, ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are
            supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image. If the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``,
            ``size`` will be overruled so that the longer edge is equal to
            ``max_size``.
            As a result, the smaller edge may be shorter than ``size``. This
            is only supported if ``size`` is an int (or a sequence of length
            1 in torchscript mode).
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True`` (default): will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The default value changed from ``None`` to ``True`` in
            v0.17, for the PIL and Tensor backends to be consistent.

    Returns:
        PIL Image or Tensor: Resized image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(resize)

    if isinstance(interpolation, int):
        interpolation = _interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if isinstance(size, (list, tuple)):
        if len(size) not in [1, 2]:
            raise ValueError(
                f"Size must be an int or a 1 or 2 element tuple/list, not a {len(size)} element tuple/list"
            )
        if max_size is not None and len(size) != 1:
            raise ValueError(
                "max_size should only be passed if size specifies the length of the smaller edge, "
                "i.e. size should be an int or a sequence of length 1 in torchscript mode."
            )

    _, image_height, image_width = get_dimensions(img)
    if isinstance(size, int):
        size = [size]
    output_size = _compute_resized_output_size((image_height, image_width), size, max_size)

    if [image_height, image_width] == output_size:
        return img

    if not isinstance(img, torch.Tensor):
        if antialias is False:
            warnings.warn("Anti-alias option is always applied for PIL Image input. Argument antialias is ignored.")
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil.resize(img, size=output_size, interpolation=pil_interpolation)

    return F_t.resize(img, size=output_size, interpolation=interpolation.value, antialias=antialias)


class Llama4ImageProcessorFast:
  image_mean = [0.5, 0.5, 0.5]
  image_std = [0.5, 0.5, 0.5]
  size = SizeDict()
  size.height, size.width = 336, 336
  do_resize = True
  do_rescale = True
  do_normalize = True
  do_convert_rgb = True
  max_patches = 16
  resize_to_max_canvas = False

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def rescale_and_normalize(
      self,
      images: List["torch.Tensor"],
      do_rescale: bool,
      rescale_factor: float,
      do_normalize: bool,
      image_mean: Union[float, List[float]],
      image_std: Union[float, List[float]],
  ) -> "torch.Tensor":
    """
    Rescale and normalize images.
    Override to rescale and normalize the images in torch.bfloat16 as in the original implementation
    """
    if do_rescale and do_normalize:
      images = images.to(dtype=torch.bfloat16) * rescale_factor
      images = self.normalize(images, image_mean, image_std)
    elif do_rescale:
      images = images * rescale_factor
    elif do_normalize:
      images = self.normalize(images, image_mean, image_std)

    return images
  
  def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        if size.shortest_edge and size.longest_edge:
            # Resize the image so that the shortest edge or the longest edge is of the given size
            # while maintaining the aspect ratio of the original image.
            new_size = get_size_with_aspect_ratio(
                image.size()[-2:],
                size.shortest_edge,
                size.longest_edge,
            )
        elif size.shortest_edge:
            new_size = get_resize_output_image_size(
                image,
                size=size.shortest_edge,
                default_to_square=False,
                # input_data_format=ChannelDimension.FIRST,
                input_data_format="channels_first",
            )
        elif size.max_height and size.max_width:
            new_size = get_image_size_for_max_height_width(image.size()[-2:], size.max_height, size.max_width)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys, or 'max_height' and 'max_width', or 'shortest_edge' key. Got"
                f" {size}."
            )
        return F.resize(image, new_size, interpolation=interpolation, antialias=antialias)

  def preprocess(
      self,
      images: List["torch.Tensor"],
      # resize_to_max_canvas: bool,
      # interpolation: Optional["F.InterpolationMode"],
      # do_rescale: bool,
      # rescale_factor: float,
      # do_normalize: bool,
      # image_mean: Optional[Union[float, List[float]]],
      # image_std: Optional[Union[float, List[float]]],
      # return_tensors: Optional[Union[str, TensorType]],
      # **kwargs,
  ):
    "A new doc string for demonstration"
    interpolation = "bilinear"
    do_rescale = True
    rescale_factor = 1 / 255.0
    do_normalize = True
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.5, 0.5, 0.5)
    return_tensors = "pt"
    images_np = np.array(images).transpose(2, 0, 1)
    images = [torch.from_numpy(images_np)]
    possible_resolutions = find_supported_resolutions(max_num_chunks=self.max_patches, patch_size=self.size)
    possible_resolutions = torch.tensor(possible_resolutions)
    # process images by batch, grouped by shape
    grouped_images, grouped_images_index = group_images_by_shape(images)
    grouped_processed_images = {}
    grouped_aspect_ratios = {}
    for shape, stacked_images in grouped_images.items():
      image_size = stacked_images.shape[-2:]
      target_size = get_best_fit(image_size, possible_resolutions, resize_to_max_canvas=self.resize_to_max_canvas)
      # If target_size requires upscaling, we might want to limit the upscaling to max_upscaling_size
      max_upscaling_size = None if self.resize_to_max_canvas else self.size.height
      if max_upscaling_size is not None:
        new_target_height = min(max(image_size[0], max_upscaling_size), target_size[0])
        new_target_width = min(max(image_size[1], max_upscaling_size), target_size[1])
        target_size_without_distortion = (new_target_height, new_target_width)

      # resize to target_size while preserving aspect ratio
      new_size_without_distortion = get_max_res_without_distortion(image_size, target_size_without_distortion)
      new_size_without_distortion = SizeDict(
          height=max(new_size_without_distortion[0], 1), width=max(new_size_without_distortion[1], 1)
      )

      processed_images = self.resize(
          stacked_images,
          new_size_without_distortion,
          interpolation=interpolation,
      )

      # pad to target_size to be able to split into tiles
      processed_images = pad_to_best_fit(processed_images, target_size)
      processed_images = self.rescale_and_normalize(
          processed_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
      )

      ratio_h, ratio_w = (
          target_size[0] // self.size.height,
          target_size[1] // self.size.height,
      )
      # split into tiles
      processed_images = split_to_tiles(processed_images, ratio_h, ratio_w)
      grouped_processed_images[shape] = processed_images
      grouped_aspect_ratios[shape] = torch.tensor([[ratio_h, ratio_w]] * stacked_images.shape[0])

      # add a global tile to the processed tile if there are more than one tile
      if ratio_h * ratio_w > 1:
        global_tiles = self.resize(
            stacked_images,
            self.size,
            interpolation=interpolation,
        )
        global_tiles = self.rescale_and_normalize(
            global_tiles, do_rescale, rescale_factor, do_normalize, image_mean, image_std
        )
        grouped_processed_images[shape] = torch.cat([processed_images, global_tiles.unsqueeze(1)], dim=1)
    processed_images = reorder_images(grouped_processed_images, grouped_images_index)
    aspect_ratios_list = reorder_images(grouped_aspect_ratios, grouped_images_index)

    processed_images = torch.cat(processed_images, dim=0) if return_tensors else processed_images
    aspect_ratios = torch.stack(aspect_ratios_list, dim=0) if return_tensors else aspect_ratios_list
    # return BatchFeature(data={"pixel_values": processed_images, "aspect_ratios": aspect_ratios}, tensor_type=return_tensors)
    return processed_images


def pre_process_llama4_image(image):
  processor = Llama4ImageProcessorFast()
  image = processor.preprocess(image)
  return image


def pre_process_image(image, model_name):
  """Pre-process image according to different model's requirements.
  Args:
    image: The jnp.array image [H, W, C] to pre-process.
    model_name: The config.model_name that specifies the image preprocess ways.
  Returns:
    The pre-processed image in jnp.array [H, W, C].
  """
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    return pre_process_gemma3_image(image)
  elif model_name in ["llama4-17b-16e"]:
    return pre_process_llama4_image(image)
  else:
    raise ValueError(f"Model {model_name} does not support multimodal inference.")


def reformat_prompt(prompt, model_name):
  """Reformat prompt for different models."""
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    return formatted_prompt
  else:
    return prompt


def get_image_offsets(model_name):
  """Get the increase in total token count after inserting image token placeholders"""
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    return GEMMA_NUM_TOKENS_PER_MEDIA - 1  # -1 because <start_of_image> is already present in the input tokens.
  else:
    return 0


def prepare_text_for_image_fusion(texts, model_name):
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    return add_extra_tokens_for_images_gemma3(texts)
  else:
    raise ValueError(f"Model {model_name} does not support multimodal inference.")


def add_extra_tokens_for_images_gemma3(
    tokens: np.ndarray | jnp.ndarray,
    *,
    max_num_images: int = 1,
):  # -> Int['B L+(max_num_images * (num_tokens_per_image + 3))']:
  r"""Add the extra image tokens to the text tokens.

  If the model has images, we expand each `<start_of_image>` token by the image
  placeholder tokens.

  Example:

  ```python
  input = [..., x, <start_of_image>, y, ...]
  output = [
      ..., x, \n\n, <start_of_image>, SOFT_TOKEN_PLACEHOLDER,
      SOFT_TOKEN_PLACEHOLDER, ..., SOFT_TOKEN_PLACEHOLDER,
      SOFT_TOKEN_PLACEHOLDER, <end_of_image>, \n\n, y, ...
  ]
  ```

  The `\n\n` tokens are added to match how the model was trained.

  Args:
    tokens: The text tokens.
    max_num_images: The maximum number of images in the batch.
    num_tokens_per_image: The number of soft tokens per image.

  Returns:
    The text tokens with the extra image tokens.
  """

  # New tokens which will be inserted for each image.
  mm_tokens = [
      GEMMA_NEW_LINE_TOKEN,
      GEMMA_BEGIN_IMAGE_TOKEN,
      *[GEMMA_TOKEN_PLACEHOLDER] * GEMMA_NUM_PLACEHOLDER_TOKENS_PER_IMAGE,
      GEMMA_END_IMAGE_TOKEN,
      GEMMA_NEW_LINE_TOKEN,
  ]

  return insert_sequence(
      at=GEMMA_BEGIN_IMAGE_TOKEN,
      sequence=mm_tokens,
      tokens=tokens,
      max_num_images=max_num_images,
  )


def insert_sequence(
    tokens: np.ndarray | jnp.ndarray,
    *,
    at: int,
    sequence: List[int],
    max_num_images: int,
) -> np.ndarray | jnp.ndarray:
  """Insert a sequence of tokens at a given position."""
  tokens_dim = len(tokens.shape)
  if tokens_dim == 1:
    tokens = tokens[None, :]
  _, length = tokens.shape

  mm_tokens = jnp.array(sequence, dtype=jnp.int32)

  # `-1` because `<start_of_image>` is already present in the input tokens.
  offset_by = len(mm_tokens) - 1

  # Maximum length, if all images are present.
  length_with_mm = length + max_num_images * offset_by

  mm_start = tokens == at

  # Get the text tokens correctly placed at their final position.
  # The `<start_of_image>` are removed and expanded to leave space for the MM
  # tokens.
  # tokens = [..., x, <start_of_image>, y, ...]
  # new_text_tokens = [..., x, 0, 0, 0, ..., 0, 0, 0, y, ...]
  new_text_tokens = _get_new_text_tokens(
      mm_start=mm_start,
      text_tokens=tokens,
      offset_by=offset_by,
      length_with_mm=length_with_mm,
  )

  # Get the mm tokens placeholders, correctly placed at their final position.
  # new_mm_tokens = [
  #     ..., 0, 0, \n\n, <start_of_image>, ..., <end_of_image>, \n\n, 0, 0, ...
  # ]
  new_mm_tokens = _get_new_mm_tokens(
      mm_start=mm_start,
      mm_tokens_to_insert=mm_tokens,
      max_num_images=max_num_images,
      offset_by=offset_by,
      length_with_mm=length_with_mm,
  )

  # Merge the text and MM tokens.
  new_tokens = new_text_tokens + new_mm_tokens
  if tokens_dim < len(new_tokens.shape):
    new_tokens = jnp.squeeze(new_tokens)
  return new_tokens


def _get_new_text_tokens(
    *,
    mm_start: np.ndarray | jnp.ndarray,
    text_tokens: np.ndarray | jnp.ndarray,
    offset_by: int,
    length_with_mm: int,
) -> np.ndarray | jnp.ndarray:
  """Get new text tokens."""
  # Jax vmap does not support positional arguments, so need the
  # _get_new_text_tokens_inner indirection.
  return jax.vmap(_get_new_text_tokens_inner, in_axes=(0, 0, None, None))(mm_start, text_tokens, offset_by, length_with_mm)


def _get_new_text_tokens_inner(
    mm_start: np.ndarray | jnp.ndarray,
    text_tokens: np.ndarray | jnp.ndarray,
    offset_by: int,
    length_with_mm: int,
) -> np.ndarray | jnp.ndarray:
  """`_get_new_text_tokens_positions` without batch dimension."""

  # Empty buffer in which text and MM tokens will be inserted.
  tokens_with_mm = jnp.zeros((length_with_mm,), dtype=jnp.int32)

  # Shift the original tokens, so that the new soft tokens can be inserted.
  new_text_tokens_pos = _get_new_text_tokens_positions(
      offset_on=mm_start,
      offset_by=offset_by,
  )

  tokens_with_mm = tokens_with_mm.at[new_text_tokens_pos].set(text_tokens)

  # Remove the `<start_of_image>` tokens (will be added afterwards when
  # merging with `_get_new_mm_tokens`).
  first_mm_pos = tokens_with_mm[0]
  new_start_mm_pos = new_text_tokens_pos * mm_start
  tokens_with_mm = tokens_with_mm.at[new_start_mm_pos].set(0)
  tokens_with_mm = tokens_with_mm.at[0].set(first_mm_pos)

  return tokens_with_mm


def _get_new_text_tokens_positions(
    *,
    offset_on: np.ndarray | jnp.ndarray,
    offset_by: int,
) -> np.ndarray | jnp.ndarray:
  """Create the positions of the new tokens.

  Input: `[x, x, x, offset_on, x, x, offset_on, x]`
  Output: `[0, 1, 2, 3, 4+Offset, 5+Offset, 6+Offset, 7+Offset^2]`

  Args:
    offset_on: The token to offset on.
    offset_by: The number of tokens to offset by.

  Returns:
    The new positions of the tokens.
  """
  offset = jnp.cumsum(offset_on, axis=-1) * offset_by
  new_positions = jnp.arange(offset_on.shape[-1]) + offset
  # Do not shift the `<start_of_image>` token, it will be overwritten by the MM
  # tokens.
  new_positions -= offset_by * offset_on
  return new_positions


def _get_new_mm_tokens(
    *,
    mm_start: np.ndarray | jnp.ndarray,
    mm_tokens_to_insert: np.ndarray | jnp.ndarray,
    max_num_images: int,
    offset_by: int,
    length_with_mm: int,
) -> np.ndarray | jnp.ndarray:
  """batch dimension inclusive new mm_tokens"""
  # Jax vmap does not support positional argiments, so need the
  # _get_new_mm_tokens_inner indirection.
  return jax.vmap(_get_new_mm_tokens_inner, in_axes=(0, None, None, None, None))(
      mm_start, mm_tokens_to_insert, max_num_images, offset_by, length_with_mm
  )


def _get_new_mm_tokens_inner(
    mm_start: np.ndarray | jnp.ndarray,
    mm_tokens_to_insert: np.ndarray | jnp.ndarray,
    max_num_images: int,
    offset_by: int,
    length_with_mm: int,
) -> np.ndarray | jnp.ndarray:
  """`_get_new_mm_tokens` without batch dimension."""
  # Empty buffer row, which will be merged with the final tokens.
  row = jnp.zeros((length_with_mm,), dtype=jnp.int32)

  ones = jnp.ones((len(mm_tokens_to_insert),), dtype=jnp.int32)

  (offset,) = jnp.nonzero(mm_start, size=max_num_images)

  # Because not all elements in the batch do have the same number of images
  # we need to mask out the `offset == 0` values.
  # This means that `<start_of_images>` can never be the first token, but this
  # should never happen in practice as sequences should start by `<bos>`
  mask = offset != 0
  mask = jnp.einsum("...x,y->xy", mask, ones)

  # After the mask is created, offset each individual images
  offset += jnp.arange(len(offset)) * offset_by

  new_positions = jnp.einsum("x,y->xy", offset, ones)
  new_positions += jnp.arange(len(mm_tokens_to_insert))

  new_positions = new_positions * mask

  # Because not all elements in the batch do have the same number of images
  # we need to mask out the `offset == 0` values.
  # This means that `<start_of_images>` can never be the first token, but this
  # should never happen in practice as sequences should start by `<bos>`
  row = row.at[new_positions].set(mm_tokens_to_insert)
  row = row.at[0].set(0)
  return row


def merge_mm_embeddings(
    text_embeddings: np.ndarray | jnp.ndarray,
    vision_embeddings: np.ndarray | jnp.ndarray,
    mask,
) -> np.ndarray | jnp.ndarray:
  """Merge the text and MM tokens.

  Args:
    tokens: The text tokens.
    mm_tokens: The MM tokens.

  Returns:
    The merged tokens.
  """
  return jax.vmap(_merge_mm_embeddings_inner, in_axes=(0, 0, 0))(text_embeddings, vision_embeddings, mask)


def _merge_mm_embeddings_inner(text_embeddings, vision_embeddings, mask):
  """`merge_embeddings` without batch dimension."""

  # Rearrange the vision embeddings from [num_images, num_toks_per_image, d] to [num_images * num_toks_per_image, d]
  num_images, num_toks_per_image, d = vision_embeddings.shape
  vision_embeddings = jnp.reshape(vision_embeddings, (num_images * num_toks_per_image, d))

  # len(vision_embeddings) == max_num_images * num_tokens_per_image
  target_pos = jnp.nonzero(mask, size=len(vision_embeddings))

  # Save and restore the first position overwritten if there's no MM tokens.
  first_pos = text_embeddings[0]

  merged = text_embeddings.at[target_pos, :].set(vision_embeddings)

  merged = merged.at[0].set(first_pos)

  return merged
