# Copyright 2023–2025 Google LLC
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

"""Qwen3-Omni-specific preprocessing utilities for multimodal features. 

Original implementation from HuggingFace: Qwen/Qwen3-Omni-30B-A3B-Instruct.
"""

import decord
import jax
import librosa
import math
import numpy as np
from PIL import Image
from typing import Optional, Union
from dataclasses import dataclass

from MaxText import max_logging
from MaxText.multimodal import utils as mm_utils

# Image constants.
IMAGE_MEAN = 127.5
IMAGE_STD = 127.5
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

# Video constants.
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 128000 * 28 * 28 * 0.9
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

# Audio constants.
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
DITHER = 0.0


@dataclass
class Qwen3OmniPreprocessorOutput(mm_utils.PreprocessorOutput):
  """Holds the output of Qwen3-Omni image preprocessor.

  Attributes:
    Inherited from `mm_utils.PreprocessorOutput`.
  """

  # Image attributes.
  num_images: int = 0
  pixel_values: None | np.ndarray = None
  pixel_grid_thw: None | np.ndarray = None
  # Video attributes.
  num_videos: int = 0
  video_values: None | np.ndarray = None
  video_grid_thw: None | np.ndarray = None
  video_second_per_grid: None | np.ndarray = None
  # Audio attributes.
  num_audios: int = 0
  audio_values: None | np.ndarray = None
  audio_mask: None | np.ndarray = None


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
  """Rescales the image so that the following conditions are met:

  1. Both dimensions (height and width) are divisible by 'factor'.

  2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

  3. The aspect ratio of the image is maintained as closely as possible.

  """
  if max(height, width) / min(height, width) > MAX_RATIO:
    raise ValueError(
        f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
    )
  h_bar = round(height / factor) * factor
  w_bar = round(width / factor) * factor
  if h_bar * w_bar > max_pixels:
    beta = math.sqrt((height * width) / max_pixels)
    h_bar = max(factor, math.floor(height / beta / factor) * factor)
    w_bar = max(factor, math.floor(width / beta / factor) * factor)
  elif h_bar * w_bar < min_pixels:
    beta = math.sqrt(min_pixels / (height * width))
    h_bar = math.ceil(height * beta / factor) * factor
    w_bar = math.ceil(width * beta / factor) * factor
  return h_bar, w_bar


def pre_process_qwen3_image(image: np.ndarray | list[np.ndarray], config):
  """Performs a bi-linear resize (with anti-aliasing) and normalizes the image."""
  patch_size = config.patch_size_for_vit
  merge_size = config.spatial_merge_size_for_vit
  temporal_patch_size = config.temporal_patch_size_for_vit
  resample_method = Image.BICUBIC

  images_in = [image] if isinstance(image, np.ndarray) else image
  images_out = []
  grids_thw = []

  for img in images_in:
    pil_img = Image.fromarray(img)
    # Qwen3-Omni performs one resize during fetch_image and another resize before patchify.
    resized_height_1, resized_width_1 = smart_resize(
        height=img.shape[0],
        width=img.shape[1],
        factor=IMAGE_FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    pil_img = pil_img.resize((resized_width_1, resized_height_1))
    resized_height_2, resized_width_2 = smart_resize(
        height=resized_height_1,
        width=resized_width_1,
        factor=patch_size * merge_size,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    resized_img_pil = pil_img.resize((resized_width_2, resized_height_2), resample=resample_method)
    resized_img_np = np.array(resized_img_pil).astype(np.float32)

    img_np = mm_utils.normalize_images(resized_img_np, mean=IMAGE_MEAN, std=IMAGE_STD)
    img_np = np.permute_dims(img_np, (2, 0, 1))  # HWC to NCHW
    img_np = np.expand_dims(img_np, axis=(0, 1))  # add batch dimension
    img_np = np.repeat(img_np, temporal_patch_size, axis=1)  # add temporal dimension

    grid_t = 2 // temporal_patch_size
    grid_h, grid_w = resized_height_2 // patch_size, resized_width_2 // patch_size
    batch_size = img_np.shape[0]
    channel = img_np.shape[2]

    img_np = np.reshape(
        img_np,
        (
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        ),
    )
    img_np = np.permute_dims(img_np, (0, 1, 4, 7, 5, 8, 3, 2, 6, 9))  # HWC to CHW
    img_np = np.reshape(
        img_np, (batch_size, grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
    )
    img_grid_thw = np.asarray([grid_t, grid_h, grid_w], dtype=np.int32)
    images_out.append(img_np)
    grids_thw.append(img_grid_thw)

  return images_out[0][0, :, :], grids_thw[0]


def calculate_video_frame_range(
    ele: dict,
    total_frames: int,
    video_fps: float,
) -> tuple[int, int, int]:
  """
  Calculate the start and end frame indices based on the given time range.

  Args:
      ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
      total_frames (int): Total number of frames in the video.
      video_fps (float): Frames per second of the video.

  Returns:
      tuple: A tuple containing (start_frame, end_frame, frame_count).

  Raises:
      ValueError: If input parameters are invalid or the time range is inconsistent.
  """
  if video_fps <= 0:
    raise ValueError("video_fps must be a positive number")
  if total_frames <= 0:
    raise ValueError("total_frames must be a positive integer")

  video_start = ele.get("video_start", None)
  video_end = ele.get("video_end", None)
  if video_start is None and video_end is None:
    return 0, total_frames - 1, total_frames

  max_duration = total_frames / video_fps
  # Process start frame
  if video_start is not None:
    video_start_clamped = max(0.0, min(video_start, max_duration))
    start_frame = math.ceil(video_start_clamped * video_fps)
  else:
    start_frame = 0
  # Process end frame
  if video_end is not None:
    video_end_clamped = max(0.0, min(video_end, max_duration))
    end_frame = math.floor(video_end_clamped * video_fps)
    end_frame = min(end_frame, total_frames - 1)
  else:
    end_frame = total_frames - 1

  # Validate frame order
  if start_frame >= end_frame:
    raise ValueError(
        f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
        f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
        f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
    )

  return start_frame, end_frame, end_frame - start_frame + 1


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
  """Calculate the number of frames for video used for model inputs.

  Args:
      ele (dict): a dict contains the configuration of video.
          support either `fps` or `nframes`:
              - nframes: the number of frames to extract for model inputs.
              - fps: the fps to extract frames for model inputs.
                  - min_frames: the minimum number of frames of the video, only used when fps is provided.
                  - max_frames: the maximum number of frames of the video, only used when fps is provided.
      total_frames (int): the original total number of frames of the video.
      video_fps (int | float): the original fps of the video.

  Returns:
      int: the number of frames for video used for model inputs.
  """

  def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

  def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

  def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

  assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
  if "nframes" in ele:
    nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
  else:
    fps = ele.get("fps", FPS)
    min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
    max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
    nframes = total_frames / video_fps * fps
    if nframes > total_frames:
      max_logging.log(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
    nframes = min(max(nframes, min_frames), max_frames, total_frames)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)
  if not FRAME_FACTOR <= nframes <= total_frames:
    raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
  return nframes


def _read_video_decord(video_path, video_start=0.0, video_end=None) -> tuple[np.ndarray, float]:
  """Read video using decord.VideoReader (torch-free version)

  Args:
    video: the path of video. support "file://", "http://", "https://" and local path.
    video_start: the start time of video.
    video_end: the end time of video.

  Returns:
      tuple: (numpy.ndarray with shape (T, C, H, W), sample_fps as float)
  """
  video_config = {
      "video": video_path,
      "video_start": video_start,
      "video_end": video_end,
  }
  vr = decord.VideoReader(video_path)
  total_frames, video_fps = len(vr), vr.get_avg_fps()
  start_frame, end_frame, total_frames = calculate_video_frame_range(
      video_config,
      total_frames,
      video_fps,
  )
  nframes = smart_nframes(video_config, total_frames=total_frames, video_fps=video_fps)

  # Use numpy linspace instead of torch.linspace
  idx = np.linspace(start_frame, end_frame, nframes).round().astype(int).tolist()

  video = vr.get_batch(idx).asnumpy()
  # Convert from THWC to TCHW format using numpy
  video = np.transpose(video, (0, 3, 1, 2))

  sample_fps = nframes / max(total_frames, 1e-6) * video_fps
  return video, sample_fps


def preprocess_video(video, config):
  """Preprocess the video for Qwen3-Omni model."""
  patch_size = config.patch_size_for_vit
  merge_size = config.spatial_merge_size_for_vit
  temporal_patch_size = config.temporal_patch_size_for_vit

  nframes, channel, height, width = video.shape
  max_pixels = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR), int(VIDEO_MIN_PIXELS * 1.05))
  resized_height, resized_width = smart_resize(
      height,
      width,
      factor=IMAGE_FACTOR,
      min_pixels=VIDEO_MIN_PIXELS,
      max_pixels=max_pixels,
  )

  with jax.default_device(jax.devices("cpu")[0]):
    # Resize once during fetch_video and another resize before patchify in original implementation.
    video = video.astype(np.uint8)
    jax_resized_video = jax.image.resize(video, (nframes, channel, resized_height, resized_width), method="bicubic")
    jax_resized_video = jax_resized_video.astype(np.uint8)
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=VIDEO_MAX_PIXELS,
    )
    jax_resized_video = jax.image.resize(
        jax_resized_video, (nframes, channel, resized_height, resized_width), method="bicubic"
    )

  resized_video = np.array(jax_resized_video)
  resized_video = mm_utils.normalize_images(
      resized_video,
      mean=127.5,
      std=127.5,
  )
  resized_video = np.expand_dims(resized_video, axis=0)  # Add batch dimension
  batch_size, grid_t, channel = resized_video.shape[:3]
  grid_t = grid_t // temporal_patch_size
  grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

  resized_video = np.reshape(
      resized_video,
      (
          batch_size,
          grid_t,
          temporal_patch_size,
          channel,
          grid_h // merge_size,
          merge_size,
          patch_size,
          grid_w // merge_size,
          merge_size,
          patch_size,
      ),
  )
  resized_video = np.permute_dims(resized_video, (0, 1, 4, 7, 5, 8, 3, 2, 6, 9))  # HWC to CHW
  resized_video = np.reshape(
      resized_video, (batch_size, grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
  )
  processed_grid = np.asarray([[grid_t, grid_h, grid_w]], dtype=np.int32)

  return resized_video[0, :, :], processed_grid


def amplitude_to_db(
    spectrogram_array: np.ndarray,
    reference: float = 1.0,
    min_value: float = 1e-5,
    db_range: Optional[float] = None,
) -> np.ndarray:
  """
  Converts an amplitude spectrogram to the decibel scale. This computes `20 * log10(spectrogram / reference)`, using
  basic logarithm properties for numerical stability.

  The motivation behind applying the log function on the (mel) spectrogram is that humans do not hear loudness on a
  linear scale. Generally to double the perceived volume of a sound we need to put 8 times as much energy into it.
  This means that large variations in energy may not sound all that different if the sound is loud to begin with.
  This compression operation makes the (mel) spectrogram features match more closely what humans actually hear.

  Args:
      spectrogram (`np.ndarray`):
          The input amplitude (mel) spectrogram.
      reference (`float`, *optional*, defaults to 1.0):
          Sets the input spectrogram value that corresponds to 0 dB. For example, use `np.max(spectrogram)` to set
          the loudest part to 0 dB. Must be greater than zero.
      min_value (`float`, *optional*, defaults to `1e-5`):
          The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking
          `log(0)`. The default of `1e-5` corresponds to a minimum of -100 dB. Must be greater than zero.
      db_range (`float`, *optional*):
          Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
          peak value and the smallest value will never be more than 80 dB. Must be greater than zero.

  Returns:
      `np.ndarray`: the spectrogram in decibels
  """
  if reference <= 0.0:
    raise ValueError("reference must be greater than zero")
  if min_value <= 0.0:
    raise ValueError("min_value must be greater than zero")

  reference = max(min_value, reference)

  spectrogram_array = np.clip(spectrogram_array, a_min=min_value, a_max=None)
  spectrogram_array = 20.0 * (np.log10(spectrogram_array) - np.log10(reference))

  if db_range is not None:
    if db_range <= 0.0:
      raise ValueError("db_range must be greater than zero")
    spectrogram_array = np.clip(spectrogram_array, a_min=spectrogram_array.max() - db_range, a_max=None)

  return spectrogram_array


def power_to_db(
    spectrogram_array: np.ndarray,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: Optional[float] = None,
) -> np.ndarray:
  """
  Converts a power spectrogram to the decibel scale. This computes `10 * log10(spectrogram / reference)`, using basic
  logarithm properties for numerical stability.

  The motivation behind applying the log function on the (mel) spectrogram is that humans do not hear loudness on a
  linear scale. Generally to double the perceived volume of a sound we need to put 8 times as much energy into it.
  This means that large variations in energy may not sound all that different if the sound is loud to begin with.
  This compression operation makes the (mel) spectrogram features match more closely what humans actually hear.

  Based on the implementation of `librosa.power_to_db`.

  Args:
      spectrogram (`np.ndarray`):
          The input power (mel) spectrogram. Note that a power spectrogram has the amplitudes squared!
      reference (`float`, *optional*, defaults to 1.0):
          Sets the input spectrogram value that corresponds to 0 dB. For example, use `np.max(spectrogram)` to set
          the loudest part to 0 dB. Must be greater than zero.
      min_value (`float`, *optional*, defaults to `1e-10`):
          The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking
          `log(0)`. The default of `1e-10` corresponds to a minimum of -100 dB. Must be greater than zero.
      db_range (`float`, *optional*):
          Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
          peak value and the smallest value will never be more than 80 dB. Must be greater than zero.

  Returns:
      `np.ndarray`: the spectrogram in decibels
  """
  if reference <= 0.0:
    raise ValueError("reference must be greater than zero")
  if min_value <= 0.0:
    raise ValueError("min_value must be greater than zero")

  reference = max(min_value, reference)

  spectrogram_array = np.clip(spectrogram_array, a_min=min_value, a_max=None)
  spectrogram_array = 10.0 * (np.log10(spectrogram_array) - np.log10(reference))

  if db_range is not None:
    if db_range <= 0.0:
      raise ValueError("db_range must be greater than zero")
    spectrogram_array = np.clip(spectrogram_array, a_min=spectrogram_array.max() - db_range, a_max=None)

  return spectrogram_array


def spectrogram(
    waveform: np.ndarray,
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
    fft_length: Optional[int] = None,
    power: Optional[float] = 1.0,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    dither: float = 0.0,
    preemphasis: Optional[float] = None,
    mel_filters: Optional[np.ndarray] = None,
    mel_floor: float = 1e-10,
    log_mel: Optional[str] = None,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: Optional[float] = None,
    remove_dc_offset: bool = False,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
  """
  Calculates a spectrogram over one waveform using the Short-Time Fourier Transform.

  This function can create the following kinds of spectrograms:

    - amplitude spectrogram (`power = 1.0`)
    - power spectrogram (`power = 2.0`)
    - complex-valued spectrogram (`power = None`)
    - log spectrogram (use `log_mel` argument)
    - mel spectrogram (provide `mel_filters`)
    - log-mel spectrogram (provide `mel_filters` and `log_mel`)

  How this works:

    1. The input waveform is split into frames of size `frame_length` that are partially overlapping by `frame_length
       - hop_length` samples.
    2. Each frame is multiplied by the window and placed into a buffer of size `fft_length`.
    3. The DFT is taken of each windowed frame.
    4. The results are stacked into a spectrogram.

  We make a distinction between the following "blocks" of sample data, each of which may have a different lengths:

    - The analysis frame. This is the size of the time slices that the input waveform is split into.
    - The window. Each analysis frame is multiplied by the window to avoid spectral leakage.
    - The FFT input buffer. The length of this determines how many frequency bins are in the spectrogram.

  In this implementation, the window is assumed to be zero-padded to have the same size as the analysis frame. A
  padded window can be obtained from `window_function()`. The FFT input buffer may be larger than the analysis frame,
  typically the next power of two.

  Note: This function is not optimized for speed yet. It should be mostly compatible with `librosa.stft` and
  `torchaudio.functional.transforms.Spectrogram`, although it is more flexible due to the different ways spectrograms
  can be constructed.

  Args:
      waveform (`np.ndarray` of shape `(length,)`):
          The input waveform. This must be a single real-valued, mono waveform.
      window (`np.ndarray` of shape `(frame_length,)`):
          The windowing function to apply, including zero-padding if necessary. The actual window length may be
          shorter than `frame_length`, but we're assuming the array has already been zero-padded.
      frame_length (`int`):
          The length of the analysis frames in samples. With librosa this is always equal to `fft_length` but we also
          allow smaller sizes.
      hop_length (`int`):
          The stride between successive analysis frames in samples.
      fft_length (`int`, *optional*):
          The size of the FFT buffer in samples. This determines how many frequency bins the spectrogram will have.
          For optimal speed, this should be a power of two. If `None`, uses `frame_length`.
      power (`float`, *optional*, defaults to 1.0):
          If 1.0, returns the amplitude spectrogram. If 2.0, returns the power spectrogram. If `None`, returns
          complex numbers.
      center (`bool`, *optional*, defaults to `True`):
          Whether to pad the waveform so that frame `t` is centered around time `t * hop_length`. If `False`, frame
          `t` will start at time `t * hop_length`.
      pad_mode (`str`, *optional*, defaults to `"reflect"`):
          Padding mode used when `center` is `True`. Possible values are: `"constant"` (pad with zeros), `"edge"`
          (pad with edge values), `"reflect"` (pads with mirrored values).
      onesided (`bool`, *optional*, defaults to `True`):
          If True, only computes the positive frequencies and returns a spectrogram containing `fft_length // 2 + 1`
          frequency bins. If False, also computes the negative frequencies and returns `fft_length` frequency bins.
      dither (`float`, *optional*, defaults to 0.0):
          Adds dithering. In other words, adds a small Gaussian noise to each frame.
          E.g. use 4.0 to add dithering with a normal distribution centered
          around 0.0 with standard deviation 4.0, 0.0 means no dithering.
          Dithering has similar effect as `mel_floor`. It reduces the high log_mel_fbank
          values for signals with hard-zero sections, when VAD cutoff is present in the signal.
      preemphasis (`float`, *optional*)
          Coefficient for a low-pass filter that applies pre-emphasis before the DFT.
      mel_filters (`np.ndarray` of shape `(num_freq_bins, num_mel_filters)`, *optional*):
          The mel filter bank. If supplied, applies a this filter bank to create a mel spectrogram.
      mel_floor (`float`, *optional*, defaults to 1e-10):
          Minimum value of mel frequency banks.
      log_mel (`str`, *optional*):
          How to convert the spectrogram to log scale. Possible options are: `None` (don't convert), `"log"` (take
          the natural logarithm) `"log10"` (take the base-10 logarithm), `"dB"` (convert to decibels). Can only be
          used when `power` is not `None`.
      reference (`float`, *optional*, defaults to 1.0):
          Sets the input spectrogram value that corresponds to 0 dB. For example, use `np.max(spectrogram)` to set
          the loudest part to 0 dB. Must be greater than zero.
      min_value (`float`, *optional*, defaults to `1e-10`):
          The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking
          `log(0)`. For a power spectrogram, the default of `1e-10` corresponds to a minimum of -100 dB. For an
          amplitude spectrogram, the value `1e-5` corresponds to -100 dB. Must be greater than zero.
      db_range (`float`, *optional*):
          Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
          peak value and the smallest value will never be more than 80 dB. Must be greater than zero.
      remove_dc_offset (`bool`, *optional*):
          Subtract mean from waveform on each frame, applied before pre-emphasis. This should be set to `true` in
          order to get the same results as `torchaudio.compliance.kaldi.fbank` when computing mel filters.
      dtype (`np.dtype`, *optional*, defaults to `np.float32`):
          Data type of the spectrogram tensor. If `power` is None, this argument is ignored and the dtype will be
          `np.complex64`.

  Returns:
      `nd.array` containing a spectrogram of shape `(num_frequency_bins, length)` for a regular spectrogram or shape
      `(num_mel_filters, length)` for a mel spectrogram.
  """
  window_length = len(window)

  if fft_length is None:
    fft_length = frame_length

  if frame_length > fft_length:
    raise ValueError(f"frame_length ({frame_length}) may not be larger than fft_length ({fft_length})")

  if window_length != frame_length:
    raise ValueError(f"Length of the window ({window_length}) must equal frame_length ({frame_length})")

  if hop_length <= 0:
    raise ValueError("hop_length must be greater than zero")

  if waveform.ndim != 1:
    raise ValueError(f"Input waveform must have only one dimension, shape is {waveform.shape}")

  if np.iscomplexobj(waveform):
    raise ValueError("Complex-valued input waveforms are not currently supported")

  if power is None and mel_filters is not None:
    raise ValueError(
        "You have provided `mel_filters` but `power` is `None`. "
        "Mel spectrogram computation is not yet supported for complex-valued spectrogram."
        "Specify `power` to fix this issue."
    )

  # center pad the waveform
  if center:
    padding = [(int(frame_length // 2), int(frame_length // 2))]
    waveform = np.pad(waveform, padding, mode=pad_mode)

  # promote to float64, since np.fft uses float64 internally
  waveform = waveform.astype(np.float64)
  window = window.astype(np.float64)

  # split waveform into frames of frame_length size
  num_frames = int(1 + np.floor((waveform.size - frame_length) / hop_length))

  num_frequency_bins = (fft_length // 2) + 1 if onesided else fft_length
  spectrogram_array = np.empty((num_frames, num_frequency_bins), dtype=np.complex64)

  # rfft is faster than fft
  fft_func = np.fft.rfft if onesided else np.fft.fft
  buffer = np.zeros(fft_length)

  timestep = 0
  for frame_idx in range(num_frames):
    buffer[:frame_length] = waveform[timestep : timestep + frame_length]

    if dither != 0.0:
      buffer[:frame_length] += dither * np.random.randn(frame_length)

    if remove_dc_offset:
      buffer[:frame_length] = buffer[:frame_length] - buffer[:frame_length].mean()

    if preemphasis is not None:
      buffer[1:frame_length] -= preemphasis * buffer[: frame_length - 1]
      buffer[0] *= 1 - preemphasis

    buffer[:frame_length] *= window

    spectrogram_array[frame_idx] = fft_func(buffer)
    timestep += hop_length

  # note: ** is much faster than np.power
  if power is not None:
    spectrogram_array = np.abs(spectrogram_array, dtype=np.float64) ** power

  spectrogram_array = spectrogram_array.T

  if mel_filters is not None:
    spectrogram_array = np.maximum(mel_floor, np.dot(mel_filters.T, spectrogram_array))

  if power is not None and log_mel is not None:
    if log_mel == "log":
      spectrogram_array = np.log(spectrogram_array)
    elif log_mel == "log10":
      spectrogram_array = np.log10(spectrogram_array)
    elif log_mel == "dB":
      if power == 1.0:
        spectrogram_array = amplitude_to_db(spectrogram_array, reference, min_value, db_range)
      elif power == 2.0:
        spectrogram_array = power_to_db(spectrogram_array, reference, min_value, db_range)
      else:
        raise ValueError(f"Cannot use log_mel option '{log_mel}' with power {power}")
    else:
      raise ValueError(f"Unknown log_mel option: {log_mel}")

    spectrogram_array = np.asarray(spectrogram_array, dtype)

  return spectrogram_array


def hertz_to_mel(freq: Union[float, np.ndarray], mel_scale: str = "htk") -> Union[float, np.ndarray]:
  """
  Convert frequency from hertz to mels.

  Args:
      freq (`float` or `np.ndarray`):
          The frequency, or multiple frequencies, in hertz (Hz).
      mel_scale (`str`, *optional*, defaults to `"htk"`):
          The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.

  Returns:
      `float` or `np.ndarray`: The frequencies on the mel scale.
  """

  if mel_scale not in ["slaney", "htk", "kaldi"]:
    raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

  if mel_scale == "htk":
    return 2595.0 * np.log10(1.0 + (freq / 700.0))
  elif mel_scale == "kaldi":
    return 1127.0 * np.log(1.0 + (freq / 700.0))

  min_log_hertz = 1000.0
  min_log_mel = 15.0
  logstep = 27.0 / np.log(6.4)
  mels = 3.0 * freq / 200.0

  if isinstance(freq, np.ndarray):
    log_region = freq >= min_log_hertz
    mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
  elif freq >= min_log_hertz:
    mels = min_log_mel + np.log(freq / min_log_hertz) * logstep

  return mels


def _create_triangular_filter_bank(fft_freqs: np.ndarray, filter_freqs: np.ndarray) -> np.ndarray:
  """
  Creates a triangular filter bank.

  Adapted from *torchaudio* and *librosa*.

  Args:
      fft_freqs (`np.ndarray` of shape `(num_frequency_bins,)`):
          Discrete frequencies of the FFT bins in Hz.
      filter_freqs (`np.ndarray` of shape `(num_mel_filters,)`):
          Center frequencies of the triangular filters to create, in Hz.

  Returns:
      `np.ndarray` of shape `(num_frequency_bins, num_mel_filters)`
  """
  filter_diff = np.diff(filter_freqs)
  slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
  down_slopes = -slopes[:, :-2] / filter_diff[:-1]
  up_slopes = slopes[:, 2:] / filter_diff[1:]
  return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


def mel_to_hertz(mels: Union[float, np.ndarray], mel_scale: str = "htk") -> Union[float, np.ndarray]:
  """
  Convert frequency from mels to hertz.

  Args:
      mels (`float` or `np.ndarray`):
          The frequency, or multiple frequencies, in mels.
      mel_scale (`str`, *optional*, `"htk"`):
          The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.

  Returns:
      `float` or `np.ndarray`: The frequencies in hertz.
  """

  if mel_scale not in ["slaney", "htk", "kaldi"]:
    raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

  if mel_scale == "htk":
    return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
  elif mel_scale == "kaldi":
    return 700.0 * (np.exp(mels / 1127.0) - 1.0)

  min_log_hertz = 1000.0
  min_log_mel = 15.0
  logstep = np.log(6.4) / 27.0
  freq = 200.0 * mels / 3.0

  if isinstance(mels, np.ndarray):
    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
  elif mels >= min_log_mel:
    freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))

  return freq


def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
    triangularize_in_mel_space: bool = False,
) -> np.ndarray:
  """
  Creates a frequency bin conversion matrix used to obtain a mel spectrogram. This is called a *mel filter bank*, and
  various implementation exist, which differ in the number of filters, the shape of the filters, the way the filters
  are spaced, the bandwidth of the filters, and the manner in which the spectrum is warped. The goal of these
  features is to approximate the non-linear human perception of the variation in pitch with respect to the frequency.

  Different banks of mel filters were introduced in the literature. The following variations are supported:

  - MFCC FB-20: introduced in 1980 by Davis and Mermelstein, it assumes a sampling frequency of 10 kHz and a speech
    bandwidth of `[0, 4600]` Hz.
  - MFCC FB-24 HTK: from the Cambridge HMM Toolkit (HTK) (1995) uses a filter bank of 24 filters for a speech
    bandwidth of `[0, 8000]` Hz. This assumes sampling rate ≥ 16 kHz.
  - MFCC FB-40: from the Auditory Toolbox for MATLAB written by Slaney in 1998, assumes a sampling rate of 16 kHz and
    speech bandwidth of `[133, 6854]` Hz. This version also includes area normalization.
  - HFCC-E FB-29 (Human Factor Cepstral Coefficients) of Skowronski and Harris (2004), assumes a sampling rate of
    12.5 kHz and speech bandwidth of `[0, 6250]` Hz.

  This code is adapted from *torchaudio* and *librosa*. Note that the default parameters of torchaudio's
  `melscale_fbanks` implement the `"htk"` filters while librosa uses the `"slaney"` implementation.

  Args:
      num_frequency_bins (`int`):
          Number of frequency bins (should be the same as `n_fft // 2 + 1` where `n_fft` is the size of the Fourier
          Transform used to compute the spectrogram).
      num_mel_filters (`int`):
          Number of mel filters to generate.
      min_frequency (`float`):
          Lowest frequency of interest in Hz.
      max_frequency (`float`):
          Highest frequency of interest in Hz. This should not exceed `sampling_rate / 2`.
      sampling_rate (`int`):
          Sample rate of the audio waveform.
      norm (`str`, *optional*):
          If `"slaney"`, divide the triangular mel weights by the width of the mel band (area normalization).
      mel_scale (`str`, *optional*, defaults to `"htk"`):
          The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.
      triangularize_in_mel_space (`bool`, *optional*, defaults to `False`):
          If this option is enabled, the triangular filter is applied in mel space rather than frequency space. This
          should be set to `true` in order to get the same results as `torchaudio` when computing mel filters.

  Returns:
      `np.ndarray` of shape (`num_frequency_bins`, `num_mel_filters`): Triangular filter bank matrix. This is a
      projection matrix to go from a spectrogram to a mel spectrogram.
  """
  if norm is not None and norm != "slaney":
    raise ValueError('norm must be one of None or "slaney"')

  if num_frequency_bins < 2:
    raise ValueError(f"Require num_frequency_bins: {num_frequency_bins} >= 2")

  if min_frequency > max_frequency:
    raise ValueError(f"Require min_frequency: {min_frequency} <= max_frequency: {max_frequency}")

  # center points of the triangular mel filters
  mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
  mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
  mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
  filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)

  if triangularize_in_mel_space:
    # frequencies of FFT bins in Hz, but filters triangularized in mel space
    fft_bin_width = sampling_rate / ((num_frequency_bins - 1) * 2)
    fft_freqs = hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_scale)
    filter_freqs = mel_freqs
  else:
    # frequencies of FFT bins in Hz
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

  mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

  if norm is not None and norm == "slaney":
    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    mel_filters *= np.expand_dims(enorm, 0)

  if (mel_filters.max(axis=0) == 0.0).any():
    print(
        "At least one mel filter has all zero values. "
        f"The value for `num_mel_filters` ({num_mel_filters}) may be set too high. "
        f"Or, the value for `num_frequency_bins` ({num_frequency_bins}) may be set too low."
    )

  return mel_filters


def window_function(
    window_length: int,
    name: str = "hann",
    periodic: bool = True,
    frame_length: Optional[int] = None,
    center: bool = True,
) -> np.ndarray:
  """
  Returns an array containing the specified window. This window is intended to be used with `stft`.

  The following window types are supported:

      - `"boxcar"`: a rectangular window
      - `"hamming"`: the Hamming window
      - `"hann"`: the Hann window
      - `"povey"`: the Povey window

  Args:
      window_length (`int`):
          The length of the window in samples.
      name (`str`, *optional*, defaults to `"hann"`):
          The name of the window function.
      periodic (`bool`, *optional*, defaults to `True`):
          Whether the window is periodic or symmetric.
      frame_length (`int`, *optional*):
          The length of the analysis frames in samples. Provide a value for `frame_length` if the window is smaller
          than the frame length, so that it will be zero-padded.
      center (`bool`, *optional*, defaults to `True`):
          Whether to center the window inside the FFT buffer. Only used when `frame_length` is provided.

  Returns:
      `np.ndarray` of shape `(window_length,)` or `(frame_length,)` containing the window.
  """
  length = window_length + 1 if periodic else window_length

  if name == "boxcar":
    window = np.ones(length)
  elif name in ["hamming", "hamming_window"]:
    window = np.hamming(length)
  elif name in ["hann", "hann_window"]:
    window = np.hanning(length)
  elif name == "povey":
    window = np.power(np.hanning(length), 0.85)
  else:
    raise ValueError(f"Unknown window function '{name}'")

  if periodic:
    window = window[:-1]

  if frame_length is None:
    return window

  if window_length > frame_length:
    raise ValueError(f"Length of the window ({window_length}) may not be larger than frame_length ({frame_length})")

  padded_window = np.zeros(frame_length)
  offset = (frame_length - window_length) // 2 if center else 0
  padded_window[offset : offset + window_length] = window
  return padded_window


def _np_extract_fbank_features(waveform_batch: np.ndarray) -> np.ndarray:
  """
  Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch
  implementation with 1e-5 tolerance.
  """
  log_spec_batch = []
  mel_filters = mel_filter_bank(
      num_frequency_bins=1 + 400 // 2,
      num_mel_filters=128,
      min_frequency=0.0,
      max_frequency=8000.0,
      sampling_rate=16000,
      norm="slaney",
      mel_scale="slaney",
  )
  for waveform in waveform_batch:
    log_spec = spectrogram(
        waveform,
        window_function(400, "hann"),
        frame_length=400,
        hop_length=160,
        power=2.0,
        dither=0.0,
        mel_filters=mel_filters,
        log_mel="log10",
    )
    log_spec = log_spec[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    log_spec_batch.append(log_spec)
  log_spec_batch = np.array(log_spec_batch)
  return log_spec_batch


def _load_audio(data_path: str) -> np.ndarray:
  """Load audio from a file path.

  Args:
      data_path (str): The path to the audio file.

  Returns:
      np.ndarray: The loaded audio waveform.
  """
  audio = librosa.load(data_path, sr=SAMPLE_RATE)[0]
  return audio


def pre_process_audio_qwen3_omni(audio_array):
  """Preprocess audio for Qwen3-Omni model."""
  audio_features = np.expand_dims(audio_array, axis=0)  # Add batch dimension
  audio_features = _np_extract_fbank_features(audio_features)
  audio_features_mask = np.ones((audio_features.shape[0], audio_features.shape[2]), dtype=np.int32)
  return audio_features, audio_features_mask


def preprocess_mm_data_qwen3_omni(config):
  """Placeholder for multimodal data preprocessing."""
  processor_outputs = Qwen3OmniPreprocessorOutput()

  if config.image_path is not None:
    images = [mm_utils.load_image_from_path(p) for p in config.image_path.split(",")]
    pixel_values, pixel_grid_thw = pre_process_qwen3_image(images, config)
    processor_outputs.pixel_values = pixel_values
    processor_outputs.pixel_grid_thw = pixel_grid_thw
    processor_outputs.num_images = len(images)

  if config.video_path is not None:
    video_array, _ = _read_video_decord(config.video_path)
    video_processed, video_grid_thw = preprocess_video(video_array, config)
    processor_outputs.video_values = video_processed
    processor_outputs.video_grid_thw = video_grid_thw
    processor_outputs.video_second_per_grid = np.asarray([config.temporal_patch_size_for_vit], dtype=np.float32)
    processor_outputs.num_videos = 1  # Only one video for now.

  if config.audio_path is not None or (config.video_path is not None and config.use_audio_in_video):
    mt_audio = _load_audio(config.video_path)
    mt_audio, mt_audio_mask = pre_process_audio_qwen3_omni(mt_audio)
    processor_outputs.audio_values = mt_audio
    processor_outputs.audio_mask = mt_audio_mask

  return processor_outputs
