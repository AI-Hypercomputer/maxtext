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

"""Qwen3-Omni-specific preprocessing utilities for multimodal features. 

Original implementation from HuggingFace: Qwen/Qwen3-Omni-30B-A3B-Instruct.
"""

import math
import os
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image

try:
  import decord  # pytype: disable=import-error
except ImportError:
  decord = None

from MaxText.multimodal import utils as mm_utils
from maxtext.utils import max_logging

# Image constants.
IMAGE_MEAN = 127.5  # Mean value for image normalization.
IMAGE_STD = 127.5  # Standard deviation for image normalization.
IMAGE_FACTOR = 28  # Resize factor for image dimensions (patch_size).
MIN_PIXELS = 4 * 28 * 28  # Minimum image pixels: 4 patches × patch_size².
MAX_PIXELS = 16384 * 28 * 28  # Maximum image pixels: 16384 patches × patch_size².
MAX_RATIO = 200  # Maximum allowed aspect ratio for images.

# Video constants.
VIDEO_MIN_PIXELS = 128 * 28 * 28  # Minimum video pixels: 128 patches × patch_size².
VIDEO_MAX_PIXELS = 768 * 28 * 28  # Maximum video pixels: 768 patches × patch_size².
VIDEO_TOTAL_PIXELS = 128000 * 28 * 28 * 0.9  # Total video pixels budget: 128000 patches × patch_size² × 0.9.
FRAME_FACTOR = 2  # Frame count must be divisible by this factor.
FPS = 2.0  # Default frames per second for video sampling.
FPS_MIN_FRAMES = 4  # Minimum number of frames to extract from video.
FPS_MAX_FRAMES = 768  # Maximum number of frames to extract from video.

# Audio constants.
SAMPLE_RATE = 16000  # Audio sampling rate in Hz.
N_FFT = 400  # Number of FFT points for spectrogram computation.
HOP_LENGTH = 160  # Number of samples between successive frames.
DITHER = 0.0  # Amount of dithering to apply to audio signal.

# Qwen3OmniMoe-specific processing
QWEN3_OMNI_VISION_START_TOKEN = 151652  # <|vision_start|>
QWEN3_OMNI_VISION_END_TOKEN = 151653  # <|vision_eos|>
QWEN3_OMNI_IMAGE_TOKEN = 151655  # <|image_pad|>
QWEN3_OMNI_VIDEO_TOKEN = 151656  # <|video_pad|>
QWEN3_OMNI_AUDIO_START_TOKEN = 151669  # <|audio_start|>
QWEN3_OMNI_AUDIO_END_TOKEN = 151648  # <|audio_eos|>
QWEN3_OMNI_AUDIO_TOKEN = 151675  # <|audio_pad|>
QWEN3_TEMPORAL_PATCH_SIZE = 2
QWEN3_OMNI_IMAGE_SIZE = 768


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

  # Images are concatenated along the sequence dimension e.g. (seq1 + seq2, 1536)
  concatenated_images = np.concatenate([img[0] for img in images_out], axis=0)
  return concatenated_images, np.stack(grids_thw)


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

  Raises:
      FileNotFoundError: If the video file does not exist.
      RuntimeError: If the video file cannot be read.
  """
  if decord is None:
    raise ImportError("decord is required for video processing but not installed.")
  if not os.path.isfile(video_path):
    raise FileNotFoundError(f"Video file not found at path {video_path}. Please specify a valid video file path")
  video_config = {
      "video": video_path,
      "video_start": video_start,
      "video_end": video_end,
  }
  try:
    vr = decord.VideoReader(video_path)
  except Exception as e:
    raise RuntimeError(f"Failed to read video from {video_path}: {e}") from e
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
  resized_height_1, resized_width_1 = smart_resize(
      height,
      width,
      factor=IMAGE_FACTOR,
      min_pixels=VIDEO_MIN_PIXELS,
      max_pixels=max_pixels,
  )

  # First resize - using PIL to match HuggingFace behavior
  resized_frames = []
  for frame_idx in range(nframes):
    # Convert from CHW to HWC for PIL
    frame = np.transpose(video[frame_idx], (1, 2, 0))
    pil_frame = Image.fromarray(frame.astype(np.uint8))
    pil_frame = pil_frame.resize((resized_width_1, resized_height_1), Image.BICUBIC)
    # Keep as float32 to preserve values outside [0, 255] from interpolation
    resized_frames.append(np.array(pil_frame, dtype=np.float32))

  resized_video = np.stack(resized_frames)

  # Second resize
  resized_height_2, resized_width_2 = smart_resize(
      resized_height_1,
      resized_width_1,
      factor=patch_size * merge_size,
      min_pixels=VIDEO_MIN_PIXELS,
      max_pixels=VIDEO_MAX_PIXELS,
  )

  # Second resize - process each channel separately to preserve float values
  final_frames = []
  for frame in resized_video:
    channels = []
    for c in range(frame.shape[2]):
      # Process each channel separately using PIL 'F' mode (float32)
      channel_data = frame[:, :, c]
      pil_frame = Image.fromarray(channel_data, mode="F")
      pil_frame = pil_frame.resize((resized_width_2, resized_height_2), Image.BICUBIC)
      channels.append(np.array(pil_frame, dtype=np.float32))
    final_frames.append(np.stack(channels, axis=2))

  resized_video = np.stack(final_frames)
  # Convert back to TCHW format
  resized_video = np.transpose(resized_video, (0, 3, 1, 2))

  resized_height, resized_width = resized_height_2, resized_width_2
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


def _np_extract_fbank_features(waveform_batch: np.ndarray) -> np.ndarray:
  """
  Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch
  implementation with 1e-5 tolerance.
  """
  log_spec_batch = []
  mel_filters = mm_utils.mel_filter_bank(
      num_frequency_bins=1 + N_FFT // 2,
      num_mel_filters=128,
      min_frequency=0.0,
      max_frequency=8000.0,
      sampling_rate=SAMPLE_RATE,
      norm="slaney",
      mel_scale="slaney",
  )
  for waveform in waveform_batch:
    log_spec = mm_utils.spectrogram(
        waveform,
        mm_utils.window_function(N_FFT, "hann"),
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
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

  if config.video_path is not None and config.use_audio_in_video:
    # TODO(hengtaoguo): add support for separate audio files. Now only extract audio from video files.
    mt_audio = mm_utils.load_audio(config.video_path, sample_rate=SAMPLE_RATE)
    mt_audio, mt_audio_mask = pre_process_audio_qwen3_omni(mt_audio)
    processor_outputs.audio_values = mt_audio
    processor_outputs.audio_mask = mt_audio_mask

  return processor_outputs


def add_extra_tokens_for_qwen3_omni(
    tokens: np.ndarray | list,
    image_grid_thw: np.ndarray | None = None,
    video_grid_thw: np.ndarray | None = None,
    audio_lengths: np.ndarray | None = None,
    spatial_merge_size: int = 2,
    use_audio_in_video: bool = False,
    second_per_grids: np.ndarray | None = None,
    position_id_per_seconds: int = 25,
):
  """Add extra tokens for Qwen3-Omni multimodal sequences.

  Expands special tokens (<|image_pad|>, <|video_pad|>, <|audio_pad|>) into
  the correct number of placeholder tokens based on grid dimensions and merge size.

  For audio-in-video mode, interleaves audio and video tokens based on temporal ordering.

  Args:
    tokens: Input token sequence (1D array or list).
    image_grid_thw: Image dimensions (num_images, 3) with [temporal, height, width].
    video_grid_thw: Video dimensions (num_videos, 3) with [temporal, height, width].
    audio_lengths: Pre-computed audio token counts (num_audios,).
    spatial_merge_size: Number of patches merged spatially (e.g., 2 for 2x2→1).
    use_audio_in_video: If True, interleave audio and video tokens.
    second_per_grids: Time interval per temporal grid (num_videos,).
    position_id_per_seconds: Temporal granularity (tokens per second).

  Returns:
    Expanded token sequence with correct number of image/video/audio tokens.
  """
  if not isinstance(tokens, np.ndarray):
    tokens = np.asarray(tokens)

  tokens = tokens.flatten()  # Ensure 1D

  # Merge lengths for computing number of tokens
  merge_length = spatial_merge_size**2

  # Convert to list for easier manipulation
  token_list = tokens.tolist()
  new_tokens = []

  image_idx = 0
  video_idx = 0
  audio_idx = 0

  i = 0
  while i < len(token_list):
    token = token_list[i]

    # Handle image tokens
    if token == QWEN3_OMNI_IMAGE_TOKEN and image_grid_thw is not None and image_idx < len(image_grid_thw):
      grid = image_grid_thw[image_idx]
      num_image_tokens = int((grid[0] * grid[1] * grid[2]) // merge_length)
      new_tokens.extend([QWEN3_OMNI_IMAGE_TOKEN] * num_image_tokens)
      image_idx += 1

    # Handle audio-in-video: <|vision_start|><|video_pad|><|vision_end|>
    elif (
        use_audio_in_video
        and token == QWEN3_OMNI_VISION_START_TOKEN
        and i + 2 < len(token_list)
        and token_list[i + 1] == QWEN3_OMNI_VIDEO_TOKEN
        and token_list[i + 2] == QWEN3_OMNI_VISION_END_TOKEN
        and video_grid_thw is not None
        and video_idx < len(video_grid_thw)
    ):

      if audio_lengths is None or audio_idx >= len(audio_lengths):
        raise ValueError("audio_lengths required for audio-in-video mode")
      if second_per_grids is None or video_idx >= len(second_per_grids):
        raise ValueError("second_per_grids required for audio-in-video mode")

      audio_length = audio_lengths[audio_idx]
      audio_token_indices = np.arange(audio_length)

      curr_video_grid = video_grid_thw[video_idx]
      height = curr_video_grid[1] // spatial_merge_size
      width = curr_video_grid[2] // spatial_merge_size
      num_frames = curr_video_grid[0]

      video_token_indices = np.arange(num_frames).reshape(-1, 1, 1)
      video_token_indices = np.broadcast_to(video_token_indices, (num_frames, height, width)).flatten()
      video_token_indices = video_token_indices * second_per_grids[video_idx] * position_id_per_seconds

      new_tokens.append(QWEN3_OMNI_VISION_START_TOKEN)
      new_tokens.append(QWEN3_OMNI_AUDIO_START_TOKEN)

      video_data_idx = 0
      audio_data_idx = 0

      while video_data_idx < len(video_token_indices) and audio_data_idx < len(audio_token_indices):
        if video_token_indices[video_data_idx] <= audio_token_indices[audio_data_idx]:
          new_tokens.append(QWEN3_OMNI_VIDEO_TOKEN)
          video_data_idx += 1
        else:
          new_tokens.append(QWEN3_OMNI_AUDIO_TOKEN)
          audio_data_idx += 1

      while video_data_idx < len(video_token_indices):
        new_tokens.append(QWEN3_OMNI_VIDEO_TOKEN)
        video_data_idx += 1

      while audio_data_idx < len(audio_token_indices):
        new_tokens.append(QWEN3_OMNI_AUDIO_TOKEN)
        audio_data_idx += 1

      new_tokens.append(QWEN3_OMNI_AUDIO_END_TOKEN)
      new_tokens.append(QWEN3_OMNI_VISION_END_TOKEN)

      video_idx += 1
      audio_idx += 1
      i += 2

    # Handle video tokens (without audio-in-video)
    elif token == QWEN3_OMNI_VIDEO_TOKEN and video_grid_thw is not None and video_idx < len(video_grid_thw):
      grid = video_grid_thw[video_idx]
      num_video_tokens = int((grid[0] * grid[1] * grid[2]) // merge_length)
      new_tokens.extend([QWEN3_OMNI_VIDEO_TOKEN] * num_video_tokens)
      video_idx += 1

    # Handle audio tokens (standalone, not in video)
    elif token == QWEN3_OMNI_AUDIO_TOKEN and audio_lengths is not None and audio_idx < len(audio_lengths):
      num_audio_tokens = int(audio_lengths[audio_idx])
      new_tokens.extend([QWEN3_OMNI_AUDIO_TOKEN] * num_audio_tokens)
      audio_idx += 1

    # All other tokens pass through unchanged
    else:
      new_tokens.append(token)

    i += 1

  return np.array(new_tokens, dtype=np.int32)


def get_dummy_image_shape_for_init_qwen3_omni(batch_size):
  """Return the shape of the dummy image for Qwen3-Omni model's initialization."""
  image_shape = (
      batch_size,
      mm_utils.NUM_IMAGE_CHANNELS,
      QWEN3_TEMPORAL_PATCH_SIZE,
      QWEN3_OMNI_IMAGE_SIZE,  # image_size_for_vit (height)
      QWEN3_OMNI_IMAGE_SIZE,  # video_num_frames
  )
  return image_shape


def get_dummy_audio_shape_for_init_qwen3_omni(config):
  """Return the shape of the dummy audio for Qwen3-Omni model's initialization."""
  # Audio shape: (batch, num_mel_bins, audio_length)
  # audio_length must be divisible by n_window * 2
  chunk_size = config.n_window_for_audio * 2
  audio_length = chunk_size * 4  # 4 chunks
  audio_shape = (config.micro_batch_size_to_train_on, config.num_mel_bins_for_audio, audio_length)
  return audio_shape


# ==============================================================================
# Qwen3-Omni Multimodal Position ID Utilities
# ==============================================================================
def _get_feat_extract_output_lengths(input_lengths: jax.Array) -> jax.Array:
  """Computes the output length of the audio encoder convolutional layers.

  The audio encoder processes audio in chunks of 100 samples, applying
  multiple convolutional layers with stride 2. Each full 100-sample chunk
  produces 13 output tokens, and remaining samples are processed separately.

  Args:
    input_lengths: Input audio sequence lengths. Shape: (batch,) or scalar.

  Returns:
    Output sequence lengths after audio encoding. Same shape as input.
  """
  input_lengths_leave = input_lengths % 100
  feat_lengths = (input_lengths_leave - 1) // 2 + 1
  output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
  return output_lengths


def get_llm_pos_ids_for_vision(
    start_idx: int | jax.Array,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: jax.Array,
    grid_hs: jax.Array,
    grid_ws: jax.Array,
) -> jax.Array:
  """Computes 3D position IDs (temporal, height, width) for vision tokens.

  Creates position embeddings for a grid of vision tokens representing an
  image or video. For each temporal frame, generates a spatial grid of
  (height, width) positions.

  Args:
    start_idx: Starting position ID value to add as offset.
    vision_idx: Index of the current image/video being processed.
    spatial_merge_size: Number of patches merged spatially (e.g., 2 means 2x2 patches → 1 token).
    t_index: Temporal position for each frame. Shape: (num_frames,).
    grid_hs: Height dimensions for all images/videos. Shape: (num_images,).
    grid_ws: Width dimensions for all images/videos. Shape: (num_images,).

  Returns:
    3D position IDs with shape (3, num_vision_tokens) where:
      - dim 0: temporal positions
      - dim 1: height positions
      - dim 2: width positions

  Example:
    If spatial_merge_size=2, grid_h=4, grid_w=4, num_frames=2:
      - After merge: 2x2 grid per frame
      - Total tokens: 2 frames x 2 x 2 = 8 tokens
      - Output shape: (3, 8)
      - t_index: [0, 0, 0, 0, 50, 50, 50, 50]
      - h_index: [0, 0, 1, 1, 0, 0, 1, 1]
      - w_index: [0, 1, 0, 1, 0, 1, 0, 1]
  """
  # Calculate effective spatial dimensions after merging patches
  llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
  llm_grid_w = grid_ws[vision_idx] // spatial_merge_size

  # Create height indices: [0, 0, ..., 0 (W times), 1, 1, ..., 1 (W times), ...]
  # Shape: (num_frames, llm_grid_h, 1) -> expand -> (num_frames, llm_grid_h, llm_grid_w) -> flatten
  h_index = jnp.arange(llm_grid_h).reshape(1, -1, 1).repeat(len(t_index), axis=0).repeat(llm_grid_w, axis=2).flatten()

  # Create width indices: [0, 1, 2, ..., W-1, 0, 1, 2, ..., W-1, ...]
  # Shape: (num_frames, 1, llm_grid_w) -> expand -> (num_frames, llm_grid_h, llm_grid_w) -> flatten
  w_index = jnp.arange(llm_grid_w).reshape(1, 1, -1).repeat(len(t_index), axis=0).repeat(llm_grid_h, axis=1).flatten()

  # Create temporal indices: [t0, t0, ..., t0 (HxW times), t1, t1, ..., t1 (HxW times), ...]
  # Shape: (num_frames, 1) -> expand -> (num_frames, llm_grid_h * llm_grid_w) -> flatten
  t_index_expanded = t_index.reshape(-1, 1).repeat(llm_grid_h * llm_grid_w, axis=1).flatten()

  # Stack all three dimensions and add starting offset
  _llm_pos_ids = jnp.stack([t_index_expanded, h_index, w_index])
  llm_pos_ids = _llm_pos_ids + start_idx

  return llm_pos_ids


def get_chunked_index(token_indices: jax.Array, tokens_per_chunk: int, remove_index: int) -> list[tuple[int, int]]:
  """Splits token index list into chunks based on token value ranges.

  Given a list of monotonically increasing token indices, returns a list of
  (start, end) index tuples representing slices where token values fall within
  successive ranges of `tokens_per_chunk`.

  Args:
    token_indices: Monotonically increasing array of token index values. Shape: (seq_len,).
    tokens_per_chunk: Chunk size threshold (e.g., 100 means first chunk has values < 100).
    remove_index: Offset to subtract from token_indices before chunking.

  Returns:
    List of (start_idx, end_idx) tuples, each representing a chunk.

  Example:
    token_indices = [5, 10, 52, 105, 150, 250]
    tokens_per_chunk = 100
    remove_index = 0

    Result: [(0, 3), (3, 5), (5, 6)]
      - Chunk 0: indices 0-3 (values 5, 10, 52 are < 100)
      - Chunk 1: indices 3-5 (values 105, 150 are >= 100 and < 200)
      - Chunk 2: indices 5-6 (value 250 is >= 200)
  """
  chunks = []
  i = 0
  start_idx = 0
  current_chunk = 1

  while i < len(token_indices):
    if token_indices[i] - remove_index >= current_chunk * tokens_per_chunk:
      chunks.append((start_idx, i))
      start_idx = i
      current_chunk += 1
    i += 1

  # Append final chunk
  chunks.append((start_idx, len(token_indices)))

  return chunks


def get_rope_index(
    input_ids: np.ndarray,
    image_grid_thw: np.ndarray | None = None,
    video_grid_thw: np.ndarray | None = None,
    attention_mask: np.ndarray | None = None,
    use_audio_in_video: bool = False,
    audio_lengths: np.ndarray | None = None,
    second_per_grids: np.ndarray | None = None,
    spatial_merge_size: int = 2,
    position_id_per_seconds: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
  """Calculate 3D RoPE position indices for multimodal sequences.

  This function computes position IDs that encode both sequential (text) and
  spatial-temporal (vision/audio) structure for Qwen3-Omni multimodal inputs.

  For pure text sequences:
    - All 3 dimensions receive the same sequential positions: [0, 1, 2, 3, 4]

  For multimodal sequences with vision:
    - Vision tokens get 3D positions (temporal, height, width)
    - Text tokens continue sequentially from max(vision_pos) + 1
    - Example with video (3 temporal patches, 2x2 spatial):
        Vision temporal: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
        Vision height:   [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        Vision width:    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        Text positions:  [101, 102, 103, 104, 105]

  Args:
    input_ids: Input token IDs. Shape: (batch, seq_len).
    image_grid_thw: Image dimensions (temporal, height, width). Shape: (num_images, 3).
    video_grid_thw: Video dimensions (temporal, height, width). Shape: (num_videos, 3).
    attention_mask: Padding mask (1 = real token, 0 = padding). Shape: (batch, seq_len).
    use_audio_in_video: If True, audio tokens are interleaved with video tokens.
    audio_lengths: Audio sequence lengths. Shape: (num_audios,).
    second_per_grids: Time interval per temporal grid (for videos). Shape: (num_videos,).
    spatial_merge_size: Number of patches merged spatially (e.g., 2 for 2x2→1).
    position_id_per_seconds: Temporal granularity (tokens per second, typically 25).

  Returns:
    A tuple of:
      - position_ids: 3D position IDs. Shape: (3, batch, seq_len).
      - mrope_position_deltas: Position offset for each sequence. Shape: (batch, 1).

  Raises:
    ValueError: If multimodal tokens are present but grid info is missing.
  """
  batch_size, seq_len = input_ids.shape

  # Handle text-only case (no multimodal content)
  if image_grid_thw is None and video_grid_thw is None:
    if attention_mask is None:
      attention_mask = np.ones_like(input_ids)

    # Create sequential 1D positions
    position_ids = np.cumsum(attention_mask.astype(np.float32), axis=-1) - 1
    position_ids = np.where(attention_mask == 0, 1.0, position_ids)

    # Expand to 3D (same value in all dimensions for text-only)
    position_ids = np.broadcast_to(position_ids[np.newaxis, :, :], (3, batch_size, seq_len))

    # Calculate deltas for each sequence
    max_position_ids = np.max(position_ids, axis=(0, 2), keepdims=True).transpose(1, 0, 2)  # (batch, 1, 1)
    mrope_position_deltas = max_position_ids.squeeze(-1) + 1 - np.sum(attention_mask, axis=-1, keepdims=True)

    return position_ids, mrope_position_deltas

  # Multimodal case: process each sequence in batch
  if attention_mask is None:
    attention_mask = np.ones_like(input_ids)

  attention_mask_bool = attention_mask == 1
  position_ids = np.zeros((3, batch_size, seq_len), dtype=jnp.float32)
  mrope_position_deltas = []

  # Process each sequence in the batch
  for i in range(batch_size):
    # Get valid tokens (non-padding)
    valid_input_ids = input_ids[i][attention_mask_bool[i]]

    # Count multimodal elements in this sequence
    vision_start_indices = np.where(valid_input_ids == QWEN3_OMNI_VISION_START_TOKEN)[0]
    vision_tokens = valid_input_ids[vision_start_indices + 1] if len(vision_start_indices) > 0 else np.array([])

    audio_nums = np.sum(valid_input_ids == QWEN3_OMNI_AUDIO_START_TOKEN).item()
    image_nums = np.sum(vision_tokens == QWEN3_OMNI_IMAGE_TOKEN).item() if len(vision_tokens) > 0 else 0
    video_nums = (
        (
            np.sum(vision_tokens == QWEN3_OMNI_AUDIO_START_TOKEN).item()
            if use_audio_in_video
            else np.sum(vision_tokens == QWEN3_OMNI_VIDEO_TOKEN).item()
        )
        if len(vision_tokens) > 0
        else 0
    )

    input_tokens = valid_input_ids.tolist()
    llm_pos_ids_list = []
    st = 0
    remain_images = image_nums
    remain_videos = video_nums
    remain_audios = audio_nums
    image_idx = 0
    video_idx = 0
    audio_idx = 0

    multimodal_nums = image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums

    # Process each multimodal element
    for _ in range(multimodal_nums):
      st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0

      # Find next vision or audio start token
      if (QWEN3_OMNI_IMAGE_TOKEN in input_tokens or QWEN3_OMNI_VIDEO_TOKEN in input_tokens) and (
          remain_videos > 0 or remain_images > 0
      ):
        try:
          ed_vision_start = input_tokens.index(QWEN3_OMNI_VISION_START_TOKEN, st)
        except ValueError:
          ed_vision_start = len(input_tokens) + 1
      else:
        ed_vision_start = len(input_tokens) + 1

      if QWEN3_OMNI_AUDIO_TOKEN in input_tokens and remain_audios > 0:
        try:
          ed_audio_start = input_tokens.index(QWEN3_OMNI_AUDIO_START_TOKEN, st)
        except ValueError:
          ed_audio_start = len(input_tokens) + 1
      else:
        ed_audio_start = len(input_tokens) + 1

      min_ed = min(ed_vision_start, ed_audio_start)

      # Add text tokens before multimodal element
      text_len = min_ed - st
      if text_len > 0:
        text_pos = np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx
        llm_pos_ids_list.append(text_pos)
        st_idx += text_len

      # Determine BOS/EOS token lengths
      if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
        bos_len, eos_len = 2, 2  # Audio in video
      else:
        bos_len, eos_len = 1, 1

      # Add BOS token(s)
      bos_pos = np.arange(bos_len).reshape(1, -1).repeat(3, axis=0) + st_idx
      llm_pos_ids_list.append(bos_pos)
      st_idx += bos_len

      # Process modality-specific content
      # Audio Only
      if min_ed == ed_audio_start:
        audio_len = _get_feat_extract_output_lengths(audio_lengths[audio_idx]).item()
        audio_pos = np.arange(audio_len).reshape(1, -1).repeat(3, axis=0) + st_idx
        llm_pos_ids_list.append(audio_pos)

        st += int(text_len + bos_len + audio_len + eos_len)
        audio_idx += 1
        remain_audios -= 1

      # Image Only
      elif min_ed == ed_vision_start and input_tokens[ed_vision_start + 1] == QWEN3_OMNI_IMAGE_TOKEN:
        grid_t = image_grid_thw[image_idx, 0].item()
        grid_hs = image_grid_thw[:, 1]
        grid_ws = image_grid_thw[:, 2]
        t_index = np.arange(grid_t, dtype=np.float32) * 1 * position_id_per_seconds

        image_pos = get_llm_pos_ids_for_vision(st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws)
        llm_pos_ids_list.append(image_pos)

        image_len = int(np.prod(image_grid_thw[image_idx]).item() // (spatial_merge_size**2))
        st += int(text_len + bos_len + image_len + eos_len)
        image_idx += 1
        remain_images -= 1

      # Video Only
      elif min_ed == ed_vision_start and input_tokens[ed_vision_start + 1] == QWEN3_OMNI_VIDEO_TOKEN:
        grid_t = video_grid_thw[video_idx, 0].item()
        grid_hs = video_grid_thw[:, 1]
        grid_ws = video_grid_thw[:, 2]
        t_index = np.arange(grid_t, dtype=np.float32) * second_per_grids[video_idx].item() * position_id_per_seconds

        video_pos = get_llm_pos_ids_for_vision(st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws)
        llm_pos_ids_list.append(video_pos)

        video_len = int(np.prod(video_grid_thw[video_idx]).item() // (spatial_merge_size**2))
        st += int(text_len + bos_len + video_len + eos_len)
        video_idx += 1
        remain_videos -= 1

      # Audio in Video (interleaved)
      elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
        audio_len = _get_feat_extract_output_lengths(audio_lengths[audio_idx]).item()
        audio_llm_pos_ids = np.arange(audio_len).reshape(1, -1).repeat(3, axis=0) + st_idx

        grid_t = video_grid_thw[video_idx, 0].item()
        grid_hs = video_grid_thw[:, 1]
        grid_ws = video_grid_thw[:, 2]
        t_index = np.arange(grid_t, dtype=np.float32) * second_per_grids[video_idx].item() * position_id_per_seconds

        video_llm_pos_ids = get_llm_pos_ids_for_vision(st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws)

        # Interleave audio and video based on temporal ordering
        video_data_index = 0
        audio_data_index = 0
        while video_data_index < video_llm_pos_ids.shape[1] and audio_data_index < audio_llm_pos_ids.shape[1]:
          if video_llm_pos_ids[0, video_data_index] <= audio_llm_pos_ids[0, audio_data_index]:
            llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_data_index + 1])
            video_data_index += 1
          else:
            llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1])
            audio_data_index += 1

        # Append remaining tokens
        if video_data_index < video_llm_pos_ids.shape[1]:
          llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index:])
        if audio_data_index < audio_llm_pos_ids.shape[1]:
          llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index:])

        video_len = int(np.prod(video_grid_thw[video_idx]).item() // (spatial_merge_size**2))
        st += int(text_len + bos_len + audio_len + video_len + eos_len)

        audio_idx += 1
        video_idx += 1
        remain_videos -= 1
        remain_audios -= 1

      # Add EOS token(s)
      st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0
      eos_pos = np.arange(eos_len).reshape(1, -1).repeat(3, axis=0) + st_idx
      llm_pos_ids_list.append(eos_pos)

    # Add any remaining text tokens
    if st < len(input_tokens):
      st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0
      text_len = len(input_tokens) - st
      remaining_text_pos = np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx
      llm_pos_ids_list.append(remaining_text_pos)

    # Concatenate all position IDs for this sequence
    llm_positions = np.concatenate(llm_pos_ids_list, axis=1)

    # Place into position_ids tensor at valid positions
    valid_positions = np.where(attention_mask_bool[i])[0]
    position_ids[:, i, valid_positions] = llm_positions

    # Calculate delta for this sequence
    mrope_position_deltas.append(llm_positions.max().item() + 1 - len(valid_input_ids))

  mrope_position_deltas = np.array(mrope_position_deltas).reshape(batch_size, 1)

  return position_ids, mrope_position_deltas
