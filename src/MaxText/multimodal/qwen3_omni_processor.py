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

import math
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
  import decord  # pytype: disable=import-error
except ImportError:
  decord = None

from MaxText import max_logging
from MaxText.multimodal import utils as mm_utils

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
