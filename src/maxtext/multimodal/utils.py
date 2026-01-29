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

"""General utility functions for multimodal processing."""

import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image

try:
  import librosa  # pytype: disable=import-error
except ImportError:
  librosa = None

NUM_IMAGE_CHANNELS = 3  # RGB


@dataclass
class PreprocessorOutput:
  """Holds the output of a multimodal preprocessor.

  Args:
    pixel_values: A JAX array containing the processed image pixel data.
                  The shape and format depend on the specific model and
                  preprocessing steps (e.g., [H, W, C] for Gemma3 or
                  [NUM_TILES, C, TILE_SIZE, TILE_SIZE] for Llama4).
    pixel_mask: An optional JAX array of shape (NUM_TILES,) indicating valid
                tiles in the image.
    aspect_ratios: An optional JAX array of shape (batch_size, 2) representing
                   the aspect ratio [ratio_h, ratio_w] of the processed image(s).
                   This is particularly relevant for models like Llama4 that handle
                   images by tiling.
    num_images: Number of images in the output.
    audio_values: An optional array containing processed audio features.
    audio_mask: An optional array indicating valid audio segments.
  """

  pixel_values: None | np.ndarray = None
  pixel_mask: None | np.ndarray = None
  aspect_ratios: None | np.ndarray = None
  num_images: int = 0
  # Audio attributes.
  audio_values: None | np.ndarray = None
  audio_mask: None | np.ndarray = None


def convert_to_RGB(image):
  """Convert image to RGB format."""
  if image.mode != "RGB":
    image = image.convert("RGB")
  return image


def load_image_from_path(image_path):
  """Loads an image from a given file path and returns a np.array."""
  if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image not found at path {image_path}. Please specify a valid image path")
  try:
    image = Image.open(image_path).convert("RGB")
    image.load()  # Load image data to catch errors early
    return np.asarray(image)

  except (IOError, OSError) as e:
    raise IOError(f"Error loading image from {image_path}") from e


def normalize_images(images, mean, std):
  """Normalize the image by subtracting mean and dividing by std.

  Args:
    images: The images to normalize (np.ndarray).
    mean: Scalar float or tuple/array of floats for per-channel normalization.
          E.g., 127.5 or (127.5, 127.5, 127.5) for RGB.
    std: Scalar float or tuple/array of floats for per-channel normalization.
         E.g., 127.5 or (127.5, 127.5, 127.5) for RGB.
  Returns:
    The normalized images (modifies input in-place and returns it).
  """
  images -= np.asarray(mean)
  images /= np.asarray(std)
  return images


def merge_mm_embeddings(
    text_embeddings: np.ndarray | jnp.ndarray,
    multimodal_embeddings: np.ndarray | jnp.ndarray,
    mask,
    token_masks: np.ndarray | jnp.ndarray | None = None,
) -> np.ndarray | jnp.ndarray:
  """Merges text and multimodal (vision/audio) embeddings based on a mask.

  This function handles two primary formats for multimodal embeddings:
  1. Tiled Format (e.g., Llama4 vision): Embeddings are provided as a batch of
     images and their tiles, with shape (B * N, T, K, D). These are flattened
     into a single sequence of tokens per batch item.
  2. Simple Format (e.g., Gemma3 vision, Qwen3-Omni audio): Embeddings are provided as
     (B, N, K, D) and are flattened into a sequence of tokens.

  Args:
    text_embeddings: (B, S, D) array of text embeddings.
    multimodal_embeddings: Multimodal embeddings (vision/audio) in one of two formats:
      - (B * N, T, K, D) for tiled inputs.
      - (B, N, K, D) for simple inputs.
      (B=batch_size, S=seq_len, D=embedding_dim, N=num_images/audio_chunks,
       T=num_tiles, K=toks_per_image/audio_chunk)
    mask: (B, S) boolean or integer array where non-zero positions
      indicate where multimodal embeddings should be placed.
    token_masks: (Optional) A mask for the multimodal tokens.
      - (B * N, T) for tiled inputs, indicating valid tiles.
      - If None, all multimodal embeddings are assumed to be valid.

  Returns:
    A (B, S, D) array of merged embeddings.
  """
  # Input Validation and Shape Unpacking
  batch_size, _, d_model = text_embeddings.shape
  # The number of tokens per image/tile/audio_chunk is the second to last dimension.
  num_toks_per_token = multimodal_embeddings.shape[-2]

  if d_model != multimodal_embeddings.shape[-1]:
    raise ValueError(
        "Embedding dimension mismatch between text and multimodal embeddings:"
        f" {d_model} vs {multimodal_embeddings.shape[-1]}"
    )

  # Reshape Multimodal Embeddings to a unified (B, S_mm, D) format
  # This single reshape robustly handles both documented cases:
  # Case 1: (B * N, T, K, D) -> (B, N*T*K, D)
  # Case 2: (B, N, K, D) -> (B, N*K, D)
  flat_multimodal_embeddings = multimodal_embeddings.reshape(batch_size, -1, d_model)

  # Process Optional Token Masks
  flat_token_masks_processed = None
  if token_masks is not None:
    # Handle the tiled case where token_masks batch dimension is (B * N)
    if token_masks.shape[0] != batch_size:
      if token_masks.shape[0] % batch_size != 0:
        raise ValueError(
            "Batch dimension of token_masks must be a multiple of the text"
            f" batch size. Got {token_masks.shape[0]} and {batch_size}."
        )
      # Reshape from (B * N, T) to (B, N * T)
      flat_tile_masks = token_masks.reshape(batch_size, -1)
    else:
      # This handles cases where token_masks is already (B, ...)
      flat_tile_masks = token_masks.reshape(batch_size, -1)

    # Expand the tile-level mask to a token-level mask to match the embeddings.
    # A mask of shape (B, N*T) becomes (B, N*T*K) by repeating each element K times.
    flat_token_masks_processed = jnp.repeat(flat_tile_masks, repeats=num_toks_per_token, axis=1)

  # Vmap the inner merge function over the batch dimension
  return jax.vmap(
      _merge_mm_embeddings_inner,  # Assumes this function is defined elsewhere
      in_axes=(0, 0, 0, None if flat_token_masks_processed is None else 0),
  )(text_embeddings, flat_multimodal_embeddings, mask, flat_token_masks_processed)


def _merge_mm_embeddings_inner(
    text_embeddings: jnp.ndarray,
    multimodal_embeddings: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    token_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """`merge_mm_embeddings` without batch dimension."""

  if token_mask is not None:
    # This logic packs valid multimodal tokens to the front of the array.
    # It correctly handles cases where some multimodal tokens are just padding.
    sort_indices = jnp.argsort(-token_mask)  # Sorts descending, putting 1s first
    multimodal_embeddings = multimodal_embeddings[sort_indices]

  # Find positions in the text sequence to place the multimodal embeddings.
  # The `size` argument ensures a fixed shape for JIT compilation.
  target_pos = jnp.nonzero(mask, size=multimodal_embeddings.shape[0])
  target_pos = target_pos[0]  # jnp.nonzero returns a tuple of arrays

  # Save the embedding at the first position.
  first_pos_embedding = text_embeddings[0]

  # Perform the insertion.
  merged = text_embeddings.at[target_pos, :].set(multimodal_embeddings)

  # Restore the first position's embedding, in case it was overwritten.
  merged = merged.at[0].set(first_pos_embedding)

  return merged


# Following audio functions derived from the HuggingFace implementation
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
    dtype: type = np.float32,
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


def load_audio(data_path: str, sample_rate: int = 16000) -> np.ndarray:
  """Load audio from a file path.

  Args:
      data_path (str): The path to the audio file or video file.
      sample_rate (int): The target sample rate in Hz. Default is 16000.

  Returns:
      np.ndarray: The loaded audio waveform.

  Raises:
      FileNotFoundError: If the audio file does not exist.
      RuntimeError: If the audio file cannot be loaded.
  """
  if not os.path.isfile(data_path):
    raise FileNotFoundError(f"Audio file not found at path {data_path}. Please specify a valid audio file path")
  if librosa is None:
    raise ImportError("librosa is required for audio processing but not installed.")
  try:
    audio = librosa.load(data_path, sr=sample_rate)[0]
    return audio
  except Exception as e:
    raise RuntimeError(f"Failed to load audio from {data_path}: {e}") from e
