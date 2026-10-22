"""Ming audio shape and token handling tests without publisher HF dependencies."""

from types import SimpleNamespace
import numpy as np
import pytest
from maxtext.models import ming
from maxtext.multimodal import processor_ming


@pytest.mark.parametrize("length", [1, 17, 2999, 3000, 3001, 6001])
def test_projected_token_count(length):
  """Preprocessing and encoder/projector agree at chunk and stride boundaries."""
  expected = sum(
      ming.projector_out_length(ming.whisper_conv_out_length(chunk), 3)
      for chunk in ming.audio_chunk_lengths(length, 3000)
  )
  assert processor_ming.get_num_audio_tokens(length) == expected


def test_audio_token_expansion_and_offsets():
  output = SimpleNamespace(audio_lengths=np.array([3], dtype=np.int32))
  pad = processor_ming.AUDIO_PAD_ID
  tokens = np.array([11, pad, 12], dtype=np.int32)
  np.testing.assert_array_equal(processor_ming.add_extra_tokens_for_ming(tokens, output), [11, pad, pad, pad, 12])
  assert processor_ming.get_audio_offsets_ming(output) == 2
  with pytest.raises(ValueError, match="Found 0"):
    processor_ming.add_extra_tokens_for_ming(np.array([11, 12]), output)


def test_prompt_places_audio_before_text():
  prompt = processor_ming.reformat_prompt_ming("transcribe", num_audios=1)
  assert prompt.index(processor_ming.AUDIO_PAD_TOKEN) < prompt.index("transcribe")
  assert processor_ming.reformat_prompt_ming("<|audio|>transcribe", num_audios=1) == prompt
