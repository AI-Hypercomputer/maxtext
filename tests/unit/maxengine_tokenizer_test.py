"""Verify custom-code opt-in without loading model weights or a tokenizer."""

# pylint: disable=protected-access

from types import SimpleNamespace
from unittest.mock import patch
import pytest
from maxtext.inference.maxengine import maxengine


@pytest.mark.parametrize("trusted", [False, True])
def test_hf_tokenizer_custom_code_opt_in(trusted):
  engine = SimpleNamespace(config=SimpleNamespace(tokenizer_trust_remote_code=trusted))
  metadata = SimpleNamespace(
      tokenizer_type=maxengine.TokenizerType.huggingface, path="/tmp/custom-model", access_token=""
  )
  hf = SimpleNamespace(pad_token_id=None, unk_token_id=1, eos_token_id=2)
  with (
      patch.object(maxengine._token_params_ns, "_IS_STUB", False, create=True),
      patch.object(maxengine.engine_api, "_IS_STUB", False, create=True),
      patch("transformers.AutoTokenizer.from_pretrained", return_value=hf) as loader,
  ):
    result = maxengine.MaxEngine.build_tokenizer(engine, metadata)
  expected = {"token": ""}
  if trusted:
    expected["trust_remote_code"] = True
  loader.assert_called_once_with(metadata.path, **expected)
  assert result.metadata is metadata
  assert result.tokenizer.pad_token_id == 1
