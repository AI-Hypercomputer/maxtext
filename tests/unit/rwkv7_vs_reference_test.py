# Copyright 2026 Google LLC
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

"""RWKV-7 against a NumPy reference of BlinkDL's model.

A BlinkDL-format state dict goes through the RWKV-7 converter's mapping and
writer, then MaxText's model loader (`load_parameters_path`), for every WKV
implementation. The WKV kernels' own math (resets, gradients, chunking) is
tested in `rwkv7_wkv_kernel_test.py`.
"""

import numpy as np
import pytest

from maxtext.checkpoint_conversion.standalone_scripts import convert_rwkv7_unscanned
from maxtext.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path

# ----------------------------------------------------------------------
# NumPy reference of RWKV-7 (x070), float64, sequence mode.
# Adapted from the non-CUDA path of BlinkDL's RWKV-LM (Apache-2.0),
# https://github.com/BlinkDL/RWKV-LM/blob/9a75f9f037afa4418ee6283b584b92b1adb89ca1/RWKV-v7/rwkv_v7_demo.py
# Parameters use BlinkDL's checkpoint names; `nn.Linear` weights are (out, in).
# ----------------------------------------------------------------------


def _layer_norm(x, sd, prefix, eps=1e-5):
  x = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)
  return x * sd[prefix + "weight"] + sd[prefix + "bias"]


def _sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))


def _shift(x):
  """The previous token's activations (zeros before the first)."""
  return np.concatenate([np.zeros_like(x[:, :1]), x[:, :-1]], axis=1)


def _time_mix(sd, p, x, v_first, layer, head_size):
  """One layer's time mixing (`RWKV_Tmix_x070`); returns `(output, v_first)`."""
  batch, seq_len, emb = x.shape
  heads = emb // head_size

  def vec(name):
    return sd[p + name].reshape(emb)

  xr, xw, xk, xv, xa, xg = (x + (_shift(x) - x) * vec(n) for n in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"))

  r = xr @ sd[p + "receptance.weight"].T
  w = -np.logaddexp(0.0, -(vec("w0") + np.tanh(xw @ sd[p + "w1"]) @ sd[p + "w2"])) - 0.5
  w = np.exp(-np.exp(w))
  k = xk @ sd[p + "key.weight"].T
  v = xv @ sd[p + "value.weight"].T
  if layer == 0:
    v_first = v
  else:
    v = v + (v_first - v) * _sigmoid(vec("v0") + (xv @ sd[p + "v1"]) @ sd[p + "v2"])
  a = _sigmoid(vec("a0") + (xa @ sd[p + "a1"]) @ sd[p + "a2"])
  g = _sigmoid(xg @ sd[p + "g1"]) @ sd[p + "g2"]
  kk = (k * vec("k_k")).reshape(batch, seq_len, heads, head_size)
  kk = kk / np.maximum(np.linalg.norm(kk, axis=-1, keepdims=True), 1e-12)
  k = k * (1 + (a - 1) * vec("k_a"))

  r_h, w_h, k_h, v_h, a_h = (t.reshape(batch, seq_len, heads, head_size) for t in (r, w, k, v, a))
  state = np.zeros((batch, heads, head_size, head_size))
  out = np.zeros_like(r_h)
  for t in range(seq_len):
    # S = S diag(w) + (S @ -kk) (kk * a)^T + v k^T;  y = S @ r
    sab = np.einsum("bhij,bhj->bhi", state, -kk[:, t])
    state = (
        state * w_h[:, t, :, None, :]
        + sab[..., None] * (kk[:, t] * a_h[:, t])[:, :, None, :]
        + v_h[:, t, :, :, None] * k_h[:, t, :, None, :]
    )
    out[:, t] = np.einsum("bhij,bhj->bhi", state, r_h[:, t])

  # GroupNorm over each head (eps 64e-5), then the r·k bonus.
  out = (out - out.mean(-1, keepdims=True)) / np.sqrt(out.var(-1, keepdims=True) + 64e-5)
  out = out * sd[p + "ln_x.weight"].reshape(heads, head_size) + sd[p + "ln_x.bias"].reshape(heads, head_size)
  out = out + np.sum(r_h * k_h * sd[p + "r_k"], axis=-1, keepdims=True) * v_h
  return (out.reshape(batch, seq_len, emb) * g) @ sd[p + "output.weight"].T, v_first


def _channel_mix(sd, p, x):
  k = x + (_shift(x) - x) * sd[p + "x_k"].reshape(-1)
  return np.square(np.maximum(k @ sd[p + "key.weight"].T, 0.0)) @ sd[p + "value.weight"].T


def reference_logits(sd, tokens, head_size):
  """Logits `(B, T, vocab)` for `tokens` `(B, T)`, each row from a zero state."""
  n_layer = 1 + max(int(name.split(".")[1]) for name in sd if name.startswith("blocks."))
  x = sd["emb.weight"][np.asarray(tokens)]
  v_first = None
  for i in range(n_layer):
    p = f"blocks.{i}."
    if i == 0:
      x = _layer_norm(x, sd, p + "ln0.")
    att, v_first = _time_mix(sd, p + "att.", _layer_norm(x, sd, p + "ln1."), v_first, i, head_size)
    x = x + att
    x = x + _channel_mix(sd, p + "ffn.", _layer_norm(x, sd, p + "ln2."))
  return _layer_norm(x, sd, "ln_out.") @ sd["head.weight"].T


# ----------------------------------------------------------------------
# END: NumPy reference
# ----------------------------------------------------------------------

EMB, HEAD_SIZE, LAYERS, VOCAB, RANK = 64, 16, 2, 128, 8
TOL = 1e-4  # relative to the largest logit; float32 vs. the float64 reference


def random_state_dict(seed=0):
  """A BlinkDL-format state dict of small random values (checkpoints aren't zeros)."""
  rng = np.random.default_rng(seed)
  heads, shapes = EMB // HEAD_SIZE, {"emb.weight": (VOCAB, EMB), "head.weight": (VOCAB, EMB)}
  shapes.update({f"ln_out.{n}": (EMB,) for n in ("weight", "bias")})
  for i in range(LAYERS):
    p = f"blocks.{i}."
    for ln in ("ln0", "ln1", "ln2") if i == 0 else ("ln1", "ln2"):
      shapes.update({f"{p}{ln}.weight": (EMB,), f"{p}{ln}.bias": (EMB,)})
    for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "a0", "v0", "k_k", "k_a"):
      shapes[f"{p}att.{name}"] = (1, 1, EMB)
    for name in ("w", "a", "v", "g"):
      shapes.update({f"{p}att.{name}1": (EMB, RANK), f"{p}att.{name}2": (RANK, EMB)})
    shapes[f"{p}att.r_k"] = (heads, HEAD_SIZE)
    for name in ("receptance", "key", "value", "output"):
      shapes[f"{p}att.{name}.weight"] = (EMB, EMB)
    shapes.update({f"{p}att.ln_x.weight": (EMB,), f"{p}att.ln_x.bias": (EMB,), f"{p}ffn.x_k": (1, 1, EMB)})
    shapes.update({f"{p}ffn.key.weight": (4 * EMB, EMB), f"{p}ffn.value.weight": (EMB, 4 * EMB)})
  return {name: rng.uniform(-0.5, 0.5, shape) for name, shape in shapes.items()}


def config(**overrides):
  """The shipped `rwkv7-0.1b` config at tiny dims, float32 throughout."""
  args = {
      "run_name": "rwkv7_vs_reference",
      "model_name": "rwkv7-0.1b",
      "override_model_config": True,
      "base_emb_dim": EMB,
      "base_mlp_dim": 4 * EMB,
      "base_num_decoder_layers": LAYERS,
      "base_num_query_heads": EMB // HEAD_SIZE,
      "base_num_kv_heads": EMB // HEAD_SIZE,
      "head_dim": HEAD_SIZE,
      "rwkv7_head_size": HEAD_SIZE,
      "vocab_size": VOCAB,
      **{f"rwkv7_{gate}_lora_rank": RANK for gate in ("decay", "iclr", "value", "gate")},
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "skip_jax_distributed_system": True,
      **overrides,
  }
  return pyconfig.initialize([None, get_test_config_path()] + [f"{k}={v}" for k, v in args.items()])


def max_rel_dev(got, want):
  got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
  return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


@pytest.fixture(name="checkpoint", scope="module")
def fixture_checkpoint(tmp_path_factory):
  """`(state_dict, load_parameters_path)`: random weights written by the converter's own writer."""
  state_dict = random_state_dict()
  out = str(tmp_path_factory.mktemp("rwkv7") / "ckpt")
  save_weights_to_checkpoint(out, convert_rwkv7_unscanned.convert_rwkv7_params(state_dict), 1, True, True)
  return state_dict, f"{out}/0/items"


@pytest.mark.parametrize("wkv_impl", ["naive", "pallas", "pallas_chunked"])
def test_loaded_checkpoint_matches_reference(checkpoint, wkv_impl):
  """Converted checkpoint -> MaxText's model loader -> logits equal the reference's."""
  state_dict, path = checkpoint
  cfg = config(load_parameters_path=path, rwkv7_wkv_impl=wkv_impl)
  model = model_creation_utils.from_pretrained(
      cfg, mesh=maxtext_utils.get_mesh_from_config(cfg), model_mode=MODEL_MODE_TRAIN
  )
  tokens = np.random.default_rng(1).integers(0, VOCAB, (2, 20)).astype(np.int32)  # one full kernel chunk plus a tail
  logits = model(
      decoder_input_tokens=tokens,
      decoder_positions=np.broadcast_to(np.arange(20), tokens.shape),
      enable_dropout=False,
      model_mode=MODEL_MODE_TRAIN,
  )
  assert max_rel_dev(logits, reference_logits(state_dict, tokens, HEAD_SIZE)) < TOL
