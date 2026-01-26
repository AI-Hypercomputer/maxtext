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


"""
To run the test
  # pip install torch numpy transformers sympy
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  python3 -m pytest -v --pyargs tests.unit.engram_vs_reference_test -rP -s

reference: https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py
"""


from typing import List, Dict, Any
from dataclasses import dataclass, field
import math
import os
import unittest
from absl.testing import parameterized

import numpy as np
from sympy import isprime

from tokenizers import normalizers, Regex
from transformers import AutoTokenizer
import torch
from torch import nn

from flax import nnx
import jax
import jax.numpy as jnp

from MaxText.layers.engram import Engram as EngramJAX
from MaxText.layers.engram import ShortConv as ShortConvJAX
from MaxText.layers.engram import MultiHeadEmbedding as MultiHeadEmbeddingJAX

from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText import pyconfig
from MaxText import maxtext_utils
from jax.sharding import Mesh


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

V = 129280
# V = 1000
hc_mult = 1
max_ngram_size = 2  # >= 2

@dataclass
class EngramConfig:
  tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
  engram_vocab_size: List[int] = field(default_factory=lambda: [V * 5, V * 5])
  max_ngram_size: int = max_ngram_size
  n_embed_per_ngram: int = 512
  n_head_per_ngram: int = 8
  layer_ids: List[int] = field(default_factory=lambda: [1, 15])
  pad_id: int = 2
  seed: int = 0
  kernel_size: int = 4

@dataclass
class BackBoneConfig:
  hidden_size: int = 1024
  hc_mult: int = hc_mult
  vocab_size: int = V
  num_layers: int = 30

engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()


# -----------------------------------------------------------------------------
# Torch Reference Implementation
# -----------------------------------------------------------------------------

class CompressedTokenizer:

  def __init__(
      self,
      tokenizer_name_or_path,
  ):
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

    SENTINEL = "\uE000"
    self.normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ]
    )

    self.lookup_table, self.num_new_token = self._build_lookup_table()

  def __len__(self):
    return self.num_new_token

  def _build_lookup_table(self):
    old2new = {}
    key2new = {}
    new_tokens = []

    vocab_size = len(self.tokenizer)
    for tid in range(vocab_size):
      text = self.tokenizer.decode([tid], skip_special_tokens=False)

      if "�" in text:
        key = self.tokenizer.convert_ids_to_tokens(tid)
      else:
        norm = self.normalizer.normalize_str(text)
        key = norm if norm else text

      nid = key2new.get(key)
      if nid is None:
        nid = len(new_tokens)
        key2new[key] = nid
        new_tokens.append(key)
      old2new[tid] = nid

    lookup = np.empty(vocab_size, dtype=np.int64)
    for tid in range(vocab_size):
      lookup[tid] = old2new[tid]

    return lookup, len(new_tokens)

  def _compress(self, input_ids):
    arr = np.asarray(input_ids, dtype=np.int64)
    pos_mask = arr >= 0
    out = arr.copy()
    valid_ids = arr[pos_mask]
    out[pos_mask] = self.lookup_table[valid_ids]
    return out

  def __call__(self, input_ids):
    return self._compress(input_ids)


class ShortConv(nn.Module):

  def __init__(
      self,
      hidden_size: int,
      kernel_size: int = 4,
      dilation: int = 1,
      norm_eps: float = 1e-5,
      hc_mult: int = 4,
      activation: bool = True,
  ):
    super().__init__()
    self.hc_mult = hc_mult
    self.activation = activation

    total_channels = hidden_size * hc_mult
    self.conv = nn.Conv1d(
        in_channels=total_channels,
        out_channels=total_channels,
        kernel_size=kernel_size,
        groups=total_channels,
        bias=False,
        padding=(kernel_size - 1) * dilation,
        dilation=dilation,
    )

    self.norms = nn.ModuleList([nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)])

    if self.activation:
      self.act_fn = nn.SiLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Input:  (B,L,HC_MULT,D)
    Output: (B,L,HC_MULT,D)
    """
    B, T, G, C = x.shape

    assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

    normed_chunks = []
    for i in range(G):
      chunk = x[:, :, i, :]
      normed_chunks.append(self.norms[i](chunk))

    x_norm = torch.cat(normed_chunks, dim=-1)
    x_bct = x_norm.transpose(1, 2)
    y_bct = self.conv(x_bct)
    y_bct = y_bct[..., :T]

    if self.activation:
      y_bct = self.act_fn(y_bct)
    y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()

    return y


def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

class NgramHashMapping:
    def __init__(
        self, 
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,  
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

class MultiHeadEmbedding(nn.Module):

  def __init__(self, list_of_N: List[int], D: int):
    super().__init__()
    self.num_heads = len(list_of_N)
    self.embedding_dim = D

    offsets = [0]
    for n in list_of_N[:-1]:
      offsets.append(offsets[-1] + n)

    self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

    total_N = sum(list_of_N)
    self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    shifted_input_ids = input_ids + self.offsets
    output = self.embedding(shifted_input_ids)

    return output


class Engram(nn.Module):

  def __init__(self, layer_id):
    super().__init__()
    self.layer_id = layer_id
    self.hash_mapping = NgramHashMapping(
        engram_vocab_size=engram_cfg.engram_vocab_size,
        max_ngram_size=engram_cfg.max_ngram_size,
        n_embed_per_ngram=engram_cfg.n_embed_per_ngram,
        n_head_per_ngram=engram_cfg.n_head_per_ngram,
        layer_ids=engram_cfg.layer_ids,
        tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
        pad_id=engram_cfg.pad_id,
        seed=engram_cfg.seed,
    )
    print(
        "DEBUG torch Layer Vocab Sizes", [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y]
    )
    self.multi_head_embedding = MultiHeadEmbedding(
        list_of_N=[x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
        D=engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
    )
    self.short_conv = ShortConv(
        hidden_size=backbone_config.hidden_size,
        kernel_size=engram_cfg.kernel_size,
        dilation=engram_cfg.max_ngram_size,
        hc_mult=backbone_config.hc_mult,
    )
    engram_hidden_size = (engram_cfg.max_ngram_size - 1) * engram_cfg.n_embed_per_ngram
    self.value_proj = nn.Linear(engram_hidden_size, backbone_config.hidden_size)
    self.key_projs = nn.ModuleList(
        [nn.Linear(engram_hidden_size, backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)]
    )
    self.norm1 = nn.ModuleList(
        [nn.RMSNorm(backbone_config.hidden_size, eps=1e-6) for _ in range(backbone_config.hc_mult)]
    )
    self.norm2 = nn.ModuleList(
        [nn.RMSNorm(backbone_config.hidden_size, eps=1e-6) for _ in range(backbone_config.hc_mult)]
    )

  def forward(self, hidden_states, input_ids):
    """
    hidden_states: [B, L, HC_MULT, D]
    input_ids: [B, L]
    """
    hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids)[self.layer_id])
    print("pt1", hash_input_ids)
    embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
    print("pt2", embeddings)
    gates = []
    for hc_idx in range(backbone_config.hc_mult):
      key = self.key_projs[hc_idx](embeddings)
      normed_key = self.norm1[hc_idx](key)
      print("pt3", normed_key)
      query = hidden_states[:, :, hc_idx, :]
      normed_query = self.norm2[hc_idx](query)
      print("pt4", normed_query)
      gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
      print("pt5", gate)
      gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
      print("pt6", gate)
      gate = gate.sigmoid().unsqueeze(-1)
      print("pt7", gate)
      gates.append(gate)
    gates = torch.stack(gates, dim=2)
    value = gates * self.value_proj(embeddings).unsqueeze(2)
    output = value + self.short_conv(value)
    return output


class TransformerBlock(nn.Module):

  def __init__(self, layer_id):
    super().__init__()
    self.attn = lambda x: x
    self.moe = lambda x: x
    self.engram = None
    if layer_id in engram_cfg.layer_ids:
      self.engram = Engram(layer_id=layer_id)

  def forward(self, input_ids, hidden_states):
    if self.engram is not None:
      hidden_states = self.engram(hidden_states=hidden_states, input_ids=input_ids) + hidden_states
    hidden_states = self.attn(hidden_states) + hidden_states
    hidden_states = self.moe(hidden_states) + hidden_states
    return hidden_states


# -----------------------------------------------------------------------------
# Test JAX Module: Helper
# -----------------------------------------------------------------------------


"""
Tests for Engram: N-gram Hashing and Injection Layer
"""


def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  return jnp.asarray(pt_tensor.detach().cpu().numpy())


def init_torch_weights(module, std=1):
  """
  Initialize all parameters in the module with N(0,std).
  This simple strategy is intended only for unit test.
  """
  with torch.no_grad():
    for _, param in module.named_parameters():
      torch.nn.init.normal_(param, mean=0.0, std=std)


def get_cfg_and_mesh():  # config, run_name, dtype, batch_size, seq_len
  """Returns MaxText configuration and mesh."""
  cfg = pyconfig.initialize(
      [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
      run_name="hi",
      # enable_checkpointing=False,
      # model_name="default",
      dtype="float32",
      # high precision
      weight_dtype="float32",
      matmul_precision="highest",
      float32_qk_product=True,
      float32_logits=True,
      # per_device_batch_size=batch_size,
      # max_target_length=seq_len,
      # max_prefill_predict_length=seq_len,
      # attention="dot_product",
      # **asdict(config),
  )
  devices_array = maxtext_utils.create_device_mesh(cfg)
  mesh = Mesh(devices_array, cfg.mesh_axes)
  return cfg, mesh


# -----------------------------------------------------------------------------
# Test JAX Module: MultiHeadEmbedding
# -----------------------------------------------------------------------------


def get_mhe_weights(pt_layer):
  """
  Extracts weights from PyTorch MultiHeadEmbedding.
  """
  return {
      "embedding": {"embedding": to_jax(pt_layer.embedding.weight)}
      # Note: 'offsets' is a Variable/Buffer, not a Param,
      # so we generally don't load it via update() unless we explicitly sync states.
      # In this specific module, offsets are derived deterministically from list_of_N
      # at __init__, so loading weights isn't strictly necessary for offsets
      # as long as list_of_N is the same.
  }


class MultiHeadEmbeddingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    np.random.seed(42)

  @parameterized.named_parameters(
      {"testcase_name": "basic", "vocab_sizes": [100, 200, 150], "dim": 32, "batch": 2, "seq_len": 10},
      {"testcase_name": "single_head", "vocab_sizes": [500], "dim": 64, "batch": 1, "seq_len": 5},
  )
  def test_mhe_equivalence(self, vocab_sizes, dim, batch, seq_len):
    num_heads = len(vocab_sizes)

    # 1. Init PyTorch
    MultiHeadEmbeddingPT = MultiHeadEmbedding
    pt_model = MultiHeadEmbeddingPT(vocab_sizes, dim)
    init_torch_weights(pt_model)
    pt_model.eval()

    # 2. Init JAX
    rngs = nnx.Rngs(params=0)
    # jax_model = MultiHeadEmbeddingJAX(vocab_sizes, dim, rngs)

    cfg, mesh = get_cfg_and_mesh()
    jax_model = MultiHeadEmbeddingJAX(vocab_sizes, dim, cfg, mesh, rngs)
    print(jax_model)

    # 3. Transfer Weights
    weights = get_mhe_weights(pt_model)
    nnx.update(jax_model, weights)

    # 4. Input Data
    # Input indices must be within the range of each specific head's vocab.
    # e.g., if vocab_sizes=[100, 200], input for head 0 must be < 100.
    input_np = np.zeros((batch, seq_len, num_heads), dtype=np.int32)
    for i, v_size in enumerate(vocab_sizes):
      input_np[:, :, i] = np.random.randint(0, v_size, (batch, seq_len))

    x_pt = torch.from_numpy(input_np).long()
    x_jax = jnp.array(input_np)

    # 5. Forward Pass
    with torch.no_grad():
      y_pt = pt_model(x_pt)

    y_jax = jax_model(x_jax)

    # 6. Debug / Verify Offsets
    # Check if JAX offsets match PT offsets
    jax_offsets = jax_model.offsets.value
    pt_offsets = pt_model.offsets.cpu().numpy()
    np.testing.assert_array_equal(jax_offsets, pt_offsets, err_msg="Offsets mismatch")

    # 7. Verify Output
    # Shape: (B, L, H, D)
    self.assertEqual(y_jax.shape, (batch, seq_len, num_heads, dim))

    diff = np.abs(y_jax - to_jax(y_pt)).max()
    print(f"\nTest: {self._testMethodName} | Max Diff: {diff:.6f}")

    np.testing.assert_allclose(
        y_jax, to_jax(y_pt), rtol=1e-5, atol=1e-5, err_msg=f"MultiHeadEmbedding output mismatch. Max Diff: {diff}"
    )


# -----------------------------------------------------------------------------
# Test JAX Module: ShortConv
# -----------------------------------------------------------------------------


# def get_shortconv_weights(pt_layer):
#   """
#   Extracts weights from PyTorch ShortConv and formats them for JAX ShortConv.
#   """
#   # 1. Conv Weights
#   # PyTorch Conv1d (Depthwise): (Out, 1, K) where groups=Out
#   # JAX Conv (General): (K, In, Out) -> For depthwise: (K, 1, Out)
#   # We permute (2, 1, 0): (Out, 1, K) -> (K, 1, Out)
#   conv_w = pt_layer.conv.weight.permute(2, 1, 0)

#   # 2. Norm Weights
#   # wrong: Norms are in a ModuleList, JAX expects a collection indexed by string "0", "1", etc.
#   norm_weights = {i: {"scale": to_jax(n.weight)} for i, n in enumerate(pt_layer.norms)}

#   return {"conv": {"kernel": to_jax(conv_w)}, "norms": norm_weights}


# nnx.List expects integer indices in the State dictionary.
# Use Integer keys (0, 1) instead of String keys ("0", "1")
def to_nnx_list_dict(weight_list):
  return {i: w for i, w in enumerate(weight_list)}


def get_shortconv_weights(pt_layer):
  conv_weight = pt_layer.conv.weight.permute(2, 1, 0)
  short_conv_norms = [{"scale": to_jax(n.weight)} for n in pt_layer.norms]
  return {"conv": {"kernel": to_jax(conv_weight)}, "norms": to_nnx_list_dict(short_conv_norms)}


class ShortConvTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    np.random.seed(42)

  @parameterized.named_parameters(
      {"testcase_name": "base", "hidden_size": 32, "hc_mult": 4, "kernel_size": 4, "dilation": 1},
      {"testcase_name": "dilated", "hidden_size": 16, "hc_mult": 2, "kernel_size": 3, "dilation": 2},
      {"testcase_name": "no_activation", "hidden_size": 32, "hc_mult": 4, "kernel_size": 4, "dilation": 1},
  )
  def test_shortconv_equivalence(self, hidden_size, hc_mult, kernel_size, dilation):
    batch_size = 2
    seq_len = 10
    activation = True

    # Handle the named parameter check for no_activation
    if "no_activation" in self._testMethodName:
      activation = False

    # 1. Init PyTorch
    ShortConvPT = ShortConv
    pt_model = ShortConvPT(hidden_size, kernel_size, dilation, hc_mult=hc_mult, activation=activation)
    init_torch_weights(pt_model)
    pt_model.eval()

    # 2. Init JAX
    rngs = nnx.Rngs(params=0)
    jax_model = ShortConvJAX(hidden_size, kernel_size, dilation, hc_mult=hc_mult, activation=activation, rngs=rngs)

    # 3. Transfer Weights
    weights = get_shortconv_weights(pt_model)
    nnx.update(jax_model, weights)

    # 4. Input Data
    # Shape: (B, L, G, C)
    x_pt = torch.randn(batch_size, seq_len, hc_mult, hidden_size)
    x_jax = to_jax(x_pt)

    # 5. Forward Pass
    with torch.no_grad():
      y_pt = pt_model(x_pt)

    y_jax = jax_model(x_jax)

    # 6. Verify
    # # Check output shape
    # self.assertEqual(y_jax.shape, (batch_size, seq_len, hc_mult, hidden_size))

    # # Check values
    # diff = np.abs(y_jax - to_jax(y_pt)).max()
    # print(f"\nTest: {self._testMethodName} | Max Diff: {diff:.6f}")

    # err_msg=f"ShortConv output mismatch. Max Diff: {diff}

    np.testing.assert_allclose(y_jax, to_jax(y_pt), rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------
# Test JAX Module: Engram
# -----------------------------------------------------------------------------


# Map Linear layers with Bias
def map_linear(pt_linear):
  d = {"kernel": to_jax(pt_linear.weight.T)}
  if pt_linear.bias is not None:
    d["bias"] = to_jax(pt_linear.bias)
  return d


def get_jax_engram_weights(pt_engram) -> dict:

  key_projs_weights = [map_linear(proj) for proj in pt_engram.key_projs]
  norm1_weights = [{"scale": to_jax(n.weight)} for n in pt_engram.norm1]
  norm2_weights = [{"scale": to_jax(n.weight)} for n in pt_engram.norm2]

  return {
      "mhe": get_mhe_weights(pt_engram.multi_head_embedding),
      "value_proj": map_linear(pt_engram.value_proj),
      "key_projs": to_nnx_list_dict(key_projs_weights),
      "norm1": to_nnx_list_dict(norm1_weights),
      "norm2": to_nnx_list_dict(norm2_weights),
      "short_conv": get_shortconv_weights(pt_engram.short_conv),
  }


class EngramTest(parameterized.TestCase):
  """Verifies JAX Engram implementation matches PyTorch reference."""

  def setUp(self):
    super().setUp()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    np.random.seed(42)

    self.batch_size = 2
    self.seq_len = 8
    self.layer_id = 1

    # Mocking Configs
    self.e_cfg = EngramConfig(layer_ids=[self.layer_id])
    self.b_cfg = BackBoneConfig()

    self.nnx_rng = nnx.Rngs(params=0)

  @parameterized.named_parameters(
      {"testcase_name": "standard_run", "batch_size": 2, "seq_len": 16},
  )
  def test_engram_match(self, batch_size, seq_len):
    # 1. Setup PyTorch Reference

    EngramPT = Engram
    pt_layer = EngramPT(layer_id=self.layer_id)
    init_torch_weights(pt_layer)
    pt_layer.eval()

    # "deepseek-ai/DeepSeek-V3"
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path, trust_remote_code=True)

    # 2. Setup JAX NNX Implementation
    cfg, mesh = get_cfg_and_mesh()
    jax_layer = EngramJAX(
        layer_id=self.layer_id,
        config=self.e_cfg,
        backbone_cfg=self.b_cfg,
        rngs=self.nnx_rng,
        cfg=cfg,
        mesh=mesh,
        tokenizer=tokenizer,
    )

    print("torch_layer", pt_layer.state_dict())
    print("jax_layer", jax_layer)

    # 3. Synchronize Weights
    jax_weights = get_jax_engram_weights(pt_layer)
    nnx.update(jax_layer, jax_weights)

    # 4. Prepare Inputs
    # Create random input_ids and hidden_states
    input_ids_np = np.random.randint(0, 1000, (batch_size, seq_len))

    pt_input_ids = torch.from_numpy(input_ids_np)
    pt_hidden_states = torch.randn(batch_size, seq_len, self.b_cfg.hc_mult, self.b_cfg.hidden_size, dtype=torch.float32)

    jax_hidden_states = to_jax(pt_hidden_states)

    # 5. Run Inference
    with torch.no_grad():
      pt_out = pt_layer(pt_hidden_states, pt_input_ids)

    jax_out = jax_layer(jax_hidden_states, to_jax(pt_input_ids))

    # 6. Numerical Comparison
    print(f"\nPT Output Mean: {pt_out.mean().item():.6f}")
    print(f"JAX Output Mean: {jax_out.mean():.6f}")

    # We expect high similarity since we copied weights exactly
    np.testing.assert_allclose(
        to_jax(pt_out), jax_out, rtol=1e-4, atol=1e-4, err_msg="Engram output mismatch between PT and JAX"
    )
    print("✅ Engram Layer Equivalence: PASS")


if __name__ == "__main__":
  unittest.main()
