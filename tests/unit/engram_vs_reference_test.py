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
Tests for Engram: CompressedTokenizer, NgramHashMapping, MultiHeadEmbedding, ShortConv, Engram

Reference implementation: 
https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py

To run the test:
  python3 -m pip install torch numpy transformers sympy
  python3 -m pytest -v --pyargs tests.unit.engram_vs_reference_test -rP -s
"""


from typing import List
from dataclasses import dataclass, field
import math
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
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from MaxText import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

from maxtext.layers.engram import CompressedTokenizer as CompressedTokenizerJAX
from maxtext.layers.engram import NgramHashMapping as NgramHashMappingJAX
from maxtext.layers.engram import MultiHeadEmbedding as MultiHeadEmbeddingJAX
from maxtext.layers.engram import ShortConv as ShortConvJAX
from maxtext.layers.engram import Engram as EngramJAX


def setUpModule():
  """
  Enable 64-bit precision for JAX in test. Set before JAX operations
  to prevent downcasting from int64 to int32 for correctness comparison.
  """
  jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class Config:
  """MaxText config"""

  base_emb_dim: int = 1024
  tokenizer_path: str = "deepseek-ai/DeepSeek-V3"
  mhc_expansion_rate: int = 4
  # Engram
  engram_max_ngram_size: int = 3  # max_ngram_size >=2, use 2...N
  # List of minimum head vocab sizes for each n-gram order
  engram_vocab_bases: List[int] = field(default_factory=lambda: [129280 * 5, 129280 * 5])
  engram_layers: List[int] = field(default_factory=lambda: [1, 15])
  engram_kernel_size: int = 4  # conv kernel size
  engram_head_dim: int = 32
  engram_num_heads: int = 8  # num heads per n-gram
  # Hashing
  # This can be replaced with tokenizer.pad_token_id
  engram_pad_id: int = 2
  engram_seed: int = 0


class EngramConfig:
  """Torch Engram Config"""

  def __init__(self, config):
    self.tokenizer_name_or_path = config.tokenizer_path
    self.engram_vocab_size = config.engram_vocab_bases
    self.max_ngram_size = config.engram_max_ngram_size
    self.n_embed_per_ngram = config.engram_head_dim * config.engram_num_heads
    self.n_head_per_ngram = config.engram_num_heads
    self.layer_ids = config.engram_layers
    self.pad_id = config.engram_pad_id
    self.seed = config.engram_seed
    self.kernel_size = config.engram_kernel_size


class BackBoneConfig:
  """Torch Backbone Config"""

  def __init__(self, config):

    self.hidden_size = config.base_emb_dim
    self.hc_mult = config.mhc_expansion_rate


# -----------------------------------------------------------------------------
# Torch Reference Implementation
# -----------------------------------------------------------------------------

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable


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

    self.compressed_tokenizer = CompressedTokenizer(tokenizer_name_or_path=tokenizer_name_or_path)
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
      r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
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
          found_prime = find_next_prime(current_prime_search_start, seen_primes)
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
      if k == 0:
        return x
      shifted = np.pad(x, ((0, 0), (k, 0)), mode="constant", constant_values=self.pad_id)[:, :T]
      return shifted

    base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

    all_hashes = []

    for n in range(2, self.max_ngram_size + 1):
      n_gram_index = n - 2
      tokens = base_shifts[:n]
      mix = tokens[0] * multipliers[0]
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

  # added argument: engram_cfg, backbone_config
  def __init__(self, layer_id, backbone_config, engram_cfg):
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
    self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
    self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])

  # added argument: backbone_config
  def forward(self, hidden_states, input_ids, backbone_config):
    """
    hidden_states: [B, L, HC_MULT, D]
    input_ids: [B, L]
    """
    hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids)[self.layer_id])
    embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
    gates = []
    for hc_idx in range(backbone_config.hc_mult):
      key = self.key_projs[hc_idx](embeddings)
      normed_key = self.norm1[hc_idx](key)
      query = hidden_states[:, :, hc_idx, :]
      normed_query = self.norm2[hc_idx](query)
      gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
      gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
      gate = gate.sigmoid().unsqueeze(-1)
      gates.append(gate)
    gates = torch.stack(gates, dim=2)
    value = gates * self.value_proj(embeddings).unsqueeze(2)
    output = value + self.short_conv(value)
    return output


# pylint: enable=missing-class-docstring
# pylint: enable=missing-function-docstring
# pylint: enable=unused-variable

# -----------------------------------------------------------------------------
# Test JAX Module: Helper
# -----------------------------------------------------------------------------


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


def get_cfg_and_mesh(config):
  """Returns MaxText configuration and mesh."""
  cfg = pyconfig.initialize(
      [None, get_test_config_path()],
      run_name="",
      enable_checkpointing=False,
      model_name="default",
      dtype="float32",
      # high precision
      weight_dtype="float32",
      matmul_precision="highest",
      float32_qk_product=True,
      float32_logits=True,
      base_emb_dim=config.base_emb_dim,
  )
  devices_array = maxtext_utils.create_device_mesh(cfg)
  mesh = Mesh(devices_array, cfg.mesh_axes)
  return cfg, mesh


# -----------------------------------------------------------------------------
# Test JAX Module (non-parametric): CompressedTokenizer, NgramHashMapping
# -----------------------------------------------------------------------------


class CompressedTokenizerTest(parameterized.TestCase):

  def test_tokenierzer_match(self):
    np.random.seed(42)
    tokenizer_path = "deepseek-ai/DeepSeek-V3"
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # Input
    batch_size, seq_len = 4, 128
    input_ids = np.random.randint(0, len(hf_tokenizer), (batch_size, seq_len))
    # 1. PyTorch
    pt_tokenizer = CompressedTokenizer(tokenizer_path)
    pt_lookup_table = pt_tokenizer.lookup_table
    pt_out = pt_tokenizer(input_ids)
    # 2. JAX
    jax_tokenizer = CompressedTokenizerJAX(hf_tokenizer)
    jax_lookup_table = jax_tokenizer.lookup_table
    jax_out = jax_tokenizer(input_ids)
    # 3. Compare
    np.testing.assert_equal(jax_lookup_table, pt_lookup_table)
    np.testing.assert_equal(len(pt_tokenizer), len(jax_tokenizer))
    np.testing.assert_array_equal(pt_out, jax_out)


class NgramHashMappingTest(parameterized.TestCase):

  def test_hash_match(self):
    np.random.seed(42)
    self.config = Config()
    self.engram_cfg = EngramConfig(self.config)
    self.backbone_config = BackBoneConfig(self.config)
    tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, trust_remote_code=True)
    # Input
    batch_size, seq_len = 4, 128
    input_ids = np.random.randint(0, len(tokenizer), (batch_size, seq_len))
    # 1. PyTorch
    pt_hash_mapping = NgramHashMapping(
        engram_vocab_size=self.engram_cfg.engram_vocab_size,
        max_ngram_size=self.engram_cfg.max_ngram_size,
        n_embed_per_ngram=self.engram_cfg.n_embed_per_ngram,
        n_head_per_ngram=self.engram_cfg.n_head_per_ngram,
        layer_ids=self.engram_cfg.layer_ids,
        tokenizer_name_or_path=self.engram_cfg.tokenizer_name_or_path,
        pad_id=self.engram_cfg.pad_id,
        seed=self.engram_cfg.seed,
    )
    pt_out = pt_hash_mapping.hash(input_ids)
    # 2. JAX
    jax_hash_mapping = NgramHashMappingJAX(
        engram_vocab_bases=self.config.engram_vocab_bases,
        max_ngram_size=self.config.engram_max_ngram_size,
        engram_num_heads=self.config.engram_num_heads,
        layer_ids=self.config.engram_layers,
        tokenizer=tokenizer,
        pad_id=self.config.engram_pad_id,
        seed=self.config.engram_seed,
    )
    jax_out = jax_hash_mapping(input_ids)
    # 3. Compare
    # keys are layer_ids
    self.assertDictEqual(jax_hash_mapping.vocab_size_across_layers, pt_hash_mapping.vocab_size_across_layers)
    np.testing.assert_equal(pt_out, jax_out)


# -----------------------------------------------------------------------------
# Test JAX Module: MultiHeadEmbedding
# -----------------------------------------------------------------------------


def to_jax_mhe(pt_layer):
  """
  Extracts weights from PyTorch MultiHeadEmbedding.
  """
  return {"embedding": {"embedding": to_jax(pt_layer.embedding.weight)}}


class MultiHeadEmbeddingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    np.random.seed(42)
    self.rngs = nnx.Rngs(params=0)

  @parameterized.named_parameters(
      {"testcase_name": "multiple_head", "vocab_sizes": [100, 200, 150]},
      {"testcase_name": "single_head", "vocab_sizes": [500]},
  )
  def test_mhe_match(self, vocab_sizes, head_dim=32):
    # vocab_sizes: a flattened list of sizes for all heads across all n-grams
    # Input
    num_total_heads = len(vocab_sizes)
    # indices must be within the range of each specific head's vocab.
    batch_size, seq_len = 4, 128
    input_np = np.zeros((batch_size, seq_len, num_total_heads), dtype=np.int32)
    for i, v_size in enumerate(vocab_sizes):
      input_np[:, :, i] = np.random.randint(0, v_size, (batch_size, seq_len))
    x_pt = torch.from_numpy(input_np).long()
    x_jax = jnp.array(input_np)

    # 1. PyTorch
    pt_model = MultiHeadEmbedding(vocab_sizes, head_dim)
    init_torch_weights(pt_model)
    pt_model.eval()
    with torch.no_grad():
      y_pt = pt_model(x_pt)

    # 2. JAX
    config = Config()
    cfg, mesh = get_cfg_and_mesh(config)
    jax_model = MultiHeadEmbeddingJAX(cfg, mesh, vocab_sizes, head_dim, self.rngs)
    # weight transfer
    weights = to_jax_mhe(pt_model)
    nnx.update(jax_model, weights)
    # forward
    y_jax = jax_model(x_jax)

    # 3. Compare
    # Check offsets
    jax_offsets = jax_model.offsets
    if hasattr(jax_offsets, "val"):
      jax_offsets = jax_offsets.val
    np.testing.assert_array_equal(jax_offsets, to_jax(pt_model.offsets))
    # Check outputs
    np.testing.assert_allclose(y_jax, to_jax(y_pt), rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------
# Test JAX Module: ShortConv
# -----------------------------------------------------------------------------


def to_jax_norm(pt_norm):
  """Extracts scale parameter from a norm layer."""
  return {"scale": to_jax(pt_norm.weight)}


def to_jax_stack(pt_module_list, transform_fn, axis=0):
  """
  Applies transform_fn to a list of modules and stacks the
  resulting PyTrees along axis. If 0, create a new 0th dimension.
  """
  jax_trees = [transform_fn(m) for m in pt_module_list]
  # Stacks all keys (kernel, bias, etc.) along axis
  return jax.tree.map(lambda *xs: jnp.stack(xs, axis=axis), *jax_trees)


def to_jax_shortconv(pt_layer):
  """
  Converts a ShortConv layer containing a Conv and a ModuleList of Norms.
  """
  return {
      # [Out, In//Groups, Kernel] -> [Kernel, In//Groups, Out]
      "conv": {"kernel": to_jax(pt_layer.conv.weight.permute(2, 1, 0))},
      # List of norms -> Stacked norm, shape [Groups, Channels]
      "norm": to_jax_stack(pt_layer.norms, to_jax_norm),
  }


class ShortConvTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    np.random.seed(42)
    self.nnx_rngs = nnx.Rngs(params=0)

  @parameterized.named_parameters(
      {"testcase_name": "multi_branch", "mhc_expansion_rate": 4},
      {"testcase_name": "single_branch", "mhc_expansion_rate": 1},
  )
  def test_shortconv_match(self, mhc_expansion_rate, dilation=2, hidden_size=16, kernel_size=3):

    # Input, Shape [B, S, G, D]
    batch_size, seq_len = 4, 128
    x_pt = torch.randn(batch_size, seq_len, mhc_expansion_rate, hidden_size)
    x_jax = to_jax(x_pt)

    config = Config()
    cfg, _ = get_cfg_and_mesh(config)

    # 1. PyTorch
    pt_model = ShortConv(
        hidden_size, kernel_size, dilation, hc_mult=mhc_expansion_rate, norm_eps=cfg.normalization_layer_epsilon
    )
    init_torch_weights(pt_model)
    pt_model.eval()
    with torch.no_grad():
      y_pt = pt_model(x_pt)

    # 2. JAX
    jax_model = ShortConvJAX(
        cfg, hidden_size, kernel_size, dilation, mhc_expansion_rate=mhc_expansion_rate, rngs=self.nnx_rngs
    )
    # weight transfer
    weights = to_jax_shortconv(pt_model)
    nnx.update(jax_model, weights)
    # forward
    y_jax = jax_model(x_jax)

    # 3. Compare
    np.testing.assert_allclose(y_jax, to_jax(y_pt), rtol=1e-3, atol=1e-3)


# -----------------------------------------------------------------------------
# Test JAX Module: Engram
# -----------------------------------------------------------------------------


def to_jax_linear(pt_linear):
  """(Out, In) -> {'kernel': (In, Out), 'bias': (Out)}"""
  out = {"kernel": to_jax(pt_linear.weight.T)}
  if pt_linear.bias is not None:
    out["bias"] = to_jax(pt_linear.bias)
  return out


def to_jax_engram(pt_engram) -> dict:
  return {
      "multi_head_embedding": to_jax_mhe(pt_engram.multi_head_embedding),
      "value_proj": to_jax_linear(pt_engram.value_proj),
      # Result shapes: Kernel [In, G, Out], Bias [G, Out]
      "key_proj": to_jax_stack(pt_engram.key_projs, to_jax_linear, axis=-2),
      # Result shapes: Scale [G, D]
      "k_norm": to_jax_stack(pt_engram.norm1, to_jax_norm),
      "q_norm": to_jax_stack(pt_engram.norm2, to_jax_norm),
      "short_conv": to_jax_shortconv(pt_engram.short_conv),
  }


class EngramTest(parameterized.TestCase):
  """Verifies JAX Engram implementation matches PyTorch reference."""

  def setUp(self):
    super().setUp()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    np.random.seed(42)
    self.nnx_rng = nnx.Rngs(params=0)
    self.layer_id = 1  # must belong to config.engram_layers

  @parameterized.named_parameters(
      {"testcase_name": "multi_branch", "mhc_expansion_rate": 4},
      {"testcase_name": "single_branch", "mhc_expansion_rate": 1},
  )
  def test_engram_match(self, mhc_expansion_rate):
    # Config
    self.config = Config(mhc_expansion_rate=mhc_expansion_rate)
    self.engram_cfg = EngramConfig(self.config)
    self.backbone_config = BackBoneConfig(self.config)
    # Inputs
    # random input_ids [B, S]
    batch_size, seq_len = 4, 128
    input_ids_np = np.random.randint(0, 1000, (batch_size, seq_len))
    pt_input_ids = torch.from_numpy(input_ids_np)
    # hidden_states [B, S, G, D]
    pt_hidden_states = torch.randn(
        batch_size, seq_len, self.backbone_config.hc_mult, self.backbone_config.hidden_size, dtype=torch.float32
    )

    # 1. PyTorch
    pt_layer = Engram(layer_id=self.layer_id, backbone_config=self.backbone_config, engram_cfg=self.engram_cfg)
    init_torch_weights(pt_layer)
    pt_layer.eval()
    # forward
    with torch.no_grad():
      pt_out = pt_layer(pt_hidden_states, pt_input_ids, self.backbone_config)

    # 2. JAX
    # data pipeline
    tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, trust_remote_code=True)
    jax_hash_mapping = NgramHashMappingJAX(
        engram_vocab_bases=self.config.engram_vocab_bases,
        max_ngram_size=self.config.engram_max_ngram_size,
        engram_num_heads=self.config.engram_num_heads,
        layer_ids=self.config.engram_layers,
        tokenizer=tokenizer,
        pad_id=self.config.engram_pad_id,
        seed=self.config.engram_seed,
    )
    vocab_sizes = jax_hash_mapping.get_vocab_sizes(self.layer_id)  # layer specific
    jax_hash_input_ids = jax_hash_mapping(input_ids_np)[self.layer_id]  # layer specific

    # setup model
    cfg, mesh = get_cfg_and_mesh(self.config)
    jax_layer = EngramJAX(
        rngs=self.nnx_rng,
        config=cfg,
        mesh=mesh,
        vocab_sizes=vocab_sizes,
        engram_num_heads=self.config.engram_num_heads,
        engram_head_dim=self.config.engram_head_dim,
        engram_max_ngram_size=self.config.engram_max_ngram_size,
        engram_kernel_size=self.config.engram_kernel_size,
        mhc_expansion_rate=self.config.mhc_expansion_rate,
    )
    # weight transfer
    jax_weights = to_jax_engram(pt_layer)
    nnx.update(jax_layer, jax_weights)
    # forward
    jax_out = jax_layer(to_jax(pt_hidden_states), jax_hash_input_ids)

    # 3. Compare
    np.testing.assert_allclose(to_jax(pt_out), jax_out, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  unittest.main()
