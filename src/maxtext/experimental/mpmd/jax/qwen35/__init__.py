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

"""Pure-JAX Qwen3.5 text decoder (MaxText `decoder_block: qwen3_5`).

    config.py     Qwen35Config (with `shard`, a ShardPolicy, and `gdn_core`),
                  tiny / qwen35_397b / qwen35_35b, from_maxtext; DeepSeek's
                  shared-config Protocol
    _ops.py       every sharding-sensitive call site (one small function each),
                  through src/maxtext/experimental/mpmd/jax/sharding_mode.py
    layers.py     zero-centred / plain RMSNorm (both `norm_numerics`)
    gdn.py        Gated DeltaNet: causal conv, chunked gated delta rule with a
                  custom-VJP unit-lower-triangular inverse, the layer
    attention.py  gated full attention + MaxText's two rotary embeddings, by
                  use_mrope: the text mRoPE (Qwen3.5's ymls) and the partial
                  RoPE (PartialRotaryEmbedding: Qwen3.8-2.4T's)
    moe.py        softmax-top-k router, routed experts, sigmoid-gated shared expert
    model.py      decoder layer, 4-layer block scan with block/layer remat,
                  forward, parameter avals
    init.py       parameter init by MaxText's rules where they shape the math
    train.py      loss and train step (the DeepSeek port's trainer pieces)
    convert.py    MaxText NNX state -> parameter tree (either MaxText layout)

Same style as `src/maxtext/experimental/mpmd/jax/deepseek/`: plain functions over parameter pytrees and a
frozen config.  The MPMD pipeline that runs these layers is
`src/maxtext/experimental/mpmd/jax/ds_ep_pipeline.py` (`--model qwen3.8-2.4t-a95b`; its
`ds_ep.qwen_mixer_part` calls this package's mixers and norms).  The model
traces with no mesh, under an Auto stage mesh and under an Explicit one
(`qwen35_explicit_test`).  The tests are
`tests/unit/experimental/mpmd/jax/qwen35_*_test.py`;
`qwen35_maxtext_parity_test` (scheduled_only: MaxText's scheduled CI runs it) holds
the port to MaxText's own `qwen3_5` model: bf16 forward logits bit-exact
(`megablox=false`), fp32 forward logits to 1e-6 and fp32 gradients to
max(1e-5, 4 x a measured floor), and the GDN's pieces bit for bit against
MaxText's own functions.

Semantics follow MaxText where it differs from HF: the mRoPE keeps fp32
cos/sin (the partial RoPE rounds them to the activation dtype, as HF does),
the GDN rounds as MaxText does (`gdn.py`), and padding (segment 0) is zeroed
after the GDN's input projections (HF zeroes padded hidden states first).
Packed documents: attention masks across segments, and so does the GDN --
its conv window and recurrent state restart at every document -- under
`Qwen35Config.gdn_packing`, MaxText's `packing` (its default); without it
they run on across documents.  `config.from_maxtext` reads the GDN's two
MaxText options (`gdn_packing`, `gdn_chunk_inverse`) off the MaxText config,
and takes how the two RMSNorm kinds round as a parameter (`norm_numerics`:
'fp32', the default, is MaxText's since 2026-09-10; 'bf16' is MaxText's before
then, which the port was first written against -- neither is HF's: HF's
zero-centred norm rounds as 'fp32' does, its GDN gated norm as 'bf16' does).
"""
