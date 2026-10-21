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

"""Quantized training via qwix, applied after construction.

qwix works by interception: `qwix.quantize_model` traces one forward pass and
rewrites the `dot_general` / `einsum` calls under matching module paths to
quantize their operands on the fly. Nothing in a model file changes -- no
`dot_general` injection, no quantized layer classes -- which is what keeps
rule 4 of `m3/README.md` intact. Parameters stay `nnx.Param` and keep their
paths, and `maybe_quantize` checks the latter rather than assuming it.

The rule set is the legacy one for the same `config.quantization` values:
every matmul under the decoder layers, int8 or fp8, with backward-pass
quantization of the same type. The tied-embedding logits matmul and the
router projection are left in full precision, as in legacy.
"""

from flax import nnx
import jax
import jax.numpy as jnp
import qwix

_QTYPES = {
    "int8": jnp.int8,
    "fp8": jnp.float8_e4m3fn,
    "fp8_e4m3": jnp.float8_e4m3fn,
    "fp8_e5m2": jnp.float8_e5m2,
}


def quantization_rules(config) -> list[qwix.QtRule] | None:
  """qwix rules for `config.quantization`, or `None` when quantization is off."""
  if not config.quantization:
    return None
  if config.quantization not in _QTYPES:
    raise ValueError(f"m3 supports quantization in {sorted(_QTYPES)}; got {config.quantization!r}.")
  qtype = _QTYPES[config.quantization]
  return [
      # The router decides which experts run; quantizing it changes routing, so
      # it stays in full precision. qwix matches rules first-hit, so this one
      # goes before the catch-all.
      qwix.QtRule(module_path=r".*/gate$", weight_qtype=None, act_qtype=None, bwd_qtype=None, op_names=("dot_general",)),
      qwix.QtRule(
          module_path="decoder/layers.*",
          weight_qtype=qtype,
          act_qtype=qtype,
          bwd_qtype=qtype,
          op_names=("dot_general", "einsum"),
      ),
  ]


def _param_paths(model: nnx.Module) -> list[tuple]:
  return [path for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param))]


def maybe_quantize(model: nnx.Module, config, *example_args, **example_kwargs) -> nnx.Module:
  """Returns the qwix-quantized `model`, or `model` itself when quantization is off.

  `example_args` / `example_kwargs` are a batch of the shapes the model will
  be called with; qwix traces one forward pass over them to find the matmuls
  to intercept. Raises if quantization changed the parameter tree, since every
  path-keyed consumer downstream (checkpointing, HF mappings) would otherwise
  break quietly.
  """
  rules = quantization_rules(config)
  if rules is None:
    return model
  before = _param_paths(model)
  quantized = qwix.quantize_model(model, qwix.QtProvider(rules), *example_args, **example_kwargs)
  after = _param_paths(quantized)
  if before != after:
    raise RuntimeError(f"qwix changed the parameter tree: {sorted(set(before) ^ set(after))[:5]} ...")
  return quantized


def example_batch(config) -> tuple[jax.Array, jax.Array]:
  """`(tokens, positions)` of the training shape, for `maybe_quantize`'s trace."""
  shape = (config.micro_batch_size_to_train_on, config.max_target_length)
  tokens = jnp.ones(shape, dtype=jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(shape[1], dtype=jnp.int32), shape)
  return tokens, positions
