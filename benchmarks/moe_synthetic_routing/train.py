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

"""Benchmark pre-training entrypoint with synthetic MoE routing injection.

Wraps `DataLoader.load_next_batch` and `maxtext_utils.get_shaped_batch` at
runtime when `synthetic_routing` (`lognormal` or `balanced`) is requested on
the command line, then delegates directly to `maxtext.trainers.pre_train.train`.
"""

from collections.abc import Sequence
import contextlib
import functools
import sys
import weakref

from absl import app
from benchmarks.moe_synthetic_routing import synthetic_routing
import jax
import jax.numpy as jnp
from maxtext.common import data_loader
from maxtext.trainers.pre_train import train as pre_train
from maxtext.utils import maxtext_utils


@contextlib.contextmanager
def install_synthetic_routing(syn_cfg: synthetic_routing.SyntheticRoutingCfg):
  """Installs runtime hooks for `forced_routed_experts` when `syn_cfg.enabled`."""
  if not syn_cfg.enabled:
    yield
    return

  orig_get_shaped_batch = maxtext_utils.get_shaped_batch
  orig_load_next_batch = data_loader.DataLoader.load_next_batch

  @functools.wraps(orig_get_shaped_batch)
  def wrapped_get_shaped_batch(config, batch_sharding=None, is_eval=False):
    shaped_batch = orig_get_shaped_batch(config, batch_sharding=batch_sharding, is_eval=is_eval)
    if not is_eval:
      routing_shape = shaped_batch["inputs"].shape + (
          int(config.num_decoder_layers),
          int(config.num_experts_per_tok),
      )
      shaped_batch = dict(shaped_batch)
      shaped_batch["forced_routed_experts"] = jax.ShapeDtypeStruct(routing_shape, jnp.int32, sharding=batch_sharding)
    return shaped_batch

  pools_by_loader: weakref.WeakKeyDictionary[object, synthetic_routing.SyntheticRouting] = weakref.WeakKeyDictionary()

  @functools.wraps(orig_load_next_batch)
  def wrapped_load_next_batch(self, *args, **kwargs):
    example_batch = orig_load_next_batch(self, *args, **kwargs)
    sr = pools_by_loader.get(self)
    if sr is None:
      sr = synthetic_routing.SyntheticRouting(self.config, self.input_data_shardings, syn_cfg=syn_cfg)
      pools_by_loader[self] = sr
    example_batch = dict(example_batch)
    example_batch["forced_routed_experts"] = sr.next()
    return example_batch

  maxtext_utils.get_shaped_batch = wrapped_get_shaped_batch
  data_loader.DataLoader.load_next_batch = wrapped_load_next_batch
  try:
    yield
  finally:
    maxtext_utils.get_shaped_batch = orig_get_shaped_batch
    data_loader.DataLoader.load_next_batch = orig_load_next_batch


def _run_with_cfg(syn_cfg: synthetic_routing.SyntheticRoutingCfg, remaining_argv: Sequence[str]) -> None:
  with install_synthetic_routing(syn_cfg):
    pre_train.main(remaining_argv)


def main(argv: Sequence[str] | None = None) -> None:
  raw_argv = list(sys.argv if argv is None else argv)
  syn_cfg, remaining_argv = synthetic_routing.parse_synthetic_routing_args(raw_argv)
  _run_with_cfg(syn_cfg, remaining_argv)


def _cli_entry() -> None:
  syn_cfg, remaining_argv = synthetic_routing.parse_synthetic_routing_args(sys.argv)
  app.run(functools.partial(_run_with_cfg, syn_cfg), argv=remaining_argv)


if __name__ == "__main__":
  _cli_entry()
