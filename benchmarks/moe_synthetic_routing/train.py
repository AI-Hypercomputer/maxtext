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
the command line, and optionally wraps `moe_replica_planner.compute_plan_iter`
for the overhead-control ablation arm (`--synthetic_routing_overhead_control=true`),
then delegates directly to `maxtext.trainers.pre_train.train`.
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
  """Installs runtime hooks for `forced_routed_experts` and optional overhead-control ablation."""
  if not syn_cfg.enabled and not syn_cfg.synthetic_routing_overhead_control:
    yield
    return

  orig_get_shaped_batch = maxtext_utils.get_shaped_batch
  orig_load_next_batch = data_loader.DataLoader.load_next_batch
  mrp_module = None
  orig_compute_plan_iter = None

  if syn_cfg.synthetic_routing_overhead_control:
    from maxtext.layers import moe_replica_planner as mrp_module  # pylint: disable=import-outside-toplevel

    orig_compute_plan_iter = mrp_module.compute_plan_iter

    @functools.wraps(orig_compute_plan_iter)
    def overhead_control_plan_iter(counts_src, num_ep, num_replicas, *args, **kwargs):
      plan = orig_compute_plan_iter(counts_src, num_ep, num_replicas, *args, **kwargs)
      plan.route_table = mrp_module.identity_route_table(num_ep, counts_src.shape[1], num_replicas)
      return plan

    mrp_module.compute_plan_iter = overhead_control_plan_iter

  if syn_cfg.enabled:

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
    if mrp_module is not None and orig_compute_plan_iter is not None:
      mrp_module.compute_plan_iter = orig_compute_plan_iter


def main(argv: Sequence[str] | None = None, *, use_absl_app: bool = False) -> None:
  """Parses synthetic-routing CLI flags and runs MaxText pre-training."""
  syn_cfg, remaining_argv = synthetic_routing.parse_synthetic_routing_args(sys.argv if argv is None else argv)
  with install_synthetic_routing(syn_cfg):
    if use_absl_app:
      app.run(pre_train.main, argv=remaining_argv)
    else:
      pre_train.main(remaining_argv)


if __name__ == "__main__":
  main(use_absl_app=True)
