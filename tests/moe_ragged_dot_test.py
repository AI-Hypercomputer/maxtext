#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for MoE ragged dot operations."""

import csv

import functools
import os
from typing import Any, Callable, List, Tuple

from absl import flags
from absl import logging
import jax
from jax import sharding
from jax.experimental import layout
import jax.numpy as jnp
from maxtext.common import common_types
from maxtext.configs import pyconfig
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.utils import maxtext_utils
import numpy as np

from google3.platforms.xla.tests.tpu.python import jax_ragged_dot_test_util
from google3.pyglib import gfile
from google3.testing.pybase import googletest
from google3.testing.pybase import parameterized


_ENABLE_MOSAIC_EMITTERS = flags.DEFINE_string(
    "enable_mosaic_emitters",
    "",
    "Enable mosaic emitters.",
)
_RUN_BENCHMARK = flags.DEFINE_bool(
    "run_benchmark",
    False,
    "Whether to run XProf benchmarks.",
)


Layout = layout.Format
Mesh = sharding.Mesh
Config = common_types.Config
DType = common_types.DType
NdInitializer = initializers.NdInitializer
if jax.__version_info__ >= (0, 6, 3):
  DLL = layout.Layout
else:
  DLL = layout.DeviceLocalLayout  # type: ignore


def make_config(model_name: str, sparse_matmul: bool, megablox: bool):
  return pyconfig.initialize(
      [None, "third_party/py/maxtext/src/maxtext/configs/base.yml"],
      run_name=f"sparse_matmul_{sparse_matmul}_megablox_{megablox}_test",
      enable_checkpointing=False,
      per_device_batch_size=8,
      model_name=model_name,
      sparse_matmul=sparse_matmul,
      megablox=megablox,
  )


def create_routed_moe_args(cfg):
  """Creates arguments for routed MoE."""
  key = jax.random.PRNGKey(42)
  rng_model, rng_hidden_states = jax.random.split(key, 2)

  def uniform(k, shape):
    return jax.random.uniform(k, shape, cfg.dtype)

  hidden_states = uniform(
      rng_hidden_states,
      (
          int(cfg.per_device_batch_size),
          cfg.max_target_length,
          cfg.base_emb_dim,
      ),
  )
  moe_variables = {
      "params": {
          "gate": {
              "kernel": uniform(rng_model, (cfg.base_emb_dim, cfg.num_experts)),
          },
          "wi_0": uniform(rng_model, (cfg.num_experts, cfg.base_emb_dim, cfg.base_mlp_dim)),
          "wi_1": uniform(rng_model, (cfg.num_experts, cfg.base_emb_dim, cfg.base_mlp_dim)),
          "wo": uniform(rng_model, (cfg.num_experts, cfg.base_mlp_dim, cfg.base_emb_dim)),
      }
  }

  return [moe_variables, hidden_states]


def create_routed_and_shared_moe_args(cfg):
  """Creates arguments for routed and shared MoE."""

  key = jax.random.PRNGKey(42)
  rng_model, rng_hidden_states = jax.random.split(key, 2)

  def uniform(k, shape):
    return jax.random.uniform(k, shape, cfg.dtype)

  hidden_states = uniform(
      rng_hidden_states,
      (
          int(cfg.per_device_batch_size),
          cfg.max_target_length,
          cfg.base_emb_dim,
      ),
  )

  shared_expert_mlp_dim = (
      cfg.base_mlp_dim if cfg.decoder_block == common_types.DecoderBlockType.GEMMA4 else cfg.base_moe_mlp_dim
  )
  moe_variables = {
      "params": {
          "MoeBlock_0": {
              "gate": {
                  "kernel": uniform(rng_model, (cfg.base_emb_dim, cfg.num_experts)),
                  "bias": uniform(rng_model, (cfg.num_experts,)),
              },
              "wi_0": uniform(
                  rng_model,
                  (cfg.num_experts, cfg.base_emb_dim, cfg.base_moe_mlp_dim),
              ),
              "wi_1": uniform(
                  rng_model,
                  (cfg.num_experts, cfg.base_emb_dim, cfg.base_moe_mlp_dim),
              ),
              "wo": uniform(
                  rng_model,
                  (cfg.num_experts, cfg.base_moe_mlp_dim, cfg.base_emb_dim),
              ),
          },
          "shared_experts": {
              "wi_0": {
                  "kernel": uniform(
                      rng_model,
                      (
                          cfg.base_emb_dim,
                          shared_expert_mlp_dim * cfg.shared_experts,
                      ),
                  ),
              },
              "wi_1": {
                  "kernel": uniform(
                      rng_model,
                      (
                          cfg.base_emb_dim,
                          shared_expert_mlp_dim * cfg.shared_experts,
                      ),
                  ),
              },
              "wo": {
                  "kernel": uniform(
                      rng_model,
                      (
                          shared_expert_mlp_dim * cfg.shared_experts,
                          cfg.base_emb_dim,
                      ),
                  ),
              },
          },
      }
  }

  if not cfg.routed_bias:
    moe_variables["params"]["MoeBlock_0"]["gate"].pop("bias", None)

  return [moe_variables, hidden_states]


def create_routed_moe(cfg):
  """Creates routed MoE block execution function."""
  devices_array = maxtext_utils.create_device_mesh(cfg)
  mesh = Mesh(devices_array, cfg.mesh_axes)

  moe_block = moe.get_routed_moe(
      config=cfg,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      mesh=mesh,
      kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed", None),
      intermediate_dim=cfg.base_mlp_dim,
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
  )

  @jax.jit
  def impl(moe_variables, hidden_states):
    return moe_block.apply(moe_variables, hidden_states)

  return impl


def create_route_and_shared_moe(cfg):
  """Creates routed and shared MoE block execution function."""
  devices_array = maxtext_utils.create_device_mesh(cfg)

  mesh = Mesh(devices_array, cfg.mesh_axes)

  moe_block = moe.get_routed_and_shared_moe(
      config=cfg,
      mesh=mesh,
      kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed", None),
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
  )

  @jax.jit
  def impl(moe_variables, hidden_states):
    return moe_block.apply(moe_variables, hidden_states)

  return impl


def _compile(f, *args, **kwargs):
  @functools.partial(jax.jit, in_shardings=Layout(DLL.AUTO))
  def run(*args, **kwargs):
    return f(*args, **kwargs)

  executable = run.lower(*args, **kwargs).compile(
      jax.stages.CompilerOptions({"xla_tpu_enable_mosaic_emitters": (f"values: '{_ENABLE_MOSAIC_EMITTERS.value}'")})
  )
  return executable


def compile_config(cfg, fn, args):
  """Compiles JAX function for the given configuration."""
  try:
    compiled_fn = _compile(fn, *args)
    return compiled_fn
  except jax.errors.JaxRuntimeError as e:
    logging.exception("Failed to compile: %s.", cfg)
    str_e = str(e)
    is_vmem_oom = ("Ran out of memory in memory space vmem" in str_e) or (
        "RESOURCE_EXHAUSTED" in str_e and "space=vmem" in str_e
    )
    if not is_vmem_oom:
      raise
  return None


def benchmark(model_name: str, configs: List[Tuple[str, Callable[..., Any]]], *args, **kwargs):
  """Benchmarks the specified model configurations."""
  if not _RUN_BENCHMARK.value:
    return
  header = ["config", "mean (us)", "std", "session_id"]
  table = [header]
  for name, fn in configs:
    mean, std, session_id = jax_ragged_dot_test_util._run_under_xprof(fn, *args, **kwargs)  # pylint: disable=protected-access

    table.append(
        [
            name,
            f"{mean:0.3f}",
            f"{std:0.3f}",
            f"http://xprof/trace_viewer/{session_id}",
        ]
    )

  # Save the results to a file.
  directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
  filepath = os.path.join(directory, f"{model_name}_benchmark_results.txt")  # pyrefly: ignore[no-matching-overload]
  with gfile.Open(filepath, "w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(table)


class RoutedMoETest(parameterized.TestCase):

  @parameterized.parameters(
      ("mixtral-8x7b",),
      ("gpt-oss-20b",),
  )
  def test_moe_output(self, model_name: str):
    ragged = make_config(model_name, sparse_matmul=True, megablox=False)
    args = create_routed_moe_args(ragged)
    ragged_fn = compile_config(ragged, create_routed_moe(ragged), args)

    dense = make_config(model_name, sparse_matmul=False, megablox=False)
    dense_fn = compile_config(dense, create_routed_moe(dense), args)

    mblx = make_config(model_name, sparse_matmul=True, megablox=True)
    mblx_fn = compile_config(mblx, create_routed_moe(mblx), args)

    ragged_out, _, _ = ragged_fn(*args)
    logging.vlog(1, "ragged_out: %s", ragged_out)

    dense_out, _, _ = dense_fn(*args)
    logging.vlog(1, "dense_out: %s", dense_out)
    np.testing.assert_allclose(
        ragged_out.astype(jnp.float32),
        dense_out.astype(jnp.float32),
        rtol=6e-2,
        equal_nan=False,
    )

    mblx_out, _, _ = mblx_fn(*args)
    logging.vlog(1, "mblx_out: %s", mblx_out)
    np.testing.assert_allclose(
        mblx_out.astype(jnp.float32),
        dense_out.astype(jnp.float32),
        rtol=6e-2,
        equal_nan=False,
    )

    benchmark(
        model_name,
        [
            ("ragged", ragged_fn),
            ("dense", dense_fn),
            ("mblx", mblx_fn),
        ],
        *args,
    )


class RoutedAndSharedMoETest(parameterized.TestCase):

  @parameterized.parameters(
      ("llama4-17b-16e",),
      ("deepseek2-16b",),
      # We are not testing deepseek3-671b, since it is too large to run on TPUs
      # available in Forge.
      ("deepseek3-test",),
  )
  def test_moe_output(self, model_name: str):
    try:
      jax.config.update("jax_ragged_dot_use_ragged_dot_instruction", True)
    except AttributeError:
      logging.info("jax_ragged_dot_use_ragged_dot_instruction does not exist, possibly" " due to old JAX version.")
    ragged = make_config(model_name, sparse_matmul=True, megablox=False)
    args = create_routed_and_shared_moe_args(ragged)
    ragged_fn = compile_config(ragged, create_route_and_shared_moe(ragged), args)

    dense = make_config(model_name, sparse_matmul=False, megablox=False)
    dense_fn = compile_config(dense, create_route_and_shared_moe(dense), args)

    mblx = make_config(model_name, sparse_matmul=True, megablox=True)
    mblx_fn = compile_config(mblx, create_route_and_shared_moe(mblx), args)

    benchmark(
        model_name,
        [
            ("ragged", ragged_fn),
            ("dense", dense_fn),
            ("mblx", mblx_fn),
        ],
        *args,
    )

    ragged_out, _, _ = ragged_fn(*args)
    logging.vlog(1, "ragged_out: %s", ragged_out)

    dense_out, _, _ = dense_fn(*args)
    logging.vlog(1, "dense_out: %s", dense_out)
    np.testing.assert_allclose(
        ragged_out.astype(jnp.float32),
        dense_out.astype(jnp.float32),
        # TODO(b/327672581): Investigate numeric difference for deepseek2-16b.
        rtol=0.2 if model_name == "deepseek2-16b" else 6e-2,
        equal_nan=False,
    )

    mblx_out, _, _ = mblx_fn(*args)
    logging.vlog(1, "mblx_out: %s", mblx_out)
    np.testing.assert_allclose(
        mblx_out.astype(jnp.float32),
        dense_out.astype(jnp.float32),
        # TODO(b/327672581): Investigate numeric difference for deepseek2-16b.
        rtol=0.2 if model_name == "deepseek2-16b" else 6e-2,
        equal_nan=False,
    )


if __name__ == "__main__":
  googletest.main()
