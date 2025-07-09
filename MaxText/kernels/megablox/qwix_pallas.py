
"""A thin wrapper around jax.experimental.pallas that understands Qwix QArrays."""

from __future__ import annotations
import collections
from typing import Any, Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# ---- Qwix -------------------------------------------------------------------
from qwix.core.qarray import QArray, TransposedQArray          #
BlockSpec       = pl.BlockSpec
no_block_spec   = pl.no_block_spec
tree_util       = jax.tree_util


ArgAndBlockSpec = collections.namedtuple('ArgAndBlockSpec', 'arg block_spec')


# -----------------------------------------------------------------------------#
# Helper predicates
# -----------------------------------------------------------------------------#
_is_qarray              = lambda x: isinstance(x, QArray)
_is_transposed_qarray   = lambda x: isinstance(x, TransposedQArray)
_is_arg_and_block_spec  = lambda x: isinstance(x, ArgAndBlockSpec)


# -----------------------------------------------------------------------------#
# BlockSpec builder – no-op today, but keeps the API parallel to AQT’s version
# -----------------------------------------------------------------------------#
def _make_qarray_blockspec(arg: Any, block_spec: BlockSpec | Any) -> BlockSpec | Any:
  """
  If *arg* is a QArray and *block_spec* is a Pallas BlockSpec, we could
  customise the spec to match the qvalue buffer layout.  For now we simply
  return the original spec untouched, which is adequate for correctness
  because QArray.qvalue has exactly the same shape as the original tensor.
  """
  return block_spec


# -----------------------------------------------------------------------------#
# Transpose helper – placeholder until Qwix grows a dedicated fast path
# -----------------------------------------------------------------------------#
def _transpose_tensor_for_memory_saving(
    arg: Any, block_spec: BlockSpec
) -> ArgAndBlockSpec:
  """
  Mirrors AQT's memory-saving transpose shim.

  Qwix doesn't yet provide a built-in “transposed view” helper for Pallas,
  so we just pass the tensors through unchanged.  The wrapper still returns
  an *ArgAndBlockSpec* so downstream logic stays identical.
  """
  return ArgAndBlockSpec(arg, block_spec)



def pallas_call(
    f: Callable[..., None],
    *pl_call_args,
    grid_spec: pl.GridSpec | None = None,
    in_specs: Any = no_block_spec,
    **pl_call_kwargs,
):
  """`pl.pallas_call` wrapper that accepts Qwix QArrays.

  All signatures mirror the original AQT helper so MaxText code can simply
  replace:

      from aqt.jax.v2.pallas import pallas_call

  with

      from qwix_pallas import pallas_call
  """

  # If the caller passed a GridSpec, use the in_specs stored inside it.
  if grid_spec is not None:
    if jax.__version_info__ >= (0, 4, 31):
      in_specs = grid_spec.in_specs
    else:  # Pre-0.4.31 kept trees flat.
      in_specs = tree_util.tree_unflatten(grid_spec.in_specs_tree,
                                          grid_spec.in_specs)

  @jax.jit
  def wrapped(*args):

    # ---------------------------------------------#
    # Prefetch scalars (special TPU grid spec)
    # ---------------------------------------------#
    prefetch_args = ()
    if isinstance(grid_spec, pltpu.PrefetchScalarGridSpec):
      prefetch_args, args = (
          args[: grid_spec.num_scalar_prefetch],
          args[grid_spec.num_scalar_prefetch :],
      )

    # ---------------------------------------------#
    # 1) Flatten args & specs, keeping QArray leaves
    # ---------------------------------------------#
    flat_args,  args_treedef  = tree_util.tree_flatten(args, is_leaf=_is_qarray)
    flat_specs, specs_treedef = tree_util.tree_flatten(in_specs)

    # ---------------------------------------------#
    # 2) Build per-arg BlockSpecs for QArrays
    # ---------------------------------------------#
    flat_specs = tree_util.tree_map(
        _make_qarray_blockspec, flat_args, flat_specs, is_leaf=_is_qarray
    )

    # ---------------------------------------------#
    # 3) (Optional) transpose for better memory access
    # ---------------------------------------------#
    flat_pairs = tree_util.tree_map(
        _transpose_tensor_for_memory_saving,
        flat_args,
        flat_specs,
    )

    flat_args  = tree_util.tree_map(lambda x: x.arg,
                                    flat_pairs,
                                    is_leaf=_is_arg_and_block_spec)
    flat_specs = tree_util.tree_map(lambda x: x.block_spec,
                                    flat_pairs,
                                    is_leaf=_is_arg_and_block_spec)

    # ---------------------------------------------#
    # 4) Re-pack the PyTrees
    # ---------------------------------------------#
    args          = tree_util.tree_unflatten(args_treedef,  flat_args)
    kernel_specs  = tree_util.tree_unflatten(specs_treedef, flat_specs)

    # ---------------------------------------------#
    # 5) Define the actual kernel wrapper
    # ---------------------------------------------#
    def kernel(*k_args):
      # AQT unwraps TransposedTensor here; replicate for TransposedQArray
      k_args = jax.tree_util.tree_map(
          lambda a: getattr(a, "untransposed", a),
          k_args,
          is_leaf=_is_transposed_qarray,
      )
      return f(*k_args)

    # If caller handed us a GridSpec, make sure its in_specs field is current
    if grid_spec is not None:
      if jax.__version_info__ >= (0, 4, 31):
        grid_spec.in_specs = kernel_specs
      else:
        flat, tree = tree_util.tree_flatten(kernel_specs)
        grid_spec.in_specs, grid_spec.in_specs_tree = tuple(flat), tree

    # ---------------------------------------------#
    # 6) Call Pallas
    # ---------------------------------------------#
    compiled = pl.pallas_call(
        kernel,
        *pl_call_args,
        grid_spec=grid_spec,
        in_specs=kernel_specs if grid_spec is None else no_block_spec,
        **pl_call_kwargs,
    )

    return compiled(*prefetch_args, *args)

  # end wrapped

  return wrapped
