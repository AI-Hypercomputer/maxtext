from typing import Any, Callable, Sequence
import jax
from jax._src.api import TransferToMemoryKind
import jax.numpy as jnp


def _eval_jaxpr_and_transform(
    jaxpr: jax.core.Jaxpr,
    consts: Sequence[Any],
    output_transform: Callable[
        [
            jax.typing.ArrayLike | None,
            Sequence[tuple[jax.tree_util.KeyPath, Any]],
        ],
        dict[str, Any],
    ],
    out_paths: Sequence[jax.tree_util.KeyPath],
    *args,
    push_into_scan: bool,
    path_filter: Callable[[jax.tree_util.KeyPath], bool],
    host_outputs: Sequence[str],
) -> Sequence[Any]:
  """Evaluates a Jaxpr and applies an output transform to the results."""
  env = {}
  transformed_values = {}
  transformed_vars = set()

  def to_be_transformed(var):
    """Returns whether the variable should be transformed."""
    try:
      path = out_paths[jaxpr.outvars.index(var)]
      return path_filter(path)
    except ValueError:
      return False

  def read(var):
    """Reads a value from the evaluation environment."""
    # Literals are values baked into the Jaxpr
    if isinstance(var, jax.core.Literal):
      return var.val
    return env[var]

  def write(var, val):
    """Writes a value to the evaluation environment."""
    env[var] = val

  # Bind args and consts to environment
  jax.util.safe_map(write, jaxpr.invars, args)
  jax.util.safe_map(write, jaxpr.constvars, consts)

  def process_scan(eqn):
    invals = jax.util.safe_map(read, eqn.invars)
    # Push the tree map into the scan loop and lower to while loop.
    num_carry = eqn.params['num_carry']
    num_consts = eqn.params['num_consts']
    scan_jaxpr = eqn.params['jaxpr']
    reverse = eqn.params['reverse']
    length = eqn.params['length']
    if eqn.params['unroll'] != 1:
      # TODO(davelacey): Support unroll != 1
      raise ValueError('unroll != 1 is not supported.')

    # Determine the new shapes for the transformed scanned outputs (ys)
    # This requires us to trace the transform function for each transfomed
    # output and store each resulting jaxpr to evaluate during the loop.
    ys_inner_avals = scan_jaxpr.out_avals[num_carry:]
    num_orig_ys = len(ys_inner_avals)
    ys_outvars = eqn.outvars[num_carry:]
    ys_indices_to_transform = [
        i for i, var in enumerate(ys_outvars) if to_be_transformed(var)
    ]

    def path_from_y_index(i):
      return out_paths[jaxpr.outvars.index(ys_outvars[i])]

    def transform(index, ys_inner_avals):
      ys_transform_in = [
          (path_from_y_index(i), val)
          for i, val in zip(ys_indices_to_transform, ys_inner_avals)
      ]
      return output_transform(index, ys_transform_in)

    index_aval = jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32)
    transform_jaxpr, transformed_shape = jax.make_jaxpr(
        transform, return_shape=True
    )(index_aval, [ys_inner_avals[i] for i in ys_indices_to_transform])
    # Determine which output avals should be in host memory.
    transform_output_on_host = jax.util.safe_map(
        lambda x: x[0][0].key in host_outputs,
        jax.tree_util.tree_flatten_with_path(transformed_shape)[0],
    )
    offload_to_host_in_loop = jax.util.safe_map(
        lambda x: x[0][0].key in host_outputs and x[1].shape,
        jax.tree_util.tree_flatten_with_path(transformed_shape)[0],
    )

    def make_stacked_zeros(x):
      return jnp.zeros((length, *x.shape), x.dtype)

    ys_init = tuple(jax.tree.map(make_stacked_zeros, ys_inner_avals))
    transformed_ys_init = tuple(
        jax.tree.map(make_stacked_zeros, transform_jaxpr.out_avals)
    )
    transformed_ys_init = tuple(
        jax.device_put(x, TransferToMemoryKind('pinned_host')) if on_host else x
        for x, on_host in zip(transformed_ys_init, offload_to_host_in_loop)
    )
    ys_init = ys_init + transformed_ys_init
    consts, carrys_init, xs = jax.util.split_list(
        invals, [num_consts, num_carry]
    )

    def cond_fn(while_carry):
      return while_carry[0] < length

    def body_fn(while_carry):
      counter_, carrys, ys = while_carry
      counter = length - counter_ - 1 if reverse else counter_
      sliced = [
          jax.lax.dynamic_index_in_dim(x, counter, keepdims=False) for x in xs
      ]
      outs = jax.core.jaxpr_as_fun(scan_jaxpr)(*(consts + carrys + sliced))
      new_carrys, slices = outs[:num_carry], outs[num_carry:]
      new_ys = tuple(
          jax.lax.dynamic_update_index_in_dim(y, slice, counter, 0)
          for slice, y in zip(slices, ys, strict=False)
      )

      if ys_indices_to_transform:
        slices_to_transform = [slices[i] for i in ys_indices_to_transform]
        transform_slices = jax.core.jaxpr_as_fun(transform_jaxpr)(
            counter, *slices_to_transform
        )
        transform_slices = tuple(
            jax.device_put(x, TransferToMemoryKind('pinned_host'))
            if on_host
            else x
            for x, on_host in zip(transform_slices, offload_to_host_in_loop)
        )
        new_ys = new_ys + tuple(
            jax.lax.dynamic_update_index_in_dim(y, slice, counter, 0)
            for slice, y in zip(transform_slices, ys[num_orig_ys:], strict=True)
        )
      return (counter_ + 1, new_carrys, new_ys)

    i_init = jnp.zeros((), dtype=jnp.int32)
    _, carrys_out, ys_out = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (i_init, carrys_init, ys_init),
    )
    transformed_ys_out = ys_out[num_orig_ys:]
    transformed_ys_out = tuple(
        jax.device_put(x, TransferToMemoryKind('pinned_host'))
        if not already_transferred and on_host
        else x
        for x, already_transferred, on_host in zip(
            transformed_ys_out,
            offload_to_host_in_loop,
            transform_output_on_host,
        )
    )
    ys_out = ys_out[:num_orig_ys] + transformed_ys_out

    # After the loop write the results to the environment (not including the
    # transformed outputs).
    jax.util.safe_map(write, eqn.outvars[:num_carry], carrys_out)
    jax.util.safe_map(write, eqn.outvars[num_carry:], ys_out[:num_orig_ys])

    transform_treedef = jax.tree_util.tree_structure(transformed_shape)
    transform_output = jax.tree.unflatten(
        transform_treedef, ys_out[num_orig_ys:]
    )
    nonlocal transformed_vars, transformed_values
    transformed_vars.update((ys_outvars[i] for i in ys_indices_to_transform))
    transformed_values.update(transform_output)

  def scan_produces_transformed_output(eqn):
    return any(
        v in jaxpr.outvars for v in eqn.outvars[eqn.params['num_carry'] :]
    )

  def process_eqn(eqn):
    invals = jax.util.safe_map(read, eqn.invars)

    if (
        eqn.primitive == jax.lax.scan_p
        and push_into_scan
        and scan_produces_transformed_output(eqn)
    ):
      process_scan(eqn)
    else:
      outvals = eqn.primitive.bind(*invals, **eqn.params)
      # Primitives may return multiple outputs or not
      if not eqn.primitive.multiple_results:
        outvals = [outvals]

      # Write the results of the primitive into the environment
      jax.util.safe_map(write, eqn.outvars, outvals)

  for eqn in jaxpr.eqns:
    process_eqn(eqn)

  # Find the outputs that were not transformed in a loop and transform them.
  remaining_vals_to_transform = [
      (out_paths[i], read(v))
      for i, v in enumerate(jaxpr.outvars)
      if v not in transformed_vars and path_filter(out_paths[i])
  ]
  if remaining_vals_to_transform:
    remaining_transformed = output_transform(None, remaining_vals_to_transform)
    transformed_values.update(remaining_transformed)

  return ([read(v) for v in jaxpr.outvars], transformed_values)


def apply_and_transform_output(
    f: Callable[..., Any],
    g: Callable[
        [
            jax.typing.ArrayLike | None,
            Sequence[tuple[jax.tree_util.KeyPath, Any]],
        ],
        dict[str, Any],
    ],
    push_into_scan: bool = True,
    path_filter: Callable[[jax.tree_util.KeyPath], bool] = lambda x: True,
    host_outputs: Sequence[str] = (),
) -> Callable[..., tuple[Any, dict[str, Any]]]:
  """Create a function that applies a function and transform the output.

  This function returns a wrapper that applies `f` to the arguments and
  then applies `g` to the output of `f`.

  `g` acts on the flattened output of `f`, in essence it can be thought of a
  bit like:

  ```
  output = f(*args, **kwargs)
  paths_and_vals = jax.tree_util.tree_flatten_with_path(x)
  filtered_paths_and_vals = [
      (path, val) for path, val in paths_and_vals if path_filter(path)
  ]
  transformed = g(_, filtered_paths_and_vals)
  ```

  The key feature of this wrapper is
  that it can "push" `g` into scan loops within `f` so that `g` is applied to
  each slice of the output within the scan loop with the results of `g` are
  stacked. In this can `g` will be called multiple times - once for each scan
  loop that produces outputs and once for any outputs that are not produced
  within a scan loop. The final result is the aggregated dictionary from all
  the calls to `g`.

  Args:
    f: The function to apply.
    g: The function to map over the output tree. This function should take an
      index parameter that is either None or the index of the scan loop that
      produced the value and a sequence of (path, value) pairs. It should return
      a dictionary.
    push_into_scan: If True, push the tree map into scan primitives that produce
      the output. In this case `g` is applied to each slice of the output within
      the scan loop.
    path_filter: A function that returns True if the value at the path should be
      transformed.
    host_outputs: A list of dict keys of transform outputs that should be placed
      in host memory.

  Returns:
    A function that applies `f` then applies `g` to the output.
  """

  def wrapped(*args):
    closed_jaxpr, return_shape = jax.make_jaxpr(f, return_shape=True)(*args)
    in_vals, _ = jax.tree.flatten(args)
    out_paths_and_shapes, _ = jax.tree_util.tree_flatten_with_path(return_shape)
    out_paths = jax.util.safe_map(lambda x: x[0], out_paths_and_shapes)
    out_vals, transformed_values = _eval_jaxpr_and_transform(
        closed_jaxpr.jaxpr,
        closed_jaxpr.consts,
        g,
        out_paths,
        *in_vals,
        push_into_scan=push_into_scan,
        path_filter=path_filter,
        host_outputs=host_outputs,
    )
    out_treedef = jax.tree_util.tree_structure(return_shape)
    return jax.tree.unflatten(out_treedef, out_vals), transformed_values

  return wrapped
