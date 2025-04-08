# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from pathlib import Path
from typing import Optional
from collections import namedtuple
import random as pyrandom

import jax
from jax import numpy as jnp
from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

# kernel ###############################################################################################################


def gpu_ragged_dot_kernel(
    # inputs
    x_ref,  # [m, k]
    A_ref,  # [k, n]
    group_sizes_ref,  # [g]
    group_offset_ref,  # [g]
    A_scale_ref,  # [n]
    # outputs
    y_ref,  # [k, n]
    # static problem shapes
    m: int,
    k: int,
    n: int,
    # hyperparameters
    block_c: int = 8,
    block_k: int = 1024,
    block_n: int = 128,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: "jnp.dtype" = jnp.float32,
):
    pid = namedtuple("pid", ["i", "j"])(pl.program_id(0), pl.program_id(1))
    size = namedtuple("size", ["m", "k", "n"])(m, k, n)  # pack into named tuple to not lose indices later
    group_sz = group_sizes_ref[pid.i]
    compute_dtype = compute_dtype if compute_dtype is not None else x_ref.dtype

    dim_nums = (((1,), (0,)), ((), ()))
    _dot_fn = functools.partial(jax.lax.dot_general, dimension_numbers=dim_nums, preferred_element_type=acc_dtype)

    @pl.when(group_sz > 0)
    def _():
        # row index into lhs and output
        start_ridx = jnp.where(pid.i == 0, 0, group_offset_ref[jnp.maximum(pid.i - 1, 0)])

        def outer_compute(r_offset, _):
            ridx = start_ridx + r_offset * block_c  # r_offset is 0,1,2,... need to map it to actual row indices
            lhs_rows_mask = (r_offset * block_c + jnp.arange(block_c)) < group_sz
            lhs_rows_idx = pl.ds(ridx, block_c)
            rhs_cols_idx = pl.ds(0, block_n)
            rhs_cols_mask = (block_n * pid.j + jnp.arange(block_n)) < size.n

            def inner_compute(k, acc):
                inner_idx = pl.ds(k * block_k, block_k)
                inner_mask = (k * block_k + jnp.arange(block_k)) < size.k
                x = pl.load(x_ref, (lhs_rows_idx, inner_idx), mask=lhs_rows_mask[:, None] & inner_mask[None, :])
                A = pl.load(A_ref, (inner_idx, rhs_cols_idx), mask=inner_mask[:, None] & rhs_cols_mask[None, :])
                return acc + _dot_fn(x.astype(compute_dtype), A.astype(compute_dtype)).astype(acc.dtype)

            acc = jnp.zeros((block_c, block_n), dtype=acc_dtype)
            acc = jax.lax.fori_loop(0, pl.cdiv(size.k, block_k), inner_compute, acc)
            if A_scale_ref is not None:
                acc = acc * pl.load(A_scale_ref, rhs_cols_idx, mask=rhs_cols_mask).astype(acc.dtype)
            acc = acc.astype(y_ref.dtype)
            pl.store(y_ref, (lhs_rows_idx, rhs_cols_idx), acc, mask=lhs_rows_mask[:, None] & rhs_cols_mask[None, :])
            return None

        jax.lax.fori_loop(0, pl.cdiv(group_sz, block_c), outer_compute, None)

# main routine #########################################################################################################

@functools.partial(jax.jit, static_argnums=list(range(4, 12)))
def gpu_ragged_dot(
    x: jax.Array,  # [m, k]
    A: jax.Array,  # [g, k, n]
    group_sizes: jax.Array,  # [g]
    A_scale: jax.Array | None = None,  # [g, n]  or None
    block_c: int = 8,
    block_k: int = 128,
    block_n: int = 64,
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
    """Compute grouped matmul on GPU via a Pallas lowering."""
    assert A.ndim == 3 and x.ndim == 2
    assert A.shape[:1] == group_sizes.shape
    if A_scale is not None:
        assert A_scale.shape == (A.shape[0], A.shape[-1])  # one scale per A column

    # normalize the block sizes for GPU
    block_c, block_k, block_n = min(block_c, x.shape[0]), min(block_k, x.shape[-1]), min(block_n, A.shape[-1])
    block_c, block_k, block_n = map(pl.next_power_of_2, (block_c, block_k, block_n))
    block_k, block_n = max(block_k, 16), max(block_n, 16)

    size = namedtuple("size", ["m", "k", "n"])(x.shape[0], x.shape[-1], A.shape[-1])
    group_offsets = jnp.cumsum(group_sizes, -1)  # we'll read 1 down always
    in_specs = [
        pl.BlockSpec(x.shape, lambda i, j: (0, 0)),
        pl.BlockSpec((None, size.k, block_n), lambda i, j: (i, 0, j)),
        pl.BlockSpec((group_sizes.size,), lambda i, j: (0,)),
        pl.BlockSpec((group_offsets.size,), lambda i, j: (0,)),
        pl.BlockSpec((None, block_n), lambda i, j: (i, j)) if A_scale is not None else None,
    ]

    out_shape = jax.ShapeDtypeStruct((size.m, size.n), dtype=x.dtype)
    out_specs = pl.BlockSpec((size.m, block_n), lambda i, j: (0, j))
    grid = (A.shape[0], pl.cdiv(size.n, block_n))
    block_sizes = dict(block_c=block_c, block_k=block_k, block_n=block_n)
    dtype_spec = dict(compute_dtype=compute_dtype, acc_dtype=acc_dtype)
    y = pl.pallas_call(
        functools.partial(gpu_ragged_dot_kernel, **size._asdict(), **block_sizes, **dtype_spec),
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=interpret,
        compiler_params=plgpu.TritonCompilerParams(num_warps=num_warps, num_stages=num_stages),
    )(x, A, group_sizes, group_offsets, A_scale)
    return y


# reference implementation #############################################################################################


@functools.partial(jax.jit, static_argnames=("block_c", "block_k", "block_n"))
def gpu_ragged_dot_ref(
    x: jax.Array,
    A: jax.Array,
    group_sizes: jax.Array,
    A_scale: jax.Array | None = None,
    block_c: int = 0,
    block_k: int = 0,
    block_n: int = 0,
) -> jax.Array:
    del block_c, block_k, block_n
    ret = jax.lax.ragged_dot(x, A, group_sizes)
    if A_scale is not None:
        indices = jnp.repeat(jnp.arange(A.shape[0]), group_sizes, total_repeat_length=x.shape[0])
        A_scale = jnp.take_along_axis(A_scale, indices[:, None], 0)
        ret = ret * A_scale
    return ret


# tests ################################################################################################################


def test_main(interpret, profile=True, dtype_mode="int8"):
    seed = 21
    n, k, g, m = 128, 7168, 32, 256  # deepseek-like 8-expert parallelism 8-tensor parallelism

    keys = iter(random.split(random.key(seed), 1024))
    x = random.normal(next(keys), (n, k), dtype=jnp.bfloat16)
    A = random.normal(next(keys), (g, k, m), dtype=jnp.bfloat16)
    #x = random.normal(next(keys), (n, k), dtype=jnp.float32)
    #A = random.normal(next(keys), (g, k, m), dtype=jnp.float32)

    A = A / jnp.linalg.norm(A, axis=-1)[..., None]
    if dtype_mode == "int8":
        print("A is in int8")
        A = jnp.round(A * 127).astype(jnp.int8)
    # A_scale = random.normal(next(keys), (g, m), dtype=jnp.float16)
    A_scale = None

    block_c, block_k, block_n = 8, 128, 64

    group_sizes = jnp.exp(1e1 * random.uniform(next(keys), g))
    group_sizes = jnp.round(n * (group_sizes / jnp.sum(group_sizes))).astype(jnp.int32)
    print(group_sizes)
    print(jnp.sum(group_sizes))
    assert jnp.sum(group_sizes) <= n

    opts = dict(block_c=block_c, block_k=block_k, block_n=block_n)
    for _ in range(2):
        ret = gpu_ragged_dot(x, A, group_sizes, A_scale=A_scale, **opts, interpret=interpret).block_until_ready()
        ret_ref = gpu_ragged_dot_ref(x, A, group_sizes, A_scale=A_scale).block_until_ready()
        print(f"error = {float(jnp.linalg.norm(ret - ret_ref) / (jnp.linalg.norm(ret_ref) + 1e-5)):.4e}")

    with jnp.printoptions(linewidth=int(1e9), threshold=int(1e9)):
        print((ret - ret_ref)[:16, :4])
    if not profile:
        return
    with jax.profiler.trace(str(Path("~/profiles/gpu_ragged_dot").expanduser())):
        for _ in range(3):
            ret = gpu_ragged_dot(x, A, group_sizes, A_scale=A_scale, **opts).block_until_ready()
            s = jnp.linalg.norm(ret).block_until_ready()  # no-op barrier operation

        for _ in range(3):
            ret = gpu_ragged_dot_ref(x, A, group_sizes, A_scale=A_scale).block_until_ready()
            s = jnp.linalg.norm(ret).block_until_ready()  # no-op barrier operation


def _numeric_test_case(seed, interpret, n, k, g, m, block_c, block_k, block_n):
    keys = iter(random.split(random.key(seed), 1024))
    x = random.normal(next(keys), (n, k), dtype=jnp.bfloat16)
    A = random.normal(next(keys), (g, k, m), dtype=jnp.bfloat16)
    A = A / jnp.linalg.norm(A, axis=-1)[..., None]
    A = jnp.round(A * 127).astype(jnp.int8)

    group_sizes = jnp.exp(1e1 * random.uniform(next(keys), g))
    group_sizes = jnp.round(n * (group_sizes / jnp.sum(group_sizes))).astype(jnp.int32)
    assert jnp.sum(group_sizes) <= n

    opts = dict(block_c=block_c, block_k=block_k, block_n=block_n)
    try:
        ret = gpu_ragged_dot(x, A, group_sizes, **opts, interpret=interpret).block_until_ready()
    except jax.errors.JaxRuntimeError:
        return float("nan")
    ret_ref = gpu_ragged_dot_ref(x, A, group_sizes).block_until_ready()
    error = float(jnp.linalg.norm(ret - ret_ref) / (jnp.linalg.norm(ret_ref) + 1e-5))
    return error


def test_numerics():
    from tqdm import tqdm

    tests = [
        (seed, n, k, g, m, block_c, k // k_splits, n // n_splits)
        for seed in [0, 1, 2]
        for n in [128, 64, 32]
        for k in [128, 7168]
        for g in [32, 64, 256]
        for m in [7168, 128]
        for block_c in [8, 16]
        for k_splits in [1, 2, 4, 8]
        for n_splits in [1, 2, 4, 8]
        if k // k_splits <= 1024 and not (m == 7168 and k == 7168)
    ]
    pyrandom.shuffle(tests)

    max_error = 0
    it = 0
    for seed, n, k, g, m, block_c, block_k, block_n in tqdm(tests):
        error = _numeric_test_case(seed, False, n, k, g, m, block_c, block_k, block_n)
        error = max(error, max_error)
        if max_error > 1e-4:
            raise ValueError(f"failing {(seed, n, k, g, m, block_c, block_k, block_n)=} with {error=:.4}")
        if (it + 1) % 100 == 0:
            tqdm.write(f"{max_error = :.4e}")
        it += 1


if __name__ == "__main__":
    test_main()