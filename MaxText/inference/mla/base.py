


import dataclasses
from dataclasses import field
from functools import partial
from typing import Callable
import jax

from jax import tree_util
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

import common_types


Config = common_types.Config
AxisName = str | tuple[str, ...] | None
Axes = tuple[AxisName, ...]

# module reload friendly check for type(x) == cls
is_type = lambda x, cls: (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)
is_param = lambda x: is_type(x, ArrayInfo)

@dataclasses.dataclass
class ShardingRules:
    """Mapping from logical data axes to physical mesh axes.

    To manage the different shardings in the model, we define the "logical"
    dimensions of various arrays (each dimension for each layer's weights,
    etc.). Each of these logical axes may then be sharded over a physical mesh
    axis, i.e. over multiple devices. For example, any values with a batch
    dimension should always be sharded over the batch axis of the mesh.

    Defining the shardings this way allows us to easily try out new sharding
    strategies by just changing this mapping. The rest of the code handles
    taking this mapping and eventually turning it into the correct JAX shardings
    and sharding contraints.
    """
    batch: AxisName = None
    sequence: AxisName = None
    head_dim: AxisName = None
    vocab_in: AxisName = None
    vocab_out: AxisName = None
    act_embed: AxisName = None
    act_heads: AxisName = None
   

def logical_to_physical(logical: Axes, rules: ShardingRules) -> jax.sharding.PartitionSpec:
    """Returns how to physically shard a given sequence of logical array dimensions (i.e. the logical shape of an array)."""
    spec = [getattr(rules, axis) if axis is not None else None for axis in logical]
    # `spec` may contain tuples, flatten to check that `spec` maps each physical mesh axis to at most one logical array
    # axis.
    flat_axes = jax.tree.leaves(spec)
    if len(set(flat_axes)) != len(flat_axes):
        raise ValueError(f"Colliding physical axes from translating logical spec {logical} -> {spec}")
    return P(*spec)


def logical_to_sharding(logical: Axes, mesh: jax.sharding.Mesh, rules: ShardingRules) -> jax.sharding.Sharding:
    """Returns the sharding for a given sequence of logical array dimensions (i.e. the logical shape of an array)."""
    return jax.sharding.NamedSharding(mesh, logical_to_physical(logical, rules))


def jax_pytree_struct(cls, meta_fields: tuple = ()):
    """jax.tree_util.register_dataclass wrapper that automatically infers data_fields."""
    assert not dataclasses.is_dataclass(cls)
    cls = dataclasses.dataclass(cls)
    all_fields = tuple(f.name for f in dataclasses.fields(cls) if f.init)
    data_fields = tuple(f for f in all_fields if f not in meta_fields)
    return tree_util.register_dataclass(cls, data_fields=data_fields, meta_fields=meta_fields)


@partial(jax_pytree_struct, meta_fields=("shape", "dtype", "logical_axes", "initializer", "metadata"))
class ArrayInfo:
    """Metadata describing a jax.Array, including its sharding.

    We create ArrayInfos before creating actual arrays, e.g. for model weights, so we can use the sharding and other
    metadata to set things up so we can efficiently create the actual arrays with the correct shardings.

    An alternative approach would be to use jax.eval_shape to more automatically generate the metadata we need. We use
    the ArrayInfo approach instead to decouple data and its sharding from the functions we'll apply the data to.

    """
    shape: tuple[int, ...]
    dtype: "jnp.dtype"
    logical_axes: tuple
    initializer: Callable | None = None
    metadata: dict = field(default_factory=dict)


class _Init:
    """Base class for pytree data structures that will eventually contain jax.Arrays (e.g. layer definitions).

    Each subclass is responsible for defining abstract(), which returns an "abstract" version of the pytree containing
    ArrayInfos (i.e. metadata) instead of actual data. This class then helps generate the shardings and actual data.
    """

    @classmethod
    def abstract(cls, cfg: Config, *args, **kw):
        """Returns an instance of this class with ArrayInfos instead of jax.Arrays."""
        raise NotImplementedError

    @classmethod
    def shardings(cls, cfg: Config, *args, **kw):
        """Returns an instance of this class with Shardings instead of jax.Arrays.

        This is used to generate the Shardings needed for each array.
        """
        abstract = cls.abstract(cfg, *args, **kw)
        return jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules),
            abstract,
            is_leaf=is_param,
        )

    @classmethod
    def init(cls, key: jax.random.PRNGKey, cfg: Config, *args, **kw):
        """Returns a pytree of randomly-initialized jax.Arrays corresponding to abstract()."""
        abstract = cls.abstract(cfg, *args, **kw)
        shardings = jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules),
            abstract,
            is_leaf=is_param,
        )

        @partial(jax.jit, out_shardings=shardings)
        def _init():
            num_leaves = len(jax.tree.leaves(abstract, is_leaf=is_param))  # one new RNG key per tensor
            key_iter = iter(jax.random.split(key, num_leaves))
            return jax.tree.map(
                lambda info: info.initializer(next(key_iter), info.shape, info.dtype),
                abstract,
                is_leaf=is_param,
            )

        return _init()