import jax


def to_host(tree):
    shardings = jax.tree.map(lambda x: x.sharding.with_memory_kind("pinned_host"), tree)
    return jax.device_put(tree, shardings)


def to_device(tree):
    shardings = jax.tree.map(lambda x: x.sharding.with_memory_kind("device"), tree)
    return jax.device_put(tree, shardings)


def selective_offload(tree, predicate):
    def maybe_offload(path, leaf):
        p = ".".join(_path_key_to_str(key) for key in path)
        if predicate(p):
            return jax.device_put(leaf, leaf.sharding.with_memory_kind("pinned_host"))
        return leaf

    return jax.tree_util.tree_map_with_path(maybe_offload, tree)


def _path_key_to_str(key):
    match key:
        case jax.tree_util.DictKey(key=k):
            return str(k)
        case jax.tree_util.SequenceKey(idx=i):
            return str(i)
        case jax.tree_util.GetAttrKey(name=n):
            return n
        case jax.tree_util.FlattenedIndexKey(key=k):
            return str(k)
        case _:
            return repr(key)
