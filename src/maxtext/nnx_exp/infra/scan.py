from dataclasses import dataclass, field

from flax import nnx
import jax
import jax.numpy as jnp

from maxtext.nnx_exp.infra.remat import _make_remat, _resolve_policy


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScannedLayers:
    graphdef: object = field(metadata={"static": True})
    state: object


def _stack_states(states):
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)


def _split_layer_rngs(rngs, num_layers):
    split = rngs.split(num_layers)
    return [
        nnx.Rngs(**{name: stream.key[i] for name, stream in split.items()})
        for i in range(num_layers)
    ]


def create_scanned_layers(layer_cls, config, num_layers, *, rngs, **kwargs):
    states = []
    graphdef = None
    for layer_rngs in _split_layer_rngs(rngs, num_layers):
        layer = layer_cls(config, rngs=layer_rngs, **kwargs)
        graphdef_i, state_i = nnx.split(layer)
        if graphdef is None:
            graphdef = graphdef_i
        states.append(state_i)
    return ScannedLayers(graphdef=graphdef, state=_stack_states(states))


def create_scanned_remat_layers(layer_cls, config, num_layers, *, rngs, policy="full", **kwargs):
    remat_policy = _resolve_policy(policy)
    states = []
    graphdef = None
    for layer_rngs in _split_layer_rngs(rngs, num_layers):
        layer = _make_remat(layer_cls(config, rngs=layer_rngs, **kwargs), remat_policy)
        graphdef_i, state_i = nnx.split(layer)
        if graphdef is None:
            graphdef = graphdef_i
        states.append(state_i)
    return ScannedLayers(graphdef=graphdef, state=_stack_states(states))


def scan_forward(x, blocks, positions, mask, use_remat=False, remat_policy=None):
    def forward(carry, layer_state):
        layer = nnx.merge(blocks.graphdef, layer_state)
        if use_remat:
            return nnx.remat(layer, policy=remat_policy)(carry, positions, mask), None
        return layer(carry, positions, mask), None

    return jax.lax.scan(forward, x, blocks.state)[0]
