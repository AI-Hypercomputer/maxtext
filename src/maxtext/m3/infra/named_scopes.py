# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Name scopes for profiles and HLO dumps.

Without them, an m3 profile reads `fusion.512` rather than
`decoder/layers/3/mlp/wi_0/dot_general`. An NNX module is a plain Python object
that does not know the attribute it hangs from, and nothing in `nnx.Module`
opens a scope.

`add_named_scopes` supplies the missing piece after construction. It walks the
model and, for every submodule, swaps in a subclass whose `__call__` runs under
a `jax.named_scope` of the attribute-path segment the module sits at. Nesting
does the rest: the ops of `model.decoder.layers.layers[3].mlp.wi_0` come out as
`decoder/layers/3/mlp/wi_0/...`. Other public methods add their own name under
the module's: `self.attn_hyper_connection.before_block(x)` reads
`attn_hyper_connection/before_block/...`. A module calling its own method, as
`apply_block` calls `self.before_block`, does not repeat the module segment:
`attn_hyper_connection/apply_block/before_block/...`. An `nnx.List` is iterated
rather than called, so it opens no scope of its own; its items carry the list's
attribute name instead, so `self.prefix_layers[0]` and `self.suffix_layers[0]`
read `prefix_layers/0/...` and `suffix_layers/0/...`. The exception is the list
inside `nnx.Sequential`, whose `__call__` already opens that segment. Code a
module writes inline, such as the attention scores, lands in the scope of the
module that wrote it. Under scan the layer scope reads
`decoder/layers/while/body/.../mlp/wi_0`.

Scopes are HLO metadata. The computation, parameter tree and parameter paths
do not change, and `isinstance` checks against the original classes still
hold. Apply this after scan: scan requires identical layer graphdefs, and the
scoped class of layer 3 is not the scoped class of layer 4.

A module has one scope. One module instance held under two attribute paths
(`self.b = self.a`) is named after the first path `nnx.iter_graph` reaches, so
ops issued through `self.b(...)` read `a/...`. Call sites that need to tell the
uses apart can open their own `jax.named_scope`. Modules passed as arguments,
like the tied embedding in Qwen3, are not affected: they are scoped under the
module that calls them.
"""

import contextlib
import functools
import inspect
import threading

from flax import nnx
import jax

_SCOPE_ATTR = "_m3_named_scope"

# Per thread, the scoped modules whose methods are running, innermost last.
_running = threading.local()


def _scoped(fn, module_scope: str, method_scope: str | None):
  """`fn` run under `module_scope/method_scope`, minus `module_scope` when the module calls itself."""

  @functools.wraps(fn)
  def wrapper(self, *args, **kwargs):
    stack = _running.__dict__.setdefault("stack", [])
    with contextlib.ExitStack() as scopes:
      if not stack or stack[-1] is not self:
        scopes.enter_context(jax.named_scope(module_scope))
      if method_scope is not None:
        scopes.enter_context(jax.named_scope(method_scope))
      stack.append(self)
      try:
        return fn(self, *args, **kwargs)
      finally:
        stack.pop()

  return wrapper


def _methods_to_scope(cls: type) -> dict[str, object]:
  """`__call__` and the public methods defined by `cls` below `nnx.Module`, by name."""
  methods: dict[str, object] = {}
  for klass in cls.__mro__:
    if klass in (nnx.Module, nnx.Object, object):
      break
    for name, value in vars(klass).items():
      if name in methods or not inspect.isfunction(value):
        continue
      if name == "__call__" or not name.startswith("_"):
        methods[name] = value
  return methods


@functools.cache
def _scoped_class(cls: type, scope: str) -> type:
  """Subclass of unscoped `cls` whose methods open a named scope. Cached so graphdefs stay stable."""
  namespace: dict[str, object] = {_SCOPE_ATTR: scope}
  for name, fn in _methods_to_scope(cls).items():
    namespace[name] = _scoped(fn, scope, None if name == "__call__" else name)
  return type(cls.__name__, (cls,), namespace)


def _unscoped(cls: type) -> type:
  """The class `add_named_scopes` started from."""
  while _SCOPE_ATTR in vars(cls):
    cls = cls.__bases__[0]
  return cls


def scope_of(module: nnx.Module) -> str | None:
  """The named scope `add_named_scopes` gave `module`, or None."""
  return getattr(type(module), _SCOPE_ATTR, None)


def _scope_for(path: tuple, nodes: dict) -> str:
  """The scope for the node at `path`: its last segment, prefixed by its list's name if it sits in an `nnx.List`."""
  if len(path) >= 2 and isinstance(nodes.get(path[:-1]), nnx.List):
    # The list is iterated, not called, so nothing else names it. `nnx.Sequential` names it from its `__call__`.
    if not isinstance(nodes.get(path[:-2]), nnx.Sequential):
      return f"{path[-2]}/{path[-1]}"
  return str(path[-1])


def add_named_scopes(model: nnx.Module) -> nnx.Module:
  """Gives every submodule of `model` a `jax.named_scope` named after its attribute path segment.

  In place and idempotent. The root keeps its class: it has no attribute name. A
  module reachable from several paths is visited once and named after the first;
  see the module docstring.
  """
  nodes = dict(nnx.iter_graph(model))
  for path, node in nodes.items():
    if not path or not isinstance(node, nnx.Module):
      continue
    scope = _scope_for(path, nodes)
    if scope_of(node) == scope:
      continue
    # Bypass `nnx.Object.__setattr__`, which records attributes as graph state.
    object.__setattr__(node, "__class__", _scoped_class(_unscoped(type(node)), scope))
  return model
