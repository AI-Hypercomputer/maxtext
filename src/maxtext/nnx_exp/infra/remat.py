from collections.abc import Callable
from flax import nnx
import jax
from jax import checkpoint_policies
from typing import Any, TypeAlias


def _make_remat(module, policy=None):
    def init(self, module, policy):
        self.module = module
        self.policy = policy

    def call(self, *args, **kwargs):
        return nnx.remat(self.module, policy=self.policy)(*args, **kwargs)

    Remat = type("Remat", (nnx.Module,), {"__init__": init, "__call__": call})
    return Remat(module, policy)


def _resolve_policy(policy):
    match policy:
        case "full":
            return None
        case {"save": save, "offload": offload}:
            _require_activation_offload_backend()
            return checkpoint_policies.save_and_offload_only_these_names(
                names_which_can_be_saved=save,
                names_which_can_be_offloaded=offload,
                offload_src="device",
                offload_dst="pinned_host",
            )
        case {"save": save}:
            return checkpoint_policies.save_only_these_names(*save)
        case {"offload": offload}:
            _require_activation_offload_backend()
            return checkpoint_policies.save_and_offload_only_these_names(
                names_which_can_be_saved=[],
                names_which_can_be_offloaded=offload,
                offload_src="device",
                offload_dst="pinned_host",
            )
        case _ if callable(policy):
            return policy
        case _:
            raise ValueError(f"Unknown remat policy: {policy}")


RematPolicy: TypeAlias = str | dict[str, Any] | Callable[..., Any]


def _require_activation_offload_backend():
    if jax.devices()[0].platform == "cpu":
        raise ValueError("Activation offload via remat requires a non-CPU backend.")


def apply_remat(model, policy: RematPolicy = "full"):
    new_layers = []
    for layer in model.layers.iter_children():
        new_layers.append(_make_remat(layer, remat_policy))
    model.layers = nnx.Sequential(new_layers)


def maybe_apply_remat(model, policy: RematPolicy | None = None):
    if policy is not None:
        apply_remat(model, policy)
