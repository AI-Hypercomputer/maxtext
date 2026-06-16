"""Small MoE runtime config types shared without importing model config."""

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import jax


MoEExecutor: TypeAlias = Literal["dense", "ragged_dot", "megablox"]
RoutingMode: TypeAlias = Literal["dropless", "capped"]


@dataclass(frozen=True)
class MegabloxConfig:
    tile_batch_seq: int = 128
    tile_activation_dim: int = 128
    tile_weight_dim: int = 128
    interpret: bool | None = None

    def tiling(self) -> tuple[int, int, int]:
        return (
            self.tile_batch_seq,
            self.tile_activation_dim,
            self.tile_weight_dim,
        )

    def use_interpret(self) -> bool:
        match self.interpret:
            case True:
                return True
            case False:
                return False
            case None:
                return jax.default_backend() != "tpu"


@dataclass(frozen=True)
class MoERuntimeConfig:
    executor: MoEExecutor = "ragged_dot"
    capacity_factor: float = -1.0
    megablox: MegabloxConfig = field(default_factory=MegabloxConfig)
